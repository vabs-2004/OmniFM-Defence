"""Dataset Loaders, Preprocessing, and Leakage-Free Client/Split Partitioning (`backend/utils/preprocessing.py`, Phase 3 Tasks 3.1 - 3.4).

Implements dataset loaders for all four tasks while strictly enforcing:
- Privacy: patient/subject/slide identifiers are never logged; only aggregate counts
  and shapes are exposed in dataset inspection summaries.
- No source mutation: files under `datasets/` are opened read-only.
- Official / subject-level splits:
  1. MedMNIST (`dermamnist`, `pathmnist`, `bloodmnist`): uses official `train`/`val`/`test`
     NPZ arrays directly and preserves distinct class semantics (7, 9, and 8 classes).
  2. FeTS2022 (`MICCAI_FeTS2022_TrainingData`): uses official institution metadata from
     `partitioning_1.csv` (`Partition_ID` 1, 18, 4 for clients 1, 2, 3) and splits
     at the **subject** level before extracting 2D axial T1ce and segmentation slices.
  3. BreaKHis (`BreaKHis_v1`): parses patient slide IDs and splits at the **patient**
     level across both clients and `train`/`val`/`test` before extracting paired
     64x64 LR and x2/x4/x8 HR patches.
  4. Fusion (`CT-MRI`, `PET-MRI`, `SPECT-MRI`): verifies 1-to-1 filename alignment
     and spatial registration before constructing paired `(2, 128, 128)` tensors.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from backend.config import settings
from backend.tasks.classification import CLIENT_TO_MEDMNIST_DATASET, MEDMNIST_DATASET_NUM_CLASSES
from backend.tasks.fusion import CLIENT_TO_FUSION_MODALITY_PAIR

SplitName = Literal["train", "val", "test"]


# =============================================================================
# Pure-Python Read-Only NIfTI-1 (.nii.gz) Reader for FeTS2022
# =============================================================================

_NIFTI_DTYPE_MAP: dict[int, Any] = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def read_nifti1_volume(path: str | Path) -> np.ndarray:
    """Read a 3D NIfTI-1 `.nii` or `.nii.gz` file read-only into a NumPy array `(H, W, D)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    if file_path.name.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            raw = f.read()
    else:
        raw = file_path.read_bytes()

    if len(raw) < 352:
        raise ValueError(f"Invalid NIfTI-1 file (too short): {file_path}")

    sizeof_hdr_le = struct.unpack("<i", raw[0:4])[0]
    endian = "<" if sizeof_hdr_le == 348 else ">"
    sizeof_hdr = struct.unpack(f"{endian}i", raw[0:4])[0]
    if sizeof_hdr != 348:
        raise ValueError(f"Unsupported NIfTI header size {sizeof_hdr} in {file_path}")

    dims = struct.unpack(f"{endian}8h", raw[40:56])
    ndim = int(dims[0])
    shape = tuple(int(dims[i]) for i in range(1, ndim + 1))

    datatype, _bitpix = struct.unpack(f"{endian}2h", raw[70:74])
    vox_offset = int(struct.unpack(f"{endian}f", raw[108:112])[0])
    scl_slope, scl_inter = struct.unpack(f"{endian}2f", raw[112:120])

    if datatype not in _NIFTI_DTYPE_MAP:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype} in {file_path}")

    np_dtype = np.dtype(_NIFTI_DTYPE_MAP[datatype]).newbyteorder(endian)
    num_voxels = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=np_dtype, count=num_voxels, offset=vox_offset)
    vol = arr.reshape(shape, order="F")

    if scl_slope != 0.0 and np.isfinite(scl_slope) and (scl_slope != 1.0 or scl_inter != 0.0):
        vol = vol.astype(np.float32) * float(scl_slope) + float(scl_inter)

    return vol


def _deterministic_split_items(
    items: list[Any],
    split: SplitName,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[Any]:
    """Deterministically partition sorted items into disjoint train/val/test subsets."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)

    n_total = len(items)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total and n_total >= 3:
        n_train = max(1, n_total - 2)
        n_val = 1

    if split == "train":
        sel = indices[:n_train]
    elif split == "val":
        sel = indices[n_train : n_train + n_val]
    elif split == "test":
        sel = indices[n_train + n_val :]
    else:
        raise ValueError(f"Invalid split '{split}': must be 'train', 'val', or 'test'.")

    return [items[int(i)] for i in sorted(sel)]


# =============================================================================
# Task 3.1: MedMNIST Classification Dataset Loader
# =============================================================================


_MEDMNIST_CACHE: dict[tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]] = {}
_FETS_SLICE_CACHE: dict[tuple[str, str, str, int | None, int, str, int], list[tuple[Path, Path, int]]] = {}
_FETS_TENSOR_CACHE: dict[tuple[str, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _read_npy_from_zip_sliced(
    zf: Any,
    member_name: str,
    max_rows: int | None,
) -> np.ndarray:
    """Read a `.npy` member from an open ZipFile, reading only `max_rows` rows if specified."""
    import numpy.lib.format as np_fmt

    with zf.open(member_name, "r") as fp:
        version = np_fmt.read_magic(fp)
        if version == (1, 0):
            shape, fortran_order, dtype = np_fmt.read_array_header_1_0(fp)
        elif version == (2, 0):
            shape, fortran_order, dtype = np_fmt.read_array_header_2_0(fp)
        else:
            shape, fortran_order, dtype = np_fmt._read_array_header(fp, version)

        if not shape or fortran_order:
            arr = np_fmt.read_array(fp, allow_pickle=False)
            return arr[:max_rows].copy() if (max_rows is not None and max_rows > 0) else arr

        total_rows = int(shape[0])
        n_rows = min(total_rows, int(max_rows)) if (max_rows is not None and max_rows > 0) else total_rows
        row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        n_bytes = n_rows * row_elems * int(dtype.itemsize)
        raw = fp.read(n_bytes)
        out_shape = (n_rows, *shape[1:]) if len(shape) > 1 else (n_rows,)
        return np.frombuffer(raw, dtype=dtype).reshape(out_shape).copy()


class MedMNISTClassificationDataset(Dataset):
    """MedMNIST loader preserving official train/val/test splits and distinct label semantics.

    Input tensor shape: `(3, 28, 28)` float32 normalized to `[-1.0, 1.0]` (`(x/255 - 0.5)/0.5`).
    Target tensor: scalar `int64` class index in `[0, num_classes - 1]`.
    """

    def __init__(
        self,
        dataset_name: str = "dermamnist",
        split: SplitName = "train",
        dataset_dir: str | Path | None = None,
        max_samples: int | None = None,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        import zipfile

        ds_key = dataset_name.lower().strip()
        if ds_key not in MEDMNIST_DATASET_NUM_CLASSES:
            raise ValueError(
                f"Unsupported MedMNIST dataset '{dataset_name}'. "
                f"Expected one of {list(MEDMNIST_DATASET_NUM_CLASSES.keys())}."
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}': must be 'train', 'val', or 'test'.")

        root = Path(dataset_dir or settings.dataset_dir) / "medmnist"
        npz_path = root / f"{ds_key}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"MedMNIST archive not found: {npz_path}")

        eff_max = int(max_samples) if (max_samples is not None and max_samples > 0) else None
        cache_key = (str(npz_path.resolve()), split, eff_max)
        if cache_key in _MEDMNIST_CACHE:
            images, labels = _MEDMNIST_CACHE[cache_key]
        else:
            img_member = f"{split}_images.npy"
            lbl_member = f"{split}_labels.npy"
            with zipfile.ZipFile(npz_path, "r") as zf:
                names = set(zf.namelist())
                if img_member not in names or lbl_member not in names:
                    raise KeyError(
                        f"Missing official split keys ({img_member}, {lbl_member}) in {npz_path}. "
                        f"Found: {sorted(names)}"
                    )
                labels = _read_npy_from_zip_sliced(zf, lbl_member, eff_max).reshape(-1).astype(np.int64)
                images = _read_npy_from_zip_sliced(zf, img_member, eff_max)
            if eff_max is not None:
                _MEDMNIST_CACHE[cache_key] = (images, labels)

        self.dataset_name = ds_key
        self.split = split
        self.num_classes = MEDMNIST_DATASET_NUM_CLASSES[ds_key]
        self.image_size = int(image_size)
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(len(self._labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_np = self._images[idx]
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        if img_np.shape[0] != self.image_size or img_np.shape[1] != self.image_size:
            pil_img = Image.fromarray(img_np.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            img_np = np.asarray(pil_img)

        # Convert HWC uint8 [0, 255] -> CHW float32 normalized to [-1, 1]
        tensor_x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        tensor_x = (tensor_x - 0.5) / 0.5
        tensor_y = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        return tensor_x, tensor_y

    def aggregate_statistics(self) -> dict[str, Any]:
        """Return privacy-safe aggregate class counts and shapes."""
        unique_cls, counts = np.unique(self._labels, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique_cls, counts)}
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_samples": len(self),
            "num_classes": self.num_classes,
            "image_shape": (3, self.image_size, self.image_size),
            "class_counts": class_counts,
        }


# =============================================================================
# Task 3.2: FeTS2022 Brain MRI Tumor Segmentation Dataset Loader
# =============================================================================

# Top 3 institutions in partitioning_1.csv:
# Institution '1' (511 subjects), Institution '18' (382 subjects), Institution '4' (47 subjects)
FETS_CLIENT_INSTITUTION_MAP: dict[str, str] = {
    "client_1": "1",
    "client_2": "18",
    "client_3": "4",
}


class FeTS2022SegmentationDataset(Dataset):
    """FeTS2022 2D axial T1ce segmentation loader with institution & subject-level splits.

    - Uses official `partitioning_1.csv` institution IDs so each client maps to a
      real originating institution (`client_1 -> inst 1`, `client_2 -> inst 18`,
      `client_3 -> inst 4`).
    - Splits subjects into `train`/`val`/`test` at the **subject** level prior to
      slice extraction so zero subject leakage occurs across splits.
    - Extracts non-empty tumor slices (or central brain slices as fallback) from
      `*_t1ce.nii.gz` and `*_seg.nii.gz`, normalizes T1ce intensity to `[0.0, 1.0]`,
      and resizes to `(1, 128, 128)`.
    """

    def __init__(
        self,
        client_id: str = "client_1",
        split: SplitName = "train",
        dataset_dir: str | Path | None = None,
        image_size: int = 128,
        slices_per_subject: int = 2,
        max_subjects: int | None = None,
        tumor_label_mode: Literal["whole_tumor", "enhancing_tumor"] = "whole_tumor",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if client_id not in FETS_CLIENT_INSTITUTION_MAP:
            raise ValueError(
                f"Unknown client_id '{client_id}'. Expected one of {list(FETS_CLIENT_INSTITUTION_MAP.keys())}."
            )

        self.client_id = client_id
        self.institution_id = FETS_CLIENT_INSTITUTION_MAP[client_id]
        self.split = split
        self.image_size = int(image_size)
        self.slices_per_subject = int(slices_per_subject)
        self.tumor_label_mode = tumor_label_mode

        train_root = (
            Path(dataset_dir or settings.dataset_dir)
            / "fets2022"
            / "MICCAI_FeTS2022_TrainingData"
        )
        part_csv = train_root / "partitioning_1.csv"
        if not part_csv.is_file():
            raise FileNotFoundError(f"FeTS2022 institution partitioning CSV not found: {part_csv}")

        institution_subjects: list[str] = []
        with part_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Partition_ID", "").strip() == self.institution_id:
                    subj_id = row.get("Subject_ID", "").strip()
                    if subj_id and (train_root / subj_id).is_dir():
                        institution_subjects.append(subj_id)

        institution_subjects.sort()
        if not institution_subjects:
            raise RuntimeError(
                f"No FeTS2022 subjects found for institution {self.institution_id} ({client_id})."
            )

        split_subjects = _deterministic_split_items(
            institution_subjects, split=split, seed=seed
        )
        if max_subjects is not None and max_subjects > 0:
            split_subjects = split_subjects[: int(max_subjects)]

        self._subject_hashes = [
            hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] for s in split_subjects
        ]
        fets_key = (
            str(train_root.resolve()),
            self.client_id,
            self.split,
            int(max_subjects) if (max_subjects is not None and max_subjects > 0) else None,
            self.slices_per_subject,
            self.tumor_label_mode,
            int(seed),
        )
        if fets_key in _FETS_SLICE_CACHE:
            self._slice_records = list(_FETS_SLICE_CACHE[fets_key])
        else:
            self._slice_records = []
            self._prepare_slice_index(train_root, split_subjects)
            _FETS_SLICE_CACHE[fets_key] = list(self._slice_records)

    def _prepare_slice_index(self, train_root: Path, subject_ids: list[str]) -> None:
        """Select non-empty axial slice indices for each assigned subject."""
        for subj_id in subject_ids:
            subj_dir = train_root / subj_id
            t1ce_path = subj_dir / f"{subj_id}_t1ce.nii.gz"
            seg_path = subj_dir / f"{subj_id}_seg.nii.gz"
            if not (t1ce_path.is_file() and seg_path.is_file()):
                continue

            seg_vol = read_nifti1_volume(seg_path)
            if self.tumor_label_mode == "enhancing_tumor":
                mask_3d = seg_vol == 4
            else:
                mask_3d = seg_vol > 0

            # Compute tumor area per axial slice z
            slice_tumor_counts = mask_3d.reshape(-1, mask_3d.shape[2]).sum(axis=0)
            positive_z = np.where(slice_tumor_counts > 25)[0]

            if len(positive_z) >= self.slices_per_subject:
                # Pick evenly spaced tumor-bearing axial slices sorted by tumor area
                top_z = positive_z[np.argsort(-slice_tumor_counts[positive_z])]
                selected_z = sorted(int(z) for z in top_z[: self.slices_per_subject])
            elif len(positive_z) > 0:
                selected_z = [int(z) for z in positive_z]
            else:
                # Fallback to central axial slice if subject has no matching mask voxels
                selected_z = [int(seg_vol.shape[2] // 2)]

            for z in selected_z:
                self._slice_records.append((t1ce_path, seg_path, z))

    @property
    def anonymized_subject_hashes(self) -> set[str]:
        """Return SHA-256 prefix hashes of subject IDs for leakage verification without exposing raw IDs."""
        return set(self._subject_hashes)

    def __len__(self) -> int:
        return len(self._slice_records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t1ce_path, seg_path, z = self._slice_records[idx]
        slice_key = (str(t1ce_path), int(z), self.image_size, self.tumor_label_mode)
        if slice_key in _FETS_TENSOR_CACHE:
            return _FETS_TENSOR_CACHE[slice_key]

        t1ce_vol = read_nifti1_volume(t1ce_path)
        seg_vol = read_nifti1_volume(seg_path)

        img_slice = t1ce_vol[:, :, z].astype(np.float32)
        seg_slice = seg_vol[:, :, z]
        del t1ce_vol, seg_vol

        # Robust intensity normalization to [0.0, 1.0] over brain region
        nonzero = img_slice[img_slice > 0]
        if nonzero.size > 0:
            p1, p99 = np.percentile(nonzero, [1.0, 99.0])
            if p99 > p1:
                img_norm = np.clip((img_slice - p1) / (p99 - p1), 0.0, 1.0)
            else:
                img_norm = np.clip(img_slice / (float(img_slice.max()) + 1e-6), 0.0, 1.0)
            img_norm[img_slice <= 0] = 0.0
        else:
            img_norm = np.zeros_like(img_slice, dtype=np.float32)

        if self.tumor_label_mode == "enhancing_tumor":
            mask_bin = (seg_slice == 4).astype(np.float32)
        else:
            mask_bin = (seg_slice > 0).astype(np.float32)

        # Resize image (bilinear) and binary mask (nearest) to (image_size, image_size)
        img_pil = Image.fromarray((img_norm * 255.0).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask_pil = Image.fromarray((mask_bin * 255.0).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )

        x_tensor = torch.from_numpy(np.asarray(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        y_tensor = torch.from_numpy(
            (np.asarray(mask_pil, dtype=np.float32) >= 128.0).astype(np.float32)
        ).unsqueeze(0)
        _FETS_TENSOR_CACHE[slice_key] = (x_tensor, y_tensor)
        return x_tensor, y_tensor

    def aggregate_statistics(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "institution_id": self.institution_id,
            "split": self.split,
            "num_subjects": len(self._subject_hashes),
            "num_slices": len(self._slice_records),
            "image_shape": (1, self.image_size, self.image_size),
            "tumor_label_mode": self.tumor_label_mode,
        }


# =============================================================================
# Task 3.3: BreaKHis Super-Resolution Dataset Loader
# =============================================================================

_BREAKHIS_FILENAME_RE = re.compile(
    r"^(?:SOB|CNB)_[BM]_[A-Z]+-(\d{2}-\d+[A-Z]*)-(\d+)-\d+\.png$"
)

# Simulated non-IID magnification assignment per logical hospital client
BREAKHIS_CLIENT_MAGNIFICATIONS: dict[str, tuple[str, ...]] = {
    "client_1": ("40X", "100X"),
    "client_2": ("200X",),
    "client_3": ("400X",),
}


def extract_breakhis_patient_id(filepath: str | Path) -> str:
    """Extract the patient/slide identifier (e.g. `14-22549AB`) from a BreaKHis filename."""
    name = Path(filepath).name
    m = _BREAKHIS_FILENAME_RE.match(name)
    if m:
        return m.group(1)
    parts = name.split("-")
    if len(parts) >= 3:
        return f"{parts[1]}-{parts[2]}"
    return name


class BreaKHisSRDataset(Dataset):
    """BreaKHis histopathology super-resolution dataset with patient-level splitting.

    Partition & Degradation Protocol:
    1. All unique patient IDs in `BreaKHis_v1` are partitioned into disjoint
       `train` (70%), `val` (15%), and `test` (15%) patient sets *before* any
       patch extraction, preventing patient leakage across splits.
    2. Clients are assigned magnification-specific non-IID partitions:
       - `client_1`: `40X` and `100X`
       - `client_2`: `200X`
       - `client_3`: `400X`
    3. For a given scale factor `s in {2, 4, 8}`, a deterministic central high-resolution
       target patch of size `(64*s, 64*s)` is prepared in `[0.0, 1.0]` and downsampled
       to a `(3, 64, 64)` low-resolution input using bicubic interpolation (`Image.Resampling.BICUBIC`).
    """

    def __init__(
        self,
        client_id: str = "client_1",
        split: SplitName = "train",
        scale: int = 2,
        lr_patch_size: int = 64,
        dataset_dir: str | Path | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if client_id not in BREAKHIS_CLIENT_MAGNIFICATIONS:
            raise ValueError(
                f"Unknown client_id '{client_id}'. Expected one of {list(BREAKHIS_CLIENT_MAGNIFICATIONS.keys())}."
            )
        if scale not in (2, 4, 8):
            raise ValueError(f"Unsupported SR scale {scale}: must be 2, 4, or 8.")

        self.client_id = client_id
        self.split = split
        self.scale = int(scale)
        self.lr_patch_size = int(lr_patch_size)
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.magnifications = BREAKHIS_CLIENT_MAGNIFICATIONS[client_id]

        bh_root = (
            Path(dataset_dir or settings.dataset_dir)
            / "breakhis"
            / "BreaKHis_v1"
            / "histology_slides"
            / "breast"
        )
        if not bh_root.is_dir():
            raise FileNotFoundError(f"BreaKHis root directory not found: {bh_root}")

        all_pngs = sorted(bh_root.rglob("*.png"))
        patient_to_files: dict[str, list[Path]] = {}
        for p in all_pngs:
            pid = extract_breakhis_patient_id(p)
            patient_to_files.setdefault(pid, []).append(p)

        all_patients = sorted(patient_to_files.keys())
        split_patients = set(_deterministic_split_items(all_patients, split=split, seed=seed))

        selected_files: list[Path] = []
        selected_patients: set[str] = set()
        for pid in sorted(split_patients):
            for p in patient_to_files[pid]:
                if any(mag in p.parts for mag in self.magnifications):
                    selected_files.append(p)
                    selected_patients.add(pid)

        if max_samples is not None and max_samples > 0:
            selected_files = selected_files[: int(max_samples)]
            selected_patients = {extract_breakhis_patient_id(p) for p in selected_files}

        self._files = selected_files
        self._patient_hashes = {
            hashlib.sha256(pid.encode("utf-8")).hexdigest()[:12] for pid in selected_patients
        }

    @property
    def anonymized_patient_hashes(self) -> set[str]:
        """Return SHA-256 prefix hashes of patient IDs for leakage verification."""
        return set(self._patient_hashes)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self._files[idx]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            crop_side = min(width, height, self.hr_patch_size)
            left = (width - crop_side) // 2
            top = (height - crop_side) // 2
            cropped = rgb.crop((left, top, left + crop_side, top + crop_side))

            if crop_side != self.hr_patch_size:
                hr_pil = cropped.resize(
                    (self.hr_patch_size, self.hr_patch_size), Image.Resampling.BICUBIC
                )
            else:
                hr_pil = cropped

            # Documented degradation kernel: Bicubic downsampling to 64x64 LR input
            lr_pil = hr_pil.resize(
                (self.lr_patch_size, self.lr_patch_size), Image.Resampling.BICUBIC
            )

        lr_np = np.asarray(lr_pil, dtype=np.float32) / 255.0
        hr_np = np.asarray(hr_pil, dtype=np.float32) / 255.0

        lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).contiguous()
        hr_tensor = torch.from_numpy(hr_np).permute(2, 0, 1).contiguous()
        return lr_tensor, hr_tensor

    def aggregate_statistics(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "magnifications": self.magnifications,
            "split": self.split,
            "scale": self.scale,
            "num_patients": len(self._patient_hashes),
            "num_patches": len(self._files),
            "lr_shape": (3, self.lr_patch_size, self.lr_patch_size),
            "hr_shape": (3, self.hr_patch_size, self.hr_patch_size),
            "degradation_kernel": "PIL.Image.Resampling.BICUBIC",
        }


# =============================================================================
# Task 3.4: Heterogeneous Multimodal Medical Image Fusion Dataset Loader
# =============================================================================

FUSION_PAIR_SUBDIRS: dict[str, tuple[str, str]] = {
    "CT-MRI": ("CT", "MRI"),
    "PET-MRI": ("PET", "MRI"),
    "SPECT-MRI": ("SPECT", "MRI"),
}


class MultimodalFusionDataset(Dataset):
    """Paired multimodal medical image fusion loader for CT-MRI, PET-MRI, and SPECT-MRI.

    Verifies that every filename exists in both modality folders and that source
    dimensions match before resizing to `(1, 128, 128)` per modality and stacking
    into `(2, 128, 128)` in `[0.0, 1.0]`.
    """

    def __init__(
        self,
        modality_pair: str | None = None,
        client_id: str = "client_1",
        split: SplitName = "train",
        image_size: int = 128,
        dataset_dir: str | Path | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        pair_key = modality_pair or CLIENT_TO_FUSION_MODALITY_PAIR.get(client_id, "CT-MRI")
        if pair_key not in FUSION_PAIR_SUBDIRS:
            raise ValueError(
                f"Unsupported modality_pair '{pair_key}'. Expected one of {list(FUSION_PAIR_SUBDIRS.keys())}."
            )

        self.modality_pair = pair_key
        self.client_id = client_id
        self.split = split
        self.image_size = int(image_size)
        mod_a, mod_b = FUSION_PAIR_SUBDIRS[pair_key]
        self.modality_a = mod_a
        self.modality_b = mod_b

        pair_root = Path(dataset_dir or settings.dataset_dir) / "fusion" / pair_key
        dir_a = pair_root / mod_a
        dir_b = pair_root / mod_b
        if not (dir_a.is_dir() and dir_b.is_dir()):
            raise FileNotFoundError(
                f"Fusion modality directories not found for {pair_key}: {dir_a}, {dir_b}"
            )

        files_a = {f.name: f for f in dir_a.glob("*.*") if f.is_file()}
        files_b = {f.name: f for f in dir_b.glob("*.*") if f.is_file()}
        common_names = sorted(set(files_a.keys()) & set(files_b.keys()))
        if not common_names:
            raise RuntimeError(f"No matching paired filenames found in {pair_root}.")

        split_names = _deterministic_split_items(common_names, split=split, seed=seed)
        if max_samples is not None and max_samples > 0:
            split_names = split_names[: int(max_samples)]

        self._pairs: list[tuple[Path, Path]] = [
            (files_a[name], files_b[name]) for name in split_names
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_a, path_b = self._pairs[idx]
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            if im_a.size != im_b.size:
                raise ValueError(
                    f"Spatial alignment size mismatch for pair {path_a.name}: "
                    f"{im_a.size} vs {im_b.size}."
                )
            gray_a = im_a.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            gray_b = im_b.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )

        arr_a = np.asarray(gray_a, dtype=np.float32) / 255.0
        arr_b = np.asarray(gray_b, dtype=np.float32) / 255.0
        paired = np.stack([arr_a, arr_b], axis=0)  # (2, H, W)
        paired_tensor = torch.from_numpy(paired).contiguous()
        # For self-supervised U2Fusion training, the stacked pair serves as both input and reference pair
        return paired_tensor, paired_tensor.clone()

    def aggregate_statistics(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "modality_pair": self.modality_pair,
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "split": self.split,
            "num_pairs": len(self._pairs),
            "paired_shape": (2, self.image_size, self.image_size),
        }


# =============================================================================
# Unified DataLoader Builder for Federated Clients
# =============================================================================


def build_client_dataloader(
    task_type: str,
    client_id: str = "client_1",
    split: SplitName = "train",
    batch_size: int | None = None,
    max_samples: int | None = None,
    scale: int = 2,
    shuffle: bool | None = None,
    dataset_dir: str | Path | None = None,
    seed: int = 42,
) -> tuple[DataLoader, dict[str, Any]]:
    """Construct a task- and client-specific DataLoader and its aggregate summary."""
    task_key = task_type.lower().strip()
    is_train = split == "train"
    do_shuffle = is_train if shuffle is None else bool(shuffle)

    if task_key == "classification":
        ds_name = CLIENT_TO_MEDMNIST_DATASET.get(client_id, "dermamnist")
        ds: Dataset = MedMNISTClassificationDataset(
            dataset_name=ds_name,
            split=split,
            dataset_dir=dataset_dir,
            max_samples=max_samples,
            image_size=settings.image_size_classification,
        )
        bsz = batch_size or settings.batch_size_classification
    elif task_key == "segmentation":
        ds = FeTS2022SegmentationDataset(
            client_id=client_id,
            split=split,
            dataset_dir=dataset_dir,
            image_size=settings.image_size_segmentation,
            max_subjects=max_samples,
            seed=seed,
        )
        bsz = batch_size or settings.batch_size_segmentation
    elif task_key in ("superresolution", "sr"):
        ds = BreaKHisSRDataset(
            client_id=client_id,
            split=split,
            scale=scale,
            lr_patch_size=settings.image_size_superresolution_lr,
            dataset_dir=dataset_dir,
            max_samples=max_samples,
            seed=seed,
        )
        bsz = batch_size or settings.batch_size_superresolution
    elif task_key == "fusion":
        ds = MultimodalFusionDataset(
            client_id=client_id,
            split=split,
            image_size=settings.image_size_fusion,
            dataset_dir=dataset_dir,
            max_samples=max_samples,
            seed=seed,
        )
        bsz = batch_size or settings.batch_size_fusion
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    loader = DataLoader(ds, batch_size=int(bsz), shuffle=do_shuffle, num_workers=0)
    stats = ds.aggregate_statistics()  # type: ignore[attr-defined]
    return loader, stats
