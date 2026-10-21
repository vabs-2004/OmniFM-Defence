"""Medical Imaging Prediction Endpoints (`backend/api/endpoints/predict.py`, Task 6.4).

Implements:
- `POST /api/predict/classify`
- `POST /api/predict/segment`
- `POST /api/predict/superresolution`
- `POST /api/predict/fusion`

Enforces strict reference-metric guards: Dice/IoU and PSNR/SSIM are NEVER fabricated or
returned for unlabeled/unreferenced inference uploads.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
import torch
import torch.nn.functional as F

from backend.config import settings
from backend.services.artifact_service import (
    CheckpointNotFoundError,
    decode_image_bytes_to_tensor,
    encode_tensor_to_base64_png,
    resolve_and_load_adapter,
)
from backend.tasks.classification import (
    CLIENT_TO_MEDMNIST_DATASET,
    MEDMNIST_DATASET_NUM_CLASSES,
)
from backend.utils.metrics import compute_fusion_metrics, evaluate_task_predictions

router = APIRouter(tags=["Predictions"])

DATASET_TO_CLIENT = {v: k for k, v in CLIENT_TO_MEDMNIST_DATASET.items()}

MEDMNIST_CLASS_NAMES: dict[str, list[str]] = {
    "dermamnist": [
        "Actinic keratoses & intraepithelial carcinoma",
        "Basal cell carcinoma",
        "Benign keratosis-like lesions",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic nevi",
        "Vascular lesions",
    ],
    "pathmnist": [
        "Adipose (ADI)",
        "Background (BACK)",
        "Debris (DEB)",
        "Lymphocytes (LYM)",
        "Mucus (MUC)",
        "Smooth muscle (MUS)",
        "Normal colon mucosa (NORM)",
        "Cancer-associated stroma (STR)",
        "Colorectal adenocarcinoma epithelium (TUM)",
    ],
    "bloodmnist": [
        "Basophil",
        "Eosinophil",
        "Erythroblast",
        "Immature granulocytes",
        "Lymphocyte",
        "Monocyte",
        "Neutrophil",
        "Platelet",
    ],
}

VALID_MODALITY_PAIRS = {
    "CT-MRI": "client_1",
    "MRI-CT": "client_1",
    "PET-MRI": "client_2",
    "MRI-PET": "client_2",
    "SPECT-MRI": "client_3",
    "MRI-SPECT": "client_3",
}


def _get_db_url(request: Request) -> str:
    return getattr(request.app.state, "database_url", None) or settings.database_url


@router.post("/api/predict/classify")
async def predict_classify(
    request: Request,
    file: UploadFile = File(..., description="Input medical image (28x28 or resized)"),
    client_id: str = Form(default="client_1", description="client_1 | client_2 | client_3"),
    dataset_name: str | None = Form(
        default=None, description="Optional dataset head: dermamnist | pathmnist | bloodmnist"
    ),
    experiment_id: str | None = Form(default=None),
    round_num: int | None = Form(default=None),
    allow_untrained_fallback: bool = Form(default=True),
) -> dict[str, Any]:
    """Run multi-class classification with dataset/client-specific head."""
    resolved_client = client_id.strip()
    if dataset_name:
        ds_key = dataset_name.lower().strip()
        if ds_key not in DATASET_TO_CLIENT:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported dataset_name '{dataset_name}'. Valid options: {list(DATASET_TO_CLIENT.keys())}.",
            )
        resolved_client = DATASET_TO_CLIENT[ds_key]
    elif resolved_client not in CLIENT_TO_MEDMNIST_DATASET:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported client_id '{resolved_client}'. Valid options: {list(CLIENT_TO_MEDMNIST_DATASET.keys())}.",
        )

    raw_bytes = await file.read()
    try:
        image_tensor, prep_meta = decode_image_bytes_to_tensor(
            raw_bytes, channels=3, target_size=(28, 28)
        )
        adapter, model_meta = resolve_and_load_adapter(
            task_type="classification",
            experiment_id=experiment_id,
            round_num=round_num,
            client_id=resolved_client,
            allow_untrained_fallback=allow_untrained_fallback,
            database_url=_get_db_url(request),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ds_name = CLIENT_TO_MEDMNIST_DATASET[resolved_client]
    with torch.no_grad():
        # Match MedMNIST [-1, 1] input normalization
        norm_input = (image_tensor - 0.5) / 0.5
        fwd = adapter.forward_features_and_predict(
            norm_input, dataset_name=ds_name, client_id=resolved_client
        )
        probs = F.softmax(fwd.prediction.float(), dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())

    class_names = MEDMNIST_CLASS_NAMES.get(
        ds_name, [str(i) for i in range(MEDMNIST_DATASET_NUM_CLASSES[ds_name])]
    )
    probs_list = [round(float(p), 6) for p in probs.tolist()]

    labeled_probs = [
        {"class_index": i, "class_name": class_names[i], "probability": probs_list[i]}
        for i in range(len(probs_list))
    ]

    return {
        "task_type": "classification",
        "client_id": resolved_client,
        "dataset_name": ds_name,
        "predicted_class": pred_idx,
        "predicted_label": class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx),
        "confidence": round(conf, 6),
        "probabilities": probs_list,
        "class_probabilities": labeled_probs,
        "model_version": model_meta["model_version"],
        "model_metadata": model_meta,
        "preprocessing": prep_meta,
        "clinical_disclaimer": (
            "Research prototype only: softmax confidence is uncalibrated and does NOT represent "
            "clinical diagnostic certainty."
        ),
    }


@router.post("/api/predict/segment")
async def predict_segment(
    request: Request,
    file: UploadFile = File(..., description="Input MRI slice image"),
    reference_mask: UploadFile | None = File(
        default=None, description="Optional ground-truth segmentation mask for Dice/IoU evaluation"
    ),
    client_id: str = Form(default="client_1"),
    experiment_id: str | None = Form(default=None),
    round_num: int | None = Form(default=None),
    allow_untrained_fallback: bool = Form(default=True),
) -> dict[str, Any]:
    """Run brain tumor MRI slice segmentation.

    Dice and IoU are computed ONLY when `reference_mask` is supplied.
    """
    raw_bytes = await file.read()
    try:
        image_tensor, prep_meta = decode_image_bytes_to_tensor(
            raw_bytes, channels=1, target_size=(128, 128)
        )
        adapter, model_meta = resolve_and_load_adapter(
            task_type="segmentation",
            experiment_id=experiment_id,
            round_num=round_num,
            client_id=client_id,
            allow_untrained_fallback=allow_untrained_fallback,
            database_url=_get_db_url(request),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with torch.no_grad():
        fwd = adapter.forward_features_and_predict(image_tensor)
        prob_map = torch.sigmoid(fwd.prediction.float())
        binary_mask = (prob_map >= 0.5).float()

    metrics: dict[str, Any] | None = None
    metrics_status = "unlabeled_inference_no_reference_mask"

    if reference_mask is not None:
        mask_bytes = await reference_mask.read()
        if mask_bytes:
            try:
                gt_tensor, _ = decode_image_bytes_to_tensor(
                    mask_bytes, channels=1, target_size=(128, 128)
                )
                gt_binary = (gt_tensor >= 0.5).float()
                eval_res = evaluate_task_predictions("segmentation", prob_map, gt_binary)
                metrics = {
                    "dice": round(float(eval_res["dice"]), 6),
                    "iou": round(float(eval_res["iou"]), 6),
                    "reference_provided": True,
                }
                metrics_status = "measured_against_reference_mask"
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid reference_mask file: {exc}"
                ) from exc

    tumor_fraction = float(binary_mask.mean().item())
    return {
        "task_type": "segmentation",
        "client_id": client_id,
        "mask_base64": encode_tensor_to_base64_png(binary_mask),
        "probability_map_base64": encode_tensor_to_base64_png(prob_map),
        "input_base64": encode_tensor_to_base64_png(image_tensor),
        "tumor_pixel_fraction": round(tumor_fraction, 6),
        "metrics": metrics,
        "metrics_status": metrics_status,
        "model_version": model_meta["model_version"],
        "model_metadata": model_meta,
        "preprocessing": prep_meta,
    }


@router.post("/api/predict/superresolution")
async def predict_superresolution(
    request: Request,
    file: UploadFile = File(..., description="Low-resolution histopathology image"),
    scale: int = Form(default=2, description="Upsampling factor: 2, 4, or 8"),
    reference_hr: UploadFile | None = File(
        default=None, description="Optional paired high-resolution reference image for PSNR/SSIM"
    ),
    client_id: str = Form(default="client_1"),
    experiment_id: str | None = Form(default=None),
    round_num: int | None = Form(default=None),
    allow_untrained_fallback: bool = Form(default=True),
) -> dict[str, Any]:
    """Run medical image super-resolution at x2, x4, or x8 scale.

    PSNR, SSIM, and L1 are computed ONLY when `reference_hr` is supplied.
    """
    if int(scale) not in (2, 4, 8):
        raise HTTPException(
            status_code=400, detail=f"Invalid super-resolution scale '{scale}'. Must be 2, 4, or 8."
        )

    raw_bytes = await file.read()
    try:
        lr_tensor, prep_meta = decode_image_bytes_to_tensor(
            raw_bytes, channels=3, target_size=(64, 64)
        )
        adapter, model_meta = resolve_and_load_adapter(
            task_type="superresolution",
            experiment_id=experiment_id,
            round_num=round_num,
            client_id=client_id,
            scale_factor=int(scale),
            allow_untrained_fallback=allow_untrained_fallback,
            database_url=_get_db_url(request),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with torch.no_grad():
        fwd = adapter.forward_features_and_predict(lr_tensor, scale=int(scale))
        sr_tensor = torch.clamp(fwd.prediction.float(), 0.0, 1.0)

    hr_h = 64 * int(scale)
    hr_w = 64 * int(scale)

    metrics: dict[str, Any] | None = None
    metrics_status = "unreferenced_inference_no_hr_target"

    if reference_hr is not None:
        hr_bytes = await reference_hr.read()
        if hr_bytes:
            try:
                hr_tensor, _ = decode_image_bytes_to_tensor(
                    hr_bytes, channels=3, target_size=(hr_h, hr_w)
                )
                eval_res = evaluate_task_predictions("superresolution", sr_tensor, hr_tensor)
                metrics = {
                    "psnr": round(float(eval_res["psnr"]), 6),
                    "ssim": round(float(eval_res["ssim"]), 6),
                    "l1": round(float(eval_res["l1"]), 6),
                    "reference_provided": True,
                }
                metrics_status = "measured_against_paired_hr_reference"
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid reference_hr file: {exc}"
                ) from exc

    return {
        "task_type": "superresolution",
        "client_id": client_id,
        "scale": int(scale),
        "input_shape": list(lr_tensor.shape),
        "output_shape": list(sr_tensor.shape),
        "input_base64": encode_tensor_to_base64_png(lr_tensor),
        "enhanced_base64": encode_tensor_to_base64_png(sr_tensor),
        "metrics": metrics,
        "metrics_status": metrics_status,
        "model_version": model_meta["model_version"],
        "model_metadata": model_meta,
        "preprocessing": prep_meta,
    }


@router.post("/api/predict/fusion")
async def predict_fusion(
    request: Request,
    file_a: UploadFile = File(..., description="Source modality A image (e.g., CT, PET, SPECT)"),
    file_b: UploadFile = File(..., description="Source modality B image (e.g., MRI)"),
    modality_pair: str = Form(default="CT-MRI", description="CT-MRI | PET-MRI | SPECT-MRI"),
    is_registered: bool = Form(
        default=True, description="Whether the uploaded source images are spatially co-registered"
    ),
    experiment_id: str | None = Form(default=None),
    round_num: int | None = Form(default=None),
    allow_untrained_fallback: bool = Form(default=True),
) -> dict[str, Any]:
    """Run multi-modal image fusion for a verified pair of source modalities."""
    pair_key = modality_pair.upper().strip()
    if pair_key not in VALID_MODALITY_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported modality_pair '{modality_pair}'. Valid options: ['CT-MRI', 'PET-MRI', 'SPECT-MRI'].",
        )
    resolved_client = VALID_MODALITY_PAIRS[pair_key]

    bytes_a = await file_a.read()
    bytes_b = await file_b.read()
    try:
        tensor_a, meta_a = decode_image_bytes_to_tensor(bytes_a, channels=1, target_size=(128, 128))
        tensor_b, meta_b = decode_image_bytes_to_tensor(bytes_b, channels=1, target_size=(128, 128))
        adapter, model_meta = resolve_and_load_adapter(
            task_type="fusion",
            experiment_id=experiment_id,
            round_num=round_num,
            client_id=resolved_client,
            allow_untrained_fallback=allow_untrained_fallback,
            database_url=_get_db_url(request),
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stacked_pair = torch.cat([tensor_a, tensor_b], dim=1)  # (1, 2, 128, 128)
    with torch.no_grad():
        fwd = adapter.forward_features_and_predict(stacked_pair)
        fused_tensor = torch.clamp(fwd.prediction.float(), 0.0, 1.0)

    # Compute source-similarity and no-reference metrics only when images are declared registered
    metrics: dict[str, Any] | None = None
    metrics_status = "unregistered_pair_metrics_withheld"
    if is_registered:
        eval_res = compute_fusion_metrics(fused_tensor, paired_sources=stacked_pair)
        metrics = {
            "ssim_to_sources": round(float(eval_res["ssim"]), 6),
            "vif_to_sources": round(float(eval_res["vif"]), 6),
            "l1_to_sources": round(float(eval_res["l1"]), 6),
            "spatial_frequency_no_ref": round(float(eval_res["spatial_frequency_no_ref"]), 6),
            "metric_protocol": eval_res["metric_protocol"],
        }
        metrics_status = "computed_on_registered_source_pair"

    return {
        "task_type": "fusion",
        "client_id": resolved_client,
        "modality_pair": pair_key,
        "is_registered": bool(is_registered),
        "registration_note": (
            "Co-registered 128x128 axial slice pair."
            if is_registered
            else "Unregistered input pair: source-comparison metrics withheld."
        ),
        "source_a_base64": encode_tensor_to_base64_png(tensor_a),
        "source_b_base64": encode_tensor_to_base64_png(tensor_b),
        "fused_base64": encode_tensor_to_base64_png(fused_tensor),
        "metrics": metrics,
        "metrics_status": metrics_status,
        "model_version": model_meta["model_version"],
        "model_metadata": model_meta,
        "preprocessing": {"source_a": meta_a, "source_b": meta_b},
    }
