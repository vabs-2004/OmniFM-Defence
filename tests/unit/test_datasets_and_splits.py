"""Unit tests for Phase 3 (Tasks 3.1 - 3.5): Dataset loaders, splits, shapes, value ranges, and leakage checks."""

from __future__ import annotations

import unittest

import torch

from backend.utils.preprocessing import (
    BreaKHisSRDataset,
    FeTS2022SegmentationDataset,
    MedMNISTClassificationDataset,
    MultimodalFusionDataset,
    build_client_dataloader,
)


class TestDatasetLoadersAndSplits(unittest.TestCase):
    def test_medmnist_official_splits_and_distinct_label_semantics(self) -> None:
        expected_classes = {"dermamnist": 7, "pathmnist": 9, "bloodmnist": 8}
        for ds_name, n_cls in expected_classes.items():
            ds_train = MedMNISTClassificationDataset(
                dataset_name=ds_name, split="train", max_samples=16
            )
            ds_val = MedMNISTClassificationDataset(
                dataset_name=ds_name, split="val", max_samples=8
            )
            ds_test = MedMNISTClassificationDataset(
                dataset_name=ds_name, split="test", max_samples=8
            )

            self.assertEqual(len(ds_train), 16)
            self.assertEqual(len(ds_val), 8)
            self.assertEqual(len(ds_test), 8)
            self.assertEqual(ds_train.num_classes, n_cls)

            x, y = ds_train[0]
            self.assertEqual(x.shape, (3, 28, 28))
            self.assertEqual(y.ndim, 0)
            self.assertTrue(0 <= int(y.item()) < n_cls)
            self.assertGreaterEqual(float(x.min().item()), -1.0001)
            self.assertLessEqual(float(x.max().item()), 1.0001)

            stats = ds_train.aggregate_statistics()
            self.assertEqual(stats["num_classes"], n_cls)
            self.assertEqual(stats["num_samples"], 16)

    def test_fets2022_institution_and_subject_level_split_no_leakage(self) -> None:
        # Verify disjoint subjects across train/val/test for client_1 (Institution 1)
        ds_train = FeTS2022SegmentationDataset(
            client_id="client_1", split="train", max_subjects=4, slices_per_subject=1
        )
        ds_val = FeTS2022SegmentationDataset(
            client_id="client_1", split="val", max_subjects=2, slices_per_subject=1
        )
        ds_test = FeTS2022SegmentationDataset(
            client_id="client_1", split="test", max_subjects=2, slices_per_subject=1
        )

        self.assertEqual(
            len(ds_train.anonymized_subject_hashes & ds_val.anonymized_subject_hashes), 0
        )
        self.assertEqual(
            len(ds_train.anonymized_subject_hashes & ds_test.anonymized_subject_hashes), 0
        )
        self.assertEqual(
            len(ds_val.anonymized_subject_hashes & ds_test.anonymized_subject_hashes), 0
        )

        # Verify disjoint subjects across clients (Institution 1 vs Institution 18 vs Institution 4)
        c2_train = FeTS2022SegmentationDataset(
            client_id="client_2", split="train", max_subjects=2, slices_per_subject=1
        )
        c3_train = FeTS2022SegmentationDataset(
            client_id="client_3", split="train", max_subjects=2, slices_per_subject=1
        )
        self.assertEqual(
            len(ds_train.anonymized_subject_hashes & c2_train.anonymized_subject_hashes), 0
        )
        self.assertEqual(
            len(ds_train.anonymized_subject_hashes & c3_train.anonymized_subject_hashes), 0
        )

        img, mask = ds_train[0]
        self.assertEqual(img.shape, (1, 128, 128))
        self.assertEqual(mask.shape, (1, 128, 128))
        self.assertGreaterEqual(float(img.min().item()), 0.0)
        self.assertLessEqual(float(img.max().item()), 1.0001)
        unique_mask = set(torch.unique(mask).tolist())
        self.assertTrue(unique_mask.issubset({0.0, 1.0}))
        # Verify non-empty tumor slice policy found tumor voxels
        self.assertGreater(float(mask.sum().item()), 0.0)

    def test_breakhis_patient_level_split_no_leakage_and_multi_scale_shapes(self) -> None:
        ds_train = BreaKHisSRDataset(
            client_id="client_1", split="train", scale=2, max_samples=6
        )
        ds_val = BreaKHisSRDataset(
            client_id="client_1", split="val", scale=2, max_samples=4
        )
        ds_test = BreaKHisSRDataset(
            client_id="client_1", split="test", scale=2, max_samples=4
        )

        # Zero patient overlap across train, val, and test splits
        self.assertEqual(
            len(ds_train.anonymized_patient_hashes & ds_val.anonymized_patient_hashes), 0
        )
        self.assertEqual(
            len(ds_train.anonymized_patient_hashes & ds_test.anonymized_patient_hashes), 0
        )
        self.assertEqual(
            len(ds_val.anonymized_patient_hashes & ds_test.anonymized_patient_hashes), 0
        )

        # Verify x2, x4, x8 paired LR/HR shapes and value ranges
        for scale in (2, 4, 8):
            ds_s = BreaKHisSRDataset(
                client_id="client_2", split="train", scale=scale, max_samples=2
            )
            lr, hr = ds_s[0]
            self.assertEqual(lr.shape, (3, 64, 64))
            self.assertEqual(hr.shape, (3, 64 * scale, 64 * scale))
            self.assertGreaterEqual(float(lr.min().item()), 0.0)
            self.assertLessEqual(float(lr.max().item()), 1.0001)
            self.assertGreaterEqual(float(hr.min().item()), 0.0)
            self.assertLessEqual(float(hr.max().item()), 1.0001)

    def test_fusion_verified_modality_pairs_and_shapes(self) -> None:
        for cid, expected_pair in [
            ("client_1", "CT-MRI"),
            ("client_2", "PET-MRI"),
            ("client_3", "SPECT-MRI"),
        ]:
            ds = MultimodalFusionDataset(client_id=cid, split="train", max_samples=4)
            self.assertEqual(ds.modality_pair, expected_pair)
            self.assertEqual(len(ds), 4)
            x, y = ds[0]
            self.assertEqual(x.shape, (2, 128, 128))
            self.assertEqual(y.shape, (2, 128, 128))
            self.assertGreaterEqual(float(x.min().item()), 0.0)
            self.assertLessEqual(float(x.max().item()), 1.0001)

    def test_one_batch_from_each_loader_via_build_client_dataloader(self) -> None:
        for task, expected_x, expected_y in [
            ("classification", (2, 3, 28, 28), (2,)),
            ("segmentation", (2, 1, 128, 128), (2, 1, 128, 128)),
            ("superresolution", (2, 3, 64, 64), (2, 3, 128, 128)),
            ("fusion", (2, 2, 128, 128), (2, 2, 128, 128)),
        ]:
            loader, stats = build_client_dataloader(
                task_type=task,
                client_id="client_1",
                split="train",
                batch_size=2,
                max_samples=2,
                scale=2,
            )
            batch_x, batch_y = next(iter(loader))
            self.assertEqual(tuple(batch_x.shape), expected_x)
            self.assertEqual(tuple(batch_y.shape), expected_y)
            self.assertTrue(torch.isfinite(batch_x).all())
            self.assertTrue(torch.isfinite(batch_y).all())
            self.assertIsInstance(stats, dict)


if __name__ == "__main__":
    unittest.main()
