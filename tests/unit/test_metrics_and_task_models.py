"""Unit and real-batch smoke tests for Phase 4 (Tasks 4.1 - 4.6): Task models, metrics, and VRAM measurement."""

from __future__ import annotations

import unittest

import torch

from backend.config import settings
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks import create_task_adapter
from backend.utils.metrics import (
    compute_accuracy,
    compute_dice,
    compute_fusion_metrics,
    compute_iou,
    compute_l1,
    compute_macro_f1,
    compute_psnr,
    compute_ssim,
    compute_vif,
    evaluate_task_predictions,
)
from backend.utils.preprocessing import build_client_dataloader


class TestMetricsAndTaskModels(unittest.TestCase):
    # =========================================================================
    # Task 4.5: Known Toy Examples & Reference-Requirement Guardrails
    # =========================================================================
    def test_metrics_known_toy_examples_and_reference_guards(self) -> None:
        # 1. Classification: Accuracy & Macro-F1
        preds = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2, 2])
        # Correct on indices 0, 1, 3 -> 3/4 = 0.75
        self.assertAlmostEqual(compute_accuracy(preds, targets), 0.75, places=6)
        # Class 0: TP=1, FP=0, FN=0 -> F1=1.0
        # Class 1: TP=1, FP=1, FN=0 -> F1=2/3
        # Class 2: TP=1, FP=0, FN=1 -> F1=2/3
        # Macro-F1 = (1 + 2/3 + 2/3) / 3 = 7/9
        self.assertAlmostEqual(compute_macro_f1(preds, targets, num_classes=3), 7.0 / 9.0, places=5)

        # 2. Segmentation: Dice & IoU
        mask_a = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        mask_b = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        # Intersection = 1, Sum = 4 -> Dice = 2/4 = 0.5; Union = 3 -> IoU = 1/3
        self.assertAlmostEqual(compute_dice(mask_a, mask_b), 0.5, places=5)
        self.assertAlmostEqual(compute_iou(mask_a, mask_b), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(compute_dice(mask_a, mask_a), 1.0, places=5)
        self.assertAlmostEqual(compute_iou(mask_a, mask_a), 1.0, places=5)

        # 3. Super-Resolution: PSNR, SSIM, VIF, L1
        img_ref = torch.rand(2, 3, 32, 32)
        self.assertAlmostEqual(compute_l1(img_ref, img_ref), 0.0, places=6)
        self.assertEqual(compute_psnr(img_ref, img_ref), 100.0)
        self.assertAlmostEqual(compute_ssim(img_ref, img_ref), 1.0, places=4)
        self.assertAlmostEqual(compute_vif(img_ref, img_ref), 1.0, places=4)

        # Known MSE = 0.01 -> PSNR = 10 * log10(1.0 / 0.01) = 20.0 dB
        img_shifted = torch.clamp(img_ref * 0.0 + 0.5, 0.0, 1.0)
        img_noisy = img_shifted + 0.1
        self.assertAlmostEqual(compute_psnr(img_noisy, img_shifted), 20.0, places=4)
        self.assertAlmostEqual(compute_l1(img_noisy, img_shifted), 0.1, places=5)

        # 4. Reference-required guardrails: passing target=None must raise ValueError
        for fn in (
            compute_accuracy,
            compute_macro_f1,
            compute_dice,
            compute_iou,
            compute_psnr,
            compute_ssim,
            compute_vif,
            compute_l1,
        ):
            with self.assertRaises(ValueError):
                fn(img_ref, None)

        # 5. Fusion no-reference vs paired-source protocol identification
        fused_sample = torch.rand(1, 1, 32, 32)
        no_ref_metrics = compute_fusion_metrics(fused_sample, paired_sources=None)
        self.assertEqual(no_ref_metrics["metric_protocol"], "no_reference_only")
        self.assertFalse(no_ref_metrics["has_source_pair_reference"])
        self.assertNotIn("ssim", no_ref_metrics)

        pair_sample = torch.rand(1, 2, 32, 32)
        pair_metrics = compute_fusion_metrics(fused_sample, paired_sources=pair_sample)
        self.assertEqual(
            pair_metrics["metric_protocol"], "paired_source_preservation_plus_no_ref"
        )
        self.assertTrue(pair_metrics["has_source_pair_reference"])
        self.assertIn("ssim", pair_metrics)
        self.assertIn("vif", pair_metrics)

    # =========================================================================
    # Tasks 4.1 - 4.4 & 4.6: Real-Batch Forward/Backward + AMP + VRAM Check
    # =========================================================================
    def test_real_batch_forward_backward_per_task_with_vram_measurement(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda" and settings.precision_policy == "fp16_amp"

        bank = SpectralKnowledgeBank(
            embedding_dim=settings.embedding_dim,
            max_size=100,
            top_k=settings.gskr_top_k,
        )
        for t in ("classification", "segmentation", "superresolution", "fusion"):
            bank.add_prototypes(
                torch.randn(4, settings.embedding_dim),
                task_type=t,
                client_id="seed_client",
                round_num=0,
            )

        task_configs = [
            ("classification", {"batch_size": 4, "max_samples": 4}),
            ("segmentation", {"batch_size": 2, "max_samples": 2}),
            ("superresolution", {"batch_size": 2, "max_samples": 2, "scale": 2}),
            ("fusion", {"batch_size": 2, "max_samples": 2}),
        ]

        print(f"\n=== Task 4.6 Real-Batch Forward/Backward Smoke Report (device={device}, precision={settings.precision_policy}) ===")
        for task_name, loader_kwargs in task_configs:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            loader, _stats = build_client_dataloader(
                task_type=task_name,
                client_id="client_1",
                split="train",
                **loader_kwargs,
            )
            batch_x, batch_y = next(iter(loader))
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            adapter = create_task_adapter(
                task_name,
                embedding_dim=settings.embedding_dim,
                lambda_align=settings.spalign_lambda,
            ).to(device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                fw_out = adapter(batch_x, knowledge_bank=bank, client_id="client_1")
                loss_breakdown = adapter.compute_total_loss(
                    fw_out, batch_y, inputs=batch_x
                )

            scaler.scale(loss_breakdown.total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            self.assertTrue(torch.isfinite(loss_breakdown.total_loss).item())
            self.assertTrue(torch.isfinite(fw_out.prediction).all().item())

            metrics = evaluate_task_predictions(
                task_type=task_name,
                predictions=fw_out.prediction,
                targets=batch_y,
                num_classes=fw_out.extra.get("num_classes"),
            )

            peak_vram_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"Task={task_name:15s} | PeakVRAM={peak_vram_mb:7.2f} MB | "
                f"Loss={loss_breakdown.to_log_dict()} | Metrics={metrics}"
            )

            del adapter, optimizer, batch_x, batch_y, fw_out
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
