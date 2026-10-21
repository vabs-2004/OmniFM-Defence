"""Unit and smoke tests for Phase 2 (Tasks 2.1 - 2.7): OmniFM components and task adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from backend.omnifm.eca import EmbeddingCrossAttention
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.omnifm.psp import PERSONALIZED_PROMPT_PARAM_NAME, PrefixSuffixSpectralPrompting
from backend.omnifm.spalign import SPAlignLoss
from backend.omnifm.spectral import SpectralEncoder
from backend.tasks import (
    ClassificationTaskAdapter,
    FusionTaskAdapter,
    SegmentationTaskAdapter,
    SuperResolutionTaskAdapter,
    create_task_adapter,
)


class TestOmniFMComponents(unittest.TestCase):
    # =========================================================================
    # Task 2.1: SpectralEncoder
    # =========================================================================
    def test_spectral_encoder_shapes_odd_even_channels_and_gradients(self) -> None:
        for dim in (64, 128):
            encoder = SpectralEncoder(embedding_dim=dim, low_pass_ratio=0.25)
            # Test RGB/grayscale/paired and odd/even dimensions + batch size 1
            for shape in [(1, 1, 27, 29), (2, 3, 28, 28), (2, 2, 64, 64), (1, 3, 65, 63)]:
                x = torch.randn(*shape, requires_grad=True)
                out = encoder(x)
                self.assertEqual(out.shape, (shape[0], dim))
                self.assertTrue(torch.isfinite(out).all())

                # Verify unit L2 norm
                norms = out.norm(p=2, dim=-1)
                self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

                loss = out.sum()
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                encoder.zero_grad()

    # =========================================================================
    # Task 2.2: GSKR & SpectralKnowledgeBank
    # =========================================================================
    def test_gskr_empty_one_undersized_full_reload_and_self_retrieval_protection(self) -> None:
        bank = SpectralKnowledgeBank(embedding_dim=32, max_size=4, top_k=3)
        q = torch.randn(2, 32)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)

        # 1. Empty bank
        res_empty = bank.retrieve(q, top_k=3)
        self.assertFalse(res_empty.has_valid)
        self.assertEqual(res_empty.prototypes.shape, (2, 0, 32))

        # 2. One prototype (fewer than k=3)
        p1 = torch.nn.functional.normalize(torch.randn(1, 32), p=2, dim=-1)
        bank.add_prototypes(p1, task_type="classification", client_id="client_1", round_num=1)
        res_one = bank.retrieve(q, top_k=3, task_type="classification")
        self.assertTrue(res_one.has_valid)
        self.assertEqual(res_one.prototypes.shape, (2, 1, 32))

        # 3. Task-aware filtering: querying another task returns empty
        res_other_task = bank.retrieve(q, top_k=3, task_type="segmentation")
        self.assertFalse(res_other_task.has_valid)
        self.assertEqual(res_other_task.prototypes.shape, (2, 0, 32))

        # 4. Self-retrieval protection: if query q[0] is added to bank, avoid_exact_self masks it
        bank.add_prototypes(q[0:1], task_type="classification", client_id="client_2", round_num=1)
        res_no_self = bank.retrieve(q[0:1], top_k=3, task_type="classification", avoid_exact_self=True)
        self.assertTrue(res_no_self.has_valid)
        self.assertEqual(res_no_self.prototypes.shape, (1, 1, 32))
        self.assertFalse(torch.allclose(res_no_self.prototypes[0, 0], q[0], atol=1e-4))

        # 5. Capacity pruning (max_size=4) and save/load determinism
        extra = torch.randn(5, 32)
        bank.add_prototypes(extra, task_type="classification", client_id="client_3", round_num=2)
        self.assertEqual(len(bank), 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "knowledge_bank.pt"
            bank.save(save_path)
            reloaded = SpectralKnowledgeBank.load(save_path)
            self.assertEqual(len(reloaded), 4)
            self.assertTrue(torch.equal(bank.embeddings, reloaded.embeddings))

            r_orig = bank.retrieve(q, top_k=2, task_type="classification")
            r_load = reloaded.retrieve(q, top_k=2, task_type="classification")
            self.assertTrue(torch.allclose(r_orig.prototypes, r_load.prototypes, atol=1e-6))

    # =========================================================================
    # Task 2.3: EmbeddingCrossAttention (ECA)
    # =========================================================================
    def test_eca_shapes_and_backward(self) -> None:
        eca = EmbeddingCrossAttention(embedding_dim=64, num_heads=4)
        q_backbone = torch.randn(2, 16, 64, requires_grad=True)
        retrieved = torch.randn(2, 2, 64)
        local_s = torch.randn(2, 64, requires_grad=True)

        fused, prefix = eca(q_backbone, retrieved_prototypes=retrieved, local_spectral_token=local_s)
        self.assertEqual(fused.shape, (2, 16, 64))
        self.assertEqual(prefix.shape, (2, 1, 64))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.isfinite(prefix).all())

        (fused.sum() + prefix.sum()).backward()
        self.assertIsNotNone(q_backbone.grad)
        self.assertIsNotNone(local_s.grad)
        self.assertTrue(torch.isfinite(q_backbone.grad).all())

    # =========================================================================
    # Task 2.4: PrefixSuffixSpectralPrompting (PSP)
    # =========================================================================
    def test_psp_token_alignment_and_personalized_parameter_identification(self) -> None:
        psp = PrefixSuffixSpectralPrompting(
            embedding_dim=64,
            num_prefix_tokens=2,
            num_suffix_tokens=3,
            num_heads=4,
        )
        backbone_tokens = torch.randn(2, 16, 64, requires_grad=True)
        prefix_tokens = torch.randn(2, 2, 64, requires_grad=True)

        out = psp(backbone_tokens, prefix_tokens=prefix_tokens)
        self.assertEqual(out.prompted_sequence.shape, (2, 2 + 16 + 3, 64))
        self.assertEqual(out.spatial_tokens.shape, (2, 16, 64))
        self.assertEqual(out.pooled_token.shape, (2, 64))

        (out.spatial_tokens.sum() + out.pooled_token.sum()).backward()
        self.assertIsNotNone(psp.client_suffix_prompt.grad)
        self.assertTrue(torch.isfinite(psp.client_suffix_prompt.grad).all())

        pers = psp.get_personalized_parameter_names(prefix="omnifm.psp")
        self.assertIn(f"omnifm.psp.{PERSONALIZED_PROMPT_PARAM_NAME}", pers)
        self.assertTrue(
            PrefixSuffixSpectralPrompting.is_personalized_parameter(
                f"omnifm.psp.{PERSONALIZED_PROMPT_PARAM_NAME}"
            )
        )
        self.assertFalse(
            PrefixSuffixSpectralPrompting.is_personalized_parameter(
                "omnifm.psp.shared_prefix_embed"
            )
        )

    # =========================================================================
    # Task 2.5: SPAlignLoss
    # =========================================================================
    def test_spalign_valid_and_no_prototype_behavior(self) -> None:
        spalign = SPAlignLoss(lambda_align=0.1)
        task_loss = torch.tensor(1.5, requires_grad=True)
        s_local = torch.zeros(2, 16, requires_grad=True)

        # Case 1: Empty / no prototypes -> align_loss == 0, total_loss == task_loss
        empty_protos = torch.empty((2, 0, 16))
        b_empty = spalign(task_loss, local_spectral_tokens=s_local, retrieved_prototypes=empty_protos)
        self.assertFalse(b_empty.has_valid_prototypes)
        self.assertAlmostEqual(float(b_empty.align_loss.item()), 0.0)
        self.assertAlmostEqual(float(b_empty.total_loss.item()), 1.5, places=6)

        # Case 2: Valid prototypes -> mean(||0 - 1||^2) over 16 dims = 16.0
        valid_protos = torch.ones((2, 2, 16))
        b_valid = spalign(task_loss, local_spectral_tokens=s_local, retrieved_prototypes=valid_protos)
        self.assertTrue(b_valid.has_valid_prototypes)
        self.assertAlmostEqual(float(b_valid.align_loss.item()), 16.0, places=5)
        self.assertAlmostEqual(float(b_valid.total_loss.item()), 1.5 + 0.1 * 16.0, places=5)

        log_dict = b_valid.to_log_dict()
        self.assertIn("task_loss", log_dict)
        self.assertIn("align_loss", log_dict)
        self.assertIn("total_loss", log_dict)

        b_valid.total_loss.backward()
        self.assertIsNotNone(s_local.grad)
        self.assertTrue(torch.isfinite(s_local.grad).all())

    # =========================================================================
    # Tasks 2.6 & 2.7: Task Adapters Forward/Backward Smoke Test (CPU + CUDA)
    # =========================================================================
    def test_all_four_task_adapters_forward_backward_and_parameter_policies(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            bank = SpectralKnowledgeBank(embedding_dim=64, max_size=50, top_k=2)
            for t_name in ("classification", "segmentation", "superresolution", "fusion"):
                bank.add_prototypes(
                    torch.randn(3, 64),
                    task_type=t_name,
                    client_id="client_seed",
                    round_num=0,
                )

            # 1. Classification (28x28 RGB)
            cls_adapter = ClassificationTaskAdapter(embedding_dim=64, num_heads=4).to(device)
            x_cls = torch.randn(2, 3, 28, 28, device=device)
            y_cls = torch.tensor([0, 3], dtype=torch.long, device=device)
            out_cls = cls_adapter(x_cls, knowledge_bank=bank, dataset_name="dermamnist")
            self.assertEqual(out_cls.prediction.shape, (2, 7))
            loss_cls = cls_adapter.compute_total_loss(out_cls, y_cls)
            loss_cls.total_loss.backward()
            # Verify heads and suffix prompts are excluded from shared parameters
            shared_cls = cls_adapter.get_shared_parameter_names()
            pers_cls = cls_adapter.get_personalized_parameter_names()
            self.assertTrue(any("heads.dermamnist" in p for p in pers_cls))
            self.assertTrue(any(PERSONALIZED_PROMPT_PARAM_NAME in p for p in pers_cls))
            self.assertFalse(any("heads." in s for s in shared_cls))

            # 2. Segmentation (128x128 1-ch MRI)
            seg_adapter = SegmentationTaskAdapter(
                base_channels=16, embedding_dim=64, num_heads=4
            ).to(device)
            x_seg = torch.randn(2, 1, 128, 128, device=device)
            y_seg = (torch.rand(2, 1, 128, 128, device=device) > 0.5).float()
            out_seg = seg_adapter(x_seg, knowledge_bank=bank)
            self.assertEqual(out_seg.prediction.shape, (2, 1, 128, 128))
            loss_seg = seg_adapter.compute_total_loss(out_seg, y_seg)
            loss_seg.total_loss.backward()

            # 3. Super-Resolution (64x64 3-ch -> x2 128x128 and x4 256x256)
            sr_adapter = SuperResolutionTaskAdapter(
                feat_channels=32, num_blocks=2, default_scale=2, embedding_dim=64, num_heads=4
            ).to(device)
            x_sr = torch.rand(1, 3, 64, 64, device=device)
            y_sr = torch.rand(1, 3, 128, 128, device=device)
            out_sr = sr_adapter(x_sr, knowledge_bank=bank, scale=2)
            self.assertEqual(out_sr.prediction.shape, (1, 3, 128, 128))
            loss_sr = sr_adapter.compute_total_loss(out_sr, y_sr)
            loss_sr.total_loss.backward()

            # 4. Fusion (128x128 2-ch paired -> 128x128 1-ch fused)
            fus_adapter = FusionTaskAdapter(
                feat_channels=32, embedding_dim=64, num_heads=4
            ).to(device)
            x_fus = torch.rand(2, 2, 128, 128, device=device)
            out_fus = fus_adapter(x_fus, knowledge_bank=bank)
            self.assertEqual(out_fus.prediction.shape, (2, 1, 128, 128))
            loss_fus = fus_adapter.compute_total_loss(out_fus, x_fus, inputs=x_fus)
            loss_fus.total_loss.backward()

            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                print(f"\n[Task 2.7 CUDA Smoke Check] Measured Peak VRAM: {peak_mb:.2f} MB")
                self.assertGreater(peak_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
