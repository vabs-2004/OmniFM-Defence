"""Unit tests for Phase 5 (Tasks 5.1 - 5.5, 5.8) and Phase 6 (Task 6.1):
Parameter-sharing policy, weighted FedAvg, RoundManager state machine, SQLite persistence,
and algorithmic verification of FedProx and FedPer baselines.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import select
import torch

from backend.config import load_settings
from backend.database.db import session_scope
from backend.database.models import (
    ArtifactMetadata,
    ClientUpdate,
    Experiment,
    GlobalModelVersion,
    MetricRecord,
    TrainingRound,
)
from backend.fl.aggregation import (
    AggregationValidationError,
    weighted_fedavg,
)
from backend.fl.client import FederatedHospitalClient
from backend.fl.protocols import (
    ClientUpdatePayload,
    build_parameter_sharing_policy,
)
from backend.fl.round_manager import RoundManager, RoundStateTransitionError
from backend.fl.server import FederatedLearningServer
from backend.tasks import ClassificationTaskAdapter, SegmentationTaskAdapter


class TestFLAggregationAndBaselines(unittest.TestCase):
    # =========================================================================
    # Task 5.1 & 5.3: Parameter-Sharing Policy & Weighted FedAvg Toy-Tensor Tests
    # =========================================================================
    def test_parameter_sharing_policy_and_weighted_fedavg_toy_tensors(self) -> None:
        cls_adapter = ClassificationTaskAdapter(embedding_dim=32, num_heads=4)
        policy = build_parameter_sharing_policy(cls_adapter, algorithm="omnifm")
        manifest = policy.to_manifest_dict()

        self.assertGreater(manifest["shared_count"], 0)
        self.assertEqual(manifest["personalized_count"], 1)  # client_suffix_prompt
        self.assertEqual(manifest["incompatible_head_count"], 6)  # 3 heads * (weight, bias)

        # Toy tensor weighted FedAvg: n1=10, n2=30 -> weights 0.25 and 0.75
        state_1 = {
            "backbone.conv1.weight": torch.ones((2, 2), dtype=torch.float32) * 2.0,
            "omnifm.psp.client_suffix_prompt": torch.ones((1, 1, 32), dtype=torch.float32) * 100.0,
            "heads.dermamnist.weight": torch.ones((7, 10), dtype=torch.float32),
        }
        state_2 = {
            "backbone.conv1.weight": torch.ones((2, 2), dtype=torch.float32) * 6.0,
            "omnifm.psp.client_suffix_prompt": torch.ones((1, 1, 32), dtype=torch.float32) * 200.0,
            "heads.pathmnist.weight": torch.ones((9, 10), dtype=torch.float32),
        }

        agg = weighted_fedavg([state_1, state_2], sample_counts=[10, 30], policy=policy)
        # Excluded keys must NOT be in aggregated state
        self.assertNotIn("omnifm.psp.client_suffix_prompt", agg)
        self.assertNotIn("heads.dermamnist.weight", agg)
        self.assertNotIn("heads.pathmnist.weight", agg)

        # Weighted average: 0.25 * 2.0 + 0.75 * 6.0 = 0.5 + 4.5 = 5.0
        expected = torch.ones((2, 2), dtype=torch.float32) * 5.0
        self.assertTrue(torch.allclose(agg["backbone.conv1.weight"], expected, atol=1e-6))

        # Validation: NaN/Inf and invalid sample count must raise AggregationValidationError
        with self.assertRaises(AggregationValidationError):
            weighted_fedavg([state_1], sample_counts=[0], policy=policy)

        bad_nan = {"backbone.conv1.weight": torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])}
        with self.assertRaises(AggregationValidationError):
            weighted_fedavg([bad_nan], sample_counts=[5])

    # =========================================================================
    # Task 5.4: RoundManager Duplicate Submission, Quorum, and Duplicate Aggregation
    # =========================================================================
    def test_round_manager_guards_incomplete_and_duplicate_aggregation(self) -> None:
        seg_adapter = SegmentationTaskAdapter(base_channels=8, embedding_dim=32, num_heads=4)
        policy = build_parameter_sharing_policy(seg_adapter, algorithm="omnifm")
        shared_state = seg_adapter.extract_shared_state_dict()

        mgr = RoundManager(
            experiment_id="exp_mgr_test",
            round_num=1,
            expected_client_ids=("client_1", "client_2"),
            policy=policy,
            require_all_clients=True,
        )

        u1 = ClientUpdatePayload(
            experiment_id="exp_mgr_test",
            round_num=1,
            client_id="client_1",
            task_type="segmentation",
            algorithm="omnifm",
            num_samples=4,
            shared_state_dict=shared_state,
            update_hash="0x" + "a" * 64,
            spectral_prototype=torch.randn(1, 32),
            train_loss=0.5,
            task_loss=0.4,
            align_loss=0.1,
            eval_metrics={"dice": 0.5},
            duration_sec=0.1,
            peak_vram_mb=10.0,
        )
        mgr.accept_client_update(u1)
        self.assertFalse(mgr.check_quorum_met())

        # Duplicate submission from client_1 must fail
        with self.assertRaises(AggregationValidationError):
            mgr.accept_client_update(u1)

        # Incomplete round aggregation must fail
        with self.assertRaises(RoundStateTransitionError):
            mgr.validate_and_aggregate()

        u2 = ClientUpdatePayload(
            experiment_id="exp_mgr_test",
            round_num=1,
            client_id="client_2",
            task_type="segmentation",
            algorithm="omnifm",
            num_samples=6,
            shared_state_dict=shared_state,
            update_hash="0x" + "b" * 64,
            spectral_prototype=torch.randn(1, 32),
            train_loss=0.4,
            task_loss=0.3,
            align_loss=0.1,
            eval_metrics={"dice": 0.6},
            duration_sec=0.1,
            peak_vram_mb=10.0,
        )
        mgr.accept_client_update(u2)
        self.assertTrue(mgr.check_quorum_met())

        agg_state = mgr.validate_and_aggregate()
        self.assertGreater(len(agg_state), 0)

        # Duplicate aggregation must fail
        with self.assertRaises(RoundStateTransitionError):
            mgr.validate_and_aggregate()

    # =========================================================================
    # Task 5.8: Algorithmic Verification of FedProx and FedPer Baselines
    # =========================================================================
    def test_fedprox_and_fedper_implement_true_algorithmic_behavior(self) -> None:
        seg_adapter = SegmentationTaskAdapter(base_channels=8, embedding_dim=32, num_heads=4)

        # 1. Verify FedPer policy excludes decoder head (`unet.out_head.*`, `unet.dec1.*`)
        #    while FedAvg shares them.
        policy_fedavg = build_parameter_sharing_policy(seg_adapter, algorithm="fedavg")
        policy_fedper = build_parameter_sharing_policy(seg_adapter, algorithm="fedper")
        self.assertIn("unet.out_head.weight", policy_fedavg.shared_param_names)
        self.assertNotIn("unet.out_head.weight", policy_fedper.shared_param_names)
        self.assertIn("unet.out_head.weight", policy_fedper.personalized_param_names)

        # 2. Verify FedProx proximal penalty term `(mu / 2) * ||w - w_global||^2`
        #    is strictly positive and produces restoring gradients toward `w_global`
        #    when local weights diverge from `w_global`.
        global_ref = {
            k: v.detach().clone()
            for k, v in seg_adapter.state_dict().items()
            if k in policy_fedavg.shared_param_names
        }
        # Zero diff -> proximal penalty == 0
        prox_zero = FederatedHospitalClient.compute_fedprox_term(
            seg_adapter, global_ref, policy_fedavg.shared_param_names, mu=0.1
        )
        self.assertAlmostEqual(float(prox_zero.item()), 0.0, places=6)

        # Perturb a shared weight by +1.0 -> proximal penalty > 0 and gradient > 0
        with torch.no_grad():
            for name, param in seg_adapter.named_parameters():
                if name == "unet.enc1.block.0.weight":
                    param.add_(1.0)
                    num_el = param.numel()
                    break

        seg_adapter.zero_grad()
        prox_pos = FederatedHospitalClient.compute_fedprox_term(
            seg_adapter, global_ref, policy_fedavg.shared_param_names, mu=0.1
        )
        # Expected: 0.5 * 0.1 * (num_el * 1.0^2) = 0.05 * num_el
        self.assertAlmostEqual(float(prox_pos.item()), 0.05 * num_el, places=4)
        prox_pos.backward()
        for name, param in seg_adapter.named_parameters():
            if name == "unet.enc1.block.0.weight":
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.allclose(param.grad, torch.ones_like(param) * 0.1, atol=1e-5))

    # =========================================================================
    # Task 5.5 & 6.1: SQLite Persistence & Checkpoint/Bank Reload Survival
    # =========================================================================
    def test_sqlite_persistence_and_checkpoint_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_url = f"sqlite:///{(tmp_path / 'test_omnifm.db').as_posix()}"
            cfg = load_settings(
                env_file=None,
                overrides={
                    "database_url": db_url,
                    "artifact_dir": str(tmp_path / "artifacts"),
                    "knowledge_bank_path": str(tmp_path / "bank" / "knowledge_bank.pt"),
                    "embedding_dim": 32,
                    "eca_num_heads": 4,
                },
            )

            server = FederatedLearningServer(
                experiment_id="exp_persist_01",
                task_type="classification",
                algorithm="omnifm",
                client_ids=("client_1",),
                cfg=cfg,
                database_url=db_url,
            )
            res = server.run_round(
                round_num=1,
                local_epochs=1,
                batch_size=4,
                max_train_samples=4,
                max_val_samples=4,
                device="cpu",
            )
            self.assertEqual(res.state.value, "COMPLETED")
            self.assertEqual(res.knowledge_bank_size, 1)

            # Verify checkpoint and bank reload on a fresh server instance
            reloaded_server = FederatedLearningServer(
                experiment_id="exp_persist_01",
                task_type="classification",
                algorithm="omnifm",
                client_ids=("client_1",),
                cfg=cfg,
                database_url=db_url,
            )
            payload = reloaded_server.reload_from_checkpoint(round_num=1)
            self.assertEqual(payload["model_hash"], res.global_model_hash)
            self.assertEqual(len(reloaded_server.knowledge_bank), 1)

            # Verify SQLite records persisted across all tables
            with session_scope(db_url) as session:
                exp_row = session.execute(
                    select(Experiment).where(Experiment.experiment_id == "exp_persist_01")
                ).scalar_one()
                self.assertEqual(exp_row.task_type, "classification")

                rnd_row = session.execute(
                    select(TrainingRound).where(
                        TrainingRound.experiment_id == "exp_persist_01",
                        TrainingRound.round_num == 1,
                    )
                ).scalar_one()
                self.assertTrue(rnd_row.aggregated)
                self.assertEqual(rnd_row.global_model_hash, res.global_model_hash)

                cu_rows = (
                    session.execute(
                        select(ClientUpdate).where(
                            ClientUpdate.experiment_id == "exp_persist_01"
                        )
                    )
                    .scalars()
                    .all()
                )
                self.assertEqual(len(cu_rows), 1)

                gm_rows = (
                    session.execute(
                        select(GlobalModelVersion).where(
                            GlobalModelVersion.experiment_id == "exp_persist_01"
                        )
                    )
                    .scalars()
                    .all()
                )
                # Round 0 + Round 1
                self.assertEqual(len(gm_rows), 2)

                metric_rows = (
                    session.execute(
                        select(MetricRecord).where(
                            MetricRecord.experiment_id == "exp_persist_01"
                        )
                    )
                    .scalars()
                    .all()
                )
                self.assertGreaterEqual(len(metric_rows), 2)

                art_rows = (
                    session.execute(
                        select(ArtifactMetadata).where(
                            ArtifactMetadata.experiment_id == "exp_persist_01"
                        )
                    )
                    .scalars()
                    .all()
                )
                self.assertGreaterEqual(len(art_rows), 3)


if __name__ == "__main__":
    unittest.main()
