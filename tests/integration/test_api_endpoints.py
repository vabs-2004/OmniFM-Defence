"""Integration Tests for FastAPI Endpoints (`tests/integration/test_api_endpoints.py`, Tasks 6.2 - 6.6)."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from backend.blockchain.web3_connector import BlockchainError
from backend.config import settings
from backend.database.db import dispose_engines
from backend.main import create_app
from backend.services.experiment_service import TrainingJobManager


def _make_png_bytes(width: int = 32, height: int = 32, channels: int = 3, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    if channels == 1:
        arr = rng.integers(20, 220, size=(height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        arr = rng.integers(20, 220, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAPIEndpointsIntegration(unittest.TestCase):
    """Verify Tasks 6.2 - 6.6 API routes, error handling, reference-metric guards, and audit verification."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.db_url = f"sqlite:///{(self.tmp_path / 'api_test.db').as_posix()}"
        self.orig_artifact_dir = settings.artifact_dir
        settings.artifact_dir = self.tmp_path / "artifacts"

        # Simulate unavailable blockchain (raises BlockchainError on write attempts)
        self.mock_w3 = MagicMock()
        self.mock_w3.is_connected.return_value = False
        self.mock_w3.is_contract_deployed.return_value = False
        self.mock_w3.store_global_model.side_effect = BlockchainError("Ganache RPC node unavailable")
        self.mock_w3.submit_model_update.side_effect = BlockchainError("Ganache RPC node unavailable")
        self.mock_w3.trigger_aggregation.side_effect = BlockchainError("Ganache RPC node unavailable")

        self.job_mgr = TrainingJobManager()
        self.app = create_app(
            database_url=self.db_url,
            web3_connector=self.mock_w3,
            job_manager_instance=self.job_mgr,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        settings.artifact_dir = self.orig_artifact_dir
        dispose_engines(self.db_url)
        self.tmp_dir.cleanup()

    def test_docs_health_and_hospital_registration_security(self) -> None:
        """Task 6.2: /docs loads, /api/health reports status without secrets, /api/hospital rejects private keys."""
        docs_res = self.client.get("/docs")
        self.assertEqual(docs_res.status_code, 200)
        self.assertIn("Swagger UI", docs_res.text)

        health_res = self.client.get("/api/health")
        self.assertEqual(health_res.status_code, 200)
        h_data = health_res.json()
        self.assertEqual(h_data["status"], "ok")
        self.assertTrue(h_data["database"]["connected"])
        self.assertIn("gpu", h_data)
        self.assertIn("blockchain", h_data)
        self.assertFalse(h_data["blockchain"]["connected"])
        # Ensure no private keys appear anywhere in health output
        self.assertNotIn("private_key", str(h_data).lower())

        # Reject private_key in hospital registration body
        bad_reg = self.client.post(
            "/api/hospital/register",
            json={
                "client_id": "client_1",
                "hospital_name": "Test Hospital",
                "modality": "Dermatology",
                "private_key": "0xdeadbeef",
            },
        )
        self.assertEqual(bad_reg.status_code, 400)
        self.assertIn("Private keys", bad_reg.json()["detail"])

        # Valid hospital registration
        ok_reg = self.client.post(
            "/api/hospital/register",
            json={
                "client_id": "client_1",
                "hospital_name": "Metro General Hospital A",
                "modality": "Dermatology & Brain MRI",
            },
        )
        self.assertEqual(ok_reg.status_code, 200)
        self.assertEqual(ok_reg.json()["client_id"], "client_1")
        self.assertEqual(ok_reg.json()["hospital_name"], "Metro General Hospital A")

        list_res = self.client.get("/api/hospital")
        self.assertEqual(list_res.status_code, 200)
        self.assertGreaterEqual(list_res.json()["total"], 3)

    def test_training_lifecycle_errors_cancel_and_resume(self) -> None:
        """Task 6.3 & 6.6: validate start, failure visibility, missing checkpoint on resume, and valid resume."""
        # 1. Invalid task_type -> 400
        bad_start = self.client.post(
            "/api/training/start",
            json={"task_type": "invalid_task", "num_rounds": 1},
        )
        self.assertEqual(bad_start.status_code, 400)

        # 2. Simulated training failure -> status FAILED with visible error_message, and resume fails (404) without checkpoint
        fail_start = self.client.post(
            "/api/training/start",
            json={
                "task_type": "classification",
                "num_rounds": 1,
                "training_id": "exp_fail_test",
                "run_async": False,
                "config": {"simulate_failure": True, "device": "cpu"},
            },
        )
        self.assertEqual(fail_start.status_code, 200)
        status_fail = self.client.get("/api/training/status", params={"training_id": "exp_fail_test"})
        self.assertEqual(status_fail.status_code, 200)
        self.assertEqual(status_fail.json()["status"], "FAILED")
        self.assertIn("Simulated training failure", status_fail.json()["error_message"])

        # Resume on failed job with 0 checkpoints -> 404 CheckpointNotFoundError
        resume_missing = self.client.post(
            "/api/training/exp_fail_test/resume",
            json={"run_async": False},
        )
        self.assertEqual(resume_missing.status_code, 404)

        # 3. Normal 1-round synchronous run -> COMPLETED, logs & metrics populated
        ok_start = self.client.post(
            "/api/training/start",
            json={
                "task_type": "classification",
                "num_rounds": 1,
                "training_id": "exp_cls_api",
                "run_async": False,
                "config": {
                    "algorithm": "omnifm",
                    "batch_size": 4,
                    "max_samples": 8,
                    "max_batches": 1,
                    "local_epochs": 1,
                    "device": "cpu",
                },
            },
        )
        self.assertEqual(ok_start.status_code, 200)
        self.assertEqual(ok_start.json()["status"], "COMPLETED")

        status_ok = self.client.get("/api/training/status", params={"training_id": "exp_cls_api"})
        self.assertEqual(status_ok.status_code, 200)
        st_body = status_ok.json()
        self.assertEqual(st_body["current_round"], 1)
        self.assertIsNotNone(st_body["latest_checkpoint"])
        self.assertIn("accuracy", st_body["latest_metrics"])

        logs_res = self.client.get("/api/training/exp_cls_api/logs")
        self.assertEqual(logs_res.status_code, 200)
        self.assertGreaterEqual(logs_res.json()["total_lines"], 2)

        # 4. Audit log, artifact verification, experiments list, and metrics history (Task 6.5)
        exp_list = self.client.get("/api/experiments")
        self.assertEqual(exp_list.status_code, 200)
        self.assertGreaterEqual(exp_list.json()["total"], 2)

        metrics_res = self.client.get(
            "/api/metrics/history",
            params={"task_type": "classification", "training_id": "exp_cls_api"},
        )
        self.assertEqual(metrics_res.status_code, 200)
        m_body = metrics_res.json()
        self.assertEqual(len(m_body["global_history"]), 1)
        self.assertEqual(len(m_body["client_history"]), 3)

        audit_res = self.client.get("/api/blockchain/log", params={"experiment_id": "exp_cls_api"})
        self.assertEqual(audit_res.status_code, 200)
        self.assertIn("audit_disclaimer", audit_res.json())
        self.assertGreaterEqual(audit_res.json()["total"], 4)

        # Verify global checkpoint hash matches
        verify_ok = self.client.post(
            "/api/blockchain/verify",
            json={"experiment_id": "exp_cls_api", "round_num": 1},
        )
        self.assertEqual(verify_ok.status_code, 200)
        self.assertTrue(verify_ok.json()["match"])
        self.assertIn("audit_disclaimer", verify_ok.json())

        # Verify client update hash comparison with mismatched provided_hash -> match=False
        verify_mismatch = self.client.post(
            "/api/blockchain/verify",
            json={
                "experiment_id": "exp_cls_api",
                "round_num": 1,
                "client_id": "client_1",
                "provided_hash": "0x" + "11" * 32,
            },
        )
        self.assertEqual(verify_mismatch.status_code, 200)
        self.assertFalse(verify_mismatch.json()["match"])

    def test_prediction_endpoints_and_reference_metric_guards(self) -> None:
        """Task 6.4 & 6.6: test classify, segment, superresolution, and fusion endpoints + metric guards."""
        img_rgb = _make_png_bytes(32, 32, channels=3, seed=1)
        img_gray = _make_png_bytes(64, 64, channels=1, seed=2)
        img_gray_b = _make_png_bytes(64, 64, channels=1, seed=3)

        # 1. Missing checkpoint when allow_untrained_fallback=False -> 404
        missing_ckpt = self.client.post(
            "/api/predict/classify",
            files={"file": ("sample.png", img_rgb, "image/png")},
            data={"client_id": "client_1", "allow_untrained_fallback": "false"},
        )
        self.assertEqual(missing_ckpt.status_code, 404)

        # 2. Classification prediction
        cls_res = self.client.post(
            "/api/predict/classify",
            files={"file": ("derma.png", img_rgb, "image/png")},
            data={"dataset_name": "dermamnist"},
        )
        self.assertEqual(cls_res.status_code, 200)
        cls_body = cls_res.json()
        self.assertEqual(cls_body["dataset_name"], "dermamnist")
        self.assertEqual(len(cls_body["probabilities"]), 7)
        self.assertAlmostEqual(sum(cls_body["probabilities"]), 1.0, places=4)
        self.assertIn("clinical_disclaimer", cls_body)

        # 3. Segmentation prediction WITHOUT reference mask -> metrics MUST be None
        seg_no_ref = self.client.post(
            "/api/predict/segment",
            files={"file": ("mri.png", img_gray, "image/png")},
        )
        self.assertEqual(seg_no_ref.status_code, 200)
        self.assertIsNone(seg_no_ref.json()["metrics"])
        self.assertTrue(seg_no_ref.json()["mask_base64"].startswith("data:image/png;base64,"))

        # Segmentation prediction WITH reference mask -> Dice & IoU returned
        seg_with_ref = self.client.post(
            "/api/predict/segment",
            files={
                "file": ("mri.png", img_gray, "image/png"),
                "reference_mask": ("mask.png", img_gray_b, "image/png"),
            },
        )
        self.assertEqual(seg_with_ref.status_code, 200)
        self.assertIsNotNone(seg_with_ref.json()["metrics"])
        self.assertIn("dice", seg_with_ref.json()["metrics"])
        self.assertIn("iou", seg_with_ref.json()["metrics"])

        # 4. Super-Resolution prediction WITHOUT reference HR -> metrics MUST be None
        sr_no_ref = self.client.post(
            "/api/predict/superresolution",
            files={"file": ("patch.png", img_rgb, "image/png")},
            data={"scale": "2"},
        )
        self.assertEqual(sr_no_ref.status_code, 200)
        self.assertIsNone(sr_no_ref.json()["metrics"])
        self.assertEqual(sr_no_ref.json()["output_shape"], [1, 3, 128, 128])

        # Super-Resolution prediction WITH reference HR -> PSNR & SSIM returned
        sr_with_ref = self.client.post(
            "/api/predict/superresolution",
            files={
                "file": ("patch.png", img_rgb, "image/png"),
                "reference_hr": ("hr.png", img_rgb, "image/png"),
            },
            data={"scale": "2"},
        )
        self.assertEqual(sr_with_ref.status_code, 200)
        self.assertIsNotNone(sr_with_ref.json()["metrics"])
        self.assertIn("psnr", sr_with_ref.json()["metrics"])
        self.assertIn("ssim", sr_with_ref.json()["metrics"])

        # Invalid SR scale -> 400
        sr_bad_scale = self.client.post(
            "/api/predict/superresolution",
            files={"file": ("patch.png", img_rgb, "image/png")},
            data={"scale": "3"},
        )
        self.assertEqual(sr_bad_scale.status_code, 400)

        # 5. Fusion prediction with registered pair vs unregistered pair
        fus_reg = self.client.post(
            "/api/predict/fusion",
            files={
                "file_a": ("ct.png", img_gray, "image/png"),
                "file_b": ("mri.png", img_gray_b, "image/png"),
            },
            data={"modality_pair": "CT-MRI", "is_registered": "true"},
        )
        self.assertEqual(fus_reg.status_code, 200)
        self.assertIsNotNone(fus_reg.json()["metrics"])
        self.assertIn("spatial_frequency_no_ref", fus_reg.json()["metrics"])

        fus_unreg = self.client.post(
            "/api/predict/fusion",
            files={
                "file_a": ("ct.png", img_gray, "image/png"),
                "file_b": ("mri.png", img_gray_b, "image/png"),
            },
            data={"modality_pair": "CT-MRI", "is_registered": "false"},
        )
        self.assertEqual(fus_unreg.status_code, 200)
        self.assertIsNone(fus_unreg.json()["metrics"])


if __name__ == "__main__":
    unittest.main()
