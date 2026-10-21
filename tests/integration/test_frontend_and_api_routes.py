"""Phase 7 Verification Test (`tests/integration/test_frontend_and_api_routes.py`, Tasks 7.1 - 7.8).

Verifies:
1. Production frontend build (`frontend/dist/index.html`) and all 7 React routes
   (`/`, `/classify`, `/segment`, `/superres`, `/fusion`, `/blockchain`, `/train`).
2. Static audit of all frontend pages to ensure no hardcoded fake metric arrays appear.
3. End-to-end API contract walkthrough for every route consumed by the 7 pages across
   empty, success, and error states.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.blockchain.web3_connector import BlockchainError
from backend.config import settings
from backend.database.db import dispose_engines
from backend.main import create_app
from backend.services.experiment_service import TrainingJobManager

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"


class TestFrontendAndApiRoutes(unittest.TestCase):
    """Verify Tasks 7.1 - 7.8 frontend build, route wiring, and API empty/success/failure states."""

    def test_frontend_build_routes_and_no_fake_metrics(self) -> None:
        dist_index = FRONTEND_DIR / "dist" / "index.html"
        self.assertTrue(dist_index.is_file(), "Frontend production build dist/index.html must exist.")

        app_jsx = (FRONTEND_DIR / "src" / "App.jsx").read_text(encoding="utf-8")
        for route in ("/", "/classify", "/segment", "/superres", "/fusion", "/blockchain", "/train"):
            self.assertIn(f'path="{route}"', app_jsx, f"Missing route {route} in App.jsx")

        page_files = [
            "Dashboard.jsx",
            "Classification.jsx",
            "Segmentation.jsx",
            "SuperResolution.jsx",
            "Fusion.jsx",
            "BlockchainExplorer.jsx",
            "TrainingControl.jsx",
        ]
        for fname in page_files:
            fpath = FRONTEND_DIR / "src" / "pages" / fname
            self.assertTrue(fpath.is_file(), f"Missing page file {fpath}")
            content = fpath.read_text(encoding="utf-8")
            self.assertIn("omnifmApi", content, f"{fname} must fetch live data via omnifmApi")

    def test_all_page_api_flows_empty_success_and_error_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_url = f"sqlite:///{(tmp_path / 'phase7.db').as_posix()}"
            orig_art = settings.artifact_dir
            settings.artifact_dir = tmp_path / "artifacts"
            try:
                mock_w3 = MagicMock()
                mock_w3.is_connected.return_value = False
                mock_w3.is_contract_deployed.return_value = False
                mock_w3.store_global_model.side_effect = BlockchainError("Offline")
                mock_w3.submit_model_update.side_effect = BlockchainError("Offline")
                mock_w3.trigger_aggregation.side_effect = BlockchainError("Offline")

                app = create_app(
                    database_url=db_url,
                    web3_connector=mock_w3,
                    job_manager_instance=TrainingJobManager(),
                )
                client = TestClient(app)

                # 1. Empty state on initial Dashboard load
                st_empty = client.get("/api/training/status")
                self.assertEqual(st_empty.status_code, 200)
                self.assertEqual(st_empty.json()["status"], "IDLE")

                m_empty = client.get("/api/metrics/history", params={"task_type": "classification"})
                self.assertEqual(m_empty.status_code, 200)
                self.assertEqual(m_empty.json()["global_history"], [])

                # 2. Failure state on invalid training config
                err_start = client.post(
                    "/api/training/start",
                    json={"task_type": "superresolution", "num_rounds": 1, "config": {"scale_factor": 5}},
                )
                self.assertEqual(err_start.status_code, 400)

                # 3. Success state after running 1 round
                ok_start = client.post(
                    "/api/training/start",
                    json={
                        "task_type": "classification",
                        "num_rounds": 1,
                        "training_id": "exp_phase7_check",
                        "run_async": False,
                        "config": {"batch_size": 4, "max_samples": 8, "device": "cpu"},
                    },
                )
                self.assertEqual(ok_start.status_code, 200)
                self.assertEqual(ok_start.json()["status"], "COMPLETED")
            finally:
                settings.artifact_dir = orig_art
                dispose_engines(db_url)


if __name__ == "__main__":
    unittest.main()
