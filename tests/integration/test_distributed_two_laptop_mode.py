"""Integration and Regression Tests for Single-Laptop Failsafe (Mode A) and Two-Laptop Distributed Federation (Mode B)."""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from fastapi.testclient import TestClient
from pydantic import SecretStr
import torch

from backend.blockchain.web3_connector import TxReceiptRecord
from backend.config import Settings
from backend.database.db import init_db, session_scope
from backend.database.models import (
    BlockchainTransaction,
    ClientUpdate,
    Experiment,
    GlobalModelVersion,
    TrainingRound,
)
from backend.fl.remote_client import RemoteHospitalWorker
from backend.fl.server import FederatedLearningServer
from backend.main import create_app
from backend.services.experiment_service import TrainingJobManager
from backend.utils.hashing import hash_state_dict, serialize_state_dict_canonical


class FakeTwoClientBlockchainConnector:
    """Deterministic fake blockchain connector recording 2-client quorum calls and receipts."""

    def __init__(self) -> None:
        self.expected_clients_calls: list[tuple[str, int]] = []
        self.submitted_updates: list[tuple[str, int, str, str]] = []
        self.aggregated_rounds: list[tuple[str, int]] = []
        self.stored_models: list[tuple[str, int, str]] = []
        self._tx_counter = 0

    def _next_receipt(self, fn_name: str) -> TxReceiptRecord:
        self._tx_counter += 1
        return TxReceiptRecord(
            action=fn_name,
            tx_hash="0x" + f"{self._tx_counter:064x}",
            block_number=100 + self._tx_counter,
            gas_used=50000,
            status="CONFIRMED",
            sender="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            contract_address="0xbEFE9C52315C84beC3085D6B2Ac261DE4b624650",
        )

    def is_connected(self) -> bool:
        return True

    def is_contract_deployed(self) -> bool:
        return True

    def set_experiment_expected_clients(
        self, experiment_id: str, expected_count: int
    ) -> TxReceiptRecord:
        self.expected_clients_calls.append((experiment_id, int(expected_count)))
        return self._next_receipt("setExperimentExpectedClients")

    def submit_model_update(
        self,
        experiment_id: str,
        round_num: int,
        task_type: str,
        model_hash: str,
        client_id: str,
        **kwargs: Any,
    ) -> TxReceiptRecord:
        self.submitted_updates.append((experiment_id, round_num, client_id, model_hash))
        return self._next_receipt("submitModelUpdate")

    def trigger_aggregation(self, experiment_id: str, round_num: int) -> TxReceiptRecord:
        self.aggregated_rounds.append((experiment_id, round_num))
        return self._next_receipt("triggerAggregation")

    def store_global_model(
        self,
        experiment_id: str,
        round_num: int,
        task_type: str,
        global_model_hash: str,
    ) -> TxReceiptRecord:
        self.stored_models.append((experiment_id, round_num, global_model_hash))
        return self._next_receipt("storeGlobalModel")


class TestDistributedTwoLaptopAndSingleLaptopFailsafe(unittest.TestCase):
    """Verify Mode A (Single-Laptop Default) and Mode B (Two-Laptop Distributed) behavior."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name)
        self.db_path = self.root_path / "test_omnifm.db"
        self.db_url = f"sqlite:///{self.db_path}"
        self.artifacts_dir = self.root_path / "artifacts"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.experiments_dir = self.artifacts_dir / "experiments"
        self.logs_dir = self.root_path / "logs"
        init_db(self.db_url)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _make_settings(self, token: str = "") -> Settings:
        return Settings(
            database_url=self.db_url,
            artifact_dir=self.artifacts_dir,
            knowledge_bank_path=self.artifacts_dir / "knowledge_bank.pt",
            blockchain_audit_enabled=False,
            fl_client_auth_token=SecretStr(token),
            distributed_round_timeout_sec=15,
        )

    def test_mode_a_single_laptop_default_remains_untouched(self) -> None:
        """Mode A (default) runs 3 local clients sequentially with no token or remote config."""
        cfg = self._make_settings(token="")
        mgr = TrainingJobManager()
        res = mgr.start_training(
            task_type="classification",
            num_rounds=1,
            config={
                "algorithm": "omnifm",
                "local_epochs": 1,
                "batch_size": 4,
                "max_samples": 8,
                "device": "cpu",
            },
            training_id="exp_mode_a_single_laptop",
            cfg=cfg,
            database_url=self.db_url,
            run_async=False,
        )
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["execution_mode"], "single_laptop")
        self.assertEqual(res["remote_client_ids"], [])

        status = mgr.get_training_status("exp_mode_a_single_laptop", database_url=self.db_url)
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["current_round"], 1)
        self.assertEqual(status["execution_mode"], "single_laptop")

    def test_distributed_mode_requires_explicit_non_empty_token(self) -> None:
        """Starting Mode B without FL_CLIENT_AUTH_TOKEN fails fast with ValueError / HTTP 400."""
        cfg_no_token = self._make_settings(token="")
        mgr = TrainingJobManager()
        with self.assertRaises(ValueError) as ctx:
            mgr.start_training(
                task_type="classification",
                num_rounds=1,
                config={
                    "algorithm": "omnifm",
                    "client_ids": ["client_1", "client_2"],
                    "remote_client_ids": ["client_2"],
                    "device": "cpu",
                },
                cfg=cfg_no_token,
                database_url=self.db_url,
                run_async=False,
            )
        self.assertIn("FL_CLIENT_AUTH_TOKEN", str(ctx.exception))

    def test_remote_endpoints_enforce_token_authentication(self) -> None:
        """Remote endpoints return 403 when token unconfigured and 401 on missing/wrong token."""
        app = create_app()
        app.state.database_url = self.db_url
        app.state.job_manager = TrainingJobManager()

        # Case 1: FL_CLIENT_AUTH_TOKEN is empty -> 403 Forbidden
        app.state.settings = self._make_settings(token="")
        client = TestClient(app)
        r403 = client.get("/api/fl/rounds/current", params={"client_id": "client_2"})
        self.assertEqual(r403.status_code, 403)

        # Case 2: FL_CLIENT_AUTH_TOKEN is set, but header missing or wrong -> 401 Unauthorized
        app.state.settings = self._make_settings(token="secret-demo-token-2026")
        r401_missing = client.get("/api/fl/rounds/current", params={"client_id": "client_2"})
        self.assertEqual(r401_missing.status_code, 401)

        r401_wrong = client.get(
            "/api/fl/rounds/current",
            params={"client_id": "client_2"},
            headers={"X-OmniFM-Client-Token": "wrong-token"},
        )
        self.assertEqual(r401_wrong.status_code, 401)

        # Valid token -> 200 OK (IDLE when no job is running)
        r200 = client.get(
            "/api/fl/rounds/current",
            params={"client_id": "client_2"},
            headers={"X-OmniFM-Client-Token": "secret-demo-token-2026"},
        )
        self.assertEqual(r200.status_code, 200)
        self.assertEqual(r200.json()["status"], "IDLE")

    def test_mode_b_two_laptop_distributed_round_and_validation(self) -> None:
        """Full 2-client distributed training round with local client_1 + remote CPU client_2 and blockchain audit."""
        token = "omnifm-two-laptop-token-98765"
        cfg = self._make_settings(token=token)
        fake_chain = FakeTwoClientBlockchainConnector()
        mgr = TrainingJobManager()

        app = create_app()
        app.state.settings = cfg
        app.state.database_url = self.db_url
        app.state.job_manager = mgr
        app.state.web3_connector = fake_chain
        http_client = TestClient(app)

        # Start a 2-round distributed experiment asynchronously (client_1 local, client_2 remote)
        start_resp = http_client.post(
            "/api/training/start",
            json={
                "task_type": "classification",
                "num_rounds": 1,
                "training_id": "exp_two_laptop_demo",
                "run_async": True,
                "config": {
                    "algorithm": "omnifm",
                    "client_ids": ["client_1", "client_2"],
                    "remote_client_ids": ["client_2"],
                    "local_epochs": 1,
                    "batch_size": 4,
                    "max_samples": 8,
                    "device": "cpu",
                    "remote_round_timeout_sec": 30.0,
                },
            },
        )
        self.assertEqual(start_resp.status_code, 200, start_resp.text)
        start_data = start_resp.json()
        self.assertEqual(start_data["execution_mode"], "distributed")
        self.assertEqual(start_data["remote_client_ids"], ["client_2"])

        # Wait until coordinator finishes local client_1 and is waiting for remote client_2
        task_payload: dict[str, Any] | None = None
        for _ in range(100):
            poll_resp = http_client.get(
                "/api/fl/rounds/current",
                params={"client_id": "client_2", "experiment_id": "exp_two_laptop_demo"},
                headers={"X-OmniFM-Client-Token": token},
            )
            self.assertEqual(poll_resp.status_code, 200)
            data = poll_resp.json()
            if data.get("awaiting_submission"):
                task_payload = data
                break
            time.sleep(0.05)

        self.assertIsNotNone(task_payload, "Coordinator did not reach AWAITING_ submission state")
        assert task_payload is not None
        self.assertEqual(task_payload["round_num"], 1)

        # Test negative validation cases before valid submission:
        toy_state = {"small_weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        toy_bytes = serialize_state_dict_canonical(toy_state)
        toy_b64 = base64.b64encode(toy_bytes).decode("ascii")
        toy_hash = hash_state_dict(toy_state)

        # 1. Wrong round number -> 409
        bad_round_resp = http_client.post(
            "/api/fl/rounds/exp_two_laptop_demo/99/submit",
            headers={"X-OmniFM-Client-Token": token},
            json={
                "client_id": "client_2",
                "num_samples": 8,
                "shared_state_b64": toy_b64,
                "update_hash": toy_hash,
            },
        )
        self.assertEqual(bad_round_resp.status_code, 409)

        # 2. Unknown experiment_id -> 404
        bad_exp_resp = http_client.post(
            "/api/fl/rounds/exp_nonexistent/1/submit",
            headers={"X-OmniFM-Client-Token": token},
            json={
                "client_id": "client_2",
                "num_samples": 8,
                "shared_state_b64": toy_b64,
                "update_hash": toy_hash,
            },
        )
        self.assertEqual(bad_exp_resp.status_code, 404)

        # 3. Corrupted update_hash -> 400
        bad_hash_resp = http_client.post(
            "/api/fl/rounds/exp_two_laptop_demo/1/submit",
            headers={"X-OmniFM-Client-Token": token},
            json={
                "client_id": "client_2",
                "num_samples": 8,
                "shared_state_b64": toy_b64,
                "update_hash": "f" * 64,
            },
        )
        self.assertEqual(bad_hash_resp.status_code, 400)

        # 4. NaN tensor injection -> 400
        nan_state = {"bad_param": torch.tensor([float("nan"), 1.0], dtype=torch.float32)}
        nan_bytes = serialize_state_dict_canonical(nan_state)
        nan_b64 = base64.b64encode(nan_bytes).decode("ascii")
        nan_resp = http_client.post(
            "/api/fl/rounds/exp_two_laptop_demo/1/submit",
            headers={"X-OmniFM-Client-Token": token},
            json={
                "client_id": "client_2",
                "num_samples": 8,
                "shared_state_b64": nan_b64,
                "update_hash": hash_state_dict(nan_state),
            },
        )
        self.assertEqual(nan_resp.status_code, 400)

        # Now run RemoteHospitalWorker (Laptop B CPU worker) to train and submit round 1
        worker = RemoteHospitalWorker(
            coordinator_url="",
            client_id="client_2",
            token=token,
            experiment_id="exp_two_laptop_demo",
            device="cpu",
            http_client=http_client,
        )
        exec_res = worker.poll_and_execute_once()
        self.assertEqual(exec_res["action"], "submitted")
        self.assertEqual(exec_res["round_num"], 1)
        self.assertEqual(exec_res["num_samples"], 8)

        # Verify personalized heads and client_suffix_prompt remained on Laptop B
        fl_client_b = worker._fl_clients[("exp_two_laptop_demo", "classification", "omnifm")]
        local_keys = set(fl_client_b.personalized_state.keys())
        self.assertTrue(any(k.startswith("heads.") for k in local_keys))
        self.assertTrue(any("client_suffix_prompt" in k for k in local_keys))

        # Wait for coordinator background thread to finish aggregation and mark COMPLETED
        for _ in range(200):
            st = http_client.get(
                "/api/training/status", params={"training_id": "exp_two_laptop_demo"}
            ).json()
            if st["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.1)

        final_status = http_client.get(
            "/api/training/status", params={"training_id": "exp_two_laptop_demo"}
        ).json()
        self.assertEqual(final_status["status"], "COMPLETED", final_status.get("error_message"))
        self.assertEqual(final_status["current_round"], 1)
        self.assertIsNotNone(final_status["latest_checkpoint"])

        # Verify SQLite records and blockchain quorum/audit calls
        self.assertEqual(fake_chain.expected_clients_calls, [("exp_two_laptop_demo", 2)])
        self.assertEqual(len(fake_chain.submitted_updates), 2)
        self.assertEqual(fake_chain.aggregated_rounds, [("exp_two_laptop_demo", 1)])
        self.assertEqual([m[1] for m in fake_chain.stored_models], [0, 1])

        with session_scope(self.db_url) as session:
            updates = (
                session.query(ClientUpdate)
                .filter_by(experiment_id="exp_two_laptop_demo", round_num=1)
                .all()
            )
            self.assertEqual({u.client_id for u in updates}, {"client_1", "client_2"})
            for u in updates:
                self.assertEqual(u.blockchain_status, "CONFIRMED")
                self.assertIsNotNone(u.tx_hash)

            txs = (
                session.query(BlockchainTransaction)
                .filter_by(experiment_id="exp_two_laptop_demo")
                .all()
            )
            self.assertGreaterEqual(len(txs), 4)
            self.assertTrue(all(t.status == "CONFIRMED" for t in txs))

    def test_remote_client_timeout_marks_round_failed(self) -> None:
        """If remote client_2 does not submit before timeout, round transitions to FAILED."""
        cfg = self._make_settings(token="timeout-test-token")
        server = FederatedLearningServer(
            experiment_id="exp_timeout_test",
            task_type="classification",
            algorithm="omnifm",
            client_ids=("client_1", "client_2"),
            remote_client_ids=("client_2",),
            remote_round_timeout_sec=0.3,
            cfg=cfg,
            database_url=self.db_url,
        )
        server.initialize_round_zero(total_rounds=1)
        with self.assertRaises(TimeoutError):
            server.run_round(
                round_num=1,
                local_epochs=1,
                batch_size=4,
                max_train_samples=4,
                max_val_samples=4,
                device="cpu",
                remote_timeout_sec=0.3,
            )


if __name__ == "__main__":
    unittest.main()
