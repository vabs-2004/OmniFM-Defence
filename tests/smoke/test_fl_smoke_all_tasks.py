"""One-Round Federated Learning Smoke Tests for Phase 5 (Tasks 5.6 & 5.7).

1. Task 5.6: Runs 1-round classification smoke test on GPU (when available) sequentially
   with 1 client (`client_1`), then 2 clients (`client_1`, `client_2`), then all 3 clients
   (`client_1`, `client_2`, `client_3`), verifying compatible aggregation, global hash
   verification, SQLite persistence, and measured peak VRAM.
2. Task 5.7: Runs 1-round 3-client smoke tests for `segmentation`, `superresolution`,
   and `fusion` with full on-chain blockchain audit verification via local Hardhat RPC.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import torch

from backend.blockchain.web3_connector import Web3Connector
from backend.config import load_settings
from backend.fl.server import FederatedLearningServer
from backend.utils.hashing import verify_state_dict_hash
from tests.integration.test_web3_connector import (
    DEV_CLIENT1_KEY,
    DEV_CLIENT2_KEY,
    DEV_CLIENT3_KEY,
    DEV_SERVER_KEY,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestFLSmokeAllTasks(unittest.TestCase):
    rpc_proc: subprocess.Popen[str] | None = None
    rpc_url: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        port = _find_free_port()
        cls.rpc_url = f"http://127.0.0.1:{port}"
        blockchain_dir = Path(__file__).resolve().parents[2] / "blockchain"
        env = os.environ.copy()
        env["RPC_PORT"] = str(port)
        env["RPC_HOST"] = "127.0.0.1"

        cls.rpc_proc = subprocess.Popen(
            ["node", "scripts/rpc_server.js"],
            cwd=str(blockchain_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("Timed out starting local RPC server for FL smoke test.")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.rpc_proc is not None and cls.rpc_proc.poll() is None:
            cls.rpc_proc.terminate()
            try:
                cls.rpc_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.rpc_proc.kill()

    def _setup_connector_and_cfg(self, tmp_path: Path) -> tuple[Web3Connector, object]:
        db_url = f"sqlite:///{(tmp_path / 'smoke.db').as_posix()}"
        base_cfg = load_settings(
            env_file=None,
            overrides={
                "ganache_url": self.rpc_url,
                "ganache_chain_id": 1337,
                "server_private_key": DEV_SERVER_KEY,
                "client1_private_key": DEV_CLIENT1_KEY,
                "client2_private_key": DEV_CLIENT2_KEY,
                "client3_private_key": DEV_CLIENT3_KEY,
                "database_url": db_url,
                "artifact_dir": str(tmp_path / "artifacts"),
                "knowledge_bank_path": str(tmp_path / "bank" / "knowledge_bank.pt"),
            },
        )
        connector = Web3Connector(cfg=base_cfg)
        contract_addr, _ = connector.deploy_contract()
        base_cfg.contract_address = contract_addr

        for cid, key in [
            ("client_1", DEV_CLIENT1_KEY),
            ("client_2", DEV_CLIENT2_KEY),
            ("client_3", DEV_CLIENT3_KEY),
        ]:
            connector.register_hospital(
                cid, f"Hospital_{cid}", connector.get_wallet_address(key)
            )
        return connector, base_cfg

    # =========================================================================
    # Task 5.6: Classification 1-Client -> 2-Client -> 3-Client Vertical Slice
    # =========================================================================
    def test_classification_one_two_three_clients_smoke(self) -> None:
        print("\n=== Task 5.6 Classification 1 -> 2 -> 3 Client One-Round Smoke Report ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            connector, cfg = self._setup_connector_and_cfg(tmp_path)

            for client_subset in [
                ("client_1",),
                ("client_1", "client_2"),
                ("client_1", "client_2", "client_3"),
            ]:
                exp_id = f"cls_smoke_{len(client_subset)}c"
                # Configure expected client count on-chain for this experiment
                fn = connector.contract.get_function_by_signature(
                    "setExperimentExpectedClients(string,uint256)"
                )(exp_id, len(client_subset))
                connector._send_transaction(
                    tx_call=fn,
                    private_key=DEV_SERVER_KEY,
                    action="set_expected_clients",
                )

                server = FederatedLearningServer(
                    experiment_id=exp_id,
                    task_type="classification",
                    algorithm="omnifm",
                    client_ids=client_subset,
                    cfg=cfg,  # type: ignore[arg-type]
                    web3_connector=connector,
                    audit_required=True,
                    knowledge_bank_path=tmp_path / f"bank_{exp_id}.pt",
                )
                res = server.run_round(
                    round_num=1,
                    local_epochs=1,
                    batch_size=8,
                    max_train_samples=16,
                    max_val_samples=8,
                )

                self.assertEqual(res.state.value, "COMPLETED")
                self.assertEqual(len(res.client_updates), len(client_subset))
                self.assertEqual(res.audit_status, "CONFIRMED")
                self.assertTrue(
                    verify_state_dict_hash(server.global_shared_state, res.global_model_hash)
                )
                self.assertEqual(
                    connector.get_global_model_hash(exp_id, 1), res.global_model_hash
                )
                print(
                    f"Clients={len(client_subset)} ({','.join(client_subset)}) | "
                    f"GlobalHash={res.global_model_hash[:14]}... | "
                    f"PeakVRAM={res.peak_vram_mb:6.2f} MB | "
                    f"Metrics={res.global_metrics}"
                )

    # =========================================================================
    # Task 5.7: Segmentation, Super-Resolution, and Fusion 3-Client Smoke Tests
    # =========================================================================
    def test_segmentation_superresolution_and_fusion_one_round_smoke(self) -> None:
        print("\n=== Task 5.7 Segmentation, SR, and Fusion 3-Client One-Round Smoke Report ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            connector, cfg = self._setup_connector_and_cfg(tmp_path)

            task_runs = [
                ("segmentation", {"batch_size": 2, "max_train_samples": 2, "max_val_samples": 2}),
                ("superresolution", {"batch_size": 2, "max_train_samples": 4, "max_val_samples": 2, "scale": 2}),
                ("fusion", {"batch_size": 2, "max_train_samples": 4, "max_val_samples": 2}),
            ]

            for task_name, r_kwargs in task_runs:
                exp_id = f"{task_name}_smoke_3c"
                server = FederatedLearningServer(
                    experiment_id=exp_id,
                    task_type=task_name,
                    algorithm="omnifm",
                    client_ids=("client_1", "client_2", "client_3"),
                    cfg=cfg,  # type: ignore[arg-type]
                    web3_connector=connector,
                    audit_required=True,
                    knowledge_bank_path=tmp_path / f"bank_{exp_id}.pt",
                )
                res = server.run_round(round_num=1, local_epochs=1, **r_kwargs)
                self.assertEqual(res.state.value, "COMPLETED")
                self.assertEqual(len(res.client_updates), 3)
                self.assertEqual(res.audit_status, "CONFIRMED")
                self.assertEqual(res.knowledge_bank_size, 3)
                self.assertTrue(
                    verify_state_dict_hash(server.global_shared_state, res.global_model_hash)
                )

                events = connector.get_audit_events(experiment_id=exp_id, round_num=1)
                self.assertEqual(
                    sum(1 for e in events if e["event_type"] == "ModelUpdateSubmitted"), 3
                )
                self.assertEqual(
                    sum(1 for e in events if e["event_type"] == "AggregationTriggered"), 1
                )
                self.assertEqual(
                    sum(1 for e in events if e["event_type"] == "GlobalModelStored"), 1
                )
                print(
                    f"Task={task_name:15s} | GlobalHash={res.global_model_hash[:14]}... | "
                    f"PeakVRAM={res.peak_vram_mb:6.2f} MB | "
                    f"Metrics={res.global_metrics}"
                )


if __name__ == "__main__":
    unittest.main()
