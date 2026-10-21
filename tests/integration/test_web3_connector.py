"""Integration test for Task 1.5: Web3Connector against local Hardhat/Ganache RPC node."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import unittest
from pathlib import Path

import torch

from backend.blockchain.web3_connector import (
    BlockchainTransactionError,
    Web3Connector,
)
from backend.config import load_settings
from backend.utils.hashing import hash_state_dict

# Standard Hardhat/Anvil local development-only test private keys (NEVER use in production)
DEV_SERVER_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEV_CLIENT1_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
DEV_CLIENT2_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
DEV_CLIENT3_KEY = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
DEV_OUTSIDER_KEY = "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestWeb3ConnectorIntegration(unittest.TestCase):
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
        ready = False
        while time.time() < deadline:
            if cls.rpc_proc.poll() is not None:
                out = cls.rpc_proc.stdout.read() if cls.rpc_proc.stdout else ""
                raise RuntimeError(f"Local RPC server exited early: {out}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    ready = True
                    break
            except OSError:
                time.sleep(0.2)

        if not ready:
            cls.tearDownClass()
            raise RuntimeError("Timed out waiting for local Hardhat RPC server to start.")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.rpc_proc is not None and cls.rpc_proc.poll() is None:
            cls.rpc_proc.terminate()
            try:
                cls.rpc_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.rpc_proc.kill()

    def test_register_three_clients_submit_hashes_aggregate_and_read_events(self) -> None:
        cfg = load_settings(
            env_file=None,
            overrides={
                "ganache_url": self.rpc_url,
                "ganache_chain_id": 1337,
                "server_private_key": DEV_SERVER_KEY,
                "client1_private_key": DEV_CLIENT1_KEY,
                "client2_private_key": DEV_CLIENT2_KEY,
                "client3_private_key": DEV_CLIENT3_KEY,
            },
        )
        connector = Web3Connector(cfg=cfg)
        self.assertTrue(connector.is_connected())

        contract_addr, deploy_receipt = connector.deploy_contract()
        self.assertEqual(deploy_receipt.status, "CONFIRMED")
        self.assertTrue(contract_addr.startswith("0x"))

        # Verify chain ID and bytecode
        chain_info = connector.verify_chain_and_contract()
        self.assertEqual(chain_info["chain_id"], 1337)
        self.assertGreater(chain_info["bytecode_len"], 100)

        # Register 3 dummy hospital clients
        clients = [
            ("client_1", "Hospital One", connector.get_wallet_address(DEV_CLIENT1_KEY)),
            ("client_2", "Hospital Two", connector.get_wallet_address(DEV_CLIENT2_KEY)),
            ("client_3", "Hospital Three", connector.get_wallet_address(DEV_CLIENT3_KEY)),
        ]
        for cid, cname, cwallet in clients:
            rec = connector.register_hospital(cid, cname, cwallet)
            self.assertEqual(rec.status, "CONFIRMED")
            hosp = connector.get_hospital(cid)
            self.assertTrue(hosp["registered"])
            self.assertEqual(hosp["wallet"], cwallet)

        # Round 0 global model hash
        exp_id = "exp_integration_01"
        r0_hash = hash_state_dict({"backbone.weight": torch.zeros(4, 4)})
        r0_rec = connector.store_global_model(exp_id, 0, "classification", r0_hash)
        self.assertEqual(r0_rec.status, "CONFIRMED")
        self.assertEqual(connector.get_global_model_hash(exp_id, 0), r0_hash)

        # Submit 3 client update hashes for round 1
        c1_hash = hash_state_dict({"backbone.weight": torch.ones(4, 4) * 1.0})
        c2_hash = hash_state_dict({"backbone.weight": torch.ones(4, 4) * 2.0})
        c3_hash = hash_state_dict({"backbone.weight": torch.ones(4, 4) * 3.0})

        connector.submit_model_update(exp_id, 1, "classification", c1_hash, "client_1")
        connector.submit_model_update(exp_id, 1, "classification", c2_hash, "client_2")
        self.assertFalse(connector.check_all_submitted(exp_id, 1))

        # Incomplete round aggregation must fail
        with self.assertRaises(BlockchainTransactionError):
            connector.trigger_aggregation(exp_id, 1)

        # Unregistered / wrong wallet submission must fail
        with self.assertRaises(BlockchainTransactionError):
            connector.submit_model_update(
                exp_id,
                1,
                "classification",
                c3_hash,
                "client_3",
                client_private_key=DEV_OUTSIDER_KEY,
            )

        # Third valid submission completes quorum
        connector.submit_model_update(exp_id, 1, "classification", c3_hash, "client_3")
        self.assertTrue(connector.check_all_submitted(exp_id, 1))

        # Duplicate submission must fail
        with self.assertRaises(BlockchainTransactionError):
            connector.submit_model_update(exp_id, 1, "classification", c1_hash, "client_1")

        # Record aggregation and store round 1 global hash
        agg_rec = connector.trigger_aggregation(exp_id, 1)
        self.assertEqual(agg_rec.status, "CONFIRMED")

        r1_global_hash = hash_state_dict({"backbone.weight": torch.ones(4, 4) * 2.0})
        gm_rec = connector.store_global_model(exp_id, 1, "classification", r1_global_hash)
        self.assertEqual(gm_rec.status, "CONFIRMED")
        self.assertEqual(connector.get_global_model_hash(exp_id, 1), r1_global_hash)

        # Read back audit events and verify all lifecycle records match
        all_events = connector.get_audit_events()
        event_types = [e["event_type"] for e in all_events]
        self.assertEqual(event_types.count("HospitalRegistered"), 3)
        self.assertEqual(event_types.count("ModelUpdateSubmitted"), 3)
        self.assertEqual(event_types.count("AggregationTriggered"), 1)
        self.assertEqual(event_types.count("GlobalModelStored"), 2)

        r1_events = connector.get_audit_events(experiment_id=exp_id, round_num=1)
        submitted_hashes = {
            e["client_id"]: e["model_hash"]
            for e in r1_events
            if e["event_type"] == "ModelUpdateSubmitted"
        }
        self.assertEqual(
            submitted_hashes,
            {"client_1": c1_hash, "client_2": c2_hash, "client_3": c3_hash},
        )

        # Verify helper methods added during audit
        self.assertTrue(connector.is_contract_deployed())
        self.assertEqual(
            connector.get_contract_owner(),
            connector.get_wallet_address(DEV_SERVER_KEY),
        )

        # Verify global & client model hashes on-chain
        ver_global_ok = connector.verify_model_hash(exp_id, 1, r1_global_hash)
        self.assertTrue(ver_global_ok["exists"])
        self.assertTrue(ver_global_ok["match"])

        ver_global_mismatch = connector.verify_model_hash(exp_id, 1, c1_hash)
        self.assertTrue(ver_global_mismatch["exists"])
        self.assertFalse(ver_global_mismatch["match"])

        ver_client_ok = connector.verify_model_hash(exp_id, 1, c1_hash, client_id="client_1")
        self.assertTrue(ver_client_ok["exists"])
        self.assertTrue(ver_client_ok["match"])

        ver_not_on_chain = connector.verify_model_hash("nonexistent_exp", 99, r1_global_hash)
        self.assertFalse(ver_not_on_chain["exists"])
        self.assertFalse(ver_not_on_chain["match"])

        # Missing client private key must fail explicitly rather than falling back to accounts[0]
        cfg_no_c2 = load_settings(
            env_file=None,
            overrides={
                "ganache_url": self.rpc_url,
                "ganache_chain_id": 1337,
                "contract_address": contract_addr,
                "server_private_key": DEV_SERVER_KEY,
                "client1_private_key": DEV_CLIENT1_KEY,
                "client2_private_key": "",
            },
        )
        connector_no_c2 = Web3Connector(cfg=cfg_no_c2, contract_address=contract_addr)
        with self.assertRaises(BlockchainTransactionError) as ctx:
            connector_no_c2.submit_model_update(
                "exp_missing_key", 1, "classification", c2_hash, "client_2"
            )
        self.assertIn("CLIENT2_PRIVATE_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

