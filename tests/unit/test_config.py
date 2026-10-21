"""Unit tests for backend.config validated settings."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.config import Settings, load_settings, settings


class TestConfigSettings(unittest.TestCase):
    def test_default_settings_import_and_redact_secrets(self) -> None:
        self.assertEqual(settings.embedding_dim, 128)
        self.assertEqual(settings.gskr_top_k, 2)
        self.assertAlmostEqual(settings.spalign_lambda, 0.1)
        self.assertEqual(settings.precision_policy, "fp16_amp")

        dummy_key = "0x" + "a" * 64
        cfg = load_settings(env_file=None, overrides={"server_private_key": dummy_key})
        safe = cfg.safe_dict()
        self.assertEqual(safe["server_private_key"], "***REDACTED***")
        self.assertNotIn("a" * 64, repr(cfg))
        self.assertNotIn("a" * 64, str(safe))

    def test_invalid_configurations_raise_actionable_errors(self) -> None:
        with self.assertRaises(ValidationError) as ctx1:
            Settings(ganache_chain_id=0)
        self.assertIn("GANACHE_CHAIN_ID", str(ctx1.exception))

        with self.assertRaises(ValidationError) as ctx2:
            Settings(contract_address="not-an-address")
        self.assertIn("CONTRACT_ADDRESS", str(ctx2.exception))

        with self.assertRaises(ValidationError) as ctx3:
            Settings(embedding_dim=130, eca_num_heads=4)
        self.assertIn("divisible", str(ctx3.exception))

        with self.assertRaises(ValidationError) as ctx4:
            Settings(blockchain_audit_enabled=True, contract_address="")
        self.assertIn("BLOCKCHAIN_AUDIT_ENABLED", str(ctx4.exception))

        with self.assertRaises(ValidationError) as ctx5:
            Settings(precision_policy="invalid_precision")  # type: ignore[arg-type]
        self.assertIn("precision_policy", str(ctx5.exception))


if __name__ == "__main__":
    unittest.main()
