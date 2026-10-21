"""Unit tests for Task 0.5: logging, seed/config capture, and basic health check."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from backend.config import load_settings
from backend.utils.logging import (
     get_system_health,
    save_run_metadata,
    set_global_seed,
    setup_logger,
)


class TestLoggingSeedAndHealth(unittest.TestCase):
    def test_seed_reproducibility(self) -> None:
        set_global_seed(12345, deterministic=True)
        t1 = torch.randn(4, 4)
        set_global_seed(12345, deterministic=True)
        t2 = torch.randn(4, 4)
        self.assertTrue(torch.equal(t1, t2))

    def test_config_capture_and_log_redaction(self) -> None:
        fake_key = "0x" + "b" * 64
        fake_subject = "FeTS2022_01234"
        fake_slide = "SOB_B_A-14-22549AB-40-001"

        cfg = load_settings(
            env_file=None,
            overrides={
                "server_private_key": fake_key,
                "random_seed": 99,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "test_run.log"
            meta_file = tmp_path / "run_meta.json"

            logger = setup_logger("omnifm_test_05", log_file=log_file)
            logger.info(
                "Starting test run with key=%s subject=%s slide=%s",
                fake_key,
                fake_subject,
                fake_slide,
            )
            for handler in logger.handlers:
                handler.flush()

            metadata = save_run_metadata(
                meta_file,
                cfg=cfg,
                seed=99,
                extra={"note": f"checked {fake_subject} and {fake_key}"},
            )

            self.assertEqual(metadata["seed"], 99)
            self.assertEqual(metadata["config"]["random_seed"], 99)
            self.assertEqual(metadata["config"]["server_private_key"], "***REDACTED***")

            raw_meta = meta_file.read_text(encoding="utf-8")
            self.assertNotIn("b" * 64, raw_meta)
            self.assertNotIn(fake_subject, raw_meta)

            raw_log = log_file.read_text(encoding="utf-8")
            self.assertIn("[REDACTED_KEY]", raw_log)
            self.assertIn("[REDACTED_SUBJECT_ID]", raw_log)
            self.assertIn("[REDACTED_SLIDE_ID]", raw_log)
            self.assertNotIn("b" * 64, raw_log)
            self.assertNotIn(fake_subject, raw_log)
            self.assertNotIn(fake_slide, raw_log)

            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def test_basic_health_check(self) -> None:
        health = get_system_health()
        self.assertEqual(health["status"], "ok")
        self.assertIn("gpu", health)
        self.assertIn("cuda_available", health["gpu"])
        self.assertIn("blockchain", health)
        self.assertNotIn("private_key", json.dumps(health))


if __name__ == "__main__":
    unittest.main()
