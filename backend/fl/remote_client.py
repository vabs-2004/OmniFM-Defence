"""Standalone Remote Hospital Federated Client Worker (`backend/fl/remote_client.py`).

Runs an independent hospital client process (e.g., `client_2` on Laptop B CPU)
that polls the Laptop A coordinator over HTTP, downloads the global shared
state (`OMNIFM_STATE_V1` canonical binary), trains locally on its private
hospital dataset partition while keeping `heads.*` and `client_suffix_prompt`
strictly local, and uploads its validated parameter update and SHA-256 hash.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from typing import Any

# Set single-threaded BLAS/OpenMP defaults on Windows before importing numpy/torch
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if "--device" in sys.argv:
    try:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].strip().lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except Exception:
        pass

import httpx
import torch

from backend.config import settings
from backend.fl.client import FederatedHospitalClient
from backend.omnifm.gskr import PrototypeMetadata, SpectralKnowledgeBank
from backend.utils.hashing import (
    deserialize_state_dict_canonical,
    hash_state_dict,
    serialize_state_dict_canonical,
)
from backend.utils.logging import set_global_seed, setup_logger

logger = setup_logger("omnifm.fl.remote_client")


def _reconstruct_knowledge_bank(
    bank_b64: str | None,
    metadata_list: list[dict[str, Any]] | None,
    embedding_dim: int = 128,
) -> SpectralKnowledgeBank:
    """Reconstruct a `SpectralKnowledgeBank` from canonical binary tensor + metadata."""
    bank = SpectralKnowledgeBank(embedding_dim=embedding_dim)
    if not bank_b64 or not metadata_list:
        return bank
    raw_bytes = base64.b64decode(bank_b64.encode("ascii"), validate=True)
    tensor_dict = deserialize_state_dict_canonical(raw_bytes)
    proto_tensor = tensor_dict.get("embeddings")
    if proto_tensor is None:
        proto_tensor = tensor_dict.get("prototypes")
    if proto_tensor is None or proto_tensor.ndim != 2:
        return bank
    count = min(int(proto_tensor.shape[0]), len(metadata_list))
    bank._embeddings = proto_tensor[:count].detach().cpu().to(dtype=torch.float32).clone()
    bank._metadata = [
        PrototypeMetadata(
            prototype_id=str(metadata_list[idx].get("prototype_id", f"proto_{idx}")),
            task_type=str(metadata_list[idx].get("task_type", "classification")),
            client_id=str(metadata_list[idx].get("client_id", "unknown")),
            round_num=int(metadata_list[idx].get("round_num", 0)),
            sample_count=int(metadata_list[idx].get("sample_count", 1)),
        )
        for idx in range(count)
    ]
    bank._counter = count
    return bank


class RemoteHospitalWorker:
    """Stateful remote hospital client worker that preserves local personalized parameters across rounds."""

    def __init__(
        self,
        coordinator_url: str,
        client_id: str = "client_2",
        token: str | None = None,
        experiment_id: str | None = None,
        device: str = "cpu",
        http_client: Any | None = None,
        request_timeout_sec: float = 60.0,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.client_id = client_id.strip()
        resolved_token = (
            token if token is not None else settings.fl_client_auth_token.get_secret_value()
        ).strip()
        if not resolved_token:
            raise ValueError(
                "RemoteHospitalWorker requires a non-empty authentication token "
                "(--token or FL_CLIENT_AUTH_TOKEN environment variable)."
            )
        self.token = resolved_token
        self.experiment_id = experiment_id.strip() if experiment_id else None
        self.device_str = device.strip().lower()
        if self.device_str == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                torch.set_num_threads(min(4, max(1, os.cpu_count() or 2)))
            except Exception:
                pass
        self.device = torch.device(self.device_str)
        self._http_client = http_client
        self.request_timeout_sec = float(request_timeout_sec)

        # Persisted per-(experiment_id, task_type, algorithm) so local heads & suffix prompt persist across rounds
        self._fl_clients: dict[tuple[str, str, str], FederatedHospitalClient] = {}
        self._completed_rounds: set[tuple[str, int]] = set()

    def _headers(self) -> dict[str, str]:
        return {"X-OmniFM-Client-Token": self.token}

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.coordinator_url}{path}" if self.coordinator_url else path
        if self._http_client is not None:
            resp = self._http_client.get(url, params=params, headers=self._headers())
        else:
            with httpx.Client(timeout=self.request_timeout_sec) as client:
                resp = client.get(url, params=params, headers=self._headers())
        if resp.status_code >= 400:
            raise RuntimeError(
                f"GET {path} failed with HTTP {resp.status_code}: {resp.text}"
            )
        return dict(resp.json())

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.coordinator_url}{path}" if self.coordinator_url else path
        if self._http_client is not None:
            resp = self._http_client.post(url, json=payload, headers=self._headers())
        else:
            with httpx.Client(timeout=self.request_timeout_sec) as client:
                resp = client.post(url, json=payload, headers=self._headers())
        if resp.status_code >= 400:
            raise RuntimeError(
                f"POST {path} failed with HTTP {resp.status_code}: {resp.text}"
            )
        return dict(resp.json())

    def _get_or_create_fl_client(
        self,
        experiment_id: str,
        task_type: str,
        algorithm: str,
        adapter_kwargs: dict[str, Any],
        seed: int,
    ) -> FederatedHospitalClient:
        key = (experiment_id, task_type, algorithm)
        if key not in self._fl_clients:
            set_global_seed(seed)
            self._fl_clients[key] = FederatedHospitalClient(
                client_id=self.client_id,
                task_type=task_type,
                algorithm=algorithm,  # type: ignore[arg-type]
                cfg=settings,
                adapter_kwargs=adapter_kwargs,
            )
            logger.info(
                "Initialized local FederatedHospitalClient for experiment=%s, client=%s, task=%s, algorithm=%s on device=%s",
                experiment_id,
                self.client_id,
                task_type,
                algorithm,
                self.device,
            )
        return self._fl_clients[key]

    def poll_and_execute_once(self) -> dict[str, Any]:
        """Poll the coordinator once; if a round is awaiting this client, train locally and submit."""
        params: dict[str, Any] = {"client_id": self.client_id}
        if self.experiment_id:
            params["experiment_id"] = self.experiment_id

        task_info = self._get_json("/api/fl/rounds/current", params=params)
        status = str(task_info.get("status", "IDLE")).upper()
        exp_id = task_info.get("experiment_id")
        round_num = task_info.get("round_num")
        awaiting = bool(task_info.get("awaiting_submission", False))

        if not awaiting or exp_id is None or round_num is None:
            return {
                "action": "polled",
                "status": status,
                "experiment_id": exp_id,
                "round_num": round_num,
                "awaiting_submission": False,
            }

        round_key = (str(exp_id), int(round_num))
        if round_key in self._completed_rounds:
            return {
                "action": "already_submitted_locally",
                "status": status,
                "experiment_id": exp_id,
                "round_num": round_num,
                "awaiting_submission": False,
            }

        task_type = str(task_info["task_type"])
        algorithm = str(task_info["algorithm"])
        hyper = dict(task_info.get("hyperparameters") or task_info.get("round_config") or {})
        seed = int(hyper.get("seed", 42))
        adapter_kwargs = dict(task_info.get("adapter_kwargs") or hyper.get("adapter_kwargs") or {})

        shared_b64_str = task_info.pop("shared_state_b64", "")
        raw_shared = base64.b64decode(shared_b64_str, validate=True)
        del shared_b64_str
        global_shared_state = deserialize_state_dict_canonical(raw_shared)
        del raw_shared
        recomputed_global_hash = hash_state_dict(global_shared_state)
        expected_global_hash = str(
            task_info.get("global_shared_hash") or task_info.get("shared_state_hash") or ""
        ).strip().lower()
        if expected_global_hash and recomputed_global_hash.lower() != expected_global_hash:
            raise ValueError(
                f"Downloaded global shared state hash mismatch in round {round_num}: "
                f"expected {expected_global_hash}, got {recomputed_global_hash}."
            )

        fl_client = self._get_or_create_fl_client(
            experiment_id=str(exp_id),
            task_type=task_type,
            algorithm=algorithm,
            adapter_kwargs=adapter_kwargs,
            seed=seed,
        )

        knowledge_bank = _reconstruct_knowledge_bank(
            bank_b64=task_info.get("knowledge_bank_b64"),
            metadata_list=task_info.get("knowledge_bank_metadata"),
            embedding_dim=int(settings.embedding_dim),
        )

        batch_size = hyper.get("batch_size")
        max_train_samples = hyper.get("max_train_samples", 16)
        max_val_samples = hyper.get("max_val_samples", 16)
        scale = int(hyper.get("scale", 2))
        local_epochs = int(hyper.get("local_epochs", 1))
        lr = float(hyper.get("lr", 1e-3))
        fedprox_mu = float(hyper.get("fedprox_mu", 0.01))

        logger.info(
            "Starting local training on %s for experiment=%s round=%d client=%s (task=%s, alg=%s, epochs=%d)",
            self.device,
            exp_id,
            int(round_num),
            self.client_id,
            task_type,
            algorithm,
            local_epochs,
        )
        t0 = time.monotonic()
        update = fl_client.train_round(
            experiment_id=str(exp_id),
            round_num=int(round_num),
            shared_state_dict=global_shared_state,
            knowledge_bank=knowledge_bank,
            local_epochs=local_epochs,
            batch_size=int(batch_size) if batch_size is not None else None,
            max_train_samples=int(max_train_samples) if max_train_samples is not None else None,
            max_val_samples=int(max_val_samples) if max_val_samples is not None else None,
            lr=lr,
            fedprox_mu=fedprox_mu,
            scale=scale,
            device=self.device,
            seed=seed,
        )
        elapsed = time.monotonic() - t0

        # Strictly verify personalized parameters never leave Laptop B
        for key in update.shared_state_dict:
            if key.startswith("heads.") or "client_suffix_prompt" in key:
                raise RuntimeError(
                    f"Privacy policy violation: local parameter '{key}' found in shared_state_dict!"
                )

        shared_bytes = serialize_state_dict_canonical(update.shared_state_dict)
        shared_b64 = base64.b64encode(shared_bytes).decode("ascii")

        proto_b64: str | None = None
        if update.spectral_prototype is not None:
            proto_bytes = serialize_state_dict_canonical(
                {"spectral_prototype": update.spectral_prototype}
            )
            proto_b64 = base64.b64encode(proto_bytes).decode("ascii")

        eval_metrics_float: dict[str, float] = {}
        for k, v in (update.eval_metrics or {}).items():
            try:
                eval_metrics_float[str(k)] = float(v)
            except (TypeError, ValueError):
                pass

        submit_payload = {
            "client_id": self.client_id,
            "num_samples": int(update.num_samples),
            "shared_state_b64": shared_b64,
            "spectral_prototype_b64": proto_b64,
            "update_hash": update.update_hash,
            "train_loss": float(update.train_loss),
            "task_loss": float(update.task_loss),
            "align_loss": float(update.align_loss),
            "eval_metrics": eval_metrics_float,
            "peak_vram_mb": float(update.peak_vram_mb),
        }

        ack = self._post_json(
            f"/api/fl/rounds/{exp_id}/{int(round_num)}/submit",
            payload=submit_payload,
        )
        self._completed_rounds.add(round_key)
        logger.info(
            "Submitted round %d update for client=%s in %.2fs (hash=%s..., loss=%.4f, tx=%s)",
            int(round_num),
            self.client_id,
            elapsed,
            update.update_hash[:12],
            update.train_loss,
            ack.get("tx_hash") or ack.get("blockchain_status"),
        )
        return {
            "action": "submitted",
            "status": ack.get("status", "ACCEPTED"),
            "experiment_id": exp_id,
            "round_num": int(round_num),
            "update_hash": update.update_hash,
            "num_samples": update.num_samples,
            "train_loss": update.train_loss,
            "eval_metrics": update.eval_metrics,
            "ack": ack,
        }

    def run_until_complete(
        self,
        poll_interval_sec: float = 2.0,
        max_idle_polls: int = 300,
        max_retries: int = 10,
    ) -> list[dict[str, Any]]:
        """Continuously poll the coordinator and execute assigned rounds until experiment completion."""
        submissions: list[dict[str, Any]] = []
        idle_count = 0
        consecutive_errors = 0

        logger.info(
            "RemoteHospitalWorker polling %s as client_id='%s' on device='%s'...",
            self.coordinator_url or "<in-process>",
            self.client_id,
            self.device,
        )

        while True:
            try:
                res = self.poll_and_execute_once()
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    "Poll/submit error (%d/%d): %s",
                    consecutive_errors,
                    max_retries,
                    exc,
                )
                if consecutive_errors >= max_retries:
                    raise RuntimeError(
                        f"RemoteHospitalWorker exceeded max_retries ({max_retries}): {exc}"
                    ) from exc
                time.sleep(poll_interval_sec)
                continue

            action = res.get("action")
            status = str(res.get("status", "IDLE")).upper()

            if action == "submitted":
                submissions.append(res)
                idle_count = 0
                continue

            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                if submissions or self.experiment_id:
                    logger.info(
                        "Experiment '%s' reached terminal status '%s'. Remote worker exiting cleanly.",
                        res.get("experiment_id"),
                        status,
                    )
                    return submissions

            idle_count += 1
            if idle_count >= max_idle_polls:
                logger.info(
                    "Reached max_idle_polls (%d) with status='%s'. Exiting.",
                    max_idle_polls,
                    status,
                )
                return submissions

            time.sleep(poll_interval_sec)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniFM Standalone Remote Hospital Client Worker (Laptop B CPU)"
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        required=True,
        help="Base URL of Laptop A coordinator (e.g., http://192.168.1.10:8000)",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default="client_2",
        help="Logical hospital client ID (default: client_2)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional explicit experiment ID to bind to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for local training (default: cpu)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Shared client authentication token (defaults to FL_CLIENT_AUTH_TOKEN in .env)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds when waiting for next round (default: 2.0)",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=600,
        help="Maximum idle polls before exiting if no round is active (default: 600)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=15,
        help="Maximum consecutive HTTP errors before aborting (default: 15)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    worker = RemoteHospitalWorker(
        coordinator_url=args.coordinator_url,
        client_id=args.client_id,
        token=args.token,
        experiment_id=args.experiment_id,
        device=args.device,
    )
    submissions = worker.run_until_complete(
        poll_interval_sec=args.poll_interval,
        max_idle_polls=args.max_idle_polls,
        max_retries=args.max_retries,
    )
    logger.info(
        "RemoteHospitalWorker finished: %d round(s) submitted by %s.",
        len(submissions),
        args.client_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
