"""CLI Runner for Phase 8 Experiments, Baselines, Ablations, and Results Export (`backend/fl/run_experiments.py`, Tasks 8.1 - 8.4).

Usage examples (PowerShell):
  .\venv\Scripts\python.exe -m backend.fl.run_experiments --mode main --task classification --rounds 20
  .\venv\Scripts\python.exe -m backend.fl.run_experiments --mode baselines --task classification --rounds 20
  .\venv\Scripts\python.exe -m backend.fl.run_experiments --mode ablations --task classification --rounds 20
  .\venv\Scripts\python.exe -m backend.fl.run_experiments --mode export
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from sqlalchemy import asc, select
import torch

from backend.blockchain.web3_connector import Web3Connector
from backend.config import settings
from backend.database.db import init_db, session_scope
from backend.database.models import Experiment, GlobalModelVersion, MetricRecord
from backend.fl.protocols import AlgorithmType
from backend.fl.server import FederatedLearningServer
from backend.utils.logging import setup_logger

logger = setup_logger("omnifm.experiments")

ALL_TASKS = ("classification", "segmentation", "superresolution", "fusion")


def run_single_experiment(
    experiment_id: str,
    task_type: str,
    algorithm: AlgorithmType = "omnifm",
    num_rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int | None = None,
    max_train_samples: int | None = 64,
    max_val_samples: int | None = 32,
    lr: float = 1e-3,
    fedprox_mu: float = 0.01,
    scale: int = 2,
    seed: int = 42,
    audit_on_chain: bool = False,
    adapter_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a multi-round federated experiment and persist checkpoints, audit records, and metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w3: Web3Connector | None = None
    if audit_on_chain or settings.blockchain_audit_enabled:
        try:
            candidate = Web3Connector(cfg=settings)
            if candidate.is_connected() and candidate.is_contract_deployed():
                w3 = candidate
                logger.info("Connected to blockchain node at %s", settings.ganache_url)
            else:
                logger.warning("Blockchain node/contract not ready; recording audit metadata in SQLite only.")
        except Exception as exc:
            logger.warning("Blockchain connection skipped: %s", exc)

    server = FederatedLearningServer(
        experiment_id=experiment_id,
        task_type=task_type,
        algorithm=algorithm,
        client_ids=("client_1", "client_2", "client_3"),
        cfg=settings,
        web3_connector=w3,
        audit_required=False,
        adapter_kwargs=adapter_kwargs or {},
        seed=seed,
    )

    t0 = time.perf_counter()
    r0_hash = server.initialize_round_zero(total_rounds=num_rounds)
    logger.info(
        "Started %s | task=%s | alg=%s | rounds=%d | device=%s | round0_hash=%s",
        experiment_id,
        task_type,
        algorithm,
        num_rounds,
        device,
        r0_hash[:14],
    )

    last_metrics: dict[str, Any] = {}
    peak_vram = 0.0
    for r in range(1, num_rounds + 1):
        res = server.run_round(
            round_num=r,
            local_epochs=local_epochs,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            lr=lr,
            fedprox_mu=fedprox_mu,
            scale=scale,
            device=device,
        )
        last_metrics = res.global_metrics
        peak_vram = max(peak_vram, res.peak_vram_mb)
        logger.info(
            "[%s] Round %d/%d complete | hash=%s | peak_vram=%.2f MB | metrics=%s",
            experiment_id,
            r,
            num_rounds,
            res.global_model_hash[:14],
            res.peak_vram_mb,
            res.global_metrics,
        )

    elapsed = round(time.perf_counter() - t0, 2)
    with session_scope(settings.database_url) as session:
        exp = session.execute(
            select(Experiment).where(Experiment.experiment_id == experiment_id)
        ).scalar_one_or_none()
        if exp is not None:
            exp.status = "COMPLETED"
            exp.current_round = num_rounds
            exp.peak_vram_mb = max(float(exp.peak_vram_mb), peak_vram)
            exp.updated_at = datetime.now(timezone.utc)

    logger.info(
        "Completed %s in %.2fs | peak_vram=%.2f MB | final_metrics=%s",
        experiment_id,
        elapsed,
        peak_vram,
        last_metrics,
    )
    return {
        "experiment_id": experiment_id,
        "task_type": task_type,
        "algorithm": algorithm,
        "rounds": num_rounds,
        "elapsed_sec": elapsed,
        "peak_vram_mb": peak_vram,
        "final_metrics": last_metrics,
    }


def export_results(output_dir: str | Path = "./results") -> tuple[Path, Path]:
    """Export all measured SQLite experiment and round records to `results/results.md` and `results/metrics_summary.csv`."""
    init_db(settings.database_url)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "metrics_summary.csv"
    md_file = out_path / "results.md"

    with session_scope(settings.database_url) as session:
        exps = (
            session.execute(select(Experiment).order_by(asc(Experiment.id)))
            .scalars()
            .all()
        )
        metrics = (
            session.execute(
                select(MetricRecord)
                .where(MetricRecord.scope == "global")
                .order_by(asc(MetricRecord.experiment_id), asc(MetricRecord.round_num))
            )
            .scalars()
            .all()
        )
        gmvs = (
            session.execute(select(GlobalModelVersion))
            .scalars()
            .all()
        )

    hash_by_exp_round = {(g.experiment_id, g.round_num): g.model_hash for g in gmvs}

    csv_rows: list[dict[str, Any]] = []
    for m in metrics:
        row: dict[str, Any] = {
            "experiment_id": m.experiment_id,
            "task_type": m.task_type,
            "algorithm": m.algorithm,
            "round_num": m.round_num,
            "global_model_hash": hash_by_exp_round.get((m.experiment_id, m.round_num), ""),
            "train_loss": m.metrics_json.get("train_loss", ""),
            "task_loss": m.metrics_json.get("task_loss", ""),
            "align_loss": m.metrics_json.get("align_loss", ""),
            "accuracy": m.metrics_json.get("accuracy", ""),
            "macro_f1": m.metrics_json.get("macro_f1", ""),
            "dice": m.metrics_json.get("dice", ""),
            "iou": m.metrics_json.get("iou", ""),
            "psnr": m.metrics_json.get("psnr", ""),
            "ssim": m.metrics_json.get("ssim", ""),
            "vif": m.metrics_json.get("vif", ""),
            "l1": m.metrics_json.get("l1", ""),
            "spatial_frequency_no_ref": m.metrics_json.get("spatial_frequency_no_ref", ""),
            "timestamp": m.created_at.isoformat() if m.created_at else "",
        }
        csv_rows.append(row)

    fieldnames = [
        "experiment_id",
        "task_type",
        "algorithm",
        "round_num",
        "global_model_hash",
        "train_loss",
        "task_loss",
        "align_loss",
        "accuracy",
        "macro_f1",
        "dice",
        "iou",
        "psnr",
        "ssim",
        "vif",
        "l1",
        "spatial_frequency_no_ref",
        "timestamp",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    lines = [
        "# OmniFM Blockchain Federated Learning — Measured Experimental Results",
        "",
        f"- **Generated At:** `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- **Hardware:** `{gpu_name}` (`PyTorch {torch.__version__}`)",
        "- **Client Setup:** 3 sequential logical hospital nodes (`client_1`, `client_2`, `client_3`)",
        "- **Data Integrity Policy:** All values below are directly exported from SQLite `experiments` and `metric_records` tables.",
        "",
        "## 1. Experiment Summary Table",
        "",
        "| Experiment ID | Task | Algorithm | Status | Rounds | Seed | Peak VRAM (MB) | Final Global Metrics |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]

    latest_metric_by_exp: dict[str, dict[str, Any]] = {}
    for m in metrics:
        latest_metric_by_exp[m.experiment_id] = m.metrics_json

    for exp in exps:
        final_m = latest_metric_by_exp.get(exp.experiment_id, {})
        lines.append(
            f"| `{exp.experiment_id}` | `{exp.task_type}` | `{exp.algorithm}` | "
            f"`{exp.status}` | {exp.current_round}/{exp.total_rounds} | {exp.seed} | "
            f"{float(exp.peak_vram_mb or 0.0):.2f} | `{final_m}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Metric Definitions & Reference Protocols",
            "",
            "- **Classification (`dermamnist`, `pathmnist`, `bloodmnist`):** Multi-class `accuracy` and unweighted `macro_f1`. Classifier heads have distinct class semantics (7, 9, and 8 classes) and are kept local (`personalized_parameters`); only the shared ResNet-18 + OmniFM backbone is aggregated.",
            "- **Segmentation (`FeTS2022` T1ce MRI 128×128):** Foreground binary `dice` and `iou` evaluated strictly on labeled validation slices.",
            "- **Super-Resolution (`BreaKHis` 64×64 LR -> x2/x4/x8 HR):** Full-reference `psnr` (dB), `ssim`, and `l1` evaluated strictly on patient-split paired HR test patches.",
            "- **Multimodal Fusion (`CT-MRI`, `PET-MRI`, `SPECT-MRI` 128×128):** Source-pair structural preservation (`ssim`, `vif`, `l1` averaged across both co-registered input modalities) plus no-reference `spatial_frequency_no_ref`.",
            "",
            "## 3. Research Limitations",
            "",
            "- On-chain SHA-256 hashes and Ganache/Hardhat transactions verify artifact immutability and audit provenance after recording, not cryptographic proof of honest client training.",
            "- Single-GPU sequential client execution simulates multi-hospital federation under a 6 GB VRAM ceiling.",
        ]
    )

    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported %d metric rows to %s and %s", len(csv_rows), csv_file, md_file)
    return md_file, csv_file


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniFM Phase 8 Experiment & Baseline Runner")
    parser.add_argument(
        "--mode",
        choices=("main", "baselines", "ablations", "export"),
        default="main",
        help="Execution mode: main (OmniFM), baselines (FedAvg/FedProx/FedPer), ablations, or export",
    )
    parser.add_argument(
        "--task",
        choices=(*ALL_TASKS, "all"),
        default="classification",
        help="Target medical imaging task (or 'all')",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--max-train-samples", type=int, default=64, help="Max training samples per client per round (0 for full)")
    parser.add_argument("--max-val-samples", type=int, default=32, help="Max validation samples per client per round (0 for full)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client learning rate")
    parser.add_argument("--scale", type=int, default=2, choices=(2, 4, 8), help="Super-resolution scale factor")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--audit", action="store_true", help="Record transactions on local blockchain RPC if connected")

    args = parser.parse_args()

    if args.mode == "export":
        export_results()
        return

    tasks = ALL_TASKS if args.task == "all" else (args.task,)
    max_tr = None if args.max_train_samples <= 0 else args.max_train_samples
    max_va = None if args.max_val_samples <= 0 else args.max_val_samples

    for t in tasks:
        base_adapter_kwargs: dict[str, Any] = {}
        if t == "superresolution":
            base_adapter_kwargs["default_scale"] = args.scale

        if args.mode == "main":
            exp_id = f"omnifm_{t}_r{args.rounds}_s{args.seed}"
            run_single_experiment(
                experiment_id=exp_id,
                task_type=t,
                algorithm="omnifm",
                num_rounds=args.rounds,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                max_train_samples=max_tr,
                max_val_samples=max_va,
                lr=args.lr,
                scale=args.scale,
                seed=args.seed,
                audit_on_chain=args.audit,
                adapter_kwargs=base_adapter_kwargs,
            )

        elif args.mode == "baselines":
            for alg in ("fedavg", "fedprox", "fedper"):
                exp_id = f"{alg}_{t}_r{args.rounds}_s{args.seed}"
                run_single_experiment(
                    experiment_id=exp_id,
                    task_type=t,
                    algorithm=alg,  # type: ignore[arg-type]
                    num_rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    max_train_samples=max_tr,
                    max_val_samples=max_va,
                    lr=args.lr,
                    scale=args.scale,
                    seed=args.seed,
                    audit_on_chain=args.audit,
                    adapter_kwargs=base_adapter_kwargs,
                )

        elif args.mode == "ablations":
            ablation_specs = [
                ("no_gskr", {"enable_gskr": False}),
                ("no_eca", {"enable_eca": False}),
                ("no_psp", {"enable_psp": False}),
                ("no_spalign", {"enable_spalign": False}),
            ]
            for tag, flags in ablation_specs:
                exp_id = f"ablation_{tag}_{t}_r{args.rounds}_s{args.seed}"
                kw = {**base_adapter_kwargs, **flags}
                run_single_experiment(
                    experiment_id=exp_id,
                    task_type=t,
                    algorithm="omnifm",
                    num_rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    max_train_samples=max_tr,
                    max_val_samples=max_va,
                    lr=args.lr,
                    scale=args.scale,
                    seed=args.seed,
                    audit_on_chain=args.audit,
                    adapter_kwargs=kw,
                )

    export_results()


if __name__ == "__main__":
    main()
