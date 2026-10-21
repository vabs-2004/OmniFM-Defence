"""Sequential Logical Hospital Client Trainer (`backend/fl/client.py`, Tasks 5.2 & 5.8).

Each logical client:
- Loads only its assigned local dataset partition (`client_1`, `client_2`, `client_3`).
- Retains personalized parameters (`client_suffix_prompt`, incompatible dataset
  classification heads, and FedPer personalized decoder layers) locally across rounds.
- Supports `"omnifm"`, `"fedavg"`, `"fedprox"` (with proximal term `(mu / 2) * ||w - w_global||^2`),
  and `"fedper"` baselines.
- Releases GPU memory at the end of its round execution so sequential simulation
  fits comfortably inside the 6 GB RTX 3050 VRAM budget.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from backend.config import Settings, settings
from backend.fl.aggregation import filter_shared_state_dict
from backend.fl.protocols import (
    AlgorithmType,
    ClientUpdatePayload,
    ParameterSharingPolicy,
    build_parameter_sharing_policy,
)
from backend.omnifm.gskr import SpectralKnowledgeBank
from backend.tasks import BaseTaskAdapter, create_task_adapter
from backend.utils.hashing import hash_state_dict
from backend.utils.metrics import evaluate_task_predictions
from backend.utils.preprocessing import build_client_dataloader


class FederatedHospitalClient:
    """Simulated hospital client executing local training sequentially on CPU or GPU."""

    def __init__(
        self,
        client_id: str,
        task_type: str,
        algorithm: AlgorithmType = "omnifm",
        hospital_name: str | None = None,
        cfg: Settings | None = None,
        local_checkpoint_dir: str | Path | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client_id = client_id
        self.task_type = task_type.lower().strip()
        self.algorithm: AlgorithmType = algorithm
        self.hospital_name = hospital_name or f"Hospital_{client_id}"
        self.cfg = cfg or settings
        self.adapter_kwargs = dict(adapter_kwargs or {})

        base_ckpt_dir = Path(local_checkpoint_dir or (self.cfg.artifact_dir / "clients" / client_id))
        base_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.local_checkpoint_dir = base_ckpt_dir

        # Client-local personalized & incompatible-head state persisted across rounds
        self._personalized_state: dict[str, torch.Tensor] = {}

    @property
    def personalized_state(self) -> dict[str, torch.Tensor]:
        """Return copy of client-local personalized parameters."""
        return {k: v.detach().cpu().clone() for k, v in self._personalized_state.items()}

    def _build_local_adapter(self) -> BaseTaskAdapter:
        """Instantiate the task adapter configured for `self.algorithm` and any ablation overrides."""
        is_omnifm = self.algorithm == "omnifm"
        kwargs: dict[str, Any] = {
            "embedding_dim": self.cfg.embedding_dim,
            "lambda_align": self.cfg.spalign_lambda if is_omnifm else 0.0,
            "enable_gskr": is_omnifm,
            "enable_eca": is_omnifm,
            "enable_psp": is_omnifm,
            "enable_spalign": is_omnifm,
        }
        kwargs.update(self.adapter_kwargs)
        return create_task_adapter(self.task_type, **kwargs)

    def load_shared_and_local_state(
        self,
        adapter: BaseTaskAdapter,
        shared_state_dict: Mapping[str, torch.Tensor],
        policy: ParameterSharingPolicy,
    ) -> None:
        """Load server-broadcast shared parameters in-place while preserving client-local personalized parameters."""
        target_state = adapter.state_dict()
        with torch.no_grad():
            # 1. Apply server shared parameters (never overwriting excluded personalized keys)
            for k, v in shared_state_dict.items():
                if k not in policy.excluded_from_aggregation and k in target_state:
                    target_state[k].copy_(v, non_blocking=True)

            # 2. Restore client-local personalized parameters from previous round if available
            for k, v in self._personalized_state.items():
                if k in target_state and target_state[k].shape == v.shape:
                    target_state[k].copy_(v, non_blocking=True)

    def _save_personalized_state(
        self,
        adapter: BaseTaskAdapter,
        policy: ParameterSharingPolicy,
        experiment_id: str,
        round_num: int,
    ) -> None:
        """Extract and persist client-local personalized parameters after local training."""
        full_state = adapter.state_dict()
        with torch.no_grad():
            for k in policy.excluded_from_aggregation:
                if k in full_state:
                    src = full_state[k].detach()
                    if k in self._personalized_state and self._personalized_state[k].shape == src.shape:
                        self._personalized_state[k].copy_(src, non_blocking=True)
                    else:
                        self._personalized_state[k] = src.to(device="cpu").clone()
        ckpt_file = self.local_checkpoint_dir / f"{experiment_id}_r{round_num}_personalized.pt"
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._personalized_state, ckpt_file)

    @staticmethod
    def compute_fedprox_term(
        adapter: BaseTaskAdapter,
        global_shared_params: Mapping[str, torch.Tensor],
        shared_param_names: frozenset[str],
        mu: float,
    ) -> torch.Tensor:
        """Compute FedProx proximal penalty `(mu / 2) * sum(||w - w_global||^2)` over shared parameters."""
        if mu <= 0.0:
            return torch.zeros((), dtype=torch.float32)

        prox_sq_sum: torch.Tensor | None = None
        for name, param in adapter.named_parameters():
            if name in shared_param_names and name in global_shared_params:
                ref = global_shared_params[name].to(device=param.device, dtype=torch.float32)
                diff_sq = (param.float() - ref).square().sum()
                prox_sq_sum = diff_sq if prox_sq_sum is None else (prox_sq_sum + diff_sq)

        if prox_sq_sum is None:
            return torch.zeros((), dtype=torch.float32)
        return 0.5 * float(mu) * prox_sq_sum

    def train_round(
        self,
        experiment_id: str,
        round_num: int,
        shared_state_dict: Mapping[str, torch.Tensor],
        knowledge_bank: SpectralKnowledgeBank | None = None,
        policy: ParameterSharingPolicy | None = None,
        local_epochs: int | None = None,
        batch_size: int | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        lr: float = 1e-3,
        fedprox_mu: float = 0.01,
        scale: int = 2,
        device: torch.device | str | None = None,
        seed: int = 42,
        reusable_adapter: BaseTaskAdapter | None = None,
    ) -> ClientUpdatePayload:
        """Execute one federated round of local training and evaluation for this hospital client."""
        t0 = time.perf_counter()
        dev = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        use_amp = dev.type == "cuda" and self.cfg.precision_policy == "fp16_amp"

        if dev.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.reset_peak_memory_stats(dev)

        owns_adapter = reusable_adapter is None
        adapter = reusable_adapter if reusable_adapter is not None else self._build_local_adapter()
        adapter.to(dev)
        active_policy = policy or build_parameter_sharing_policy(adapter, algorithm=self.algorithm)
        self.load_shared_and_local_state(adapter, shared_state_dict, active_policy)

        # Keep reference to shared parameters if FedProx is active
        global_shared_ref: Mapping[str, torch.Tensor] = (
            shared_state_dict if self.algorithm == "fedprox" else {}
        )

        train_loader, _train_stats = build_client_dataloader(
            task_type=self.task_type,
            client_id=self.client_id,
            split="train",
            batch_size=batch_size,
            max_samples=max_train_samples,
            scale=scale,
            seed=seed + round_num,
        )
        val_loader, _val_stats = build_client_dataloader(
            task_type=self.task_type,
            client_id=self.client_id,
            split="val",
            batch_size=batch_size,
            max_samples=max_val_samples or max_train_samples,
            scale=scale,
            shuffle=False,
            seed=seed,
        )

        epochs = int(local_epochs if local_epochs is not None else self.cfg.default_local_epochs)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        adapter.train()
        total_samples = 0
        total_loss_accum = 0.0
        task_loss_accum = 0.0
        align_loss_accum = 0.0
        num_steps = 0
        collected_spectral_tokens: list[torch.Tensor] = []

        for epoch_idx in range(epochs):
            for batch_x, batch_y in train_loader:
                bsz = int(batch_x.shape[0])
                if epoch_idx == 0:
                    total_samples += bsz

                batch_x = batch_x.to(dev, non_blocking=True)
                batch_y = batch_y.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=dev.type,
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    fw_out = adapter(
                        batch_x,
                        knowledge_bank=knowledge_bank,
                        client_id=self.client_id,
                        scale=scale,
                    )
                    loss_breakdown = adapter.compute_total_loss(
                        fw_out,
                        batch_y,
                        inputs=batch_x,
                    )
                    step_loss = loss_breakdown.total_loss
                    if self.algorithm == "fedprox":
                        prox_term = self.compute_fedprox_term(
                            adapter=adapter,
                            global_shared_params=global_shared_ref,
                            shared_param_names=active_policy.shared_param_names,
                            mu=fedprox_mu,
                        )
                        step_loss = step_loss + prox_term.to(device=dev)

                scaler.scale(step_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss_accum += float(step_loss.detach().item())
                task_loss_accum += float(loss_breakdown.task_loss.detach().item())
                align_loss_accum += float(loss_breakdown.align_loss.detach().item())
                num_steps += 1

                if epoch_idx == epochs - 1:
                    collected_spectral_tokens.append(
                        fw_out.conditioned.local_spectral_token.detach().cpu().float()
                    )

        # Evaluate on validation split
        eval_metrics = self._evaluate(adapter, val_loader, knowledge_bank, scale, dev, use_amp)

        # Save personalized parameters locally and extract shared parameters
        self._save_personalized_state(adapter, active_policy, experiment_id, round_num)
        updated_shared = filter_shared_state_dict(adapter.state_dict(), active_policy)
        upd_hash = hash_state_dict(updated_shared)

        # Compute mean L2-normalized spectral prototype for server knowledge bank
        mean_proto: torch.Tensor | None = None
        if collected_spectral_tokens and adapter.omnifm.enable_gskr:  # type: ignore[attr-defined]
            cat_tokens = torch.cat(collected_spectral_tokens, dim=0)
            mean_proto = F.normalize(cat_tokens.mean(dim=0, keepdim=True), p=2.0, dim=-1, eps=1e-8)

        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(dev) / (1024**2), 2)
            if dev.type == "cuda"
            else 0.0
        )
        duration_sec = round(time.perf_counter() - t0, 4)

        # Release optimizer/scaler state
        del optimizer, scaler
        if owns_adapter:
            del adapter
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        denom = max(1, num_steps)
        return ClientUpdatePayload(
            experiment_id=experiment_id,
            round_num=int(round_num),
            client_id=self.client_id,
            task_type=self.task_type,
            algorithm=self.algorithm,
            num_samples=max(1, total_samples),
            shared_state_dict=updated_shared,
            update_hash=upd_hash,
            spectral_prototype=mean_proto,
            train_loss=round(total_loss_accum / denom, 6),
            task_loss=round(task_loss_accum / denom, 6),
            align_loss=round(align_loss_accum / denom, 6),
            eval_metrics=eval_metrics,
            duration_sec=duration_sec,
            peak_vram_mb=peak_vram_mb,
        )

    @torch.no_grad()
    def _evaluate(
        self,
        adapter: BaseTaskAdapter,
        val_loader: Any,
        knowledge_bank: SpectralKnowledgeBank | None,
        scale: int,
        dev: torch.device,
        use_amp: bool,
    ) -> dict[str, Any]:
        adapter.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        num_classes: int | None = None

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)
            with torch.amp.autocast(
                device_type=dev.type,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                fw_out = adapter(
                    batch_x,
                    knowledge_bank=knowledge_bank,
                    client_id=self.client_id,
                    scale=scale,
                )
            all_preds.append(fw_out.prediction.detach().cpu().float())
            all_targets.append(batch_y.detach().cpu())
            if "num_classes" in fw_out.extra:
                num_classes = int(fw_out.extra["num_classes"])

        if not all_preds:
            return {}

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        return evaluate_task_predictions(
            task_type=self.task_type,
            predictions=cat_preds,
            targets=cat_targets,
            num_classes=num_classes,
        )
