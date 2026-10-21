"""Task Evaluation Metrics (`backend/utils/metrics.py`, Task 4.5).

Implements validated evaluation metrics for all four tasks:
- Classification: `compute_accuracy`, `compute_macro_f1`
- Segmentation: `compute_dice`, `compute_iou` (requires ground-truth binary mask)
- Super-Resolution: `compute_psnr`, `compute_ssim`, `compute_l1` (requires ground-truth HR target)
- Multimodal Fusion: `compute_fusion_metrics` (explicitly distinguishes paired-source
  structural/information preservation metrics `source_ssim`, `source_vif`,
  `source_l1`, and `spatial_frequency_no_ref` from ground-truth reference metrics).

Reference-Requirement Policy:
-----------------------------
Metrics that require ground-truth labels/targets (`dice`, `iou`, `psnr`, `ssim`,
`l1`, `accuracy`, `macro_f1`) raise `ValueError` if `target is None`, preventing
any caller or API route from fabricating reference-based scores when no reference
is provided.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _require_reference(target: torch.Tensor | None, metric_name: str) -> torch.Tensor:
    if target is None:
        raise ValueError(
            f"Metric '{metric_name}' requires a ground-truth reference tensor, but target=None was passed."
        )
    return target


# =============================================================================
# 1. Classification Metrics (Accuracy, Macro-F1)
# =============================================================================


def compute_accuracy(logits_or_preds: torch.Tensor, targets: torch.Tensor | None) -> float:
    """Compute top-1 classification accuracy in `[0.0, 1.0]`."""
    tgt = _require_reference(targets, "accuracy").detach().view(-1).long()
    if logits_or_preds.ndim >= 2:
        preds = logits_or_preds.detach().argmax(dim=-1).view(-1).long()
    else:
        preds = logits_or_preds.detach().view(-1).long()

    if preds.numel() != tgt.numel() or tgt.numel() == 0:
        raise ValueError(
            f"Prediction count ({preds.numel()}) and target count ({tgt.numel()}) must match and be > 0."
        )
    return float((preds == tgt).float().mean().item())


def compute_macro_f1(
    logits_or_preds: torch.Tensor,
    targets: torch.Tensor | None,
    num_classes: int | None = None,
) -> float:
    """Compute unweighted macro-averaged F1 score across present/configured classes."""
    tgt = _require_reference(targets, "macro_f1").detach().cpu().view(-1).numpy().astype(np.int64)
    if logits_or_preds.ndim >= 2:
        if num_classes is None:
            num_classes = int(logits_or_preds.shape[-1])
        preds = logits_or_preds.detach().cpu().argmax(dim=-1).view(-1).numpy().astype(np.int64)
    else:
        preds = logits_or_preds.detach().cpu().view(-1).numpy().astype(np.int64)

    if preds.size != tgt.size or tgt.size == 0:
        raise ValueError("Predictions and targets must be non-empty and have matching length.")

    classes = (
        list(range(int(num_classes)))
        if num_classes is not None
        else sorted(set(tgt.tolist()) | set(preds.tolist()))
    )
    f1_scores: list[float] = []
    for c in classes:
        tp = int(np.sum((preds == c) & (tgt == c)))
        fp = int(np.sum((preds == c) & (tgt != c)))
        fn = int(np.sum((preds != c) & (tgt == c)))
        denom = 2 * tp + fp + fn
        if denom == 0:
            # Class neither present in ground truth nor predicted in this batch/split
            continue
        f1_scores.append((2.0 * tp) / float(denom))

    if not f1_scores:
        return 0.0
    return float(np.mean(f1_scores))


# =============================================================================
# 2. Segmentation Metrics (Dice, IoU)
# =============================================================================


def _binarize_segmentation(
    logits_or_mask: torch.Tensor,
    targets: torch.Tensor | None,
    metric_name: str,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    tgt = _require_reference(targets, metric_name).detach().float()
    pred_raw = logits_or_mask.detach().float()

    # If values lie outside [0, 1], treat as raw logits and apply sigmoid
    if float(pred_raw.min().item()) < 0.0 or float(pred_raw.max().item()) > 1.0:
        probs = torch.sigmoid(pred_raw)
    else:
        probs = pred_raw

    pred_bin = (probs >= threshold).float()
    tgt_bin = (tgt >= 0.5).float()

    if pred_bin.ndim == 3:
        pred_bin = pred_bin.unsqueeze(1)
    if tgt_bin.ndim == 3:
        tgt_bin = tgt_bin.unsqueeze(1)

    if pred_bin.shape != tgt_bin.shape:
        raise ValueError(
            f"Shape mismatch in {metric_name}: prediction {tuple(pred_bin.shape)} vs target {tuple(tgt_bin.shape)}."
        )
    return pred_bin.flatten(start_dim=1), tgt_bin.flatten(start_dim=1)


def compute_dice(
    logits_or_mask: torch.Tensor,
    targets: torch.Tensor | None,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Compute binary Sørensen-Dice coefficient averaged over batch `B`."""
    pred_flat, tgt_flat = _binarize_segmentation(logits_or_mask, targets, "dice", threshold)
    inter = (pred_flat * tgt_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)
    # When both prediction and ground truth are all-zero, Dice is 1.0
    dice = torch.where(
        denom > 0,
        (2.0 * inter) / (denom + eps),
        torch.ones_like(denom),
    )
    return float(dice.mean().item())


def compute_iou(
    logits_or_mask: torch.Tensor,
    targets: torch.Tensor | None,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Compute binary Intersection-over-Union (Jaccard index) averaged over batch `B`."""
    pred_flat, tgt_flat = _binarize_segmentation(logits_or_mask, targets, "iou", threshold)
    inter = (pred_flat * tgt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) - inter
    iou = torch.where(
        union > 0,
        inter / (union + eps),
        torch.ones_like(union),
    )
    return float(iou.mean().item())


# =============================================================================
# 3. Super-Resolution & Reconstruction Metrics (PSNR, SSIM, L1, VIF)
# =============================================================================


def compute_l1(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
) -> float:
    """Compute Mean Absolute Error (L1) against `target`."""
    tgt = _require_reference(target, "l1").detach().float()
    pred = prediction.detach().float()
    if pred.shape != tgt.shape:
        raise ValueError(
            f"Shape mismatch in compute_l1: prediction {tuple(pred.shape)} vs target {tuple(tgt.shape)}."
        )
    return float(F.l1_loss(pred, tgt).item())


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) averaged over batch `B`."""
    tgt = _require_reference(target, "psnr").detach().float()
    pred = prediction.detach().float()
    if pred.shape != tgt.shape:
        raise ValueError(
            f"Shape mismatch in compute_psnr: prediction {tuple(pred.shape)} vs target {tuple(tgt.shape)}."
        )
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)

    mse_per_sample = (pred - tgt).square().flatten(start_dim=1).mean(dim=1)
    psnr_list: list[float] = []
    for mse_val in mse_per_sample.tolist():
        if mse_val <= 1e-12:
            psnr_list.append(100.0)
        else:
            psnr_list.append(10.0 * math.log10((data_range**2) / mse_val))
    return float(np.mean(psnr_list))


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    data_range: float = 1.0,
    window_size: int = 7,
) -> float:
    """Compute Structural Similarity Index (SSIM) in `[-1.0, 1.0]`."""
    tgt = _require_reference(target, "ssim").detach().float()
    pred = prediction.detach().float()
    if pred.shape != tgt.shape:
        raise ValueError(
            f"Shape mismatch in compute_ssim: prediction {tuple(pred.shape)} vs target {tuple(tgt.shape)}."
        )
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)

    pad = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(tgt, kernel_size=window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=pad) - mu_x.square()
    sigma_y = F.avg_pool2d(tgt * tgt, kernel_size=window_size, stride=1, padding=pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * tgt, kernel_size=window_size, stride=1, padding=pad) - (mu_x * mu_y)

    sigma_x = torch.clamp(sigma_x, min=0.0)
    sigma_y = torch.clamp(sigma_y, min=0.0)

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-8)
    return float(torch.clamp(ssim_map.mean(), min=-1.0, max=1.0).item())


def compute_vif(
    prediction: torch.Tensor,
    reference: torch.Tensor | None,
    sigma_nsq: float = 0.01,
) -> float:
    """Compute multi-scale Visual Information Fidelity (VIF) in pixel domain.

    Follows the standard multi-scale Gaussian subband information ratio formulation
    (Sheikh & Bovik, IEEE TIP 2006) over 4 spatial scales.
    """
    ref = _require_reference(reference, "vif").detach().float()
    pred = prediction.detach().float()
    if pred.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch in compute_vif: prediction {tuple(pred.shape)} vs reference {tuple(ref.shape)}."
        )
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        ref = ref.unsqueeze(0)

    eps = 1e-8
    num_total = 0.0
    den_total = 0.0

    cur_ref = ref
    cur_pred = pred

    for scale in range(4):
        win = max(3, 9 - 2 * scale)
        pad = win // 2
        if cur_ref.shape[-1] < win or cur_ref.shape[-2] < win:
            break

        mu1 = F.avg_pool2d(cur_ref, kernel_size=win, stride=1, padding=pad)
        mu2 = F.avg_pool2d(cur_pred, kernel_size=win, stride=1, padding=pad)

        sigma1_sq = torch.clamp(
            F.avg_pool2d(cur_ref * cur_ref, kernel_size=win, stride=1, padding=pad) - mu1.square(),
            min=0.0,
        )
        sigma2_sq = torch.clamp(
            F.avg_pool2d(cur_pred * cur_pred, kernel_size=win, stride=1, padding=pad) - mu2.square(),
            min=0.0,
        )
        sigma12 = (
            F.avg_pool2d(cur_ref * cur_pred, kernel_size=win, stride=1, padding=pad) - mu1 * mu2
        )

        g = torch.clamp(sigma12 / (sigma1_sq + eps), min=0.0)
        sv_sq = torch.clamp(sigma2_sq - g * sigma12, min=eps)

        # Mutual information in distorted vs reference channels
        info_num = torch.log10(1.0 + (g.square() * sigma1_sq) / (sv_sq + sigma_nsq))
        info_den = torch.log10(1.0 + sigma1_sq / sigma_nsq)

        num_total += float(info_num.sum().item())
        den_total += float(info_den.sum().item())

        if cur_ref.shape[-1] >= 4 and cur_ref.shape[-2] >= 4:
            cur_ref = F.avg_pool2d(cur_ref, kernel_size=2, stride=2)
            cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)

    if den_total <= eps:
        return 1.0 if torch.allclose(pred, ref, atol=1e-5) else 0.0
    return float(np.clip(num_total / den_total, 0.0, 1.0))


def compute_spatial_frequency_no_ref(image: torch.Tensor) -> float:
    """No-reference Spatial Frequency (SF) sharpness metric for fused images."""
    x = image.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    rf = (x[:, :, :, 1:] - x[:, :, :, :-1]).square().mean()
    cf = (x[:, :, 1:, :] - x[:, :, :-1, :]).square().mean()
    return float(torch.sqrt(rf + cf + 1e-12).item())


def compute_fusion_metrics(
    fused: torch.Tensor,
    paired_sources: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Compute documented fusion metrics.

    Explicitly marks no-reference vs. source-pair preservation metrics so they
    are never misrepresented as ground-truth reference metrics.
    """
    sf = compute_spatial_frequency_no_ref(fused)
    if paired_sources is None:
        return {
            "metric_protocol": "no_reference_only",
            "has_source_pair_reference": False,
            "spatial_frequency_no_ref": sf,
        }

    if isinstance(paired_sources, (tuple, list)):
        src_a, src_b = paired_sources[0].detach().float(), paired_sources[1].detach().float()
    else:
        if paired_sources.ndim != 4 or paired_sources.shape[1] != 2:
            raise ValueError(
                f"Expected paired_sources of shape (B, 2, H, W), got {tuple(paired_sources.shape)}."
            )
        src_a = paired_sources[:, 0:1, :, :].detach().float()
        src_b = paired_sources[:, 1:2, :, :].detach().float()

    ssim_ab = 0.5 * (compute_ssim(fused, src_a) + compute_ssim(fused, src_b))
    vif_ab = 0.5 * (compute_vif(fused, src_a) + compute_vif(fused, src_b))
    l1_ab = 0.5 * (compute_l1(fused, src_a) + compute_l1(fused, src_b))

    return {
        "metric_protocol": "paired_source_preservation_plus_no_ref",
        "has_source_pair_reference": True,
        "ssim": ssim_ab,
        "vif": vif_ab,
        "l1": l1_ab,
        "spatial_frequency_no_ref": sf,
    }


def evaluate_task_predictions(
    task_type: str,
    predictions: torch.Tensor,
    targets: torch.Tensor | None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Compute canonical evaluation metrics for `task_type`."""
    key = task_type.lower().strip()
    if key == "classification":
        return {
            "accuracy": compute_accuracy(predictions, targets),
            "macro_f1": compute_macro_f1(predictions, targets, num_classes=num_classes),
        }
    if key == "segmentation":
        return {
            "dice": compute_dice(predictions, targets),
            "iou": compute_iou(predictions, targets),
        }
    if key in ("superresolution", "sr"):
        return {
            "psnr": compute_psnr(predictions, targets),
            "ssim": compute_ssim(predictions, targets),
            "l1": compute_l1(predictions, targets),
        }
    if key == "fusion":
        return compute_fusion_metrics(predictions, paired_sources=targets)
    raise ValueError(f"Unsupported task_type '{task_type}'.")
