"""Global Spectral Knowledge Retrieval (`backend/omnifm/gskr.py`, Task 2.2).

Manages the persisted server-side spectral knowledge bank and performs top-k
cosine-similarity prototype retrieval with task-aware filtering, empty/small-bank
handling, capacity pruning, and self-retrieval protection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PrototypeMetadata:
    """Metadata attached to a stored spectral prototype in the knowledge bank."""

    prototype_id: str
    task_type: str
    client_id: str
    round_num: int
    sample_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GSKRRetrievalResult:
    """Result of querying the Global Spectral Knowledge Bank for a batch `(B, D)`.

    Attributes:
        prototypes: Retrieved prototypes of shape `(B, K_actual, D)` where
            `0 <= K_actual <= top_k`. When the eligible bank is empty, `K_actual == 0`.
        similarities: Cosine similarities of shape `(B, K_actual)`.
        indices: Selected indices in the filtered bank of shape `(B, K_actual)`.
        has_valid: True when `K_actual > 0` (at least one valid prototype retrieved).
    """

    prototypes: torch.Tensor
    similarities: torch.Tensor
    indices: torch.Tensor
    has_valid: bool


class SpectralKnowledgeBank:
    """Server-managed repository of L2-normalized spectral prototypes `(M, D)`.

    Supports task-aware filtering, FIFO capacity pruning, and disk persistence
    to `backend/fl/knowledge_bank.pt`.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        max_size: int = 1000,
        top_k: int = 2,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}.")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        self.embedding_dim = int(embedding_dim)
        self.max_size = int(max_size)
        self.top_k = int(top_k)

        self._embeddings: torch.Tensor = torch.empty((0, self.embedding_dim), dtype=torch.float32)
        self._metadata: list[PrototypeMetadata] = []
        self._counter: int = 0

    def __len__(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def embeddings(self) -> torch.Tensor:
        """Return stored prototypes tensor on CPU of shape `(M, D)`."""
        return self._embeddings

    @property
    def metadata(self) -> list[PrototypeMetadata]:
        """Return copy of prototype metadata list."""
        return list(self._metadata)

    def clear(self) -> None:
        """Remove all stored prototypes and reset bank."""
        self._embeddings = torch.empty((0, self.embedding_dim), dtype=torch.float32)
        self._metadata.clear()

    def add_prototypes(
        self,
        embeddings: torch.Tensor,
        task_type: str = "unspecified",
        client_id: str = "unknown",
        round_num: int = 0,
        sample_count: int = 1,
        prototype_ids: list[str] | None = None,
    ) -> list[str]:
        """Add one or more spectral prototypes `(N, D)` or `(D,)` and prune to `max_size`.

        Prototypes are detached, cast to float32 on CPU, and L2-normalized before storage.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected prototype tensor of shape (N, {self.embedding_dim}), "
                f"got {tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("Cannot add non-finite (NaN/Inf) spectral prototypes to knowledge bank.")

        count = int(embeddings.shape[0])
        if count == 0:
            return []

        if prototype_ids is not None and len(prototype_ids) != count:
            raise ValueError(
                f"Length of prototype_ids ({len(prototype_ids)}) must match embeddings count ({count})."
            )

        norm_emb = F.normalize(
            embeddings.detach().cpu().float(),
            p=2.0,
            dim=-1,
            eps=1e-8,
        )

        added_ids: list[str] = []
        new_meta: list[PrototypeMetadata] = []
        for idx in range(count):
            self._counter += 1
            pid = (
                prototype_ids[idx]
                if prototype_ids is not None
                else f"proto_{task_type}_{client_id}_r{round_num}_{self._counter}"
            )
            added_ids.append(pid)
            new_meta.append(
                PrototypeMetadata(
                    prototype_id=pid,
                    task_type=task_type,
                    client_id=client_id,
                    round_num=int(round_num),
                    sample_count=int(sample_count),
                )
            )

        if self._embeddings.shape[0] == 0:
            self._embeddings = norm_emb
        else:
            self._embeddings = torch.cat([self._embeddings, norm_emb], dim=0)
        self._metadata.extend(new_meta)

        # FIFO capacity pruning: retain most recent `max_size` prototypes
        if self._embeddings.shape[0] > self.max_size:
            excess = int(self._embeddings.shape[0]) - self.max_size
            self._embeddings = self._embeddings[excess:].contiguous()
            self._metadata = self._metadata[excess:]

        return added_ids

    def retrieve(
        self,
        query_tokens: torch.Tensor,
        top_k: int | None = None,
        task_type: str | None = None,
        exclude_prototype_ids: set[str] | None = None,
        exclude_client_round: tuple[str, int] | None = None,
        avoid_exact_self: bool = True,
    ) -> GSKRRetrievalResult:
        """Retrieve up to `top_k` global spectral prototypes via cosine similarity.

        Retrieval Ordering & Self-Retrieval Protection:
        - In the federated lifecycle, clients query the knowledge bank snapshot from
          completed rounds *before* new local prototypes from the current round are
          inserted.
        - Additionally, callers can pass `exclude_prototype_ids`, `exclude_client_round=(client_id, round_num)`,
          or `avoid_exact_self=True` (which masks out any candidate prototype whose
          cosine similarity with the query is `>= 1.0 - 1e-6` when other candidates exist,
          or masks exact-identity vectors inserted from the same query).
        """
        if query_tokens.ndim != 2 or query_tokens.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected query_tokens of shape (B, {self.embedding_dim}), "
                f"got {tuple(query_tokens.shape)}."
            )

        bsz = int(query_tokens.shape[0])
        k_req = int(top_k if top_k is not None else self.top_k)
        if k_req < 1:
            raise ValueError(f"top_k must be >= 1, got {k_req}.")

        device = query_tokens.device
        dtype = query_tokens.dtype

        if self.is_empty:
            return GSKRRetrievalResult(
                prototypes=torch.empty((bsz, 0, self.embedding_dim), device=device, dtype=dtype),
                similarities=torch.empty((bsz, 0), device=device, dtype=dtype),
                indices=torch.empty((bsz, 0), device=device, dtype=torch.long),
                has_valid=False,
            )

        # Filter eligible prototypes by task_type and exclusion rules
        eligible_indices: list[int] = []
        for idx, meta in enumerate(self._metadata):
            if task_type is not None and meta.task_type != task_type:
                continue
            if exclude_prototype_ids and meta.prototype_id in exclude_prototype_ids:
                continue
            if (
                exclude_client_round is not None
                and meta.client_id == exclude_client_round[0]
                and meta.round_num == exclude_client_round[1]
            ):
                continue
            eligible_indices.append(idx)

        if not eligible_indices:
            return GSKRRetrievalResult(
                prototypes=torch.empty((bsz, 0, self.embedding_dim), device=device, dtype=dtype),
                similarities=torch.empty((bsz, 0), device=device, dtype=dtype),
                indices=torch.empty((bsz, 0), device=device, dtype=torch.long),
                has_valid=False,
            )

        bank_subset = self._embeddings[eligible_indices].to(device=device, dtype=torch.float32)
        bank_norm = F.normalize(bank_subset, p=2.0, dim=-1, eps=1e-8)  # (M_elig, D)
        q_norm = F.normalize(query_tokens.detach().float(), p=2.0, dim=-1, eps=1e-8)  # (B, D)

        # Cosine similarity matrix: (B, M_elig)
        sim_matrix = torch.matmul(q_norm, bank_norm.transpose(0, 1))

        if avoid_exact_self:
            # Check for exact self-match (L2 distance < 1e-6) and mask it out
            # so a query vector that was just added to the bank cannot retrieve itself.
            dists = torch.cdist(q_norm, bank_norm, p=2.0)  # (B, M_elig)
            self_mask = dists < 1e-5
            if self_mask.any():
                sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))

        # Check how many valid candidates remain per batch item
        valid_counts = (sim_matrix > float("-inf")).sum(dim=-1)
        min_valid = int(valid_counts.min().item())
        if min_valid == 0:
            return GSKRRetrievalResult(
                prototypes=torch.empty((bsz, 0, self.embedding_dim), device=device, dtype=dtype),
                similarities=torch.empty((bsz, 0), device=device, dtype=dtype),
                indices=torch.empty((bsz, 0), device=device, dtype=torch.long),
                has_valid=False,
            )

        k_actual = min(k_req, min_valid)
        top_sims, top_rel_idx = torch.topk(sim_matrix, k=k_actual, dim=-1, largest=True, sorted=True)

        # Gather prototypes: (B, K_actual, D)
        expanded_idx = top_rel_idx.unsqueeze(-1).expand(bsz, k_actual, self.embedding_dim)
        expanded_bank = bank_norm.unsqueeze(0).expand(bsz, bank_norm.shape[0], self.embedding_dim)
        retrieved = torch.gather(expanded_bank, dim=1, index=expanded_idx).to(dtype=dtype)

        eligible_tensor = torch.tensor(eligible_indices, device=device, dtype=torch.long)
        abs_indices = eligible_tensor[top_rel_idx]

        return GSKRRetrievalResult(
            prototypes=retrieved,
            similarities=top_sims.to(dtype=dtype),
            indices=abs_indices,
            has_valid=True,
        )

    def save(self, path: str | Path) -> Path:
        """Persist the knowledge bank embeddings and metadata to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "embedding_dim": self.embedding_dim,
            "max_size": self.max_size,
            "top_k": self.top_k,
            "counter": self._counter,
            "embeddings": self._embeddings.cpu(),
            "metadata": [m.to_dict() for m in self._metadata],
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "SpectralKnowledgeBank":
        """Load a persisted knowledge bank from disk."""
        in_path = Path(path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Knowledge bank file not found at {in_path}")
        payload = torch.load(in_path, map_location="cpu", weights_only=False)
        bank = cls(
            embedding_dim=int(payload["embedding_dim"]),
            max_size=int(payload["max_size"]),
            top_k=int(payload["top_k"]),
        )
        bank._counter = int(payload.get("counter", 0))
        bank._embeddings = payload["embeddings"].float().cpu()
        bank._metadata = [PrototypeMetadata(**m) for m in payload["metadata"]]
        return bank


class GSKRModule(nn.Module):
    """PyTorch module wrapper around `SpectralKnowledgeBank` for forward-pass retrieval."""

    def __init__(self, bank: SpectralKnowledgeBank | None = None, embedding_dim: int = 128, top_k: int = 2) -> None:
        super().__init__()
        self.bank = bank or SpectralKnowledgeBank(embedding_dim=embedding_dim, top_k=top_k)

    def forward(
        self,
        spectral_tokens: torch.Tensor,
        task_type: str | None = None,
        top_k: int | None = None,
        avoid_exact_self: bool = True,
    ) -> GSKRRetrievalResult:
        return self.bank.retrieve(
            query_tokens=spectral_tokens,
            top_k=top_k,
            task_type=task_type,
            avoid_exact_self=avoid_exact_self,
        )
