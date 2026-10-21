"""OmniFM Spectral Conditioning Package (`backend/omnifm`)."""

from backend.omnifm.composer import OmniFMConditionedOutput, OmniFMSpatialConditioner
from backend.omnifm.eca import EmbeddingCrossAttention
from backend.omnifm.gskr import (
    GSKRModule,
    GSKRRetrievalResult,
    PrototypeMetadata,
    SpectralKnowledgeBank,
)
from backend.omnifm.psp import (
    PERSONALIZED_PROMPT_PARAM_NAME,
    PSPOutput,
    PrefixSuffixSpectralPrompting,
)
from backend.omnifm.spalign import SPAlignLoss, SPAlignLossBreakdown
from backend.omnifm.spectral import SpectralEncoder

__all__ = [
    "EmbeddingCrossAttention",
    "GSKRModule",
    "GSKRRetrievalResult",
    "OmniFMConditionedOutput",
    "OmniFMSpatialConditioner",
    "PERSONALIZED_PROMPT_PARAM_NAME",
    "PSPOutput",
    "PrefixSuffixSpectralPrompting",
    "PrototypeMetadata",
    "SPAlignLoss",
    "SPAlignLossBreakdown",
    "SpectralEncoder",
    "SpectralKnowledgeBank",
]
