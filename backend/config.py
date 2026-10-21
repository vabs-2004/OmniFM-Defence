"""Validated configuration settings for OmniFM Blockchain Federated Learning.

Loads settings from environment variables and an optional `.env` file while
protecting sensitive fields (private keys) from accidental exposure in logs,
string representations, or serialized dictionaries.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
HEX_KEY_RE = re.compile(r"^(0x)?[a-fA-F0-9]{64}$")

PrecisionPolicy = Literal["fp32", "fp16_amp", "bf16_amp"]


class Settings(BaseModel):
    """Application, model, federated learning, and blockchain settings."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    # Blockchain settings
    ganache_url: str = Field(default="http://127.0.0.1:8545")
    ganache_chain_id: int = Field(default=1337)
    contract_address: str = Field(default="")
    blockchain_audit_enabled: bool = Field(default=False)
    server_private_key: SecretStr = Field(default=SecretStr(""))
    client1_private_key: SecretStr = Field(default=SecretStr(""))
    client2_private_key: SecretStr = Field(default=SecretStr(""))
    client3_private_key: SecretStr = Field(default=SecretStr(""))

    # Database and API settings
    database_url: str = Field(default="sqlite:///./omnifm.db")
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # OmniFM spectral / retrieval / prompt / alignment hyperparameters
    spalign_lambda: float = Field(default=0.1)
    gskr_top_k: int = Field(default=2)
    embedding_dim: int = Field(default=128)
    eca_num_heads: int = Field(default=4)
    spectral_low_pass_ratio: float = Field(default=0.25)
    max_knowledge_bank_size: int = Field(default=1000)

    # Paths
    artifact_dir: Path = Field(default=Path("./artifacts"))
    dataset_dir: Path = Field(default=Path("./datasets"))
    knowledge_bank_path: Path = Field(default=Path("./backend/fl/knowledge_bank.pt"))

    # Federated learning & precision policy
    num_clients: int = Field(default=3)
    default_num_rounds: int = Field(default=20)
    smoke_num_rounds: int = Field(default=5)
    default_local_epochs: int = Field(default=1)
    random_seed: int = Field(default=42)
    precision_policy: PrecisionPolicy = Field(default="fp16_amp")

    # Task batch sizes
    batch_size_classification: int = Field(default=32)
    batch_size_segmentation: int = Field(default=4)
    batch_size_superresolution: int = Field(default=4)
    batch_size_fusion: int = Field(default=2)

    # Task default spatial resolutions
    image_size_classification: int = Field(default=28)
    image_size_segmentation: int = Field(default=128)
    image_size_superresolution_lr: int = Field(default=64)
    image_size_fusion: int = Field(default=128)

    @field_validator("ganache_url")
    @classmethod
    def validate_ganache_url(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://") or v.startswith("ws://") or v.startswith("wss://")):
            raise ValueError(
                f"Invalid GANACHE_URL '{v}': must start with http://, https://, ws://, or wss://."
            )
        return v

    @field_validator("ganache_chain_id")
    @classmethod
    def validate_chain_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Invalid GANACHE_CHAIN_ID '{v}': must be a positive integer (e.g. 1337).")
        return v

    @field_validator("contract_address")
    @classmethod
    def validate_contract_address(cls, v: str) -> str:
        v = v.strip()
        if v and not ETH_ADDRESS_RE.match(v):
            raise ValueError(
                f"Invalid CONTRACT_ADDRESS '{v}': must be empty or a 42-character 0x-prefixed hex address."
            )
        return v

    @field_validator(
        "server_private_key",
        "client1_private_key",
        "client2_private_key",
        "client3_private_key",
        mode="before",
    )
    @classmethod
    def validate_private_key(cls, v: Any) -> SecretStr:
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v or "").strip()
        if raw and not HEX_KEY_RE.match(raw):
            raise ValueError(
                "Invalid private key format: must be empty or a 64-hex-character development key (optional 0x prefix)."
            )
        return SecretStr(raw)

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        v = v.strip()
        if not v or "://" not in v:
            raise ValueError(
                f"Invalid DATABASE_URL '{v}': must be a valid SQLAlchemy database URL (e.g. sqlite:///./omnifm.db)."
            )
        return v

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Invalid API_PORT '{v}': must be between 1 and 65535.")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            origins = [item.strip() for item in v.split(",") if item.strip()]
            if not origins:
                raise ValueError("CORS_ORIGINS cannot be empty when specified as a string.")
            return origins
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        raise ValueError("Invalid CORS_ORIGINS: must be a comma-separated string or list of strings.")

    @field_validator("spalign_lambda")
    @classmethod
    def validate_spalign_lambda(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Invalid SPALIGN_LAMBDA '{v}': must be >= 0.0.")
        return v

    @field_validator("gskr_top_k")
    @classmethod
    def validate_gskr_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Invalid GSKR_TOP_K '{v}': must be >= 1.")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        if v < 8:
            raise ValueError(f"Invalid EMBEDDING_DIM '{v}': must be >= 8.")
        return v

    @field_validator("eca_num_heads")
    @classmethod
    def validate_eca_num_heads(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Invalid ECA_NUM_HEADS '{v}': must be >= 1.")
        return v

    @field_validator("spectral_low_pass_ratio")
    @classmethod
    def validate_low_pass_ratio(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"Invalid SPECTRAL_LOW_PASS_RATIO '{v}': must be in the range (0.0, 1.0]."
            )
        return v

    @field_validator("max_knowledge_bank_size")
    @classmethod
    def validate_max_bank_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Invalid MAX_KNOWLEDGE_BANK_SIZE '{v}': must be >= 1.")
        return v

    @field_validator("num_clients", "default_num_rounds", "smoke_num_rounds", "default_local_epochs")
    @classmethod
    def validate_positive_fl_counts(cls, v: int, info: Any) -> int:
        if v < 1:
            raise ValueError(f"Invalid {info.field_name.upper()} '{v}': must be >= 1.")
        return v

    @field_validator(
        "batch_size_classification",
        "batch_size_segmentation",
        "batch_size_superresolution",
        "batch_size_fusion",
    )
    @classmethod
    def validate_batch_sizes(cls, v: int, info: Any) -> int:
        if v < 1:
            raise ValueError(f"Invalid {info.field_name.upper()} '{v}': batch size must be >= 1.")
        return v

    @field_validator(
        "image_size_classification",
        "image_size_segmentation",
        "image_size_superresolution_lr",
        "image_size_fusion",
    )
    @classmethod
    def validate_image_sizes(cls, v: int, info: Any) -> int:
        if v < 8:
            raise ValueError(f"Invalid {info.field_name.upper()} '{v}': image size must be >= 8.")
        return v

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "Settings":
        if self.embedding_dim % self.eca_num_heads != 0:
            raise ValueError(
                f"Invalid EMBEDDING_DIM ({self.embedding_dim}) and ECA_NUM_HEADS ({self.eca_num_heads}): "
                "EMBEDDING_DIM must be divisible by ECA_NUM_HEADS."
            )
        if self.gskr_top_k > self.max_knowledge_bank_size:
            raise ValueError(
                f"Invalid GSKR_TOP_K ({self.gskr_top_k}) > MAX_KNOWLEDGE_BANK_SIZE ({self.max_knowledge_bank_size})."
            )
        if self.blockchain_audit_enabled and not self.contract_address:
            raise ValueError(
                "BLOCKCHAIN_AUDIT_ENABLED is true, but CONTRACT_ADDRESS is empty. "
                "Deploy the contract and set CONTRACT_ADDRESS or disable blockchain_audit_enabled."
            )
        return self

    def safe_dict(self) -> dict[str, Any]:
        """Return a dictionary of settings safe for logging/persistence with secrets redacted."""
        data = self.model_dump()
        for key in (
            "server_private_key",
            "client1_private_key",
            "client2_private_key",
            "client3_private_key",
        ):
            secret_val: SecretStr = getattr(self, key)
            data[key] = "***REDACTED***" if secret_val.get_secret_value() else ""
        for path_key in ("artifact_dir", "dataset_dir", "knowledge_bank_path"):
            data[path_key] = str(data[path_key])
        return data


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_settings(
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load and validate settings from `.env`, environment variables, and explicit overrides."""
    raw_values: dict[str, Any] = {}

    if env_file is not None:
        env_path = Path(env_file)
        if not env_path.is_file() and env_file == ".env" and DEFAULT_ENV_FILE.is_file():
            env_path = DEFAULT_ENV_FILE
        if env_path.is_file():
            for k, v in dotenv_values(env_path).items():
                if v is not None and v != "":
                    raw_values[k.lower()] = v

    for field_name in Settings.model_fields:
        env_key = field_name.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            raw_values[field_name] = os.environ[env_key]

    if overrides:
        for k, v in overrides.items():
            raw_values[k.lower()] = v

    return Settings(**raw_values)


settings = load_settings()
