
"""
Deterministic SHA-256 artifact and PyTorch state_dict hashing.

Canonical serialization format:
    1. Magic header: b"OMNIFM_STATE_V1\\x00"
    2. Number of tensors: uint32 big-endian
    3. Tensors in sorted UTF-8 key order:
       a. Key length: uint32 big-endian
       b. UTF-8 key bytes
       c. Dtype string length: uint32 big-endian
       d. Canonical dtype string bytes
       e. Tensor rank: uint32 big-endian
       f. Shape dimensions: uint64 big-endian
       g. Raw tensor byte length: uint64 big-endian
       h. Contiguous, little-endian tensor bytes

The bfloat16 dtype is serialized through a uint16 view to preserve its
exact bit representation.

hash_state_dict() streams this format directly into SHA-256 rather than
building one enormous serialized copy in memory.

IMPORTANT:
The serialization format must remain stable for compatibility with hashes
stored in checkpoints, databases, and blockchain audit records.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import torch


MAGIC_HEADER = b"OMNIFM_STATE_V1\x00"
DEFAULT_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def _validate_tensor(tensor: torch.Tensor, key: str = "") -> None:
    """Validate that a state_dict value is a supported dense tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor in state_dict"
            f"{f'[{key!r}]' if key else ''}, "
            f"got {type(tensor).__name__}."
        )

    if tensor.layout != torch.strided:
        raise TypeError(
            f"Only dense strided tensors are supported for hashing; "
            f"got layout {tensor.layout} for {key!r}."
        )


def _canonical_numpy_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a CPU tensor to a NumPy array with canonical little-endian
    representation, preserving the underlying tensor values and dtype.
    """
    if tensor.device.type != "cpu":
        raise ValueError("_canonical_numpy_array expects a CPU tensor.")

    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16).numpy()
    else:
        arr = tensor.numpy()

    # NumPy's byteorder may be '=', '<', '>', or '|'.
    # Normalize multi-byte data to little-endian.
    is_big_endian = (
        arr.dtype.byteorder == ">"
        or (
            arr.dtype.byteorder == "="
            and struct.pack("=H", 1) == b"\x00\x01"
        )
    )

    if is_big_endian and arr.dtype.itemsize > 1:
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))

    return arr


def _iter_tensor_canonical_byte_chunks(
    tensor: torch.Tensor,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[bytes]:
    """
    Yield a tensor's canonical raw bytes in bounded-size chunks.

    Contiguous tensors are processed in slices. GPU tensors are transferred
    to CPU one slice at a time, avoiding a full GPU-to-CPU copy.

    Non-contiguous tensors are first made contiguous to preserve the same
    logical row-major element order as the original serializer.
    """
    _validate_tensor(tensor)

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")

    source = tensor.detach()

    if not source.is_contiguous():
        source = source.contiguous()

    flat = source.view(-1)

    element_size = source.element_size()
    elements_per_chunk = max(1, chunk_bytes // element_size)

    for start in range(0, flat.numel(), elements_per_chunk):
        end = min(start + elements_per_chunk, flat.numel())

        # Only the current slice is transferred to CPU.
        cpu_chunk = flat[start:end].to(device="cpu")

        arr = _canonical_numpy_array(cpu_chunk)

        # Each temporary byte string is bounded approximately by chunk_bytes.
        yield arr.tobytes(order="C")


def _tensor_to_canonical_bytes(
    tensor: torch.Tensor,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> bytes:
    """
    Return a tensor's canonical raw bytes.

    Compatibility helper. This function still materializes the complete
    tensor byte string; use the streaming hash API for large tensors.
    """
    return b"".join(
        _iter_tensor_canonical_byte_chunks(
            tensor,
            chunk_bytes=chunk_bytes,
        )
    )


def _iter_state_dict_canonical_chunks(
    state_dict: Mapping[str, torch.Tensor],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[bytes]:
    """
    Yield the canonical serialized state_dict as byte chunks.

    The emitted byte sequence matches the original canonical serialization
    format. Tensor data is streamed rather than accumulated in a list.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "state_dict must be a mapping of parameter names to torch.Tensor."
        )

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")

    # Header and number of tensors.
    yield MAGIC_HEADER
    yield struct.pack(">I", len(state_dict))

    # Deterministic ordering independent of dictionary insertion order.
    for key in sorted(state_dict.keys()):
        if not isinstance(key, str):
            raise TypeError(
                f"state_dict key must be str, got {type(key).__name__}."
            )

        tensor = state_dict[key]
        _validate_tensor(tensor, key)

        key_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("utf-8")
        shape = tuple(int(dim) for dim in tensor.shape)

        # This is the number of raw tensor bytes without materializing them.
        raw_byte_length = tensor.numel() * tensor.element_size()

        # Key framing.
        yield struct.pack(">I", len(key_bytes))
        yield key_bytes

        # Dtype framing.
        yield struct.pack(">I", len(dtype_bytes))
        yield dtype_bytes

        # Rank and dimensions.
        yield struct.pack(">I", len(shape))

        if shape:
            yield struct.pack(f">{len(shape)}Q", *shape)

        # Raw tensor byte length.
        yield struct.pack(">Q", raw_byte_length)

        # Stream tensor bytes in bounded-size chunks.
        yield from _iter_tensor_canonical_byte_chunks(
            tensor,
            chunk_bytes=chunk_bytes,
        )


def serialize_state_dict_canonical(
    state_dict: Mapping[str, torch.Tensor],
) -> bytes:
    """
    Serialize a state_dict into the canonical byte sequence.

    Note: this public compatibility API necessarily materializes the complete
    serialized result. Prefer hash_state_dict() for large model states.
    """
    return b"".join(_iter_state_dict_canonical_chunks(state_dict))


def hash_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> str:
    """
    Compute the canonical 0x-prefixed SHA-256 digest of a state_dict.

    Hashes the exact canonical serialization incrementally, avoiding the
    full serialized-state allocation used by the original implementation.
    """
    hasher = hashlib.sha256()

    for chunk in _iter_state_dict_canonical_chunks(
        state_dict,
        chunk_bytes=chunk_bytes,
    ):
        hasher.update(chunk)

    return "0x" + hasher.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Compute the 0x-prefixed SHA-256 digest of raw bytes."""
    return "0x" + hashlib.sha256(data).hexdigest()


def hash_file(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> str:
    """Compute the 0x-prefixed SHA-256 digest of a file incrementally."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact file not found for hashing: {path}"
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)

            if not block:
                break

            hasher.update(block)

    return "0x" + hasher.hexdigest()


def normalize_bytes32_hex(hash_str: str | bytes) -> str:
    """
    Validate and normalize a SHA-256 digest to 0x-prefixed lowercase hex.

    Accepts a 32-byte value or a 64-character hexadecimal string, optionally
    prefixed with 0x. The all-zero digest is rejected as in the original API.
    """
    if isinstance(hash_str, (bytes, bytearray)):
        if len(hash_str) != 32:
            raise ValueError(
                f"Expected 32-byte hash, got {len(hash_str)} bytes."
            )

        cleaned = bytes(hash_str).hex()

    elif isinstance(hash_str, str):
        cleaned = hash_str.strip().lower()

        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]

    else:
        raise TypeError(
            "Hash must be a hexadecimal string or a 32-byte value."
        )

    if len(cleaned) != 64 or any(
        char not in "0123456789abcdef" for char in cleaned
    ):
        raise ValueError(
            "Invalid SHA-256 hash: expected exactly 64 hexadecimal characters."
        )

    if cleaned == "0" * 64:
        raise ValueError(
            "Zero hash (0x00..00) is not a valid artifact hash."
        )

    return "0x" + cleaned


def to_bytes32(hash_str: str | bytes) -> bytes:
    """Convert a valid SHA-256 digest to 32 raw bytes for Solidity bytes32."""
    normalized = normalize_bytes32_hex(hash_str)
    return bytes.fromhex(normalized[2:])


def verify_state_dict_hash(
    state_dict: Mapping[str, torch.Tensor],
    expected_hash: str | bytes,
) -> bool:
    """Verify a state_dict against its expected canonical SHA-256 digest."""
    computed = hash_state_dict(state_dict)
    expected = normalize_bytes32_hex(expected_hash)

    return computed == expected


def verify_file_hash(
    file_path: str | Path,
    expected_hash: str | bytes,
) -> bool:
    """Verify a file against its expected SHA-256 digest."""
    computed = hash_file(file_path)
    expected = normalize_bytes32_hex(expected_hash)

    return computed == expected
