"""Unit tests for Task 1.6: deterministic state_dict and artifact hashing."""

from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from backend.utils.hashing import (
    deserialize_state_dict_canonical,
    hash_file,
    hash_state_dict,
    serialize_state_dict_canonical,
    verify_file_hash,
    verify_state_dict_hash,
)


class TestDeterministicHashing(unittest.TestCase):
    def test_identical_state_dict_produces_identical_hash_regardless_of_key_order(self) -> None:
        w = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.float32)
        b = torch.tensor([0.5, -0.5], dtype=torch.float32)

        state_a = OrderedDict([("layer.weight", w.clone()), ("layer.bias", b.clone())])
        state_b = OrderedDict([("layer.bias", b.clone()), ("layer.weight", w.clone())])

        hash_a = hash_state_dict(state_a)
        hash_b = hash_state_dict(state_b)

        self.assertEqual(hash_a, hash_b)
        self.assertTrue(hash_a.startswith("0x"))
        self.assertEqual(len(hash_a), 66)
        self.assertTrue(verify_state_dict_hash(state_a, hash_b))

    def test_changed_bytes_dtype_shape_or_key_changes_hash(self) -> None:
        base_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        base_state = {"param": base_tensor.clone()}
        base_hash = hash_state_dict(base_state)

        # 1. Changed tensor value
        changed_val = {"param": torch.tensor([1.0, 2.0, 3.0, 4.0001], dtype=torch.float32)}
        self.assertNotEqual(base_hash, hash_state_dict(changed_val))

        # 2. Changed tensor shape with identical raw bytes (1x4 vs 2x2)
        changed_shape = {"param": base_tensor.clone().reshape(2, 2)}
        self.assertNotEqual(base_hash, hash_state_dict(changed_shape))

        # 3. Changed tensor dtype (float32 vs float64 or int32 view)
        changed_dtype = {"param": base_tensor.clone().to(torch.float64)}
        self.assertNotEqual(base_hash, hash_state_dict(changed_dtype))

        reinterpreted_dtype = {"param": base_tensor.clone().view(torch.int32)}
        self.assertNotEqual(base_hash, hash_state_dict(reinterpreted_dtype))

        # 4. Changed parameter key name
        changed_key = {"param_renamed": base_tensor.clone()}
        self.assertNotEqual(base_hash, hash_state_dict(changed_key))

    def test_file_hashing_and_verification(self) -> None:
        state = {"encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            torch.save(state, ckpt_path)
            digest = hash_file(ckpt_path)
            self.assertTrue(verify_file_hash(ckpt_path, digest))
            self.assertFalse(verify_file_hash(ckpt_path, "0x" + "1" * 64))

    def test_canonical_serialization_and_deserialization_roundtrip_and_guards(self) -> None:
        orig_state = {
            "conv.weight": torch.randn(4, 3, 3, 3, dtype=torch.float32),
            "fp16.bias": torch.tensor([0.25, -0.75, 1.5], dtype=torch.float16),
            "bf16.scale": torch.tensor([2.0, -4.0], dtype=torch.bfloat16),
            "bn.num_batches_tracked": torch.tensor(12, dtype=torch.int64),
        }
        expected_hash = hash_state_dict(orig_state)
        raw_bytes = serialize_state_dict_canonical(orig_state)

        restored = deserialize_state_dict_canonical(raw_bytes)
        self.assertEqual(set(restored.keys()), set(orig_state.keys()))
        self.assertEqual(hash_state_dict(restored), expected_hash)
        for k, t in orig_state.items():
            self.assertEqual(restored[k].dtype, t.dtype)
            self.assertEqual(tuple(restored[k].shape), tuple(t.shape))
            self.assertEqual(restored[k].device.type, "cpu")
            self.assertTrue(torch.equal(restored[k], t.cpu()))

        # Invalid magic header rejected
        with self.assertRaises(ValueError):
            deserialize_state_dict_canonical(b"BAD_HEADER_V1\x00\x00\x00" + raw_bytes[16:])

        # Truncated payload rejected
        with self.assertRaises(ValueError):
            deserialize_state_dict_canonical(raw_bytes[:-5])

        # Trailing garbage rejected
        with self.assertRaises(ValueError):
            deserialize_state_dict_canonical(raw_bytes + b"\x00")

        # Max byte limit enforced
        with self.assertRaises(ValueError):
            deserialize_state_dict_canonical(raw_bytes, max_bytes=32)


if __name__ == "__main__":
    unittest.main()
