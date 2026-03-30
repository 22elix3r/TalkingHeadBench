"""
Shared pytest fixtures for Sub-env 3 tests.

Exposes ``synthetic_lora_path`` so it can be used by both
``test_node7_extractor.py`` and ``test_subenv3.py`` without duplication.
"""

from __future__ import annotations

import pytest
import torch
import safetensors.torch


@pytest.fixture
def synthetic_lora_path(tmp_path):
    """Write a minimal 3-layer LoRA ``.safetensors`` file and return its path.

    Layout (PEFT-style keys):
      layer_{i}.lora_A.weight  shape (8, 32)   → treated as A^T → A: (32, 8)
      layer_{i}.lora_B.weight  shape (64, 32)

    So:  rank = 8,  in_features = 32,  out_features = 64,  layers = 3.
    """
    torch.manual_seed(0)
    tensors = {}
    for i in range(3):
        # PEFT convention:
        #   lora_A.weight shape = (rank, in_features) = (8, 32)
        #   lora_B.weight shape = (out_features, rank) = (64, 8)
        # _find_lora_pairs sees A.shape[0] < A.shape[1] → transposes to (32, 8)
        # Then QR((32,8)) → Q:(32,8), R:(8,8); B:(64,8) @ R_a:(8,8) ✓
        A_peft = torch.randn(8, 32)    # (rank, in_features)
        B_peft = torch.randn(64, 8)    # (out_features, rank)
        tensors[f"layer_{i}.lora_A.weight"] = A_peft.contiguous()
        tensors[f"layer_{i}.lora_B.weight"] = B_peft.contiguous()
    path = tmp_path / "test_lora.safetensors"
    safetensors.torch.save_file(tensors, str(path))
    return path
