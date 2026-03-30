"""
Deep smoke tests for Node 7 (Weight Signal Extractor).

Verifies the mathematical invariants of the extracted canonical weight signals
using synthetic LoRA files generated via torch and safetensors.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import safetensors.torch

from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.utils.canonical import CanonicalComponents


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_layer_norms_positive(synthetic_lora_path):
    """All Frobenius norms of canonical updates must be strictly positive for random weights."""
    result = extract_weight_signals(synthetic_lora_path)
    assert len(result.layer_norms) == 3
    assert all(v > 0.0 for v in result.layer_norms.values())


def test_layer_sparsity_in_range(synthetic_lora_path):
    """Sparsity fraction must be in [0.0, 1.0]."""
    result = extract_weight_signals(synthetic_lora_path)
    assert all(0.0 <= v <= 1.0 for v in result.layer_sparsity.values())


def test_histogram_sums_to_one(synthetic_lora_path):
    """The weight magnitude histogram must be a valid probability distribution."""
    result = extract_weight_signals(synthetic_lora_path)
    assert sum(result.weight_magnitude_histogram) == pytest.approx(1.0, abs=1e-5)


def test_layer_correlation_matrix_is_square(synthetic_lora_path):
    """n_layers x n_layers correlation matrix."""
    result = extract_weight_signals(synthetic_lora_path)
    n = len(result.layer_norms)
    assert len(result.layer_correlation_matrix) == n
    assert all(len(row) == n for row in result.layer_correlation_matrix)


def test_layer_correlation_diagonal_is_one(synthetic_lora_path):
    """Pearson correlation with self must be 1.0."""
    result = extract_weight_signals(synthetic_lora_path)
    for i, row in enumerate(result.layer_correlation_matrix):
        # Pearson correlation can be slightly off due to float precision
        assert abs(row[i] - 1.0) < 1e-4


def test_dominant_directions_leq_rank(synthetic_lora_path):
    """Dominant direction count cannot exceed the theoretical max (rank * layers)."""
    result = extract_weight_signals(synthetic_lora_path)
    # The extractor currently uses a mean, but we check against the global upper bound per prompt.
    assert result.canonical_dominant_directions <= result.lora_rank * len(result.layer_norms)


def test_overfitting_signature_reflects_rank_utilization(tmp_path):
    """Collapsed rank + high norms must result in a high overfitting signature (> 0.5)."""
    # A is (in, rank) = (32, 8). We collapse it to rank 1.
    A = torch.zeros(32, 8)
    A[:, 0] = 1.0
    # B is (out, rank) = (64, 8). Inflate update norm to trigger signature.
    B = torch.randn(64, 8) * 50.0
    
    # Create 2 layers to satisfy the extractor's correlation matrix logic (n > 1)
    # Use .clone() to avoid safetensors RuntimeError regarding shared memory.
    tensors = {
        "layer_0.lora_A.weight": A.T.contiguous().clone(),
        "layer_0.lora_B.weight": B.contiguous().clone(),
        "layer_1.lora_A.weight": A.T.contiguous().clone(),
        "layer_1.lora_B.weight": B.contiguous().clone()
    }
    path = tmp_path / "collapsed_lora.safetensors"
    safetensors.torch.save_file(tensors, str(path))
    
    result = extract_weight_signals(path)
    # rank_util = 1/8 = 0.125. (1 - 0.125) = 0.875.
    # norm will be high (> 10), so norm_factor = 1.0.
    # signature should be ~0.875.
    assert result.overfitting_signature > 0.5


def test_gradient_noise_nonnegative(synthetic_lora_path):
    """Gradient noise estimate (min/max S ratio) must be non-negative."""
    result = extract_weight_signals(synthetic_lora_path)
    assert result.gradient_noise_estimate >= 0.0


def test_target_modules_matches_layer_count(synthetic_lora_path):
    """Verify target_modules list matches the number of LoRA pairs in the file."""
    result = extract_weight_signals(synthetic_lora_path)
    assert len(result.target_modules) == 3


def test_canonical_not_called_on_nonlora_keys(tmp_path):
    """Non-2D or non-LoRA keys in safetensors must be ignored by the extractor."""
    A = torch.randn(8, 32)
    B = torch.randn(64, 8)
    # 2 layers + a non-LoRA key. Use .clone() to avoid shared memory issues.
    tensors = {
        "layer_0.lora_A.weight": A.clone(),
        "layer_0.lora_B.weight": B.clone(),
        "layer_1.lora_A.weight": A.clone(),
        "layer_1.lora_B.weight": B.clone(),
        "norm.weight": torch.randn(64)  # non-2D tensor (vector)
    }
    path = tmp_path / "mixed_keys.safetensors"
    safetensors.torch.save_file(tensors, str(path))
    
    # Patch the function imported inside the extractor module
    target = "src.envs.subenv3.node7_weight_extractor.canonicalize_lora_factors"
    with patch(target) as mock_canon:
        # Provide a dummy return value
        mock_canon.return_value = CanonicalComponents(
            U=torch.randn(64, 8),
            S=torch.randn(8),
            Vt=torch.randn(8, 32),
            Q=torch.randn(32, 8)
        )
        
        extract_weight_signals(path)
        
        # Should be called once per layer (2 total), not for norm.weight
        assert mock_canon.call_count == 2
