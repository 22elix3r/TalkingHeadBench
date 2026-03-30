"""
Unit tests for Node 7: Weight Signal Extractor
(src/envs/subenv3/node7_weight_extractor.py)

All tests use the ``synthetic_lora_path`` fixture defined in tests/conftest.py —
no real model weights are needed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.schemas.subenv3 import WeightSignalObservation
from src.utils.canonical import canonicalize_lora_factors


# ``synthetic_lora_path`` is injected from tests/conftest.py


# ---------------------------------------------------------------------------
# Test 1 — return type and basic structural invariants
# ---------------------------------------------------------------------------


def test_returns_valid_observation(synthetic_lora_path):
    """extract_weight_signals must return a WeightSignalObservation with
    lora_rank == 8 and exactly 3 layer_norm entries (one per LoRA layer).
    """
    result = extract_weight_signals(synthetic_lora_path)

    assert isinstance(result, WeightSignalObservation)
    assert result.lora_rank == 8
    assert len(result.layer_norms) == 3


# ---------------------------------------------------------------------------
# Test 2 — canonicalize_lora_factors called once per layer
# ---------------------------------------------------------------------------


def test_canonical_called_per_layer(synthetic_lora_path):
    """canonicalize_lora_factors must be invoked exactly 3 times (one per layer)."""
    target = "src.envs.subenv3.node7_weight_extractor.canonicalize_lora_factors"
    with patch(target, wraps=canonicalize_lora_factors) as mock_canon:
        extract_weight_signals(synthetic_lora_path)
        assert mock_canon.call_count == 3


# ---------------------------------------------------------------------------
# Test 3 — token mapping is None when no tokenizer config supplied
# ---------------------------------------------------------------------------


def test_token_mapping_none_without_config(synthetic_lora_path):
    """Without a tokenizer config, token_position_to_phoneme must be None."""
    result = extract_weight_signals(synthetic_lora_path)
    assert result.token_position_to_phoneme is None


# ---------------------------------------------------------------------------
# Test 4 — token mapping loaded and cast from JSON config
# ---------------------------------------------------------------------------


def test_token_mapping_loaded_from_config(synthetic_lora_path, tmp_path):
    """A tokenizer config with 'token_position_to_phoneme' must produce
    a dict[int, str] with string keys parsed as integers.
    """
    config = tmp_path / "tokenizer.json"
    config.write_text(
        json.dumps({"token_position_to_phoneme": {"0": "AH", "1": "EE"}})
    )
    result = extract_weight_signals(
        synthetic_lora_path, tokenizer_config_path=config
    )
    assert result.token_position_to_phoneme == {0: "AH", 1: "EE"}


# ---------------------------------------------------------------------------
# Test 5 — rank utilization values in [0, 1]
# ---------------------------------------------------------------------------


def test_rank_utilization_in_range(synthetic_lora_path):
    """Every layer_rank_utilization value must be in [0.0, 1.0]."""
    result = extract_weight_signals(synthetic_lora_path)
    assert result.layer_rank_utilization, "layer_rank_utilization must be non-empty"
    assert all(
        0.0 <= v <= 1.0 for v in result.layer_rank_utilization.values()
    ), f"Out-of-range values: {result.layer_rank_utilization}"


# ---------------------------------------------------------------------------
# Test 6 — canonical entropy non-negative
# ---------------------------------------------------------------------------


def test_canonical_entropy_nonnegative(synthetic_lora_path):
    """Every canonical_entropy_per_layer value must be ≥ 0.0 (Shannon entropy ≥ 0)."""
    result = extract_weight_signals(synthetic_lora_path)
    assert result.canonical_entropy_per_layer, "canonical_entropy_per_layer must be non-empty"
    assert all(
        v >= 0.0 for v in result.canonical_entropy_per_layer.values()
    ), f"Negative entropy found: {result.canonical_entropy_per_layer}"


# ---------------------------------------------------------------------------
# Test 7 — overfitting signature in [0, 1]
# ---------------------------------------------------------------------------


def test_overfitting_signature_in_range(synthetic_lora_path):
    """overfitting_signature must be in [0.0, 1.0]."""
    result = extract_weight_signals(synthetic_lora_path)
    assert 0.0 <= result.overfitting_signature <= 1.0, (
        f"overfitting_signature={result.overfitting_signature} out of range"
    )


# ---------------------------------------------------------------------------
# Supplementary — structural sanity checks
# ---------------------------------------------------------------------------


def test_layer_keys_consistent(synthetic_lora_path):
    """layer_norms, layer_sparsity, layer_rank_utilization, and
    canonical_entropy_per_layer must all share the same key set.
    """
    result = extract_weight_signals(synthetic_lora_path)
    keys = set(result.layer_norms)
    assert set(result.layer_sparsity) == keys
    assert set(result.layer_rank_utilization) == keys
    assert set(result.canonical_entropy_per_layer) == keys


def test_target_modules_count(synthetic_lora_path):
    """target_modules must list exactly 3 module names."""
    result = extract_weight_signals(synthetic_lora_path)
    assert len(result.target_modules) == 3


def test_weight_file_id_matches_filename(synthetic_lora_path):
    """weight_file_id must equal the filename of the provided path."""
    result = extract_weight_signals(synthetic_lora_path)
    assert result.weight_file_id == synthetic_lora_path.name


def test_missing_file_raises(tmp_path):
    """extract_weight_signals must raise FileNotFoundError for nonexistent paths."""
    with pytest.raises(FileNotFoundError):
        extract_weight_signals(tmp_path / "nonexistent.safetensors")
