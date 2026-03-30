"""
Node 7: Weight Signal Extractor — Sub-env 3.

Extracts canonical weight statistics from a trained LoRA ``.safetensors`` file.

**All statistics are computed from canonical SVD components (U, S, Vt, Q)**
produced by ``canonicalize_lora_factors()`` imported from
``src/utils/canonical.py``.  Raw A/B matrices are never used for statistics
directly — they are fed through the W2T QR → SVD pipeline first, resolving
column-space factorization ambiguity.

``token_position_to_phoneme`` is loaded from the tokenizer config JSON shipped
alongside the ``.safetensors`` file.  It is NOT derived from the weights.  If
``tokenizer_config_path`` is ``None``, the field is set to ``None``.

Statistic formulas:
  - Layer norms:         ``torch.linalg.norm(canonical_effective_update)``
    - Layer entropy:       inverted normalized entropy of canonical singular values
  - Rank utilization:    from canonical SVD S directly (no extra computation)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file

from src.schemas.subenv2 import SyntheticWeightDescriptor
from src.schemas.subenv3 import WeightSignalObservation
from src.utils.canonical import (
    CanonicalComponents,
    canonicalize_lora_factors,
    layer_entropy_from_singular_values,
    singular_direction_anomaly_scores,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of a layer's max singular value below which an S entry is "near zero".
_SPARSITY_THRESHOLD_RATIO: float = 0.01

# Cumulative energy fraction for counting "dominant" singular directions.
_DOMINANT_ENERGY_FRACTION: float = 0.90

# Standard-deviation multipliers for flagging anomalous Vt-row entropy.
_ENTROPY_ANOMALY_SIGMA: float = 2.0

# Number of histogram bins for the canonical S distribution.
_HISTOGRAM_BINS: int = 20


# ---------------------------------------------------------------------------
# Private helpers — safetensors parsing
# ---------------------------------------------------------------------------


def _find_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Parse a LoRA state-dict into ``{layer_name: (A, B)}`` pairs.

    Supports two common key conventions:
      - PEFT-style:  ``...lora_A...weight`` / ``...lora_B...weight``
      - Kohya-style: ``...lora_down.weight`` / ``...lora_up.weight``

    ``A`` is the "down" (input) projection; ``B`` is the "up" (output)
    projection, matching the LoRA convention ``delta_W = B @ A``.

    Keys that are not 2-D tensors are silently skipped (e.g. alpha scalars).
    """
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}

    for key in state_dict:
        if state_dict[key].ndim != 2:
            continue
        lo = key.lower()
        if "lora_a" in lo or "lora_down" in lo:
            # Derive canonical layer name by stripping the LoRA suffix
            base = (
                key.replace(".lora_A.weight", "")
                   .replace(".lora_A.default.weight", "")
                   .replace(".lora_down.weight", "")
            )
            a_keys[base] = key
        elif "lora_b" in lo or "lora_up" in lo:
            base = (
                key.replace(".lora_B.weight", "")
                   .replace(".lora_B.default.weight", "")
                   .replace(".lora_up.weight", "")
            )
            b_keys[base] = key

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base in a_keys:
        if base in b_keys:
            A = state_dict[a_keys[base]]   # shape: (rank, in_features) → transpose → (in_features, rank)
            B = state_dict[b_keys[base]]   # shape: (out_features, rank)
            # PEFT stores A as (rank, in),  B as (out, rank).
            # canonicalize_lora_factors expects A: (in, rank), B: (out, rank).
            if A.shape[0] < A.shape[1]:
                A = A.T
            pairs[base] = (A, B)

    return pairs


# ---------------------------------------------------------------------------
# Private helpers — per-layer statistics (all from canonical components)
# ---------------------------------------------------------------------------


def _layer_norm_from_canonical(cc: CanonicalComponents) -> float:
    """Frobenius norm of the canonical effective update.

    Equivalent to ``torch.linalg.norm(U @ diag(S) @ Vt)``, which simplifies
    to ``torch.linalg.norm(S)`` since U and Vt are orthonormal.
    """
    return float(torch.linalg.norm(cc.S))


def _layer_sparsity_from_canonical(cc: CanonicalComponents) -> float:
    """Fraction of canonical singular values near zero.

    A value is «near zero» when it is below
    ``_SPARSITY_THRESHOLD_RATIO × max(S)``.
    """
    s = cc.S
    if s.numel() == 0:
        return 1.0
    threshold = _SPARSITY_THRESHOLD_RATIO * float(s.max())
    near_zero = int((s < threshold).sum())
    return near_zero / s.numel()


def _layer_rank_utilization_from_canonical(cc: CanonicalComponents) -> float:
    """Effective rank / nominal rank from canonical SVD S.

    Effective rank = number of singular values above the sparsity threshold.
    Nominal rank = total number of singular values (= LoRA rank).
    """
    s = cc.S
    if s.numel() == 0:
        return 0.0
    threshold = _SPARSITY_THRESHOLD_RATIO * float(s.max())
    effective_rank = int((s >= threshold).sum())
    return effective_rank / s.numel()


def _layer_entropy_from_canonical(cc: CanonicalComponents) -> float:
    """Inverted normalized entropy of canonical singular values.

    Returns a value in [0, 1] where:
      - 0.0 indicates uniform singular values (healthy utilization),
      - 1.0 indicates concentrated energy (rank collapse tendency).
    """
    return layer_entropy_from_singular_values(cc.S)


def _per_row_entropies(cc: CanonicalComponents) -> list[float]:
    """Per-direction anomaly scores from normalized singular values.

    Values are in [0, 1]; higher values indicate singular directions that
    dominate layer energy.
    """
    return singular_direction_anomaly_scores(cc.S)


def _u_column_norm_variance(cc: CanonicalComponents) -> float:
    """Variance of U column norms across the canonical left singular vectors."""
    u = cc.U  # (out_features, rank)
    if u.numel() == 0:
        return 0.0
    col_norms = torch.linalg.norm(u, dim=0)   # (rank,)
    return float(col_norms.var())


def _dominant_directions(cc: CanonicalComponents, fraction: float = _DOMINANT_ENERGY_FRACTION) -> int:
    """Number of canonical singular values that capture ``fraction`` of total energy."""
    s = cc.S
    if s.numel() == 0:
        return 0
    energy = s ** 2
    total = float(energy.sum())
    if total < 1e-12:
        return 0
    cumsum = torch.cumsum(energy, dim=0)
    k = int((cumsum < fraction * total).sum()) + 1
    return min(k, s.numel())


# ---------------------------------------------------------------------------
# Private helpers — cross-layer statistics
# ---------------------------------------------------------------------------


def _layer_update_vector(cc: CanonicalComponents) -> np.ndarray:
    """Flatten the canonical effective update to a 1-D vector for correlation."""
    # Effective update = U @ diag(S) @ Vt; we use S as a compact representation.
    return cc.S.detach().cpu().float().numpy()


def _correlation_matrix(vectors: list[np.ndarray]) -> list[list[float]]:
    """Pairwise Pearson correlation matrix of layer canonical S vectors.

    Vectors are zero-padded or truncated to the shortest common length before
    correlation is computed, since different layers may have different ranks.
    """
    if not vectors:
        return []
    min_len = min(len(v) for v in vectors)
    mat = np.stack([v[:min_len] for v in vectors], axis=0)  # (n_layers, min_len)
    try:
        corr = np.corrcoef(mat)                             # (n_layers, n_layers)
    except Exception:
        n = len(vectors)
        corr = np.eye(n)
    return [[float(corr[i, j]) for j in range(corr.shape[1])] for i in range(corr.shape[0])]


def _attention_head_specialization(
    layer_pairs: dict[str, tuple[CanonicalComponents, int]],
) -> dict[str, float]:
    """Per-layer attention head specialization score.

    For each attention-related layer, measures how much the canonical energy
    is concentrated in a subset of singular directions relative to the number
    of heads.  A score near 1.0 indicates one head dominates; near 0.0
    indicates uniform distribution across heads.

    Args:
        layer_pairs: ``{layer_name: (CanonicalComponents, n_heads)}`` where
            ``n_heads`` is inferred from the layer name (see implementation).

    Returns:
        ``{layer_name: specialization_score}`` for attention layers only.
    """
    result: dict[str, float] = {}
    for name, (cc, n_heads) in layer_pairs.items():
        if n_heads < 2:
            continue
        s = cc.S.detach().cpu().float().numpy()
        if len(s) == 0:
            continue
        energy = s ** 2
        total = energy.sum()
        if total < 1e-12:
            continue
        # Fraction of energy in the top (rank // n_heads) directions
        head_size = max(1, len(s) // n_heads)
        top_energy = energy[:head_size].sum()
        result[name] = float(top_energy / total)
    return result


def _weight_magnitude_histogram(all_s_values: list[float], n_bins: int = _HISTOGRAM_BINS) -> list[float]:
    """Binned histogram of all canonical singular values (S) across layers."""
    if not all_s_values:
        return [0.0] * n_bins
    arr = np.array(all_s_values, dtype=np.float32)
    counts, _ = np.histogram(arr, bins=n_bins)
    total = counts.sum()
    return [float(c / total) if total > 0 else 0.0 for c in counts]


def _gradient_noise_estimate(layer_canonical: dict[str, CanonicalComponents]) -> float:
    """Estimate gradient noise from the flatness of the canonical S spectrum.

    A noisy training signal causes a more uniform singular value spectrum (high
    ratio of minimum to maximum S).  Returns a value in [0.0, 1.0] where 1.0
    is maximum estimated noise.
    """
    ratios: list[float] = []
    for cc in layer_canonical.values():
        s = cc.S.detach().cpu().float().numpy()
        if len(s) < 2 or s[0] < 1e-8:
            continue
        ratios.append(float(s[-1] / s[0]))   # min / max  (S is descending from SVD)
    return float(np.mean(ratios)) if ratios else 0.0


def _overfitting_signature(
    rank_utilizations: dict[str, float],
    layer_norms: dict[str, float],
) -> float:
    """Overfitting signature: low rank utilization combined with high norms.

    Returns a value in [0.0, 1.0] where 1.0 indicates a strong overfitting
    pattern (collapsed canonical directions + inflated update norms).
    """
    if not rank_utilizations:
        return 0.0
    mean_util = float(np.mean(list(rank_utilizations.values())))
    mean_norm = float(np.mean(list(layer_norms.values()))) if layer_norms else 0.0
    # Low utilization (near 0) and high norm (uncapped) → high overfitting score.
    # Norm is normalised with a soft cap at 10.0.
    norm_factor = float(np.clip(mean_norm / 10.0, 0.0, 1.0))
    return float(np.clip((1.0 - mean_util) * norm_factor, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _infer_n_heads(layer_name: str) -> int:
    """Heuristically infer the number of attention heads from a layer name."""
    # Common naming patterns: attn.q_proj, self_attn.k_proj, attention.query
    _ATTN_KEYWORDS = ("q_proj", "k_proj", "v_proj", "query", "key", "value", "attn")
    lo = layer_name.lower()
    if not any(kw in lo for kw in _ATTN_KEYWORDS):
        return 0  # not an attention layer — signal to skip head specialization
    # Return a reasonable default; could be overridden via config in production.
    return 8


def extract_weight_signals(
    weight_path: Path,
    tokenizer_config_path: Optional[Path] = None,
    dataset_health_summary: Optional[SyntheticWeightDescriptor] = None,
    suspected_anomalous_phonemes: Optional[list[str]] = None,
) -> WeightSignalObservation:
    """Extract canonical weight statistics from a LoRA ``.safetensors`` file.

    ``canonicalize_lora_factors()`` is called for every LoRA (A, B) pair
    before any statistics are computed.  All ``canonical_*`` fields in the
    returned observation are derived exclusively from the canonical SVD
    components (U, S, Vt, Q), never from raw A/B matrices.

    Args:
        weight_path: Path to the ``.safetensors`` LoRA weight file.
        tokenizer_config_path: Path to the audio tokenizer config JSON that
            ships alongside the ``.safetensors`` file.  Must contain a
            ``"token_position_to_phoneme"`` key mapping int positions to
            phoneme strings.  If ``None``, the field is set to ``None``.
        dataset_health_summary: Optional ``SyntheticWeightDescriptor`` from
            Sub-env 2 Node 6, forwarded as prior context.
        suspected_anomalous_phonemes: Optional list of phonemes flagged by
            Sub-env 2, forwarded as prior context.

    Returns:
        A fully populated :class:`WeightSignalObservation`.

    Raises:
        FileNotFoundError: If ``weight_path`` does not exist.
        ValueError: If no valid LoRA (A, B) pairs are found in the file.
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    weight_file_id = weight_path.name

    # ------------------------------------------------------------------
    # Load state dict and parse LoRA pairs
    # ------------------------------------------------------------------
    state_dict: dict[str, torch.Tensor] = load_file(str(weight_path))

    lora_pairs = _find_lora_pairs(state_dict)
    if not lora_pairs:
        raise ValueError(
            f"No valid LoRA (A, B) pairs found in '{weight_path}'. "
            "Check that the file contains lora_A/lora_B or lora_down/lora_up keys."
        )

    target_modules = sorted(lora_pairs.keys())

    # Infer rank from first pair
    first_A, _ = next(iter(lora_pairs.values()))
    lora_rank = first_A.shape[1]  # A is (in_features, rank)

    total_parameters = sum(
        A.numel() + B.numel() for A, B in lora_pairs.values()
    )

    # ------------------------------------------------------------------
    # Canonicalize every (A, B) pair — all subsequent stats use cc only
    # ------------------------------------------------------------------
    layer_canonical: dict[str, CanonicalComponents] = {}
    for name, (A, B) in lora_pairs.items():
        layer_canonical[name] = canonicalize_lora_factors(A.float(), B.float())

    # ------------------------------------------------------------------
    # Layer-wise statistics (all from canonical components)
    # ------------------------------------------------------------------
    layer_norms: dict[str, float] = {}
    layer_sparsity: dict[str, float] = {}
    layer_rank_utilization: dict[str, float] = {}
    canonical_entropy_per_layer: dict[str, float] = {}

    # Collect all S values for histogram and per-direction anomaly scores
    all_s_values: list[float] = []
    all_row_entropies: list[tuple[int, float]] = []  # (position_idx, anomaly_score)

    for name, cc in layer_canonical.items():
        layer_norms[name] = _layer_norm_from_canonical(cc)
        layer_sparsity[name] = _layer_sparsity_from_canonical(cc)
        layer_rank_utilization[name] = _layer_rank_utilization_from_canonical(cc)
        canonical_entropy_per_layer[name] = _layer_entropy_from_canonical(cc)

        all_s_values.extend(cc.S.detach().cpu().float().tolist())

        for row_idx, h in enumerate(_per_row_entropies(cc)):
            all_row_entropies.append((row_idx, h))

    # ------------------------------------------------------------------
    # High-entropy token positions (anomalous singular directions across layers)
    # ------------------------------------------------------------------
    if all_row_entropies:
        entropies_only = np.array([e for _, e in all_row_entropies], dtype=np.float32)
        mu = float(entropies_only.mean())
        sigma = float(entropies_only.std())
        threshold = max(mu + _ENTROPY_ANOMALY_SIGMA * sigma, 0.1)
        high_entropy_token_positions: list[int] = sorted(
            set(
                pos
                for pos, h in all_row_entropies
                if h > threshold
            )
        )
    else:
        high_entropy_token_positions = []

    # ------------------------------------------------------------------
    # Token-to-phoneme mapping (from tokenizer config, NOT from weights)
    # ------------------------------------------------------------------
    token_position_to_phoneme: Optional[dict[int, str]] = None
    if tokenizer_config_path is not None:
        tokenizer_config_path = Path(tokenizer_config_path)
        with tokenizer_config_path.open("r", encoding="utf-8") as fh:
            tok_cfg = json.load(fh)
        raw_map: dict = tok_cfg.get("token_position_to_phoneme", {})
        token_position_to_phoneme = {int(k): str(v) for k, v in raw_map.items()}

    # ------------------------------------------------------------------
    # Canonical U-component statistics (aggregated across all layers)
    # ------------------------------------------------------------------
    u_norm_variances = [_u_column_norm_variance(cc) for cc in layer_canonical.values()]
    canonical_output_norm_variance = float(np.mean(u_norm_variances)) if u_norm_variances else 0.0

    dominant_per_layer = [_dominant_directions(cc) for cc in layer_canonical.values()]
    canonical_dominant_directions = int(np.mean(dominant_per_layer)) if dominant_per_layer else 0

    # ------------------------------------------------------------------
    # Cross-layer correlation matrix (over canonical S vectors)
    # ------------------------------------------------------------------
    layer_vectors = [_layer_update_vector(cc) for cc in layer_canonical.values()]
    layer_correlation_matrix = _correlation_matrix(layer_vectors)

    # ------------------------------------------------------------------
    # Attention head specialization
    # ------------------------------------------------------------------
    attn_pairs: dict[str, tuple[CanonicalComponents, int]] = {
        name: (cc, _infer_n_heads(name))
        for name, cc in layer_canonical.items()
        if _infer_n_heads(name) >= 2
    }
    attention_head_specialization = _attention_head_specialization(attn_pairs)

    # ------------------------------------------------------------------
    # Training quality signals
    # ------------------------------------------------------------------
    weight_magnitude_histogram = _weight_magnitude_histogram(all_s_values)
    gradient_noise_estimate = _gradient_noise_estimate(layer_canonical)
    overfitting_sig = _overfitting_signature(layer_rank_utilization, layer_norms)

    # ------------------------------------------------------------------
    # Assemble observation
    # ------------------------------------------------------------------
    return WeightSignalObservation(
        weight_file_id=weight_file_id,
        lora_rank=lora_rank,
        target_modules=target_modules,
        total_parameters=total_parameters,
        # Layer-wise (canonical)
        layer_norms=layer_norms,
        layer_sparsity=layer_sparsity,
        layer_rank_utilization=layer_rank_utilization,
        # Canonical Vt analysis
        canonical_entropy_per_layer=canonical_entropy_per_layer,
        high_entropy_token_positions=high_entropy_token_positions,
        # Token-to-phoneme (from tokenizer config only)
        token_position_to_phoneme=token_position_to_phoneme,
        # Canonical U analysis
        canonical_output_norm_variance=canonical_output_norm_variance,
        canonical_dominant_directions=canonical_dominant_directions,
        # Cross-layer patterns
        layer_correlation_matrix=layer_correlation_matrix,
        attention_head_specialization=attention_head_specialization,
        # Training quality
        weight_magnitude_histogram=weight_magnitude_histogram,
        gradient_noise_estimate=gradient_noise_estimate,
        overfitting_signature=overfitting_sig,
        # Sub-env 2 context
        dataset_health_summary=dataset_health_summary,
        suspected_anomalous_phonemes=suspected_anomalous_phonemes,
    )
