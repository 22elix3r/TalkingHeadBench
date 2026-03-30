"""
W2T-style canonical LoRA decomposition (QR → SVD).

Used by the Node 7 Weight Signal Extractor environment pre-processing step.
All LoRA factor signals must pass through ``canonicalize_lora_factors`` before
any statistics are computed so that equivalent (A, B) matrix pairs produce
identical canonical representations, resolving column-space factorization
ambiguity as described in the W2T paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class CanonicalComponents:
    """Container for the four outputs of the QR → SVD canonical decomposition.

    Attributes:
        U:  Left singular vectors of the effective LoRA update ``B @ R_a``.
            Shape: ``(out_features, rank)``.
        S:  Singular values of the effective LoRA update, sorted descending.
            Shape: ``(rank,)``.
        Vt: Right singular vectors (transposed) of the effective update.
            Rows represent canonical input-space directions.
            Shape: ``(rank, in_features)``.
        Q:  Orthonormal matrix from the QR factorisation of ``A``.
            Shape: ``(in_features, rank)``.
    """

    U: Tensor
    S: Tensor
    Vt: Tensor
    Q: Tensor


def canonicalize_lora_factors(A: Tensor, B: Tensor) -> CanonicalComponents:
    """W2T-style canonical decomposition of a LoRA (A, B) weight pair: QR → SVD.

    Resolves column-space factorization ambiguity so that equivalent (A, B)
    pairs — i.e. those representing the same effective weight update
    ``delta_W = B @ A`` up to an invertible gauge transformation — produce
    identical canonical representations.  This is the core pre-processing step
    required before computing any per-layer statistics on LoRA factors.

    Algorithm
    ---------
    1. **QR step** — decompose ``A`` to resolve column-space ambiguity::

           Q_a, R_a = QR(A)

    2. **Effective update** — absorb the upper-triangular factor into ``B``::

           effective_update = B @ R_a

    3. **SVD step** — factorise the gauge-fixed effective update::

           U, S, Vt = SVD(effective_update)

    Args:
        A: The LoRA "down" projection matrix.  Must be a 2-D tensor of shape
            ``(in_features, rank)``.
        B: The LoRA "up" projection matrix.  Must be a 2-D tensor of shape
            ``(out_features, rank)``.

    Returns:
        A :class:`CanonicalComponents` dataclass with fields ``U``, ``S``,
        ``Vt``, and ``Q`` as described above.

    Raises:
        AssertionError: If either ``A`` or ``B`` is not a 2-D tensor.

    Example:
        >>> import torch
        >>> rank, in_f, out_f = 4, 64, 128
        >>> A = torch.randn(in_f, rank)
        >>> B = torch.randn(out_f, rank)
        >>> cc = canonicalize_lora_factors(A, B)
        >>> cc.S.shape
        torch.Size([4])
    """
    assert A.ndim == 2, (
        f"A must be a 2-D tensor, got shape {tuple(A.shape)} (ndim={A.ndim})"
    )
    assert B.ndim == 2, (
        f"B must be a 2-D tensor, got shape {tuple(B.shape)} (ndim={B.ndim})"
    )

    Q_a, R_a = torch.linalg.qr(A)                                          # Step 1: resolve column-space ambiguity
    effective_update = B @ R_a                                              # Step 2: form the effective LoRA update
    U, S, Vt = torch.linalg.svd(effective_update, full_matrices=False)     # Step 3: SVD
    return CanonicalComponents(U=U, S=S, Vt=Vt, Q=Q_a)


def layer_entropy_from_singular_values(S: Tensor) -> float:
    """Return inverted normalized entropy of the singular-value distribution.

    Interprets singular values as an energy distribution across canonical
    directions and computes:

    1) raw entropy of normalized singular values,
    2) normalization by log(rank),
    3) inversion so higher means more anomalous concentration.

    Returns:
      - 0.0 for uniform singular values (healthy utilization)
      - 1.0 for extreme concentration into one direction (rank collapse)
    """
    from scipy.stats import entropy as scipy_entropy

    s_np = S.detach().cpu().float().numpy()
    if s_np.size == 0:
        return 0.0

    s_np = np.clip(s_np, a_min=0.0, a_max=None)
    total = float(s_np.sum())
    if total <= 1e-8:
        return 0.0

    probs = s_np / (total + 1e-8)
    raw_entropy = float(scipy_entropy(probs))
    max_entropy = float(np.log(len(probs)))
    if max_entropy < 1e-8:
        return 0.0

    normalized = raw_entropy / max_entropy
    return float(np.clip(1.0 - normalized, 0.0, 1.0))


def singular_direction_anomaly_scores(S: Tensor) -> list[float]:
    """Return per-direction anomaly scores from normalized singular values.

    Uses normalized singular values as concentration scores in [0, 1].
    Higher values indicate directions that dominate layer energy.
    """
    s_np = S.detach().cpu().float().numpy()
    if s_np.size == 0:
        return []

    s_np = np.clip(s_np, a_min=0.0, a_max=None)
    total = float(s_np.sum())
    if total <= 1e-8:
        return [0.0 for _ in range(len(s_np))]

    probs = s_np / (total + 1e-8)
    return [float(v) for v in probs]
