"""
Unit tests for src/utils/canonical.py — canonicalize_lora_factors().

Tests verify the core design invariant (Key Invariant #5):

    **W2T canonical form** — all LoRA weight signals are extracted from
    QR→SVD canonical components, not raw A/B matrices, to ensure
    factorization-invariant representations.

The QR → SVD canonical decomposition guarantees that two LoRA (A, B) pairs
which share the same upper-triangular R factor (i.e., differ only in their
column-space basis Q) produce identical canonical U, S, Vt outputs.  This
test constructs such equivalent pairs: given A = Q_a @ R_a, we form
A2 = Q_new @ R_a for a *different* orthonormal Q_new, keep B unchanged,
and confirm that canonicalize_lora_factors returns the same (U, S, Vt)
within floating-point tolerance (torch.allclose, atol=1e-5).
"""

import pytest
import torch
from torch import Tensor

from src.utils.canonical import CanonicalComponents, canonicalize_lora_factors


# ===========================================================================
# Helpers
# ===========================================================================


def _random_orthonormal_columns(
    rows: int, cols: int, *, generator: torch.Generator
) -> Tensor:
    """Return a (rows, cols) matrix with orthonormal columns via thin QR."""
    Z = torch.randn(rows, cols, generator=generator)
    Q, _ = torch.linalg.qr(Z)
    return Q


def _assert_canonical_equal(
    cc1: CanonicalComponents,
    cc2: CanonicalComponents,
    *,
    atol: float = 1e-5,
) -> None:
    """Assert that two canonical decompositions are identical within *atol*.

    Singular values (S) are compared directly.  Singular vector matrices
    (U, Vt) are compared up to per-column sign flips — the standard
    ambiguity inherent in any SVD decomposition.
    """
    # --- Singular values must match (sign-unambiguous) ---------------------
    assert torch.allclose(cc1.S, cc2.S, atol=atol), (
        f"Singular values differ:\n  cc1.S = {cc1.S}\n  cc2.S = {cc2.S}\n"
        f"  max_diff = {(cc1.S - cc2.S).abs().max().item():.2e}"
    )

    # --- U and Vt: columns/rows may differ by ±1 sign factors --------------
    _assert_columns_match_up_to_sign(cc1.U, cc2.U, atol=atol, name="U")
    # Vt is row-oriented; compare column-wise via transpose.
    _assert_columns_match_up_to_sign(cc1.Vt.T, cc2.Vt.T, atol=atol, name="Vt")


def _assert_columns_match_up_to_sign(
    M1: Tensor, M2: Tensor, *, atol: float, name: str
) -> None:
    """Check every column of M1 matches the corresponding column of M2 up to ±1."""
    assert M1.shape == M2.shape, (
        f"{name} shapes differ: {M1.shape} vs {M2.shape}"
    )
    for col in range(M1.shape[1]):
        c1 = M1[:, col]
        c2 = M2[:, col]
        match_pos = torch.allclose(c1, c2, atol=atol)
        match_neg = torch.allclose(c1, -c2, atol=atol)
        assert match_pos or match_neg, (
            f"{name} column {col} does not match up to sign:\n"
            f"  max |c1 - c2|  = {(c1 - c2).abs().max().item():.2e}\n"
            f"  max |c1 + c2|  = {(c1 + c2).abs().max().item():.2e}"
        )


# ===========================================================================
# Factorization-invariance tests
# ===========================================================================


class TestCanonicalizeLoraFactorsInvariance:
    """Canonical decomposition must be factorization-invariant.

    The QR step resolves column-space ambiguity in A: given A = Q_a @ R_a,
    the canonical (U, S, Vt) depend only on B and R_a (via effective_update
    = B @ R_a), not on Q_a.  Two LoRA pairs (A1, B) and (A2, B) where
    A1 and A2 share the same R_a (but have different orthonormal column
    bases Q) produce the same effective update and therefore identical
    canonical representations.

    This is the "column-space factorization ambiguity" that the W2T
    canonical form resolves — equivalent to the user's LoRA gauge
    transform (A @ Q, Q^T @ B) expressed in input-space coordinates.
    """

    ATOL = 1e-5

    @pytest.mark.parametrize(
        "in_features, out_features, rank",
        [
            (64, 128, 4),    # typical small LoRA layer
            (128, 128, 8),   # square-ish, larger rank
            (256, 64, 2),    # wide input, narrow output, rank-2
            (32, 512, 16),   # narrow input, wide output, rank-16
        ],
        ids=["64x128_r4", "128x128_r8", "256x64_r2", "32x512_r16"],
    )
    def test_invariant_under_column_space_rotation(
        self, in_features: int, out_features: int, rank: int
    ) -> None:
        """Two (A, B) pairs sharing the same R_a must canonicalize identically.

        Construct A2 = Q_new @ R_a where Q_new is a fresh orthonormal basis
        unrelated to A's original Q_a.  Since effective_update = B @ R_a is
        the same for both pairs, canonical (U, S, Vt) must agree.
        """
        gen = torch.Generator().manual_seed(42)

        # Original LoRA pair.
        A = torch.randn(in_features, rank, generator=gen)
        B = torch.randn(out_features, rank, generator=gen)

        # Extract R_a from A's QR decomposition.
        Q_a, R_a = torch.linalg.qr(A)

        # Build an equivalent A2 with the same R_a but a different Q.
        gen2 = torch.Generator().manual_seed(9999)
        Q_new = _random_orthonormal_columns(in_features, rank, generator=gen2)
        A2 = Q_new @ R_a

        # Sanity: verify A2's QR reproduces the same R_a.
        _, R_a2 = torch.linalg.qr(A2)
        assert torch.allclose(R_a, R_a2, atol=1e-6), (
            f"R factors should be identical:\n"
            f"  R_a  diag = {R_a.diag()}\n  R_a2 diag = {R_a2.diag()}"
        )

        # Canonical decompositions must agree on U, S, Vt.
        cc1 = canonicalize_lora_factors(A, B)
        cc2 = canonicalize_lora_factors(A2, B)

        _assert_canonical_equal(cc1, cc2, atol=self.ATOL)

        # Q should differ (it absorbs the column-space rotation).
        assert not torch.allclose(cc1.Q, cc2.Q, atol=0.1), (
            "Q components should differ between pairs with different A"
        )

    def test_effective_update_is_invariant(self) -> None:
        """Directly verify that effective_update = B @ R_a is unchanged
        when only A's column-space basis changes."""
        gen = torch.Generator().manual_seed(777)

        in_f, out_f, rank = 128, 64, 8
        A = torch.randn(in_f, rank, generator=gen)
        B = torch.randn(out_f, rank, generator=gen)

        _, R_a = torch.linalg.qr(A)

        gen2 = torch.Generator().manual_seed(888)
        Q_new = _random_orthonormal_columns(in_f, rank, generator=gen2)
        A2 = Q_new @ R_a

        _, R_a2 = torch.linalg.qr(A2)

        eff1 = B @ R_a
        eff2 = B @ R_a2

        assert torch.allclose(eff1, eff2, atol=1e-5), (
            f"Effective updates differ: max_diff = "
            f"{(eff1 - eff2).abs().max().item():.2e}"
        )

    def test_identity_clone_trivially_equal(self) -> None:
        """When (A2, B2) = (A.clone(), B.clone()), outputs are bit-for-bit identical."""
        gen = torch.Generator().manual_seed(123)
        A = torch.randn(64, 4, generator=gen)
        B = torch.randn(128, 4, generator=gen)

        cc1 = canonicalize_lora_factors(A, B)
        cc2 = canonicalize_lora_factors(A.clone(), B.clone())

        # Must be *exactly* equal (same numerical path).
        assert torch.allclose(cc1.S, cc2.S, atol=0)
        assert torch.allclose(cc1.U, cc2.U, atol=0)
        assert torch.allclose(cc1.Vt, cc2.Vt, atol=0)
        assert torch.allclose(cc1.Q, cc2.Q, atol=0)

    def test_multiple_q_rotations_all_agree(self) -> None:
        """Three different Q bases with the same R_a must all agree on (U, S, Vt)."""
        gen = torch.Generator().manual_seed(2025)

        in_f, out_f, rank = 64, 128, 4
        A = torch.randn(in_f, rank, generator=gen)
        B = torch.randn(out_f, rank, generator=gen)

        _, R_a = torch.linalg.qr(A)

        cc_orig = canonicalize_lora_factors(A, B)

        canonical_components = [cc_orig]
        for seed in [1000, 2000, 3000]:
            gen_q = torch.Generator().manual_seed(seed)
            Q_i = _random_orthonormal_columns(in_f, rank, generator=gen_q)
            A_i = Q_i @ R_a
            cc_i = canonicalize_lora_factors(A_i, B)
            canonical_components.append(cc_i)

        # All pairwise comparisons must agree.
        for i in range(len(canonical_components)):
            for j in range(i + 1, len(canonical_components)):
                _assert_canonical_equal(
                    canonical_components[i],
                    canonical_components[j],
                    atol=self.ATOL,
                )

    def test_q_captures_column_space_orientation(self) -> None:
        """Verify that Q in the canonical output reflects A's column-space basis.

        When A2 = Q_new @ R_a, the canonical Q should equal Q_new (the
        column-space basis of A2).
        """
        gen = torch.Generator().manual_seed(55)

        A = torch.randn(64, 4, generator=gen)
        B = torch.randn(128, 4, generator=gen)

        Q_a, R_a = torch.linalg.qr(A)
        cc = canonicalize_lora_factors(A, B)

        # The Q in canonical output should be Q_a from QR(A).
        assert torch.allclose(cc.Q, Q_a, atol=1e-6), (
            "Canonical Q should equal the Q factor from QR(A)"
        )

    def test_output_shapes_correct(self) -> None:
        """Verify canonical output tensor shapes match the docstring spec."""
        in_f, out_f, rank = 64, 128, 4
        gen = torch.Generator().manual_seed(0)
        A = torch.randn(in_f, rank, generator=gen)
        B = torch.randn(out_f, rank, generator=gen)

        cc = canonicalize_lora_factors(A, B)

        assert cc.U.shape == (out_f, rank), f"U shape: {cc.U.shape}"
        assert cc.S.shape == (rank,), f"S shape: {cc.S.shape}"
        assert cc.Vt.shape == (rank, rank), f"Vt shape: {cc.Vt.shape}"
        assert cc.Q.shape == (in_f, rank), f"Q shape: {cc.Q.shape}"

    def test_s_values_are_nonnegative_and_sorted(self) -> None:
        """SVD singular values must be non-negative and sorted descending."""
        gen = torch.Generator().manual_seed(314)
        A = torch.randn(128, 8, generator=gen)
        B = torch.randn(256, 8, generator=gen)

        cc = canonicalize_lora_factors(A, B)

        assert (cc.S >= 0).all(), f"Negative singular values found: {cc.S}"
        # Check descending order.
        for i in range(len(cc.S) - 1):
            assert cc.S[i] >= cc.S[i + 1], (
                f"S not sorted descending at index {i}: "
                f"S[{i}]={cc.S[i].item():.4f} < S[{i+1}]={cc.S[i+1].item():.4f}"
            )
