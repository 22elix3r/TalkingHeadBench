"""
Unit tests for src/utils/grader_utils.py.

Tests cover every edge-case listed in the task spec for both:
  - set_f1()           — F1 over sets with correct empty-set handling
  - jaccard_similarity() — Jaccard index with matching empty-set guards

Floating-point assertions use pytest.approx(abs=1e-6) throughout so the 1e-8
epsilon baked into set_f1's denominator does not cause false failures.
"""

import pytest

from src.utils.grader_utils import jaccard_similarity, set_f1


# ===========================================================================
# Helpers
# ===========================================================================

def _f1_exact(predicted: set, true: set) -> float:
    """Reference implementation of set_f1 without the 1e-8 epsilon, used to
    compute expected values for partial-overlap cases."""
    if not predicted and not true:
        return 1.0
    if not predicted or not true:
        return 0.0
    p = len(predicted & true) / len(predicted)
    r = len(predicted & true) / len(true)
    return 2 * p * r / (p + r)


# ===========================================================================
# set_f1 tests
# ===========================================================================


class TestSetF1:
    """Tests for set_f1(predicted, true)."""

    # ---- Empty-set corner cases -------------------------------------------

    def test_both_empty_returns_one(self):
        """Both sides empty means both agree nothing to report → 1.0."""
        assert set_f1(set(), set()) == 1.0

    def test_predicted_empty_true_nonempty_returns_zero(self):
        """Agent predicted nothing when there were true items → 0.0 (all missed)."""
        assert set_f1(set(), {"a", "b", "c"}) == 0.0

    def test_true_empty_predicted_nonempty_returns_zero(self):
        """Agent hallucinated items when there were none → 0.0 (all false positives)."""
        assert set_f1({"x", "y"}, set()) == 0.0

    # ---- Perfect match ----------------------------------------------------

    def test_perfect_match_single_element(self):
        """Single element, exact match → 1.0."""
        result = set_f1({"a"}, {"a"})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_perfect_match_multiple_elements(self):
        """Multiple elements, exact set match → 1.0."""
        items = {"identity_collapse", "background_bleed", "temporal_jitter"}
        result = set_f1(items.copy(), items.copy())
        assert result == pytest.approx(1.0, abs=1e-6)

    # ---- Partial overlap (precision ≠ recall) -----------------------------

    def test_partial_overlap_precision_lt_recall(self):
        """Predicted is a superset of true (high recall, low precision).

        predicted = {A, B, C}  true = {A, B}
          precision = 2/3   recall = 2/2 = 1.0
          F1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
        """
        predicted = {"A", "B", "C"}
        true = {"A", "B"}
        expected = _f1_exact(predicted, true)   # 0.8
        result = set_f1(predicted, true)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_partial_overlap_recall_lt_precision(self):
        """Predicted is a subset of true (high precision, low recall).

        predicted = {A, B}  true = {A, B, C}
          precision = 2/2 = 1.0   recall = 2/3
          F1 = 2 * 1 * (2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
        """
        predicted = {"A", "B"}
        true = {"A", "B", "C"}
        expected = _f1_exact(predicted, true)   # 0.8
        result = set_f1(predicted, true)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_partial_overlap_asymmetric(self):
        """Predicted overlaps partially but precision ≠ recall ≠ simple fractions.

        predicted = {A, B, C}  true = {A, B, D, E}
          intersection = {A, B}  → 2
          precision = 2/3   recall = 2/4 = 0.5
          F1 = 2 * (2/3) * 0.5 / (2/3 + 0.5) = (2/3) / (7/6) = 4/7
        """
        predicted = {"A", "B", "C"}
        true = {"A", "B", "D", "E"}
        expected = _f1_exact(predicted, true)   # 4/7 ≈ 0.571428...
        result = set_f1(predicted, true)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result == pytest.approx(4 / 7, abs=1e-5)

    def test_no_overlap_nonempty_sets(self):
        """Disjoint non-empty sets → F1 = 0.0 (numerator is zero)."""
        result = set_f1({"x", "y"}, {"a", "b"})
        assert result == pytest.approx(0.0, abs=1e-6)

    # ---- Works on non-string element types --------------------------------

    def test_works_with_tuple_elements(self):
        """set_f1 should work on any hashable element type (e.g. tuples)."""
        predicted = {("EE", "smile"), ("OW", "jaw_drift")}
        true = {("EE", "smile"), ("AH", "brow_raise")}
        # intersection = 1, predicted = 2, true = 2
        # precision = 0.5, recall = 0.5, F1 = 0.5
        result = set_f1(predicted, true)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_works_with_frozenset_elements(self):
        """set_f1 should work when elements are frozensets (cluster identification)."""
        predicted = {frozenset(["A", "B"]), frozenset(["C"])}
        true = {frozenset(["A", "B"]), frozenset(["D"])}
        # intersection = 1, precision = 0.5, recall = 0.5, F1 = 0.5
        result = set_f1(predicted, true)
        assert result == pytest.approx(0.5, abs=1e-6)


# ===========================================================================
# jaccard_similarity tests
# ===========================================================================


class TestJaccardSimilarity:
    """Tests for jaccard_similarity(set_a, set_b)."""

    # ---- Empty-set corner cases -------------------------------------------

    def test_both_empty_returns_one(self):
        """Both empty sets are vacuously identical → 1.0."""
        assert jaccard_similarity(set(), set()) == 1.0

    def test_set_a_empty_returns_zero(self):
        """set_a empty, set_b non-empty → 0.0 (no overlap possible)."""
        assert jaccard_similarity(set(), {"a", "b"}) == 0.0

    def test_set_b_empty_returns_zero(self):
        """set_b empty, set_a non-empty → 0.0 (no overlap possible)."""
        assert jaccard_similarity({"x"}, set()) == 0.0

    # ---- Structural cases -------------------------------------------------

    def test_disjoint_sets_returns_zero(self):
        """Disjoint non-empty sets → intersection = 0 → Jaccard = 0.0."""
        result = jaccard_similarity({"a", "b"}, {"c", "d"})
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_identical_single_element(self):
        """Identical single-element sets → Jaccard = 1.0."""
        result = jaccard_similarity({"only"}, {"only"})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_identical_multi_element(self):
        """Identical multi-element sets → Jaccard = 1.0."""
        s = {"EE", "OW", "AH", "IY"}
        result = jaccard_similarity(s.copy(), s.copy())
        assert result == pytest.approx(1.0, abs=1e-6)

    # ---- Partial overlap --------------------------------------------------

    def test_partial_overlap_half(self):
        """Two elements overlap out of four distinct → Jaccard = 0.5.

        set_a = {1, 2, 3}  set_b = {2, 3, 4}
          intersection = {2, 3} → 2
          union         = {1, 2, 3, 4} → 4
          Jaccard = 2/4 = 0.5
        """
        result = jaccard_similarity({1, 2, 3}, {2, 3, 4})
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_partial_overlap_one_third(self):
        """One element overlaps out of five distinct → Jaccard = 1/5.

        set_a = {A, B, C}  set_b = {C, D, E}
          intersection = {C} → 1
          union         = {A, B, C, D, E} → 5
          Jaccard = 1/5 = 0.2
        """
        result = jaccard_similarity({"A", "B", "C"}, {"C", "D", "E"})
        assert result == pytest.approx(1 / 5, abs=1e-6)

    def test_subset_jaccard(self):
        """When set_a ⊂ set_b, Jaccard = |set_a| / |set_b|.

        set_a = {X, Y}  set_b = {X, Y, Z, W}
          Jaccard = 2/4 = 0.5
        """
        result = jaccard_similarity({"X", "Y"}, {"X", "Y", "Z", "W"})
        assert result == pytest.approx(0.5, abs=1e-6)

    # ---- Works on non-string element types --------------------------------

    def test_works_with_frozenset_elements(self):
        """jaccard_similarity must work on sets of frozensets (cluster grading)."""
        a = {frozenset(["EE", "IY"]), frozenset(["OW"])}
        b = {frozenset(["EE", "IY"]), frozenset(["AH"])}
        # intersection = 1, union = 3 → Jaccard = 1/3
        result = jaccard_similarity(a, b)
        assert result == pytest.approx(1 / 3, abs=1e-6)

    def test_works_with_tuple_elements(self):
        """jaccard_similarity should work on any hashable element type."""
        a = {(1, "a"), (2, "b"), (3, "c")}
        b = {(2, "b"), (3, "c"), (4, "d")}
        # intersection = 2, union = 4 → Jaccard = 0.5
        result = jaccard_similarity(a, b)
        assert result == pytest.approx(0.5, abs=1e-6)

    # ---- Symmetry property ------------------------------------------------

    def test_symmetry(self):
        """Jaccard similarity is symmetric: J(A, B) == J(B, A)."""
        a = {"dog", "cat", "fish"}
        b = {"cat", "bird"}
        assert jaccard_similarity(a, b) == pytest.approx(jaccard_similarity(b, a), abs=1e-9)
