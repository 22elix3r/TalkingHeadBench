"""
Smoke tests: grader scoring arithmetic.

Verifies that every scoring formula produces the *exact* numeric value
mandated by the spec — not just "in-range" but within floating-point
tolerance.  No mocking is used; all functions are exercised with
carefully-chosen inputs that isolate a single scoring component.

Sections
--------
1. set_f1  precision/recall asymmetry
2. Ordinal risk calibration  (Node 2 — grade_anomaly_detection)
3. Ordinal safety calibration (Node 9 — grade_behavioral_audit)
4. Node 5 disposition grader boundary cases (grade_clip_disposition)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grader functions
# ---------------------------------------------------------------------------
from src.envs.subenv1.node3_grader import grade_anomaly_detection
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node9_grader import grade_behavioral_audit

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
from src.utils.grader_utils import jaccard_similarity, set_f1  # noqa: F401

# ---------------------------------------------------------------------------
# Schemas — ground-truth
# ---------------------------------------------------------------------------
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)

# ---------------------------------------------------------------------------
# Schemas — sub-env 1
# ---------------------------------------------------------------------------
from src.schemas.subenv1 import (
    DirectionalFix,
    ImageDiagnosticsAction,
    ParameterAnomaly,
    ParamAnomalyAction,
    ParamAnomalyObservation,
    ReferenceAuditHandoff,
)

# ---------------------------------------------------------------------------
# Schemas — sub-env 2
# ---------------------------------------------------------------------------
from src.schemas.subenv2 import (
    ClipDispositionAction,
    ClipEvidenceDossier,
    ClipDispositionObservation,
    ClipSignalObservation,
    DatasetHealthHandoff,
    SyntheticWeightDescriptor,
)

# ---------------------------------------------------------------------------
# Schemas — sub-env 3
# ---------------------------------------------------------------------------
from src.schemas.subenv3 import (
    BehaviorTriggerPrediction,
    BehavioralAuditHandoff,
    LayerAnomalyFlag,
    MitigationRecommendation,
    PhonemeCluster,
    PhonemeRiskAction,
    PhonemeRiskEntry,
    TokenAnomalyFlag,
    WeightEvidenceDossier,
    WeightSignalObservation,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_empty_param_gt(config_risk_level: str = "safe") -> GroundTruthParamAnnotation:
    """Ground-truth with no anomalies / failure modes / fixes at given risk level."""
    return GroundTruthParamAnnotation(
        config_risk_level=config_risk_level,  # type: ignore[arg-type]
        anomalies=[],
        predicted_failure_modes=[],
        valid_fix_directions=[],
    )


def _make_param_action(config_risk_level: str) -> ParamAnomalyAction:
    """Agent action with no anomalies / failure modes / fixes at given risk level."""
    return ParamAnomalyAction(
        config_risk_level=config_risk_level,  # type: ignore[arg-type]
        anomalies=[],
        predicted_failure_modes=[],
        directional_fixes=[],
        summary="",
    )


def _make_empty_behavioral_gt(model_behavioral_safety: str = "safe") -> GroundTruthBehavioralAnnotation:
    """Ground-truth with no rankings / triggers / clusters at given safety level."""
    return GroundTruthBehavioralAnnotation(
        phoneme_risk_ranking=[],
        predicted_behavior_triggers=[],
        risky_phoneme_clusters=[],
        model_behavioral_safety=model_behavioral_safety,
        valid_mitigation_set=set(),
    )


def _make_phoneme_risk_action(model_behavioral_safety: str) -> PhonemeRiskAction:
    """Agent action with no rankings / triggers / clusters at given safety level."""
    return PhonemeRiskAction(
        phoneme_risk_ranking=[],
        predicted_behavior_triggers=[],
        risky_phoneme_clusters=[],
        model_behavioral_safety=model_behavioral_safety,  # type: ignore[arg-type]
        mitigation_recommendations=[],
        summary="",
    )


def _score_risk_only(agent_risk: str, gt_risk: str) -> float:
    """Return just the risk_calibration component of grade_anomaly_detection.

    With empty anomalies/failure_modes/fixes on both sides:
      - anomaly_detection  = 1.0  (both empty → perfect)
      - failure_prediction = 1.0  (both empty → perfect)
      - fix_quality        = 1.0  (both empty → perfect)
      - risk_calibration   = ordinal_score

    composite = 0.30*1.0 + 0.25*1.0 + 0.30*1.0 + 0.15*risk_calibration
              = 0.85 + 0.15*risk_calibration

    So: risk_calibration = (composite - 0.85) / 0.15
    """
    action = _make_param_action(agent_risk)
    gt = _make_empty_param_gt(gt_risk)
    composite = grade_anomaly_detection(action, gt)
    return (composite - 0.85) / 0.15


def _score_safety_only(agent_safety: str, gt_safety: str) -> float:
    """Return just the safety_calibration component of grade_behavioral_audit.

    With empty lists on both sides:
      - ranking_quality    = 0/5 = 0.0  (both empty → 0 overlap)
      - trigger_prediction = 1.0  (both empty → perfect)
      - cluster_ident      = 1.0  (both empty → perfect)
      - safety_calibration = ordinal_score
      - mitigation_quality = 1.0  (both empty → perfect)

    composite = 0.15*0.0 + 0.30*1.0 + 0.20*1.0 + 0.15*safety_calibration + 0.20*1.0
              = 0.70 + 0.15*safety_calibration

    So: safety_calibration = (composite - 0.70) / 0.15
    """
    action = _make_phoneme_risk_action(agent_safety)
    gt = _make_empty_behavioral_gt(gt_safety)
    composite = grade_behavioral_audit(action, gt)
    return (composite - 0.70) / 0.15


def _make_clip_gt(
    disposition: str = "reject",
    confidence: float = 0.80,
    disposition_ambiguity: float = 0.0,
    valid_fix_steps: list[str] | None = None,
    valid_override_justifications: list[str] | None = None,
    expected_reasoning_elements: list[str] | None = None,
) -> GroundTruthClipAnnotation:
    # Default to a single dummy element that will never appear in the empty
    # dataset_impact_reasoning string, so the reasoning score is always 0.0
    # unless the caller explicitly sets expected_reasoning_elements.
    if expected_reasoning_elements is None:
        expected_reasoning_elements = ["__REQUIRED_KEYWORD__"]
    return GroundTruthClipAnnotation(
        disposition=disposition,  # type: ignore[arg-type]
        confidence=confidence,
        disposition_ambiguity=disposition_ambiguity,
        valid_fix_steps=valid_fix_steps or [],
        valid_override_justifications=valid_override_justifications or [],
        expected_reasoning_elements=expected_reasoning_elements,
    )


def _make_clip_action(
    disposition: str,
    confidence: float = 0.80,
    defer_reason: str | None = None,
    fix_instructions: list[str] | None = None,
    estimated_fix_effort: str | None = None,
    rejection_reasons: list[str] | None = None,
    override_decision: str = "not_applicable",
    override_justification: str | None = None,
    dataset_impact_reasoning: str = "",
) -> ClipDispositionAction:
    return ClipDispositionAction(
        disposition=disposition,  # type: ignore[arg-type]
        confidence=confidence,
        rejection_reasons=rejection_reasons,
        fix_instructions=fix_instructions,
        estimated_fix_effort=estimated_fix_effort,  # type: ignore[arg-type]
        defer_reason=defer_reason,
        dataset_impact_reasoning=dataset_impact_reasoning,
        override_decision=override_decision,  # type: ignore[arg-type]
        override_justification=override_justification,
    )


# ===========================================================================
# 1. set_f1 precision / recall asymmetry
# ===========================================================================


class TestSetF1:
    """Verify the harmonic-mean formula for asymmetric predicted/true sets."""

    def test_f1_precision_dominated(self):
        """predicted ⊂ true → precision=1.0, recall=0.2 → F1 = 1/3."""
        predicted = {"A"}
        true = {"A", "B", "C", "D", "E"}
        # precision = 1/1 = 1.0,  recall = 1/5 = 0.2
        # F1 = 2 * 1.0 * 0.2 / (1.0 + 0.2) = 0.4 / 1.2 = 1/3
        assert abs(set_f1(predicted, true) - 1 / 3) < 1e-6

    def test_f1_recall_dominated(self):
        """true ⊂ predicted → precision=0.2, recall=1.0 → F1 = 1/3 (symmetric)."""
        predicted = {"A", "B", "C", "D", "E"}
        true = {"A"}
        # precision = 1/5 = 0.2,  recall = 1/1 = 1.0
        # F1 = 2 * 0.2 * 1.0 / (0.2 + 1.0) = 0.4 / 1.2 = 1/3
        assert abs(set_f1(predicted, true) - 1 / 3) < 1e-6

    def test_f1_symmetry(self):
        """set_f1(p, t) == set_f1(t, p) for partially overlapping sets."""
        p = {"A", "B", "C"}
        t = {"B", "C", "D"}
        assert abs(set_f1(p, t) - set_f1(t, p)) < 1e-9

    def test_f1_perfect(self):
        """Identical non-empty sets → F1 ≈ 1.0 (epsilon in denominator is negligible)."""
        s = {"X", "Y", "Z"}
        assert abs(set_f1(s, s) - 1.0) < 1e-6

    def test_f1_both_empty(self):
        """Both sets empty → 1.0 (both sides agree nothing to report)."""
        assert set_f1(set(), set()) == 1.0

    def test_f1_one_empty(self):
        """Exactly one set empty → 0.0."""
        assert set_f1(set(), {"A"}) == 0.0
        assert set_f1({"A"}, set()) == 0.0


# ===========================================================================
# 2. Ordinal risk calibration — Node 2 (grade_anomaly_detection)
# ===========================================================================


class TestOrdinalRiskCalibration:
    """Risk levels: ["safe", "marginal", "risky", "dangerous"] (max distance = 3)."""

    def test_ordinal_risk_exact_match(self):
        """Same level → ordinal distance 0 → calibration score 1.0."""
        for level in ("safe", "marginal", "risky", "dangerous"):
            calib = _score_risk_only(level, level)
            assert abs(calib - 1.0) < 1e-9, f"Failed for {level}"

    def test_ordinal_risk_adjacent(self):
        """safe → marginal: distance=1 → score = 1 - 1/3 = 2/3."""
        calib = _score_risk_only("marginal", "safe")
        assert abs(calib - 2 / 3) < 1e-6

    def test_ordinal_risk_max_distance(self):
        """safe → dangerous: distance=3 → score = 1 - 3/3 = 0.0."""
        calib = _score_risk_only("dangerous", "safe")
        assert abs(calib - 0.0) < 1e-9

    def test_ordinal_risk_max_distance_full_grade(self):
        """Confirm the 0.15 weight contribution when risk_calibration = 0.0.

        With all other components perfect (empty sets on both sides, so
        anomaly_detection=1.0, failure_mode=1.0, fix_quality=1.0) the
        composite should be exactly 0.85.
        """
        action = _make_param_action("dangerous")
        gt = _make_empty_param_gt("safe")
        score = grade_anomaly_detection(action, gt)
        assert abs(score - 0.85) < 1e-9

    def test_ordinal_risk_correct_level_full_grade(self):
        """When risk_calibration = 1.0 and others = 1.0, composite = 1.0."""
        action = _make_param_action("safe")
        gt = _make_empty_param_gt("safe")
        score = grade_anomaly_detection(action, gt)
        assert abs(score - 1.0) < 1e-6


# ===========================================================================
# 3. Ordinal safety calibration — Node 9 (grade_behavioral_audit)
# ===========================================================================


class TestOrdinalSafetyCalibration:
    """Safety levels: ["safe","minor_concerns","moderate_risk","high_risk","unsafe"] (max dist=4)."""

    def test_ordinal_safety_exact_match(self):
        """Same level → score = 1.0."""
        for level in ("safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"):
            calib = _score_safety_only(level, level)
            assert abs(calib - 1.0) < 1e-9, f"Failed for {level}"

    def test_ordinal_safety_safe_to_unsafe(self):
        """safe → unsafe: distance=4 → score = 1 - 4/4 = 0.0."""
        calib = _score_safety_only("unsafe", "safe")
        assert abs(calib - 0.0) < 1e-9

    def test_ordinal_safety_safe_to_high_risk(self):
        """safe → high_risk: distance=3 → score = 1 - 3/4 = 0.25."""
        calib = _score_safety_only("high_risk", "safe")
        assert abs(calib - 0.25) < 1e-6

    def test_ordinal_safety_minor_to_moderate(self):
        """minor_concerns → moderate_risk: distance=1 → score = 1 - 1/4 = 0.75."""
        calib = _score_safety_only("moderate_risk", "minor_concerns")
        assert abs(calib - 0.75) < 1e-6

    def test_ordinal_safety_max_full_grade(self):
        """safe/unsafe worst case: safety_calibration=0.0 → composite = 0.70."""
        action = _make_phoneme_risk_action("unsafe")
        gt = _make_empty_behavioral_gt("safe")
        score = grade_behavioral_audit(action, gt)
        assert abs(score - 0.70) < 1e-9

    def test_ordinal_safety_perfect_full_grade(self):
        """Matching safety, empty lists → composite = 0.70 + 0.15 = 0.85."""
        action = _make_phoneme_risk_action("safe")
        gt = _make_empty_behavioral_gt("safe")
        score = grade_behavioral_audit(action, gt)
        assert abs(score - 0.85) < 1e-9


# ===========================================================================
# 4. Node 5 disposition grader boundary cases
# ===========================================================================


class TestDispositionGraderBoundaries:
    """Verify each branch of grade_clip_disposition produces the exact score."""

    # --- Correct disposition + confidence calibration ---

    def test_disposition_correct_wellcalibrated(self):
        """Correct disposition, |confidence diff| < 0.15 → base = 0.40."""
        # No fix/reasoning bonus (reasoning elements empty → 0 matched → 0.0)
        action = _make_clip_action(disposition="reject", confidence=0.82)
        gt = _make_clip_gt(disposition="reject", confidence=0.80)
        # diff = 0.02 < 0.15 → +0.40; no fix, no reasoning → 0.40 total
        assert abs(grade_clip_disposition(action, gt) - 0.40) < 1e-9

    def test_disposition_correct_miscalibrated(self):
        """Correct disposition, |confidence diff| >= 0.15 → base = 0.28."""
        action = _make_clip_action(disposition="reject", confidence=0.82)
        gt = _make_clip_gt(disposition="reject", confidence=0.60)
        # diff = 0.22 >= 0.15 → +0.28; no fix, no reasoning → 0.28 total
        assert abs(grade_clip_disposition(action, gt) - 0.28) < 1e-9

    # --- Wrong disposition corner cases ---

    def test_fix_when_gt_is_reject(self):
        """fix vs reject → partial credit +0.20 base (no fix_instructions → no bonus)."""
        action = _make_clip_action(disposition="fix", confidence=0.70)
        gt = _make_clip_gt(disposition="reject", confidence=0.70)
        # base = 0.20; fix_instructions is None → section skipped; empty reasoning → 0.0
        assert abs(grade_clip_disposition(action, gt) - 0.20) < 1e-9

    def test_accept_when_gt_is_reject(self):
        """accept vs reject → +0.00 base, no bonus, no penalty."""
        action = _make_clip_action(disposition="accept", confidence=0.70)
        gt = _make_clip_gt(disposition="reject", confidence=0.70)
        assert abs(grade_clip_disposition(action, gt) - 0.00) < 1e-9

    # --- defer branch ---

    def test_defer_ambiguous_with_reason(self):
        """defer on ambiguous case (ambiguity=0.7 >= 0.5) with defer_reason → +0.15."""
        action = _make_clip_action(
            disposition="defer",
            confidence=0.50,
            defer_reason="unclear",
        )
        gt = _make_clip_gt(
            disposition="reject",
            confidence=0.70,
            disposition_ambiguity=0.7,
        )
        assert abs(grade_clip_disposition(action, gt) - 0.15) < 1e-9

    def test_defer_ambiguous_no_reason(self):
        """defer on ambiguous case, defer_reason=None → +0.10."""
        action = _make_clip_action(disposition="defer", confidence=0.50)
        gt = _make_clip_gt(
            disposition="reject",
            confidence=0.70,
            disposition_ambiguity=0.7,
        )
        assert abs(grade_clip_disposition(action, gt) - 0.10) < 1e-9

    def test_defer_unambiguous(self):
        """defer on unambiguous case (ambiguity=0.1 < 0.5) → avoidance penalty −0.05,
        clamped to 0.0 (score starts at 0.0, subtract 0.05, clamp → 0.0)."""
        action = _make_clip_action(disposition="defer", confidence=0.50)
        gt = _make_clip_gt(
            disposition="reject",
            confidence=0.70,
            disposition_ambiguity=0.1,
        )
        # max(0.0 - 0.05, 0.0) = 0.0
        assert abs(grade_clip_disposition(action, gt) - 0.00) < 1e-9

    # --- Override penalty ---

    def test_override_penalty_no_justification(self):
        """override_decision='applied' with empty override labels → no penalty.

        By design, override penalties are only evaluated when
        valid_override_justifications is non-empty in ground truth.
        With an empty valid list (annotation placeholder), override is
        treated as "not evaluated".
        """
        action = _make_clip_action(
            disposition="reject",
            confidence=0.80,
            override_decision="applied",
            override_justification=None,
        )
        gt = _make_clip_gt(disposition="reject", confidence=0.80)
        assert abs(grade_clip_disposition(action, gt) - 0.40) < 1e-9

    def test_override_penalty_invalid_justification(self):
        """override_decision='applied', justification not in valid set → penalty −0.05.

        Base: correct disposition, well-calibrated → +0.40.
        Override penalty: −0.05.
        Expected final: 0.40 − 0.05 = 0.35.
        """
        action = _make_clip_action(
            disposition="reject",
            confidence=0.80,
            override_decision="applied",
            override_justification="not in valid list",
        )
        gt = _make_clip_gt(
            disposition="reject",
            confidence=0.80,
            valid_override_justifications=["specific valid reason"],
        )
        assert abs(grade_clip_disposition(action, gt) - 0.35) < 1e-9
