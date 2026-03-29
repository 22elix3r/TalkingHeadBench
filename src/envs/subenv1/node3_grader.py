"""
Node 3 (Grader): Reference Audit Grader — Sub-env 1.

This module implements the grader logic for Sub-env 1.  The Node 3 grader
evaluates two upstream agent outputs:

  - ``grade_image_diagnostics``  — Node 1 (Image Diagnostician)
  - ``grade_anomaly_detection``  — Node 2 (Parameter Anomaly Detector)

Only ``grade_anomaly_detection`` is included per the current task scope; the
Node 1 grader lives alongside it for completeness and is implemented directly
from the spec.

Risk calibration uses **ordinal distance**, not binary equality.
``failure_mode_prediction`` delegates to ``set_f1`` from
``src/utils/grader_utils`` — it is not re-implemented here.
"""

from __future__ import annotations

from src.schemas.ground_truth import GroundTruthImageAnnotation, GroundTruthParamAnnotation
from src.schemas.subenv1 import (
    DirectionalFix,
    ImageDiagnosticsAction,
    ParamAnomalyAction,
    ReferenceAuditHandoff,
)
from src.utils.grader_utils import set_f1

# Ordered risk levels used for ordinal-distance calibration (Node 2).
_RISK_LEVELS: list[str] = ["safe", "marginal", "risky", "dangerous"]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _evaluate_directional_fixes(
    agent_fixes: list[DirectionalFix],
    valid_fixes: list[DirectionalFix],
) -> float:
    """Score the agent's directional fixes against the groud-truth valid set.

    A fix is considered valid when both its ``target`` and ``direction``
    match an entry in ``valid_fixes``.  The score is the precision of
    the agent's fix list against that valid set:

        score = |matched| / |agent_fixes|

    Special cases mirror the ``set_f1`` empty-set convention:
    - Both empty  → 1.0  (both sides agree: no fixes needed)
    - Agent empty only → 0.0  (agent missed all required fixes)
    - Valid empty only → 0.0  (agent hallucinated fixes)

    Args:
        agent_fixes: Fixes recommended by the agent (``ParamAnomalyAction``).
        valid_fixes: Ground-truth valid fix directions
            (``GroundTruthParamAnnotation.valid_fix_directions``).

    Returns:
        A float in [0.0, 1.0].
    """
    if not agent_fixes and not valid_fixes:
        return 1.0
    if not agent_fixes or not valid_fixes:
        return 0.0

    valid_pairs: set[tuple[str, str]] = {
        (fix.target, fix.direction) for fix in valid_fixes
    }
    matched = sum(
        1 for fix in agent_fixes if (fix.target, fix.direction) in valid_pairs
    )
    return matched / len(agent_fixes)


# ---------------------------------------------------------------------------
# Node 2 grader
# ---------------------------------------------------------------------------


def grade_anomaly_detection(
    agent_action: ParamAnomalyAction,
    ground_truth: GroundTruthParamAnnotation,
) -> float:
    """Grade the Parameter Anomaly Detector agent output (Node 2).

    Evaluates four dimensions and returns a weighted composite score in
    [0.0, 1.0]:

    1. **Anomaly detection** (weight 0.30) — macro-averaged F1 over the set of
       flagged parameter names.  Handles three cases explicitly:
       - Both empty → 1.0 (agent correctly found nothing).
       - True set empty → 0.0 (agent hallucinated anomalies).
       - Otherwise → 0.5 * recall + 0.5 * precision.

    2. **Failure mode prediction** (weight 0.25) — set F1 over predicted
       failure mode strings, delegated to ``set_f1()`` from
       ``src.utils.grader_utils``.

    3. **Directional fix quality** (weight 0.30) — precision of the agent's
       (target, direction) fix pairs against the ground-truth valid set.

    4. **Risk level calibration** (weight 0.15) — ordinal distance between the
       agent's ``config_risk_level`` and the ground-truth level on the ordered
       scale ``["safe", "marginal", "risky", "dangerous"]``.  NOT binary —
       adjacent misses are penalised less than distant misses.

    Args:
        agent_action: The ``ParamAnomalyAction`` produced by the Node 2 agent.
        ground_truth: The ``GroundTruthParamAnnotation`` for this test case.

    Returns:
        A float in [0.0, 1.0] representing the composite grader score.
    """
    scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Anomaly detection — F1 over flagged parameters
    # ------------------------------------------------------------------
    predicted_params: set[str] = {a.parameter for a in agent_action.anomalies}
    true_params: set[str] = {a.parameter for a in ground_truth.anomalies}

    if not predicted_params and not true_params:
        scores["anomaly_detection"] = 1.0
    elif not true_params:
        scores["anomaly_detection"] = 0.0  # agent hallucinated anomalies
    else:
        recall = len(predicted_params & true_params) / len(true_params)
        precision = (
            len(predicted_params & true_params) / len(predicted_params)
            if predicted_params
            else 0.0
        )
        scores["anomaly_detection"] = 0.5 * recall + 0.5 * precision

    # ------------------------------------------------------------------
    # 2. Failure mode prediction — set F1
    # ------------------------------------------------------------------
    scores["failure_mode_prediction"] = set_f1(
        set(agent_action.predicted_failure_modes),
        set(ground_truth.predicted_failure_modes),
    )

    # ------------------------------------------------------------------
    # 3. Directional fix quality
    # ------------------------------------------------------------------
    scores["fix_quality"] = _evaluate_directional_fixes(
        agent_action.directional_fixes,
        ground_truth.valid_fix_directions,
    )

    # ------------------------------------------------------------------
    # 4. Risk level calibration — ordinal distance (NOT binary)
    # ------------------------------------------------------------------
    agent_idx = _RISK_LEVELS.index(agent_action.config_risk_level)
    true_idx = _RISK_LEVELS.index(ground_truth.config_risk_level)
    scores["risk_calibration"] = 1.0 - abs(agent_idx - true_idx) / (
        len(_RISK_LEVELS) - 1
    )

    # ------------------------------------------------------------------
    # Weighted composite
    # ------------------------------------------------------------------
    return (
        0.30 * scores["anomaly_detection"]
        + 0.25 * scores["failure_mode_prediction"]
        + 0.30 * scores["fix_quality"]
        + 0.15 * scores["risk_calibration"]
    )


# ---------------------------------------------------------------------------
# Node 3 — produce ReferenceAuditHandoff (public entry-point)
# ---------------------------------------------------------------------------


def produce_reference_audit_handoff(
    node1_action: ImageDiagnosticsAction,
    node2_action: ParamAnomalyAction,
    img_gt: GroundTruthImageAnnotation,
    param_gt: GroundTruthParamAnnotation,
) -> ReferenceAuditHandoff:
    """Grade both upstream nodes and return the Sub-env 1 → 2 coupling object.

    Computes ``node1_score`` (image diagnostics) and ``node2_score`` (anomaly
    detection), combines them with equal weight into ``subenv1_score``, and
    packages the result into a :class:`ReferenceAuditHandoff`.

    This is the public equivalent of ``pipeline._build_reference_audit_handoff``
    and is intended for direct use in unit and integration tests.

    Args:
        node1_action: Output of the Image Diagnostician agent (Node 1).
        node2_action: Output of the Parameter Anomaly Detector agent (Node 2).
        img_gt: Ground-truth annotation for Node 1 grading.
        param_gt: Ground-truth annotation for Node 2 grading.

    Returns:
        A fully populated :class:`ReferenceAuditHandoff`.
    """
    # --- grade both nodes (re-use local graders) ----------------------------
    node1_score = _grade_image_diagnostics_local(node1_action, img_gt)
    node2_score = grade_anomaly_detection(node2_action, param_gt)
    subenv1_score = 0.50 * node1_score + 0.50 * node2_score

    # --- risk profile from Node 2 risk level --------------------------------
    risk_map = {"safe": "low", "marginal": "low", "risky": "medium", "dangerous": "high"}
    risk_profile = risk_map.get(node2_action.config_risk_level, "medium")

    # --- config quality: invert risk ordinal --------------------------------
    risk_ordinal = {"safe": 1.0, "marginal": 0.75, "risky": 0.40, "dangerous": 0.10}
    config_quality_score = risk_ordinal.get(node2_action.config_risk_level, 0.5)

    # --- estimated drift risk: proportion of severe anomalies ---------------
    total_anomalies = len(node2_action.anomalies)
    severe = sum(1 for a in node2_action.anomalies if a.severity == "severe")
    estimated_drift_risk = severe / total_anomalies if total_anomalies else 0.0

    return ReferenceAuditHandoff(
        image_usability_score=node1_action.image_usability_score,
        regime=node1_action.regime_classification,
        identified_risk_factors=node1_action.identified_risk_factors,
        config_quality_score=config_quality_score,
        risk_profile=risk_profile,
        estimated_drift_risk=estimated_drift_risk,
        prompt_strength=max(0.0, 1.0 - float(len(node1_action.prompt_issues)) * 0.1),
        recommended_config={},
        subenv1_score=subenv1_score,
    )


# ---------------------------------------------------------------------------
# Private helper (image diagnostics grader — mirrors pipeline._grade_image_diagnostics)
# ---------------------------------------------------------------------------


def _grade_image_diagnostics_local(
    agent_action: ImageDiagnosticsAction,
    ground_truth: GroundTruthImageAnnotation,
) -> float:
    """Local copy of the Node 1 grader (avoids a circular import with pipeline)."""
    scores: dict[str, float] = {}

    if agent_action.regime_classification == ground_truth.regime_classification:
        scores["regime_accuracy"] = 1.0
    elif agent_action.regime_classification in ground_truth.acceptable_regimes:
        scores["regime_accuracy"] = 0.7
    else:
        scores["regime_accuracy"] = 0.0

    predicted_risks = set(agent_action.identified_risk_factors)
    true_risks = set(ground_truth.identified_risk_factors)
    scores["risk_factor_recall"] = (
        len(predicted_risks & true_risks) / len(true_risks) if true_risks else 1.0
    )

    valid_mods = set(ground_truth.valid_prompt_modifications)
    agent_mods = set(agent_action.recommended_prompt_modifications)
    if agent_mods:
        scores["prompt_modification_validity"] = len(agent_mods & valid_mods) / len(agent_mods)
    else:
        scores["prompt_modification_validity"] = 0.0 if valid_mods else 1.0

    return (
        0.35 * scores["regime_accuracy"]
        + 0.35 * scores["risk_factor_recall"]
        + 0.30 * scores["prompt_modification_validity"]
    )
