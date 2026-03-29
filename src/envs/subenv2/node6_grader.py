"""
Node 6 (Grader): Dataset Health Grader — Sub-env 2.

This module implements the per-clip disposition grader that evaluates the
output of the Clip Disposition Recommender agent (Node 5).

``grade_clip_disposition`` scores a single clip decision.  The Dataset Health
Grader (Node 6) aggregates scores across all clips and produces the
``DatasetHealthHandoff``.

Scoring contract
----------------
- Maximum achievable score: 0.80 (0.40 base + 0.20 fix quality + 0.20 reasoning)
- Override misuse subtracts up to 0.10 from the running total.
- Final score is clamped to [0.0, 1.0] — never negative.
- The ``defer`` branch checks ``ground_truth.disposition_ambiguity >= 0.5``;
  deferring on an unambiguous case is penalised as avoidance.
- The override penalty applies **only** when ``override_decision == "applied"``.
"""

from __future__ import annotations

from src.schemas.ground_truth import GroundTruthClipAnnotation
from src.schemas.subenv2 import ClipDispositionAction


def grade_clip_disposition(
    agent_action: ClipDispositionAction,
    ground_truth: GroundTruthClipAnnotation,
) -> float:
    """Grade a single clip disposition recommendation (Node 5).

    Evaluates four dimensions and returns a clamped composite score in
    [0.0, 1.0]:

    **1. Base disposition** (0.40 max)
      - Correct disposition + confidence within 0.15 of ground truth → +0.40
      - Correct disposition but miscalibrated confidence             → +0.28
      - ``"fix"`` when ground truth is ``"reject"``                  → +0.20
        (partial credit: fix is better than blind accept)
      - ``"defer"`` on an ambiguous case (ambiguity ≥ 0.5):
          - with ``defer_reason``    → +0.15
          - without ``defer_reason`` → +0.10
      - ``"defer"`` on an unambiguous case (ambiguity < 0.5)         → −0.05
        (avoidance penalty; clamped so total never goes below 0.0)
      - ``"accept"`` when ground truth is ``"reject"``               → +0.00
        (worst case)

    **2. Fix instruction quality** (0.20 max)
      Applies only when ``disposition == "fix"`` and ``fix_instructions`` is
      non-empty.  Precision = fraction of agent steps present in
      ``ground_truth.valid_fix_steps``:
      - precision ≥ 0.8 → +0.20
      - precision ≥ 0.5 → +0.10
      - precision <  0.5 → +0.00

    **3. Dataset impact reasoning** (0.20 max)
      Checks how many of ``ground_truth.expected_reasoning_elements`` appear
      (case-insensitive substring) in ``dataset_impact_reasoning``:
      - ≥ 80 % matched → +0.20
      - ≥ 1  matched   → +0.10
      - none matched   → +0.00

    **4. Override misuse penalty** (applies only when ``override_decision == "applied"``)
      - No ``override_justification`` provided        → −0.10
      - Justification not in valid set                → −0.05
      - Valid justification present                   → no penalty

    Args:
        agent_action: The ``ClipDispositionAction`` produced by the Node 5 agent.
        ground_truth: The ``GroundTruthClipAnnotation`` for this test case.

    Returns:
        A float in [0.0, 1.0] representing the per-clip grader score.
    """
    score: float = 0.0

    # ------------------------------------------------------------------
    # 1. Base disposition (0.40 max)
    # ------------------------------------------------------------------
    if agent_action.disposition == ground_truth.disposition:
        calibrated = abs(agent_action.confidence - ground_truth.confidence) < 0.15
        score += 0.40 if calibrated else 0.28

    elif agent_action.disposition == "fix" and ground_truth.disposition == "reject":
        score += 0.20  # partial: fix is better than blind accept

    elif agent_action.disposition == "defer":
        if ground_truth.disposition_ambiguity >= 0.5:
            # Ambiguous case — deferring is reasonable; reward more if documented
            score += 0.15 if agent_action.defer_reason else 0.10
        else:
            # Unambiguous case — defer is avoidance; penalise
            score = max(score - 0.05, 0.0)

    elif agent_action.disposition == "accept" and ground_truth.disposition == "reject":
        score += 0.00  # worst case — no credit

    # ------------------------------------------------------------------
    # 2. Fix instruction quality (0.20 max)
    # ------------------------------------------------------------------
    if agent_action.disposition == "fix" and agent_action.fix_instructions:
        valid_steps = sum(
            1
            for step in agent_action.fix_instructions
            if step in ground_truth.valid_fix_steps
        )
        fix_precision = valid_steps / len(agent_action.fix_instructions)

        if fix_precision >= 0.8:
            score += 0.20
        elif fix_precision >= 0.5:
            score += 0.10
        # else: 0.00 — no addition

    # ------------------------------------------------------------------
    # 3. Dataset impact reasoning (0.20 max)
    # ------------------------------------------------------------------
    kw_elements = ground_truth.expected_reasoning_elements
    agent_text = agent_action.dataset_impact_reasoning.lower()
    matched = sum(1 for kw in kw_elements if kw in agent_text)

    if matched >= len(kw_elements) * 0.8:
        score += 0.20
    elif matched >= 1:
        score += 0.10
    # else: 0.00 — no addition

    # ------------------------------------------------------------------
    # 4. Override misuse penalty
    # Applies ONLY when override_decision == "applied".
    # ------------------------------------------------------------------
    if agent_action.override_decision == "applied":
        if not agent_action.override_justification:
            score -= 0.10
        elif (
            agent_action.override_justification
            not in ground_truth.valid_override_justifications
        ):
            score -= 0.05
        # Valid justification present → no penalty

    # ------------------------------------------------------------------
    # Clamp — score must never go below 0.0
    # ------------------------------------------------------------------
    return max(score, 0.0)
