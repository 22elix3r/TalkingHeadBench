"""
Pydantic v2 ground-truth annotation schemas for TalkingHeadBench.

These schemas define what the graders expect.  Use them for all test-set
annotations.  Four schemas cover all three sub-environments:

  - GroundTruthImageAnnotation    — Node 1/3 (Sub-env 1 image + prompt)
  - GroundTruthParamAnnotation    — Node 2/3 (Sub-env 1 parameter audit)
  - GroundTruthClipAnnotation     — Node 5/6 (Sub-env 2 clip disposition)
  - GroundTruthBehavioralAnnotation — Node 8/9 (Sub-env 3 behavioral audit)

Cross-module imports bring in the reusable sub-types defined alongside their
respective agent schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.schemas.subenv1 import DirectionalFix, ParameterAnomaly
from src.schemas.subenv3 import (
    BehaviorTriggerPrediction,
    PhonemeCluster,
    PhonemeRiskEntry,
)


class GroundTruthImageAnnotation(BaseModel):
    """Ground-truth annotation for Node 1 (Image Diagnostician) evaluation.

    ``valid_prompt_modifications`` is a curated set of acceptable modifications
    per test case, enabling deterministic grading without an LLM judge.
    ``acceptable_regimes`` lists borderline alternatives that receive partial
    credit (0.7) from the Node 3 grader.
    """

    regime_classification: str
    acceptable_regimes: list[str]           # borderline alternatives for partial credit
    identified_risk_factors: list[str]
    valid_prompt_modifications: list[str]   # curated set — enables deterministic grading


class GroundTruthParamAnnotation(BaseModel):
    """Ground-truth annotation for Node 2 (Parameter Anomaly Detector) evaluation."""

    config_risk_level: Literal["safe", "marginal", "risky", "dangerous"]
    anomalies: list[ParameterAnomaly]
    predicted_failure_modes: list[str]
    valid_fix_directions: list[DirectionalFix]


class GroundTruthClipAnnotation(BaseModel):
    """Ground-truth annotation for Node 5 (Clip Disposition Recommender) evaluation.

    ``disposition_ambiguity`` should be set by human annotators during test-set
    construction:
      - 0.0 = unambiguous (clear-cut accept or reject)
      - 1.0 = genuinely contested (expert disagreement)

    ``valid_override_justifications`` is matched via exact string comparison in
    the reference grader; replace with semantic similarity in production since
    ``override_justification`` is free-text.
    """

    disposition: Literal["accept", "reject", "fix", "defer"]
    confidence: float
    disposition_ambiguity: float            # 0.0 = unambiguous, 1.0 = genuinely contested
    valid_fix_steps: list[str]
    valid_override_justifications: list[str]
    expected_reasoning_elements: list[str]  # keywords grader checks for in reasoning


class GroundTruthBehavioralAnnotation(BaseModel):
    """Ground-truth annotation for Node 8 (Phoneme Risk Assessor) evaluation.

    ``valid_mitigation_set`` is a set of ``(target, action)`` pairs checked by
    the Node 9 grader against the agent's mitigation recommendations.
    """

    phoneme_risk_ranking: list[PhonemeRiskEntry]    # top-k reference
    predicted_behavior_triggers: list[BehaviorTriggerPrediction]
    risky_phoneme_clusters: list[PhonemeCluster]
    model_behavioral_safety: str
    valid_mitigation_set: set[tuple[str, str]]       # (target, action) pairs
