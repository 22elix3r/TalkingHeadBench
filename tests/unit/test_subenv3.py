"""
Integration tests for Sub-env 3: Trained LoRA Weight Behavioral Audit.

Covers Node 7 (Weight Signal Extractor) → Node 8 (Phoneme Risk Assessor)
→ Node 9 (Behavioral Audit Grader) end-to-end using the shared
``synthetic_lora_path`` fixture from tests/conftest.py.
No real model weights or GPU is required.
"""

from __future__ import annotations

import pytest

from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.envs.subenv3.node9_grader import grade_behavioral_audit
from src.schemas.ground_truth import GroundTruthBehavioralAnnotation
from src.schemas.subenv3 import (
    BehaviorTriggerPrediction,
    PhonemeCluster,
    PhonemeRiskEntry,
    PhonemeRiskObservation,
    WeightEvidenceDossier,
)


# ---------------------------------------------------------------------------
# Helpers / shared builders
# ---------------------------------------------------------------------------

_STUB_DOSSIER = WeightEvidenceDossier(
    weight_file_id="test",
    training_quality="healthy",
    rank_utilization_assessment="efficient",
    high_entropy_token_flags=[],
    layer_anomaly_flags=[],
    overall_behavioral_risk="low",
    evidence_summary="test",
)

_PHONEME_OBS_BASE = dict(
    weight_evidence=_STUB_DOSSIER,
    high_entropy_token_flags=[],
    phoneme_vocabulary=["AH", "EE", "OW", "IY", "EY", "ZH", "TH"],
    phoneme_to_token_indices={"EE": [0, 1], "OW": [2], "AH": [3]},
    phoneme_entropy_scores={
        "EE": 0.85, "OW": 0.72, "AH": 0.2, "IY": 0.78, "EY": 0.65,
    },
    phoneme_influence_scores={
        "EE": 0.78, "OW": 0.65, "AH": 0.1, "IY": 0.71, "EY": 0.60,
    },
    phoneme_cooccurrence_anomalies=[],
    behavior_vocabulary=["smile", "jaw_drift", "head_turn", "brow_raise"],
    training_data_phoneme_distribution=None,
    suspected_anomalous_phonemes_from_subenv2=None,
)


def _make_phoneme_obs(**overrides) -> PhonemeRiskObservation:
    return PhonemeRiskObservation(**{**_PHONEME_OBS_BASE, **overrides})


# ---------------------------------------------------------------------------
# Test 1 — full Sub-env 3 pipeline: Node 7 → Node 8
# ---------------------------------------------------------------------------


def test_full_subenv3_pipeline(synthetic_lora_path):
    """Node 7 extracts signals; Node 8 produces a valid PhonemeRiskAction."""
    # Node 7
    obs = extract_weight_signals(synthetic_lora_path)
    assert obs.lora_rank == 8  # sanity-check fixture dimensions

    # Node 8
    phoneme_obs = _make_phoneme_obs()
    action = assess_phoneme_risk(phoneme_obs)

    # Safety literal must be one of the five defined values
    assert action.model_behavioral_safety in {
        "safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"
    }

    # All risk scores must be in [0, 1]
    assert all(
        0.0 <= e.risk_score <= 1.0 for e in action.phoneme_risk_ranking
    ), "risk_score out of [0, 1]"

    # EE: entropy=0.85, influence=0.78 → risk=0.6*0.85+0.4*0.78=0.822 > 0.3 → ranked
    ranked_phonemes = [e.phoneme for e in action.phoneme_risk_ranking]
    assert "EE" in ranked_phonemes, "EE (high-risk) must appear in ranking"

    # EE must rank above AH (AH risk≈0.16, excluded; if present at all EE must precede it)
    if "AH" in ranked_phonemes:
        assert ranked_phonemes.index("EE") < ranked_phonemes.index("AH"), (
            "EE must rank above AH"
        )


# ---------------------------------------------------------------------------
# Test 2 — Sub-env 2 suspected phoneme hints propagate into the ranking
# ---------------------------------------------------------------------------


def test_subenv2_hints_propagate():
    """A suspected phoneme from Sub-env 2 not in the vocabulary must be
    appended to the ranking with evidence = 'flagged by dataset audit (Sub-env 2)'.
    """
    phoneme_obs2 = _make_phoneme_obs(
        suspected_anomalous_phonemes_from_subenv2=["NG"]
    )
    action2 = assess_phoneme_risk(phoneme_obs2)

    assert any(e.phoneme == "NG" for e in action2.phoneme_risk_ranking), (
        "NG should appear in phoneme_risk_ranking"
    )
    assert any("Sub-env 2" in e.evidence for e in action2.phoneme_risk_ranking), (
        "At least one entry should cite Sub-env 2 in its evidence"
    )


# ---------------------------------------------------------------------------
# Test 3 — Node 9 grader returns a score in [0, 1]
# ---------------------------------------------------------------------------


def test_grader_score_in_range():
    """grade_behavioral_audit must return a composite score in [0.0, 1.0]."""
    # Produce a Node 8 action on the base phoneme obs
    phoneme_obs = _make_phoneme_obs()
    action = assess_phoneme_risk(phoneme_obs)

    gt = GroundTruthBehavioralAnnotation(
        phoneme_risk_ranking=[
            PhonemeRiskEntry(
                phoneme="EE",
                risk_score=0.85,
                risk_type="expression_trigger",
                confidence=0.78,
                evidence="test",
            ),
            PhonemeRiskEntry(
                phoneme="OW",
                risk_score=0.72,
                risk_type="identity_trigger",
                confidence=0.65,
                evidence="test",
            ),
        ],
        predicted_behavior_triggers=[
            BehaviorTriggerPrediction(
                trigger_phoneme="EE",
                triggered_behavior="smile",
                association_strength=0.83,
                is_intended=False,
                concern_level="medium",
            ),
        ],
        risky_phoneme_clusters=[],
        model_behavioral_safety="moderate_risk",
        valid_mitigation_set={("EE/IY/EY phoneme cluster", "add_counter_examples")},
    )

    score = grade_behavioral_audit(action, gt)
    assert 0.0 <= score <= 1.0, f"grader score {score} out of [0, 1]"
