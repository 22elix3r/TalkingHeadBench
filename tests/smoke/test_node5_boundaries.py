"""
Smoke tests: Node 5 (Clip Disposition Recommender) boundary conditions.

Covers:
  1. Disposition routing based on quality score and phoneme value.
  2. Specific overrides for negative training impact.
  3. Fix instruction generation and effort estimation.
  4. Reasoning string content.
  5. Confidence range validation.
"""

from __future__ import annotations

import pytest
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.schemas.subenv2 import ClipDispositionObservation, ClipEvidenceDossier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dossier(**overrides) -> ClipEvidenceDossier:
    """Create a high-quality dossier by default."""
    defaults = dict(
        clip_id="test_clip",
        identity_drift_severity="none",
        temporal_instability_flag=False,
        lip_sync_quality="good",
        unique_phoneme_value=0.5,
        dataset_redundancy_score=0.2,
        estimated_training_impact="positive",
        primary_rejection_reason=None,
        evidence_summary="Baseline high-quality clip evidence.",
    )
    return ClipEvidenceDossier(**{**defaults, **overrides})


def make_obs(dossier: ClipEvidenceDossier, **ctx_overrides) -> ClipDispositionObservation:
    """Wrap a dossier in an observation context."""
    defaults = dict(
        evidence_dossier=dossier,
        minimum_clips_needed=20,
        phoneme_gap_severity={"ZH": 2, "TH": 1},
        pose_gap_severity={},
        budget_remaining=10,
        reference_risk_profile="medium",
        estimated_drift_risk=0.4,
        marginal_training_damage=0.2,
        marginal_coverage_gain=0.5,
    )
    return ClipDispositionObservation(**{**defaults, **ctx_overrides})


# ===========================================================================
# Disposition routing
# ===========================================================================

def test_high_quality_accepts():
    """High quality (no drift, stable, good sync, low redundancy) -> accept."""
    dossier = make_dossier(
        identity_drift_severity="none",
        temporal_instability_flag=False,
        lip_sync_quality="good",
        unique_phoneme_value=0.9,
        dataset_redundancy_score=0.05
    )
    # quality = (1.0 + 0.3 + 0.3 + 2*0.9 + 0.2*0.95)/2 = (1.0 + 0.3 + 0.3 + 0.18 + 0.19)/2 = 1.97/2 = 0.985
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "accept"
    assert action.confidence > 0.7


def test_severe_drift_low_phoneme_rejects():
    """Severe drift + low value + high redundancy -> reject or fix (mostly reject if quality < 0.3)."""
    dossier = make_dossier(
        identity_drift_severity="severe",
        temporal_instability_flag=True,
        lip_sync_quality="poor",
        unique_phoneme_value=0.1,
        dataset_redundancy_score=0.95
    )
    # quality = (0.0 + 0.0 + 0.0 + 0.2*0.1 + 0.2*0.05)/2 = (0.02 + 0.01)/2 = 0.015
    # quality < 0.3 and unique_phoneme_value (0.1) <= 0.3 -> reject
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "reject"


def test_severe_drift_rare_phonemes_fixes():
    """Severe quality issues but high phoneme value -> fix (rare phonemes salvageable)."""
    dossier = make_dossier(
        identity_drift_severity="severe",
        temporal_instability_flag=True,
        lip_sync_quality="poor",
        unique_phoneme_value=0.8,
        dataset_redundancy_score=0.2
    )
    # quality = (0.0 + 0.0 + 0.0 + 0.2*0.8 + 0.2*0.8)/2 = (0.16 + 0.16)/2 = 0.16
    # quality < 0.3 and unique_phoneme_value (0.8) > 0.3 -> fix
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "fix"


def test_fix_generates_instructions():
    """Any clip routed to 'fix' must have actionable instructions."""
    dossier = make_dossier(
        identity_drift_severity="moderate",
        temporal_instability_flag=True,
        unique_phoneme_value=0.8
    )
    # quality = (0.3 + 0.0 + 0.3 + 0.16 + 0.16)/2 = 0.92/2 = 0.46
    # 0.3 <= quality < 0.5 -> fix
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "fix"
    assert action.fix_instructions is not None
    assert len(action.fix_instructions) >= 2  # instability fix + retained value info


def test_defer_has_reason():
    """Borderline quality + neutral impact -> defer with reason."""
    dossier = make_dossier(
        identity_drift_severity="moderate",
        temporal_instability_flag=False,
        dataset_redundancy_score=0.5,
        estimated_training_impact="neutral"
    )
    # quality = (0.3 + 0.3 + 0.3 + 0.2*0.5 + 0.2*0.5)/2 = 1.1/2 = 0.55
    # 0.5 <= quality < 0.7 and impact neutral -> defer
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "defer"
    assert action.defer_reason is not None
    assert "quality borderline" in action.defer_reason


def test_override_applied_for_negative_impact():
    """Accepting/fixing despite negative impact -> override applied."""
    dossier = make_dossier(
        identity_drift_severity="none",
        estimated_training_impact="negative",
        primary_rejection_reason="Artifacts found in backdrop"
    )
    # quality ~ 0.8 -> accept
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "accept"
    assert action.override_decision == "applied"
    assert action.override_justification is not None
    assert "Artifacts" in action.override_justification


def test_override_not_applicable_for_positive_impact():
    """Positive impact -> no override needed."""
    dossier = make_dossier(estimated_training_impact="positive")
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.override_decision == "not_applicable"


@pytest.mark.parametrize("drift,stability,sync,val,redun", [
    ("none", False, "good", 0.9, 0.1),
    ("severe", True, "poor", 0.1, 0.9),
    ("moderate", False, "poor", 0.5, 0.5),
    ("minor", True, "good", 0.3, 0.2),
    ("none", True, "poor", 0.8, 0.3),
    ("severe", False, "good", 0.1, 0.1),
    ("moderate", True, "good", 0.9, 0.8),
    ("minor", False, "poor", 0.2, 0.4),
    ("none", False, "poor", 0.5, 0.9),
    ("severe", True, "good", 0.9, 0.05),
])
def test_confidence_always_in_range(drift, stability, sync, val, redun):
    """Quality (confidence) must stay in [0, 1]."""
    dossier = make_dossier(
        identity_drift_severity=drift,
        temporal_instability_flag=stability,
        lip_sync_quality=sync,
        unique_phoneme_value=val,
        dataset_redundancy_score=redun
    )
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert 0.0 <= action.confidence <= 1.0


def test_dataset_impact_mentions_phoneme_gap():
    """Reasoning should reflect the provided phoneme gaps."""
    obs = make_obs(make_dossier(), phoneme_gap_severity={"ZH": 2, "NG": 1})
    action = recommend_clip_disposition(obs)
    assert "ZH" in action.dataset_impact_reasoning
    assert "NG" in action.dataset_impact_reasoning or "phoneme" in action.dataset_impact_reasoning.lower()


def test_fix_effort_trivial_for_single_flag():
    """Only temporal instability -> trivial fix effort."""
    dossier = make_dossier(
        identity_drift_severity="none",      # score 1.0
        temporal_instability_flag=True,      # score 0.0 (flag present)
        lip_sync_quality="good",             # score 0.3
        unique_phoneme_value=0.5,           # score 0.1
        dataset_redundancy_score=0.2,        # score 0.16
        # total quality = (1.0 + 0.0 + 0.3 + 0.1 + 0.16)/2 = 1.56/2 = 0.78
        # wait, quality 0.78 -> accept. I need quality in fix range [0.3, 0.5) OR (<0.3 with val>0.3)
    )
    # adjustment to get quality ~ 0.4
    dossier = make_dossier(
        identity_drift_severity="moderate",  # 0.3
        temporal_instability_flag=True,      # 0.0
        lip_sync_quality="acceptable",       # 0.3
        unique_phoneme_value=0.5,           # 0.1
        dataset_redundancy_score=0.5,        # 0.1
        # quality = (0.3 + 0.0 + 0.3 + 0.1 + 0.1)/2 = 0.8/2 = 0.4 -> fix
    )
    obs = make_obs(dossier)
    action = recommend_clip_disposition(obs)
    assert action.disposition == "fix"
    assert action.estimated_fix_effort == "trivial"
