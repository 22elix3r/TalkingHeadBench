"""
Integration tests for Sub-env 1: Reference Image + Prompt Audit.

Covers all three nodes end-to-end without any file I/O or external
dependencies.  The fixture is a single non-frontal reference image
observation that flows through Node 1 → Node 2 → Node 3.
"""

import pytest

from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import produce_reference_audit_handoff
from src.schemas.ground_truth import GroundTruthImageAnnotation, GroundTruthParamAnnotation
from src.schemas.subenv1 import ImageDiagnosticsObservation, ParamAnomalyObservation


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def non_frontal_obs() -> ImageDiagnosticsObservation:
    """Non-frontal reference image with weak lighting and conflicting descriptors."""
    return ImageDiagnosticsObservation(
        face_occupancy_ratio=0.45,
        estimated_yaw_degrees=32.0,
        estimated_pitch_degrees=5.0,
        background_complexity_score=0.35,
        lighting_uniformity_score=0.31,
        skin_tone_bucket=3,
        occlusion_detected=False,
        image_resolution=(1280, 720),
        estimated_sharpness=0.6,
        prompt_token_count=45,
        prompt_semantic_density=0.6,
        conflicting_descriptors=["dramatic lighting / natural look"],
        identity_anchoring_strength=0.3,
    )


# ---------------------------------------------------------------------------
# Test 1 — Node 1: Image Diagnostician
# ---------------------------------------------------------------------------


def test_node1_non_frontal_regime(non_frontal_obs):
    action = diagnose_image(non_frontal_obs)

    assert action.regime_classification == "non_frontal"
    assert len(action.identified_risk_factors) >= 1
    assert 0.0 <= action.image_usability_score <= 1.0
    assert len(action.recommended_prompt_modifications) >= 1


# ---------------------------------------------------------------------------
# Test 2 — Node 2: severe anomaly detected
# ---------------------------------------------------------------------------


def test_node2_detects_severe_anomaly(non_frontal_obs):
    # Run Node 1 first so we have real downstream values
    action = diagnose_image(non_frontal_obs)

    obs = ParamAnomalyObservation(
        proposed_config={"denoise_alt": 0.25, "cfg": 7.5, "eta": 0.15},
        regime=action.regime_classification,
        identified_risk_factors=action.identified_risk_factors,
        image_usability_score=action.image_usability_score,
        face_occupancy_ratio=0.45,
        estimated_yaw_degrees=32.0,
        background_complexity_score=0.35,
        lighting_uniformity_score=0.31,
        occlusion_detected=False,
        prompt_identity_anchoring=0.3,
        prompt_token_count=45,
        conflicting_descriptors=[],
    )
    anomaly_action = detect_param_anomalies(obs)

    assert any(a.severity == "severe" for a in anomaly_action.anomalies)
    assert anomaly_action.config_risk_level in ("risky", "dangerous")
    assert "reference_token_dropout" in anomaly_action.predicted_failure_modes
    assert len(anomaly_action.directional_fixes) >= 1


# ---------------------------------------------------------------------------
# Test 3 — Node 2: safe config produces no anomalies
# ---------------------------------------------------------------------------


def test_node2_safe_config(non_frontal_obs):
    action = diagnose_image(non_frontal_obs)

    obs_safe = ParamAnomalyObservation(
        proposed_config={"denoise_alt": 0.5, "cfg": 5.0, "eta": 0.05},
        regime=action.regime_classification,
        identified_risk_factors=action.identified_risk_factors,
        image_usability_score=action.image_usability_score,
        face_occupancy_ratio=0.45,
        estimated_yaw_degrees=32.0,
        background_complexity_score=0.35,
        lighting_uniformity_score=0.31,
        occlusion_detected=False,
        prompt_identity_anchoring=0.3,
        prompt_token_count=45,
        conflicting_descriptors=[],
    )
    safe_action = detect_param_anomalies(obs_safe)

    assert safe_action.config_risk_level == "safe"
    assert safe_action.anomalies == []


# ---------------------------------------------------------------------------
# Test 4 — Node 3: grader produces a valid ReferenceAuditHandoff
# ---------------------------------------------------------------------------


def test_node3_grader_produces_handoff(non_frontal_obs):
    # Produce real node outputs
    action = diagnose_image(non_frontal_obs)

    obs = ParamAnomalyObservation(
        proposed_config={"denoise_alt": 0.25, "cfg": 7.5, "eta": 0.15},
        regime=action.regime_classification,
        identified_risk_factors=action.identified_risk_factors,
        image_usability_score=action.image_usability_score,
        face_occupancy_ratio=0.45,
        estimated_yaw_degrees=32.0,
        background_complexity_score=0.35,
        lighting_uniformity_score=0.31,
        occlusion_detected=False,
        prompt_identity_anchoring=0.3,
        prompt_token_count=45,
        conflicting_descriptors=[],
    )
    anomaly_action = detect_param_anomalies(obs)

    img_gt = GroundTruthImageAnnotation(
        regime_classification="non_frontal",
        acceptable_regimes=["complex_background"],
        identified_risk_factors=[
            "yaw exceeds 25° — lateral pose reduces reference token coverage"
        ],
        valid_prompt_modifications=[
            "resolve conflicting descriptors: dramatic lighting / natural look"
        ],
    )
    param_gt = GroundTruthParamAnnotation(
        config_risk_level="dangerous",
        anomalies=[],
        predicted_failure_modes=["reference_token_dropout"],
        valid_fix_directions=[],
    )

    handoff = produce_reference_audit_handoff(action, anomaly_action, img_gt, param_gt)

    assert 0.0 <= handoff.subenv1_score <= 1.0
    assert handoff.risk_profile in ("low", "medium", "high")
    assert 0.0 <= handoff.estimated_drift_risk <= 1.0
