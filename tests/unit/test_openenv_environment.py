"""Unit tests for OpenEnv environment reset/step behavior and provenance fields."""

from __future__ import annotations

import pytest

from server.talking_head_environment import TalkingHeadEnvironment
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.schemas.subenv1 import ImageDiagnosticsAction, ParamAnomalyAction
from src.schemas.subenv2 import ClipDispositionAction
from src.schemas.subenv3 import PhonemeRiskObservation


def _node1_action() -> ImageDiagnosticsAction:
    return ImageDiagnosticsAction(
        regime_classification="frontal_simple",
        identified_risk_factors=[],
        prompt_issues=[],
        recommended_prompt_modifications=[],
        image_usability_score=0.85,
        reasoning="Reference image looks usable with no critical risks.",
    )


def _node2_action() -> ParamAnomalyAction:
    return ParamAnomalyAction(
        config_risk_level="safe",
        anomalies=[],
        predicted_failure_modes=[],
        directional_fixes=[],
        summary="Configuration appears stable for this regime.",
    )


def test_reset_benchmark_mode_exposes_provenance():
    env = TalkingHeadEnvironment()

    obs = env.reset(seed=7)

    assert obs.done is False
    assert obs.node == "node1_image_diagnostician"
    assert obs.mode == "benchmark"
    assert obs.api_version == "1.0"
    assert obs.is_deterministic is True
    assert obs.provenance["bundle_source"] == "benchmark_test_set"


def test_reset_custom_bundle_exposes_custom_mode():
    env = TalkingHeadEnvironment()
    custom_bundle = {
        "case_id": "custom-1",
        "image_observation": {
            "face_occupancy_ratio": 0.45,
            "estimated_yaw_degrees": 4.0,
            "estimated_pitch_degrees": -1.0,
            "background_complexity_score": 0.25,
            "lighting_uniformity_score": 0.82,
            "skin_tone_bucket": 3,
            "occlusion_detected": False,
            "image_resolution": [1024, 1024],
            "estimated_sharpness": 0.8,
            "prompt_token_count": 24,
            "prompt_semantic_density": 0.65,
            "conflicting_descriptors": [],
            "identity_anchoring_strength": 0.9,
        },
        "param_config": {"cfg": 7.0, "denoise_alt": 0.3, "eta": 0.1},
        "ingestion_metadata": {"created_at_unix": 1, "api_version": "1.0"},
        "source_files": {"reference_image": "ref.png"},
    }

    obs = env.reset(custom_bundle=custom_bundle)

    assert obs.done is False
    assert obs.mode == "custom"
    assert obs.case_id == "custom-1"
    assert obs.provenance["bundle_source"] == "custom_bundle"
    assert "ingestion_metadata" in obs.provenance


def test_full_episode_step_flow_reaches_done():
    env = TalkingHeadEnvironment()

    obs0 = env.reset(seed=1)
    assert obs0.node == "node1_image_diagnostician"

    obs1 = env.step(_node1_action())
    assert obs1.done is False
    assert obs1.node == "node2_param_anomaly_detector"

    obs2 = env.step(_node2_action())
    assert obs2.done is True
    assert obs2.node == "episode_complete"
    assert 0.0 <= obs2.reward <= 1.0


def test_clip_mode_starts_at_node5_and_completes():
    env = TalkingHeadEnvironment()

    obs0 = env.reset(seed=2, mode="clips")
    assert obs0.done is False
    assert obs0.node == "node5_clip_disposition_recommender"

    action = ClipDispositionAction(
        disposition="accept",
        confidence=0.8,
        rejection_reasons=None,
        fix_instructions=None,
        estimated_fix_effort=None,
        defer_reason=None,
        dataset_impact_reasoning="Clip provides usable coverage without severe risks.",
        override_decision="not_applicable",
        override_justification=None,
    )
    obs1 = env.step(action)
    assert obs1.done is True
    assert obs1.node == "episode_complete"
    assert 0.0 <= obs1.reward <= 1.0


def test_weight_mode_starts_at_node8_and_completes():
    env = TalkingHeadEnvironment()

    obs0 = env.reset(seed=3, mode="weights")
    assert obs0.done is False
    assert obs0.node == "node8_phoneme_risk_assessor"

    node8_obs = PhonemeRiskObservation.model_validate(obs0.signals)
    node8_action = assess_phoneme_risk(node8_obs)

    obs1 = env.step(node8_action)
    assert obs1.done is True
    assert obs1.node == "episode_complete"
    assert 0.0 <= obs1.reward <= 1.0
    assert obs1.mode == "benchmark"


def test_reset_with_unknown_ingestion_id_fails():
    env = TalkingHeadEnvironment()

    with pytest.raises(RuntimeError, match="Unknown ingestion_id"):
        env.reset(ingestion_id="does-not-exist")
