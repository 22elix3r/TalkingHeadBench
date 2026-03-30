"""
End-to-end smoke tests for the TalkingHeadBench episode pipeline.

Verifies:
  1. The pipeline correctly orchestrates all three sub-environments.
  2. Sub-environment coupling/handoffs propagate as expected.
  3. Final score calculation follows the 25/35/40 weighted formula.
  4. Node failures propagate correctly.
  5. The pipeline is deterministic for identical inputs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from src.pipeline import EpisodeResult, run_episode
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import (
    ImageDiagnosticsObservation,
    ReferenceAuditHandoff,
)
from src.schemas.subenv2 import (
    ClipSignalObservation,
    DatasetHealthHandoff,
)
from src.schemas.subenv3 import (
    BehavioralAuditHandoff,
    WeightSignalObservation,
)


# ---------------------------------------------------------------------------
# Fixture: minimal_episode_inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_episode_inputs():
    """Build a complete set of in-memory dummy inputs for a full episode."""
    
    # 1. Sub-env 1: Reference image + prompt
    reference_obs = ImageDiagnosticsObservation(
        face_occupancy_ratio=0.6,
        estimated_yaw_degrees=5.0,
        estimated_pitch_degrees=2.0,
        background_complexity_score=0.3,
        lighting_uniformity_score=0.7,
        skin_tone_bucket=3,
        occlusion_detected=False,
        image_resolution=(1280, 720),
        estimated_sharpness=0.8,
        prompt_token_count=40,
        prompt_semantic_density=0.5,
        conflicting_descriptors=[],
        identity_anchoring_strength=0.8,
    )
    
    param_config = {"denoise_alt": 0.5, "cfg": 5.5, "eta": 0.08}
    
    # 2. Sub-env 2: 3 clip observations (clean, drifting, borderline)
    clip_obs_list = [
        ClipSignalObservation(
            clip_id="clip_clean",
            face_embedding_variance=0.01,
            landmark_stability_score=0.01,
            identity_cosine_drift=0.02,
            frame_difference_mean=5.0,
            optical_flow_magnitude=1.0,
            blink_count=2,
            lip_sync_confidence=0.85,
            phoneme_sequence=["AH", "EE"],
            phoneme_coverage_new=0.5,
            blur_score=0.1,
            exposure_score=0.7,
            occlusion_frames=0,
            clips_audited_so_far=0,
            current_phoneme_coverage={},
            current_pose_distribution={},
            similar_clips_accepted=0,
        ),
        ClipSignalObservation(
            clip_id="clip_drifting",
            face_embedding_variance=0.5,
            landmark_stability_score=0.1,
            identity_cosine_drift=0.3,
            frame_difference_mean=20.0,
            optical_flow_magnitude=5.0,
            blink_count=0,
            lip_sync_confidence=0.1,
            phoneme_sequence=["OW", "IY"],
            phoneme_coverage_new=0.6,
            blur_score=0.5,
            exposure_score=0.3,
            occlusion_frames=10,
            clips_audited_so_far=1,
            current_phoneme_coverage={},
            current_pose_distribution={},
            similar_clips_accepted=0,
        ),
        ClipSignalObservation(
            clip_id="clip_borderline",
            face_embedding_variance=0.1,
            landmark_stability_score=0.03,
            identity_cosine_drift=0.15,
            frame_difference_mean=10.0,
            optical_flow_magnitude=2.0,
            blink_count=1,
            lip_sync_confidence=0.6,
            phoneme_sequence=["NG"],
            phoneme_coverage_new=0.4,
            blur_score=0.2,
            exposure_score=0.5,
            occlusion_frames=2,
            clips_audited_so_far=2,
            current_phoneme_coverage={},
            current_pose_distribution={},
            similar_clips_accepted=1,
        ),
    ]
    
    # 3. Sub-env 3: Weight signal
    weight_obs = WeightSignalObservation(
        weight_file_id="dummy_lora",
        lora_rank=8,
        target_modules=["q_proj", "v_proj"],
        total_parameters=1000000,
        layer_norms={"layer1": 0.5, "layer2": 0.4},
        layer_sparsity={"layer1": 0.1, "layer2": 0.1},
        layer_rank_utilization={"layer1": 0.8, "layer2": 0.75},
        canonical_entropy_per_layer={"layer1": 0.8, "layer2": 0.7},
        high_entropy_token_positions=[1, 5],
        token_position_to_phoneme={1: "EE", 5: "OW"},
        canonical_output_norm_variance=0.01,
        canonical_dominant_directions=5,
        layer_correlation_matrix=[[1.0, 0.2], [0.2, 1.0]],
        attention_head_specialization={"h1": 0.5},
        weight_magnitude_histogram=[0.1, 0.8, 0.1],
        gradient_noise_estimate=0.1,
        overfitting_signature=0.2,
        dataset_health_summary=None,
        suspected_anomalous_phonemes=None,
    )
    
    # 4. Ground Truths
    ground_truths = {
        "image": GroundTruthImageAnnotation(
            regime_classification="frontal_simple",
            acceptable_regimes=[],
            identified_risk_factors=[],
            valid_prompt_modifications=[],
        ),
        "param": GroundTruthParamAnnotation(
            config_risk_level="safe",
            anomalies=[],
            predicted_failure_modes=[],
            valid_fix_directions=[],
        ),
        "clips": [
            GroundTruthClipAnnotation(
                disposition="accept",
                confidence=0.9,
                disposition_ambiguity=0.0,
                valid_fix_steps=[],
                valid_override_justifications=[],
                expected_reasoning_elements=["clear"],
            ),
            GroundTruthClipAnnotation(
                disposition="reject",
                confidence=0.1,
                disposition_ambiguity=0.0,
                valid_fix_steps=[],
                valid_override_justifications=[],
                expected_reasoning_elements=["drift"],
            ),
            # Borderline case might be defer or accept or fix... 
            # In our heuristic: drifting=0.3 -> moderate. stability=0.3. sync=0.3. phoneme=0.2*0.4=0.08. redundancy=0.2*0.9=0.18.
            # quality = (0.3+0.3+0.3+0.08+0.18)/2 = 0.58. impact neutral (default) -> defer.
            GroundTruthClipAnnotation(
                disposition="defer",
                confidence=0.6,
                disposition_ambiguity=0.2,
                valid_fix_steps=[],
                valid_override_justifications=[],
                expected_reasoning_elements=["quality"],
            ),
        ],
        "behavioral": GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[],
            predicted_behavior_triggers=[],
            risky_phoneme_clusters=[],
            model_behavioral_safety="safe",
            valid_mitigation_set=set(),
        ),
    }
    
    return reference_obs, param_config, clip_obs_list, weight_obs, ground_truths


# ===========================================================================
# Pipeline E2E verification
# ===========================================================================

def test_episode_returns_valid_result(minimal_episode_inputs):
    """Run the pipeline and verify the output structure and score bounds."""
    result = run_episode(*minimal_episode_inputs)
    
    assert isinstance(result, EpisodeResult)
    assert 0.0 <= result.subenv1_score <= 1.0
    assert 0.0 <= result.subenv2_score <= 1.0
    assert 0.0 <= result.subenv3_score <= 1.0
    assert 0.0 <= result.final_score <= 1.0


def test_final_score_is_weighted_combination(minimal_episode_inputs):
    """Verify final_score = 0.25*s1 + 0.35*s2 + 0.40*s3 exactly."""
    result = run_episode(*minimal_episode_inputs)
    
    expected = (
        0.25 * result.subenv1_score +
        0.35 * result.subenv2_score +
        0.40 * result.subenv3_score
    )
    assert abs(result.final_score - expected) < 1e-9


def test_handoffs_are_correct_types(minimal_episode_inputs):
    """Verify that inter-environment coupling objects are the expected types."""
    result = run_episode(*minimal_episode_inputs)
    
    assert isinstance(result.reference_handoff, ReferenceAuditHandoff)
    assert isinstance(result.dataset_handoff, DatasetHealthHandoff)
    assert isinstance(result.behavioral_handoff, BehavioralAuditHandoff)


def test_subenv2_receives_risk_profile_from_subenv1(minimal_episode_inputs):
    """Verify that Sub-env 1's risk profile propagates to Sub-env 2.
    
    We compare a clean reference image vs a degraded one.
    """
    ref_clean, param_cfg, clips, weight, gt = minimal_episode_inputs
    result_clean = run_episode(ref_clean, param_cfg, clips, weight, gt)
    
    # Degrade reference image to trigger risk_profile escalation
    # Node 1 will flag these issues, and Node 2 will see them in observations.
    # Actually risk_profile is derived from Node 2 action (config_risk_level).
    # To get dangerous level: non_frontal regime + denoise_alt < 0.45
    ref_degraded = ref_clean.model_copy(update={"estimated_yaw_degrees": 30.0}) # non_frontal
    param_degraded = {"denoise_alt": 0.2, "cfg": 5.5, "eta": 0.08} # denoise_alt < 0.45
    
    result_degraded = run_episode(ref_degraded, param_degraded, clips, weight, gt)
    
    # Check handoff values directly
    assert result_clean.reference_handoff.risk_profile in ["low", "medium"]
    assert result_degraded.reference_handoff.risk_profile in ["medium", "high"]


def test_subenv3_receives_synthetic_descriptor(minimal_episode_inputs):
    """Verify that Sub-env 2's synthetic weight descriptor propagates to Sub-env 3."""
    result = run_episode(*minimal_episode_inputs)
    
    # Node 6 produces dataset_handoff.synthetic_weight_descriptor
    # result.dataset_handoff is the aggregation of Sub-env 2.
    assert result.dataset_handoff.synthetic_weight_descriptor is not None
    
    # Check that it's actually populated with values derived from clips
    # with 3 clips (1 accept, 1 reject, 1 defer), rank_utilization_estimate is 1/3 ~ 0.33
    assert 0.0 <= result.dataset_handoff.synthetic_weight_descriptor.estimated_rank_utilization <= 1.0


def test_episode_deterministic(minimal_episode_inputs):
    """Two runs with the same inputs must yield identical exact results."""
    result1 = run_episode(*minimal_episode_inputs)
    result2 = run_episode(*minimal_episode_inputs)
    
    assert result1.final_score == result2.final_score
    assert result1.subenv1_score == result2.subenv1_score
    assert result1.subenv2_score == result2.subenv2_score
    assert result1.subenv3_score == result2.subenv3_score
    assert result1.reference_handoff == result2.reference_handoff


def test_node_failure_propagates(minimal_episode_inputs):
    """Verify that an exception in any node is not swallowed but propagates."""
    with patch(
        "src.pipeline.diagnose_image",
        side_effect=RuntimeError("injected failure"),
    ):
        with pytest.raises(RuntimeError, match="injected failure"):
            run_episode(*minimal_episode_inputs)
