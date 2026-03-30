"""Smoke tests for the legacy pipeline bundle entry-point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.pipeline import (
    EpisodeResult,
    _assess_weight_evidence,
    _heuristic_clip_evidence_dossier,
    run_episode_from_bundle,
)
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import ImageDiagnosticsObservation
from src.schemas.subenv2 import ClipSignalObservation
from src.schemas.subenv3 import WeightSignalObservation


def test_run_episode_from_bundle_returns_episode_result():
    """Legacy bundle API should return EpisodeResult with bounded final_score."""
    image_obs = ImageDiagnosticsObservation(
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

    mocked_clip_obs = ClipSignalObservation(
        clip_id="clip_001",
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
    )

    mocked_weight_obs = WeightSignalObservation(
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

    bundle = {
        "image_obs": image_obs,
        "proposed_config": {"denoise_alt": 0.5, "cfg": 5.5, "eta": 0.08},
        "clips": [
            {
                "path": Path("dummy_clip.mp4"),
                "dataset_context": {
                    "minimum_clips_needed": 20,
                    "phoneme_gap_severity": {},
                    "pose_gap_severity": {},
                    "budget_remaining": 10,
                    "marginal_training_damage": 0.0,
                    "marginal_coverage_gain": 0.5,
                },
                "aligner_output": None,
            }
        ],
        "weight_path": Path("dummy_lora.safetensors"),
        "phoneme_obs_context": {
            "phoneme_vocabulary": ["AH", "EE", "OW"],
            "phoneme_to_token_indices": {"AH": [0], "EE": [1], "OW": [2]},
            "phoneme_entropy_scores": {"AH": 0.2, "EE": 0.3, "OW": 0.4},
            "phoneme_influence_scores": {"AH": 0.1, "EE": 0.2, "OW": 0.3},
            "phoneme_cooccurrence_anomalies": [],
            "behavior_vocabulary": ["smile", "blink"],
        },
        "agents": {
            "node1": diagnose_image,
            "node2": detect_param_anomalies,
            "node4": _heuristic_clip_evidence_dossier,
            "node5": recommend_clip_disposition,
            "node7": _assess_weight_evidence,
            "node8": assess_phoneme_risk,
        },
        "ground_truth": {
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
                    expected_reasoning_elements=[],
                )
            ],
            "behavioral": GroundTruthBehavioralAnnotation(
                phoneme_risk_ranking=[],
                predicted_behavior_triggers=[],
                risky_phoneme_clusters=[],
                model_behavioral_safety="safe",
                valid_mitigation_set=set(),
            ),
        },
    }

    with patch("src.pipeline.extract_clip_signals", return_value=mocked_clip_obs), patch(
        "src.pipeline.extract_weight_signals", return_value=mocked_weight_obs
    ):
        result = run_episode_from_bundle(bundle)

    assert isinstance(result, EpisodeResult)
    assert 0.0 <= result.final_score <= 1.0
