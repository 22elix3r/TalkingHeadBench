"""
Unit tests for all four schema modules:
  - src/schemas/subenv1
  - src/schemas/subenv2
  - src/schemas/subenv3
  - src/schemas/ground_truth

For each module:
  (a) instantiation tests — one per model using realistic dummy values
  (b) ValidationError tests — omit a required field, expect pydantic.ValidationError

Additional contract tests:
  - test_override_decision_rejects_bool
  - test_synthetic_weight_descriptor_typed
  - test_token_mapping_optional
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ── schema imports ────────────────────────────────────────────────────────────
from src.schemas.subenv1 import (
    DirectionalFix,
    ImageDiagnosticsAction,
    ImageDiagnosticsObservation,
    ParameterAnomaly,
    ParamAnomalyAction,
    ParamAnomalyObservation,
    ReferenceAuditHandoff,
)
from src.schemas.subenv2 import (
    ClipDispositionAction,
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
    DatasetHealthHandoff,
    SyntheticWeightDescriptor,
)
from src.schemas.subenv3 import (
    BehavioralAuditHandoff,
    BehaviorTriggerPrediction,
    LayerAnomalyFlag,
    MitigationRecommendation,
    PhonemeCluster,
    PhonemeRiskAction,
    PhonemeRiskEntry,
    PhonemeRiskObservation,
    TokenAnomalyFlag,
    WeightEvidenceDossier,
    WeightSignalObservation,
)
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)


# =============================================================================
# ── Helpers / shared fixtures ─────────────────────────────────────────────────
# =============================================================================

def _make_parameter_anomaly() -> ParameterAnomaly:
    return ParameterAnomaly(
        parameter="cfg",
        issue="CFG scale too high for this regime",
        severity="moderate",
        linked_failure_mode="identity_collapse",
    )


def _make_directional_fix() -> DirectionalFix:
    return DirectionalFix(
        target="cfg",
        direction="decrease",
        rationale="Reducing CFG prevents over-saturation of identity signal.",
        priority="recommended",
    )


def _make_clip_evidence_dossier() -> ClipEvidenceDossier:
    return ClipEvidenceDossier(
        clip_id="clip_001",
        identity_drift_severity="minor",
        temporal_instability_flag=False,
        lip_sync_quality="good",
        unique_phoneme_value=0.72,
        dataset_redundancy_score=0.15,
        estimated_training_impact="positive",
        primary_rejection_reason=None,
        evidence_summary="Clip passes all quality thresholds with minor identity drift.",
    )


def _make_token_anomaly_flag() -> TokenAnomalyFlag:
    return TokenAnomalyFlag(
        token_position=12,
        mapped_phoneme="AH",
        anomaly_type="excessive_influence",
        severity=0.85,
        evidence="Token 12 exhibits singular value 4.7× above layer mean.",
    )


def _make_layer_anomaly_flag() -> LayerAnomalyFlag:
    return LayerAnomalyFlag(
        layer_name="transformer.h.6.attn.c_attn",
        anomaly_type="norm_explosion",
        severity=0.91,
        evidence="Frobenius norm 12.3, expected < 3.0 for this rank.",
    )


def _make_phoneme_risk_entry() -> PhonemeRiskEntry:
    return PhonemeRiskEntry(
        phoneme="AH",
        risk_score=0.87,
        risk_type="identity_trigger",
        confidence=0.92,
        evidence="Phoneme AH mapped to token 12 with excessive singular value.",
    )


def _make_behavior_trigger_prediction() -> BehaviorTriggerPrediction:
    return BehaviorTriggerPrediction(
        trigger_phoneme="AH",
        triggered_behavior="smile",
        association_strength=0.78,
        is_intended=False,
        concern_level="high",
    )


def _make_phoneme_cluster() -> PhonemeCluster:
    return PhonemeCluster(
        phonemes=["AH", "AE", "AA"],
        cluster_risk_type="identity_trigger",
        combined_risk_score=0.82,
        interaction_description="Low-back vowels co-activate identity-anchored attention heads.",
    )


def _make_mitigation_recommendation() -> MitigationRecommendation:
    return MitigationRecommendation(
        target="AH",
        action="add_counter_examples",
        rationale="Adding diverse AH examples reduces over-association with identity.",
        priority="recommended",
    )


def _make_weight_evidence_dossier() -> WeightEvidenceDossier:
    return WeightEvidenceDossier(
        weight_file_id="lora_v1.safetensors",
        training_quality="unstable",
        rank_utilization_assessment="wasteful",
        high_entropy_token_flags=[_make_token_anomaly_flag()],
        layer_anomaly_flags=[_make_layer_anomaly_flag()],
        overall_behavioral_risk="high",
        evidence_summary="Two layers exhibit norm explosion; token 12 is high-entropy.",
    )


def _make_synthetic_weight_descriptor() -> SyntheticWeightDescriptor:
    return SyntheticWeightDescriptor(
        estimated_rank_utilization=0.65,
        suspected_overfitting_score=0.42,
        high_risk_phoneme_hints=["AH", "OW"],
        identity_consistency_estimate=0.88,
        expected_canonical_entropy_range=(0.3, 0.9),
    )


# =============================================================================
# ── Sub-env 1 tests ───────────────────────────────────────────────────────────
# =============================================================================


class TestSubenv1Instantiation:

    def test_image_diagnostics_observation(self):
        obs = ImageDiagnosticsObservation(
            face_occupancy_ratio=0.35,
            estimated_yaw_degrees=-12.5,
            estimated_pitch_degrees=3.2,
            background_complexity_score=0.41,
            lighting_uniformity_score=0.78,
            skin_tone_bucket=3,
            occlusion_detected=False,
            image_resolution=(512, 512),
            estimated_sharpness=0.82,
            prompt_token_count=28,
            prompt_semantic_density=6.0,
            conflicting_descriptors=[],
            identity_anchoring_strength=0.91,
        )
        assert isinstance(obs, ImageDiagnosticsObservation)

    def test_image_diagnostics_action(self):
        action = ImageDiagnosticsAction(
            regime_classification="frontal_simple",
            identified_risk_factors=["slight_overexposure"],
            prompt_issues=[],
            recommended_prompt_modifications=["Remove redundant lighting descriptor"],
            image_usability_score=0.88,
            reasoning="Face occupies 35% of frame with good sharpness.",
        )
        assert isinstance(action, ImageDiagnosticsAction)

    def test_parameter_anomaly(self):
        anomaly = _make_parameter_anomaly()
        assert isinstance(anomaly, ParameterAnomaly)

    def test_directional_fix(self):
        fix = _make_directional_fix()
        assert isinstance(fix, DirectionalFix)

    def test_param_anomaly_observation(self):
        obs = ParamAnomalyObservation(
            proposed_config={"cfg": 12.0, "denoise_alt": 0.1, "eta": 0.0},
            regime="frontal_simple",
            identified_risk_factors=["slight_overexposure"],
            image_usability_score=0.88,
            face_occupancy_ratio=0.35,
            estimated_yaw_degrees=-12.5,
            background_complexity_score=0.41,
            lighting_uniformity_score=0.78,
            occlusion_detected=False,
            prompt_identity_anchoring=0.91,
            prompt_token_count=28,
            conflicting_descriptors=[],
        )
        assert isinstance(obs, ParamAnomalyObservation)

    def test_param_anomaly_action(self):
        action = ParamAnomalyAction(
            config_risk_level="marginal",
            anomalies=[_make_parameter_anomaly()],
            predicted_failure_modes=["identity_collapse"],
            directional_fixes=[_make_directional_fix()],
            summary="CFG is elevated; recommend reducing to 7–8.",
        )
        assert isinstance(action, ParamAnomalyAction)

    def test_reference_audit_handoff(self):
        handoff = ReferenceAuditHandoff(
            image_usability_score=0.88,
            regime="frontal_simple",
            identified_risk_factors=["slight_overexposure"],
            config_quality_score=0.71,
            risk_profile="medium",
            estimated_drift_risk=0.32,
            prompt_strength=0.91,
            recommended_config={"cfg": 7.5, "denoise_alt": 0.15},
            subenv1_score=0.79,
        )
        assert isinstance(handoff, ReferenceAuditHandoff)


class TestSubenv1ValidationErrors:

    def test_image_diagnostics_observation_missing_field(self):
        with pytest.raises(ValidationError):
            ImageDiagnosticsObservation(
                # omit face_occupancy_ratio
                estimated_yaw_degrees=-12.5,
                estimated_pitch_degrees=3.2,
                background_complexity_score=0.41,
                lighting_uniformity_score=0.78,
                skin_tone_bucket=3,
                occlusion_detected=False,
                image_resolution=(512, 512),
                estimated_sharpness=0.82,
                prompt_token_count=28,
                prompt_semantic_density=6.0,
                conflicting_descriptors=[],
                identity_anchoring_strength=0.91,
            )

    def test_image_diagnostics_action_missing_field(self):
        with pytest.raises(ValidationError):
            ImageDiagnosticsAction(
                # omit regime_classification
                identified_risk_factors=[],
                prompt_issues=[],
                recommended_prompt_modifications=[],
                image_usability_score=0.88,
                reasoning="OK",
            )

    def test_parameter_anomaly_missing_field(self):
        with pytest.raises(ValidationError):
            ParameterAnomaly(
                parameter="cfg",
                issue="Too high",
                # omit severity
                linked_failure_mode="identity_collapse",
            )

    def test_directional_fix_missing_field(self):
        with pytest.raises(ValidationError):
            DirectionalFix(
                target="cfg",
                # omit direction
                rationale="Just because",
                priority="optional",
            )

    def test_param_anomaly_observation_missing_field(self):
        with pytest.raises(ValidationError):
            ParamAnomalyObservation(
                # omit proposed_config
                regime="frontal_simple",
                identified_risk_factors=[],
                image_usability_score=0.88,
                face_occupancy_ratio=0.35,
                estimated_yaw_degrees=-12.5,
                background_complexity_score=0.41,
                lighting_uniformity_score=0.78,
                occlusion_detected=False,
                prompt_identity_anchoring=0.91,
                prompt_token_count=28,
                conflicting_descriptors=[],
            )

    def test_param_anomaly_action_missing_field(self):
        with pytest.raises(ValidationError):
            ParamAnomalyAction(
                config_risk_level="safe",
                anomalies=[],
                predicted_failure_modes=[],
                directional_fixes=[],
                # omit summary
            )

    def test_reference_audit_handoff_missing_field(self):
        with pytest.raises(ValidationError):
            ReferenceAuditHandoff(
                image_usability_score=0.88,
                regime="frontal_simple",
                identified_risk_factors=[],
                config_quality_score=0.71,
                # omit risk_profile
                estimated_drift_risk=0.32,
                prompt_strength=0.91,
                recommended_config={},
                subenv1_score=0.79,
            )


# =============================================================================
# ── Sub-env 2 tests ───────────────────────────────────────────────────────────
# =============================================================================


class TestSubenv2Instantiation:

    def test_clip_signal_observation(self):
        obs = ClipSignalObservation(
            clip_id="clip_001",
            face_embedding_variance=0.012,
            landmark_stability_score=0.94,
            identity_cosine_drift=0.03,
            frame_difference_mean=4.2,
            optical_flow_magnitude=1.8,
            blink_count=3,
            lip_sync_confidence=0.87,
            phoneme_sequence=["AH", "B", "AW", "T"],
            phoneme_coverage_new=0.15,
            blur_score=0.76,
            exposure_score=0.71,
            occlusion_frames=2,
            clips_audited_so_far=47,
            current_phoneme_coverage={"AH": 5, "B": 3},
            current_pose_distribution={"frontal_simple": 30, "non_frontal": 8},
            similar_clips_accepted=4,
        )
        assert isinstance(obs, ClipSignalObservation)

    def test_clip_evidence_dossier(self):
        dossier = _make_clip_evidence_dossier()
        assert isinstance(dossier, ClipEvidenceDossier)

    def test_clip_disposition_observation(self):
        obs = ClipDispositionObservation(
            evidence_dossier=_make_clip_evidence_dossier(),
            minimum_clips_needed=80,
            phoneme_gap_severity={"OW": 0.8, "UH": 0.6},
            pose_gap_severity={"non_frontal": 0.7},
            budget_remaining=33,
            reference_risk_profile="medium",
            estimated_drift_risk=0.32,
            marginal_training_damage=0.05,
            marginal_coverage_gain=0.15,
        )
        assert isinstance(obs, ClipDispositionObservation)

    def test_clip_disposition_action(self):
        action = ClipDispositionAction(
            disposition="accept",
            confidence=0.91,
            rejection_reasons=None,
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason=None,
            dataset_impact_reasoning="Coverage gain outweighs minor identity drift.",
            override_decision="not_applicable",
            override_justification=None,
        )
        assert isinstance(action, ClipDispositionAction)

    def test_synthetic_weight_descriptor(self):
        desc = _make_synthetic_weight_descriptor()
        assert isinstance(desc, SyntheticWeightDescriptor)

    def test_dataset_health_handoff(self):
        handoff = DatasetHealthHandoff(
            accepted_clip_count=62,
            rejected_clip_count=11,
            fix_recommended_count=7,
            identity_consistency_score=0.86,
            phoneme_coverage_score=0.74,
            pose_diversity_score=0.61,
            overall_dataset_quality=0.77,
            suspected_anomalous_phonemes=["OW", "UH"],
            high_risk_clip_ids=["clip_014", "clip_031"],
            weight_contamination_estimate=0.18,
            synthetic_weight_descriptor=_make_synthetic_weight_descriptor(),
            subenv2_score=0.72,
        )
        assert isinstance(handoff, DatasetHealthHandoff)


class TestSubenv2ValidationErrors:

    def test_clip_signal_observation_missing_field(self):
        with pytest.raises(ValidationError):
            ClipSignalObservation(
                # omit clip_id
                face_embedding_variance=0.012,
                landmark_stability_score=0.94,
                identity_cosine_drift=0.03,
                frame_difference_mean=4.2,
                optical_flow_magnitude=1.8,
                blink_count=3,
                lip_sync_confidence=0.87,
                phoneme_sequence=[],
                phoneme_coverage_new=0.15,
                blur_score=0.76,
                exposure_score=0.71,
                occlusion_frames=2,
                clips_audited_so_far=47,
                current_phoneme_coverage={},
                current_pose_distribution={},
                similar_clips_accepted=4,
            )

    def test_clip_evidence_dossier_missing_field(self):
        with pytest.raises(ValidationError):
            ClipEvidenceDossier(
                clip_id="clip_001",
                # omit identity_drift_severity
                temporal_instability_flag=False,
                lip_sync_quality="good",
                unique_phoneme_value=0.72,
                dataset_redundancy_score=0.15,
                estimated_training_impact="positive",
                primary_rejection_reason=None,
                evidence_summary="OK",
            )

    def test_clip_disposition_observation_missing_field(self):
        with pytest.raises(ValidationError):
            ClipDispositionObservation(
                evidence_dossier=_make_clip_evidence_dossier(),
                # omit minimum_clips_needed
                phoneme_gap_severity={},
                pose_gap_severity={},
                budget_remaining=33,
                reference_risk_profile="low",
                estimated_drift_risk=0.1,
                marginal_training_damage=0.02,
                marginal_coverage_gain=0.08,
            )

    def test_clip_disposition_action_missing_field(self):
        with pytest.raises(ValidationError):
            ClipDispositionAction(
                disposition="accept",
                confidence=0.9,
                rejection_reasons=None,
                fix_instructions=None,
                estimated_fix_effort=None,
                defer_reason=None,
                # omit dataset_impact_reasoning
                override_decision="not_applicable",
                override_justification=None,
            )

    def test_synthetic_weight_descriptor_missing_field(self):
        with pytest.raises(ValidationError):
            SyntheticWeightDescriptor(
                estimated_rank_utilization=0.65,
                suspected_overfitting_score=0.42,
                high_risk_phoneme_hints=[],
                # omit identity_consistency_estimate
                expected_canonical_entropy_range=(0.3, 0.9),
            )

    def test_dataset_health_handoff_missing_field(self):
        with pytest.raises(ValidationError):
            DatasetHealthHandoff(
                accepted_clip_count=62,
                rejected_clip_count=11,
                fix_recommended_count=7,
                identity_consistency_score=0.86,
                phoneme_coverage_score=0.74,
                pose_diversity_score=0.61,
                overall_dataset_quality=0.77,
                suspected_anomalous_phonemes=[],
                high_risk_clip_ids=[],
                weight_contamination_estimate=0.18,
                # omit synthetic_weight_descriptor
                subenv2_score=0.72,
            )


# =============================================================================
# ── Sub-env 3 tests ───────────────────────────────────────────────────────────
# =============================================================================


class TestSubenv3Instantiation:

    def test_token_anomaly_flag(self):
        flag = _make_token_anomaly_flag()
        assert isinstance(flag, TokenAnomalyFlag)

    def test_layer_anomaly_flag(self):
        flag = _make_layer_anomaly_flag()
        assert isinstance(flag, LayerAnomalyFlag)

    def test_weight_signal_observation(self):
        obs = WeightSignalObservation(
            weight_file_id="lora_v1.safetensors",
            lora_rank=16,
            target_modules=["q_proj", "v_proj"],
            total_parameters=2_097_152,
            layer_norms={"transformer.h.6.attn.c_attn": 2.41},
            layer_sparsity={"transformer.h.6.attn.c_attn": 0.07},
            layer_rank_utilization={"transformer.h.6.attn.c_attn": 0.81},
            canonical_entropy_per_layer={"transformer.h.6.attn.c_attn": 3.12},
            high_entropy_token_positions=[12, 27],
            token_position_to_phoneme={12: "AH", 27: "OW"},
            canonical_output_norm_variance=0.034,
            canonical_dominant_directions=9,
            layer_correlation_matrix=[[1.0, 0.42], [0.42, 1.0]],
            attention_head_specialization={"head_0": 0.61, "head_1": 0.88},
            weight_magnitude_histogram=[0.1, 0.3, 0.4, 0.15, 0.05],
            gradient_noise_estimate=0.02,
            overfitting_signature=0.35,
            dataset_health_summary=_make_synthetic_weight_descriptor(),
            suspected_anomalous_phonemes=["OW"],
        )
        assert isinstance(obs, WeightSignalObservation)

    def test_weight_evidence_dossier(self):
        dossier = _make_weight_evidence_dossier()
        assert isinstance(dossier, WeightEvidenceDossier)

    def test_phoneme_risk_observation(self):
        obs = PhonemeRiskObservation(
            weight_evidence=_make_weight_evidence_dossier(),
            high_entropy_token_flags=[_make_token_anomaly_flag()],
            phoneme_vocabulary=["AH", "AE", "B", "OW", "UH"],
            phoneme_to_token_indices={"AH": [12], "OW": [27]},
            phoneme_entropy_scores={"AH": 3.12, "OW": 2.87},
            phoneme_influence_scores={"AH": 0.88, "OW": 0.72},
            phoneme_cooccurrence_anomalies=[("AH", "OW", 0.91)],
            behavior_vocabulary=["smile", "blink", "head_turn"],
            training_data_phoneme_distribution={"AH": 120, "OW": 80},
            suspected_anomalous_phonemes_from_subenv2=["OW"],
        )
        assert isinstance(obs, PhonemeRiskObservation)

    def test_phoneme_risk_entry(self):
        entry = _make_phoneme_risk_entry()
        assert isinstance(entry, PhonemeRiskEntry)

    def test_behavior_trigger_prediction(self):
        pred = _make_behavior_trigger_prediction()
        assert isinstance(pred, BehaviorTriggerPrediction)

    def test_phoneme_cluster(self):
        cluster = _make_phoneme_cluster()
        assert isinstance(cluster, PhonemeCluster)

    def test_mitigation_recommendation(self):
        rec = _make_mitigation_recommendation()
        assert isinstance(rec, MitigationRecommendation)

    def test_phoneme_risk_action(self):
        action = PhonemeRiskAction(
            phoneme_risk_ranking=[_make_phoneme_risk_entry()],
            predicted_behavior_triggers=[_make_behavior_trigger_prediction()],
            risky_phoneme_clusters=[_make_phoneme_cluster()],
            model_behavioral_safety="moderate_risk",
            mitigation_recommendations=[_make_mitigation_recommendation()],
            summary="AH and OW tokens exhibit elevated singular values linked to identity triggers.",
        )
        assert isinstance(action, PhonemeRiskAction)

    def test_behavioral_audit_handoff(self):
        handoff = BehavioralAuditHandoff(
            weight_file_id="lora_v1.safetensors",
            phoneme_risk_ranking=[_make_phoneme_risk_entry()],
            predicted_behavior_triggers=[_make_behavior_trigger_prediction()],
            risky_phoneme_clusters=[_make_phoneme_cluster()],
            model_behavioral_safety="moderate_risk",
            mitigation_recommendations=[_make_mitigation_recommendation()],
            ranking_quality_score=0.81,
            trigger_prediction_score=0.76,
            cluster_identification_score=0.88,
            safety_calibration_score=0.74,
            mitigation_quality_score=0.79,
            subenv3_score=0.80,
        )
        assert isinstance(handoff, BehavioralAuditHandoff)


class TestSubenv3ValidationErrors:

    def test_token_anomaly_flag_missing_field(self):
        with pytest.raises(ValidationError):
            TokenAnomalyFlag(
                token_position=12,
                mapped_phoneme="AH",
                # omit anomaly_type
                severity=0.85,
                evidence="Some evidence",
            )

    def test_layer_anomaly_flag_missing_field(self):
        with pytest.raises(ValidationError):
            LayerAnomalyFlag(
                layer_name="transformer.h.6",
                # omit anomaly_type
                severity=0.91,
                evidence="Norm explosion",
            )

    def test_weight_signal_observation_missing_field(self):
        with pytest.raises(ValidationError):
            WeightSignalObservation(
                weight_file_id="lora_v1.safetensors",
                # omit lora_rank
                target_modules=["q_proj"],
                total_parameters=1024,
                layer_norms={},
                layer_sparsity={},
                layer_rank_utilization={},
                canonical_entropy_per_layer={},
                high_entropy_token_positions=[],
                token_position_to_phoneme=None,
                canonical_output_norm_variance=0.01,
                canonical_dominant_directions=8,
                layer_correlation_matrix=[[1.0]],
                attention_head_specialization={},
                weight_magnitude_histogram=[],
                gradient_noise_estimate=0.01,
                overfitting_signature=0.1,
                dataset_health_summary=None,
                suspected_anomalous_phonemes=None,
            )

    def test_weight_evidence_dossier_missing_field(self):
        with pytest.raises(ValidationError):
            WeightEvidenceDossier(
                weight_file_id="lora_v1.safetensors",
                training_quality="healthy",
                rank_utilization_assessment="efficient",
                high_entropy_token_flags=[],
                layer_anomaly_flags=[],
                # omit overall_behavioral_risk
                evidence_summary="All clear",
            )

    def test_phoneme_risk_observation_missing_field(self):
        with pytest.raises(ValidationError):
            PhonemeRiskObservation(
                weight_evidence=_make_weight_evidence_dossier(),
                high_entropy_token_flags=[],
                # omit phoneme_vocabulary
                phoneme_to_token_indices={},
                phoneme_entropy_scores={},
                phoneme_influence_scores={},
                phoneme_cooccurrence_anomalies=[],
                behavior_vocabulary=[],
                training_data_phoneme_distribution=None,
                suspected_anomalous_phonemes_from_subenv2=None,
            )

    def test_phoneme_risk_entry_missing_field(self):
        with pytest.raises(ValidationError):
            PhonemeRiskEntry(
                phoneme="AH",
                risk_score=0.87,
                # omit risk_type
                confidence=0.92,
                evidence="High singular value",
            )

    def test_phoneme_risk_action_missing_field(self):
        with pytest.raises(ValidationError):
            PhonemeRiskAction(
                phoneme_risk_ranking=[],
                predicted_behavior_triggers=[],
                risky_phoneme_clusters=[],
                model_behavioral_safety="safe",
                mitigation_recommendations=[],
                # omit summary
            )

    def test_behavioral_audit_handoff_missing_field(self):
        with pytest.raises(ValidationError):
            BehavioralAuditHandoff(
                weight_file_id="lora_v1.safetensors",
                phoneme_risk_ranking=[],
                predicted_behavior_triggers=[],
                risky_phoneme_clusters=[],
                model_behavioral_safety="safe",
                mitigation_recommendations=[],
                ranking_quality_score=0.81,
                trigger_prediction_score=0.76,
                cluster_identification_score=0.88,
                safety_calibration_score=0.74,
                mitigation_quality_score=0.79,
                # omit subenv3_score
            )


# =============================================================================
# ── Ground truth tests ────────────────────────────────────────────────────────
# =============================================================================


class TestGroundTruthInstantiation:

    def test_ground_truth_image_annotation(self):
        ann = GroundTruthImageAnnotation(
            regime_classification="frontal_simple",
            acceptable_regimes=["non_frontal"],
            identified_risk_factors=["slight_overexposure"],
            valid_prompt_modifications=["Remove redundant lighting descriptor"],
        )
        assert isinstance(ann, GroundTruthImageAnnotation)

    def test_ground_truth_param_annotation(self):
        ann = GroundTruthParamAnnotation(
            config_risk_level="marginal",
            anomalies=[_make_parameter_anomaly()],
            predicted_failure_modes=["identity_collapse"],
            valid_fix_directions=[_make_directional_fix()],
        )
        assert isinstance(ann, GroundTruthParamAnnotation)

    def test_ground_truth_clip_annotation(self):
        ann = GroundTruthClipAnnotation(
            disposition="accept",
            confidence=0.91,
            disposition_ambiguity=0.1,
            valid_fix_steps=["Trim leading silent frames"],
            valid_override_justifications=["Phoneme gap too severe to reject"],
            expected_reasoning_elements=["phoneme_coverage", "identity_drift"],
        )
        assert isinstance(ann, GroundTruthClipAnnotation)

    def test_ground_truth_behavioral_annotation(self):
        ann = GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[_make_phoneme_risk_entry()],
            predicted_behavior_triggers=[_make_behavior_trigger_prediction()],
            risky_phoneme_clusters=[_make_phoneme_cluster()],
            model_behavioral_safety="moderate_risk",
            valid_mitigation_set={("AH", "add_counter_examples")},
        )
        assert isinstance(ann, GroundTruthBehavioralAnnotation)


class TestGroundTruthValidationErrors:

    def test_ground_truth_image_annotation_missing_field(self):
        with pytest.raises(ValidationError):
            GroundTruthImageAnnotation(
                regime_classification="frontal_simple",
                acceptable_regimes=[],
                # omit identified_risk_factors
                valid_prompt_modifications=[],
            )

    def test_ground_truth_param_annotation_missing_field(self):
        with pytest.raises(ValidationError):
            GroundTruthParamAnnotation(
                # omit config_risk_level
                anomalies=[],
                predicted_failure_modes=[],
                valid_fix_directions=[],
            )

    def test_ground_truth_clip_annotation_missing_field(self):
        with pytest.raises(ValidationError):
            GroundTruthClipAnnotation(
                disposition="accept",
                confidence=0.91,
                # omit disposition_ambiguity
                valid_fix_steps=[],
                valid_override_justifications=[],
                expected_reasoning_elements=[],
            )

    def test_ground_truth_behavioral_annotation_missing_field(self):
        with pytest.raises(ValidationError):
            GroundTruthBehavioralAnnotation(
                phoneme_risk_ranking=[],
                predicted_behavior_triggers=[],
                risky_phoneme_clusters=[],
                # omit model_behavioral_safety
                valid_mitigation_set=set(),
            )


# =============================================================================
# ── Additional contract tests ─────────────────────────────────────────────────
# =============================================================================


def test_override_decision_rejects_bool():
    """ClipDispositionAction.override_decision must be a 3-way Literal, not bool."""
    with pytest.raises(ValidationError):
        ClipDispositionAction(
            disposition="accept",
            confidence=0.9,
            rejection_reasons=None,
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason=None,
            dataset_impact_reasoning="Coverage gain outweighs risk.",
            override_decision=True,          # ← must be rejected
            override_justification=None,
        )


def test_synthetic_weight_descriptor_typed():
    """DatasetHealthHandoff accepts a SyntheticWeightDescriptor instance and
    coerces a valid dict into one (Pydantic v2 default behavior), but rejects
    a dict that is missing a required field of SyntheticWeightDescriptor."""

    common_fields = dict(
        accepted_clip_count=62,
        rejected_clip_count=11,
        fix_recommended_count=7,
        identity_consistency_score=0.86,
        phoneme_coverage_score=0.74,
        pose_diversity_score=0.61,
        overall_dataset_quality=0.77,
        suspected_anomalous_phonemes=[],
        high_risk_clip_ids=[],
        weight_contamination_estimate=0.18,
        subenv2_score=0.72,
    )

    # Typed instance must be accepted and preserved as the correct type.
    handoff = DatasetHealthHandoff(
        **common_fields,
        synthetic_weight_descriptor=_make_synthetic_weight_descriptor(),
    )
    assert isinstance(handoff.synthetic_weight_descriptor, SyntheticWeightDescriptor)

    # Pydantic v2 coerces a valid dict into the nested model — the result must
    # still be a SyntheticWeightDescriptor (not a raw dict).
    handoff_from_dict = DatasetHealthHandoff(
        **common_fields,
        synthetic_weight_descriptor={
            "estimated_rank_utilization": 0.65,
            "suspected_overfitting_score": 0.42,
            "high_risk_phoneme_hints": [],
            "identity_consistency_estimate": 0.88,
            "expected_canonical_entropy_range": (0.3, 0.9),
        },
    )
    assert isinstance(handoff_from_dict.synthetic_weight_descriptor, SyntheticWeightDescriptor)

    # A dict that is missing a required field of SyntheticWeightDescriptor must
    # still raise ValidationError (the nested schema is enforced).
    with pytest.raises(ValidationError):
        DatasetHealthHandoff(
            **common_fields,
            synthetic_weight_descriptor={
                # omit estimated_rank_utilization — required field
                "suspected_overfitting_score": 0.42,
                "high_risk_phoneme_hints": [],
                "identity_consistency_estimate": 0.88,
                "expected_canonical_entropy_range": (0.3, 0.9),
            },
        )


def test_token_mapping_optional():
    """WeightSignalObservation accepts token_position_to_phoneme=None and
    stores None without error.  The field is Optional so None is valid; it
    must be passed explicitly because Pydantic does not supply a default."""
    obs = WeightSignalObservation(
        weight_file_id="lora_v1.safetensors",
        lora_rank=16,
        target_modules=["q_proj"],
        total_parameters=1_048_576,
        layer_norms={"transformer.h.0.attn.c_attn": 1.93},
        layer_sparsity={"transformer.h.0.attn.c_attn": 0.04},
        layer_rank_utilization={"transformer.h.0.attn.c_attn": 0.78},
        canonical_entropy_per_layer={"transformer.h.0.attn.c_attn": 2.84},
        high_entropy_token_positions=[],
        token_position_to_phoneme=None,     # explicit None — no audio config available
        canonical_output_norm_variance=0.021,
        canonical_dominant_directions=7,
        layer_correlation_matrix=[[1.0]],
        attention_head_specialization={"head_0": 0.55},
        weight_magnitude_histogram=[0.2, 0.5, 0.2, 0.07, 0.03],
        gradient_noise_estimate=0.015,
        overfitting_signature=0.28,
        dataset_health_summary=None,
        suspected_anomalous_phonemes=None,
    )
    assert obs.token_position_to_phoneme is None
