"""
Schema roundtrip smoke tests.

For every Pydantic model across all four schema files, verifies that:
  1. model.model_dump() → ModelClass(**data) reconstructs an equal object.
  2. model.model_dump_json() is valid JSON and round-trips via
     ModelClass.model_validate_json(json_str).
  3. Invalid Literal values are rejected with ValidationError.
  4. Optional fields default to None when absent.
  5. Nested model coercion works as intended.
  6. The set[tuple] field in GroundTruthBehavioralAnnotation survives a
     dump/load cycle.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

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
    BehaviorTriggerPrediction,
    BehavioralAuditHandoff,
    LayerAnomalyFlag,
    MitigationRecommendation,
    PhonemeCluster,
    PhonemeRiskAction,
    PhonemeRiskEntry,
    PhonemeRiskObservation,
    TokenAnomalyFlag,
    WeightEvidenceDossier,
)
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)


# ---------------------------------------------------------------------------
# Reusable roundtrip helper
# ---------------------------------------------------------------------------

def _roundtrip(model):
    """Run both dict-roundtrip and JSON-roundtrip, assert equality."""
    cls = type(model)

    # Dict roundtrip
    data = model.model_dump()
    model2 = cls(**data)
    assert model == model2, f"{cls.__name__} dict roundtrip failed"

    # JSON roundtrip
    json_str = model.model_dump_json()
    assert json.loads(json_str) is not None  # parseable JSON
    model3 = cls.model_validate_json(json_str)
    assert model == model3, f"{cls.__name__} JSON roundtrip failed"


# ---------------------------------------------------------------------------
# Shared sub-objects (used across multiple tests)
# ---------------------------------------------------------------------------

_anomaly = ParameterAnomaly(
    parameter="cfg",
    issue="too high",
    severity="moderate",
    linked_failure_mode="background_bleed",
)

_fix = DirectionalFix(
    target="guidance_scale",
    direction="decrease",
    rationale="prevents attention bleed",
    priority="recommended",
)

_token_flag = TokenAnomalyFlag(
    token_position=3,
    mapped_phoneme="EE",
    anomaly_type="excessive_influence",
    severity=0.75,
    evidence="High Vt row norm at position 3",
)

_layer_flag = LayerAnomalyFlag(
    layer_name="q_proj",
    anomaly_type="sparsity_anomaly",
    severity=0.6,
    evidence="Sparsity 0.65 — majority near zero",
)

_phoneme_entry = PhonemeRiskEntry(
    phoneme="EE",
    risk_score=0.72,
    risk_type="expression_trigger",
    confidence=0.8,
    evidence="canonical entropy 0.85, influence score 0.65",
)

_behavior_trigger = BehaviorTriggerPrediction(
    trigger_phoneme="EE",
    triggered_behavior="smile",
    association_strength=0.72,
    is_intended=False,
    concern_level="medium",
)

_cluster = PhonemeCluster(
    phonemes=["EE", "IY"],
    cluster_risk_type="expression_trigger",
    combined_risk_score=0.71,
    interaction_description="Correlated canonical representation",
)

_mitigation = MitigationRecommendation(
    target="expression_trigger cluster: ['EE', 'IY']",
    action="add_counter_examples",
    rationale="add neutral expression clips",
    priority="recommended",
)

_synthetic_descriptor = SyntheticWeightDescriptor(
    estimated_rank_utilization=0.8,
    suspected_overfitting_score=0.2,
    high_risk_phoneme_hints=["OW"],
    identity_consistency_estimate=0.9,
    expected_canonical_entropy_range=(0.5, 1.8),
)

_clip_dossier = ClipEvidenceDossier(
    clip_id="clip_A",
    identity_drift_severity="none",
    temporal_instability_flag=False,
    lip_sync_quality="good",
    unique_phoneme_value=0.7,
    dataset_redundancy_score=0.2,
    estimated_training_impact="positive",
    primary_rejection_reason=None,
    evidence_summary="Clean clip",
)

_weight_dossier = WeightEvidenceDossier(
    weight_file_id="lora_v1",
    training_quality="healthy",
    rank_utilization_assessment="efficient",
    high_entropy_token_flags=[_token_flag],
    layer_anomaly_flags=[_layer_flag],
    overall_behavioral_risk="low",
    evidence_summary="Healthy weights",
)

_gt_image = GroundTruthImageAnnotation(
    regime_classification="frontal_simple",
    acceptable_regimes=["non_frontal"],
    identified_risk_factors=["low_sharpness"],
    valid_prompt_modifications=["add_frontal_constraint"],
)

_gt_param = GroundTruthParamAnnotation(
    config_risk_level="marginal",
    anomalies=[_anomaly],
    predicted_failure_modes=["background_bleed"],
    valid_fix_directions=[_fix],
)

_gt_clip = GroundTruthClipAnnotation(
    disposition="accept",
    confidence=0.85,
    disposition_ambiguity=0.1,
    valid_fix_steps=["trim_frames_0_5"],
    valid_override_justifications=["acceptable despite minor drift"],
    expected_reasoning_elements=["stable", "good sync"],
)

_gt_behavioral = GroundTruthBehavioralAnnotation(
    phoneme_risk_ranking=[_phoneme_entry],
    predicted_behavior_triggers=[_behavior_trigger],
    risky_phoneme_clusters=[_cluster],
    model_behavioral_safety="minor_concerns",
    valid_mitigation_set={("expression_trigger cluster: ['EE', 'IY']", "add_counter_examples")},
)


# ===========================================================================
# 1. Sub-env 1 roundtrips
# ===========================================================================

class TestSubenv1Roundtrip:

    def test_image_diagnostics_observation(self):
        model = ImageDiagnosticsObservation(
            face_occupancy_ratio=0.6,
            estimated_yaw_degrees=5.0,
            estimated_pitch_degrees=2.0,
            background_complexity_score=0.3,
            lighting_uniformity_score=0.7,
            skin_tone_bucket=3,
            occlusion_detected=False,
            image_resolution=(1280, 720),
            estimated_sharpness=0.75,
            prompt_token_count=40,
            prompt_semantic_density=0.5,
            conflicting_descriptors=[],
            identity_anchoring_strength=0.8,
        )
        _roundtrip(model)

    def test_image_diagnostics_action(self):
        model = ImageDiagnosticsAction(
            regime_classification="frontal_simple",
            identified_risk_factors=["low_sharpness"],
            prompt_issues=[],
            recommended_prompt_modifications=["add_frontal_constraint"],
            image_usability_score=0.75,
            reasoning="Face is clear and frontal.",
        )
        _roundtrip(model)

    def test_parameter_anomaly(self):
        _roundtrip(_anomaly)

    def test_directional_fix(self):
        _roundtrip(_fix)

    def test_param_anomaly_observation(self):
        model = ParamAnomalyObservation(
            proposed_config={"cfg": 8.0},
            regime="frontal_simple",
            identified_risk_factors=[],
            image_usability_score=0.75,
            face_occupancy_ratio=0.6,
            estimated_yaw_degrees=5.0,
            background_complexity_score=0.3,
            lighting_uniformity_score=0.7,
            occlusion_detected=False,
            prompt_identity_anchoring=0.8,
            prompt_token_count=40,
            conflicting_descriptors=[],
        )
        _roundtrip(model)

    def test_param_anomaly_action(self):
        model = ParamAnomalyAction(
            config_risk_level="marginal",
            anomalies=[_anomaly],
            predicted_failure_modes=["background_bleed"],
            directional_fixes=[_fix],
            summary="One moderate anomaly detected.",
        )
        _roundtrip(model)

    def test_reference_audit_handoff(self):
        model = ReferenceAuditHandoff(
            image_usability_score=0.78,
            regime="frontal_simple",
            identified_risk_factors=[],
            config_quality_score=0.75,
            risk_profile="low",
            estimated_drift_risk=0.0,
            prompt_strength=1.0,
            recommended_config={},
            subenv1_score=0.85,
        )
        _roundtrip(model)


# ===========================================================================
# 2. Sub-env 2 roundtrips
# ===========================================================================

class TestSubenv2Roundtrip:

    def test_clip_signal_observation(self):
        model = ClipSignalObservation(
            clip_id="clip_A",
            face_embedding_variance=0.01,
            landmark_stability_score=0.01,
            identity_cosine_drift=0.02,
            frame_difference_mean=5.0,
            optical_flow_magnitude=1.0,
            blink_count=2,
            lip_sync_confidence=0.9,
            phoneme_sequence=["AH", "EE"],
            phoneme_coverage_new=0.6,
            blur_score=0.1,
            exposure_score=0.7,
            occlusion_frames=0,
            clips_audited_so_far=5,
            current_phoneme_coverage={"AH": 3},
            current_pose_distribution={"frontal_simple": 4},
            similar_clips_accepted=1,
        )
        _roundtrip(model)

    def test_clip_evidence_dossier(self):
        _roundtrip(_clip_dossier)

    def test_clip_evidence_dossier_with_rejection_reason(self):
        model = ClipEvidenceDossier(
            clip_id="clip_B",
            identity_drift_severity="severe",
            temporal_instability_flag=True,
            lip_sync_quality="poor",
            unique_phoneme_value=0.1,
            dataset_redundancy_score=0.9,
            estimated_training_impact="negative",
            primary_rejection_reason="Drift too high",
            evidence_summary="Rejected clip",
        )
        _roundtrip(model)

    def test_clip_disposition_observation(self):
        model = ClipDispositionObservation(
            evidence_dossier=_clip_dossier,
            minimum_clips_needed=20,
            phoneme_gap_severity={"ZH": 2},
            pose_gap_severity={},
            budget_remaining=10,
            reference_risk_profile="low",
            estimated_drift_risk=0.1,
            marginal_training_damage=0.05,
            marginal_coverage_gain=0.6,
        )
        _roundtrip(model)

    def test_clip_disposition_action_accept(self):
        model = ClipDispositionAction(
            disposition="accept",
            confidence=0.9,
            rejection_reasons=None,
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason=None,
            dataset_impact_reasoning="Wide phoneme coverage gain.",
            override_decision="not_applicable",
            override_justification=None,
        )
        _roundtrip(model)

    def test_clip_disposition_action_fix(self):
        model = ClipDispositionAction(
            disposition="fix",
            confidence=0.45,
            rejection_reasons=None,
            fix_instructions=["trim frames 0–3"],
            estimated_fix_effort="trivial",
            defer_reason=None,
            dataset_impact_reasoning="Valuable phonemes but unstable.",
            override_decision="not_applicable",
            override_justification=None,
        )
        _roundtrip(model)

    def test_clip_disposition_action_reject(self):
        model = ClipDispositionAction(
            disposition="reject",
            confidence=0.05,
            rejection_reasons=["severe drift"],
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason=None,
            dataset_impact_reasoning="No salvageable content.",
            override_decision="not_applicable",
            override_justification=None,
        )
        _roundtrip(model)

    def test_clip_disposition_action_defer(self):
        model = ClipDispositionAction(
            disposition="defer",
            confidence=0.55,
            rejection_reasons=None,
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason="quality borderline (0.55) — manual review recommended",
            dataset_impact_reasoning="Borderline quality.",
            override_decision="not_applicable",
            override_justification=None,
        )
        _roundtrip(model)

    def test_synthetic_weight_descriptor(self):
        _roundtrip(_synthetic_descriptor)

    def test_dataset_health_handoff(self):
        model = DatasetHealthHandoff(
            accepted_clip_count=2,
            rejected_clip_count=1,
            fix_recommended_count=0,
            identity_consistency_score=0.9,
            phoneme_coverage_score=0.6,
            pose_diversity_score=0.4,
            overall_dataset_quality=0.72,
            suspected_anomalous_phonemes=["OW"],
            high_risk_clip_ids=[],
            weight_contamination_estimate=0.33,
            synthetic_weight_descriptor=_synthetic_descriptor,
            subenv2_score=0.78,
        )
        _roundtrip(model)


# ===========================================================================
# 3. Sub-env 3 roundtrips
# ===========================================================================

class TestSubenv3Roundtrip:

    def test_token_anomaly_flag(self):
        _roundtrip(_token_flag)

    def test_layer_anomaly_flag(self):
        _roundtrip(_layer_flag)

    def test_weight_evidence_dossier(self):
        _roundtrip(_weight_dossier)

    def test_phoneme_risk_observation(self):
        model = PhonemeRiskObservation(
            weight_evidence=_weight_dossier,
            high_entropy_token_flags=[_token_flag],
            phoneme_vocabulary=["AH", "EE", "OW"],
            phoneme_to_token_indices={"EE": [0], "OW": [1]},
            phoneme_entropy_scores={"EE": 0.85},
            phoneme_influence_scores={"EE": 0.7},
            phoneme_cooccurrence_anomalies=[("EE", "OW", 0.8)],
            behavior_vocabulary=["smile", "jaw_drift"],
            training_data_phoneme_distribution={"AH": 10, "EE": 5},
            suspected_anomalous_phonemes_from_subenv2=["OW"],
        )
        _roundtrip(model)

    def test_phoneme_risk_entry(self):
        _roundtrip(_phoneme_entry)

    def test_behavior_trigger_prediction(self):
        _roundtrip(_behavior_trigger)

    def test_phoneme_cluster(self):
        _roundtrip(_cluster)

    def test_mitigation_recommendation(self):
        _roundtrip(_mitigation)

    def test_phoneme_risk_action(self):
        model = PhonemeRiskAction(
            phoneme_risk_ranking=[_phoneme_entry],
            predicted_behavior_triggers=[_behavior_trigger],
            risky_phoneme_clusters=[_cluster],
            model_behavioral_safety="minor_concerns",
            mitigation_recommendations=[_mitigation],
            summary="Behavioral safety: minor_concerns. 1 phoneme flagged.",
        )
        _roundtrip(model)

    def test_behavioral_audit_handoff(self):
        model = BehavioralAuditHandoff(
            weight_file_id="lora_v1",
            phoneme_risk_ranking=[_phoneme_entry],
            predicted_behavior_triggers=[_behavior_trigger],
            risky_phoneme_clusters=[_cluster],
            model_behavioral_safety="minor_concerns",
            mitigation_recommendations=[_mitigation],
            ranking_quality_score=0.8,
            trigger_prediction_score=0.75,
            cluster_identification_score=1.0,
            safety_calibration_score=0.9,
            mitigation_quality_score=0.7,
            subenv3_score=0.82,
        )
        _roundtrip(model)


# ===========================================================================
# 4. Ground truth roundtrips
# ===========================================================================

class TestGroundTruthRoundtrip:

    def test_ground_truth_image_annotation(self):
        _roundtrip(_gt_image)

    def test_ground_truth_param_annotation(self):
        _roundtrip(_gt_param)

    def test_ground_truth_clip_annotation(self):
        _roundtrip(_gt_clip)

    def test_ground_truth_behavioral_annotation(self):
        """set[tuple] field needs special handling: model_dump serialises
        the set as a list; model_validate_json must reconstruct the set."""
        model = _gt_behavioral

        # Dict roundtrip
        data = model.model_dump()
        model2 = GroundTruthBehavioralAnnotation(**data)
        assert model2.valid_mitigation_set == model.valid_mitigation_set

        # JSON roundtrip — set is serialised as a list by Pydantic v2
        json_str = model.model_dump_json()
        model3 = GroundTruthBehavioralAnnotation.model_validate_json(json_str)
        assert model3.valid_mitigation_set == model.valid_mitigation_set


# ===========================================================================
# 5. Invalid Literal values are rejected
# ===========================================================================

class TestLiteralValuesRejected:

    def test_param_anomaly_action_invalid_risk_level(self):
        with pytest.raises(ValidationError):
            ParamAnomalyAction(
                config_risk_level="unknown_level",
                anomalies=[],
                predicted_failure_modes=[],
                directional_fixes=[],
                summary="test",
            )

    def test_clip_disposition_action_invalid_override(self):
        with pytest.raises(ValidationError):
            ClipDispositionAction(
                disposition="accept",
                confidence=0.9,
                rejection_reasons=None,
                fix_instructions=None,
                estimated_fix_effort=None,
                defer_reason=None,
                dataset_impact_reasoning="test",
                override_decision="yes",   # invalid
                override_justification=None,
            )

    def test_phoneme_risk_action_invalid_safety(self):
        with pytest.raises(ValidationError):
            PhonemeRiskAction(
                phoneme_risk_ranking=[],
                predicted_behavior_triggers=[],
                risky_phoneme_clusters=[],
                model_behavioral_safety="very_risky",  # invalid
                mitigation_recommendations=[],
                summary="test",
            )

    def test_parameter_anomaly_invalid_severity(self):
        with pytest.raises(ValidationError):
            ParameterAnomaly(
                parameter="cfg",
                issue="test",
                severity="catastrophic",  # invalid
                linked_failure_mode="background_bleed",
            )

    def test_directional_fix_invalid_direction(self):
        with pytest.raises(ValidationError):
            DirectionalFix(
                target="cfg",
                direction="maybe",  # invalid
                rationale="test",
                priority="recommended",
            )


# ===========================================================================
# 6. Optional fields default to None
# ===========================================================================

class TestOptionalFieldsDefaultToNone:

    def test_clip_disposition_action_accept_no_optionals(self):
        """Accept disposition with no optional fields set → all None."""
        action = ClipDispositionAction(
            disposition="accept",
            confidence=0.9,
            rejection_reasons=None,
            fix_instructions=None,
            estimated_fix_effort=None,
            defer_reason=None,
            dataset_impact_reasoning="Good clip.",
            override_decision="not_applicable",
            override_justification=None,
        )
        assert action.rejection_reasons is None
        assert action.fix_instructions is None
        assert action.estimated_fix_effort is None
        assert action.defer_reason is None
        assert action.override_justification is None

    def test_phoneme_risk_observation_optional_fields(self):
        """training_data_phoneme_distribution and suspected_anomalous_phonemes
        are Optional → accept None."""
        obs = PhonemeRiskObservation(
            weight_evidence=_weight_dossier,
            high_entropy_token_flags=[],
            phoneme_vocabulary=["AH"],
            phoneme_to_token_indices={},
            phoneme_entropy_scores={},
            phoneme_influence_scores={},
            phoneme_cooccurrence_anomalies=[],
            behavior_vocabulary=["smile"],
            training_data_phoneme_distribution=None,
            suspected_anomalous_phonemes_from_subenv2=None,
        )
        assert obs.training_data_phoneme_distribution is None
        assert obs.suspected_anomalous_phonemes_from_subenv2 is None

    def test_clip_evidence_dossier_primary_rejection_reason_optional(self):
        """primary_rejection_reason is Optional."""
        dossier = ClipEvidenceDossier(
            clip_id="test",
            identity_drift_severity="none",
            temporal_instability_flag=False,
            lip_sync_quality="good",
            unique_phoneme_value=0.5,
            dataset_redundancy_score=0.2,
            estimated_training_impact="positive",
            primary_rejection_reason=None,
            evidence_summary="test",
        )
        assert dossier.primary_rejection_reason is None


# ===========================================================================
# 7. Nested model validation
# ===========================================================================

class TestNestedModelValidation:

    def test_dataset_health_handoff_rejects_plain_bad_dict(self):
        """synthetic_weight_descriptor as an incompatible dict → ValidationError."""
        with pytest.raises(ValidationError):
            DatasetHealthHandoff(
                accepted_clip_count=1,
                rejected_clip_count=0,
                fix_recommended_count=0,
                identity_consistency_score=0.9,
                phoneme_coverage_score=0.6,
                pose_diversity_score=0.4,
                overall_dataset_quality=0.7,
                suspected_anomalous_phonemes=[],
                high_risk_clip_ids=[],
                weight_contamination_estimate=0.0,
                synthetic_weight_descriptor="not_a_dict",   # wrong type
                subenv2_score=0.8,
            )

    def test_dataset_health_handoff_accepts_nested_instance(self):
        """SyntheticWeightDescriptor instance coerced correctly."""
        model = DatasetHealthHandoff(
            accepted_clip_count=1,
            rejected_clip_count=0,
            fix_recommended_count=0,
            identity_consistency_score=0.9,
            phoneme_coverage_score=0.6,
            pose_diversity_score=0.4,
            overall_dataset_quality=0.7,
            suspected_anomalous_phonemes=[],
            high_risk_clip_ids=[],
            weight_contamination_estimate=0.0,
            synthetic_weight_descriptor=_synthetic_descriptor,
            subenv2_score=0.8,
        )
        assert isinstance(model.synthetic_weight_descriptor, SyntheticWeightDescriptor)

    def test_dataset_health_handoff_accepts_compatible_dict(self):
        """Pydantic v2 should coerce a compatible dict into the nested model."""
        desc_dict = _synthetic_descriptor.model_dump()
        model = DatasetHealthHandoff(
            accepted_clip_count=1,
            rejected_clip_count=0,
            fix_recommended_count=0,
            identity_consistency_score=0.9,
            phoneme_coverage_score=0.6,
            pose_diversity_score=0.4,
            overall_dataset_quality=0.7,
            suspected_anomalous_phonemes=[],
            high_risk_clip_ids=[],
            weight_contamination_estimate=0.0,
            synthetic_weight_descriptor=desc_dict,
            subenv2_score=0.8,
        )
        assert isinstance(model.synthetic_weight_descriptor, SyntheticWeightDescriptor)
        assert model.synthetic_weight_descriptor == _synthetic_descriptor


# ===========================================================================
# 8. Ground truth set[tuple] field
# ===========================================================================

class TestGroundTruthSetField:

    def test_valid_mitigation_set_is_set(self):
        """valid_mitigation_set must be a set (or frozenset) after construction."""
        gt = GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[],
            predicted_behavior_triggers=[],
            risky_phoneme_clusters=[],
            model_behavioral_safety="safe",
            valid_mitigation_set={("target_X", "add_counter_examples")},
        )
        assert isinstance(gt.valid_mitigation_set, (set, frozenset))

    def test_valid_mitigation_set_roundtrip(self):
        """Dict roundtrip preserves the set values exactly."""
        original = {
            ("expression_trigger cluster: ['EE', 'IY']", "add_counter_examples"),
            ("identity cluster: ['OW']", "flag_for_manual_review"),
        }
        gt = GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[],
            predicted_behavior_triggers=[],
            risky_phoneme_clusters=[],
            model_behavioral_safety="safe",
            valid_mitigation_set=original,
        )
        data = gt.model_dump()
        gt2 = GroundTruthBehavioralAnnotation(**data)
        assert gt2.valid_mitigation_set == gt.valid_mitigation_set

    def test_valid_mitigation_set_empty_roundtrip(self):
        """Empty set roundtrips without error."""
        gt = GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[],
            predicted_behavior_triggers=[],
            risky_phoneme_clusters=[],
            model_behavioral_safety="safe",
            valid_mitigation_set=set(),
        )
        data = gt.model_dump()
        gt2 = GroundTruthBehavioralAnnotation(**data)
        assert gt2.valid_mitigation_set == set()
