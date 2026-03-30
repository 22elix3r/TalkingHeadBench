## `src/utils/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/utils/grader_utils.py`
**Type:** Utility
**Purpose:** Shared grader utilities for TalkingHeadBench.
**Exports:** set_f1, jaccard_similarity
**Depends on:** none
**Key constraints:**
  - no live inference
  - never redefine set_f1 inline
  - never hardcode thresholds

---

## `src/utils/canonical.py`
**Type:** Utility
**Purpose:** W2T-style canonical LoRA decomposition (QR → SVD).
**Exports:** canonicalize_lora_factors, CanonicalComponents
**Depends on:** none
**Key constraints:**
  - no live inference
  - canonicalize_lora_factors must be used before any weight logic

---

## `src/schemas/__init__.py`
**Type:** Schema
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference
  - type annotations are mandatory

---

## `src/schemas/subenv1.py`
**Type:** Schema
**Purpose:** Pydantic v2 schemas for Sub-env 1: Reference Image + Prompt Audit.
**Exports:** ImageDiagnosticsObservation, ParamAnomalyAction, DirectionalFix, ImageDiagnosticsAction, ReferenceAuditHandoff, ParameterAnomaly, ParamAnomalyObservation
**Depends on:** none
**Key constraints:**
  - no live inference
  - type annotations are mandatory

---

## `src/schemas/subenv2.py`
**Type:** Schema
**Purpose:** Pydantic v2 schemas for Sub-env 2: Dataset Clip Audit.
**Exports:** ClipDispositionAction, ClipEvidenceDossier, SyntheticWeightDescriptor, DatasetHealthHandoff, ClipDispositionObservation, ClipSignalObservation
**Depends on:** none
**Key constraints:**
  - no live inference
  - override_decision is a 3-way Literal not bool
  - type annotations are mandatory

---

## `src/schemas/subenv3.py`
**Type:** Schema
**Purpose:** Pydantic v2 schemas for Sub-env 3: Trained LoRA Weight Behavioral Audit.
**Exports:** MitigationRecommendation, PhonemeRiskEntry, PhonemeCluster, WeightEvidenceDossier, BehaviorTriggerPrediction, BehavioralAuditHandoff, PhonemeRiskAction, WeightSignalObservation, LayerAnomalyFlag, PhonemeRiskObservation, TokenAnomalyFlag
**Depends on:** src.schemas.subenv2
**Key constraints:**
  - no live inference
  - type annotations are mandatory

---

## `src/schemas/ground_truth.py`
**Type:** Schema
**Purpose:** Pydantic v2 ground-truth annotation schemas for TalkingHeadBench.
**Exports:** GroundTruthBehavioralAnnotation, GroundTruthParamAnnotation, GroundTruthImageAnnotation, GroundTruthClipAnnotation
**Depends on:** src.schemas.subenv1, src.schemas.subenv3
**Key constraints:**
  - no live inference
  - type annotations are mandatory

---

## `src/envs/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/envs/subenv1/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/envs/subenv1/node1_image_diagnostician.py`
**Type:** Agent Node
**Purpose:** Node 1: Image Diagnostician — Sub-env 1.
**Exports:** diagnose_image
**Depends on:** src.schemas.subenv1
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/envs/subenv1/node2_param_anomaly.py`
**Type:** Agent Node
**Purpose:** Node 2: Parameter Anomaly Detector — Sub-env 1.
**Exports:** detect_param_anomalies
**Depends on:** src.schemas.subenv1
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/envs/subenv1/node3_grader.py`
**Type:** Grader Node
**Purpose:** Node 3 (Grader): Reference Audit Grader — Sub-env 1.
**Exports:** produce_reference_audit_handoff, grade_image_diagnostics, grade_anomaly_detection
**Depends on:** src.schemas.ground_truth, src.schemas.subenv1, src.utils.grader_utils
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/envs/subenv2/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/envs/subenv2/node4_clip_extractor.py`
**Type:** Agent Node
**Purpose:** Node 4: Clip Signal Extractor — Sub-env 2.
**Exports:** extract_clip_signals
**Depends on:** src.schemas.subenv2
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/envs/subenv2/node5_disposition.py`
**Type:** Agent Node
**Purpose:** Node 5: Clip Disposition Recommender.
**Exports:** recommend_clip_disposition
**Depends on:** src.schemas.subenv2
**Key constraints:**
  - no live inference
  - override_decision is a 3-way Literal not bool
  - never hardcode thresholds

---

## `src/envs/subenv2/node6_grader.py`
**Type:** Grader Node
**Purpose:** Node 6 (Grader): Dataset Health Grader — Sub-env 2.
**Exports:** grade_clip_disposition
**Depends on:** src.schemas.subenv2, src.schemas.ground_truth
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/envs/subenv3/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/envs/subenv3/node7_weight_extractor.py`
**Type:** Agent Node
**Purpose:** Node 7: Weight Signal Extractor — Sub-env 3.
**Exports:** extract_weight_signals
**Depends on:** src.schemas.subenv2, src.utils.canonical, src.schemas.subenv3
**Key constraints:**
  - no live inference
  - never hardcode thresholds
  - token_position_to_phoneme is loaded from tokenizer config, NOT derived from weights

---

## `src/envs/subenv3/node8_phoneme_risk.py`
**Type:** Agent Node
**Purpose:** Node 8: Phoneme Risk Assessor — Sub-env 3.
**Exports:** assess_phoneme_risk
**Depends on:** src.schemas.subenv3
**Key constraints:**
  - no live inference
  - never hardcode thresholds
  - token_position_to_phoneme is loaded from tokenizer config, NOT derived from weights

---

## `src/envs/subenv3/node9_grader.py`
**Type:** Grader Node
**Purpose:** Node 9 (Grader): Behavioral Audit Grader — Sub-env 3.
**Exports:** grade_behavioral_audit
**Depends on:** src.schemas.ground_truth, src.utils.grader_utils, src.schemas.subenv3
**Key constraints:**
  - no live inference
  - never hardcode thresholds

---

## `src/__init__.py`
**Type:** Utility
**Purpose:** Empty __init__.py stub.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - no live inference

---

## `src/pipeline.py`
**Type:** Pipeline
**Purpose:** TalkingHeadBench episode pipeline with two entry points: `run_episode` for typed in-memory observations and `run_episode_from_bundle` for legacy artifact-bundle execution (paths + agent callables) while preserving the same end-to-end scoring/handoff flow.
**Exports:** run_episode, run_episode_from_bundle, EpisodeResult, log
**Depends on:** src.envs.subenv2.node6_grader, src.envs.subenv3.node9_grader, src.schemas.subenv2, src.schemas.ground_truth, src.envs.subenv3.node8_phoneme_risk, src.schemas.subenv1, src.utils.grader_utils, src.envs.subenv2.node5_disposition, src.envs.subenv2.node4_clip_extractor, src.envs.subenv1.node2_param_anomaly, src.schemas.subenv3, src.envs.subenv3.node7_weight_extractor, src.envs.subenv1.node3_grader, src.envs.subenv1.node1_image_diagnostician
**Key constraints:**
  - no live inference
  - run_episode_from_bundle expects a structured artifact_bundle schema (required keys include image_obs, proposed_config, clips, weight_path, phoneme_obs_context, agents, and ground_truth)

---

## `src/evaluate.py`
**Type:** Harness
**Purpose:** TalkingHeadBench evaluation harness.
**Exports:** main
**Depends on:** src.envs.subenv2.node6_grader, src.envs.subenv3.node9_grader, src.schemas.subenv2, src.schemas.subenv1, src.envs.subenv3.node8_phoneme_risk, src.pipeline, src.envs.subenv2.node5_disposition, src.utils.grader_utils, src.schemas.subenv3, src.envs.subenv1.node2_param_anomaly, src.schemas.ground_truth, src.envs.subenv1.node3_grader, src.envs.subenv1.node1_image_diagnostician
**Key constraints:**
  - no live inference

---

## `conftest.py`
**Type:** Test
**Purpose:** Root conftest: ensure project root is on sys.path for `from src.…` imports.
**Exports:** ROOT
**Depends on:** none
**Key constraints:**
  - none

---

## `tests/unit/test_canonical.py`
**Type:** Test
**Purpose:** Unit tests for src/utils/canonical.py — canonicalize_lora_factors().
**Exports:** TestCanonicalizeLoraFactorsInvariance
**Depends on:** src.utils.canonical
**Key constraints:**
  - none

---

## `tests/unit/test_graders.py`
**Type:** Grader Node
**Purpose:** Unit tests for src/utils/grader_utils.py.
**Exports:** TestJaccardSimilarity, TestSetF1
**Depends on:** src.utils.grader_utils
**Key constraints:**
  - none

---

## `tests/unit/test_schemas.py`
**Type:** Test
**Purpose:** Unit tests for all four schema modules:
**Exports:** TestSubenv1ValidationErrors, test_token_mapping_optional, TestSubenv2ValidationErrors, TestSubenv1Instantiation, TestSubenv3Instantiation, TestGroundTruthValidationErrors, TestGroundTruthInstantiation, test_synthetic_weight_descriptor_typed, test_override_decision_rejects_bool, TestSubenv3ValidationErrors, TestSubenv2Instantiation
**Depends on:** src.schemas.subenv2, src.schemas.ground_truth, src.schemas.subenv1, src.schemas.subenv3
**Key constraints:**
  - none

---

## `tests/unit/test_node4_extractor.py`
**Type:** Agent Node
**Purpose:** Unit tests for Node 4: Clip Signal Extractor
**Exports:** mock_capture, test_phoneme_coverage_new_empty_dataset, test_blur_score_in_range, test_no_forced_align_path, test_raises_on_short_clip, test_phoneme_coverage_new_partial
**Depends on:** src.schemas.subenv2
**Key constraints:**
  - none

---

## `tests/unit/test_node7_extractor.py`
**Type:** Agent Node
**Purpose:** Unit tests for Node 7: Weight Signal Extractor
**Exports:** test_token_mapping_loaded_from_config, test_target_modules_count, test_overfitting_signature_in_range, test_token_mapping_none_without_config, test_rank_utilization_in_range, test_missing_file_raises, test_returns_valid_observation, test_layer_keys_consistent, test_weight_file_id_matches_filename, test_canonical_called_per_layer, test_canonical_entropy_nonnegative
**Depends on:** src.envs.subenv3.node7_weight_extractor, src.utils.canonical, src.schemas.subenv3
**Key constraints:**
  - none

---

## `tests/unit/test_subenv1.py`
**Type:** Test
**Purpose:** Integration tests for Sub-env 1: Reference Image + Prompt Audit.
**Exports:** test_node2_safe_config, test_node1_non_frontal_regime, test_node3_grader_produces_handoff, test_node2_detects_severe_anomaly, non_frontal_obs
**Depends on:** src.schemas.subenv1, src.envs.subenv1.node2_param_anomaly, src.schemas.ground_truth, src.envs.subenv1.node3_grader, src.envs.subenv1.node1_image_diagnostician
**Key constraints:**
  - none

---

## `tests/unit/test_subenv2.py`
**Type:** Test
**Purpose:** Integration tests for Sub-env 2: Dataset Clip Audit.
**Exports:** dossier_c, TestNode6DatasetHealthHandoff, dossier_a, TestNode5Disposition, dossier_b
**Depends on:** src.schemas.subenv2, src.schemas.ground_truth, src.pipeline, src.envs.subenv2.node5_disposition
**Key constraints:**
  - none

---

## `tests/unit/test_subenv3.py`
**Type:** Test
**Purpose:** Integration tests for Sub-env 3: Trained LoRA Weight Behavioral Audit.
**Exports:** test_subenv2_hints_propagate, test_grader_score_in_range, test_full_subenv3_pipeline
**Depends on:** src.envs.subenv3.node9_grader, src.envs.subenv3.node8_phoneme_risk, src.schemas.subenv3, src.schemas.ground_truth, src.envs.subenv3.node7_weight_extractor
**Key constraints:**
  - none

---

## `tests/smoke/test_grader_arithmetic.py`
**Type:** Grader Node
**Purpose:** Smoke tests: grader scoring arithmetic.
**Exports:** TestOrdinalRiskCalibration, TestSetF1, TestOrdinalSafetyCalibration, TestDispositionGraderBoundaries
**Depends on:** src.envs.subenv2.node6_grader, src.envs.subenv3.node9_grader, src.schemas.subenv2, src.schemas.subenv1, src.utils.grader_utils, src.schemas.subenv3, src.schemas.ground_truth, src.envs.subenv1.node3_grader
**Key constraints:**
  - none

---

## `tests/smoke/test_node1_boundaries.py`
**Type:** Agent Node
**Purpose:** Smoke tests: Node 1 (Image Diagnostician) boundary conditions.
**Exports:** TestRiskFactors, TestPromptIssues, base_obs, TestImageUsabilityScore, TestRegimeClassification
**Depends on:** src.schemas.subenv1, src.envs.subenv1.node1_image_diagnostician
**Key constraints:**
  - none

---

## `tests/smoke/test_node2_boundaries.py`
**Type:** Agent Node
**Purpose:** Smoke tests: Node 2 (Parameter Anomaly Detector) boundary conditions.
**Exports:** TestEmptyAndUnknownConfig, TestRiskLevelEscalation, TestDenoiseAltRule, TestFailureModeDeduplicated, TestCfgRule, base_obs, TestEtaRule, TestDirectionalFixes, run
**Depends on:** src.schemas.subenv1, src.envs.subenv1.node2_param_anomaly
**Key constraints:**
  - none

---

## `tests/smoke/test_node5_boundaries.py`
**Type:** Agent Node
**Purpose:** Smoke tests: Node 5 (Clip Disposition Recommender) boundary conditions.
**Exports:** test_defer_has_reason, test_severe_drift_rare_phonemes_fixes, test_fix_effort_trivial_for_single_flag, test_fix_generates_instructions, test_override_applied_for_negative_impact, test_confidence_always_in_range, test_severe_drift_low_phoneme_rejects, test_override_not_applicable_for_positive_impact, test_high_quality_accepts, test_dataset_impact_mentions_phoneme_gap, make_dossier, make_obs
**Depends on:** src.schemas.subenv2, src.envs.subenv2.node5_disposition
**Key constraints:**
  - none

---

## `tests/smoke/test_node7_deep.py`
**Type:** Agent Node
**Purpose:** Deep smoke tests for Node 7 (Weight Signal Extractor).
**Exports:** test_target_modules_matches_layer_count, test_layer_correlation_diagonal_is_one, test_overfitting_signature_reflects_rank_utilization, test_canonical_not_called_on_nonlora_keys, test_layer_sparsity_in_range, test_dominant_directions_leq_rank, test_layer_norms_positive, test_gradient_noise_nonnegative, test_histogram_sums_to_one, test_layer_correlation_matrix_is_square
**Depends on:** src.envs.subenv3.node7_weight_extractor, src.utils.canonical
**Key constraints:**
  - none

---

## `tests/smoke/test_node8_boundaries.py`
**Type:** Agent Node
**Purpose:** Smoke tests: Node 8 (Phoneme Risk Assessor) boundary conditions.
**Exports:** test_threshold_03_filters_low_risk, test_cluster_combined_score_is_mean, test_cluster_groups_by_risk_type, test_threshold_03_exact_boundary, test_none_hints_no_effect, test_cluster_requires_two_members, test_subenv2_hint_added_to_ranking, test_high_ee_scores_triggers_expression, base_obs, test_high_influence_alone_triggers_identity, test_safety_levels_at_exact_boundaries, test_subenv2_hint_not_duplicated, test_ranking_is_sorted_descending, test_all_zero_scores_produces_safe
**Depends on:** src.envs.subenv3.node8_phoneme_risk, src.schemas.subenv3
**Key constraints:**
  - none

---

## `tests/smoke/test_pipeline_e2e.py`
**Type:** Pipeline
**Purpose:** End-to-end smoke tests for the TalkingHeadBench episode pipeline.
**Exports:** test_handoffs_are_correct_types, test_subenv3_receives_synthetic_descriptor, test_subenv2_receives_risk_profile_from_subenv1, test_episode_returns_valid_result, test_final_score_is_weighted_combination, test_episode_deterministic, minimal_episode_inputs, test_node_failure_propagates
**Depends on:** src.schemas.subenv2, src.schemas.subenv1, src.schemas.subenv3, src.schemas.ground_truth, src.pipeline
**Key constraints:**
  - none

---

## `tests/smoke/test_schema_roundtrip.py`
**Type:** Test
**Purpose:** Schema roundtrip smoke tests.
**Exports:** TestSubenv2Roundtrip, TestOptionalFieldsDefaultToNone, TestGroundTruthSetField, TestSubenv3Roundtrip, TestLiteralValuesRejected, TestSubenv1Roundtrip, TestGroundTruthRoundtrip, TestNestedModelValidation
**Depends on:** src.schemas.subenv2, src.schemas.ground_truth, src.schemas.subenv1, src.schemas.subenv3
**Key constraints:**
  - none

---

## `tests/smoke/test_evaluate_cli.py`
**Type:** Harness
**Purpose:** CLI smoke tests for src/evaluate.py.
**Exports:** test_dry_run_subenv2_exits_zero, test_subenv1_produces_scores, test_invalid_json_exits_nonzero, test_score_values_in_range, project_root, test_missing_required_field_exits_one, test_dry_run_all_exits_zero, test_missing_test_set_argument, test_invalid_subenv_argument, test_dry_run_subenv1_exits_zero, test_dry_run_subenv3_exits_zero, test_subenv3_produces_scores, test_invalid_observation_field_exits_one, test_subenv2_produces_scores
**Depends on:** none
**Key constraints:**
  - none

---

## `tests/test_set/subenv1_cases.json`
**Type:** Data
**Purpose:** JSON test cases data for test.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `tests/test_set/subenv2_cases.json`
**Type:** Data
**Purpose:** JSON test cases data for test.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `tests/test_set/subenv3_cases.json`
**Type:** Data
**Purpose:** JSON test cases data for test.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `.gitignore`
**Type:** Config
**Purpose:** Specifies intentionally untracked files to ignore.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `requirements.txt`
**Type:** Config
**Purpose:** Lists project dependencies.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `README.md`
**Type:** Config
**Purpose:** Project overview and general introduction.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `.agent/agent.md`
**Type:** Config
**Purpose:** Context and critical design constraints for Claude.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `.agent/skills/talkingheadbench/SKILL.md`
**Type:** Config
**Purpose:** Skill definition for the benchmark.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## `.agent/skills/talkingheadbench/references/envs.md`
**Type:** Config
**Purpose:** Full environment specification and schemas.
**Exports:** none
**Depends on:** none
**Key constraints:**
  - none

---

## Dependency Graph
utils
  ↓
schemas
  ↓
envs (subenv1, subenv2, subenv3)
  ↓
pipeline
  ↓
evaluate

Unit tests target sub-components (utils, schemas, individual nodes).
Smoke tests target graders and pipelines end-to-end.

## Test Coverage Map
| File Under Test | Unit Tests | Smoke Tests | Total Assertions |
| --- | --- | --- | --- |
| src/utils/canonical.py | tests/unit/test_canonical.py | none | 17 |
| src/utils/grader_utils.py | tests/unit/test_graders.py | tests/smoke/test_grader_arithmetic.py | 53 |
| src/schemas/* | tests/unit/test_schemas.py | tests/smoke/test_schema_roundtrip.py | 50 |
| src/envs/subenv1/node1_image_diagnostician.py | tests/unit/test_subenv1.py | tests/smoke/test_node1_boundaries.py | 53 |
| src/envs/subenv1/node2_param_anomaly.py | tests/unit/test_subenv1.py | tests/smoke/test_node2_boundaries.py | 57 |
| src/envs/subenv1/node3_grader.py | tests/unit/test_subenv1.py | tests/smoke/test_grader_arithmetic.py | 40 |
| src/envs/subenv2/node4_clip_extractor.py | tests/unit/test_node4_extractor.py, tests/unit/test_subenv2.py | none | 25 |
| src/envs/subenv2/node5_disposition.py | tests/unit/test_subenv2.py | tests/smoke/test_node5_boundaries.py | 39 |
| src/envs/subenv2/node6_grader.py | tests/unit/test_subenv2.py | tests/smoke/test_grader_arithmetic.py | 46 |
| src/envs/subenv3/node7_weight_extractor.py | tests/unit/test_node7_extractor.py, tests/unit/test_subenv3.py | tests/smoke/test_node7_deep.py | 36 |
| src/envs/subenv3/node8_phoneme_risk.py | tests/unit/test_subenv3.py | tests/smoke/test_node8_boundaries.py | 31 |
| src/envs/subenv3/node9_grader.py | tests/unit/test_subenv3.py | tests/smoke/test_grader_arithmetic.py | 35 |
| src/pipeline.py | none | tests/smoke/test_pipeline_e2e.py | 18 |
| src/evaluate.py | none | tests/smoke/test_evaluate_cli.py | 25 |