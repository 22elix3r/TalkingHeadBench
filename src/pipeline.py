"""
TalkingHeadBench episode pipeline.

Orchestrates all three sub-environments in sequence and returns a fully
populated ``EpisodeResult`` with per-sub-environment scores, the final
weighted score, and the inter-sub-environment handoff objects.

Final score formula
-------------------
::

    final_score = 0.25 * subenv1_score
                + 0.35 * subenv2_score
                + 0.40 * subenv3_score

Sub-environment coupling
------------------------
- ``DatasetHealthHandoff`` (Node 6 output) feeds its
  ``synthetic_weight_descriptor`` into ``weight_obs.dataset_health_summary``
  and its ``suspected_anomalous_phonemes`` into
  ``PhonemeRiskObservation.suspected_anomalous_phonemes_from_subenv2``.

Error handling
--------------
Every node call is wrapped in ``_call_node`` which logs the node name and
re-raises on failure.  Nothing is silently swallowed.

``run_episode`` — new typed entry-point
---------------------------------------
Accepts fully pre-built observation objects; no file I/O, no callables dict::

    run_episode(
        reference_image_obs : ImageDiagnosticsObservation,
        param_config        : dict,
        clip_signal_obs_list: list[ClipSignalObservation],
        weight_obs          : WeightSignalObservation,
        ground_truths       : dict,   # keys: "image", "param", "clips", "behavioral"
    ) -> EpisodeResult

``run_episode_from_bundle`` — legacy bundle entry-point
-------------------------------------------------------
Retained for compatibility with existing integration tests.  Accepts an
``artifact_bundle`` dict (see legacy module docstring below).

``artifact_bundle`` schema (legacy)
------------------------------------
A flat-ish dict with the following required and optional keys::

    Required:
        "image_obs"               : ImageDiagnosticsObservation
        "proposed_config"         : dict   # user's generation config
        "clips"                   : list[dict], each entry:
            "path"                    : Path   # video file
            "dataset_context"         : dict   # see ClipDispositionObservation fields
            "aligner_output"          : dict | None   # MFA JSON; None → empty phoneme list
        "weight_path"             : Path   # .safetensors LoRA file
        "phoneme_obs_context"     : dict, with keys:
            "phoneme_vocabulary"          : list[str]
            "phoneme_to_token_indices"    : dict[str, list[int]]
            "phoneme_entropy_scores"      : dict[str, float]
            "phoneme_influence_scores"    : dict[str, float]
            "phoneme_cooccurrence_anomalies": list[tuple[str, str, float]]
            "behavior_vocabulary"         : list[str]
        "agents"                  : dict[str, Callable], with keys:
            "node1"   (ImageDiagnosticsObservation  → ImageDiagnosticsAction)
            "node2"   (ParamAnomalyObservation       → ParamAnomalyAction)
            "node4"   (ClipSignalObservation         → ClipEvidenceDossier)
            "node5"   (ClipDispositionObservation    → ClipDispositionAction)
            "node7"   (WeightSignalObservation       → WeightEvidenceDossier)
            "node8"   (PhonemeRiskObservation        → PhonemeRiskAction)
        "ground_truth"            : dict, with keys:
            "image"        : GroundTruthImageAnnotation
            "param"        : GroundTruthParamAnnotation
            "clips"        : list[GroundTruthClipAnnotation]  # one per clip, same order
            "behavioral"   : GroundTruthBehavioralAnnotation

    Optional:
        "tokenizer_config_path"   : Path | None   (default None)
        "training_data_phoneme_distribution": dict[str, int] | None   (default None)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import grade_anomaly_detection
from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.envs.subenv3.node9_grader import grade_behavioral_audit
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import (
    ImageDiagnosticsAction,
    ImageDiagnosticsObservation,
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
    PhonemeRiskAction,
    PhonemeRiskObservation,
    WeightEvidenceDossier,
    WeightSignalObservation,
)
from src.utils.grader_utils import set_f1

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Structured result of a complete TalkingHeadBench episode.

    Carries per-sub-environment scores, the final weighted score, and the
    three inter-sub-environment handoff objects that encode the coupling
    between sub-environments.

    Attributes:
        subenv1_score: Sub-env 1 score in [0, 1] (weight 0.25).
        subenv2_score: Sub-env 2 score in [0, 1] (weight 0.35).
        subenv3_score: Sub-env 3 score in [0, 1] (weight 0.40).
        final_score:   Weighted composite: 0.25·s1 + 0.35·s2 + 0.40·s3.
        reference_handoff:  Sub-env 1 → 2 coupling (``ReferenceAuditHandoff``).
        dataset_handoff:    Sub-env 2 → 3 coupling (``DatasetHealthHandoff``).
        behavioral_handoff: Sub-env 3 final grader output
            (``BehavioralAuditHandoff``).
    """

    subenv1_score: float
    subenv2_score: float
    subenv3_score: float
    final_score: float
    reference_handoff: ReferenceAuditHandoff
    dataset_handoff: DatasetHealthHandoff
    behavioral_handoff: BehavioralAuditHandoff


# ---------------------------------------------------------------------------
# Node-call wrapper (logs node name, always re-raises)
# ---------------------------------------------------------------------------


def _call_node(node_name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a node function, logging the node name on failure and re-raising.

    Never silently swallows any exception.

    Args:
        node_name: Human-readable node identifier for log messages.
        fn: The callable to invoke (agent, grader, or env extractor).
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        Any exception raised by ``fn``, re-raised after logging.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error("Node %s failed: %s", node_name, e)
        raise


# ---------------------------------------------------------------------------
# Sub-env 1 helpers
# ---------------------------------------------------------------------------


def _grade_image_diagnostics(
    agent_action: ImageDiagnosticsAction,
    ground_truth: GroundTruthImageAnnotation,
) -> float:
    """Grade the Image Diagnostician agent output (Node 1).

    Verbatim from the spec (references/envs.md §Node 1 Grader logic):

      1. Regime classification accuracy (0.35) — partial credit for borderline.
      2. Risk factor recall (0.35).
      3. Prompt modification validity (0.30) — deterministic set intersection.
    """
    scores: dict[str, float] = {}

    # 1. Regime classification
    if agent_action.regime_classification == ground_truth.regime_classification:
        scores["regime_accuracy"] = 1.0
    elif agent_action.regime_classification in ground_truth.acceptable_regimes:
        scores["regime_accuracy"] = 0.7
    else:
        scores["regime_accuracy"] = 0.0

    # 2. Risk factor recall
    predicted_risks = set(agent_action.identified_risk_factors)
    true_risks = set(ground_truth.identified_risk_factors)
    scores["risk_factor_recall"] = (
        len(predicted_risks & true_risks) / len(true_risks) if true_risks else 1.0
    )

    # 3. Prompt modification validity
    valid_mods = set(ground_truth.valid_prompt_modifications)
    agent_mods = set(agent_action.recommended_prompt_modifications)
    if agent_mods:
        scores["prompt_modification_validity"] = len(agent_mods & valid_mods) / len(agent_mods)
    else:
        scores["prompt_modification_validity"] = 0.0 if valid_mods else 1.0

    return (
        0.35 * scores["regime_accuracy"]
        + 0.35 * scores["risk_factor_recall"]
        + 0.30 * scores["prompt_modification_validity"]
    )


def _build_param_anomaly_obs(
    node1_action: ImageDiagnosticsAction,
    image_obs: ImageDiagnosticsObservation,
    proposed_config: dict,
) -> ParamAnomalyObservation:
    """Construct the Node 2 observation from Node 1 output + original inputs."""
    return ParamAnomalyObservation(
        proposed_config=proposed_config,
        regime=node1_action.regime_classification,
        identified_risk_factors=node1_action.identified_risk_factors,
        image_usability_score=node1_action.image_usability_score,
        face_occupancy_ratio=image_obs.face_occupancy_ratio,
        estimated_yaw_degrees=image_obs.estimated_yaw_degrees,
        background_complexity_score=image_obs.background_complexity_score,
        lighting_uniformity_score=image_obs.lighting_uniformity_score,
        occlusion_detected=image_obs.occlusion_detected,
        prompt_identity_anchoring=image_obs.identity_anchoring_strength,
        prompt_token_count=image_obs.prompt_token_count,
        conflicting_descriptors=image_obs.conflicting_descriptors,
    )


def _build_reference_audit_handoff(
    node1_action: ImageDiagnosticsAction,
    node2_action: ParamAnomalyAction,
    subenv1_score: float,
) -> ReferenceAuditHandoff:
    """Produce the Sub-env 1 → Sub-env 2 coupling object from graded node outputs."""
    risk_map = {"safe": "low", "marginal": "low", "risky": "medium", "dangerous": "high"}
    risk_profile = risk_map.get(node2_action.config_risk_level, "medium")

    # Config quality: invert the risk level ordinal
    risk_ordinal = {"safe": 1.0, "marginal": 0.75, "risky": 0.40, "dangerous": 0.10}
    config_quality_score = risk_ordinal.get(node2_action.config_risk_level, 0.5)

    # Estimated drift risk: proportion of severe anomalies
    total_anomalies = len(node2_action.anomalies)
    severe = sum(1 for a in node2_action.anomalies if a.severity == "severe")
    estimated_drift_risk = severe / total_anomalies if total_anomalies else 0.0

    return ReferenceAuditHandoff(
        image_usability_score=node1_action.image_usability_score,
        regime=node1_action.regime_classification,
        identified_risk_factors=node1_action.identified_risk_factors,
        config_quality_score=config_quality_score,
        risk_profile=risk_profile,
        estimated_drift_risk=estimated_drift_risk,
        prompt_strength=max(0.0, 1.0 - float(len(node1_action.prompt_issues)) * 0.1),
        recommended_config={},   # directional only — not prescriptive
        subenv1_score=subenv1_score,
    )


# ---------------------------------------------------------------------------
# Sub-env 2 helpers
# ---------------------------------------------------------------------------


def _heuristic_clip_evidence_dossier(obs: ClipSignalObservation) -> ClipEvidenceDossier:
    """Build a ``ClipEvidenceDossier`` heuristically from a ``ClipSignalObservation``.

    Used when no Node 4 agent callable is available (new ``run_episode`` API).
    All decisions are threshold-based rules, consistent with the Node 4 spec.

    Identity drift severity thresholds (identity_cosine_drift):
      < 0.05  → "none"
      < 0.12  → "minor"
      < 0.22  → "moderate"
      else    → "severe"

    Lip sync quality (lip_sync_confidence):
      ≥ 0.75  → "good"
      ≥ 0.50  → "acceptable"
      ≥ 0.20  → "poor"
      else    → "absent"
    """
    # Identity drift severity
    drift = obs.identity_cosine_drift
    if drift < 0.05:
        drift_severity = "none"
    elif drift < 0.12:
        drift_severity = "minor"
    elif drift < 0.22:
        drift_severity = "moderate"
    else:
        drift_severity = "severe"

    # Temporal instability: high landmark jitter or high frame difference
    temporal_instability = (
        obs.landmark_stability_score > 0.04 or obs.frame_difference_mean > 15.0
    )

    # Lip sync quality
    lsc = obs.lip_sync_confidence
    if lsc >= 0.75:
        lip_sync_quality = "good"
    elif lsc >= 0.50:
        lip_sync_quality = "acceptable"
    elif lsc >= 0.20:
        lip_sync_quality = "poor"
    else:
        lip_sync_quality = "absent"

    # Dataset redundancy: high similar_clips_accepted → redundant
    dataset_redundancy_score = min(1.0, obs.similar_clips_accepted / 10.0)

    # Estimated training impact
    if drift_severity in ("none", "minor") and not temporal_instability and lsc >= 0.50:
        training_impact = "positive"
    elif drift_severity == "severe" or (temporal_instability and lsc < 0.20):
        training_impact = "negative"
    else:
        training_impact = "neutral"

    # Primary rejection reason
    primary_rejection_reason: Optional[str] = None
    if drift_severity in ("moderate", "severe"):
        primary_rejection_reason = (
            f"identity cosine drift {drift:.3f} exceeds threshold; "
            f"severity={drift_severity}"
        )
    elif temporal_instability and lsc < 0.20:
        primary_rejection_reason = "temporal instability with absent lip sync"

    evidence_summary = (
        f"Drift={drift_severity}, temporal_instability={temporal_instability}, "
        f"lip_sync={lip_sync_quality}, phoneme_value={obs.phoneme_coverage_new:.2f}, "
        f"redundancy={dataset_redundancy_score:.2f}."
    )

    return ClipEvidenceDossier(
        clip_id=obs.clip_id,
        identity_drift_severity=drift_severity,
        temporal_instability_flag=temporal_instability,
        lip_sync_quality=lip_sync_quality,
        unique_phoneme_value=obs.phoneme_coverage_new,
        dataset_redundancy_score=dataset_redundancy_score,
        estimated_training_impact=training_impact,
        primary_rejection_reason=primary_rejection_reason,
        evidence_summary=evidence_summary,
    )


def _build_clip_disposition_obs(
    dossier: ClipEvidenceDossier,
    clip_context: dict,
) -> ClipDispositionObservation:
    """Build a ClipDispositionObservation by combining Node 4 output with context."""
    return ClipDispositionObservation(
        evidence_dossier=dossier,
        minimum_clips_needed=int(clip_context.get("minimum_clips_needed", 20)),
        phoneme_gap_severity=clip_context.get("phoneme_gap_severity", {}),
        pose_gap_severity=clip_context.get("pose_gap_severity", {}),
        budget_remaining=int(clip_context.get("budget_remaining", 0)),
        marginal_training_damage=float(clip_context.get("marginal_training_damage", 0.0)),
        marginal_coverage_gain=float(clip_context.get("marginal_coverage_gain", 0.0)),
    )


def _build_clip_disposition_obs_from_signal(
    dossier: ClipEvidenceDossier,
    clip_obs: ClipSignalObservation,
) -> ClipDispositionObservation:
    """Build ``ClipDispositionObservation`` directly from a ``ClipSignalObservation``.

    Used by ``run_episode`` (new API) where clip context is inferred from the
    observation itself rather than supplied in a separate context dict.
    """
    # Derive marginal signals from the observation
    marginal_coverage_gain = float(clip_obs.phoneme_coverage_new)
    # Damage estimate: high if severe drift or near-zero lip sync
    drift_damage_map = {"none": 0.0, "minor": 0.05, "moderate": 0.20, "severe": 0.50}
    marginal_training_damage = drift_damage_map.get(dossier.identity_drift_severity, 0.1)

    return ClipDispositionObservation(
        evidence_dossier=dossier,
        minimum_clips_needed=20,
        phoneme_gap_severity={},
        pose_gap_severity={},
        budget_remaining=max(0, 50 - clip_obs.clips_audited_so_far),
        marginal_training_damage=marginal_training_damage,
        marginal_coverage_gain=marginal_coverage_gain,
    )


def _build_dataset_health_handoff(
    clip_actions: list,
    clip_scores: list[float],
    clip_obs_list: list[ClipSignalObservation],
    subenv2_score: float,
) -> DatasetHealthHandoff:
    """Aggregate per-clip results into the Sub-env 2 → 3 coupling object."""
    accepted = sum(1 for a in clip_actions if a.disposition == "accept")
    rejected = sum(1 for a in clip_actions if a.disposition == "reject")
    fix_count = sum(1 for a in clip_actions if a.disposition == "fix")

    # Identity consistency: mean of per-clip variance signals (inverted)
    id_consistency = 1.0 - float(
        sum(o.face_embedding_variance for o in clip_obs_list) / max(len(clip_obs_list), 1)
    )
    id_consistency = max(0.0, min(1.0, id_consistency))

    # Phoneme coverage: mean phoneme_coverage_new across all clips
    phoneme_cov = float(
        sum(o.phoneme_coverage_new for o in clip_obs_list) / max(len(clip_obs_list), 1)
    )

    # Pose diversity: proxy from action confidence variance
    confidences = [a.confidence for a in clip_actions]
    pose_diversity = float(np.std(confidences)) if confidences else 0.0
    pose_diversity = max(0.0, min(1.0, pose_diversity))

    overall_quality = float(
        0.4 * id_consistency + 0.35 * phoneme_cov + 0.25 * pose_diversity
    )

    # Suspected anomalous phonemes: collect from rejected clips
    suspected_phonemes: list[str] = []
    for obs, action in zip(clip_obs_list, clip_actions):
        if action.disposition == "reject":
            suspected_phonemes.extend(obs.phoneme_sequence)
    suspected_phonemes = list(set(suspected_phonemes))

    # High-risk clip IDs: accepted clips with override applied
    high_risk_ids = [
        obs.clip_id
        for obs, action in zip(clip_obs_list, clip_actions)
        if action.disposition == "accept" and action.override_decision == "applied"
    ]

    weight_contamination = float(rejected / max(len(clip_actions), 1))

    synthetic_descriptor = SyntheticWeightDescriptor(
        estimated_rank_utilization=max(0.0, min(1.0, float(accepted / max(len(clip_actions), 1)))),
        suspected_overfitting_score=weight_contamination,
        high_risk_phoneme_hints=suspected_phonemes[:10],
        identity_consistency_estimate=id_consistency,
        expected_canonical_entropy_range=(0.5, 1.5 + pose_diversity),
    )

    return DatasetHealthHandoff(
        accepted_clip_count=accepted,
        rejected_clip_count=rejected,
        fix_recommended_count=fix_count,
        identity_consistency_score=id_consistency,
        phoneme_coverage_score=phoneme_cov,
        pose_diversity_score=pose_diversity,
        overall_dataset_quality=overall_quality,
        suspected_anomalous_phonemes=suspected_phonemes,
        high_risk_clip_ids=high_risk_ids,
        weight_contamination_estimate=weight_contamination,
        synthetic_weight_descriptor=synthetic_descriptor,
        subenv2_score=subenv2_score,
    )


# ---------------------------------------------------------------------------
# Sub-env 3 helpers
# ---------------------------------------------------------------------------


def _build_phoneme_risk_obs(
    weight_dossier: WeightEvidenceDossier,
    weight_obs: WeightSignalObservation,
    phoneme_ctx: dict,
    dataset_handoff: DatasetHealthHandoff,
) -> PhonemeRiskObservation:
    """Build the Node 8 observation from Node 7 outputs and phoneme context."""
    return PhonemeRiskObservation(
        weight_evidence=weight_dossier,
        high_entropy_token_flags=weight_dossier.high_entropy_token_flags,
        phoneme_vocabulary=phoneme_ctx["phoneme_vocabulary"],
        phoneme_to_token_indices=phoneme_ctx["phoneme_to_token_indices"],
        phoneme_entropy_scores=phoneme_ctx["phoneme_entropy_scores"],
        phoneme_influence_scores=phoneme_ctx["phoneme_influence_scores"],
        phoneme_cooccurrence_anomalies=phoneme_ctx["phoneme_cooccurrence_anomalies"],
        behavior_vocabulary=phoneme_ctx["behavior_vocabulary"],
        training_data_phoneme_distribution=phoneme_ctx.get("training_data_phoneme_distribution"),
        # ── Sub-env 2 coupling ──────────────────────────────────────────
        suspected_anomalous_phonemes_from_subenv2=dataset_handoff.suspected_anomalous_phonemes,
    )


def _build_phoneme_risk_obs_from_weight(
    weight_dossier: WeightEvidenceDossier,
    weight_obs: WeightSignalObservation,
    dataset_handoff: DatasetHealthHandoff,
) -> PhonemeRiskObservation:
    """Build the Node 8 observation from weight signals without an external phoneme context dict.

    Used by ``run_episode`` (new API) where phoneme signals are derived
    directly from ``weight_obs`` fields (canonical entropy / Vt analysis).
    Token-to-phoneme mapping is taken from ``weight_obs.token_position_to_phoneme``
    when available.
    """
    token_map: dict[int, str] = weight_obs.token_position_to_phoneme or {}

    # Derive phoneme vocabulary from token map (fall back to empty; ranking will be empty)
    phoneme_vocabulary: list[str] = sorted(set(token_map.values())) if token_map else []

    # phoneme_to_token_indices: invert the token map
    phoneme_to_token_indices: dict[str, list[int]] = {}
    for pos, ph in token_map.items():
        phoneme_to_token_indices.setdefault(ph, []).append(pos)

    # Phoneme entropy scores: average the canonical entropy of layers where
    # the phoneme's token positions are high-entropy.
    high_entropy_set = set(weight_obs.high_entropy_token_positions)
    layer_entropy_mean = (
        float(sum(weight_obs.canonical_entropy_per_layer.values()) /
              max(len(weight_obs.canonical_entropy_per_layer), 1))
        if weight_obs.canonical_entropy_per_layer else 0.0
    )
    phoneme_entropy_scores: dict[str, float] = {}
    phoneme_influence_scores: dict[str, float] = {}
    for ph, positions in phoneme_to_token_indices.items():
        # Entropy: fraction of this phoneme's token positions that are high-entropy
        if positions:
            high_frac = len([p for p in positions if p in high_entropy_set]) / len(positions)
            phoneme_entropy_scores[ph] = layer_entropy_mean * (1.0 + high_frac)
            # Influence: rank_utilization mean as a proxy for how much this phoneme shapes outputs
            phoneme_influence_scores[ph] = float(
                sum(weight_obs.layer_rank_utilization.values())
                / max(len(weight_obs.layer_rank_utilization), 1)
            ) * (1.0 + high_frac * 0.5)
        else:
            phoneme_entropy_scores[ph] = 0.0
            phoneme_influence_scores[ph] = 0.0

    # Behavior vocabulary: use dataset_handoff context if available, else minimal default
    behavior_vocabulary = ["smile", "blink", "head_turn", "jaw_drift", "brow_raise"]

    return PhonemeRiskObservation(
        weight_evidence=weight_dossier,
        high_entropy_token_flags=weight_dossier.high_entropy_token_flags,
        phoneme_vocabulary=phoneme_vocabulary,
        phoneme_to_token_indices=phoneme_to_token_indices,
        phoneme_entropy_scores=phoneme_entropy_scores,
        phoneme_influence_scores=phoneme_influence_scores,
        phoneme_cooccurrence_anomalies=[],
        behavior_vocabulary=behavior_vocabulary,
        training_data_phoneme_distribution=None,
        # ── Sub-env 2 coupling ──────────────────────────────────────────
        suspected_anomalous_phonemes_from_subenv2=dataset_handoff.suspected_anomalous_phonemes,
    )


def _build_behavioral_audit_handoff(
    node8_action: PhonemeRiskAction,
    subenv3_score: float,
    weight_file_id: str,
    component_scores: dict[str, float],
) -> BehavioralAuditHandoff:
    """Wrap the Node 9 grader result into the final handoff object."""
    return BehavioralAuditHandoff(
        weight_file_id=weight_file_id,
        phoneme_risk_ranking=node8_action.phoneme_risk_ranking,
        predicted_behavior_triggers=node8_action.predicted_behavior_triggers,
        risky_phoneme_clusters=node8_action.risky_phoneme_clusters,
        model_behavioral_safety=node8_action.model_behavioral_safety,
        mitigation_recommendations=node8_action.mitigation_recommendations,
        ranking_quality_score=component_scores.get("ranking_quality", 0.0),
        trigger_prediction_score=component_scores.get("trigger_prediction", 0.0),
        cluster_identification_score=component_scores.get("cluster_identification", 0.0),
        safety_calibration_score=component_scores.get("safety_calibration", 0.0),
        mitigation_quality_score=component_scores.get("mitigation_quality", 0.0),
        subenv3_score=subenv3_score,
    )


def _compute_behavioral_component_scores(
    node8_action: PhonemeRiskAction,
    behavioral_gt: GroundTruthBehavioralAnnotation,
) -> dict[str, float]:
    """Compute per-dimension scores for the behavioral audit handoff."""
    from src.utils.grader_utils import jaccard_similarity

    # 1. Ranking quality: top-5 set overlap / 5
    agent_top5 = {e.phoneme for e in node8_action.phoneme_risk_ranking[:5]}
    true_top5 = {e.phoneme for e in behavioral_gt.phoneme_risk_ranking[:5]}
    ranking_quality = len(agent_top5 & true_top5) / 5

    # 2. Trigger prediction: set F1 on (phoneme, behavior) tuples
    agent_triggers = {
        (t.trigger_phoneme, t.triggered_behavior)
        for t in node8_action.predicted_behavior_triggers
    }
    true_triggers = {
        (t.trigger_phoneme, t.triggered_behavior)
        for t in behavioral_gt.predicted_behavior_triggers
    }
    trigger_prediction = set_f1(agent_triggers, true_triggers)

    # 3. Cluster identification: Jaccard
    agent_clusters = {frozenset(c.phonemes) for c in node8_action.risky_phoneme_clusters}
    true_clusters = {frozenset(c.phonemes) for c in behavioral_gt.risky_phoneme_clusters}
    cluster_identification = jaccard_similarity(agent_clusters, true_clusters)

    # 4. Safety calibration: ordinal distance
    _SAFETY_LEVELS = ["safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"]
    try:
        agent_idx = _SAFETY_LEVELS.index(node8_action.model_behavioral_safety)
        true_idx = _SAFETY_LEVELS.index(behavioral_gt.model_behavioral_safety)
        safety_calibration = 1.0 - abs(agent_idx - true_idx) / (len(_SAFETY_LEVELS) - 1)
    except ValueError:
        safety_calibration = 0.0

    # 5. Mitigation quality: precision of (target, action) pairs
    agent_mits = {(m.target, m.action) for m in node8_action.mitigation_recommendations}
    valid_mits = behavioral_gt.valid_mitigation_set
    if agent_mits:
        mitigation_quality = len(agent_mits & valid_mits) / len(agent_mits)
    else:
        mitigation_quality = 0.0 if valid_mits else 1.0

    return {
        "ranking_quality": ranking_quality,
        "trigger_prediction": trigger_prediction,
        "cluster_identification": cluster_identification,
        "safety_calibration": safety_calibration,
        "mitigation_quality": mitigation_quality,
    }


# ---------------------------------------------------------------------------
# New typed entry-point
# ---------------------------------------------------------------------------


def run_episode(
    reference_image_obs: ImageDiagnosticsObservation,
    param_config: dict,
    clip_signal_obs_list: list[ClipSignalObservation],
    weight_obs: WeightSignalObservation,
    ground_truths: dict,
) -> EpisodeResult:
    """Run a full TalkingHeadBench episode with pre-built observation objects.

    Executes all 9 nodes across 3 sub-environments in order using the
    deterministic built-in node implementations.  Node agents (1, 2, 5, 7, 8)
    are called directly via their public functions; graders (3, 6, 9) are
    applied immediately after each sub-environment.

        Sub-environment coupling:
            - ``DatasetHealthHandoff.synthetic_weight_descriptor`` is injected into
        ``weight_obs.dataset_health_summary`` before Node 7 runs.
      - ``DatasetHealthHandoff.suspected_anomalous_phonemes`` is forwarded
        into ``PhonemeRiskObservation`` for Node 8.

    Sub-env 2 dossier construction:
      ``ClipEvidenceDossier`` objects are built heuristically from each
      ``ClipSignalObservation`` via ``_heuristic_clip_evidence_dossier``
      (threshold-based rules, no agent callable required).

    Args:
        reference_image_obs: Pre-extracted image + prompt signals (Node 1 input).
        param_config: User's proposed generation config dict (Node 2 input).
        clip_signal_obs_list: Pre-extracted per-clip CV signals (Node 5 inputs).
        weight_obs: Pre-extracted canonical weight statistics (Node 7 input).
        ground_truths: Dict with keys:
            ``"image"``      → ``GroundTruthImageAnnotation``
            ``"param"``      → ``GroundTruthParamAnnotation``
            ``"clips"``      → ``list[GroundTruthClipAnnotation]``
            ``"behavioral"`` → ``GroundTruthBehavioralAnnotation``

    Returns:
        An :class:`EpisodeResult` containing all scores and handoff objects.

    Raises:
        Any exception propagated from a failing node call (logged before raise).
    """
    gt = ground_truths
    gt_clips: list[GroundTruthClipAnnotation] = gt["clips"]

    # ======================================================================
    # Sub-env 1: Reference Image + Prompt Audit
    # ======================================================================
    log.info("=== Sub-env 1: Reference Image + Prompt Audit ===")

    # Node 1 — Image Diagnostician (agent)
    node1_action: ImageDiagnosticsAction = _call_node(
        "Node 1 (Image Diagnostician)", diagnose_image, reference_image_obs
    )

    # Node 2 — Parameter Anomaly Detector (agent)
    node2_obs = _build_param_anomaly_obs(node1_action, reference_image_obs, param_config)
    node2_action: ParamAnomalyAction = _call_node(
        "Node 2 (Parameter Anomaly Detector)", detect_param_anomalies, node2_obs
    )

    # Node 3 — Reference Audit Grader
    node1_score: float = _call_node(
        "Node 3 / grade_image_diagnostics",
        _grade_image_diagnostics,
        node1_action,
        gt["image"],
    )
    node2_score: float = _call_node(
        "Node 3 / grade_anomaly_detection",
        grade_anomaly_detection,
        node2_action,
        gt["param"],
    )
    subenv1_score = 0.50 * node1_score + 0.50 * node2_score

    reference_handoff: ReferenceAuditHandoff = _call_node(
        "Node 3 / build_reference_audit_handoff",
        _build_reference_audit_handoff,
        node1_action,
        node2_action,
        subenv1_score,
    )
    log.info(
        "Sub-env 1 score: %.4f  (risk_profile=%s)",
        subenv1_score,
        reference_handoff.risk_profile,
    )

    # ======================================================================
    # Sub-env 2: Dataset Clip Audit
    # ======================================================================
    log.info("=== Sub-env 2: Dataset Clip Audit (%d clips) ===", len(clip_signal_obs_list))

    clip_scores: list[float] = []
    clip_actions: list[ClipDispositionAction] = []

    for idx, clip_obs in enumerate(clip_signal_obs_list):
        clip_label = f"clip[{idx}] {clip_obs.clip_id}"

        # Build ClipEvidenceDossier heuristically from the observation
        dossier: ClipEvidenceDossier = _call_node(
            f"Node 4 heuristic ({clip_label})",
            _heuristic_clip_evidence_dossier,
            clip_obs,
        )

        # Node 5 — Clip Disposition Recommender (agent)
        disposition_obs = _build_clip_disposition_obs_from_signal(
            dossier, clip_obs
        )
        disposition_action: ClipDispositionAction = _call_node(
            f"Node 5 ({clip_label})", recommend_clip_disposition, disposition_obs
        )
        clip_actions.append(disposition_action)

        # Node 6 — per-clip grader
        clip_score: float = _call_node(
            f"Node 6 grader ({clip_label})",
            grade_clip_disposition,
            disposition_action,
            gt_clips[idx],
        )
        clip_scores.append(clip_score)
        log.debug(
            "  %s  disposition=%s  score=%.4f",
            clip_label,
            disposition_action.disposition,
            clip_score,
        )

    subenv2_score = float(sum(clip_scores) / len(clip_scores)) if clip_scores else 0.0

    dataset_handoff: DatasetHealthHandoff = _call_node(
        "Node 6 / build_dataset_health_handoff",
        _build_dataset_health_handoff,
        clip_actions,
        clip_scores,
        clip_signal_obs_list,
        subenv2_score,
    )
    log.info(
        "Sub-env 2 score: %.4f  (accepted=%d  rejected=%d  fix=%d)",
        subenv2_score,
        dataset_handoff.accepted_clip_count,
        dataset_handoff.rejected_clip_count,
        dataset_handoff.fix_recommended_count,
    )

    # ======================================================================
    # Sub-env 3: Trained LoRA Weight Behavioral Audit
    # ======================================================================
    log.info("=== Sub-env 3: Trained LoRA Weight Behavioral Audit ===")

    # ── Sub-env 2 → 3 coupling: inject dataset_health_summary ─────────────
    # Pydantic models are immutable by default; construct a new instance with
    # the synthetic_weight_descriptor injected as dataset_health_summary.
    enriched_weight_obs = weight_obs.model_copy(
        update={
            "dataset_health_summary": dataset_handoff.synthetic_weight_descriptor,
            "suspected_anomalous_phonemes": dataset_handoff.suspected_anomalous_phonemes,
        }
    )

    # Node 7 — Weight Signal Extractor agent (WeightSignalObservation → WeightEvidenceDossier)
    from src.envs.subenv3.node7_weight_extractor import _find_lora_pairs  # noqa: PLC0415
    from src.envs.subenv3 import node7_weight_extractor as _n7  # noqa: PLC0415

    # The node7 *agent* function does not exist separately from the env extractor;
    # the environment-side extraction is already done (weight_obs is pre-built).
    # We synthesise the WeightEvidenceDossier via the deterministic rule-based
    # assessment that mirrors what a node7 agent would produce.
    weight_dossier: WeightEvidenceDossier = _call_node(
        "Node 7 agent (assess_weight_evidence)",
        _assess_weight_evidence,
        enriched_weight_obs,
    )

    # Node 8 — Phoneme Risk Assessor (agent)
    phoneme_obs: PhonemeRiskObservation = _build_phoneme_risk_obs_from_weight(
        weight_dossier, enriched_weight_obs, dataset_handoff
    )
    node8_action: PhonemeRiskAction = _call_node(
        "Node 8 (Phoneme Risk Assessor)", assess_phoneme_risk, phoneme_obs
    )

    # Node 9 — Behavioral Audit Grader
    subenv3_score: float = _call_node(
        "Node 9 (Behavioral Audit Grader)",
        grade_behavioral_audit,
        node8_action,
        gt["behavioral"],
    )
    log.info(
        "Sub-env 3 score: %.4f  (safety=%s)",
        subenv3_score,
        node8_action.model_behavioral_safety,
    )

    # ======================================================================
    # Final weighted episode score
    # ======================================================================
    final_score = (
        0.25 * subenv1_score
        + 0.35 * subenv2_score
        + 0.40 * subenv3_score
    )

    component_scores = _compute_behavioral_component_scores(node8_action, gt["behavioral"])
    behavioral_handoff = _build_behavioral_audit_handoff(
        node8_action,
        subenv3_score,
        enriched_weight_obs.weight_file_id,
        component_scores,
    )

    log.info(
        "Episode complete — final_score=%.4f  "
        "(s1=%.4f × 0.25 + s2=%.4f × 0.35 + s3=%.4f × 0.40)",
        final_score,
        subenv1_score,
        subenv2_score,
        subenv3_score,
    )

    return EpisodeResult(
        subenv1_score=subenv1_score,
        subenv2_score=subenv2_score,
        subenv3_score=subenv3_score,
        final_score=final_score,
        reference_handoff=reference_handoff,
        dataset_handoff=dataset_handoff,
        behavioral_handoff=behavioral_handoff,
    )


# ---------------------------------------------------------------------------
# Helper: deterministic WeightEvidenceDossier from WeightSignalObservation
# ---------------------------------------------------------------------------


def _assess_weight_evidence(obs: WeightSignalObservation) -> WeightEvidenceDossier:
    """Produce a ``WeightEvidenceDossier`` from a ``WeightSignalObservation``.

    This function plays the role of the Node 7 *agent* in ``run_episode``.
    It is deterministic and rule-based — no LLM calls, no I/O.

    Training quality heuristic (from overfitting_signature and gradient_noise_estimate):
      overfit_sig ≥ 0.6              → "overfit"
      gradient_noise ≥ 0.5           → "unstable"
      mean rank_utilization ≤ 0.3   → "underfit"
      else                           → "healthy"

    Rank utilization assessment (mean across layers):
      ≤ 0.3   → "collapsed"
      ≤ 0.65  → "wasteful"
      else    → "efficient"

    Overall behavioral risk (from max layer_sparsity and overfitting_signature):
      overfit_sig ≥ 0.8  → "critical"
      overfit_sig ≥ 0.6  → "high"
      overfit_sig ≥ 0.3  → "medium"
      else               → "low"
    """
    from src.schemas.subenv3 import LayerAnomalyFlag, TokenAnomalyFlag

    # Training quality
    mean_util = (
        float(sum(obs.layer_rank_utilization.values()) / max(len(obs.layer_rank_utilization), 1))
        if obs.layer_rank_utilization else 0.5
    )
    if obs.overfitting_signature >= 0.6:
        training_quality = "overfit"
    elif obs.gradient_noise_estimate >= 0.5:
        training_quality = "unstable"
    elif mean_util <= 0.3:
        training_quality = "underfit"
    else:
        training_quality = "healthy"

    # Rank utilization assessment
    if mean_util <= 0.3:
        rank_assessment = "collapsed"
    elif mean_util <= 0.65:
        rank_assessment = "wasteful"
    else:
        rank_assessment = "efficient"

    # Overall behavioral risk
    sig = obs.overfitting_signature
    if sig >= 0.8:
        overall_risk = "critical"
    elif sig >= 0.6:
        overall_risk = "high"
    elif sig >= 0.3:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    # High-entropy token flags: derive from high_entropy_token_positions
    token_map = obs.token_position_to_phoneme or {}
    mean_entropy = (
        float(sum(obs.canonical_entropy_per_layer.values()) /
              max(len(obs.canonical_entropy_per_layer), 1))
        if obs.canonical_entropy_per_layer else 0.0
    )
    token_flags: list[TokenAnomalyFlag] = [
        TokenAnomalyFlag(
            token_position=pos,
            mapped_phoneme=token_map.get(pos),
            anomaly_type="excessive_influence",
            severity=min(1.0, mean_entropy * 1.5),
            evidence=(
                f"Token position {pos} flagged as high-entropy in canonical Vt analysis; "
                f"layer entropy mean {mean_entropy:.3f}"
            ),
        )
        for pos in obs.high_entropy_token_positions[:10]  # cap at 10
    ]

    # Layer anomaly flags: flag layers with sparsity > 0.5 or very high norms
    layer_flags: list[LayerAnomalyFlag] = []
    mean_norm = (
        float(sum(obs.layer_norms.values()) / max(len(obs.layer_norms), 1))
        if obs.layer_norms else 0.0
    )
    for layer_name, sparsity in obs.layer_sparsity.items():
        norm = obs.layer_norms.get(layer_name, 0.0)
        if sparsity > 0.5:
            layer_flags.append(
                LayerAnomalyFlag(
                    layer_name=layer_name,
                    anomaly_type="sparsity_anomaly",
                    severity=min(1.0, sparsity),
                    evidence=f"Sparsity {sparsity:.3f} — majority of canonical S near zero",
                )
            )
        elif mean_norm > 0 and norm > 3.0 * mean_norm:
            layer_flags.append(
                LayerAnomalyFlag(
                    layer_name=layer_name,
                    anomaly_type="norm_explosion",
                    severity=min(1.0, norm / (3.0 * mean_norm + 1e-8)),
                    evidence=(
                        f"Norm {norm:.3f} is {norm / (mean_norm + 1e-8):.1f}× "
                        f"the layer mean ({mean_norm:.3f})"
                    ),
                )
            )

    n_token = len(token_flags)
    n_layer = len(layer_flags)
    evidence_summary = (
        f"Training quality: {training_quality}. "
        f"Rank utilization: {rank_assessment} (mean {mean_util:.2f}). "
        f"{n_token} high-entropy token position(s) flagged. "
        f"{n_layer} layer anomaly/anomalies detected. "
        f"Overall risk: {overall_risk}."
    )

    return WeightEvidenceDossier(
        weight_file_id=obs.weight_file_id,
        training_quality=training_quality,
        rank_utilization_assessment=rank_assessment,
        high_entropy_token_flags=token_flags,
        layer_anomaly_flags=layer_flags,
        overall_behavioral_risk=overall_risk,
        evidence_summary=evidence_summary,
    )


# ---------------------------------------------------------------------------
# Legacy bundle entry-point (retained for existing integration tests)
# ---------------------------------------------------------------------------


def run_episode_from_bundle(artifact_bundle: dict) -> EpisodeResult:
    """Run a full TalkingHeadBench episode from a raw artifact bundle.

    This is the original entry-point, retained for backward compatibility with
    existing integration tests that supply an ``artifact_bundle`` dict with
    video paths, a ``weight_path``, and an ``agents`` callable dict.

    For new code, prefer :func:`run_episode` which accepts typed observation
    objects directly and returns a structured :class:`EpisodeResult`.

    Executes all 9 nodes across 3 sub-environments in order.  Every node call
    is wrapped so that failures are logged with the node name before the
    exception propagates — nothing is silently swallowed.

    The sub-environment coupling is:
            - ``DatasetHealthHandoff.synthetic_weight_descriptor`` → Node 7
        ``extract_weight_signals(dataset_health_summary=...)``
      - ``DatasetHealthHandoff.suspected_anomalous_phonemes`` → Node 8
        ``PhonemeRiskObservation.suspected_anomalous_phonemes_from_subenv2``

    Args:
        artifact_bundle: See module docstring for the full key schema.

    Returns:
        An :class:`EpisodeResult` containing per-sub-environment scores,
        ``final_score = 0.25 * s1 + 0.35 * s2 + 0.40 * s3``, and all
        inter-sub-environment handoff objects.
    """
    typed_bundle_keys = {
        "reference_image_obs",
        "param_config",
        "clip_signal_obs_list",
        "weight_obs",
        "ground_truths",
    }
    if typed_bundle_keys.issubset(artifact_bundle):
        reference_image_obs = artifact_bundle["reference_image_obs"]
        if isinstance(reference_image_obs, dict):
            reference_image_obs = ImageDiagnosticsObservation(**reference_image_obs)

        param_config = artifact_bundle["param_config"]
        if not isinstance(param_config, dict):
            raise TypeError(
                "Typed bundle key 'param_config' must be a dict. "
                f"Got {type(param_config).__name__}."
            )

        clip_signal_obs_list_raw = artifact_bundle["clip_signal_obs_list"]
        if not isinstance(clip_signal_obs_list_raw, list):
            raise TypeError(
                "Typed bundle key 'clip_signal_obs_list' must be a list. "
                f"Got {type(clip_signal_obs_list_raw).__name__}."
            )
        clip_signal_obs_list: list[ClipSignalObservation] = []
        for idx, clip_obs in enumerate(clip_signal_obs_list_raw):
            if isinstance(clip_obs, ClipSignalObservation):
                clip_signal_obs_list.append(clip_obs)
            elif isinstance(clip_obs, dict):
                clip_signal_obs_list.append(ClipSignalObservation(**clip_obs))
            else:
                raise TypeError(
                    "Typed bundle key 'clip_signal_obs_list' entries must be "
                    "ClipSignalObservation or dict. "
                    f"Entry {idx} is {type(clip_obs).__name__}."
                )

        weight_obs = artifact_bundle["weight_obs"]
        if isinstance(weight_obs, dict):
            weight_obs = WeightSignalObservation(**weight_obs)

        ground_truths_raw = artifact_bundle["ground_truths"]
        if not isinstance(ground_truths_raw, dict):
            raise TypeError(
                "Typed bundle key 'ground_truths' must be a dict. "
                f"Got {type(ground_truths_raw).__name__}."
            )

        gt_image = ground_truths_raw["image"]
        if isinstance(gt_image, dict):
            gt_image = GroundTruthImageAnnotation(**gt_image)

        gt_param = ground_truths_raw["param"]
        if isinstance(gt_param, dict):
            gt_param = GroundTruthParamAnnotation(**gt_param)

        gt_clips_raw = ground_truths_raw["clips"]
        if not isinstance(gt_clips_raw, list):
            raise TypeError(
                "Typed bundle key 'ground_truths[\"clips\"]' must be a list. "
                f"Got {type(gt_clips_raw).__name__}."
            )
        gt_clips: list[GroundTruthClipAnnotation] = []
        for idx, gt_clip in enumerate(gt_clips_raw):
            if isinstance(gt_clip, GroundTruthClipAnnotation):
                gt_clips.append(gt_clip)
            elif isinstance(gt_clip, dict):
                gt_clips.append(GroundTruthClipAnnotation(**gt_clip))
            else:
                raise TypeError(
                    "Typed bundle key 'ground_truths[\"clips\"]' entries must be "
                    "GroundTruthClipAnnotation or dict. "
                    f"Entry {idx} is {type(gt_clip).__name__}."
                )

        gt_behavioral = ground_truths_raw["behavioral"]
        if isinstance(gt_behavioral, dict):
            gt_behavioral = GroundTruthBehavioralAnnotation(**gt_behavioral)

        typed_ground_truths = {
            "image": gt_image,
            "param": gt_param,
            "clips": gt_clips,
            "behavioral": gt_behavioral,
        }

        return run_episode(
            reference_image_obs=reference_image_obs,
            param_config=param_config,
            clip_signal_obs_list=clip_signal_obs_list,
            weight_obs=weight_obs,
            ground_truths=typed_ground_truths,
        )

    agents = artifact_bundle["agents"]
    gt = artifact_bundle["ground_truth"]
    phoneme_ctx = artifact_bundle["phoneme_obs_context"]
    tokenizer_config_path = artifact_bundle.get("tokenizer_config_path")

    # ======================================================================
    # Sub-env 1: Reference Image + Prompt Audit
    # ======================================================================
    log.info("=== Sub-env 1: Reference Image + Prompt Audit ===")

    image_obs: ImageDiagnosticsObservation = artifact_bundle["image_obs"]
    proposed_config: dict = artifact_bundle["proposed_config"]

    # Node 1 — Image Diagnostician (agent)
    node1_action: ImageDiagnosticsAction = _call_node(
        "Node 1 (Image Diagnostician)", agents["node1"], image_obs
    )

    # Node 2 — Parameter Anomaly Detector (agent)
    node2_obs = _build_param_anomaly_obs(node1_action, image_obs, proposed_config)
    node2_action: ParamAnomalyAction = _call_node(
        "Node 2 (Parameter Anomaly Detector)", agents["node2"], node2_obs
    )

    # Node 3 — Reference Audit Grader
    node1_score: float = _call_node(
        "Node 3 / grade_image_diagnostics",
        _grade_image_diagnostics,
        node1_action,
        gt["image"],
    )
    node2_score: float = _call_node(
        "Node 3 / grade_anomaly_detection",
        grade_anomaly_detection,
        node2_action,
        gt["param"],
    )
    subenv1_score = 0.50 * node1_score + 0.50 * node2_score

    reference_handoff: ReferenceAuditHandoff = _call_node(
        "Node 3 / build_reference_audit_handoff",
        _build_reference_audit_handoff,
        node1_action,
        node2_action,
        subenv1_score,
    )
    log.info(
        "Sub-env 1 score: %.4f  (risk_profile=%s)",
        subenv1_score,
        reference_handoff.risk_profile,
    )

    # ======================================================================
    # Sub-env 2: Dataset Clip Audit
    # ======================================================================
    log.info(
        "=== Sub-env 2: Dataset Clip Audit (%d clips) ===",
        len(artifact_bundle["clips"]),
    )

    clip_scores: list[float] = []
    clip_actions: list = []
    clip_obs_list: list[ClipSignalObservation] = []
    gt_clips: list[GroundTruthClipAnnotation] = gt["clips"]

    for idx, clip_spec in enumerate(artifact_bundle["clips"]):
        clip_label = f"clip[{idx}] {Path(clip_spec['path']).name}"

        # Node 4 env pre-extraction (environment, not agent)
        raw_obs: ClipSignalObservation = _call_node(
            f"Node 4 env / extract_clip_signals ({clip_label})",
            extract_clip_signals,
            clip_spec["path"],
            clip_spec["dataset_context"],
            clip_spec.get("aligner_output"),
        )
        clip_obs_list.append(raw_obs)

        # Node 4 — Clip Signal Extractor (agent: ClipSignalObservation → ClipEvidenceDossier)
        dossier: ClipEvidenceDossier = _call_node(
            f"Node 4 agent ({clip_label})", agents["node4"], raw_obs
        )

        # Node 5 — Clip Disposition Recommender (agent)
        disposition_obs = _build_clip_disposition_obs(
            dossier, clip_spec["dataset_context"]
        )
        disposition_action = _call_node(
            f"Node 5 ({clip_label})", agents["node5"], disposition_obs
        )
        clip_actions.append(disposition_action)

        # Node 6 — per-clip grader
        clip_score: float = _call_node(
            f"Node 6 grader ({clip_label})",
            grade_clip_disposition,
            disposition_action,
            gt_clips[idx],
        )
        clip_scores.append(clip_score)
        log.debug(
            "  %s  disposition=%s  score=%.4f",
            clip_label,
            disposition_action.disposition,
            clip_score,
        )

    subenv2_score = float(sum(clip_scores) / len(clip_scores)) if clip_scores else 0.0

    dataset_handoff: DatasetHealthHandoff = _call_node(
        "Node 6 / build_dataset_health_handoff",
        _build_dataset_health_handoff,
        clip_actions,
        clip_scores,
        clip_obs_list,
        subenv2_score,
    )
    log.info(
        "Sub-env 2 score: %.4f  (accepted=%d  rejected=%d  fix=%d)",
        subenv2_score,
        dataset_handoff.accepted_clip_count,
        dataset_handoff.rejected_clip_count,
        dataset_handoff.fix_recommended_count,
    )

    # ======================================================================
    # Sub-env 3: Trained LoRA Weight Behavioral Audit
    # ======================================================================
    log.info("=== Sub-env 3: Trained LoRA Weight Behavioral Audit ===")

    weight_path = Path(artifact_bundle["weight_path"])

    # Node 7 env pre-extraction (environment)
    # ── Sub-env 2 coupling: dataset_health_summary ─────────────────────────
    weight_obs: WeightSignalObservation = _call_node(
        "Node 7 env / extract_weight_signals",
        extract_weight_signals,
        weight_path,
        tokenizer_config_path,
        dataset_handoff.synthetic_weight_descriptor,   # Sub-env 2 → 3
        dataset_handoff.suspected_anomalous_phonemes,  # Sub-env 2 → 3
    )

    # Node 7 — Weight Signal Extractor (agent: WeightSignalObservation → WeightEvidenceDossier)
    weight_dossier: WeightEvidenceDossier = _call_node(
        "Node 7 agent", agents["node7"], weight_obs
    )

    # Node 8 — Phoneme Risk Assessor (agent)
    phoneme_obs = _build_phoneme_risk_obs(
        weight_dossier, weight_obs, phoneme_ctx, dataset_handoff
    )
    node8_action: PhonemeRiskAction = _call_node(
        "Node 8 (Phoneme Risk Assessor)", agents["node8"], phoneme_obs
    )

    # Node 9 — Behavioral Audit Grader
    subenv3_score: float = _call_node(
        "Node 9 (Behavioral Audit Grader)",
        grade_behavioral_audit,
        node8_action,
        gt["behavioral"],
    )
    log.info(
        "Sub-env 3 score: %.4f  (safety=%s)",
        subenv3_score,
        node8_action.model_behavioral_safety,
    )

    # ======================================================================
    # Final weighted episode score
    # ======================================================================
    final_score = (
        0.25 * subenv1_score
        + 0.35 * subenv2_score
        + 0.40 * subenv3_score
    )
    log.info(
        "Episode complete — final_score=%.4f  "
        "(s1=%.4f × 0.25 + s2=%.4f × 0.35 + s3=%.4f × 0.40)",
        final_score,
        subenv1_score,
        subenv2_score,
        subenv3_score,
    )
    component_scores = _compute_behavioral_component_scores(
        node8_action,
        gt["behavioral"],
    )
    behavioral_handoff = _build_behavioral_audit_handoff(
        node8_action,
        subenv3_score,
        weight_obs.weight_file_id,
        component_scores,
    )

    return EpisodeResult(
        subenv1_score=subenv1_score,
        subenv2_score=subenv2_score,
        subenv3_score=subenv3_score,
        final_score=final_score,
        reference_handoff=reference_handoff,
        dataset_handoff=dataset_handoff,
        behavioral_handoff=behavioral_handoff,
    )
