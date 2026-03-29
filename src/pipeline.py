"""
TalkingHeadBench episode pipeline.

Orchestrates all three sub-environments in sequence and returns the final
weighted episode score:

    final_score = 0.25 * subenv1_score
                + 0.35 * subenv2_score
                + 0.40 * subenv3_score

Sub-environment coupling
------------------------
- ``ReferenceAuditHandoff`` (Node 3 output) feeds ``risk_profile`` and
  ``estimated_drift_risk`` into every clip's ``ClipDispositionObservation``
  as ``reference_risk_profile`` and ``estimated_drift_risk``.
- ``DatasetHealthHandoff`` (Node 6 output) feeds its
  ``synthetic_weight_descriptor`` into ``extract_weight_signals`` as
  ``dataset_health_summary``, and its ``suspected_anomalous_phonemes`` into
  ``PhonemeRiskObservation``.

Error handling
--------------
Every node call is wrapped in a ``try/except`` that logs the node name and
re-raises.  Errors are never silently swallowed.

``artifact_bundle`` schema
--------------------------
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
from pathlib import Path
from typing import Any, Callable, Optional

from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.envs.subenv3.node9_grader import grade_behavioral_audit
from src.envs.subenv1.node3_grader import grade_anomaly_detection
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
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
    DatasetHealthHandoff,
    SyntheticWeightDescriptor,
)
from src.schemas.subenv3 import (
    PhonemeRiskAction,
    PhonemeRiskObservation,
    WeightEvidenceDossier,
    WeightSignalObservation,
)
from src.utils.grader_utils import set_f1

log = logging.getLogger(__name__)


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
    except Exception as exc:
        log.error("[%s] failed with %s: %s", node_name, type(exc).__name__, exc)
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


def _build_clip_disposition_obs(
    dossier: ClipEvidenceDossier,
    clip_context: dict,
    reference_handoff: ReferenceAuditHandoff,
) -> ClipDispositionObservation:
    """Build a ClipDispositionObservation by combining Node 4 output with context."""
    return ClipDispositionObservation(
        evidence_dossier=dossier,
        minimum_clips_needed=int(clip_context.get("minimum_clips_needed", 20)),
        phoneme_gap_severity=clip_context.get("phoneme_gap_severity", {}),
        pose_gap_severity=clip_context.get("pose_gap_severity", {}),
        budget_remaining=int(clip_context.get("budget_remaining", 0)),
        # ── Sub-env 1 coupling ──────────────────────────────────────────
        reference_risk_profile=reference_handoff.risk_profile,
        estimated_drift_risk=reference_handoff.estimated_drift_risk,
        # ────────────────────────────────────────────────────────────────
        marginal_training_damage=float(clip_context.get("marginal_training_damage", 0.0)),
        marginal_coverage_gain=float(clip_context.get("marginal_coverage_gain", 0.0)),
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
    import numpy as np
    confidences = [a.confidence for a in clip_actions]
    pose_diversity = float(np.std(confidences)) if confidences else 0.0
    pose_diversity = max(0.0, min(1.0, pose_diversity))

    overall_quality = float(
        0.4 * id_consistency + 0.35 * phoneme_cov + 0.25 * pose_diversity
    )

    # Suspected anomalous phonemes: collect from clips with negative training impact
    suspected_phonemes: list[str] = []
    for obs, action in zip(clip_obs_list, clip_actions):
        if action.disposition in ("reject",):
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


def _build_behavioral_audit_handoff(
    node8_action: PhonemeRiskAction,
    subenv3_score: float,
    weight_file_id: str,
    component_scores: dict[str, float],
) -> "BehavioralAuditHandoff":
    """Wrap the Node 9 grader result into the final handoff object."""
    from src.schemas.subenv3 import BehavioralAuditHandoff
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_episode(artifact_bundle: dict) -> float:
    """Run a full TalkingHeadBench episode and return the final score.

    Executes all 9 nodes across 3 sub-environments in order.  Every node call
    is wrapped so that failures are logged with the node name before the
    exception propagates — nothing is silently swallowed.

    The sub-environment coupling is:
      - ``ReferenceAuditHandoff.risk_profile``        → each clip's
        ``ClipDispositionObservation.reference_risk_profile``
      - ``ReferenceAuditHandoff.estimated_drift_risk`` → each clip's
        ``ClipDispositionObservation.estimated_drift_risk``
      - ``DatasetHealthHandoff.synthetic_weight_descriptor`` → Node 7
        ``extract_weight_signals(dataset_health_summary=...)``
      - ``DatasetHealthHandoff.suspected_anomalous_phonemes`` → Node 8
        ``PhonemeRiskObservation.suspected_anomalous_phonemes_from_subenv2``

    Args:
        artifact_bundle: See module docstring for the full key schema.

    Returns:
        ``final_score = 0.25 * s1 + 0.35 * s2 + 0.40 * s3`` in [0.0, 1.0].
    """
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
    log.info("Sub-env 1 score: %.4f  (risk_profile=%s)", subenv1_score, reference_handoff.risk_profile)

    # ======================================================================
    # Sub-env 2: Dataset Clip Audit
    # ======================================================================
    log.info("=== Sub-env 2: Dataset Clip Audit (%d clips) ===", len(artifact_bundle["clips"]))

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
            dossier, clip_spec["dataset_context"], reference_handoff
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
        log.debug("  %s  disposition=%s  score=%.4f", clip_label, disposition_action.disposition, clip_score)

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
    log.info("Sub-env 3 score: %.4f  (safety=%s)", subenv3_score, node8_action.model_behavioral_safety)

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
        final_score, subenv1_score, subenv2_score, subenv3_score,
    )
    return final_score
