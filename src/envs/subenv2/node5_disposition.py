"""
Node 5: Clip Disposition Recommender.

Deterministic heuristic to recommend accept/reject/fix/defer for a video clip.
Based on the spec in references/envs.md and specific requested logic.
"""

from src.schemas.subenv2 import (
    ClipDispositionObservation,
    ClipDispositionAction,
)


def recommend_clip_disposition(obs: ClipDispositionObservation) -> ClipDispositionAction:
    """
    Recommend a disposition for a clip using a rule-based heuristic.
    """
    dossier = obs.evidence_dossier

    # Step 1: Base quality score [0.0, 1.0]
    drift_map = {"none": 1.0, "minor": 0.7, "moderate": 0.3, "severe": 0.0}
    drift_score = drift_map.get(dossier.identity_drift_severity, 0.0)

    stability = 0.0 if dossier.temporal_instability_flag else 0.3
    sync = 0.3 if dossier.lip_sync_quality in ["good", "acceptable"] else 0.0
    phoneme_val = 0.2 * dossier.unique_phoneme_value
    redundancy = 0.2 * (1.0 - dossier.dataset_redundancy_score)

    quality = (drift_score + stability + sync + phoneme_val + redundancy) / 2.0

    # Step 2: Disposition
    disposition = "accept"
    if quality >= 0.7:
        disposition = "accept"
    elif quality < 0.3 and dossier.unique_phoneme_value <= 0.3:
        disposition = "reject"
    elif quality < 0.3 and dossier.unique_phoneme_value > 0.3:
        disposition = "fix"
    elif 0.3 <= quality < 0.5:
        disposition = "fix"
    elif 0.5 <= quality < 0.7 and dossier.estimated_training_impact == "neutral":
        disposition = "defer"
    else:
        disposition = "accept"

    # Field assembly
    fix_instructions = None
    estimated_fix_effort = None
    if disposition == "fix":
        fix_instructions = []
        fix_count = 0
        if dossier.temporal_instability_flag:
            fix_instructions.append("trim frames with temporal instability in jaw landmark region")
            fix_count += 1
        if dossier.lip_sync_quality == "poor":
            fix_instructions.append("re-record segment — lip sync confidence below threshold")
            fix_count += 1
        if dossier.dataset_redundancy_score > 0.7:
            fix_instructions.append("clip is redundant — consider replacing with novel scenario")
            fix_count += 1

        fix_instructions.append(f"retained value: unique_phoneme_value={dossier.unique_phoneme_value:.2f}")

        # estimated_fix_effort
        if dossier.temporal_instability_flag and fix_count == 1:
            estimated_fix_effort = "trivial"
        elif fix_count >= 2:
            estimated_fix_effort = "high"
        else:
            estimated_fix_effort = "moderate"

    defer_reason = None
    if disposition == "defer":
        defer_reason = f"quality borderline ({quality:.2f}) — manual review recommended"

    # Override handling
    override_decision = "not_applicable"
    override_justification = None
    if disposition in ["accept", "fix"] and dossier.estimated_training_impact == "negative":
        override_decision = "applied"
        override_justification = f"accepting despite negative training impact: {dossier.primary_rejection_reason}"

    # Dataset impact reasoning
    dataset_impact_reasoning = (
        f"Dataset phoneme gaps: {list(obs.phoneme_gap_severity.keys())}. "
        f"Pose gaps: {list(obs.pose_gap_severity.keys())}. "
        f"This clip {'addresses' if dossier.unique_phoneme_value > 0.3 else 'does not address'} critical gaps."
    )

    return ClipDispositionAction(
        disposition=disposition,
        confidence=float(quality),
        rejection_reasons=[dossier.primary_rejection_reason] if disposition == "reject" and dossier.primary_rejection_reason else None,
        fix_instructions=fix_instructions,
        estimated_fix_effort=estimated_fix_effort,
        defer_reason=defer_reason,
        dataset_impact_reasoning=dataset_impact_reasoning,
        override_decision=override_decision,
        override_justification=override_justification,
    )
