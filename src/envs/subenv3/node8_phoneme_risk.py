"""
Node 8: Phoneme Risk Assessor — Sub-env 3.

Consumes aggregated phoneme-level canonical signals (entropy and influence
scores derived from the W2T canonical SVD decomposition) alongside the
weight evidence produced by Node 7.  Produces a structured behavioral risk
profile without calling any external model or LLM.

Risk scoring formula
--------------------
    risk_score = clip(0.6 * entropy + 0.4 * influence, 0.0, 1.0)

Only phonemes with ``risk_score > 0.3`` enter the ranking list.
Phonemes flagged by the Sub-env 2 dataset audit that did not make the
threshold are appended at a fixed risk score of 0.4 (unknown_anomaly).

Risk type decision tree (evaluated in order)
--------------------------------------------
1. entropy > 0.7 AND influence > 0.5  → "expression_trigger"
2. influence > 0.7                    → "identity_trigger"
3. entropy > 0.5                      → "motion_trigger"
4. else                               → "unknown_anomaly"

Behavioral safety from max risk score
--------------------------------------
    < 0.30  → "safe"
    < 0.50  → "minor_concerns"
    < 0.65  → "moderate_risk"
    < 0.80  → "high_risk"
    else    → "unsafe"
"""

from __future__ import annotations

from src.schemas.subenv3 import (
    BehaviorTriggerPrediction,
    MitigationRecommendation,
    PhonemeCluster,
    PhonemeRiskAction,
    PhonemeRiskEntry,
    PhonemeRiskObservation,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _risk_type(entropy: float, influence: float) -> str:
    """Determine risk type from entropy and influence scores."""
    if entropy > 0.7 and influence > 0.5:
        return "expression_trigger"
    if influence > 0.7:
        return "identity_trigger"
    if entropy > 0.5:
        return "motion_trigger"
    return "unknown_anomaly"


def _concern_level(risk_score: float) -> str:
    """Map risk score to BehaviorTriggerPrediction concern_level."""
    if risk_score > 0.75:
        return "high"
    if risk_score > 0.5:
        return "medium"
    return "low"


def _triggered_behavior(phoneme: str, risk_type: str) -> str:
    """Map a risk type (and phoneme) to a predicted triggered behavior."""
    if risk_type == "expression_trigger":
        return "smile" if phoneme in {"EE", "IY", "EY"} else "brow_raise"
    if risk_type == "identity_trigger":
        return "jaw_drift"
    # motion_trigger and fallback
    return "head_turn"


def _mitigation(cluster: PhonemeCluster) -> MitigationRecommendation:
    """Produce a MitigationRecommendation for a risky phoneme cluster."""
    rt = cluster.cluster_risk_type

    if rt == "expression_trigger":
        action = "add_counter_examples"
        rationale = (
            "add clips where speaker produces these phonemes with neutral expression"
        )
    elif rt == "identity_trigger":
        action = "flag_for_manual_review"
        rationale = (
            "identity drift may indicate deeper dataset bias — review source clips"
        )
    else:
        action = "retrain_with_more_data"
        rationale = "insufficient training coverage for this phoneme group"

    priority = "critical" if cluster.combined_risk_score > 0.7 else "recommended"

    return MitigationRecommendation(
        target=f"{rt} cluster: {cluster.phonemes}",
        action=action,
        rationale=rationale,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_phoneme_risk(obs: PhonemeRiskObservation) -> PhonemeRiskAction:
    """Assess behavioral risks from phoneme-level canonical weight signals.

    Pure deterministic function — no LLM calls, no I/O.

    Args:
        obs: Pre-extracted phoneme risk observation from Node 7 and Sub-env 2.

    Returns:
        A fully populated :class:`PhonemeRiskAction`.
    """
    # ------------------------------------------------------------------
    # 1. Build PhonemeRiskEntry for each phoneme in the vocabulary
    # ------------------------------------------------------------------
    all_entries: list[PhonemeRiskEntry] = []

    for phoneme in obs.phoneme_vocabulary:
        entropy = obs.phoneme_entropy_scores.get(phoneme, 0.0)
        influence = obs.phoneme_influence_scores.get(phoneme, 0.0)
        risk_score = min(0.6 * entropy + 0.4 * influence, 1.0)

        if risk_score <= 0.3:
            continue

        rt = _risk_type(entropy, influence)
        # Confidence: agreement between entropy and influence
        confidence = min(entropy, influence) / (max(entropy, influence) + 1e-8)
        evidence = (
            f"canonical entropy {entropy:.2f}, influence score {influence:.2f}"
        )

        all_entries.append(
            PhonemeRiskEntry(
                phoneme=phoneme,
                risk_score=risk_score,
                risk_type=rt,
                confidence=confidence,
                evidence=evidence,
            )
        )

    # Sort descending by risk_score
    all_entries.sort(key=lambda e: e.risk_score, reverse=True)

    # ------------------------------------------------------------------
    # 2. Append Sub-env 2 suspected phonemes not already in the ranking
    # ------------------------------------------------------------------
    if obs.suspected_anomalous_phonemes_from_subenv2 is not None:
        already_ranked = {e.phoneme for e in all_entries}
        for p in obs.suspected_anomalous_phonemes_from_subenv2:
            if p not in already_ranked:
                all_entries.append(
                    PhonemeRiskEntry(
                        phoneme=p,
                        risk_score=0.4,
                        risk_type="unknown_anomaly",
                        confidence=0.3,
                        evidence="flagged by dataset audit (Sub-env 2)",
                    )
                )

    phoneme_risk_ranking = all_entries

    # ------------------------------------------------------------------
    # 3. BehaviorTriggerPredictions — entries with risk_score > 0.5
    # ------------------------------------------------------------------
    trigger_predictions: list[BehaviorTriggerPrediction] = []

    for entry in phoneme_risk_ranking:
        if entry.risk_score <= 0.5:
            continue
        trigger_predictions.append(
            BehaviorTriggerPrediction(
                trigger_phoneme=entry.phoneme,
                triggered_behavior=_triggered_behavior(entry.phoneme, entry.risk_type),
                association_strength=entry.risk_score,
                is_intended=False,
                concern_level=_concern_level(entry.risk_score),
            )
        )

    # ------------------------------------------------------------------
    # 4. PhonemeCluster — group risk_score > 0.5 entries by risk_type
    # ------------------------------------------------------------------
    from collections import defaultdict

    groups: dict[str, list[PhonemeRiskEntry]] = defaultdict(list)
    for entry in phoneme_risk_ranking:
        if entry.risk_score > 0.5:
            groups[entry.risk_type].append(entry)

    risky_clusters: list[PhonemeCluster] = []
    for risk_type, members in groups.items():
        if len(members) < 2:
            continue
        phonemes_in_cluster = [m.phoneme for m in members]
        combined = sum(m.risk_score for m in members) / len(members)
        risky_clusters.append(
            PhonemeCluster(
                phonemes=phonemes_in_cluster,
                cluster_risk_type=risk_type,
                combined_risk_score=combined,
                interaction_description=(
                    f"{risk_type} cluster: phonemes {phonemes_in_cluster} show "
                    f"correlated anomalous canonical representations"
                ),
            )
        )

    # ------------------------------------------------------------------
    # 5. model_behavioral_safety from max risk score
    # ------------------------------------------------------------------
    if phoneme_risk_ranking:
        max_risk = max(e.risk_score for e in phoneme_risk_ranking)
    else:
        max_risk = 0.0

    if max_risk < 0.3:
        safety = "safe"
    elif max_risk < 0.5:
        safety = "minor_concerns"
    elif max_risk < 0.65:
        safety = "moderate_risk"
    elif max_risk < 0.8:
        safety = "high_risk"
    else:
        safety = "unsafe"

    # ------------------------------------------------------------------
    # 6. MitigationRecommendations — one per cluster
    # ------------------------------------------------------------------
    mitigation_recs: list[MitigationRecommendation] = [
        _mitigation(cluster) for cluster in risky_clusters
    ]

    # ------------------------------------------------------------------
    # 7. Summary string
    # ------------------------------------------------------------------
    top = phoneme_risk_ranking[0] if phoneme_risk_ranking else None
    summary = (
        f"Behavioral safety: {safety}. "
        f"{len(phoneme_risk_ranking)} phonemes flagged. "
        f"Top risk: {top.phoneme if top else 'none'} "
        f"({top.risk_type if top else 'n/a'})."
    )

    return PhonemeRiskAction(
        phoneme_risk_ranking=phoneme_risk_ranking,
        predicted_behavior_triggers=trigger_predictions,
        risky_phoneme_clusters=risky_clusters,
        model_behavioral_safety=safety,
        mitigation_recommendations=mitigation_recs,
        summary=summary,
    )
