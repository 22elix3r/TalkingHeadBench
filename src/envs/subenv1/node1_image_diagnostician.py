"""
Node 1: Image Diagnostician — Sub-env 1.

Pure rule-based heuristic: no LLM calls, no I/O, no side effects.
Receives pre-extracted image and prompt signals via ImageDiagnosticsObservation
and returns a structured ImageDiagnosticsAction.
"""

from src.schemas.subenv1 import ImageDiagnosticsAction, ImageDiagnosticsObservation


def diagnose_image(obs: ImageDiagnosticsObservation) -> ImageDiagnosticsAction:
    """Diagnose a reference image from pre-extracted signals.

    All decisions are deterministic and rule-based.  No model inference,
    no external calls, no mutable state.

    Args:
        obs: Pre-computed signals from the reference image and prompt text.

    Returns:
        A fully populated ImageDiagnosticsAction.
    """

    # ------------------------------------------------------------------
    # 1. Regime classification — first matching rule wins
    # ------------------------------------------------------------------
    if obs.face_occupancy_ratio < 0.25:
        regime = "low_quality"
    elif obs.occlusion_detected:
        regime = "occluded"
    elif abs(obs.estimated_yaw_degrees) > 25:
        regime = "non_frontal"
    elif obs.background_complexity_score > 0.7:
        regime = "complex_background"
    else:
        regime = "frontal_simple"

    # ------------------------------------------------------------------
    # 2. Risk factors — append all that apply (order matters for reasoning)
    # ------------------------------------------------------------------
    risk_factors: list[str] = []

    if abs(obs.estimated_yaw_degrees) > 25:
        risk_factors.append(
            "yaw exceeds 25° — lateral pose reduces reference token coverage"
        )
    if obs.lighting_uniformity_score < 0.4:
        risk_factors.append(
            "lighting uniformity low — shadow regions risk inconsistent skin rendering"
        )
    if obs.face_occupancy_ratio < 0.4:
        risk_factors.append(
            "face occupancy below 0.4 — background competes for attention budget"
        )
    if obs.occlusion_detected:
        risk_factors.append(
            "occlusion detected — reference tokens may encode obstruction"
        )
    if obs.estimated_sharpness < 0.3:
        risk_factors.append(
            "low sharpness — may cause blurred identity encoding"
        )

    # ------------------------------------------------------------------
    # 3. Prompt issues — conflicting descriptors + weak identity anchoring
    # ------------------------------------------------------------------
    prompt_issues: list[str] = []

    for item in obs.conflicting_descriptors:
        prompt_issues.append(f"'{item}' — contradictory descriptors")

    if obs.identity_anchoring_strength < 0.4:
        prompt_issues.append(
            "no explicit identity descriptor — prompt relies entirely on reference image"
        )

    # ------------------------------------------------------------------
    # 4. Image usability score — weighted sum, clipped to [0.0, 1.0]
    # ------------------------------------------------------------------
    raw_score = (
        0.30 * min(obs.face_occupancy_ratio, 1.0)
        + 0.20 * obs.lighting_uniformity_score
        + 0.20 * obs.estimated_sharpness
        + 0.20 * (1.0 - obs.background_complexity_score)
        + 0.10 * (0.0 if obs.occlusion_detected else 1.0)
    )
    usability_score = max(0.0, min(1.0, raw_score))

    # ------------------------------------------------------------------
    # 5. Reasoning — single f-string naming regime and primary risk factor
    # ------------------------------------------------------------------
    primary_risk = risk_factors[0] if risk_factors else "no significant risk factors"
    reasoning = (
        f"Regime '{regime}': primary concern is {primary_risk}."
    )

    # ------------------------------------------------------------------
    # 6. Recommended prompt modifications — one entry per prompt issue,
    #    rephrased as an actionable suggestion
    # ------------------------------------------------------------------
    recommended_modifications: list[str] = []

    for issue in prompt_issues:
        if "contradictory" in issue or "conflicting" in issue or "contradict" in issue:
            # Extract the descriptor name from "'{item}' — contradictory descriptors"
            item = issue.split("'")[1]
            recommended_modifications.append(f"resolve conflicting descriptors: {item}")
        elif "identity" in issue:
            recommended_modifications.append(
                "add explicit identity anchoring term"
            )

    return ImageDiagnosticsAction(
        regime_classification=regime,
        identified_risk_factors=risk_factors,
        prompt_issues=prompt_issues,
        recommended_prompt_modifications=recommended_modifications,
        image_usability_score=round(usability_score, 4),
        reasoning=reasoning,
    )
