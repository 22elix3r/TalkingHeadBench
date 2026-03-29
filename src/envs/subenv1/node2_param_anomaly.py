"""
Node 2: Parameter Anomaly Detector — Sub-env 1.

Pure rule-based heuristic: no LLM calls, no I/O, no side effects.
Receives the user's proposed generation config alongside diagnostic signals
forwarded from Node 1, and returns a structured ParamAnomalyAction.
"""

from src.schemas.subenv1 import (
    DirectionalFix,
    ParameterAnomaly,
    ParamAnomalyAction,
    ParamAnomalyObservation,
)


def detect_param_anomalies(obs: ParamAnomalyObservation) -> ParamAnomalyAction:
    """Detect parameter anomalies from a proposed generation config.

    Only keys present in ``obs.proposed_config`` are evaluated.  All decisions
    are deterministic and rule-based — no model inference, no external calls,
    no mutable state.

    Args:
        obs: Proposed config plus diagnostic signals from Node 1.

    Returns:
        A fully populated ParamAnomalyAction.
    """

    cfg = obs.proposed_config
    anomalies: list[ParameterAnomaly] = []

    # ------------------------------------------------------------------
    # Anomaly detection — check only keys present in proposed_config
    # ------------------------------------------------------------------

    # --- denoise_alt ---
    if "denoise_alt" in cfg:
        val = cfg["denoise_alt"]
        if obs.regime == "non_frontal" and val < 0.45:
            anomalies.append(
                ParameterAnomaly(
                    parameter="denoise_alt",
                    issue="too low for non-frontal regime — reference token loses lateral coverage",
                    severity="severe",
                    linked_failure_mode="reference_token_dropout",
                )
            )
        elif obs.regime == "complex_background" and val > 0.65:
            anomalies.append(
                ParameterAnomaly(
                    parameter="denoise_alt",
                    issue="too high for complex background — risks washing out identity",
                    severity="moderate",
                    linked_failure_mode="identity_collapse",
                )
            )

    # --- cfg ---
    if "cfg" in cfg:
        val = cfg["cfg"]
        if obs.background_complexity_score > 0.6 and val > 7.0:
            anomalies.append(
                ParameterAnomaly(
                    parameter="cfg",
                    issue="elevated CFG with complex background — attention bleeds into non-face regions",
                    severity="moderate",
                    linked_failure_mode="background_bleed",
                )
            )

    # --- eta ---
    if "eta" in cfg:
        val = cfg["eta"]
        if val > 0.12 and obs.image_usability_score < 0.5:
            anomalies.append(
                ParameterAnomaly(
                    parameter="eta",
                    issue="high stochasticity with weak reference — identity drifts across frames",
                    severity="moderate",
                    linked_failure_mode="identity_collapse",
                )
            )

    # ------------------------------------------------------------------
    # Config risk level
    # ------------------------------------------------------------------
    severities = [a.severity for a in anomalies]
    moderate_count = severities.count("moderate")

    if "severe" in severities:
        config_risk_level = "dangerous"
    elif moderate_count >= 2:
        config_risk_level = "risky"
    elif moderate_count == 1:
        config_risk_level = "marginal"
    else:
        config_risk_level = "safe"

    # ------------------------------------------------------------------
    # Predicted failure modes — unique, in order of first appearance
    # ------------------------------------------------------------------
    seen: set[str] = set()
    predicted_failure_modes: list[str] = []
    for a in anomalies:
        if a.linked_failure_mode not in seen:
            seen.add(a.linked_failure_mode)
            predicted_failure_modes.append(a.linked_failure_mode)

    # ------------------------------------------------------------------
    # Directional fixes — one per anomaly, keyed on failure mode + param
    # ------------------------------------------------------------------
    directional_fixes: list[DirectionalFix] = []
    for a in anomalies:
        if a.linked_failure_mode == "reference_token_dropout":
            directional_fixes.append(
                DirectionalFix(
                    target="reference_token_strength",
                    direction="increase",
                    rationale=(
                        "compensates for lateral pose — keeps identity anchored "
                        "in side-facing frames"
                    ),
                    priority="critical",
                )
            )
        elif a.linked_failure_mode == "identity_collapse" and a.parameter == "eta":
            directional_fixes.append(
                DirectionalFix(
                    target="stochasticity (eta)",
                    direction="decrease",
                    rationale="reduces frame-to-frame identity variance",
                    priority="critical",
                )
            )
        elif a.linked_failure_mode == "identity_collapse" and a.parameter == "denoise_alt":
            directional_fixes.append(
                DirectionalFix(
                    target="denoise_alt",
                    direction="decrease",
                    rationale="prevents identity washout in complex background",
                    priority="recommended",
                )
            )
        elif a.linked_failure_mode == "background_bleed":
            directional_fixes.append(
                DirectionalFix(
                    target="guidance_scale",
                    direction="decrease",
                    rationale="prevents attention competition with background elements",
                    priority="recommended",
                )
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(anomalies)
    summary = f"Config risk: {config_risk_level}. {n} anomaly/anomalies detected."
    if anomalies:
        # Top severity: prefer severe, then moderate, then minor
        top = max(anomalies, key=lambda a: {"severe": 2, "moderate": 1, "minor": 0}[a.severity])
        summary += f" Top issue ({top.severity}): {top.issue}"

    return ParamAnomalyAction(
        config_risk_level=config_risk_level,
        anomalies=anomalies,
        predicted_failure_modes=predicted_failure_modes,
        directional_fixes=directional_fixes,
        summary=summary,
    )
