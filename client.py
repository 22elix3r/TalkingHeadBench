"""TalkingHeadBench environment client.

Provides typed helper methods for interacting with a running
TalkingHeadBench OpenEnv server.

Usage:
    from client import TalkingHeadBenchEnv

    with TalkingHeadBenchEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        obs = env.step(action_1)
        obs = env.step(action_2)
        obs = env.step(action_3)
        print(obs.reward)
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_client import EnvClient


class TalkingHeadBenchEnv(EnvClient):
    """Client wrapper for a running TalkingHeadBench OpenEnv server.

    Episode flow:
        reset()  -> ImageDiagnosticsObservation (Node 1 signals)
        step(ImageDiagnosticsAction)  -> ParamAnomalyObservation
        step(ParamAnomalyAction)      -> PhonemeRiskObservation
        step(PhonemeRiskAction)       -> done=True, final_score

    Final score formula:
        0.25 x subenv1 + 0.35 x subenv2 + 0.40 x subenv3
    """

    STEP_SCHEMAS: list[str] = [
        "ImageDiagnosticsAction",
        "ParamAnomalyAction",
        "PhonemeRiskAction",
    ]

    @staticmethod
    def expected_action_schema(step: int) -> str:
        """Return the expected action schema name for a given step index."""
        if 0 <= step < len(TalkingHeadBenchEnv.STEP_SCHEMAS):
            return TalkingHeadBenchEnv.STEP_SCHEMAS[step]
        return "unknown"

    @staticmethod
    def make_minimal_action(step: int) -> dict[str, Any]:
        """Create a minimal valid action dict for the given step.

        Useful as a fallback when the LLM fails to produce valid output.
        """
        if step == 0:
            return {
                "regime_classification": "frontal_simple",
                "identified_risk_factors": [],
                "prompt_issues": [],
                "recommended_prompt_modifications": [],
                "image_usability_score": 0.5,
                "reasoning": "Minimal fallback action.",
            }
        elif step == 1:
            return {
                "config_risk_level": "marginal",
                "anomalies": [],
                "predicted_failure_modes": [],
                "directional_fixes": [],
                "summary": "Minimal fallback action.",
            }
        elif step == 2:
            return {
                "phoneme_risk_ranking": [],
                "predicted_behavior_triggers": [],
                "risky_phoneme_clusters": [],
                "model_behavioral_safety": "minor_concerns",
                "mitigation_recommendations": [],
                "summary": "Minimal fallback action.",
            }
        else:
            return {}