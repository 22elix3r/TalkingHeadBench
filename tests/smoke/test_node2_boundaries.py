"""
Smoke tests: Node 2 (Parameter Anomaly Detector) boundary conditions.

Covers:
  1. Empty / unknown config keys produce safe output.
  2. Each rule fires only for its declared regime/context (and on the
     correct side of each threshold).
  3. Risk-level escalation logic (safe → marginal → risky → dangerous).
  4. Predicted failure-mode deduplication.
  5. Directional-fix count mirrors anomaly count.

All rules are verified at exact boundary values so we confirm ``<`` vs
``<=`` strictness throughout.
"""

from __future__ import annotations

from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.schemas.subenv1 import ParamAnomalyAction, ParamAnomalyObservation


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_DEFAULTS: dict = dict(
    proposed_config={},
    regime="frontal_simple",
    identified_risk_factors=[],
    image_usability_score=0.7,
    face_occupancy_ratio=0.6,
    estimated_yaw_degrees=5.0,
    background_complexity_score=0.3,
    lighting_uniformity_score=0.7,
    occlusion_detected=False,
    prompt_identity_anchoring=0.7,
    prompt_token_count=40,
    conflicting_descriptors=[],
)


def base_obs(**overrides) -> ParamAnomalyObservation:
    """Return a ParamAnomalyObservation in the safe zone with applied overrides.

    Defaults keep every rule dormant:
    - proposed_config empty → no parameter evaluated
    - regime "frontal_simple" → denoise_alt rule cannot fire
    - background_complexity_score 0.3 ≤ 0.6 → cfg rule cannot fire
    - image_usability_score 0.7 ≥ 0.5 → eta rule cannot fire
    """
    return ParamAnomalyObservation(**{**_DEFAULTS, **overrides})


def run(obs: ParamAnomalyObservation) -> ParamAnomalyAction:
    return detect_param_anomalies(obs)


# ===========================================================================
# 1. Empty / unknown config keys
# ===========================================================================


class TestEmptyAndUnknownConfig:
    """Only keys present in proposed_config are evaluated — unrecognised keys
    must be silently ignored, not raise or produce spurious anomalies."""

    def test_empty_config_produces_safe(self):
        """Empty proposed_config → safe level, no anomalies."""
        action = run(base_obs())
        assert action.config_risk_level == "safe"
        assert action.anomalies == []

    def test_unknown_param_keys_ignored(self):
        """Unrecognised parameter names produce no anomalies."""
        action = run(base_obs(proposed_config={"nonexistent_param": 99.9}))
        assert action.config_risk_level == "safe"
        assert action.anomalies == []

    def test_known_key_absent_is_safe_even_with_triggering_context(self):
        """If 'denoise_alt' is absent from proposed_config, the rule never runs
        even when the regime is 'non_frontal'."""
        obs = base_obs(regime="non_frontal", proposed_config={})
        action = run(obs)
        assert action.anomalies == []


# ===========================================================================
# 2. denoise_alt rule
# ===========================================================================


class TestDenoiseAltRule:
    """Rule: regime=='non_frontal' and val < 0.45  → severe / reference_token_dropout
             regime=='complex_background' and val > 0.65 → moderate / identity_collapse
    """

    # --- regime guard ---

    def test_denoise_alt_severe_only_for_non_frontal(self):
        """val=0.25 (< 0.45) with regime='frontal_simple' → no anomaly."""
        obs = base_obs(
            regime="frontal_simple",
            proposed_config={"denoise_alt": 0.25},
        )
        action = run(obs)
        assert not any(a.severity == "severe" for a in action.anomalies)

    def test_denoise_alt_severe_fires_for_non_frontal(self):
        """val=0.25 (< 0.45) with regime='non_frontal' → severe anomaly."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.25},
        )
        action = run(obs)
        assert any(
            a.severity == "severe" and a.parameter == "denoise_alt"
            for a in action.anomalies
        )

    def test_denoise_alt_non_frontal_linked_failure_mode(self):
        """Severe denoise_alt anomaly links to reference_token_dropout."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.25},
        )
        action = run(obs)
        assert any(
            a.linked_failure_mode == "reference_token_dropout"
            for a in action.anomalies
        )

    # --- exact boundary: val < 0.45 (strict) ---

    def test_denoise_alt_exact_boundary_045_no_anomaly(self):
        """val == 0.45 is NOT < 0.45 → no anomaly for non_frontal."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.45},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_denoise_alt_exact_boundary_0449_anomaly(self):
        """val == 0.449 < 0.45 → severe anomaly for non_frontal."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.449},
        )
        action = run(obs)
        assert any(
            a.severity == "severe" and a.parameter == "denoise_alt"
            for a in action.anomalies
        )

    # --- complex_background branch ---

    def test_denoise_alt_complex_background_high_value(self):
        """val=0.70 > 0.65 with regime='complex_background' → moderate anomaly."""
        obs = base_obs(
            regime="complex_background",
            proposed_config={"denoise_alt": 0.70},
        )
        action = run(obs)
        assert any(
            a.severity == "moderate" and a.parameter == "denoise_alt"
            for a in action.anomalies
        )

    def test_denoise_alt_complex_background_exact_boundary_065_no_anomaly(self):
        """val == 0.65 is NOT > 0.65 → no anomaly for complex_background."""
        obs = base_obs(
            regime="complex_background",
            proposed_config={"denoise_alt": 0.65},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_denoise_alt_complex_background_just_above_065(self):
        """val == 0.651 > 0.65 → moderate anomaly for complex_background."""
        obs = base_obs(
            regime="complex_background",
            proposed_config={"denoise_alt": 0.651},
        )
        action = run(obs)
        assert any(
            a.severity == "moderate" and a.parameter == "denoise_alt"
            for a in action.anomalies
        )


# ===========================================================================
# 3. cfg rule
# ===========================================================================


class TestCfgRule:
    """Rule: background_complexity_score > 0.6 AND cfg_val > 7.0 → moderate / background_bleed."""

    def test_cfg_only_fires_with_complex_background_above_06(self):
        """bg_complexity == 0.59 ≤ 0.6 → cfg rule cannot fire even with high cfg."""
        obs = base_obs(
            background_complexity_score=0.59,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_cfg_fires_when_bg_above_06(self):
        """bg_complexity == 0.61 > 0.6 and cfg == 8.0 > 7.0 → moderate anomaly."""
        obs = base_obs(
            background_complexity_score=0.61,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        assert any(
            a.severity == "moderate" and a.parameter == "cfg"
            for a in action.anomalies
        )

    def test_cfg_exact_bg_boundary_06_no_anomaly(self):
        """bg_complexity == 0.6 is NOT > 0.6 → no anomaly."""
        obs = base_obs(
            background_complexity_score=0.6,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_cfg_exact_val_boundary_70_no_anomaly(self):
        """cfg == 7.0 is NOT > 7.0 → no anomaly (bg_complexity safely above 0.6)."""
        obs = base_obs(
            background_complexity_score=0.7,
            proposed_config={"cfg": 7.0},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_cfg_just_above_70_fires(self):
        """cfg == 7.01 > 7.0 with bg_complexity=0.7 > 0.6 → anomaly."""
        obs = base_obs(
            background_complexity_score=0.7,
            proposed_config={"cfg": 7.01},
        )
        action = run(obs)
        assert any(a.parameter == "cfg" for a in action.anomalies)

    def test_cfg_linked_failure_mode_is_background_bleed(self):
        """cfg anomaly always links to background_bleed."""
        obs = base_obs(
            background_complexity_score=0.7,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        assert any(
            a.linked_failure_mode == "background_bleed" for a in action.anomalies
        )


# ===========================================================================
# 4. eta rule
# ===========================================================================


class TestEtaRule:
    """Rule: eta_val > 0.12 AND image_usability_score < 0.5 → moderate / identity_collapse."""

    def test_eta_only_fires_with_weak_reference_usability_below_05(self):
        """usability == 0.5 is NOT < 0.5 → eta rule cannot fire."""
        obs = base_obs(
            image_usability_score=0.5,
            proposed_config={"eta": 0.13},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_eta_fires_when_usability_below_05(self):
        """usability == 0.49 < 0.5 and eta == 0.13 > 0.12 → moderate anomaly."""
        obs = base_obs(
            image_usability_score=0.49,
            proposed_config={"eta": 0.13},
        )
        action = run(obs)
        assert any(
            a.severity == "moderate" and a.parameter == "eta"
            for a in action.anomalies
        )

    def test_eta_exact_val_boundary_012_no_anomaly(self):
        """eta == 0.12 is NOT > 0.12 → no anomaly (usability safely below 0.5)."""
        obs = base_obs(
            image_usability_score=0.4,
            proposed_config={"eta": 0.12},
        )
        action = run(obs)
        assert action.anomalies == []

    def test_eta_just_above_012_fires(self):
        """eta == 0.121 > 0.12 with usability=0.4 < 0.5 → anomaly."""
        obs = base_obs(
            image_usability_score=0.4,
            proposed_config={"eta": 0.121},
        )
        action = run(obs)
        assert any(a.parameter == "eta" for a in action.anomalies)

    def test_eta_linked_failure_mode_is_identity_collapse(self):
        """eta anomaly always links to identity_collapse."""
        obs = base_obs(
            image_usability_score=0.4,
            proposed_config={"eta": 0.13},
        )
        action = run(obs)
        assert any(
            a.linked_failure_mode == "identity_collapse" for a in action.anomalies
        )


# ===========================================================================
# 5. Config risk level escalation
# ===========================================================================


class TestRiskLevelEscalation:
    """safe → marginal → risky → dangerous strictly via severity counts."""

    def test_risk_level_safe_no_anomalies(self):
        """No anomalies → safe."""
        action = run(base_obs())
        assert action.config_risk_level == "safe"

    def test_risk_level_marginal_one_moderate(self):
        """One moderate anomaly → marginal (not risky)."""
        # cfg moderate: bg_complexity=0.7, cfg=8.0
        obs = base_obs(
            background_complexity_score=0.7,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        assert action.config_risk_level == "marginal"

    def test_risk_level_risky_requires_two_moderate(self):
        """Two moderate anomalies → risky.

        cfg (background_bleed) + eta (identity_collapse):
          bg_complexity=0.7 > 0.6, cfg=8.0 > 7.0
          usability=0.4 < 0.5, eta=0.13 > 0.12
        """
        obs = base_obs(
            background_complexity_score=0.7,
            image_usability_score=0.4,
            proposed_config={"cfg": 8.0, "eta": 0.13},
        )
        action = run(obs)
        assert action.config_risk_level == "risky"

    def test_risk_level_dangerous_requires_severe(self):
        """One severe anomaly → dangerous, regardless of moderate count."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.25},
        )
        action = run(obs)
        assert action.config_risk_level == "dangerous"

    def test_risk_level_dangerous_overrides_moderate(self):
        """Severe + moderate simultaneously → still dangerous (not risky)."""
        obs = base_obs(
            regime="non_frontal",
            background_complexity_score=0.7,
            proposed_config={"denoise_alt": 0.25, "cfg": 8.0},
        )
        action = run(obs)
        assert action.config_risk_level == "dangerous"

    def test_only_moderate_anomalies_never_dangerous(self):
        """Two moderate anomalies → risky, never dangerous."""
        obs = base_obs(
            background_complexity_score=0.7,
            image_usability_score=0.4,
            proposed_config={"cfg": 8.0, "eta": 0.13},
        )
        action = run(obs)
        assert action.config_risk_level != "dangerous"


# ===========================================================================
# 6. Failure-mode deduplication
# ===========================================================================


class TestFailureModeDeduplicated:
    """predicted_failure_modes must contain each mode at most once."""

    def test_predicted_failure_modes_deduplicated(self):
        """Two anomalies with the same linked_failure_mode='identity_collapse':
        denoise_alt in complex_background (val=0.70 > 0.65) + eta (val=0.13,
        usability=0.4 < 0.5). Only one 'identity_collapse' entry expected.
        """
        obs = base_obs(
            regime="complex_background",
            image_usability_score=0.4,
            proposed_config={"denoise_alt": 0.70, "eta": 0.13},
        )
        action = run(obs)
        # Both anomalies link to identity_collapse
        assert len(action.anomalies) == 2
        assert action.predicted_failure_modes.count("identity_collapse") == 1

    def test_predicted_failure_modes_order_is_insertion_order(self):
        """Failure modes appear in the order the anomalies were detected.

        denoise_alt is checked before eta in the implementation, so
        reference_token_dropout should come before identity_collapse.
        """
        obs = base_obs(
            regime="non_frontal",
            image_usability_score=0.4,
            proposed_config={"denoise_alt": 0.25, "eta": 0.13},
        )
        action = run(obs)
        modes = action.predicted_failure_modes
        assert modes.index("reference_token_dropout") < modes.index("identity_collapse")

    def test_multiple_distinct_failure_modes_all_present(self):
        """Three anomalies with three distinct failure modes → three entries."""
        # reference_token_dropout (denoise_alt/non_frontal)
        # background_bleed (cfg)
        # identity_collapse (eta)
        obs = base_obs(
            regime="non_frontal",
            background_complexity_score=0.7,
            image_usability_score=0.4,
            proposed_config={"denoise_alt": 0.25, "cfg": 8.0, "eta": 0.13},
        )
        action = run(obs)
        assert len(action.predicted_failure_modes) == 3
        assert set(action.predicted_failure_modes) == {
            "reference_token_dropout",
            "background_bleed",
            "identity_collapse",
        }


# ===========================================================================
# 7. Directional fix count
# ===========================================================================


class TestDirectionalFixes:
    """One directional fix is generated per anomaly for every known rule branch."""

    def test_no_fixes_for_safe_config(self):
        """No anomalies → empty directional_fixes list."""
        action = run(base_obs())
        assert action.directional_fixes == []

    def test_directional_fixes_one_per_anomaly(self):
        """Three anomalies (reference_token_dropout + background_bleed +
        identity_collapse/eta) → three directional fixes.

        Uses non_frontal to trigger denoise_alt severe + cfg moderate +
        eta moderate.  background_complexity_score > 0.6 for cfg;
        image_usability_score < 0.5 for eta.
        """
        obs = base_obs(
            regime="non_frontal",
            background_complexity_score=0.7,
            image_usability_score=0.4,
            proposed_config={"denoise_alt": 0.25, "cfg": 8.0, "eta": 0.13},
        )
        action = run(obs)
        assert len(action.anomalies) == 3
        assert len(action.directional_fixes) == 3

    def test_fix_for_reference_token_dropout_increases(self):
        """reference_token_dropout → fix direction is 'increase'."""
        obs = base_obs(
            regime="non_frontal",
            proposed_config={"denoise_alt": 0.25},
        )
        action = run(obs)
        td_fixes = [f for f in action.directional_fixes if f.target == "reference_token_strength"]
        assert len(td_fixes) == 1
        assert td_fixes[0].direction == "increase"

    def test_fix_for_background_bleed_decreases(self):
        """background_bleed → fix direction is 'decrease' on guidance_scale."""
        obs = base_obs(
            background_complexity_score=0.7,
            proposed_config={"cfg": 8.0},
        )
        action = run(obs)
        bg_fixes = [f for f in action.directional_fixes if f.target == "guidance_scale"]
        assert len(bg_fixes) == 1
        assert bg_fixes[0].direction == "decrease"

    def test_fix_for_eta_identity_collapse_decreases(self):
        """identity_collapse via eta → fix direction is 'decrease' on stochasticity."""
        obs = base_obs(
            image_usability_score=0.4,
            proposed_config={"eta": 0.13},
        )
        action = run(obs)
        eta_fixes = [f for f in action.directional_fixes if "eta" in f.target]
        assert len(eta_fixes) == 1
        assert eta_fixes[0].direction == "decrease"
