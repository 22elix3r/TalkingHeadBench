"""
Smoke tests: Node 1 (Image Diagnostician) boundary conditions.

Covers:
  1. Regime classification priority order and exact boundary values.
  2. Risk factor accumulation — all-at-once and none.
  3. Prompt issue propagation (conflicting descriptors, identity anchoring).
  4. image_usability_score formula: perfect, worst, and clipped to [0, 1].

All tests use rule-based deterministic code — no mocking required.
"""

from __future__ import annotations

import random

import pytest

from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.schemas.subenv1 import ImageDiagnosticsObservation


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_DEFAULTS: dict = dict(
    face_occupancy_ratio=0.6,
    estimated_yaw_degrees=5.0,
    estimated_pitch_degrees=2.0,
    background_complexity_score=0.3,
    lighting_uniformity_score=0.7,
    skin_tone_bucket=3,
    occlusion_detected=False,
    image_resolution=(1280, 720),
    estimated_sharpness=0.65,
    prompt_token_count=40,
    prompt_semantic_density=0.5,
    conflicting_descriptors=[],
    identity_anchoring_strength=0.7,
)


def base_obs(**overrides) -> ImageDiagnosticsObservation:
    """Return an ImageDiagnosticsObservation in frontal_simple territory.

    The defaults place every signal well inside the safe zone:
    no risk factors fire, regime resolves to ``"frontal_simple"``.
    Pass keyword overrides to move specific signals to boundary values.
    """
    return ImageDiagnosticsObservation(**{**_DEFAULTS, **overrides})


# ===========================================================================
# 1. Regime classification — priority order and exact boundaries
# ===========================================================================


class TestRegimeClassification:
    """First-match-wins priority: low_quality > occluded > non_frontal > complex_background."""

    def test_regime_low_quality_takes_priority_over_occlusion(self):
        """face_occupancy < 0.25 wins even when occlusion_detected=True."""
        obs = base_obs(face_occupancy_ratio=0.20, occlusion_detected=True)
        action = diagnose_image(obs)
        assert action.regime_classification == "low_quality"

    def test_regime_occlusion_before_non_frontal(self):
        """occlusion_detected wins over |yaw| > 25 (occupancy is safe)."""
        obs = base_obs(
            face_occupancy_ratio=0.5,
            occlusion_detected=True,
            estimated_yaw_degrees=35.0,
        )
        action = diagnose_image(obs)
        assert action.regime_classification == "occluded"

    def test_regime_non_frontal_before_complex_background(self):
        """|yaw| > 25 wins over high background_complexity."""
        obs = base_obs(
            estimated_yaw_degrees=30.0,
            background_complexity_score=0.8,
        )
        action = diagnose_image(obs)
        assert action.regime_classification == "non_frontal"

    def test_regime_complex_background(self):
        """background_complexity > 0.7 triggers complex_background when higher rules don't fire."""
        obs = base_obs(background_complexity_score=0.71)
        action = diagnose_image(obs)
        assert action.regime_classification == "complex_background"

    def test_regime_frontal_simple_baseline(self):
        """Default observation (safe on all axes) → frontal_simple."""
        action = diagnose_image(base_obs())
        assert action.regime_classification == "frontal_simple"

    # --- Exact boundary: yaw = 25 (NOT strictly >) ---

    def test_regime_exact_yaw_boundary_25_not_non_frontal(self):
        """yaw == 25.0 is NOT > 25 → stays frontal_simple."""
        obs = base_obs(estimated_yaw_degrees=25.0)
        action = diagnose_image(obs)
        assert action.regime_classification == "frontal_simple"

    def test_regime_exact_yaw_boundary_25_1_is_non_frontal(self):
        """yaw == 25.1 is > 25 → non_frontal."""
        obs = base_obs(estimated_yaw_degrees=25.1)
        action = diagnose_image(obs)
        assert action.regime_classification == "non_frontal"

    def test_regime_exact_yaw_negative_boundary(self):
        """Negative yaw: -25.0 is NOT strictly > 25 in abs → frontal_simple."""
        obs = base_obs(estimated_yaw_degrees=-25.0)
        action = diagnose_image(obs)
        assert action.regime_classification == "frontal_simple"

    def test_regime_exact_yaw_negative_over_boundary(self):
        """Negative yaw: -25.1 → |yaw| = 25.1 > 25 → non_frontal."""
        obs = base_obs(estimated_yaw_degrees=-25.1)
        action = diagnose_image(obs)
        assert action.regime_classification == "non_frontal"

    # --- Exact boundary: occupancy = 0.25 (NOT strictly <) ---

    def test_regime_exact_occupancy_boundary_025_not_low_quality(self):
        """face_occupancy_ratio == 0.25 is NOT < 0.25 → does not trigger low_quality."""
        obs = base_obs(face_occupancy_ratio=0.25)
        action = diagnose_image(obs)
        assert action.regime_classification != "low_quality"

    def test_regime_exact_occupancy_boundary_0249_is_low_quality(self):
        """face_occupancy_ratio == 0.249 < 0.25 → low_quality."""
        obs = base_obs(face_occupancy_ratio=0.249)
        action = diagnose_image(obs)
        assert action.regime_classification == "low_quality"

    # --- Exact boundary: background_complexity = 0.7 (NOT strictly >) ---

    def test_regime_background_complexity_exactly_07_not_complex(self):
        """background_complexity == 0.7 is NOT > 0.7 → frontal_simple."""
        obs = base_obs(background_complexity_score=0.7)
        action = diagnose_image(obs)
        assert action.regime_classification == "frontal_simple"

    def test_regime_background_complexity_just_above_07(self):
        """background_complexity == 0.701 > 0.7 → complex_background."""
        obs = base_obs(background_complexity_score=0.701)
        action = diagnose_image(obs)
        assert action.regime_classification == "complex_background"


# ===========================================================================
# 2. Risk factor accumulation
# ===========================================================================


class TestRiskFactors:
    """Each risk-factor rule fires independently; all can fire simultaneously."""

    def test_all_risk_factors_simultaneously(self):
        """Five conditions active → five risk factors.

        Triggering values (from spec):
          |yaw| > 25      → yaw=30.0
          lighting < 0.4  → lighting=0.3
          occupancy < 0.4 → occupancy=0.3
          occlusion=True
          sharpness < 0.3 → sharpness=0.2
        """
        obs = base_obs(
            estimated_yaw_degrees=30.0,
            lighting_uniformity_score=0.3,
            face_occupancy_ratio=0.3,
            occlusion_detected=True,
            estimated_sharpness=0.2,
        )
        action = diagnose_image(obs)
        assert len(action.identified_risk_factors) == 5

    def test_no_risk_factors_clean_image(self):
        """Defaults are all in safe ranges → empty risk-factor list."""
        action = diagnose_image(base_obs())
        assert action.identified_risk_factors == []

    # --- Individual trigger boundaries ---

    def test_risk_yaw_fires_above_25(self):
        """yaw = 25.1 (> 25) → yaw risk factor present."""
        obs = base_obs(estimated_yaw_degrees=25.1)
        action = diagnose_image(obs)
        assert any("yaw" in rf for rf in action.identified_risk_factors)

    def test_risk_yaw_not_fired_at_25(self):
        """yaw = 25.0 (NOT > 25) → yaw risk factor absent."""
        obs = base_obs(estimated_yaw_degrees=25.0)
        action = diagnose_image(obs)
        assert not any("yaw" in rf for rf in action.identified_risk_factors)

    def test_risk_lighting_fires_below_04(self):
        """lighting = 0.39 < 0.4 → lighting risk factor present."""
        obs = base_obs(lighting_uniformity_score=0.39)
        action = diagnose_image(obs)
        assert any("lighting" in rf for rf in action.identified_risk_factors)

    def test_risk_lighting_not_fired_at_04(self):
        """lighting = 0.40 (NOT < 0.4) → lighting risk factor absent."""
        obs = base_obs(lighting_uniformity_score=0.40)
        action = diagnose_image(obs)
        assert not any("lighting" in rf for rf in action.identified_risk_factors)

    def test_risk_occupancy_fires_below_04(self):
        """occupancy = 0.39 < 0.4 → occupancy risk factor present."""
        obs = base_obs(face_occupancy_ratio=0.39)
        action = diagnose_image(obs)
        assert any("occupancy" in rf for rf in action.identified_risk_factors)

    def test_risk_occupancy_not_fired_at_04(self):
        """occupancy = 0.40 (NOT < 0.4) → occupancy risk factor absent."""
        obs = base_obs(face_occupancy_ratio=0.40)
        action = diagnose_image(obs)
        assert not any("occupancy" in rf for rf in action.identified_risk_factors)

    def test_risk_sharpness_fires_below_03(self):
        """sharpness = 0.29 < 0.3 → sharpness risk factor present."""
        obs = base_obs(estimated_sharpness=0.29)
        action = diagnose_image(obs)
        assert any("sharpness" in rf for rf in action.identified_risk_factors)

    def test_risk_sharpness_not_fired_at_03(self):
        """sharpness = 0.30 (NOT < 0.3) → sharpness risk factor absent."""
        obs = base_obs(estimated_sharpness=0.30)
        action = diagnose_image(obs)
        assert not any("sharpness" in rf for rf in action.identified_risk_factors)

    def test_risk_occlusion_flag(self):
        """occlusion_detected=True → occlusion risk factor present."""
        obs = base_obs(occlusion_detected=True)
        action = diagnose_image(obs)
        assert any("occlusion" in rf for rf in action.identified_risk_factors)


# ===========================================================================
# 3. Prompt issues
# ===========================================================================


class TestPromptIssues:
    """Conflicting-descriptor propagation and identity-anchoring threshold."""

    def test_conflicting_descriptors_propagate(self):
        """Two conflicting descriptors → two prompt issues, each containing 'contradictory'."""
        obs = base_obs(
            conflicting_descriptors=[
                "dramatic lighting / natural look",
                "sharp / soft focus",
            ]
        )
        action = diagnose_image(obs)
        assert len(action.prompt_issues) >= 2
        assert all("contradictory" in issue for issue in action.prompt_issues)

    def test_conflicting_descriptor_single(self):
        """One conflicting descriptor → exactly one prompt issue with the descriptor text."""
        descriptor = "aged / youthful appearance"
        obs = base_obs(conflicting_descriptors=[descriptor])
        action = diagnose_image(obs)
        assert len(action.prompt_issues) == 1
        assert descriptor in action.prompt_issues[0]
        assert "contradictory" in action.prompt_issues[0]

    def test_identity_anchoring_below_threshold(self):
        """identity_anchoring_strength = 0.39 < 0.4 → prompt_issues contains an identity issue."""
        obs = base_obs(identity_anchoring_strength=0.39)
        action = diagnose_image(obs)
        assert any("identity" in issue for issue in action.prompt_issues)

    def test_identity_anchoring_at_threshold(self):
        """identity_anchoring_strength = 0.40 (NOT < 0.4) → no identity prompt issue."""
        obs = base_obs(identity_anchoring_strength=0.40)
        action = diagnose_image(obs)
        assert not any("identity" in issue for issue in action.prompt_issues)

    def test_identity_anchoring_above_threshold(self):
        """identity_anchoring_strength = 0.7 (default) → no identity prompt issue."""
        action = diagnose_image(base_obs())
        assert not any("identity" in issue for issue in action.prompt_issues)

    def test_no_prompt_issues_clean_input(self):
        """Empty conflicting_descriptors + strong identity anchoring → no prompt issues."""
        action = diagnose_image(base_obs())
        assert action.prompt_issues == []

    def test_recommended_modifications_one_per_conflicting_descriptor(self):
        """Each conflicting descriptor generates a 'resolve conflicting descriptors' entry."""
        obs = base_obs(
            conflicting_descriptors=["bold colours / muted palette", "wide angle / portrait"],
        )
        action = diagnose_image(obs)
        resolve_mods = [
            m for m in action.recommended_prompt_modifications
            if "resolve conflicting descriptors" in m
        ]
        assert len(resolve_mods) == 2

    def test_recommended_modifications_identity_anchor(self):
        """Weak anchoring → recommended_modifications contains an 'identity anchoring' entry."""
        obs = base_obs(identity_anchoring_strength=0.2)
        action = diagnose_image(obs)
        assert any(
            "identity" in mod for mod in action.recommended_prompt_modifications
        )


# ===========================================================================
# 4. image_usability_score
# ===========================================================================


class TestImageUsabilityScore:
    """Verify the weighted-sum formula and clipping to [0.0, 1.0]."""

    def test_usability_score_perfect_image(self):
        """All signals at their best → usability == 1.0.

        raw = 0.30*1.0 + 0.20*1.0 + 0.20*1.0 + 0.20*(1-0.0) + 0.10*1.0
            = 0.30 + 0.20 + 0.20 + 0.20 + 0.10 = 1.00
        """
        obs = base_obs(
            face_occupancy_ratio=1.0,
            lighting_uniformity_score=1.0,
            estimated_sharpness=1.0,
            background_complexity_score=0.0,
            occlusion_detected=False,
        )
        action = diagnose_image(obs)
        assert action.image_usability_score == pytest.approx(1.0)

    def test_usability_score_worst_image(self):
        """All signals at their worst → usability == 0.0.

        raw = 0.30*0.0 + 0.20*0.0 + 0.20*0.0 + 0.20*(1-1.0) + 0.10*0.0
            = 0.0
        clip → 0.0
        """
        obs = base_obs(
            face_occupancy_ratio=0.0,
            lighting_uniformity_score=0.0,
            estimated_sharpness=0.0,
            background_complexity_score=1.0,
            occlusion_detected=True,
        )
        action = diagnose_image(obs)
        assert action.image_usability_score == pytest.approx(0.0)

    def test_usability_score_known_formula(self):
        """Spot-check the weighted formula against a manually computed value.

        Inputs:
          face_occupancy_ratio       = 0.8  → 0.30 * 0.8 = 0.240
          lighting_uniformity_score  = 0.6  → 0.20 * 0.6 = 0.120
          estimated_sharpness        = 0.5  → 0.20 * 0.5 = 0.100
          background_complexity      = 0.4  → 0.20 * 0.6 = 0.120
          occlusion_detected         = False → 0.10 * 1.0 = 0.100
          raw = 0.680  → round(0.680, 4) = 0.68
        """
        obs = base_obs(
            face_occupancy_ratio=0.8,
            lighting_uniformity_score=0.6,
            estimated_sharpness=0.5,
            background_complexity_score=0.4,
            occlusion_detected=False,
        )
        action = diagnose_image(obs)
        assert action.image_usability_score == pytest.approx(0.68, abs=1e-4)

    def test_usability_score_occlusion_penalty(self):
        """occlusion_detected=True removes the 0.10 occlusion bonus.

        Using defaults (occ_ratio=0.6, light=0.7, sharpness=0.65, bg=0.3):
          no_occ  = 0.30*0.6 + 0.20*0.7 + 0.20*0.65 + 0.20*0.7 + 0.10*1.0 = 0.71
          with_occ = same - 0.10 = 0.61
        """
        obs_clean = base_obs(occlusion_detected=False)
        obs_occ   = base_obs(occlusion_detected=True)
        score_clean = diagnose_image(obs_clean).image_usability_score
        score_occ   = diagnose_image(obs_occ).image_usability_score
        assert abs(score_clean - score_occ - 0.10) < 1e-4

    @pytest.mark.parametrize("seed", range(20))
    def test_usability_score_clipped_to_range(self, seed: int):
        """Any valid observation produces a usability score in [0.0, 1.0].

        Uses 20 independent random seeds for reproducibility.
        """
        rng = random.Random(seed)

        # Draw random signals; use values that span the full range to stress
        # the clipping logic.
        obs = base_obs(
            face_occupancy_ratio=rng.uniform(0.0, 1.0),
            lighting_uniformity_score=rng.uniform(0.0, 1.0),
            estimated_sharpness=rng.uniform(0.0, 1.0),
            background_complexity_score=rng.uniform(0.0, 1.0),
            occlusion_detected=rng.choice([True, False]),
        )
        action = diagnose_image(obs)
        assert 0.0 <= action.image_usability_score <= 1.0
