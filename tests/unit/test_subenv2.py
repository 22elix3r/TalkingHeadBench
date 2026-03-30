"""
Integration tests for Sub-env 2: Dataset Clip Audit.

Covers Node 5 (Clip Disposition Recommender) and Node 6 grader
(DatasetHealthHandoff aggregation) without any real video files or heavy
model dependencies.  All objects are constructed manually from schema classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.pipeline import _build_dataset_health_handoff
from src.schemas.ground_truth import GroundTruthClipAnnotation
from src.schemas.subenv2 import (
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
    SyntheticWeightDescriptor,
)


# ---------------------------------------------------------------------------
# Shared dataset-level observation parameters (same for all three clips)
# ---------------------------------------------------------------------------

_SHARED_OBS_KWARGS = dict(
    minimum_clips_needed=20,
    phoneme_gap_severity={"ZH": 2, "TH": 1},
    pose_gap_severity={},
    budget_remaining=10,
    reference_risk_profile="medium",
    estimated_drift_risk=0.4,
    marginal_training_damage=0.2,
    marginal_coverage_gain=0.5,
)


# ---------------------------------------------------------------------------
# Evidence dossiers
# ---------------------------------------------------------------------------


@pytest.fixture
def dossier_a() -> ClipEvidenceDossier:
    """Clean clip — should be accepted."""
    return ClipEvidenceDossier(
        clip_id="clip_001",
        identity_drift_severity="none",
        temporal_instability_flag=False,
        lip_sync_quality="good",
        unique_phoneme_value=0.8,
        dataset_redundancy_score=0.1,
        estimated_training_impact="positive",
        primary_rejection_reason=None,
        evidence_summary="clean clip",
    )


@pytest.fixture
def dossier_b() -> ClipEvidenceDossier:
    """Bad clip — high drift, temporal instability, poor sync, redundant."""
    return ClipEvidenceDossier(
        clip_id="clip_002",
        identity_drift_severity="severe",
        temporal_instability_flag=True,
        lip_sync_quality="poor",
        unique_phoneme_value=0.1,
        dataset_redundancy_score=0.9,
        estimated_training_impact="negative",
        primary_rejection_reason="severe drift",
        evidence_summary="bad clip",
    )


@pytest.fixture
def dossier_c() -> ClipEvidenceDossier:
    """Borderline clip — moderate drift, acceptable sync, mid-range phoneme value."""
    return ClipEvidenceDossier(
        clip_id="clip_003",
        identity_drift_severity="moderate",
        temporal_instability_flag=False,
        lip_sync_quality="acceptable",
        unique_phoneme_value=0.6,
        dataset_redundancy_score=0.3,
        estimated_training_impact="neutral",
        primary_rejection_reason=None,
        evidence_summary="borderline clip",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(dossier: ClipEvidenceDossier) -> ClipDispositionObservation:
    """Wrap a dossier into a ClipDispositionObservation with the shared context."""
    return ClipDispositionObservation(
        evidence_dossier=dossier,
        **_SHARED_OBS_KWARGS,
    )


def _minimal_clip_signal_obs(clip_id: str) -> ClipSignalObservation:
    """Return a ClipSignalObservation with all-zero / empty values for aggregation."""
    return ClipSignalObservation(
        clip_id=clip_id,
        face_embedding_variance=0.0,
        landmark_stability_score=0.0,
        identity_cosine_drift=0.0,
        frame_difference_mean=0.0,
        optical_flow_magnitude=1.0,
        blink_count=0,
        lip_sync_confidence=0.0,
        phoneme_sequence=[],
        phoneme_coverage_new=0.5,
        blur_score=0.8,
        exposure_score=0.7,
        occlusion_frames=0,
        clips_audited_so_far=0,
        current_phoneme_coverage={},
        current_pose_distribution={},
        similar_clips_accepted=0,
    )


def _minimal_gt(
    disposition: str,
    confidence: float = 0.5,
    ambiguity: float = 0.0,
) -> GroundTruthClipAnnotation:
    """Build a minimal GroundTruthClipAnnotation for grading."""
    return GroundTruthClipAnnotation(
        disposition=disposition,
        confidence=confidence,
        disposition_ambiguity=ambiguity,
        valid_fix_steps=[],
        valid_override_justifications=[],
        expected_reasoning_elements=["zh"],
    )


# ---------------------------------------------------------------------------
# Test block 1 — Node 5: Clip Disposition Recommender
# ---------------------------------------------------------------------------


class TestNode5Disposition:
    """Tests for recommend_clip_disposition()."""

    def test_disposition_a_accept(self, dossier_a):
        """Clean clip must be accepted."""
        action = recommend_clip_disposition(_make_obs(dossier_a))
        assert action.disposition == "accept"

    def test_disposition_b_reject_or_fix(self, dossier_b):
        """Severely degraded clip must be rejected or sent for fix."""
        action = recommend_clip_disposition(_make_obs(dossier_b))
        assert action.disposition in {"reject", "fix"}

    def test_disposition_c_reasonable(self, dossier_c):
        """Borderline clip must land on fix, defer, or accept (not outright reject)."""
        action = recommend_clip_disposition(_make_obs(dossier_c))
        assert action.disposition in {"fix", "defer", "accept"}

    def test_confidence_in_range_all(self, dossier_a, dossier_b, dossier_c):
        """Confidence must be in [0.0, 1.0] for every clip."""
        for dossier in (dossier_a, dossier_b, dossier_c):
            action = recommend_clip_disposition(_make_obs(dossier))
            assert 0.0 <= action.confidence <= 1.0, (
                f"confidence={action.confidence} out of range for {dossier.clip_id}"
            )

    def test_override_decision_valid_literal(self, dossier_a, dossier_b, dossier_c):
        """override_decision must be one of the three allowed Literal values."""
        valid = {"not_applicable", "declined", "applied"}
        for dossier in (dossier_a, dossier_b, dossier_c):
            action = recommend_clip_disposition(_make_obs(dossier))
            assert action.override_decision in valid, (
                f"override_decision={action.override_decision!r} for {dossier.clip_id}"
            )

    def test_reasoning_mentions_zh_phoneme_gap(self, dossier_a):
        """dataset_impact_reasoning must mention 'ZH' (a phoneme gap in the context)."""
        action = recommend_clip_disposition(_make_obs(dossier_a))
        assert "ZH" in action.dataset_impact_reasoning, (
            f"Expected 'ZH' in reasoning, got: {action.dataset_impact_reasoning!r}"
        )

    def test_fix_has_instructions_when_fix(self, dossier_b):
        """When disposition is 'fix', fix_instructions must be non-empty."""
        action = recommend_clip_disposition(_make_obs(dossier_b))
        if action.disposition == "fix":
            assert action.fix_instructions is not None
            assert len(action.fix_instructions) > 0

    def test_reject_has_rejection_reasons(self, dossier_b):
        """When disposition is 'reject', rejection_reasons should reference the issue."""
        action = recommend_clip_disposition(_make_obs(dossier_b))
        if action.disposition == "reject":
            # Either rejection_reasons is set, or the primary_rejection_reason is surfaced
            assert (
                action.rejection_reasons is not None
            ), "Expected rejection_reasons for a reject disposition"


# ---------------------------------------------------------------------------
# Test block 2 — Node 6 grader: DatasetHealthHandoff aggregation
# ---------------------------------------------------------------------------


class TestNode6DatasetHealthHandoff:
    """Tests for _build_dataset_health_handoff() (Node 6 aggregator)."""

    @pytest.fixture
    def all_actions_and_obs(self, dossier_a, dossier_b, dossier_c):
        """Run Node 5 for all three dossiers; return (actions, clip_obs_list)."""
        dossiers = [dossier_a, dossier_b, dossier_c]
        actions = [recommend_clip_disposition(_make_obs(d)) for d in dossiers]
        obs_list = [_minimal_clip_signal_obs(d.clip_id) for d in dossiers]
        return actions, obs_list

    @pytest.fixture
    def handoff(self, all_actions_and_obs):
        """Build the DatasetHealthHandoff from the three clip results."""
        actions, obs_list = all_actions_and_obs
        clip_scores = [0.6, 0.2, 0.5]   # arbitrary plausible per-clip scores
        subenv2_score = float(np.mean(clip_scores))
        return _build_dataset_health_handoff(actions, clip_scores, obs_list, subenv2_score)

    def test_subenv2_score_in_range(self, handoff):
        """subenv2_score must be in [0.0, 1.0]."""
        assert 0.0 <= handoff.subenv2_score <= 1.0

    def test_synthetic_weight_descriptor_type(self, handoff):
        """synthetic_weight_descriptor must be a SyntheticWeightDescriptor instance."""
        assert isinstance(handoff.synthetic_weight_descriptor, SyntheticWeightDescriptor)

    def test_clip_counts_sum_to_three(self, handoff, all_actions_and_obs):
        """accepted + rejected + fix_recommended must account for all non-defer clips.

        The aggregator does not include 'defer' dispositions in any count field,
        so the sum equals (total clips - deferred clips), not necessarily 3.
        """
        actions, _ = all_actions_and_obs
        expected_accept = sum(1 for a in actions if a.disposition == "accept")
        expected_reject = sum(1 for a in actions if a.disposition == "reject")
        expected_fix = sum(1 for a in actions if a.disposition == "fix")
        non_defer = expected_accept + expected_reject + expected_fix

        assert handoff.accepted_clip_count == expected_accept
        assert handoff.rejected_clip_count == expected_reject
        assert handoff.fix_recommended_count == expected_fix
        total = (
            handoff.accepted_clip_count
            + handoff.rejected_clip_count
            + handoff.fix_recommended_count
        )
        assert total == non_defer, (
            f"Expected counts to sum to {non_defer} (non-defer clips), got "
            f"accept={handoff.accepted_clip_count} "
            f"reject={handoff.rejected_clip_count} "
            f"fix={handoff.fix_recommended_count}"
        )

    def test_weight_contamination_in_range(self, handoff):
        """weight_contamination_estimate must be in [0.0, 1.0]."""
        assert 0.0 <= handoff.weight_contamination_estimate <= 1.0

    def test_identity_consistency_score_in_range(self, handoff):
        """identity_consistency_score must be in [0.0, 1.0]."""
        assert 0.0 <= handoff.identity_consistency_score <= 1.0

    def test_overall_dataset_quality_in_range(self, handoff):
        """overall_dataset_quality must be in [0.0, 1.0]."""
        assert 0.0 <= handoff.overall_dataset_quality <= 1.0

    def test_high_risk_clip_ids_are_strings(self, handoff):
        """high_risk_clip_ids must be a list of strings."""
        assert all(isinstance(cid, str) for cid in handoff.high_risk_clip_ids)
