"""
Pydantic v2 schemas for Sub-env 2: Dataset Clip Audit.

Covers all three nodes in the sub-environment:
  - Node 4 (Agent):  Clip Signal Extractor   → ClipEvidenceDossier
  - Node 5 (Agent):  Clip Disposition Recommender
  - Node 6 (Grader): Dataset Health Grader   → DatasetHealthHandoff

All field names, types, and Literal values match the spec in
``references/envs.md`` exactly.  No fields have been added or removed.

Important: ``ClipDispositionAction.override_decision`` is a 3-way
``Literal["not_applicable", "declined", "applied"]`` — NOT a bool.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Node 4 — Clip Signal Extractor
# ---------------------------------------------------------------------------


class ClipSignalObservation(BaseModel):
    """Pre-extracted CV signals fed to the Clip Signal Extractor agent (Node 4).

    All signals are computed from raw video files using standard libraries
    (OpenCV, MediaPipe, face-alignment, ArcFace).  No model inference is
    performed by the environment — the agent does diagnostic reasoning only.
    """

    clip_id: str

    # Identity consistency (ArcFace, MediaPipe)
    face_embedding_variance: float          # variance of ArcFace embeddings across frames
    landmark_stability_score: float         # MediaPipe landmark jitter, frame-to-frame
    identity_cosine_drift: float            # cosine distance: first frame vs last frame

    # Temporal signals
    frame_difference_mean: float            # mean absolute pixel difference
    optical_flow_magnitude: float           # face region vs background region ratio
    blink_count: int                        # proxy for naturalness

    # Audio-visual alignment (if audio present)
    lip_sync_confidence: float              # Wav2Lip-style alignment score
    phoneme_sequence: list[str]             # detected phoneme sequence
    phoneme_coverage_new: float             # new phonemes this clip adds to dataset

    # Quality signals
    blur_score: float                       # Laplacian variance, resolution-normalised to [0, 1]
    exposure_score: float                   # histogram mean + clipping
    occlusion_frames: int                   # frames where face is partially occluded

    # Dataset context
    clips_audited_so_far: int
    current_phoneme_coverage: dict          # phoneme → coverage count
    current_pose_distribution: dict         # regime → accepted count
    similar_clips_accepted: int             # same regime, similar embedding


class ClipEvidenceDossier(BaseModel):
    """Evidence summary produced by the Clip Signal Extractor agent (Node 4).

    Passed downstream to the Clip Disposition Recommender (Node 5) and
    aggregated by the Dataset Health Grader (Node 6).
    """

    clip_id: str
    identity_drift_severity: Literal["none", "minor", "moderate", "severe"]
    temporal_instability_flag: bool
    lip_sync_quality: Literal["good", "acceptable", "poor", "absent"]
    unique_phoneme_value: float
    dataset_redundancy_score: float         # how much does this duplicate existing clips
    estimated_training_impact: Literal["positive", "neutral", "negative"]
    primary_rejection_reason: Optional[str]
    evidence_summary: str


# ---------------------------------------------------------------------------
# Node 5 — Clip Disposition Recommender
# ---------------------------------------------------------------------------


class ClipDispositionObservation(BaseModel):
    """Inputs to the Clip Disposition Recommender agent (Node 5).

    Combines the per-clip evidence dossier from Node 4 with dataset-level
    context and the risk profile forwarded from Sub-env 1.
    """

    evidence_dossier: ClipEvidenceDossier

    # Dataset-level context
    minimum_clips_needed: int
    phoneme_gap_severity: dict              # phonemes below threshold
    pose_gap_severity: dict                 # regimes underrepresented
    budget_remaining: int                   # clips user still needs

    # Risk context from Sub-env 1
    reference_risk_profile: str
    estimated_drift_risk: float

    # Non-local consequence signals
    marginal_training_damage: float         # estimated damage of accepting this clip
    marginal_coverage_gain: float           # coverage gain of accepting this clip


class ClipDispositionAction(BaseModel):
    """Disposition recommendation produced by the Clip Disposition Recommender (Node 5).

    The ``fix`` disposition is what distinguishes this from a binary classifier:
    the agent must supply specific, actionable repair instructions.

    ``override_decision`` is a 3-way Literal — NOT a bool:
      - ``"not_applicable"`` — signals do not conflict with the disposition.
      - ``"declined"``       — override was considered but rejected.
      - ``"applied"``        — accepting despite negative signals (requires
                               ``override_justification``).
    """

    disposition: Literal["accept", "reject", "fix", "defer"]
    confidence: float

    # If reject
    rejection_reasons: Optional[list[str]]

    # If fix — specific and actionable
    fix_instructions: Optional[list[str]]   # e.g., "trim frames 0–5"
    estimated_fix_effort: Optional[Literal["trivial", "moderate", "high"]]

    # If defer — must document uncertainty
    defer_reason: Optional[str]             # required when disposition == "defer"

    # Strategic reasoning
    dataset_impact_reasoning: str

    # Override handling — 3-way literal (NOT bool)
    override_decision: Literal[
        "not_applicable",                   # signals don't conflict with disposition
        "declined",                         # considered overriding, chose not to
        "applied",                          # accepting despite negative signals
    ]
    override_justification: Optional[str]   # required when override_decision == "applied"


# ---------------------------------------------------------------------------
# Node 6 (Grader) — Dataset Health Grader
# ---------------------------------------------------------------------------


class SyntheticWeightDescriptor(BaseModel):
    """Structured prediction of what a model trained on this dataset would exhibit.

    Not actual weights — a typed signal profile for Sub-env 3 consumption.
    Produced by the Dataset Health Grader (Node 6) and forwarded to Node 7.
    """

    estimated_rank_utilization: float
    suspected_overfitting_score: float
    high_risk_phoneme_hints: list[str]          # from rejected/flagged clips
    identity_consistency_estimate: float
    expected_canonical_entropy_range: tuple[float, float]


class DatasetHealthHandoff(BaseModel):
    """Grader output from Node 6: Dataset Health Grader.

    Carries the Sub-env 2 composite score and a structured weight descriptor
    that feeds into Sub-env 3 as prior context for the Weight Signal Extractor.
    """

    accepted_clip_count: int
    rejected_clip_count: int
    fix_recommended_count: int

    identity_consistency_score: float           # across accepted clips
    phoneme_coverage_score: float               # fraction of phoneme space covered
    pose_diversity_score: float                 # regime distribution entropy
    overall_dataset_quality: float              # composite

    # For Sub-env 3
    suspected_anomalous_phonemes: list[str]     # phonemes from low-quality clips
    high_risk_clip_ids: list[str]               # clips accepted despite warnings
    weight_contamination_estimate: float        # estimated drift in trained weights
    synthetic_weight_descriptor: SyntheticWeightDescriptor

    subenv2_score: float
