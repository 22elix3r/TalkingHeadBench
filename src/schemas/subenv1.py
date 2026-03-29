"""
Pydantic v2 schemas for Sub-env 1: Reference Image + Prompt Audit.

Covers all three nodes in the sub-environment:
  - Node 1 (Agent):  Image Diagnostician
  - Node 2 (Agent):  Parameter Anomaly Detector
  - Node 3 (Grader): Reference Audit Grader  →  ReferenceAuditHandoff

All field names, types, and Literal values match the spec in
``references/envs.md`` exactly.  No fields have been added or removed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Node 1 — Image Diagnostician
# ---------------------------------------------------------------------------


class ImageDiagnosticsObservation(BaseModel):
    """Pre-extracted signals fed to the Image Diagnostician agent (Node 1).

    Signals are derived from the reference image via MediaPipe, OpenCV, and
    face-alignment, plus CLIP tokenizer analysis of the prompt text.  No model
    inference is performed by the environment — the agent receives computed
    scalars and flags only.
    """

    # Extracted from reference image (MediaPipe, OpenCV, face-alignment)
    face_occupancy_ratio: float
    estimated_yaw_degrees: float
    estimated_pitch_degrees: float
    background_complexity_score: float
    lighting_uniformity_score: float        # low = uneven lighting
    skin_tone_bucket: int                   # Fitzpatrick 1–6
    occlusion_detected: bool                # glasses, hair, hands
    image_resolution: tuple[int, int]
    estimated_sharpness: float              # Laplacian variance, resolution-normalised to [0, 1]

    # Extracted from prompt text (CLIP tokeniser analysis)
    prompt_token_count: int
    prompt_semantic_density: float          # unique concept count
    conflicting_descriptors: list[str]      # terms that contradict each other
    identity_anchoring_strength: float      # how strongly prompt binds to identity


class ImageDiagnosticsAction(BaseModel):
    """Structured diagnosis produced by the Image Diagnostician agent (Node 1)."""

    regime_classification: Literal[
        "frontal_simple",
        "non_frontal",
        "complex_background",
        "occluded",
        "low_quality",
    ]
    identified_risk_factors: list[str]          # specific issues detected
    prompt_issues: list[str]                    # conflicting or weak terms
    recommended_prompt_modifications: list[str]
    image_usability_score: float                # 0.0–1.0
    reasoning: str


# ---------------------------------------------------------------------------
# Node 2 — Parameter Anomaly Detector
# ---------------------------------------------------------------------------


class ParameterAnomaly(BaseModel):
    """A single flagged parameter anomaly identified by the Parameter Anomaly Detector."""

    parameter: str                              # "cfg", "denoise_alt", "eta", etc.
    issue: str
    severity: Literal["minor", "moderate", "severe"]
    linked_failure_mode: str


class DirectionalFix(BaseModel):
    """A directional (not prescriptive) fix recommendation for a flagged parameter."""

    target: str
    direction: Literal["increase", "decrease", "enable", "disable", "reconsider"]
    rationale: str
    priority: Literal["critical", "recommended", "optional"]


class ParamAnomalyObservation(BaseModel):
    """Inputs to the Parameter Anomaly Detector agent (Node 2).

    Combines the user's proposed generation configuration with diagnostic
    signals forwarded from Node 1 and the reference image itself.
    """

    # User's proposed configuration
    proposed_config: dict                       # {"cfg": 8.5, "denoise_alt": 0.25, ...}

    # From Node 1
    regime: str
    identified_risk_factors: list[str]
    image_usability_score: float

    # Reference image signals
    face_occupancy_ratio: float
    estimated_yaw_degrees: float
    background_complexity_score: float
    lighting_uniformity_score: float
    occlusion_detected: bool

    # Prompt signals
    prompt_identity_anchoring: float
    prompt_token_count: int
    conflicting_descriptors: list[str]


class ParamAnomalyAction(BaseModel):
    """Structured output of the Parameter Anomaly Detector agent (Node 2)."""

    config_risk_level: Literal["safe", "marginal", "risky", "dangerous"]
    anomalies: list[ParameterAnomaly]
    predicted_failure_modes: list[
        Literal[
            "identity_collapse",
            "reference_token_dropout",
            "temporal_jitter",
            "background_bleed",
            "lip_sync_desync",
            "pose_instability",
            "overexposure_artifacts",
        ]
    ]
    directional_fixes: list[DirectionalFix]
    summary: str


# ---------------------------------------------------------------------------
# Node 3 (Grader) — Reference Audit Handoff
# ---------------------------------------------------------------------------


class ReferenceAuditHandoff(BaseModel):
    """Grader output from Node 3: Reference Audit Grader.

    Carries the Sub-env 1 composite score and a risk profile that acts as the
    hard coupling mechanism into Sub-env 2.  A degraded risk profile causes
    Sub-env 2 to receive harder clips, mirroring real-world consequences of a
    poor reference image audit.
    """

    image_usability_score: float
    regime: str
    identified_risk_factors: list[str]
    config_quality_score: float
    risk_profile: Literal["low", "medium", "high"]

    # Feed into Sub-env 2
    estimated_drift_risk: float
    prompt_strength: float
    recommended_config: dict

    subenv1_score: float
