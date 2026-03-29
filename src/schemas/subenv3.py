"""
Pydantic v2 schemas for Sub-env 3: Trained LoRA Weight Behavioral Audit.

Covers all three nodes in the sub-environment:
  - Node 7 (Agent):  Weight Signal Extractor   → WeightEvidenceDossier
  - Node 8 (Agent):  Phoneme Risk Assessor
  - Node 9 (Grader): Behavioral Audit Grader   → BehavioralAuditHandoff

All field names, types, and Literal values match the spec in
``references/envs.md`` exactly.  No fields have been added or removed.

Cross-module imports:
  - ``SyntheticWeightDescriptor`` is imported from ``subenv2`` (produced by
    Node 6 and forwarded into Node 7 as prior context).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from src.schemas.subenv2 import SyntheticWeightDescriptor


# ---------------------------------------------------------------------------
# Node 7 — Weight Signal Extractor: observation inputs
# ---------------------------------------------------------------------------


class TokenAnomalyFlag(BaseModel):
    """A single anomalous token position identified in the canonical Vt components."""

    token_position: int
    mapped_phoneme: Optional[str]
    anomaly_type: Literal[
        "excessive_influence",
        "unstable_encoding",
        "cross_token_bleed",
        "identity_entanglement",
    ]
    severity: float                         # 0.0–1.0
    evidence: str


class LayerAnomalyFlag(BaseModel):
    """A single anomalous layer identified from canonical weight statistics."""

    layer_name: str
    anomaly_type: Literal[
        "rank_collapse",
        "norm_explosion",
        "sparsity_anomaly",
        "correlation_anomaly",
    ]
    severity: float
    evidence: str


class WeightSignalObservation(BaseModel):
    """Pre-extracted canonical weight statistics fed to the Weight Signal Extractor (Node 7).

    All signals are derived from the W2T-style QR → SVD canonical decomposition
    (see ``src/utils/canonical.py``).  The agent receives statistics on canonical
    components, not raw A/B matrices.

    ``token_position_to_phoneme`` is loaded from the audio tokenizer config file
    shipped alongside ``.safetensors`` — it is NOT derivable from the weights alone.
    """

    weight_file_id: str
    lora_rank: int
    target_modules: list[str]
    total_parameters: int

    # Layer-wise statistics (computed on canonical components)
    layer_norms: dict[str, float]               # Frobenius norm of canonical update
    layer_sparsity: dict[str, float]            # fraction near-zero in canonical S
    layer_rank_utilization: dict[str, float]    # effective rank / nominal rank (from canonical SVD)

    # Canonical Vt-component analysis (post QR→SVD, per W2T)
    canonical_entropy_per_layer: dict[str, float]       # entropy of canonical Vt rows
    high_entropy_token_positions: list[int]             # token indices with anomalous values

    # Token-to-phoneme mapping
    # IMPORTANT: loaded from audio tokenizer config file shipped alongside
    # .safetensors — NOT derivable from weights alone.
    token_position_to_phoneme: Optional[dict[int, str]]

    # Canonical U-component analysis
    canonical_output_norm_variance: float       # variance of U column norms
    canonical_dominant_directions: int          # singular values capturing 90% of energy

    # Cross-layer patterns
    layer_correlation_matrix: list[list[float]]
    attention_head_specialization: dict[str, float]

    # Training quality signals
    weight_magnitude_histogram: list[float]     # binned canonical S distribution
    gradient_noise_estimate: float
    overfitting_signature: float                # high = likely overfit to small dataset

    # Context from Sub-env 2 (if pipeline ran end-to-end)
    dataset_health_summary: Optional[SyntheticWeightDescriptor]
    suspected_anomalous_phonemes: Optional[list[str]]


# ---------------------------------------------------------------------------
# Node 7 — Weight Signal Extractor: output
# ---------------------------------------------------------------------------


class WeightEvidenceDossier(BaseModel):
    """Evidence dossier produced by the Weight Signal Extractor agent (Node 7).

    Passed downstream to the Phoneme Risk Assessor (Node 8) and evaluated by
    the Behavioral Audit Grader (Node 9).
    """

    weight_file_id: str
    training_quality: Literal["healthy", "unstable", "overfit", "underfit"]
    rank_utilization_assessment: str        # "efficient", "wasteful", "collapsed"

    high_entropy_token_flags: list[TokenAnomalyFlag]
    layer_anomaly_flags: list[LayerAnomalyFlag]

    overall_behavioral_risk: Literal["low", "medium", "high", "critical"]
    evidence_summary: str


# ---------------------------------------------------------------------------
# Node 8 — Phoneme Risk Assessor: observation inputs
# ---------------------------------------------------------------------------


class PhonemeRiskObservation(BaseModel):
    """Inputs to the Phoneme Risk Assessor agent (Node 8).

    Combines the weight evidence dossier from Node 7 with the audio tokenizer
    phoneme vocabulary and aggregated phoneme-level canonical signals.
    """

    weight_evidence: WeightEvidenceDossier
    high_entropy_token_flags: list[TokenAnomalyFlag]

    # From audio tokenizer config file (NOT from weights)
    phoneme_vocabulary: list[str]
    phoneme_to_token_indices: dict[str, list[int]]

    # Aggregated phoneme-level signals (from canonical components)
    phoneme_entropy_scores: dict[str, float]
    phoneme_influence_scores: dict[str, float]
    phoneme_cooccurrence_anomalies: list[tuple[str, str, float]]

    behavior_vocabulary: list[str]          # "smile", "blink", "head_turn", etc.

    training_data_phoneme_distribution: Optional[dict[str, int]]
    suspected_anomalous_phonemes_from_subenv2: Optional[list[str]]


# ---------------------------------------------------------------------------
# Node 8 — Phoneme Risk Assessor: action sub-types and output
# ---------------------------------------------------------------------------


class PhonemeRiskEntry(BaseModel):
    """A single phoneme's risk entry in the ranked risk list."""

    phoneme: str
    risk_score: float                       # 0.0–1.0
    risk_type: Literal[
        "identity_trigger",
        "expression_trigger",
        "motion_trigger",
        "artifact_trigger",
        "unknown_anomaly",
    ]
    confidence: float
    evidence: str


class BehaviorTriggerPrediction(BaseModel):
    """A predicted phoneme → behavior association learned by the LoRA."""

    trigger_phoneme: str
    triggered_behavior: str                 # from behavior_vocabulary
    association_strength: float
    is_intended: bool
    concern_level: Literal["none", "low", "medium", "high"]


class PhonemeCluster(BaseModel):
    """A cluster of phonemes sharing a common risk pattern."""

    phonemes: list[str]
    cluster_risk_type: str
    combined_risk_score: float
    interaction_description: str


class MitigationRecommendation(BaseModel):
    """A directional mitigation recommendation for a phoneme-level behavioral risk."""

    target: str
    action: Literal[
        "retrain_with_more_data",
        "remove_from_dataset",
        "add_counter_examples",
        "reduce_lora_rank",
        "apply_weight_regularization",
        "flag_for_manual_review",
    ]
    rationale: str
    priority: Literal["critical", "recommended", "optional"]


class PhonemeRiskAction(BaseModel):
    """Structured behavioral risk profile produced by the Phoneme Risk Assessor (Node 8)."""

    phoneme_risk_ranking: list[PhonemeRiskEntry]
    predicted_behavior_triggers: list[BehaviorTriggerPrediction]
    risky_phoneme_clusters: list[PhonemeCluster]
    model_behavioral_safety: Literal[
        "safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"
    ]
    mitigation_recommendations: list[MitigationRecommendation]
    summary: str


# ---------------------------------------------------------------------------
# Node 9 (Grader) — Behavioral Audit Handoff
# ---------------------------------------------------------------------------


class BehavioralAuditHandoff(BaseModel):
    """Final grader output from Node 9: Behavioral Audit Grader.

    Carries per-dimension scores and the Sub-env 3 composite score.
    """

    weight_file_id: str

    phoneme_risk_ranking: list[PhonemeRiskEntry]
    predicted_behavior_triggers: list[BehaviorTriggerPrediction]
    risky_phoneme_clusters: list[PhonemeCluster]
    model_behavioral_safety: str
    mitigation_recommendations: list[MitigationRecommendation]

    ranking_quality_score: float
    trigger_prediction_score: float
    cluster_identification_score: float
    safety_calibration_score: float
    mitigation_quality_score: float

    subenv3_score: float
