# TalkingHeadBench — Full Environment Specification

> **Table of Contents**
> - [Sub-env 1: Reference Image + Prompt Audit](#sub-env-1)
>   - [Node 1: Image Diagnostician](#node-1-image-diagnostician)
>   - [Node 2: Parameter Anomaly Detector](#node-2-parameter-anomaly-detector)
>   - [Node 3 (G): Reference Audit Grader](#node-3-g-reference-audit-grader)
> - [Sub-env 2: Dataset Clip Audit](#sub-env-2)
>   - [Node 4: Clip Signal Extractor](#node-4-clip-signal-extractor)
>   - [Node 5: Clip Disposition Recommender](#node-5-clip-disposition-recommender)
>   - [Node 6 (G): Dataset Health Grader](#node-6-g-dataset-health-grader)
> - [Sub-env 3: Trained LoRA Weight Audit](#sub-env-3)
>   - [Node 7: Weight Signal Extractor](#node-7-weight-signal-extractor)
>   - [Node 8: Phoneme Risk Assessor](#node-8-phoneme-risk-assessor)
>   - [Node 9 (G): Behavioral Audit Grader](#node-9-g-behavioral-audit-grader)
> - [Ground Truth Schemas](#ground-truth-schemas)

---

## Sub-env 1

**User artifact:** Reference image(s) + prompt text  
**Agent task:** Diagnose quality issues and recommend optimal generation configuration

---

### Node 1: Image Diagnostician

The agent receives pre-extracted signals from the reference image and produces a
structured diagnosis. It identifies the operational regime, flags specific risk
factors, and evaluates prompt quality — all from pre-computed signals, no
generation required.

**Observation space:**

```python
class ImageDiagnosticsObservation(BaseModel):
    # Extracted from reference image (MediaPipe, OpenCV, face-alignment)
    face_occupancy_ratio: float
    estimated_yaw_degrees: float
    estimated_pitch_degrees: float
    background_complexity_score: float
    lighting_uniformity_score: float       # low = uneven lighting
    skin_tone_bucket: int                  # Fitzpatrick 1–6
    occlusion_detected: bool               # glasses, hair, hands
    image_resolution: tuple[int, int]
    estimated_sharpness: float             # Laplacian variance, resolution-normalized to [0,1]

    # Extracted from prompt text (CLIP tokenizer analysis)
    prompt_token_count: int
    prompt_semantic_density: float         # unique concept count
    conflicting_descriptors: list[str]     # terms that contradict each other
    identity_anchoring_strength: float     # how strongly prompt binds to identity
```

**Action:**

```python
class ImageDiagnosticsAction(BaseModel):
    regime_classification: Literal[
        "frontal_simple",
        "non_frontal",
        "complex_background",
        "occluded",
        "low_quality"
    ]
    identified_risk_factors: list[str]          # specific issues detected
    prompt_issues: list[str]                    # conflicting or weak terms
    recommended_prompt_modifications: list[str]
    image_usability_score: float                # 0.0–1.0
    reasoning: str
```

**Example output:**

```python
ImageDiagnosticsAction(
    regime_classification="non_frontal",
    identified_risk_factors=[
        "yaw exceeds 25° — lateral pose will reduce reference token coverage in mid-blocks",
        "lighting uniformity low (0.31) — shadow regions may cause inconsistent skin rendering",
        "face occupancy below 0.4 — background will compete for attention budget"
    ],
    prompt_issues=[
        "'dramatic lighting' conflicts with 'natural look' — contradictory descriptors",
        "no explicit identity descriptor — prompt relies entirely on reference image for identity"
    ],
    recommended_prompt_modifications=[
        "remove 'dramatic lighting' or 'natural look' — keep one lighting directive",
        "add explicit identity anchoring term to strengthen binding",
        "add 'consistent lighting' to reduce shadow-induced drift"
    ],
    image_usability_score=0.58,
    reasoning="Non-frontal pose with weak lighting creates compounding risk: lateral reference "
              "tokens will have reduced coverage, and shadow regions will introduce inconsistent "
              "skin tone encoding. The prompt lacks identity anchoring, meaning the model must "
              "rely entirely on the reference image — which is itself suboptimal due to angle."
)
```

**Grader logic:**

Ground truth includes `valid_prompt_modifications: list[str]` — a curated set
of acceptable modifications per test case — enabling deterministic evaluation
without an LLM judge.

```python
def grade_image_diagnostics(agent_action, ground_truth):
    scores = {}

    # 1. Regime classification (partial credit for borderline)
    if agent_action.regime_classification == ground_truth.regime_classification:
        scores["regime_accuracy"] = 1.0
    elif agent_action.regime_classification in ground_truth.acceptable_regimes:
        scores["regime_accuracy"] = 0.7
    else:
        scores["regime_accuracy"] = 0.0

    # 2. Risk factor recall
    predicted_risks = set(agent_action.identified_risk_factors)
    true_risks      = set(ground_truth.identified_risk_factors)
    scores["risk_factor_recall"] = (
        len(predicted_risks & true_risks) / len(true_risks) if true_risks else 1.0
    )

    # 3. Prompt modification validity (deterministic set intersection)
    valid_mods = set(ground_truth.valid_prompt_modifications)
    agent_mods = set(agent_action.recommended_prompt_modifications)
    if agent_mods:
        scores["prompt_modification_validity"] = len(agent_mods & valid_mods) / len(agent_mods)
    else:
        scores["prompt_modification_validity"] = 0.0 if valid_mods else 1.0

    return (
        0.35 * scores["regime_accuracy"]
      + 0.35 * scores["risk_factor_recall"]
      + 0.30 * scores["prompt_modification_validity"]
    )
```

**Research grounding:** Risk factors trace to TARA (attention bleed from low
face occupancy), EditYourself (reference token coverage at lateral angles), and
MoFE (angle-dependent drift). Prompt issues grounded in CLIP alignment
literature.

---

### Node 2: Parameter Anomaly Detector

The agent is a diagnostician of parameter configurations, not a prescriber of
exact values. The user arrives with their intended setup and the agent identifies
what will go wrong, providing directional guidance only — like a senior engineer
reviewing a junior's config file.

**Observation space:**

```python
class ParamAnomalyObservation(BaseModel):
    # User's proposed configuration
    proposed_config: dict  # {"cfg": 8.5, "denoise_alt": 0.25, "eta": 0.15, ...}

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
```

**Action:**

```python
class ParamAnomalyAction(BaseModel):
    config_risk_level: Literal["safe", "marginal", "risky", "dangerous"]
    anomalies: list[ParameterAnomaly]
    predicted_failure_modes: list[Literal[
        "identity_collapse",
        "reference_token_dropout",
        "temporal_jitter",
        "background_bleed",
        "lip_sync_desync",
        "pose_instability",
        "overexposure_artifacts"
    ]]
    directional_fixes: list[DirectionalFix]
    summary: str


class ParameterAnomaly(BaseModel):
    parameter: str                      # "cfg", "denoise_alt", "eta", etc.
    issue: str
    severity: Literal["minor", "moderate", "severe"]
    linked_failure_mode: str


class DirectionalFix(BaseModel):
    target: str
    direction: Literal["increase", "decrease", "enable", "disable", "reconsider"]
    rationale: str
    priority: Literal["critical", "recommended", "optional"]
```

**Example output:**

```python
ParamAnomalyAction(
    config_risk_level="risky",
    anomalies=[
        ParameterAnomaly(
            parameter="denoise_alt",
            issue="too low for non-frontal regime — reference token will lose coverage in lateral blocks",
            severity="severe",
            linked_failure_mode="reference_token_dropout"
        ),
        ParameterAnomaly(
            parameter="eta",
            issue="high stochasticity combined with low reference anchoring — identity will drift",
            severity="moderate",
            linked_failure_mode="identity_collapse"
        ),
        ParameterAnomaly(
            parameter="cfg",
            issue="elevated CFG with complex background — attention will bleed into non-face regions",
            severity="moderate",
            linked_failure_mode="background_bleed"
        )
    ],
    predicted_failure_modes=["reference_token_dropout", "identity_collapse", "background_bleed"],
    directional_fixes=[
        DirectionalFix(
            target="reference_token_strength", direction="increase",
            rationale="compensates for lateral pose — keeps identity anchored in side-facing frames",
            priority="critical"
        ),
        DirectionalFix(
            target="stochasticity (eta)", direction="decrease",
            rationale="reduces frame-to-frame identity variance — improves temporal consistency",
            priority="critical"
        ),
        DirectionalFix(
            target="guidance_scale", direction="decrease",
            rationale="prevents attention competition with complex background elements",
            priority="recommended"
        ),
    ],
    summary="High risk of identity drift and reference dropout. Primary issues: denoise_alt too "
            "low for non-frontal pose, eta too high for current reference strength."
)
```

**Grader logic:**

```python
def grade_anomaly_detection(agent_action, ground_truth):
    scores = {}

    # 1. Anomaly detection — F1 over flagged parameters
    predicted_params = {a.parameter for a in agent_action.anomalies}
    true_params      = {a.parameter for a in ground_truth.anomalies}
    if not predicted_params and not true_params:
        scores["anomaly_detection"] = 1.0
    elif not true_params:
        scores["anomaly_detection"] = 0.0  # agent hallucinated anomalies
    else:
        recall    = len(predicted_params & true_params) / len(true_params)
        precision = len(predicted_params & true_params) / len(predicted_params) if predicted_params else 0
        scores["anomaly_detection"] = 0.5 * recall + 0.5 * precision

    # 2. Failure mode prediction — set F1
    scores["failure_mode_prediction"] = set_f1(
        set(agent_action.predicted_failure_modes),
        set(ground_truth.predicted_failure_modes)
    )

    # 3. Directional fix quality
    scores["fix_quality"] = evaluate_directional_fixes(
        agent_action.directional_fixes,
        ground_truth.valid_fix_directions
    )

    # 4. Risk level calibration — ordinal distance (NOT binary)
    risk_levels = ["safe", "marginal", "risky", "dangerous"]
    agent_idx = risk_levels.index(agent_action.config_risk_level)
    true_idx  = risk_levels.index(ground_truth.config_risk_level)
    scores["risk_calibration"] = 1.0 - abs(agent_idx - true_idx) / (len(risk_levels) - 1)

    return (
        0.30 * scores["anomaly_detection"]
      + 0.25 * scores["failure_mode_prediction"]
      + 0.30 * scores["fix_quality"]
      + 0.15 * scores["risk_calibration"]
    )
```

**Why frontier models will struggle:** The agent must reason about parameter
*interactions*, not just individual ranges. Priority assignment requires
understanding downstream consequences. A parameter that is fine in isolation can
be deadly in combination with another.

---

### Node 3 (G): Reference Audit Grader

Produces a structured audit report and passes a risk profile forward into
Sub-env 2. The `risk_profile` is the hard coupling mechanism — a bad reference
image audit causes Sub-env 2 to receive harder clips, mirroring real-world
consequences.

**Output:**

```python
class ReferenceAuditHandoff(BaseModel):
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
```

---

## Sub-env 2

**User artifact:** A folder of raw video clips intended for LoRA training  
**Agent task:** Inspect each clip, identify quality issues, recommend accept / reject / fix / defer

---

### Node 4: Clip Signal Extractor

The agent receives pre-extracted signals computed directly from video files
using standard CV libraries (OpenCV, MediaPipe, face-alignment, ArcFace). No
model inference required. The agent does diagnostic reasoning, not perception.

**Observation space:**

```python
class ClipSignalObservation(BaseModel):
    clip_id: str

    # Identity consistency (ArcFace, MediaPipe)
    face_embedding_variance: float         # variance of ArcFace embeddings across frames
    landmark_stability_score: float        # MediaPipe landmark jitter, frame-to-frame
    identity_cosine_drift: float           # cosine distance: first frame vs last frame

    # Temporal signals
    frame_difference_mean: float           # mean absolute pixel difference
    optical_flow_magnitude: float          # face region vs background region ratio
    blink_count: int                       # proxy for naturalness

    # Audio-visual alignment (if audio present)
    lip_sync_confidence: float             # Wav2Lip-style alignment score
    phoneme_sequence: list[str]            # detected phoneme sequence
    phoneme_coverage_new: float            # new phonemes this clip adds to dataset

    # Quality signals
    blur_score: float                      # Laplacian variance, resolution-normalized to [0,1]
    exposure_score: float                  # histogram mean + clipping
    occlusion_frames: int                  # frames where face is partially occluded

    # Dataset context
    clips_audited_so_far: int
    current_phoneme_coverage: dict         # phoneme → coverage count
    current_pose_distribution: dict        # regime → accepted count
    similar_clips_accepted: int            # same regime, similar embedding
```

**Output — evidence dossier:**

```python
class ClipEvidenceDossier(BaseModel):
    clip_id: str
    identity_drift_severity: Literal["none", "minor", "moderate", "severe"]
    temporal_instability_flag: bool
    lip_sync_quality: Literal["good", "acceptable", "poor", "absent"]
    unique_phoneme_value: float
    dataset_redundancy_score: float        # how much does this duplicate existing clips
    estimated_training_impact: Literal["positive", "neutral", "negative"]
    primary_rejection_reason: Optional[str]
    evidence_summary: str
```

**Example output:**

```python
ClipEvidenceDossier(
    clip_id="clip_017",
    identity_drift_severity="moderate",
    temporal_instability_flag=True,
    lip_sync_quality="acceptable",
    unique_phoneme_value=0.62,
    dataset_redundancy_score=0.15,
    estimated_training_impact="negative",
    primary_rejection_reason="identity cosine drift 0.23 exceeds threshold; "
                             "landmark jitter concentrated in jaw region",
    evidence_summary="Moderate identity drift with temporal instability in jaw landmarks. "
                     "Lip sync acceptable but identity inconsistency would contaminate training. "
                     "However, clip contains ZH, TH, NG — currently underrepresented. "
                     "Recommend fix (trim unstable frames) rather than outright rejection."
)
```

---

### Node 5: Clip Disposition Recommender

The agent recommends a disposition for each clip with specific, actionable
reasons. The `fix` disposition distinguishes this from a binary classifier —
the agent provides repair instructions, which is what a real expert would do.

**Observation space:**

```python
class ClipDispositionObservation(BaseModel):
    evidence_dossier: ClipEvidenceDossier

    # Dataset-level context
    minimum_clips_needed: int
    phoneme_gap_severity: dict             # phonemes below threshold
    pose_gap_severity: dict               # regimes underrepresented
    budget_remaining: int                 # clips user still needs

    # Risk context from Sub-env 1
    reference_risk_profile: str
    estimated_drift_risk: float

    # Non-local consequence signals
    marginal_training_damage: float        # estimated damage of accepting this clip
    marginal_coverage_gain: float          # coverage gain of accepting this clip
```

**Action:**

```python
class ClipDispositionAction(BaseModel):
    disposition: Literal["accept", "reject", "fix", "defer"]
    confidence: float

    # If reject
    rejection_reasons: Optional[list[str]]

    # If fix — specific and actionable
    fix_instructions: Optional[list[str]]  # e.g., "trim frames 0–5"
    estimated_fix_effort: Optional[Literal["trivial", "moderate", "high"]]

    # If defer — must document uncertainty
    defer_reason: Optional[str]            # required when disposition == "defer"

    # Strategic reasoning
    dataset_impact_reasoning: str

    # Override handling — 3-way literal (NOT bool)
    override_decision: Literal[
        "not_applicable",                  # signals don't conflict with disposition
        "declined",                        # considered overriding, chose not to
        "applied"                          # accepting despite negative signals
    ]
    override_justification: Optional[str]  # required when override_decision == "applied"
```

**Example output:**

```python
ClipDispositionAction(
    disposition="fix",
    confidence=0.74,
    rejection_reasons=None,
    fix_instructions=[
        "trim frames 0–8 where face is partially occluded by hand",
        "trim final 12 frames where identity drift exceeds threshold",
        "remaining frames 9–41 contain clean ZH, TH phoneme coverage"
    ],
    estimated_fix_effort="trivial",
    defer_reason=None,
    dataset_impact_reasoning="Dataset has 0 clips covering ZH and only 1 covering TH. "
                             "This clip, once trimmed, provides clean coverage for both. "
                             "Rejecting outright would leave a critical phoneme gap.",
    override_decision="applied",
    override_justification="Drift is 'moderate' which would normally trigger rejection, "
                           "but drift is localized to trimmable frames. After trimming, "
                           "remaining segment has cosine drift 0.07 (within threshold)."
)
```

**Grader logic:**

```python
def grade_clip_disposition(agent_action, ground_truth):
    score = 0.0

    # 1. Base disposition (0.40 max)
    if agent_action.disposition == ground_truth.disposition:
        calibrated = abs(agent_action.confidence - ground_truth.confidence) < 0.15
        score += 0.40 if calibrated else 0.28
    elif agent_action.disposition == "fix" and ground_truth.disposition == "reject":
        score += 0.20    # partial: fix is better than blind accept
    elif agent_action.disposition == "defer":
        if ground_truth.disposition_ambiguity >= 0.5:
            score += 0.15 if agent_action.defer_reason else 0.10
        else:
            score = max(score - 0.05, 0.0)   # defer on unambiguous = avoidance
    elif agent_action.disposition == "accept" and ground_truth.disposition == "reject":
        score += 0.00    # worst case

    # 2. Fix instruction quality (0.20 max)
    if agent_action.disposition == "fix" and agent_action.fix_instructions:
        valid_steps  = sum(1 for s in agent_action.fix_instructions
                          if s in ground_truth.valid_fix_steps)
        fix_precision = valid_steps / len(agent_action.fix_instructions)
        score += 0.20 if fix_precision >= 0.8 else (0.10 if fix_precision >= 0.5 else 0.00)

    # 3. Dataset impact reasoning (0.20 max)
    kw_elements   = ground_truth.expected_reasoning_elements
    agent_text    = agent_action.dataset_impact_reasoning.lower()
    matched       = sum(1 for kw in kw_elements if kw in agent_text)
    score += 0.20 if matched >= len(kw_elements) * 0.8 else (0.10 if matched >= 1 else 0.00)

    # 4. Override misuse penalty
    if agent_action.override_decision == "applied":
        if not agent_action.override_justification:
            score -= 0.10
        elif agent_action.override_justification not in ground_truth.valid_override_justifications:
            score -= 0.05
            # Note: valid_override_justifications matching should use semantic
            # similarity in production — exact string match is fragile for free text.

    return max(score, 0.0)
```

---

### Node 6 (G): Dataset Health Grader

Produces a dataset health report and passes a structured weight descriptor to
Sub-env 3. Does not simulate actual training.

**Supporting types:**

```python
class SyntheticWeightDescriptor(BaseModel):
    """Structured prediction of what a model trained on this dataset would exhibit.
    Not actual weights — a typed signal profile for Sub-env 3 consumption."""
    estimated_rank_utilization: float
    suspected_overfitting_score: float
    high_risk_phoneme_hints: list[str]         # from rejected/flagged clips
    identity_consistency_estimate: float
    expected_canonical_entropy_range: tuple[float, float]
```

**Output:**

```python
class DatasetHealthHandoff(BaseModel):
    accepted_clip_count: int
    rejected_clip_count: int
    fix_recommended_count: int

    identity_consistency_score: float          # across accepted clips
    phoneme_coverage_score: float              # fraction of phoneme space covered
    pose_diversity_score: float                # regime distribution entropy
    overall_dataset_quality: float             # composite

    # For Sub-env 3
    suspected_anomalous_phonemes: list[str]    # phonemes from low-quality clips
    high_risk_clip_ids: list[str]              # clips accepted despite warnings
    weight_contamination_estimate: float        # estimated drift in trained weights
    synthetic_weight_descriptor: SyntheticWeightDescriptor

    subenv2_score: float
```

---

## Sub-env 3

**User artifact:** Trained LoRA weight file (`.safetensors`)  
**Agent task:** Inspect weight structure, identify behavioral anomalies, produce a risk-ranked behavioral profile

This is the **hardest task**. The user has already trained a LoRA and wants to
know: *"What hidden behaviors did this model learn? What inputs will trigger
unexpected outputs?"*

---

### Node 7: Weight Signal Extractor

All LoRA factor signals pass through a W2T-style canonical decomposition (QR →
SVD) before any statistics are computed. This resolves factorization ambiguity
so equivalent (A, B) pairs produce identical canonical representations — the
core insight of W2T. The agent receives statistics on canonical components, not
raw A/B matrices.

**Canonical preprocessing (run by the environment, not the agent):**

```python
def canonicalize_lora_factors(A: Tensor, B: Tensor) -> CanonicalComponents:
    """W2T-style canonical decomposition: QR → SVD.
    Resolves column-space factorization ambiguity so equivalent (A, B) pairs
    produce identical canonical representations."""
    Q_a, R_a = torch.linalg.qr(A)            # Step 1: resolve column-space ambiguity
    effective_update = B @ R_a                # Step 2: form the effective LoRA update
    U, S, Vt = torch.linalg.svd(effective_update, full_matrices=False)  # Step 3: SVD
    return CanonicalComponents(U=U, S=S, Vt=Vt, Q=Q_a)
```

**Observation space:**

```python
class WeightSignalObservation(BaseModel):
    weight_file_id: str
    lora_rank: int
    target_modules: list[str]
    total_parameters: int

    # Layer-wise statistics (computed on canonical components)
    layer_norms: dict[str, float]              # Frobenius norm of canonical update
    layer_sparsity: dict[str, float]           # fraction near-zero in canonical S
    layer_rank_utilization: dict[str, float]   # effective rank / nominal rank (from canonical SVD)

    # Canonical Vt-component analysis (post QR→SVD, per W2T)
    canonical_entropy_per_layer: dict[str, float]      # entropy of canonical Vt rows
    high_entropy_token_positions: list[int]             # token indices with anomalous values

    # Token-to-phoneme mapping
    # IMPORTANT: loaded from audio tokenizer config file shipped alongside
    # .safetensors — NOT derivable from weights alone.
    token_position_to_phoneme: Optional[dict[int, str]]

    # Canonical U-component analysis
    canonical_output_norm_variance: float      # variance of U column norms
    canonical_dominant_directions: int         # singular values capturing 90% of energy

    # Cross-layer patterns
    layer_correlation_matrix: list[list[float]]
    attention_head_specialization: dict[str, float]

    # Training quality signals
    weight_magnitude_histogram: list[float]    # binned canonical S distribution
    gradient_noise_estimate: float
    overfitting_signature: float               # high = likely overfit to small dataset

    # Context from Sub-env 2 (if pipeline ran end-to-end)
    dataset_health_summary: Optional[SyntheticWeightDescriptor]
    suspected_anomalous_phonemes: Optional[list[str]]
```

**Computation notes:**
- Layer norms: `torch.linalg.norm(canonical_effective_update)`
- Entropy: `scipy.stats.entropy(softmax(Vt_rows, axis=1))`
- Rank utilization: from canonical SVD directly — no additional computation needed

**Output — weight evidence dossier:**

```python
class WeightEvidenceDossier(BaseModel):
    weight_file_id: str
    training_quality: Literal["healthy", "unstable", "overfit", "underfit"]
    rank_utilization_assessment: str       # "efficient", "wasteful", "collapsed"

    high_entropy_token_flags: list[TokenAnomalyFlag]
    layer_anomaly_flags: list[LayerAnomalyFlag]

    overall_behavioral_risk: Literal["low", "medium", "high", "critical"]
    evidence_summary: str


class TokenAnomalyFlag(BaseModel):
    token_position: int
    mapped_phoneme: Optional[str]
    anomaly_type: Literal[
        "excessive_influence",
        "unstable_encoding",
        "cross_token_bleed",
        "identity_entanglement"
    ]
    severity: float                        # 0.0–1.0
    evidence: str


class LayerAnomalyFlag(BaseModel):
    layer_name: str
    anomaly_type: Literal[
        "rank_collapse",
        "norm_explosion",
        "sparsity_anomaly",
        "correlation_anomaly"
    ]
    severity: float
    evidence: str
```

**Example output:**

```python
WeightEvidenceDossier(
    weight_file_id="lora_user_042.safetensors",
    training_quality="overfit",
    rank_utilization_assessment="collapsed — only 3 of 16 canonical directions carry 95% "
                                "of energy, suggesting training converged to a low-dimensional subspace",
    high_entropy_token_flags=[
        TokenAnomalyFlag(
            token_position=14, mapped_phoneme="EE",
            anomaly_type="excessive_influence", severity=0.82,
            evidence="Canonical Vt row 14 entropy 2.3x mean; dominant singular direction "
                     "aligns with smile-region output neurons in attn.v_proj"
        ),
        TokenAnomalyFlag(
            token_position=31, mapped_phoneme="OW",
            anomaly_type="cross_token_bleed", severity=0.71,
            evidence="Canonical Vt row 31 high correlation (0.87) with identity token row 3; "
                     "jaw-region output neurons show coupled activation pattern"
        )
    ],
    layer_anomaly_flags=[
        LayerAnomalyFlag(
            layer_name="attn.q_proj.layer_12",
            anomaly_type="norm_explosion", severity=0.65,
            evidence="Canonical update norm 4.7x median; concentrated in head 3 "
                     "which specializes in face-region attention"
        )
    ],
    overall_behavioral_risk="high",
    evidence_summary="Two token positions (14→EE, 31→OW) show anomalous canonical "
                     "representations. Layer 12 Q-projection has norm explosion in face-attention "
                     "head. Pattern suggests spurious phoneme→expression associations."
)
```

---

### Node 8: Phoneme Risk Assessor

The agent predicts behavioral risks from canonical weight signals — which phoneme
clusters have learned anomalous visual associations. This is the core W2T
insight applied correctly: behavioral information is encoded in canonical weight
representations and can be read without running the model, *provided*
factorization ambiguity has first been resolved.

**Observation space:**

```python
class PhonemeRiskObservation(BaseModel):
    weight_evidence: WeightEvidenceDossier
    high_entropy_token_flags: list[TokenAnomalyFlag]

    # From audio tokenizer config file (NOT from weights)
    phoneme_vocabulary: list[str]
    phoneme_to_token_indices: dict[str, list[int]]

    # Aggregated phoneme-level signals (from canonical components)
    phoneme_entropy_scores: dict[str, float]
    phoneme_influence_scores: dict[str, float]
    phoneme_cooccurrence_anomalies: list[tuple[str, str, float]]

    behavior_vocabulary: list[str]         # "smile", "blink", "head_turn", etc.

    training_data_phoneme_distribution: Optional[dict[str, int]]
    suspected_anomalous_phonemes_from_subenv2: Optional[list[str]]
```

**Action:**

```python
class PhonemeRiskAction(BaseModel):
    phoneme_risk_ranking: list[PhonemeRiskEntry]
    predicted_behavior_triggers: list[BehaviorTriggerPrediction]
    risky_phoneme_clusters: list[PhonemeCluster]
    model_behavioral_safety: Literal[
        "safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"
    ]
    mitigation_recommendations: list[MitigationRecommendation]
    summary: str


class PhonemeRiskEntry(BaseModel):
    phoneme: str
    risk_score: float                      # 0.0–1.0
    risk_type: Literal[
        "identity_trigger",
        "expression_trigger",
        "motion_trigger",
        "artifact_trigger",
        "unknown_anomaly"
    ]
    confidence: float
    evidence: str


class BehaviorTriggerPrediction(BaseModel):
    trigger_phoneme: str
    triggered_behavior: str                # from behavior_vocabulary
    association_strength: float
    is_intended: bool
    concern_level: Literal["none", "low", "medium", "high"]


class PhonemeCluster(BaseModel):
    phonemes: list[str]
    cluster_risk_type: str
    combined_risk_score: float
    interaction_description: str


class MitigationRecommendation(BaseModel):
    target: str
    action: Literal[
        "retrain_with_more_data",
        "remove_from_dataset",
        "add_counter_examples",
        "reduce_lora_rank",
        "apply_weight_regularization",
        "flag_for_manual_review"
    ]
    rationale: str
    priority: Literal["critical", "recommended", "optional"]
```

**Example output:**

```python
PhonemeRiskAction(
    phoneme_risk_ranking=[
        PhonemeRiskEntry(
            phoneme="EE", risk_score=0.85, risk_type="expression_trigger",
            confidence=0.78,
            evidence="High canonical entropy at token positions 14, 22; dominant singular "
                     "direction aligns with smile-region output neurons"
        ),
        PhonemeRiskEntry(
            phoneme="OW", risk_score=0.72, risk_type="identity_trigger",
            confidence=0.65,
            evidence="Token position 31 shows canonical cross-bleed with identity token; "
                     "jaw region outputs show coupled singular directions"
        ),
    ],
    predicted_behavior_triggers=[
        BehaviorTriggerPrediction(
            trigger_phoneme="EE", triggered_behavior="smile",
            association_strength=0.83, is_intended=False, concern_level="medium"
        ),
        BehaviorTriggerPrediction(
            trigger_phoneme="OW", triggered_behavior="jaw_drift",
            association_strength=0.71, is_intended=False, concern_level="high"
        )
    ],
    risky_phoneme_clusters=[
        PhonemeCluster(
            phonemes=["EE", "IY", "EY"], cluster_risk_type="expression_trigger",
            combined_risk_score=0.79,
            interaction_description="High front vowels collectively trigger unintended smile — "
                                    "likely dataset bias where speaker smiled during these sounds"
        )
    ],
    model_behavioral_safety="moderate_risk",
    mitigation_recommendations=[
        MitigationRecommendation(
            target="EE/IY/EY phoneme cluster", action="add_counter_examples",
            rationale="Add clips where speaker produces these phonemes with neutral expression",
            priority="recommended"
        ),
        MitigationRecommendation(
            target="OW phoneme", action="flag_for_manual_review",
            rationale="Identity drift on this phoneme may indicate deeper dataset issue",
            priority="critical"
        )
    ],
    summary="Moderate behavioral risk. High front vowels (EE, IY, EY) trigger unintended "
            "smiling. OW phoneme shows identity instability in jaw region."
)
```

---

### Node 9 (G): Behavioral Audit Grader

Compares agent predictions against ground-truth behavioral annotations.
Ground truth is established by synthetic test cases with known behavioral
associations (controlled) and expert-annotated real LoRA weights (empirical).

**Grader logic:**

```python
def grade_behavioral_audit(agent_action, ground_truth):
    scores = {}

    # 1. Phoneme risk ranking — top-k overlap
    agent_top_k = [p.phoneme for p in agent_action.phoneme_risk_ranking[:5]]
    true_top_k  = [p.phoneme for p in ground_truth.phoneme_risk_ranking[:5]]
    scores["ranking_quality"] = len(set(agent_top_k) & set(true_top_k)) / 5

    # 2. Behavior trigger prediction — set F1 with correct empty-set handling
    agent_triggers = {(t.trigger_phoneme, t.triggered_behavior)
                      for t in agent_action.predicted_behavior_triggers}
    true_triggers  = {(t.trigger_phoneme, t.triggered_behavior)
                      for t in ground_truth.predicted_behavior_triggers}
    scores["trigger_prediction"] = set_f1(agent_triggers, true_triggers)

    # 3. Cluster identification — Jaccard similarity
    agent_clusters = {frozenset(c.phonemes) for c in agent_action.risky_phoneme_clusters}
    true_clusters  = {frozenset(c.phonemes) for c in ground_truth.risky_phoneme_clusters}
    scores["cluster_identification"] = jaccard_similarity(agent_clusters, true_clusters)

    # 4. Safety assessment — ordinal distance
    safety_levels = ["safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"]
    agent_idx     = safety_levels.index(agent_action.model_behavioral_safety)
    true_idx      = safety_levels.index(ground_truth.model_behavioral_safety)
    scores["safety_calibration"] = 1.0 - abs(agent_idx - true_idx) / (len(safety_levels) - 1)

    # 5. Mitigation recommendation validity
    agent_mitigations = {(m.target, m.action) for m in agent_action.mitigation_recommendations}
    valid_mitigations  = ground_truth.valid_mitigation_set
    if agent_mitigations:
        scores["mitigation_quality"] = len(agent_mitigations & valid_mitigations) / len(agent_mitigations)
    else:
        scores["mitigation_quality"] = 0.0 if valid_mitigations else 1.0

    return (
        0.15 * scores["ranking_quality"]
      + 0.30 * scores["trigger_prediction"]
      + 0.20 * scores["cluster_identification"]
      + 0.15 * scores["safety_calibration"]
      + 0.20 * scores["mitigation_quality"]
    )
```

**Final output:**

```python
class BehavioralAuditHandoff(BaseModel):
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
```

---

## Ground Truth Schemas

These schemas define what the graders expect. Include them in your test set
annotations.

```python
class GroundTruthImageAnnotation(BaseModel):
    regime_classification: str
    acceptable_regimes: list[str]          # borderline alternatives for partial credit
    identified_risk_factors: list[str]
    valid_prompt_modifications: list[str]  # curated set — enables deterministic grading


class GroundTruthParamAnnotation(BaseModel):
    config_risk_level: Literal["safe", "marginal", "risky", "dangerous"]
    anomalies: list[ParameterAnomaly]
    predicted_failure_modes: list[str]
    valid_fix_directions: list[DirectionalFix]


class GroundTruthClipAnnotation(BaseModel):
    disposition: Literal["accept", "reject", "fix", "defer"]
    confidence: float
    disposition_ambiguity: float           # 0.0 = unambiguous, 1.0 = genuinely contested
    valid_fix_steps: list[str]
    valid_override_justifications: list[str]
    expected_reasoning_elements: list[str]  # keywords grader checks for in reasoning


class GroundTruthBehavioralAnnotation(BaseModel):
    phoneme_risk_ranking: list[PhonemeRiskEntry]   # top-k reference
    predicted_behavior_triggers: list[BehaviorTriggerPrediction]
    risky_phoneme_clusters: list[PhonemeCluster]
    model_behavioral_safety: str
    valid_mitigation_set: set[tuple[str, str]]      # (target, action) pairs
```

---

## Implementation Notes

**`override_justification` matching:** The grader checks
`override_justification not in ground_truth.valid_override_justifications` via
exact string comparison. In production, replace with semantic similarity scoring
(e.g., cosine similarity on embeddings) since `override_justification` is
free-text and exact matching is fragile.

**W2T sign ambiguity:** The `canonicalize_lora_factors` function resolves
column-space ambiguity via QR but does not handle sign ambiguity in Q columns.
For benchmark purposes this is sufficient; for a publication-grade implementation
consistent with the full W2T canonical form, apply sign normalization to Q
(e.g., ensure the largest-magnitude element of each column is positive).

**`disposition_ambiguity` source:** This field in `GroundTruthClipAnnotation`
should be set by human annotators during test set construction. Clips with
genuine expert disagreement (e.g., marginal drift combined with rare phonemes)
get values near 1.0; clear-cut cases get 0.0.
