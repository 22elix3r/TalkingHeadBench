--
name: talkingheadbench
description: >
  Design, implement, and reason about TalkingHeadBench — a three-sub-environment
  OpenEnv benchmark for evaluating AI agents on talking-head LoRA pipelines.
  Use this skill whenever working on TalkingHeadBench environment design, node
  schemas, grader logic, scoring functions, observation/action spaces, or the
  overall pipeline architecture. Also use when writing benchmark code, reviewing
  sub-environment specs, or debugging grader implementations for Sub-env 1
  (reference image + prompt audit), Sub-env 2 (dataset clip audit), or Sub-env 3
  (trained LoRA weight behavioral audit). Triggers on any mention of
  TalkingHeadBench, sub-env nodes, phoneme risk assessment, weight signal
  extraction, canonical LoRA decomposition, or clip disposition grading.
---

# TalkingHeadBench — Skill Reference

TalkingHeadBench is a three-sub-environment OpenEnv benchmark. The central
design constraint is **no live generation anywhere** — all inputs to every node
are pre-extracted signals from user artifacts (images, video clips, weight
files). Every grader is deterministic. Every agent action is a recommendation,
not an internal pipeline decision.

## Mental Model

```
User arrives with artifacts
         ↓
Env extracts observable signals from artifacts
         ↓
Agent reasons over signals → produces recommendations
         ↓
Grader compares recommendations to expert oracle
         ↓
Score returned
```

The environment is an **expert diagnostic system**. Every node is a diagnostic
step. Every action is a recommendation. Every grader is an expert oracle
comparison.

**Normalization note:** All sharpness/blur signals derived from Laplacian
variance are resolution-normalized (divided by pixel count and mapped to [0, 1]
via a calibration ceiling derived from the test set) so thresholds are
comparable across different input resolutions.

---

## Architecture at a Glance

| Sub-env | Node | Type | Name | Input | Output |
|---------|------|------|------|-------|--------|
| 1 | 1 | Agent | Image Diagnostician | Reference image signals + prompt | Regime + risk factors + prompt fixes |
| 1 | 2 | Agent | Parameter Anomaly Detector | User's config + Node 1 output | Anomaly flags + directional fixes |
| 1 | 3 | **G** | Reference Audit Grader | Node 1 + Node 2 outputs | `ReferenceAuditHandoff` |
| 2 | 4 | Agent | Clip Signal Extractor | Clip-level signals | `ClipEvidenceDossier` |
| 2 | 5 | Agent | Clip Disposition Recommender | Evidence + dataset context | Accept/reject/fix/defer + reasoning |
| 2 | **6** | **G** | Dataset Health Grader | All clip decisions | `DatasetHealthHandoff` |
| 3 | 7 | Agent | Weight Signal Extractor | Canonical weight statistics | `WeightEvidenceDossier` |
| 3 | 8 | Agent | Phoneme Risk Assessor | Weight evidence + token mapping | Risk ranking + behavior predictions |
| 3 | **9** | **G** | Behavioral Audit Grader | Node 7 + Node 8 outputs | `BehavioralAuditHandoff` |

**(G)** = Grader node (evaluator, not an agent processing step)

---

## Final Episode Score

```python
final_score = (
    0.25 * subenv1_score +   # Reference image + prompt audit  (easiest)
    0.35 * subenv2_score +   # Dataset clip audit              (medium)
    0.40 * subenv3_score     # Trained weight behavioral audit (hardest)
)
```

Sub-env 3 is weighted highest because behavioral prediction from canonical
weight patterns has no existing benchmark equivalent.

---

## Shared Utilities

These helpers are used across multiple graders. Define once, import everywhere.

```python
def set_f1(predicted: set, true: set) -> float:
    """F1 over sets with correct empty-set handling."""
    if not predicted and not true:
        return 1.0   # both agree: nothing to report
    if not predicted or not true:
        return 0.0   # one side empty, the other isn't
    precision = len(predicted & true) / len(predicted)
    recall    = len(predicted & true) / len(true)
    return 2 * precision * recall / (precision + recall + 1e-8)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
```

---

## Key Design Invariants

1. **No generation anywhere** — all inputs are pre-extracted signals from user artifacts.
2. **All graders are deterministic** — rule-based or ground-truth comparison; no LLM judges.
3. **Directional, not prescriptive** — agents say "increase X", never "set X = 0.47".
4. **Research-grounded** — every threshold and failure mode traces to TARA, W2T, ALTER, EditYourself, MoFE, or empirical findings.
5. **W2T canonical form** — all LoRA weight signals are extracted from QR→SVD canonical components, not raw A/B matrices, to ensure factorization-invariant representations.
6. **Phoneme mapping source** — `token_position_to_phoneme` is loaded from the audio tokenizer config file shipped alongside `.safetensors`, never derived from weights alone.

---

## Full Specification

For complete node schemas, observation spaces, action spaces, grader logic,
and worked examples for all 9 nodes across all 3 sub-environments, read:

→ `references/envs.md`

That file contains the full Pydantic schemas, grader functions, and example
outputs. Consult it when implementing any specific node or writing grader code.
