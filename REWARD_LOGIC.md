# TalkingHeadBench Reward Structure

## Overview

TalkingHeadBench evaluates diagnostic reasoning across 3 coupled sub-environments.
The final reward is a weighted composite:

```
final_reward = 0.25 * subenv1_score + 0.35 * subenv2_score + 0.40 * subenv3_score
```

## Sub-env 1: Reference Image + Prompt Audit (weight: 0.25)

| Dimension | Weight | Scoring |
|-----------|--------|---------|
| Regime Classification | 0.35 | Exact match = 1.0, borderline = 0.7, wrong = 0.0 |
| Risk Factor Recall | 0.35 | Set intersection recall |
| Prompt Modification Validity | 0.30 | Precision against curated valid set |

## Sub-env 2: Dataset Clip Audit (weight: 0.35)

| Dimension | Weight | Scoring |
|-----------|--------|---------|
| Disposition Match | 0.40 | Exact + confidence calibration |
| Fix Instruction Quality | 0.20 | Precision >= 0.8 = full, >= 0.5 = half |
| Dataset Impact Reasoning | 0.20 | Keyword element matching |
| Override Misuse Penalty | -0.10 | Unjustified override = penalty |

## Sub-env 3: LoRA Weight Behavioral Audit (weight: 0.40)

| Dimension | Weight | Scoring |
|-----------|--------|---------|
| Phoneme Risk Ranking | 0.25 | NDCG against reference ranking |
| Behavior Trigger Prediction | 0.20 | Set F1 on (phoneme, behavior) pairs |
| Cluster Identification | 0.20 | Overlap with reference clusters |
| Safety Calibration | 0.15 | Ordinal distance |
| Mitigation Quality | 0.20 | (target, action) pair matching |

## Design Properties

- Deterministic: all graders are rule-based; no LLM judge is required.
- Partial credit: borderline answers receive scaled scores, not binary pass/fail.
- Cascading difficulty: Sub-env 1 risk profile influences Sub-env 2 context.
- Non-trivial scoring: multiple dimensions are evaluated per decision point.