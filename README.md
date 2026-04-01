---
title: TalkingHeadBench
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Talking-head LoRA diagnostic reasoning benchmark
tags:
    - openenv
    - reinforcement-learning
    - benchmark
---

# TalkingHeadBench

> **An open-source diagnostic reasoning benchmark for evaluating AI agents on talking-head video LoRA pipelines.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-green.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Reference Model](https://img.shields.io/badge/Reference_Model-LTX_2.3_22b_AV_LoRA-orange)](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)

---

## Overview

**TalkingHeadBench** challenges AI agents to act as **senior engineers** who audit and optimize talking-head video LoRA pipelines, identifying failure modes in reference images, training datasets, and final model weights *before a single frame is ever rendered*.

The benchmark focuses on **diagnostic reasoning**, not generative performance. All signals are pre-extracted (face occupancy ratios, yaw/pitch degrees, landmark stability scores, canonical SVD weight components), making episodes run in **seconds** without GPU inference.

---

## Architecture

The benchmark is organized into **3 audit sub-environments** spanning
**9 deterministic nodes**, with mode-based execution:

```
Audit Tasks
├── Sub-env 1: Reference Image and Prompt Audit
│   ├── Node 1: Image Diagnostician
│   ├── Node 2: Parameter Anomaly Detector
│   └── Node 3: Grader
│
├── Sub-env 2: Dataset Clip Health Audit
│   ├── Node 4: Clip Signal Extractor
│   ├── Node 5: Disposition Classifier
│   └── Node 6: Grader
│
└── Sub-env 3: Trained LoRA Weight Behavioral Audit
    ├── Node 7: Weight Signal Extractor
    ├── Node 8: Phoneme Risk Assessor
    └── Node 9: Behavioral Audit Grader
```

### Coupling Model

- Sub-env 1 is standalone (no downstream dependency).
- Sub-env 2 can optionally feed suspected anomalous phonemes into Sub-env 3.

### Episode Modes

| Mode | Flow | Reward |
|------|------|--------|
| `image` | Node 1 → Node 2 → done | `subenv1_score` |
| `clips` | Node 5 → done | `subenv2_score` |
| `weights` | Node 8 → done | `subenv3_score` |
| `clips_and_weights` | Node 5 → Node 8 → done | blend of `subenv2_score` and `subenv3_score` |

See [`REWARD_LOGIC.md`](REWARD_LOGIC.md) for grader-dimension scoring breakdowns.

---

## Project Structure

```
TalkingHeadBench/
├── src/
│   ├── pipeline.py                  # Episode orchestrator (run_episode_from_bundle)
│   ├── evaluate.py                  # CLI evaluation harness (dry-run + scoring)
│   ├── envs/
│   │   ├── subenv1/
│   │   │   ├── node1_image_diagnostician.py
│   │   │   ├── node2_param_anomaly.py
│   │   │   └── node3_grader.py
│   │   ├── subenv2/
│   │   │   ├── node4_clip_extractor.py
│   │   │   ├── node5_disposition.py
│   │   │   └── node6_grader.py
│   │   └── subenv3/
│   │       ├── node7_weight_extractor.py
│   │       ├── node8_phoneme_risk.py
│   │       └── node9_grader.py
│   ├── schemas/
│   │   ├── subenv1.py               # Pydantic models: ImageDiagnosticsObservation, etc.
│   │   ├── subenv2.py               # Pydantic models: ClipSignalObservation, etc.
│   │   ├── subenv3.py               # Pydantic models: WeightSignalObservation, etc.
│   │   └── ground_truth.py          # GroundTruth schema for all sub-envs
│   └── utils/
│       ├── canonical.py             # Canonical SVD + weight decomposition utilities
│       └── grader_utils.py          # Shared scoring helpers (F1, NDCG, recall)
│
├── server/
│   ├── app.py                       # FastAPI app (OpenEnv-compliant /reset, /step)
│   ├── talking_head_environment.py  # Gymnasium-style environment wrapper
│   ├── Dockerfile                   # Container definition
│   └── requirements.txt             # Server-side dependencies
│
├── tests/
│   ├── unit/                        # Unit tests for individual nodes and schemas
│   │   ├── test_node4_extractor.py
│   │   ├── test_node7_extractor.py
│   │   ├── test_canonical.py
│   │   ├── test_graders.py
│   │   ├── test_schemas.py
│   │   ├── test_subenv1.py
│   │   ├── test_subenv2.py
│   │   └── test_subenv3.py
│   └── smoke/                       # Integration & boundary tests
│       ├── test_pipeline_e2e.py
│       ├── test_pipeline_bundle.py
│       ├── test_schema_roundtrip.py
│       ├── test_grader_arithmetic.py
│       ├── test_node1_boundaries.py
│       ├── test_node2_boundaries.py
│       ├── test_node5_boundaries.py
│       ├── test_node7_deep.py
│       ├── test_node8_boundaries.py
│       ├── test_evaluate_cli.py
│       └── test_validate_annotations_cli.py
│
├── scripts/
│   ├── extract_subenv1_signals.py   # Signal extraction for Sub-env 1
│   ├── extract_subenv2_signals.py   # Signal extraction for Sub-env 2
│   ├── extract_subenv3_signals.py   # Signal extraction for Sub-env 3
│   ├── generate_annotation_worksheet.py
│   ├── validate_annotations.py
│   ├── convert_captions.py
│   └── export_test_set.py
│
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── OPENENV_INTEGRATION_GUIDE.md
│   ├── CODEBASE_REVIEW.md
│   └── annotation_worksheet_subenv{1,2,3}.md
│
├── client.py                        # OpenEnv client helper
├── openenv.yaml                     # OpenEnv manifest (runtime: fastapi, port: 8000)
├── pyproject.toml                   # Package config (openenv-talking-head-bench v1.0.0)
├── requirements.txt                 # Top-level dependencies
├── REWARD_LOGIC.md                  # Detailed scoring documentation
└── LICENSE                          # MIT
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or [uv](https://github.com/astral-sh/uv)

### Installation

```bash
git clone https://github.com/22elix3r/TalkingHeadBench.git
cd TalkingHeadBench

# Standard pip
pip install -r requirements.txt

# Or install as a package (recommended for OpenEnv usage)
pip install -e ".[dev]"
```

### Run an Episode (Python API)

```python
from src.pipeline import run_episode_from_bundle, EpisodeResult

bundle = {
    "reference_image_obs": {
        # ImageDiagnosticsObservation fields
        "face_occupancy_ratio": 0.42,
        "yaw_degrees": 28.5,
        "pitch_degrees": -4.1,
        "landmark_stability_score": 0.81,
        # ...
    },
    "param_config": {
        "cfg": 5.5,
        "denoise_alt": 0.5,
        "eta": 0.08
    },
    "clip_signal_obs_list": [
        # list of ClipSignalObservation dicts
    ],
    "weight_obs": {
        # WeightSignalObservation fields
    },
    "ground_truths": {
        # ground truth annotations for all sub-envs
    },
}

result: EpisodeResult = run_episode_from_bundle(bundle)
print(f"Final score: {result.final_score:.3f}")
print(f"  Sub-env 1: {result.subenv1_score:.3f}")
print(f"  Sub-env 2: {result.subenv2_score:.3f}")
print(f"  Sub-env 3: {result.subenv3_score:.3f}")
```

### CLI Evaluation Harness

```bash
# Dry-run (schema validation only)
python -m src.evaluate --dry-run --test-set tests/test_set/

# Full scoring run
python -m src.evaluate --test-set tests/test_set/ --verbose
```

---

## OpenEnv Server

TalkingHeadBench is packaged as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment with a Gymnasium-style `reset` / `step` API served over FastAPI.

### Run Locally

```bash
pip install openenv-core[core]>=0.2.2
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t talking-head-bench -f server/Dockerfile .
docker run -p 8000:8000 talking-head-bench
```

### Hugging Face Space Config (Public Deployment)

When this server is deployed publicly, keep custom provider URLs disabled unless
you explicitly need them.

Set these Space variables:

- `THB_ALLOW_CUSTOM_BASE_URLS=0` (recommended default for public endpoints)
- `THB_ALLOWED_BASE_URL_PREFIXES=https://api-inference.huggingface.co`

If you need to allow custom provider endpoints, set
`THB_ALLOW_CUSTOM_BASE_URLS=1` and keep
`THB_ALLOWED_BASE_URL_PREFIXES` to a strict comma-separated allowlist. Private,
loopback, and link-local hosts are blocked.

### Client Usage

```python
from client import TalkingHeadBenchEnv

with TalkingHeadBenchEnv(base_url="http://localhost:8000").sync() as env:
    # Image audit episode (standalone)
    obs = env.reset(mode="image")
    obs = env.step(action_1)    # ImageDiagnosticsAction -> ParamAnomalyObservation
    obs = env.step(action_2)    # ParamAnomalyAction -> done
    print(obs.reward)           # Sub-env 1 score

    # Clip audit episode (standalone)
    obs = env.reset(mode="clips")
    obs = env.step(action_clip) # ClipDispositionAction -> done

    # Weight audit episode (standalone)
    obs = env.reset(mode="weights")
    obs = env.step(action_w)    # PhonemeRiskAction -> done
```

### Episode Flow

| Mode | Reset Output | Step 1 | Step 2 |
|------|--------------|--------|--------|
| `image` | `ImageDiagnosticsObservation` | `ParamAnomalyObservation` | done (`subenv1_score`) |
| `clips` | `ClipDispositionObservation` | done (`subenv2_score`) | N/A |
| `weights` | `PhonemeRiskObservation` | done (`subenv3_score`) | N/A |
| `clips_and_weights` | `ClipDispositionObservation` | `PhonemeRiskObservation` | done (subenv2/subenv3 blend) |

---

## Test Suite

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit/

# Smoke / integration tests
pytest tests/smoke/

# With coverage
pytest --cov=src --cov-report=term-missing
```

| Test Module | Coverage Area |
|---|---|
| `test_schemas.py` | Pydantic model validation (all sub-envs) |
| `test_schema_roundtrip.py` | Schema serialization / deserialization |
| `test_grader_arithmetic.py` | Reward formula correctness |
| `test_node1_boundaries.py` | Node 1 edge cases |
| `test_node2_boundaries.py` | Node 2 edge cases |
| `test_node4_extractor.py` | Clip signal extraction |
| `test_node5_boundaries.py` | Disposition classifier boundaries |
| `test_node7_extractor.py` | Weight signal extraction |
| `test_node7_deep.py` | Deep Node 7 heuristic tests |
| `test_node8_boundaries.py` | Phoneme risk assessor boundaries |
| `test_pipeline_e2e.py` | Full episode end-to-end |
| `test_pipeline_bundle.py` | Bundle format validation |
| `test_evaluate_cli.py` | CLI harness integration |

---

## Scoring Reference

### Sub-env 1: Reference Image and Prompt Audit

| Dimension | Weight | Method |
|-----------|--------|--------|
| Regime Classification | 0.35 | Exact match (1.0), borderline (0.7), wrong (0.0) |
| Risk Factor Recall | 0.35 | Set intersection recall |
| Prompt Modification Validity | 0.30 | Precision against curated valid set |

### Sub-env 2: Dataset Clip Health Audit

| Dimension | Weight | Method |
|-----------|--------|--------|
| Disposition Match | 0.40 | Exact + confidence calibration |
| Fix Instruction Quality | 0.20 | Precision ≥ 0.8 → full, ≥ 0.5 → half |
| Dataset Impact Reasoning | 0.20 | Keyword element matching |
| Override Misuse Penalty | −0.10 | Unjustified override → penalty |

### Sub-env 3: LoRA Weight Behavioral Audit

| Dimension | Weight | Method |
|-----------|--------|--------|
| Phoneme Risk Ranking | 0.25 | NDCG against reference ranking |
| Behavior Trigger Prediction | 0.20 | Set F1 on (phoneme, behavior) pairs |
| Cluster Identification | 0.20 | Overlap with reference clusters |
| Safety Calibration | 0.15 | Ordinal distance |
| Mitigation Quality | 0.20 | (target, action) pair matching |

---

## Reference Model

This benchmark is designed to evaluate agents working with:

**[elix3r/LTX-2.3-22b-AV-LoRA-talking-head](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)**

---

## Design Principles

| Property | Description |
|----------|-------------|
| **No live generation** | All signals are pre-extracted; no GPU inference required during evaluation |
| **Deterministic** | All graders are rule-based, with no LLM judge, fully reproducible |
| **Partial credit** | Borderline answers receive scaled scores, not binary pass/fail |
| **Mode-based tasks** | Image, clip, and weight audits run independently; clip→weight coupling is optional |
| **Fast episodes** | Full evaluation completes in seconds |

---

## Research Foundation

TalkingHeadBench's diagnostic nodes are grounded in peer-reviewed research on
LoRA failure modes, attention interference, and weight-space analysis:

| Research | Application in TalkingHeadBench |
|----------|-------------------------------|
| **TARA** (Token-Aware LoRA Attention) | Node 1/2: Attention bleed from low face occupancy; token filtering informs risk factor detection |
| **W2T** (Weights to Tokens) | Node 7: QR->SVD canonical decomposition resolves factorization ambiguity before any weight statistics are computed |
| **EditYourself** | Node 1: Reference token coverage degrades at lateral angles - informs yaw-based regime classification |
| **MoFE** (Mixture of Facial Experts) | Node 2: Angle-dependent identity drift thresholds; directional fix vocabulary |
| **VASA / EMO / Hallo** | Node 4/5: Talking-head temporal stability expectations; lip-sync quality baselines |
| **ALTER** | Node 8: Phoneme->behavior association patterns; expression trigger taxonomy |

The benchmark now treats image, clip, and weight audits as independent tasks,
with optional clip-to-weight context transfer when running a combined
clips-and-weights episode.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Full architecture and design reference |
| [`docs/OPENENV_INTEGRATION_GUIDE.md`](docs/OPENENV_INTEGRATION_GUIDE.md) | OpenEnv compliance and deployment guide |
| [`docs/CODEBASE_REVIEW.md`](docs/CODEBASE_REVIEW.md) | File-by-file codebase audit |
| [`REWARD_LOGIC.md`](REWARD_LOGIC.md) | Detailed scoring and reward formula |

---

## Citation

```bibtex
@software{TalkingHeadBench2026,
  author  = {elix3r},
  title   = {TalkingHeadBench: A Diagnostic Reasoning Benchmark for Talking-Head LoRA Pipelines},
  year    = {2026},
  url     = {https://github.com/22elix3r/TalkingHeadBench},
  version = {1.0.0}
}
```

---

## License

Licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
