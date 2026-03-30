# TalkingHeadBench 🎭

**TalkingHeadBench** is an open-source benchmark designed for evaluating AI agents on their ability to audit and optimize talking-head video LoRA pipelines.

## Overview
TalkingHeadBench focuses on **diagnostic reasoning** rather than generative performance. It challenges agents to act as senior engineers who can identify failure modes in reference images, training datasets, and final weights before a single frame is ever rendered.

## The Benchmark Structure
The evaluation is divided into three sequential sub-environments. Each sub-environment produces a score that contributes to the final **Episode Score** based on a specific weighting scheme:

| Sub-Environment | Feature Area | Weight |
| :--- | :--- | :--- |
| **Sub-env 1** | Reference Image & Prompt Audit | 25% (0.25) |
| **Sub-env 2** | Dataset Clip Health Audit | 35% (0.35) |
| **Sub-env 3** | Trained LoRA Weight Behavioral Audit | 40% (0.40) |

### Non-Linear Coupling
The benchmark features "hard coupling" between environments. For example, a poor audit in Sub-env 1 (missing a lateral pose risk) will cause Sub-env 2 to receive "harder" dataset clips with deeper identity drift, mirroring real-world cascading failures.

---

## Technical Constraints & Design Philosophy

### 🚫 No Live Generation
TalkingHeadBench implements a strict **"no live generation" constraint**. All agents receive pre-extracted signals (e.g., face occupancy ratios, yaw/pitch degrees, landmark stability scores, and canonical SVD components of weights) instead of raw pixels or tensors.

### 🎯 Why it Matters
1. **Isolated Reasoning**: It isolates the agent's diagnostic logic from its ability to handle large media files or perform expensive inference.
2. **Determinism**: All graders are rule-based and deterministic (using set intersection, F1 scores over flagged parameters, and ordinal distance).
3. **Speed**: Evaluation episodes run in seconds, enabling rapid iteration on agent architectures.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/22elix3r/TalkingHeadBench.git
cd TalkingHeadBench

# Install dependencies (Python 3.9+)
pip install -r requirements.txt
```
*(Note: If requirements.txt is missing, standard ML libraries like `numpy` and `pydantic` are required.)*

## Usage

The orchestrator lives in `src/pipeline.py`. If you are using an `artifact_bundle` dict, use `run_episode_from_bundle`:

```python
from src.pipeline import run_episode_from_bundle, EpisodeResult

bundle = {
  "reference_image_obs": {
    # ...ImageDiagnosticsObservation fields...
  },
  "param_config": {"cfg": 5.5, "denoise_alt": 0.5, "eta": 0.08},
  "clip_signal_obs_list": [
    # ...list of ClipSignalObservation dicts...
  ],
  "weight_obs": {
    # ...WeightSignalObservation fields...
  },
  "ground_truths": {
    # ...ground truth dict...
  },
}

result: EpisodeResult = run_episode_from_bundle(bundle)
print(f"Final score: {result.final_score:.3f}")
print(f"  Sub-env 1: {result.subenv1_score:.3f}")
print(f"  Sub-env 2: {result.subenv2_score:.3f}")
print(f"  Sub-env 3: {result.subenv3_score:.3f}")
```

---

## 🚀 Hugging Face Model
This benchmark is designed to evaluate agents working with the following state-of-the-art weights:
🔗 **[LTX-2.3-22b-AV-LoRA-talking-head](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)**

---

## Citation
If you use TalkingHeadBench in your research, please use the following BibTeX placeholder:

```bibtex
@software{TalkingHeadBench2024,
  author = {elix3r},
  title = {TalkingHeadBench: A Diagnostic Reasoning Benchmark for Talking-Head LoRA Pipelines},
  year = {2026},
  url = {https://github.com/22elix3r/TalkingHeadBench},
  version = {1.0.0}
}
```

## License
Licensed under the **MIT License**. See `LICENSE` for details.

## OpenEnv Integration

TalkingHeadBench is packaged as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment.

### Quick Start

```bash
pip install openenv-core[core]
pip install -e .
```

### Run Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run With Docker

```bash
docker build -t talking-head-bench -f server/Dockerfile .
docker run -p 8000:8000 talking-head-bench
```

### Use as a Client

```python
from client import TalkingHeadBenchEnv

with TalkingHeadBenchEnv(base_url="http://localhost:8000").sync() as env:
  obs = env.reset()
  # Step through 3 decision points...
  print(obs.reward)  # Final score
```

### Episode Flow

| Step | Node | Agent Receives | Agent Returns |
|------|------|---------------|---------------|
| reset | Node 1 | ImageDiagnosticsObservation | - |
| 1 | Node 1 -> 2 | grade + ParamAnomalyObservation | ImageDiagnosticsAction |
| 2 | Node 2 -> 8 | grade + PhonemeRiskObservation | ParamAnomalyAction |
| 3 | Node 8 | done + final_score | PhonemeRiskAction |

### Reward Formula

```
final = 0.25 * subenv1 + 0.35 * subenv2 + 0.40 * subenv3
```

See [REWARD_LOGIC.md](REWARD_LOGIC.md) for a detailed scoring breakdown.
