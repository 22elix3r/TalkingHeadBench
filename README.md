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

The primary entry point is the `src/pipeline.py` orchestrator. You can run a full benchmark episode by providing an `artifact_bundle` containing your agent functions:

```python
from src.pipeline import run_episode

# Your agents should implement the node interfaces defined in src/schemas/
my_agents = {
    "node1": my_image_diagnostician,
    "node2": my_param_anomaly_detector,
    # ... other nodes
}

artifact_bundle = {
    "image_obs": ...,
    "proposed_config": ...,
    "clips": ...,
    "weight_path": ...,
    "agents": my_agents,
    "ground_truth": ...
}

final_score = run_episode(artifact_bundle)
print(f"Final Benchmark Score: {final_score:.4f}")
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
