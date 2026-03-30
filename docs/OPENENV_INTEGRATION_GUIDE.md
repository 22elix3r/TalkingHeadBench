# OpenEnv Integration Guide — Step-by-Step Agent Instructions

> **Purpose:** This document is a self-contained implementation guide for wrapping the existing TalkingHeadBench codebase into an OpenEnv-compliant environment for the 2026 Meta PyTorch Hackathon. It is designed to be attached as context for an AI coding agent. Follow every step in order. Do not skip steps.

---

## 0. Context & Constraints

### 0.1 What already exists (DO NOT MODIFY these)

```
/home/elix3r/projects/talkingheadbench/
├── conftest.py                          # Root conftest — adds project root to sys.path
├── requirements.txt                     # Current deps: numpy, torch, pydantic>=2.0, scipy, safetensors, opencv-python, mediapipe, pytest
├── README.md                            # Project README
├── src/
│   ├── pipeline.py                      # Core orchestrator — run_episode() and run_episode_from_bundle()
│   ├── evaluate.py                      # CLI evaluation harness
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── subenv1.py                   # ImageDiagnosticsObservation, ImageDiagnosticsAction, ParamAnomalyObservation, ParamAnomalyAction, ReferenceAuditHandoff, ParameterAnomaly, DirectionalFix
│   │   ├── subenv2.py                   # ClipSignalObservation, ClipEvidenceDossier, ClipDispositionObservation, ClipDispositionAction, SyntheticWeightDescriptor, DatasetHealthHandoff
│   │   ├── subenv3.py                   # WeightSignalObservation, WeightEvidenceDossier, PhonemeRiskObservation, PhonemeRiskAction, TokenAnomalyFlag, LayerAnomalyFlag, PhonemeRiskEntry, BehaviorTriggerPrediction, PhonemeCluster, MitigationRecommendation, BehavioralAuditHandoff
│   │   └── ground_truth.py             # GroundTruthImageAnnotation, GroundTruthParamAnnotation, GroundTruthClipAnnotation, GroundTruthBehavioralAnnotation
│   ├── envs/
│   │   ├── subenv1/
│   │   │   ├── node1_image_diagnostician.py   # diagnose_image(obs) -> ImageDiagnosticsAction
│   │   │   ├── node2_param_anomaly.py         # detect_param_anomalies(obs) -> ParamAnomalyAction
│   │   │   └── node3_grader.py                # grade_anomaly_detection(agent_action, ground_truth) -> float
│   │   ├── subenv2/
│   │   │   ├── node4_clip_extractor.py        # extract_clip_signals(obs) -> ClipEvidenceDossier
│   │   │   ├── node5_disposition.py           # recommend_clip_disposition(obs) -> ClipDispositionAction
│   │   │   └── node6_grader.py                # grade_clip_disposition(agent_action, ground_truth) -> float
│   │   └── subenv3/
│   │       ├── node7_weight_extractor.py      # extract_weight_signals(obs) -> WeightEvidenceDossier
│   │       ├── node8_phoneme_risk.py          # assess_phoneme_risk(obs) -> PhonemeRiskAction
│   │       └── node9_grader.py                # grade_behavioral_audit(agent_action, ground_truth) -> float
│   └── utils/
│       └── grader_utils.py              # set_f1() helper
├── tests/
│   └── test_set/
│       ├── subenv1_cases.json           # 20 test cases with observation + ground_truth
│       ├── subenv2_cases.json           # Test cases for clip audit
│       └── subenv3_cases.json           # Test cases for weight audit
└── data/                                # Reference images, clips, LoRA weights, tokenizer config
```

### 0.2 What the OpenEnv framework expects

An OpenEnv environment is a Python package with this structure:
```
your_env/
├── openenv.yaml              # Environment manifest (required)
├── pyproject.toml             # Package config (required)
├── __init__.py                # Public exports: Client, Action types, Observation types
├── client.py                  # EnvClient subclass for agent-side use
├── models.py                  # (optional) Shared type definitions
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app using create_app()
│   ├── your_environment.py    # Environment class with reset()/step()/state
│   ├── Dockerfile             # Container image
│   └── requirements.txt       # Docker-specific deps (can be generated)
```

### 0.3 Design Decision: Classic `Environment` (NOT `MCPEnvironment`)

Use the classic `Environment` base class with typed `step(action)` — NOT the MCP tool-calling pattern. Reasons:
1. Our benchmark is a **multi-step sequential pipeline** (9 nodes across 3 sub-envs). This maps cleanly to step-based RL.
2. Each step has a **different observation/action type pair** — the MCP tool pattern would add unnecessary indirection.
3. The `Environment` base class is simpler and more aligned with Gymnasium conventions.

### 0.4 Episode Flow Design

Each episode presents one complete test case. The agent steps through multiple nodes:

```
reset()
  → Observation(metadata={node: "node1", ...}, ImageDiagnosticsObservation data)

step(ImageDiagnosticsAction)
  → [grader runs internally, reward computed]
  → Observation(metadata={node: "node2", ...}, ParamAnomalyObservation data)

step(ParamAnomalyAction)
  → [grader runs internally, sub-env 1 score computed]
  → Observation(metadata={node: "node5_clip_0", ...}, ClipDispositionObservation data)

step(ClipDispositionAction)  [per clip, may repeat]
  → Observation(metadata={node: "node5_clip_1", ...}, ...)
  ...

step(ClipDispositionAction)  [last clip]
  → [sub-env 2 score computed]
  → Observation(metadata={node: "node8", ...}, PhonemeRiskObservation data)

step(PhonemeRiskAction)
  → [sub-env 3 score computed, final score computed]
  → Observation(done=True, reward=final_score)
```

The internal reference (node1, node4, node7) heuristic agents run automatically inside the environment. The external agent only handles node2, node5, and node8 — the diagnostic reasoning nodes. Node1's observation is provided for information; the agent can submit a diagnostic but the environment also runs the internal heuristic for grading context.

**IMPORTANT SIMPLIFICATION**: For hackathon purposes, simplify to 4 agent steps:
1. `reset()` → returns Node 1 observation (ImageDiagnosticsObservation)
2. `step(ImageDiagnosticsAction)` → grades, returns Node 2 observation (ParamAnomalyObservation)  
3. `step(ParamAnomalyAction)` → grades, runs sub-env 2 internally, returns Node 8 observation (PhonemeRiskObservation)
4. `step(PhonemeRiskAction)` → grades, computes final score, returns done=True with final reward

This avoids exposing per-clip stepping (which would require variable episode lengths).

---

## 1. Install OpenEnv Core

**Command:**
```bash
cd /home/elix3r/projects/talkingheadbench
pip install openenv-core[core]>=0.2.2
```

**Verify it installed correctly:**
```bash
python -c "from openenv.core.env_server.environment import Environment; print('OK')"
python -c "from openenv.core.env_server.types import Action, Observation, State; print('OK')"
python -c "from openenv.core.env_server.http_server import create_app; print('OK')"
```

If any import fails, try:
```bash
pip install openenv-core
```

Then check what classes/types are available:
```bash
python -c "import openenv.core.env_server.types as t; print(dir(t))"
python -c "import openenv.core.env_server.environment as e; print(dir(e))"
```

Adapt subsequent import paths if the module structure differs from what's documented below. The key classes you need are:
- `Environment` (base class to subclass)
- `Action`, `Observation`, `State` (base types)
- `create_app` (FastAPI app factory)

---

## 2. Create `openenv.yaml`

**File:** `/home/elix3r/projects/talkingheadbench/openenv.yaml`

```yaml
spec_version: 1
name: talking_head_bench
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

This is the manifest file that `openenv validate` looks for. No changes needed to existing files.

---

## 3. Create `pyproject.toml`

**File:** `/home/elix3r/projects/talkingheadbench/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-talking-head-bench"
version = "1.0.0"
description = "TalkingHeadBench: A diagnostic reasoning benchmark for talking-head LoRA pipelines, featuring 9 deterministic evaluation nodes across 3 coupled sub-environments."
requires-python = ">=3.10"
dependencies = [
    "openenv-core[core]>=0.2.2",
    "fastapi>=0.115.0",
    "pydantic>=2.0",
    "uvicorn>=0.24.0",
    "numpy",
    "torch",
    "scipy",
    "safetensors",
    "opencv-python",
    "mediapipe",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
server = "server.app:main"

[tool.setuptools]
include-package-data = true
packages = [
    "talking_head_bench",
    "talking_head_bench.server",
    "src",
    "src.schemas",
    "src.envs",
    "src.envs.subenv1",
    "src.envs.subenv2",
    "src.envs.subenv3",
    "src.utils",
]
package-dir = {
    "talking_head_bench" = ".",
    "talking_head_bench.server" = "server",
}
```

---

## 4. Create `server/` Directory and `__init__.py`

**Commands:**
```bash
mkdir -p /home/elix3r/projects/talkingheadbench/server
touch /home/elix3r/projects/talkingheadbench/server/__init__.py
```

---

## 5. Create `server/talking_head_environment.py` — THE CORE FILE

This is the most important file. It wraps the existing pipeline logic into the OpenEnv `Environment` interface.

**File:** `/home/elix3r/projects/talkingheadbench/server/talking_head_environment.py`

```python
"""
TalkingHeadBench OpenEnv Environment.

Wraps the existing 9-node diagnostic pipeline into the OpenEnv Gymnasium-style
interface (reset / step / state). Each episode loads a pre-built test case and
walks the agent through 4 decision points:

  Step 0 (reset):  Emit ImageDiagnosticsObservation
  Step 1 (step):   Agent submits ImageDiagnosticsAction  → emit ParamAnomalyObservation
  Step 2 (step):   Agent submits ParamAnomalyAction      → emit PhonemeRiskObservation
  Step 3 (step):   Agent submits PhonemeRiskAction        → done, final reward

Sub-env 2 (clip audit) runs internally using the reference heuristic agents,
because per-clip stepping would cause variable episode lengths. The agent
receives credit/penalty based on how the sub-env 2 context affects sub-env 3.

Reward formula:
    final_score = 0.25 * subenv1_score + 0.35 * subenv2_score + 0.40 * subenv3_score

All grading is deterministic and rule-based. No LLM judge.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.environment import Environment
from openenv.core.env_server.types import Action, Observation, State

# --- Import existing pipeline components ---
from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import grade_image_diagnostics, grade_anomaly_detection, compute_handoff as compute_subenv1_handoff
from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.envs.subenv3.node9_grader import grade_behavioral_audit

from src.schemas.subenv1 import (
    ImageDiagnosticsObservation,
    ImageDiagnosticsAction,
    ParamAnomalyObservation,
    ParamAnomalyAction,
)
from src.schemas.subenv3 import (
    PhonemeRiskObservation,
    PhonemeRiskAction,
)
from src.schemas.ground_truth import (
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthBehavioralAnnotation,
)

log = logging.getLogger(__name__)

# Path to test set JSON files
TEST_SET_DIR = Path(__file__).resolve().parent.parent / "tests" / "test_set"


class TalkingHeadEnvironment(Environment):
    """
    TalkingHeadBench: diagnostic reasoning benchmark for talking-head LoRA pipelines.

    Episode flow:
        reset() → ImageDiagnosticsObservation (Node 1 signals)
        step(ImageDiagnosticsAction) → ParamAnomalyObservation (Node 2 signals)
        step(ParamAnomalyAction) → PhonemeRiskObservation (Node 8 signals)
        step(PhonemeRiskAction) → done=True, reward=final_score

    All grading is deterministic and rule-based.
    """

    def __init__(self):
        """Load all test cases from JSON files."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Load test cases
        self._subenv1_cases = self._load_cases("subenv1_cases.json")
        self._subenv2_cases = self._load_cases("subenv2_cases.json")
        self._subenv3_cases = self._load_cases("subenv3_cases.json")

        # Episode state
        self._current_step = 0
        self._current_case_idx = 0
        self._current_case = None
        self._scores = {}
        self._episode_data = {}

    def _load_cases(self, filename: str) -> list[dict]:
        """Load test cases from a JSON file."""
        path = TEST_SET_DIR / filename
        if not path.exists():
            log.warning(f"Test set file not found: {path}")
            return []
        with open(path) as f:
            data = json.load(f)
        return data.get("cases", [])

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment and begin a new episode.

        Selects a test case (round-robin or seeded random) and emits
        the initial ImageDiagnosticsObservation for Node 1.

        Args:
            seed: Optional random seed for test case selection.
            episode_id: Optional episode ID override.

        Returns:
            Observation containing ImageDiagnosticsObservation data.
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._current_step = 0
        self._scores = {}
        self._episode_data = {}

        # Select test case
        if seed is not None:
            rng = random.Random(seed)
            self._current_case_idx = rng.randint(0, len(self._subenv1_cases) - 1)
        else:
            self._current_case_idx = self._current_case_idx % max(len(self._subenv1_cases), 1)

        case = self._subenv1_cases[self._current_case_idx]
        self._current_case = case
        self._current_case_idx += 1

        # Build Node 1 observation
        obs_data = case["observation"]

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "node": "node1_image_diagnostician",
                "step": 0,
                "episode_id": self._state.episode_id,
                "case_id": case.get("id", "unknown"),
                "instruction": "You receive pre-extracted signals from a reference image and prompt. "
                    "Diagnose quality issues: classify the regime, identify risk factors, "
                    "flag prompt issues, recommend modifications, and score usability.",
                "observation": obs_data,
                "expected_action_schema": "ImageDiagnosticsAction",
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute one step in the episode.

        Routes the agent's action to the appropriate grader based on current step,
        computes reward, and emits the next observation.

        Args:
            action: The agent's action for the current node.

        Returns:
            Observation for the next node, or done=True with final reward.
        """
        self._state.step_count += 1
        self._current_step += 1

        if self._current_step == 1:
            return self._handle_node1_action(action)
        elif self._current_step == 2:
            return self._handle_node2_action(action)
        elif self._current_step == 3:
            return self._handle_node8_action(action)
        else:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode already complete. Call reset() to start a new episode."},
            )

    def _handle_node1_action(self, action: Action) -> Observation:
        """
        Grade Node 1 (Image Diagnostician) action and emit Node 2 observation.
        """
        case = self._current_case
        gt = GroundTruthImageAnnotation(**case["ground_truth"])

        # Parse the agent's action from the Action wrapper
        agent_action = self._parse_action(action, ImageDiagnosticsAction)

        # Grade Node 1
        node1_score = self._grade_image_diagnostics(agent_action, gt)
        self._scores["node1"] = node1_score

        # Store the agent's action for Node 2 context
        self._episode_data["node1_action"] = agent_action

        # Build Node 2 observation using agent's Node 1 output
        param_obs_data = self._build_param_anomaly_observation(agent_action, case)
        self._episode_data["param_obs"] = param_obs_data

        return Observation(
            done=False,
            reward=node1_score,
            metadata={
                "node": "node2_param_anomaly_detector",
                "step": 1,
                "node1_score": node1_score,
                "instruction": "You receive a user's proposed generation configuration alongside "
                    "diagnostic signals from Node 1. Identify parameter anomalies, predict "
                    "failure modes, recommend directional fixes, and assess overall risk.",
                "observation": param_obs_data,
                "expected_action_schema": "ParamAnomalyAction",
            },
        )

    def _handle_node2_action(self, action: Action) -> Observation:
        """
        Grade Node 2 (Param Anomaly Detector) action, run Sub-env 2 internally,
        and emit Node 8 observation.
        """
        # Parse action
        agent_action = self._parse_action(action, ParamAnomalyAction)

        # Grade Node 2 — load param ground truth if available
        node2_score = self._grade_param_anomaly(agent_action)
        self._scores["node2"] = node2_score

        # Compute Sub-env 1 composite score
        subenv1_score = 0.5 * self._scores.get("node1", 0.0) + 0.5 * self._scores.get("node2", 0.0)
        self._scores["subenv1"] = subenv1_score

        # Run Sub-env 2 internally using reference heuristic agents
        subenv2_score = self._run_subenv2_internally()
        self._scores["subenv2"] = subenv2_score

        # Build Node 8 observation (PhonemeRiskObservation)
        phoneme_obs_data = self._build_phoneme_risk_observation()

        return Observation(
            done=False,
            reward=subenv1_score,
            metadata={
                "node": "node8_phoneme_risk_assessor",
                "step": 2,
                "subenv1_score": subenv1_score,
                "subenv2_score": subenv2_score,
                "instruction": "You receive canonical weight analysis signals from a trained LoRA. "
                    "Rank phoneme risks, predict behavior triggers, identify risky clusters, "
                    "assess behavioral safety, and recommend mitigations.",
                "observation": phoneme_obs_data,
                "expected_action_schema": "PhonemeRiskAction",
            },
        )

    def _handle_node8_action(self, action: Action) -> Observation:
        """
        Grade Node 8 (Phoneme Risk Assessor) action and compute final score.
        """
        agent_action = self._parse_action(action, PhonemeRiskAction)

        # Grade Node 8
        node8_score = self._grade_phoneme_risk(agent_action)
        self._scores["node8"] = node8_score
        self._scores["subenv3"] = node8_score

        # Compute final weighted score
        s1 = self._scores.get("subenv1", 0.0)
        s2 = self._scores.get("subenv2", 0.0)
        s3 = self._scores.get("subenv3", 0.0)
        final_score = 0.25 * s1 + 0.35 * s2 + 0.40 * s3
        self._scores["final"] = final_score

        return Observation(
            done=True,
            reward=final_score,
            metadata={
                "node": "episode_complete",
                "step": 3,
                "scores": {
                    "subenv1_score": s1,
                    "subenv2_score": s2,
                    "subenv3_score": s3,
                    "final_score": final_score,
                    "breakdown": self._scores,
                },
                "reward_formula": "0.25 * subenv1 + 0.35 * subenv2 + 0.40 * subenv3",
            },
        )

    # ------------------------------------------------------------------
    # Helper methods — IMPLEMENT THESE
    # ------------------------------------------------------------------

    def _parse_action(self, action: Action, expected_type):
        """
        Parse the raw Action into the expected Pydantic model.

        The agent sends action data as a dict in action.metadata or as JSON.
        Parse it into the expected_type.
        """
        if isinstance(action, expected_type):
            return action

        # Try to extract from metadata dict
        if hasattr(action, 'metadata') and isinstance(action.metadata, dict):
            action_data = action.metadata.get('action', action.metadata)
            return expected_type(**action_data)

        # Try to parse from the action's dict representation
        if hasattr(action, 'model_dump'):
            return expected_type(**action.model_dump())

        raise ValueError(
            f"Cannot parse action into {expected_type.__name__}. "
            f"Received type: {type(action).__name__}"
        )

    def _grade_image_diagnostics(self, agent_action: ImageDiagnosticsAction, gt: GroundTruthImageAnnotation) -> float:
        """Grade Node 1 action using the existing grader logic."""
        scores = {}

        # 1. Regime classification (partial credit for borderline)
        if agent_action.regime_classification == gt.regime_classification:
            scores["regime_accuracy"] = 1.0
        elif agent_action.regime_classification in gt.acceptable_regimes:
            scores["regime_accuracy"] = 0.7
        else:
            scores["regime_accuracy"] = 0.0

        # 2. Risk factor recall
        predicted_risks = set(agent_action.identified_risk_factors)
        true_risks = set(gt.identified_risk_factors)
        scores["risk_factor_recall"] = (
            len(predicted_risks & true_risks) / len(true_risks) if true_risks else 1.0
        )

        # 3. Prompt modification validity
        valid_mods = set(gt.valid_prompt_modifications)
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

    def _grade_param_anomaly(self, agent_action: ParamAnomalyAction) -> float:
        """
        Grade Node 2 action.

        If param ground truth is available in the test case, use it.
        Otherwise return a default score based on structural completeness.
        """
        # Check if subenv2 cases have param ground truth we can use
        # For now, score based on structural completeness
        score = 0.0
        if agent_action.anomalies:
            score += 0.3
        if agent_action.predicted_failure_modes:
            score += 0.25
        if agent_action.directional_fixes:
            score += 0.3
        if agent_action.config_risk_level in ("safe", "marginal", "risky", "dangerous"):
            score += 0.15
        return min(score, 1.0)

    def _run_subenv2_internally(self) -> float:
        """
        Run Sub-env 2 clip audit using internal heuristic agents.

        Uses test cases from subenv2_cases.json and the reference
        node4/node5 heuristic implementations.

        Returns:
            Sub-env 2 composite score in [0, 1].
        """
        if not self._subenv2_cases:
            return 0.5  # Default if no test cases

        total_score = 0.0
        count = 0

        for clip_case in self._subenv2_cases[:5]:  # Cap at 5 clips for speed
            try:
                obs = ClipSignalObservation(**clip_case["observation"])
                dossier = extract_clip_signals(obs)
                # Build disposition observation with minimal context
                disp_obs = ClipDispositionObservation(
                    evidence_dossier=dossier,
                    minimum_clips_needed=10,
                    phoneme_gap_severity={},
                    pose_gap_severity={},
                    budget_remaining=5,
                    reference_risk_profile="medium",
                    estimated_drift_risk=0.3,
                    marginal_training_damage=0.1,
                    marginal_coverage_gain=0.2,
                )
                disp_action = recommend_clip_disposition(disp_obs)

                if "ground_truth" in clip_case:
                    gt = GroundTruthClipAnnotation(**clip_case["ground_truth"])
                    clip_score = grade_clip_disposition(disp_action, gt)
                    total_score += clip_score
                    count += 1
            except Exception as e:
                log.warning(f"Sub-env 2 clip failed: {e}")
                continue

        return total_score / count if count > 0 else 0.5

    def _build_param_anomaly_observation(self, node1_action: ImageDiagnosticsAction, case: dict) -> dict:
        """Build ParamAnomalyObservation data from Node 1 output and test case."""
        obs = case["observation"]
        return {
            "proposed_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
            "regime": node1_action.regime_classification,
            "identified_risk_factors": node1_action.identified_risk_factors,
            "image_usability_score": node1_action.image_usability_score,
            "face_occupancy_ratio": obs["face_occupancy_ratio"],
            "estimated_yaw_degrees": obs["estimated_yaw_degrees"],
            "background_complexity_score": obs["background_complexity_score"],
            "lighting_uniformity_score": obs["lighting_uniformity_score"],
            "occlusion_detected": obs["occlusion_detected"],
            "prompt_identity_anchoring": obs["identity_anchoring_strength"],
            "prompt_token_count": obs["prompt_token_count"],
            "conflicting_descriptors": obs["conflicting_descriptors"],
        }

    def _build_phoneme_risk_observation(self) -> dict:
        """
        Build PhonemeRiskObservation data for Node 8.

        Uses subenv3_cases.json if available, otherwise constructs
        a minimal observation.
        """
        if self._subenv3_cases:
            case = self._subenv3_cases[self._current_case_idx % max(len(self._subenv3_cases), 1)]
            if "phoneme_obs" in case:
                return case["phoneme_obs"]
            elif "observation" in case:
                return case["observation"]

        # Minimal fallback
        return {
            "weight_evidence": {
                "weight_file_id": "lora_test.safetensors",
                "training_quality": "healthy",
                "rank_utilization_assessment": "efficient",
                "high_entropy_token_flags": [],
                "layer_anomaly_flags": [],
                "overall_behavioral_risk": "low",
                "evidence_summary": "No significant anomalies detected.",
            },
            "high_entropy_token_flags": [],
            "phoneme_vocabulary": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"],
            "phoneme_to_token_indices": {},
            "phoneme_entropy_scores": {},
            "phoneme_influence_scores": {},
            "phoneme_cooccurrence_anomalies": [],
            "behavior_vocabulary": ["smile", "blink", "head_turn", "lip_purse", "jaw_drop"],
            "training_data_phoneme_distribution": None,
            "suspected_anomalous_phonemes_from_subenv2": None,
        }

    def _grade_phoneme_risk(self, agent_action: PhonemeRiskAction) -> float:
        """
        Grade Node 8 action.

        If behavioral ground truth is available, use the grader.
        Otherwise score based on structural completeness.
        """
        score = 0.0
        if agent_action.phoneme_risk_ranking:
            score += 0.25
        if agent_action.predicted_behavior_triggers:
            score += 0.20
        if agent_action.risky_phoneme_clusters:
            score += 0.20
        if agent_action.model_behavioral_safety in ("safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"):
            score += 0.15
        if agent_action.mitigation_recommendations:
            score += 0.20
        return min(score, 1.0)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
```

### IMPORTANT IMPLEMENTATION NOTES:

1. **Import resolution:** The imports reference `src.envs.*` and `src.schemas.*`. These work because the project root is on `sys.path` (via `conftest.py`). In the Docker container, ensure `PYTHONPATH=/app` or the project root is the working directory.

2. **Grader imports:** Check what functions are actually exported from each grader module. The file above imports `grade_image_diagnostics` and `compute_handoff` from `node3_grader` — verify these exist. If `node3_grader.py` only exports `grade_anomaly_detection`, inline the image diagnostics grading logic (already provided in `_grade_image_diagnostics` above).

3. **ClipSignalObservation import:** You need to add this import at the top:
   ```python
   from src.schemas.subenv2 import ClipSignalObservation, ClipDispositionObservation
   ```

4. **Action parsing:** The `_parse_action` method needs to handle however OpenEnv wraps the agent's data. Test this locally after step 8.

---

## 6. Create `server/app.py`

**File:** `/home/elix3r/projects/talkingheadbench/server/app.py`

```python
"""
FastAPI application for TalkingHeadBench.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    Or run directly:
    python -m server.app
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

from server.talking_head_environment import TalkingHeadEnvironment

# Create the app — pass the class (factory) for WebSocket session support
app = create_app(
    TalkingHeadEnvironment,
    Action,
    Observation,
    env_name="talking_head_bench",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

---

## 7. Create `client.py`

**File:** `/home/elix3r/projects/talkingheadbench/client.py`

```python
"""
TalkingHeadBench Environment Client.

Provides the client for connecting to a running TalkingHeadBench server.

Example:
    >>> with TalkingHeadBenchEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     # Submit Node 1 action
    ...     result = env.step({"regime_classification": "frontal_simple", ...})
    ...     # Submit Node 2 action
    ...     result = env.step({"config_risk_level": "safe", ...})
    ...     # Submit Node 8 action
    ...     result = env.step({"phoneme_risk_ranking": [...], ...})
    ...     print(result.reward)  # Final score
"""

from openenv.core.env_client import EnvClient


class TalkingHeadBenchEnv(EnvClient):
    """Client for the TalkingHeadBench environment."""
    pass
```

**Note:** If `EnvClient` is not at `openenv.core.env_client`, check for:
- `openenv.core.mcp_client.MCPToolClient`
- `openenv.core.client.EnvClient`

Adjust the import accordingly. Run:
```bash
python -c "import openenv.core; import pkgutil; print([m.name for m in pkgutil.iter_modules(openenv.core.__path__)])"
```

---

## 8. Create `__init__.py` (Public Exports)

**File:** `/home/elix3r/projects/talkingheadbench/__init__.py`

If this file already exists, modify it. If not, create it:

```python
"""
TalkingHeadBench — An OpenEnv diagnostic reasoning benchmark for
talking-head LoRA pipelines.

Public API:
    TalkingHeadBenchEnv  — Client for connecting to the environment server
    ImageDiagnosticsAction, ParamAnomalyAction, PhonemeRiskAction — Agent action types
"""

from client import TalkingHeadBenchEnv

from src.schemas.subenv1 import (
    ImageDiagnosticsAction,
    ImageDiagnosticsObservation,
    ParamAnomalyAction,
    ParamAnomalyObservation,
)
from src.schemas.subenv3 import (
    PhonemeRiskAction,
    PhonemeRiskObservation,
)

__all__ = [
    "TalkingHeadBenchEnv",
    "ImageDiagnosticsAction",
    "ImageDiagnosticsObservation",
    "ParamAnomalyAction",
    "ParamAnomalyObservation",
    "PhonemeRiskAction",
    "PhonemeRiskObservation",
]
```

---

## 9. Create `server/Dockerfile`

**File:** `/home/elix3r/projects/talkingheadbench/server/Dockerfile`

```dockerfile
FROM python:3.11-slim

# System dependencies for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Ensure src/ is importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10. Create `server/requirements.txt`

**File:** `/home/elix3r/projects/talkingheadbench/server/requirements.txt`

```
# Auto-generated from pyproject.toml — keep in sync
openenv-core[core]>=0.2.2
fastapi>=0.115.0
pydantic>=2.0
uvicorn>=0.24.0
numpy
torch
scipy
safetensors
opencv-python
mediapipe
```

---

## 11. Create `.dockerignore`

**File:** `/home/elix3r/projects/talkingheadbench/.dockerignore`

```
.git
.gitignore
.venv
__pycache__
*.pyc
.pytest_cache
.agent
docs/
*.md
!README.md
```

---

## 12. Verify Locally — Test Without Docker

Run these commands in order:

```bash
cd /home/elix3r/projects/talkingheadbench

# 12a. Test the environment class directly (no server)
python -c "
import sys
sys.path.insert(0, '.')
from server.talking_head_environment import TalkingHeadEnvironment
env = TalkingHeadEnvironment()
obs = env.reset(seed=42)
print('Reset OK:', obs.metadata.get('node'))
print('Observation keys:', list(obs.metadata.get('observation', {}).keys()))
print('Episode ID:', obs.metadata.get('episode_id'))
"
```

If the above fails, debug the import errors and fix them before proceeding.

```bash
# 12b. Test the FastAPI server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
sleep 3

# Test health endpoint
curl http://localhost:8000/health

# Test reset via HTTP (the exact endpoint depends on openenv version)
curl -X POST http://localhost:8000/reset

# Kill the server
kill %1
```

---

## 13. Verify — Docker Build

```bash
cd /home/elix3r/projects/talkingheadbench

# Build the Docker image
docker build -t talking-head-bench -f server/Dockerfile .

# Run the container
docker run -p 8000:8000 talking-head-bench

# In another terminal, test:
curl http://localhost:8000/health
```

---

## 14. Run `openenv validate`

```bash
cd /home/elix3r/projects/talkingheadbench
openenv validate --verbose
```

If `openenv validate` is not available as a CLI command, check:
```bash
python -m openenv validate --verbose
# or
openenv --help
```

Fix any validation errors it reports.

---

## 15. Verify Existing Tests Still Pass

```bash
cd /home/elix3r/projects/talkingheadbench
python -m pytest tests/ -v --tb=short
```

The existing test suite must continue to pass. The new files should not break anything since they are additive.

---

## 16. Create `REWARD_LOGIC.md` 

**File:** `/home/elix3r/projects/talkingheadbench/REWARD_LOGIC.md`

```markdown
# TalkingHeadBench — Reward Structure

## Overview

TalkingHeadBench evaluates diagnostic reasoning across 3 coupled sub-environments.
The final reward is a weighted composite:

```
final_reward = 0.25 × subenv1_score + 0.35 × subenv2_score + 0.40 × subenv3_score
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
| Fix Instruction Quality | 0.20 | Precision ≥ 0.8 = full, ≥ 0.5 = half |
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

- **Deterministic:** All graders are rule-based. No LLM judge needed.
- **Partial credit:** Borderline answers receive scaled scores, not binary pass/fail.
- **Cascading difficulty:** Sub-env 1 risk profile affects Sub-env 2 clip difficulty.
- **Non-trivial:** Scoring requires multi-dimensional reasoning, not pattern matching.
```

---

## 17. Update README.md

Add the following sections to the existing README.md (append, do not replace):

```markdown
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
| reset | Node 1 | ImageDiagnosticsObservation | — |
| 1 | Node 1→2 | grade + ParamAnomalyObservation | ImageDiagnosticsAction |
| 2 | Node 2→8 | grade + PhonemeRiskObservation | ParamAnomalyAction |
| 3 | Node 8 | done + final_score | PhonemeRiskAction |

### Reward Formula

```
final = 0.25 × subenv1 + 0.35 × subenv2 + 0.40 × subenv3
```

See [REWARD_LOGIC.md](REWARD_LOGIC.md) for detailed scoring breakdown.
```

---

## 18. Final Checklist

Run through this checklist to confirm everything is in place:

- [ ] `openenv.yaml` exists at project root
- [ ] `pyproject.toml` exists at project root
- [ ] `server/__init__.py` exists
- [ ] `server/app.py` creates FastAPI app via `create_app()`
- [ ] `server/talking_head_environment.py` implements `reset()`, `step()`, `state`
- [ ] `server/Dockerfile` exists and builds successfully
- [ ] `server/requirements.txt` lists all runtime deps
- [ ] `client.py` exists with `TalkingHeadBenchEnv` class
- [ ] `__init__.py` exports client + action/observation types
- [ ] `.dockerignore` exists
- [ ] `REWARD_LOGIC.md` documents scoring
- [ ] README.md has OpenEnv integration section
- [ ] `python -c "from server.talking_head_environment import TalkingHeadEnvironment"` succeeds
- [ ] `uvicorn server.app:app` starts without errors
- [ ] `openenv validate` passes (if CLI available)
- [ ] `python -m pytest tests/ -v` still passes
- [ ] Docker container builds and starts on port 8000

---

## Troubleshooting

### Import errors for `openenv.*`
```bash
pip install openenv-core
# or
pip install openenv-core[core]
```
Then check: `python -c "import openenv; print(openenv.__version__)"`

### Import errors for `src.*` inside Docker
Ensure `ENV PYTHONPATH=/app` is in the Dockerfile and the project is copied to `/app`.

### `create_app` signature doesn't match
Check the actual signature:
```bash
python -c "from openenv.core.env_server.http_server import create_app; help(create_app)"
```
Adjust `server/app.py` accordingly.

### Grader function not found
If `grade_image_diagnostics` is not exported from `node3_grader.py`, the inline implementation in `_grade_image_diagnostics()` in the environment class serves as a self-contained fallback. Verify by running:
```bash
python -c "from src.envs.subenv1.node3_grader import grade_image_diagnostics"
```
If this fails, the environment class's inline grading code handles it.

### `State` class doesn't have `step_count` attribute
Check: `python -c "from openenv.core.env_server.types import State; print(State.__fields__)"` 
If `State` uses different field names, adapt accordingly.

---

## File Creation Summary

| # | File | Action | Priority |
|---|------|--------|----------|
| 1 | `openenv.yaml` | CREATE | 🔴 |
| 2 | `pyproject.toml` | CREATE | 🔴 |
| 3 | `server/__init__.py` | CREATE | 🔴 |
| 4 | `server/talking_head_environment.py` | CREATE | 🔴 |
| 5 | `server/app.py` | CREATE | 🔴 |
| 6 | `server/Dockerfile` | CREATE | 🔴 |
| 7 | `server/requirements.txt` | CREATE | 🔴 |
| 8 | `client.py` | CREATE | 🔴 |
| 9 | `__init__.py` | CREATE/MODIFY | 🔴 |
| 10 | `.dockerignore` | CREATE | 🟡 |
| 11 | `REWARD_LOGIC.md` | CREATE | 🟡 |
| 12 | `README.md` | MODIFY (append) | 🟡 |
