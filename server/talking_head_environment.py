"""TalkingHeadBench OpenEnv Environment.

Wraps the existing benchmark into a step-based environment with three
agent-facing episode types:

1) image audit: reset(image) -> node1 -> node2 -> done (reward=subenv1)
2) clip audit: reset(clips) -> node5 -> done (reward=subenv2)
3) weight audit: reset(weights) -> node8 -> done (reward=subenv3)
4) clips+weights: reset(clips_and_weights) -> node5 -> node8 -> done

The environment supports two modes:
- benchmark mode: pulls deterministic cases from tests/test_set
- custom mode: consumes user-ingested signal bundles (via ingestion_id)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, TypeVar
from uuid import uuid4

from models import TalkingHeadObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, State
from pydantic import ValidationError

from server.artifact_ingest import get_ingested_bundle
from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import (
    grade_anomaly_detection,
    grade_image_diagnostics,
    produce_reference_audit_handoff,
)
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.envs.subenv3.node9_grader import grade_behavioral_audit
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import (
    ImageDiagnosticsAction,
    ImageDiagnosticsObservation,
    ParamAnomalyAction,
    ParamAnomalyObservation,
)
from src.schemas.subenv2 import (
    ClipDispositionAction,
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
)
from src.schemas.subenv3 import (
    LayerAnomalyFlag,
    PhonemeRiskAction,
    PhonemeRiskObservation,
    TokenAnomalyFlag,
    WeightEvidenceDossier,
    WeightSignalObservation,
)

log = logging.getLogger(__name__)

_T = TypeVar("_T")
_TEST_SET_DIR = Path(__file__).resolve().parent.parent / "tests" / "test_set"
API_VERSION = "1.0"


class TalkingHeadEnvironment(Environment[Action, TalkingHeadObservation, State]):
    """OpenEnv adapter for TalkingHeadBench."""

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._subenv1_cases = self._load_cases("subenv1_cases.json")
        self._subenv2_cases = self._load_cases("subenv2_cases.json")
        self._subenv3_cases = self._load_cases("subenv3_cases.json")

        self._cursor = 0
        self._phase = "idle"
        self._scores: dict[str, float] = {}

        self._current_case: dict[str, Any] | None = None
        self._node1_obs: ImageDiagnosticsObservation | None = None
        self._node1_gt: GroundTruthImageAnnotation | None = None
        self._node1_action: ImageDiagnosticsAction | None = None

        self._node2_obs: ParamAnomalyObservation | None = None
        self._node2_gt: GroundTruthParamAnnotation | None = None

        self._clip_obs: ClipSignalObservation | None = None
        self._clip_gt: GroundTruthClipAnnotation | None = None

        self._subenv3_case: dict[str, Any] | None = None
        self._subenv3_gt: GroundTruthBehavioralAnnotation | None = None

        self._custom_clip_obs: list[ClipSignalObservation] | None = None
        self._custom_weight_obs: WeightSignalObservation | None = None
        self._proposed_config: dict[str, float] = self._default_param_config()
        self._mode: str = "benchmark"
        self._episode_mode: str = "image"
        self._provenance: dict[str, Any] = {}

    def _load_cases(self, filename: str) -> list[dict[str, Any]]:
        path = _TEST_SET_DIR / filename
        if not path.exists():
            log.warning("Missing test case file: %s", path)
            return []
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cases = payload.get("cases", [])
        return cases if isinstance(cases, list) else []

    def _default_param_config(self) -> dict[str, float]:
        return {"cfg": 7.5, "denoise_alt": 0.30, "eta": 0.10}

    def _reset_runtime_state(self, episode_id: Optional[str]) -> None:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._scores = {}
        self._phase = "idle"
        self._mode = "benchmark"
        self._episode_mode = "image"
        self._provenance = {
            "bundle_source": "benchmark_test_set",
            "ground_truth_source": "annotations",
        }

        self._node1_action = None
        self._node2_obs = None
        self._node2_gt = None
        self._clip_obs = None
        self._clip_gt = None
        self._subenv3_case = None
        self._subenv3_gt = None

        self._custom_clip_obs = None
        self._custom_weight_obs = None
        self._proposed_config = self._default_param_config()

    def _current_case_id(self) -> str:
        if self._current_case and "id" in self._current_case:
            return str(self._current_case.get("id"))
        return "unknown"

    def _build_agent_observation(
        self,
        *,
        done: bool,
        reward: float,
        node: str,
        step: int,
        instruction: Optional[str],
        expected_action_schema: Optional[str],
        signals: dict[str, Any],
        scores: Optional[dict[str, Any]] = None,
        reward_formula: Optional[str] = None,
        error: Optional[str] = None,
    ) -> TalkingHeadObservation:
        return TalkingHeadObservation(
            done=done,
            reward=reward,
            node=node,
            step=step,
            api_version=API_VERSION,
            mode="custom" if self._mode == "custom" else "benchmark",
            is_deterministic=True,
            case_id=self._current_case_id(),
            episode_id=self._state.episode_id,
            instruction=instruction,
            expected_action_schema=expected_action_schema,
            signals=signals,
            provenance=dict(self._provenance),
            scores=scores,
            reward_formula=reward_formula,
            error=error,
        )

    def _normalize_episode_mode(self, mode: Optional[str]) -> str:
        value = (mode or "image").strip().lower()
        valid = {"image", "clips", "weights", "clips_and_weights"}
        if value not in valid:
            raise RuntimeError(
                "mode must be one of: image, clips, weights, clips_and_weights"
            )
        return value

    def _next_index(self, total: int, seed: Optional[int]) -> int:
        if total <= 0:
            raise RuntimeError("Cannot sample from an empty case list")
        if seed is not None:
            return random.Random(seed).randrange(total)
        idx = self._cursor % total
        self._cursor += 1
        return idx

    def _infer_mode_from_custom_bundle(self) -> str:
        has_image = self._node1_obs is not None
        has_clips = bool(self._custom_clip_obs)
        has_weight = self._custom_weight_obs is not None or self._subenv3_case is not None

        if has_clips and has_weight:
            return "clips_and_weights"
        if has_clips:
            return "clips"
        if has_weight:
            return "weights"
        if has_image:
            return "image"
        raise RuntimeError("Custom bundle does not include image, clips, or weight signals.")

    def _build_clip_disposition_observation(
        self,
        clip_obs: ClipSignalObservation,
    ) -> ClipDispositionObservation:
        dossier = self._build_clip_evidence_dossier(clip_obs)
        damage_map = {"none": 0.0, "minor": 0.05, "moderate": 0.20, "severe": 0.50}
        marginal_damage = damage_map.get(dossier.identity_drift_severity, 0.1)

        return ClipDispositionObservation(
            evidence_dossier=dossier,
            minimum_clips_needed=20,
            phoneme_gap_severity={},
            pose_gap_severity={},
            budget_remaining=max(0, 50 - clip_obs.clips_audited_so_far),
            marginal_training_damage=marginal_damage,
            marginal_coverage_gain=float(clip_obs.phoneme_coverage_new),
        )

    def _start_image_episode(self, seed: Optional[int]) -> TalkingHeadObservation:
        if self._mode == "custom":
            if self._node1_obs is None:
                raise RuntimeError("Custom bundle does not include image_observation.")
            self._phase = "node1"
            return self._build_agent_observation(
                done=False,
                reward=0.0,
                node="node1_image_diagnostician",
                step=0,
                instruction=(
                    "Diagnose reference-image quality and prompt risks from the provided "
                    "signals."
                ),
                expected_action_schema="ImageDiagnosticsAction",
                signals=self._node1_obs.model_dump(mode="json"),
            )

        if not self._subenv1_cases:
            raise RuntimeError("subenv1_cases.json has no cases; cannot reset environment")

        idx = self._next_index(len(self._subenv1_cases), seed)
        self._current_case = self._subenv1_cases[idx]
        self._node1_obs = ImageDiagnosticsObservation.model_validate(
            self._current_case["observation"]
        )
        self._node1_gt = GroundTruthImageAnnotation.model_validate(
            self._current_case["ground_truth"]
        )
        self._phase = "node1"

        return self._build_agent_observation(
            done=False,
            reward=0.0,
            node="node1_image_diagnostician",
            step=0,
            instruction=(
                "Diagnose reference-image quality and prompt risks from the provided "
                "signals."
            ),
            expected_action_schema="ImageDiagnosticsAction",
            signals=self._node1_obs.model_dump(mode="json"),
        )

    def _start_clips_episode(self, seed: Optional[int]) -> TalkingHeadObservation:
        if self._mode == "custom":
            if not self._custom_clip_obs:
                raise RuntimeError("Custom bundle does not include clip_signal_observations.")
            self._clip_obs = self._custom_clip_obs[0]
            clip_obs_payload = self._build_clip_disposition_observation(self._clip_obs)
            baseline_action = recommend_clip_disposition(clip_obs_payload)
            self._clip_gt = self._synthesize_clip_ground_truth(baseline_action)
        else:
            if not self._subenv2_cases:
                raise RuntimeError("subenv2_cases.json has no cases; cannot reset environment")
            idx = self._next_index(len(self._subenv2_cases), seed)
            clip_case = self._subenv2_cases[idx]
            self._current_case = {"id": clip_case.get("id", f"subenv2-{idx}")}
            self._clip_obs = ClipSignalObservation.model_validate(clip_case["observation"])
            self._clip_gt = GroundTruthClipAnnotation.model_validate(clip_case["ground_truth"])
            clip_obs_payload = self._build_clip_disposition_observation(self._clip_obs)

        self._phase = "clips"
        return self._build_agent_observation(
            done=False,
            reward=0.0,
            node="node5_clip_disposition_recommender",
            step=0,
            instruction=(
                "Recommend clip disposition (accept/reject/fix/defer) from the dossier "
                "and dataset context."
            ),
            expected_action_schema="ClipDispositionAction",
            signals=clip_obs_payload.model_dump(mode="json"),
        )

    def _start_weights_episode(self, seed: Optional[int]) -> TalkingHeadObservation:
        if self._mode == "custom":
            if self._custom_weight_obs is None and self._subenv3_case is None:
                raise RuntimeError("Custom bundle does not include weight_observation.")
            phoneme_obs = self._build_phoneme_risk_observation()
            if self._subenv3_gt is None:
                baseline = assess_phoneme_risk(phoneme_obs)
                self._subenv3_gt = self._synthesize_behavioral_ground_truth(baseline)
        else:
            if not self._subenv3_cases:
                raise RuntimeError("subenv3_cases.json has no cases; cannot reset environment")
            idx = self._next_index(len(self._subenv3_cases), seed)
            self._subenv3_case = self._pick_subenv3_case(idx)
            self._subenv3_gt = self._resolve_behavioral_ground_truth(self._subenv3_case)
            self._current_case = self._subenv3_case
            phoneme_obs = self._build_phoneme_risk_observation()

        self._phase = "node8"
        return self._build_agent_observation(
            done=False,
            reward=0.0,
            node="node8_phoneme_risk_assessor",
            step=0,
            instruction=(
                "Assess phoneme-level behavioral risk from weight evidence and provide "
                "mitigation guidance."
            ),
            expected_action_schema="PhonemeRiskAction",
            signals=phoneme_obs.model_dump(mode="json"),
        )

    def _start_clips_and_weights_episode(self, seed: Optional[int]) -> TalkingHeadObservation:
        if self._mode == "custom":
            if not self._custom_clip_obs:
                raise RuntimeError("Custom bundle does not include clip_signal_observations.")
            if self._custom_weight_obs is None and self._subenv3_case is None:
                raise RuntimeError("Custom bundle does not include weight_observation.")
            self._clip_obs = self._custom_clip_obs[0]
            clip_obs_payload = self._build_clip_disposition_observation(self._clip_obs)
            baseline_action = recommend_clip_disposition(clip_obs_payload)
            self._clip_gt = self._synthesize_clip_ground_truth(baseline_action)
        else:
            if not self._subenv2_cases:
                raise RuntimeError("subenv2_cases.json has no cases; cannot reset environment")
            if not self._subenv3_cases:
                raise RuntimeError("subenv3_cases.json has no cases; cannot reset environment")
            idx = self._next_index(len(self._subenv2_cases), seed)
            clip_case = self._subenv2_cases[idx]
            self._subenv3_case = self._pick_subenv3_case(idx)
            self._subenv3_gt = self._resolve_behavioral_ground_truth(self._subenv3_case)
            self._current_case = {"id": clip_case.get("id", f"combo-{idx}")}
            self._clip_obs = ClipSignalObservation.model_validate(clip_case["observation"])
            self._clip_gt = GroundTruthClipAnnotation.model_validate(clip_case["ground_truth"])
            clip_obs_payload = self._build_clip_disposition_observation(self._clip_obs)

        self._phase = "clips"
        return self._build_agent_observation(
            done=False,
            reward=0.0,
            node="node5_clip_disposition_recommender",
            step=0,
            instruction=(
                "Recommend clip disposition (accept/reject/fix/defer) from the dossier "
                "and dataset context."
            ),
            expected_action_schema="ClipDispositionAction",
            signals=clip_obs_payload.model_dump(mode="json"),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> TalkingHeadObservation:
        self._reset_runtime_state(episode_id)

        requested_mode = mode if mode is not None else kwargs.get("mode")
        custom_bundle = kwargs.get("custom_bundle")
        ingestion_id = kwargs.get("ingestion_id")

        if custom_bundle is None and ingestion_id is not None:
            custom_bundle = get_ingested_bundle(str(ingestion_id))
            if custom_bundle is None:
                raise RuntimeError(f"Unknown ingestion_id: {ingestion_id}")

        if isinstance(custom_bundle, str):
            custom_bundle = json.loads(custom_bundle)

        if custom_bundle is not None:
            if not isinstance(custom_bundle, dict):
                raise RuntimeError("custom_bundle must be an object")
            self._mode = "custom"
            self._provenance = {
                "bundle_source": "ingestion_id" if ingestion_id is not None else "custom_bundle",
                "ground_truth_source": "synthesized_from_heuristics",
            }
            if ingestion_id is not None:
                self._provenance["ingestion_id"] = str(ingestion_id)
            self._initialize_custom_bundle(custom_bundle, ingestion_id)
            inferred_mode = self._infer_mode_from_custom_bundle()
            self._episode_mode = (
                self._normalize_episode_mode(requested_mode)
                if requested_mode is not None
                else inferred_mode
            )
        else:
            self._mode = "benchmark"
            self._provenance = {
                "bundle_source": "benchmark_test_set",
                "ground_truth_source": "annotations",
                "test_set_dir": str(_TEST_SET_DIR),
            }
            self._episode_mode = self._normalize_episode_mode(requested_mode)

        if self._episode_mode == "image":
            return self._start_image_episode(seed)
        if self._episode_mode == "clips":
            return self._start_clips_episode(seed)
        if self._episode_mode == "weights":
            return self._start_weights_episode(seed)
        return self._start_clips_and_weights_episode(seed)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TalkingHeadObservation:
        self._state.step_count += 1

        if self._phase == "node1":
            return self._handle_node1_action(action)
        if self._phase == "node2":
            return self._handle_node2_action(action)
        if self._phase == "clips":
            return self._handle_clips_action(action)
        if self._phase == "node8":
            return self._handle_node8_action(action)

        return self._build_agent_observation(
            done=True,
            reward=0.0,
            node="episode_complete",
            step=3,
            instruction=None,
            expected_action_schema=None,
            signals={},
            error="Episode complete. Call reset() for a new episode.",
        )

    def _handle_node1_action(self, action: Action) -> TalkingHeadObservation:
        if self._current_case is None or self._node1_obs is None or self._node1_gt is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        node1_action = self._parse_action(action, ImageDiagnosticsAction)
        self._node1_action = node1_action

        node1_score = grade_image_diagnostics(node1_action, self._node1_gt)
        self._scores["node1"] = node1_score

        self._node2_obs = self._build_param_anomaly_observation(node1_action, self._node1_obs)
        self._node2_gt = self._resolve_param_ground_truth(self._current_case, self._node2_obs)
        self._phase = "node2"

        return self._build_agent_observation(
            done=False,
            reward=node1_score,
            node="node2_param_anomaly_detector",
            step=1,
            instruction=(
                "Analyze proposed generation parameters, detect anomalies, and propose "
                "directional fixes."
            ),
            expected_action_schema="ParamAnomalyAction",
            signals=self._node2_obs.model_dump(mode="json"),
        )

    def _handle_node2_action(self, action: Action) -> TalkingHeadObservation:
        if (
            self._node1_action is None
            or self._node1_gt is None
            or self._node2_obs is None
            or self._node2_gt is None
        ):
            raise RuntimeError("Node 2 prerequisites missing. Run Node 1 step first.")

        node2_action = self._parse_action(action, ParamAnomalyAction)
        node2_score = grade_anomaly_detection(node2_action, self._node2_gt)
        self._scores["node2"] = node2_score

        subenv1_score = 0.50 * self._scores.get("node1", 0.0) + 0.50 * node2_score
        self._scores["subenv1"] = subenv1_score

        # Maintain compatibility metadata for Sub-env 1 reporting only.
        _ = produce_reference_audit_handoff(
            self._node1_action,
            node2_action,
            self._node1_gt,
            self._node2_gt,
        )

        self._phase = "done"

        return self._build_agent_observation(
            done=True,
            reward=subenv1_score,
            node="episode_complete",
            step=2,
            instruction=None,
            expected_action_schema=None,
            signals={},
            reward_formula="subenv1_score",
            scores={
                "subenv1_score": subenv1_score,
                "breakdown": self._scores,
            },
        )

    def _handle_clips_action(self, action: Action) -> TalkingHeadObservation:
        if self._clip_obs is None or self._clip_gt is None:
            raise RuntimeError("Clip phase prerequisites missing. Call reset() first.")

        clip_action = self._parse_action(action, ClipDispositionAction)
        clip_score = grade_clip_disposition(clip_action, self._clip_gt)
        self._scores["node5"] = clip_score
        self._scores["subenv2"] = clip_score

        if self._episode_mode == "clips":
            self._phase = "done"
            return self._build_agent_observation(
                done=True,
                reward=clip_score,
                node="episode_complete",
                step=1,
                instruction=None,
                expected_action_schema=None,
                signals={},
                reward_formula="subenv2_score",
                scores={
                    "subenv2_score": clip_score,
                    "breakdown": self._scores,
                },
            )

        phoneme_obs = self._build_phoneme_risk_observation()
        if self._subenv3_gt is None:
            baseline = assess_phoneme_risk(phoneme_obs)
            self._subenv3_gt = self._synthesize_behavioral_ground_truth(baseline)

        self._phase = "node8"
        return self._build_agent_observation(
            done=False,
            reward=clip_score,
            node="node8_phoneme_risk_assessor",
            step=1,
            instruction=(
                "Assess phoneme-level behavioral risk from weight evidence and provide "
                "mitigation guidance."
            ),
            expected_action_schema="PhonemeRiskAction",
            signals=phoneme_obs.model_dump(mode="json"),
        )

    def _handle_node8_action(self, action: Action) -> TalkingHeadObservation:
        if self._subenv3_gt is None:
            raise RuntimeError("Behavioral ground truth is unavailable. Reset the environment.")

        node8_action = self._parse_action(action, PhonemeRiskAction)
        node8_score = grade_behavioral_audit(node8_action, self._subenv3_gt)

        self._scores["node8"] = node8_score
        self._scores["subenv3"] = node8_score

        s2 = self._scores.get("subenv2", 0.0)
        s3 = self._scores.get("subenv3", 0.0)
        if self._episode_mode == "clips_and_weights":
            final_score = 0.50 * s2 + 0.50 * s3
            reward_formula = "0.50 * subenv2 + 0.50 * subenv3"
            scores = {
                "subenv2_score": s2,
                "subenv3_score": s3,
                "final_score": final_score,
                "breakdown": self._scores,
            }
        else:
            final_score = s3
            reward_formula = "subenv3_score"
            scores = {
                "subenv3_score": s3,
                "breakdown": self._scores,
            }

        self._scores["final"] = final_score

        self._phase = "done"

        return self._build_agent_observation(
            done=True,
            reward=final_score,
            node="episode_complete",
            step=3,
            instruction=None,
            expected_action_schema=None,
            signals={},
            reward_formula=reward_formula,
            scores=scores,
        )

    def _parse_action(self, action: Action, expected_type: type[_T]) -> _T:
        if isinstance(action, expected_type):
            return action

        payload: Any = None

        if isinstance(action, dict):
            payload = action.get("action", action)
        elif isinstance(action, Action):
            payload = action.metadata.get("action", action.metadata)
        elif hasattr(action, "metadata") and isinstance(action.metadata, dict):
            payload = action.metadata.get("action", action.metadata)
        elif hasattr(action, "model_dump"):
            payload = action.model_dump()

        if isinstance(payload, str):
            payload = json.loads(payload)

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unable to parse action payload for {expected_type.__name__}; "
                f"received {type(action).__name__}"
            )

        return expected_type.model_validate(payload)

    def _initialize_custom_bundle(
        self,
        custom_bundle: dict[str, Any],
        ingestion_id: Optional[str],
    ) -> None:
        image_payload = (
            custom_bundle.get("image_observation")
            or custom_bundle.get("image_obs")
            or custom_bundle.get("reference_image_obs")
        )
        if isinstance(image_payload, dict):
            self._node1_obs = ImageDiagnosticsObservation.model_validate(image_payload)
            self._node1_gt = self._synthesize_image_ground_truth(self._node1_obs)
        else:
            self._node1_obs = None
            self._node1_gt = None

        config_payload = custom_bundle.get("param_config") or custom_bundle.get("proposed_config")
        if isinstance(config_payload, dict):
            for key, value in config_payload.items():
                if isinstance(value, (int, float)):
                    self._proposed_config[key] = float(value)

        raw_clips = (
            custom_bundle.get("clip_signal_observations")
            or custom_bundle.get("clip_signal_obs_list")
            or []
        )
        if isinstance(raw_clips, list):
            self._custom_clip_obs = [
                ClipSignalObservation.model_validate(item)
                for item in raw_clips
                if isinstance(item, dict)
            ]
        else:
            self._custom_clip_obs = []

        custom_phoneme_case: dict[str, Any] | None = None
        raw_weight = custom_bundle.get("weight_observation") or custom_bundle.get("weight_obs")
        if isinstance(raw_weight, dict):
            try:
                self._custom_weight_obs = WeightSignalObservation.model_validate(raw_weight)
            except ValidationError:
                try:
                    phoneme_obs = PhonemeRiskObservation.model_validate(raw_weight)
                    custom_phoneme_case = {"observation": phoneme_obs.model_dump(mode="json")}
                except ValidationError as exc:
                    raise RuntimeError(
                        "weight_observation must be a WeightSignalObservation or "
                        "PhonemeRiskObservation payload"
                    ) from exc

        case_id = (
            custom_bundle.get("case_id")
            or custom_bundle.get("id")
            or ingestion_id
            or f"custom-{uuid4().hex[:8]}"
        )
        self._current_case = {"id": str(case_id)}

        metadata = custom_bundle.get("ingestion_metadata")
        if isinstance(metadata, dict):
            self._provenance["ingestion_metadata"] = metadata

        source_files = custom_bundle.get("source_files")
        if isinstance(source_files, dict):
            self._provenance["source_files"] = source_files

        self._subenv3_case = custom_phoneme_case
        self._subenv3_gt = None

    def _synthesize_image_ground_truth(
        self,
        image_obs: ImageDiagnosticsObservation,
    ) -> GroundTruthImageAnnotation:
        baseline = diagnose_image(image_obs)
        return GroundTruthImageAnnotation(
            regime_classification=baseline.regime_classification,
            acceptable_regimes=[baseline.regime_classification],
            identified_risk_factors=baseline.identified_risk_factors,
            valid_prompt_modifications=baseline.recommended_prompt_modifications,
        )

    def _pick_subenv3_case(self, idx: int) -> dict[str, Any] | None:
        if not self._subenv3_cases:
            return None
        return self._subenv3_cases[idx % len(self._subenv3_cases)]

    def _resolve_behavioral_ground_truth(
        self,
        case: dict[str, Any] | None,
    ) -> GroundTruthBehavioralAnnotation:
        if case and isinstance(case.get("ground_truth"), dict):
            try:
                return GroundTruthBehavioralAnnotation.model_validate(case["ground_truth"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to parse behavioral ground truth from case: %s", exc)

        return GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=[],
            predicted_behavior_triggers=[],
            risky_phoneme_clusters=[],
            model_behavioral_safety="safe",
            valid_mitigation_set=set(),
        )

    def _build_param_anomaly_observation(
        self,
        node1_action: ImageDiagnosticsAction,
        image_obs: ImageDiagnosticsObservation,
    ) -> ParamAnomalyObservation:
        return ParamAnomalyObservation(
            proposed_config=self._proposed_config,
            regime=node1_action.regime_classification,
            identified_risk_factors=node1_action.identified_risk_factors,
            image_usability_score=node1_action.image_usability_score,
            face_occupancy_ratio=image_obs.face_occupancy_ratio,
            estimated_yaw_degrees=image_obs.estimated_yaw_degrees,
            background_complexity_score=image_obs.background_complexity_score,
            lighting_uniformity_score=image_obs.lighting_uniformity_score,
            occlusion_detected=image_obs.occlusion_detected,
            prompt_identity_anchoring=image_obs.identity_anchoring_strength,
            prompt_token_count=image_obs.prompt_token_count,
            conflicting_descriptors=image_obs.conflicting_descriptors,
        )

    def _resolve_param_ground_truth(
        self,
        case: dict[str, Any],
        node2_obs: ParamAnomalyObservation,
    ) -> GroundTruthParamAnnotation:
        for key in ("param_ground_truth", "ground_truth_param", "node2_ground_truth"):
            if isinstance(case.get(key), dict):
                return GroundTruthParamAnnotation.model_validate(case[key])

        baseline = detect_param_anomalies(node2_obs)
        return GroundTruthParamAnnotation(
            config_risk_level=baseline.config_risk_level,
            anomalies=baseline.anomalies,
            predicted_failure_modes=list(baseline.predicted_failure_modes),
            valid_fix_directions=baseline.directional_fixes,
        )

    def _build_clip_evidence_dossier(self, obs: ClipSignalObservation) -> ClipEvidenceDossier:
        drift = obs.identity_cosine_drift
        if drift < 0.05:
            drift_severity = "none"
        elif drift < 0.12:
            drift_severity = "minor"
        elif drift < 0.22:
            drift_severity = "moderate"
        else:
            drift_severity = "severe"

        temporal_instability = (
            obs.landmark_stability_score > 0.04 or obs.frame_difference_mean > 15.0
        )

        lip_sync = obs.lip_sync_confidence
        if lip_sync >= 0.75:
            lip_sync_quality = "good"
        elif lip_sync >= 0.50:
            lip_sync_quality = "acceptable"
        elif lip_sync >= 0.20:
            lip_sync_quality = "poor"
        else:
            lip_sync_quality = "absent"

        dataset_redundancy = min(1.0, obs.similar_clips_accepted / 10.0)

        if drift_severity in ("none", "minor") and not temporal_instability and lip_sync >= 0.50:
            training_impact = "positive"
        elif drift_severity == "severe" or (temporal_instability and lip_sync < 0.20):
            training_impact = "negative"
        else:
            training_impact = "neutral"

        primary_rejection_reason = None
        if drift_severity in ("moderate", "severe"):
            primary_rejection_reason = (
                f"identity cosine drift {drift:.3f} exceeds threshold; severity={drift_severity}"
            )
        elif temporal_instability and lip_sync < 0.20:
            primary_rejection_reason = "temporal instability with absent lip sync"

        summary = (
            f"Drift={drift_severity}, temporal_instability={temporal_instability}, "
            f"lip_sync={lip_sync_quality}, phoneme_value={obs.phoneme_coverage_new:.2f}, "
            f"redundancy={dataset_redundancy:.2f}."
        )

        return ClipEvidenceDossier(
            clip_id=obs.clip_id,
            identity_drift_severity=drift_severity,
            temporal_instability_flag=temporal_instability,
            lip_sync_quality=lip_sync_quality,
            unique_phoneme_value=obs.phoneme_coverage_new,
            dataset_redundancy_score=dataset_redundancy,
            estimated_training_impact=training_impact,
            primary_rejection_reason=primary_rejection_reason,
            evidence_summary=summary,
        )

    def _synthesize_clip_ground_truth(
        self,
        action,
    ) -> GroundTruthClipAnnotation:
        valid_overrides = [action.override_justification] if action.override_justification else []
        return GroundTruthClipAnnotation(
            disposition=action.disposition,
            confidence=float(action.confidence),
            disposition_ambiguity=0.6 if action.disposition == "defer" else 0.0,
            valid_fix_steps=list(action.fix_instructions or []),
            valid_override_justifications=valid_overrides,
            expected_reasoning_elements=[],
        )

    def _run_subenv2_from_observations(
        self,
        clip_observations: list[ClipSignalObservation],
    ) -> float:
        if not clip_observations:
            return 0.5

        scores: list[float] = []

        for clip_obs in clip_observations:
            try:
                dossier = self._build_clip_evidence_dossier(clip_obs)
                damage_map = {"none": 0.0, "minor": 0.05, "moderate": 0.20, "severe": 0.50}
                marginal_damage = damage_map.get(dossier.identity_drift_severity, 0.1)

                clip_disp_obs = ClipDispositionObservation(
                    evidence_dossier=dossier,
                    minimum_clips_needed=20,
                    phoneme_gap_severity={},
                    pose_gap_severity={},
                    budget_remaining=max(0, 50 - clip_obs.clips_audited_so_far),
                    marginal_training_damage=marginal_damage,
                    marginal_coverage_gain=float(clip_obs.phoneme_coverage_new),
                )

                clip_action = recommend_clip_disposition(clip_disp_obs)
                clip_gt = self._synthesize_clip_ground_truth(clip_action)
                scores.append(grade_clip_disposition(clip_action, clip_gt))
            except Exception as exc:  # noqa: BLE001
                log.warning("Sub-env 2 custom clip evaluation failed: %s", exc)

        return sum(scores) / len(scores) if scores else 0.5

    def _run_subenv2_internally(self) -> float:
        if self._custom_clip_obs is not None:
            return self._run_subenv2_from_observations(self._custom_clip_obs)

        if not self._subenv2_cases:
            return 0.5

        scores: list[float] = []

        for clip_case in self._subenv2_cases[:5]:
            try:
                clip_obs = ClipSignalObservation.model_validate(clip_case["observation"])
                clip_gt = GroundTruthClipAnnotation.model_validate(clip_case["ground_truth"])

                dossier = self._build_clip_evidence_dossier(clip_obs)
                damage_map = {"none": 0.0, "minor": 0.05, "moderate": 0.20, "severe": 0.50}
                marginal_damage = damage_map.get(dossier.identity_drift_severity, 0.1)

                clip_disp_obs = ClipDispositionObservation(
                    evidence_dossier=dossier,
                    minimum_clips_needed=20,
                    phoneme_gap_severity={},
                    pose_gap_severity={},
                    budget_remaining=max(0, 50 - clip_obs.clips_audited_so_far),
                    marginal_training_damage=marginal_damage,
                    marginal_coverage_gain=float(clip_obs.phoneme_coverage_new),
                )

                clip_action = recommend_clip_disposition(clip_disp_obs)
                scores.append(grade_clip_disposition(clip_action, clip_gt))
            except Exception as exc:  # noqa: BLE001
                log.warning("Sub-env 2 internal clip evaluation failed: %s", exc)

        return sum(scores) / len(scores) if scores else 0.5

    def _build_weight_evidence_from_observation(
        self,
        weight_obs: WeightSignalObservation,
    ) -> WeightEvidenceDossier:
        rank_values = list(weight_obs.layer_rank_utilization.values())
        mean_rank_util = sum(rank_values) / max(1, len(rank_values))

        if weight_obs.overfitting_signature >= 0.6:
            training_quality = "overfit"
        elif weight_obs.gradient_noise_estimate >= 0.5:
            training_quality = "unstable"
        elif mean_rank_util <= 0.3:
            training_quality = "underfit"
        else:
            training_quality = "healthy"

        if mean_rank_util >= 0.75:
            rank_assessment = "efficient"
        elif mean_rank_util >= 0.45:
            rank_assessment = "wasteful"
        else:
            rank_assessment = "collapsed"

        token_map = weight_obs.token_position_to_phoneme or {}
        high_entropy_flags: list[TokenAnomalyFlag] = []
        for token_pos in weight_obs.high_entropy_token_positions[:64]:
            severity = min(1.0, 0.65 + 0.05 * (token_pos % 5))
            high_entropy_flags.append(
                TokenAnomalyFlag(
                    token_position=int(token_pos),
                    mapped_phoneme=token_map.get(int(token_pos)),
                    anomaly_type="excessive_influence",
                    severity=severity,
                    evidence="High canonical directional anomaly score for token position.",
                )
            )

        layer_flags: list[LayerAnomalyFlag] = []
        for layer_name, util in weight_obs.layer_rank_utilization.items():
            sparsity = float(weight_obs.layer_sparsity.get(layer_name, 0.0))
            layer_norm = float(weight_obs.layer_norms.get(layer_name, 0.0))

            if util < 0.35:
                layer_flags.append(
                    LayerAnomalyFlag(
                        layer_name=layer_name,
                        anomaly_type="rank_collapse",
                        severity=min(1.0, float(1.0 - util)),
                        evidence=f"Rank utilization dropped to {util:.3f}.",
                    )
                )
            if sparsity > 0.85:
                layer_flags.append(
                    LayerAnomalyFlag(
                        layer_name=layer_name,
                        anomaly_type="sparsity_anomaly",
                        severity=min(1.0, sparsity),
                        evidence=f"Layer sparsity elevated at {sparsity:.3f}.",
                    )
                )
            if layer_norm > 8.0:
                layer_flags.append(
                    LayerAnomalyFlag(
                        layer_name=layer_name,
                        anomaly_type="norm_explosion",
                        severity=min(1.0, layer_norm / 12.0),
                        evidence=f"Layer norm elevated at {layer_norm:.3f}.",
                    )
                )

        global_risk = max(
            float(weight_obs.overfitting_signature),
            float(weight_obs.gradient_noise_estimate),
            max((flag.severity for flag in high_entropy_flags), default=0.0),
            max((flag.severity for flag in layer_flags), default=0.0),
        )

        if global_risk < 0.30:
            overall_behavioral_risk = "low"
        elif global_risk < 0.55:
            overall_behavioral_risk = "medium"
        elif global_risk < 0.80:
            overall_behavioral_risk = "high"
        else:
            overall_behavioral_risk = "critical"

        evidence_summary = (
            f"training_quality={training_quality}, mean_rank_util={mean_rank_util:.3f}, "
            f"noise={weight_obs.gradient_noise_estimate:.3f}, overfit={weight_obs.overfitting_signature:.3f}."
        )

        return WeightEvidenceDossier(
            weight_file_id=weight_obs.weight_file_id,
            training_quality=training_quality,
            rank_utilization_assessment=rank_assessment,
            high_entropy_token_flags=high_entropy_flags,
            layer_anomaly_flags=layer_flags,
            overall_behavioral_risk=overall_behavioral_risk,
            evidence_summary=evidence_summary,
        )

    def _build_phoneme_risk_observation_from_weight(
        self,
        weight_obs: WeightSignalObservation,
    ) -> PhonemeRiskObservation:
        weight_evidence = self._build_weight_evidence_from_observation(weight_obs)

        token_map = weight_obs.token_position_to_phoneme or {}
        phoneme_to_token_indices: dict[str, list[int]] = {}
        for token_idx, phoneme in token_map.items():
            phoneme_to_token_indices.setdefault(str(phoneme), []).append(int(token_idx))

        phoneme_vocabulary = sorted(phoneme_to_token_indices.keys())
        if not phoneme_vocabulary:
            phoneme_vocabulary = [
                "AA",
                "AE",
                "AH",
                "AO",
                "AW",
                "AY",
                "EH",
                "ER",
                "EY",
                "IH",
                "IY",
                "OW",
                "OY",
                "UH",
                "UW",
            ]
            phoneme_to_token_indices = {ph: [] for ph in phoneme_vocabulary}

        high_entropy_set = set(weight_obs.high_entropy_token_positions)
        phoneme_entropy_scores: dict[str, float] = {}
        phoneme_influence_scores: dict[str, float] = {}

        for phoneme in phoneme_vocabulary:
            indices = phoneme_to_token_indices.get(phoneme, [])
            flagged = sum(1 for idx in indices if idx in high_entropy_set)
            ratio = flagged / max(1, len(indices))

            entropy_score = min(
                1.0,
                0.25 + 0.55 * ratio + 0.20 * float(weight_obs.overfitting_signature),
            )
            influence_score = min(
                1.0,
                0.20 + 0.50 * ratio + 0.30 * float(weight_obs.gradient_noise_estimate),
            )

            phoneme_entropy_scores[phoneme] = round(float(entropy_score), 4)
            phoneme_influence_scores[phoneme] = round(float(influence_score), 4)

        sorted_risk = sorted(
            phoneme_vocabulary,
            key=lambda ph: phoneme_entropy_scores[ph] + phoneme_influence_scores[ph],
            reverse=True,
        )
        cooccurrence: list[tuple[str, str, float]] = []
        for idx in range(max(0, min(3, len(sorted_risk) - 1))):
            left = sorted_risk[idx]
            right = sorted_risk[idx + 1]
            pair_score = (
                phoneme_entropy_scores[left]
                + phoneme_influence_scores[left]
                + phoneme_entropy_scores[right]
                + phoneme_influence_scores[right]
            ) / 4.0
            if pair_score > 0.60:
                cooccurrence.append((left, right, round(float(pair_score), 4)))

        training_distribution: dict[str, int] | None = None
        if self._custom_clip_obs:
            dist: dict[str, int] = {}
            for clip_obs in self._custom_clip_obs:
                for phoneme in clip_obs.phoneme_sequence:
                    dist[phoneme] = int(dist.get(phoneme, 0)) + 1
            training_distribution = dist if dist else None

        return PhonemeRiskObservation(
            weight_evidence=weight_evidence,
            high_entropy_token_flags=weight_evidence.high_entropy_token_flags,
            phoneme_vocabulary=phoneme_vocabulary,
            phoneme_to_token_indices=phoneme_to_token_indices,
            phoneme_entropy_scores=phoneme_entropy_scores,
            phoneme_influence_scores=phoneme_influence_scores,
            phoneme_cooccurrence_anomalies=cooccurrence,
            behavior_vocabulary=["smile", "blink", "head_turn", "jaw_drift", "brow_raise"],
            training_data_phoneme_distribution=training_distribution,
            suspected_anomalous_phonemes_from_subenv2=(
                weight_obs.suspected_anomalous_phonemes
                if weight_obs.suspected_anomalous_phonemes
                else None
            ),
        )

    def _synthesize_behavioral_ground_truth(
        self,
        baseline_action: PhonemeRiskAction,
    ) -> GroundTruthBehavioralAnnotation:
        return GroundTruthBehavioralAnnotation(
            phoneme_risk_ranking=baseline_action.phoneme_risk_ranking,
            predicted_behavior_triggers=baseline_action.predicted_behavior_triggers,
            risky_phoneme_clusters=baseline_action.risky_phoneme_clusters,
            model_behavioral_safety=baseline_action.model_behavioral_safety,
            valid_mitigation_set={
                (rec.target, rec.action)
                for rec in baseline_action.mitigation_recommendations
            },
        )

    def _build_phoneme_risk_observation(self) -> PhonemeRiskObservation:
        if self._custom_weight_obs is not None:
            return self._build_phoneme_risk_observation_from_weight(self._custom_weight_obs)

        if self._subenv3_case and isinstance(self._subenv3_case.get("observation"), dict):
            return PhonemeRiskObservation.model_validate(self._subenv3_case["observation"])

        if self._subenv3_case and isinstance(self._subenv3_case.get("phoneme_obs"), dict):
            return PhonemeRiskObservation.model_validate(self._subenv3_case["phoneme_obs"])

        return PhonemeRiskObservation.model_validate(
            {
                "weight_evidence": {
                    "weight_file_id": "fallback.safetensors",
                    "training_quality": "healthy",
                    "rank_utilization_assessment": "efficient",
                    "high_entropy_token_flags": [],
                    "layer_anomaly_flags": [],
                    "overall_behavioral_risk": "low",
                    "evidence_summary": "No significant anomalies detected.",
                },
                "high_entropy_token_flags": [],
                "phoneme_vocabulary": [
                    "AA",
                    "AE",
                    "AH",
                    "AO",
                    "AW",
                    "AY",
                    "EH",
                    "ER",
                    "EY",
                    "IH",
                    "IY",
                    "OW",
                    "OY",
                    "UH",
                    "UW",
                ],
                "phoneme_to_token_indices": {},
                "phoneme_entropy_scores": {},
                "phoneme_influence_scores": {},
                "phoneme_cooccurrence_anomalies": [],
                "behavior_vocabulary": ["smile", "blink", "head_turn"],
                "training_data_phoneme_distribution": None,
                "suspected_anomalous_phonemes_from_subenv2": None,
            }
        )

    @property
    def state(self) -> State:
        return self._state
