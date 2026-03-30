"""
TalkingHeadBench OpenEnv Environment.

Wraps the existing benchmark into a classic OpenEnv step-based environment.
Episode flow is intentionally simplified to four agent-facing steps:

1) reset() emits ImageDiagnosticsObservation (Node 1 context)
2) step(ImageDiagnosticsAction) emits ParamAnomalyObservation (Node 2 context)
3) step(ParamAnomalyAction) emits PhonemeRiskObservation (Node 8 context)
4) step(PhonemeRiskAction) returns done=True with final weighted reward

Final reward:
    final = 0.25 * subenv1 + 0.35 * subenv2 + 0.40 * subenv3
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, TypeVar
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import (
    grade_anomaly_detection,
    grade_image_diagnostics,
    produce_reference_audit_handoff,
)
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv2.node6_grader import grade_clip_disposition
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
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
)
from src.schemas.subenv3 import PhonemeRiskAction, PhonemeRiskObservation

log = logging.getLogger(__name__)

_T = TypeVar("_T")
_TEST_SET_DIR = Path(__file__).resolve().parent.parent / "tests" / "test_set"


class TalkingHeadEnvironment(Environment[Action, Observation, State]):
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

        self._subenv3_case: dict[str, Any] | None = None
        self._subenv3_gt: GroundTruthBehavioralAnnotation | None = None

    def _load_cases(self, filename: str) -> list[dict[str, Any]]:
        path = _TEST_SET_DIR / filename
        if not path.exists():
            log.warning("Missing test case file: %s", path)
            return []
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cases = payload.get("cases", [])
        return cases if isinstance(cases, list) else []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if not self._subenv1_cases:
            raise RuntimeError("subenv1_cases.json has no cases; cannot reset environment")

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._scores = {}
        self._phase = "node1"

        self._node1_action = None
        self._node2_obs = None
        self._node2_gt = None

        if seed is not None:
            rng = random.Random(seed)
            idx = rng.randrange(len(self._subenv1_cases))
        else:
            idx = self._cursor % len(self._subenv1_cases)
            self._cursor += 1

        self._current_case = self._subenv1_cases[idx]
        self._node1_obs = ImageDiagnosticsObservation.model_validate(
            self._current_case["observation"]
        )
        self._node1_gt = GroundTruthImageAnnotation.model_validate(
            self._current_case["ground_truth"]
        )

        self._subenv3_case = self._pick_subenv3_case(idx)
        self._subenv3_gt = self._resolve_behavioral_ground_truth(self._subenv3_case)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "node": "node1_image_diagnostician",
                "step": 0,
                "episode_id": self._state.episode_id,
                "case_id": self._current_case.get("id", "unknown"),
                "instruction": (
                    "Diagnose reference-image quality and prompt risks from the provided "
                    "signals."
                ),
                "observation": self._node1_obs.model_dump(mode="json"),
                "expected_action_schema": "ImageDiagnosticsAction",
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1

        if self._phase == "node1":
            return self._handle_node1_action(action)
        if self._phase == "node2":
            return self._handle_node2_action(action)
        if self._phase == "node8":
            return self._handle_node8_action(action)

        return Observation(
            done=True,
            reward=0.0,
            metadata={"error": "Episode complete. Call reset() for a new episode."},
        )

    def _handle_node1_action(self, action: Action) -> Observation:
        if self._current_case is None or self._node1_obs is None or self._node1_gt is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")

        node1_action = self._parse_action(action, ImageDiagnosticsAction)
        self._node1_action = node1_action

        node1_score = grade_image_diagnostics(node1_action, self._node1_gt)
        self._scores["node1"] = node1_score

        self._node2_obs = self._build_param_anomaly_observation(node1_action, self._node1_obs)
        self._node2_gt = self._resolve_param_ground_truth(self._current_case, self._node2_obs)
        self._phase = "node2"

        return Observation(
            done=False,
            reward=node1_score,
            metadata={
                "node": "node2_param_anomaly_detector",
                "step": 1,
                "node1_score": node1_score,
                "instruction": (
                    "Analyze proposed generation parameters, detect anomalies, and propose "
                    "directional fixes."
                ),
                "observation": self._node2_obs.model_dump(mode="json"),
                "expected_action_schema": "ParamAnomalyAction",
            },
        )

    def _handle_node2_action(self, action: Action) -> Observation:
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

        reference_handoff = produce_reference_audit_handoff(
            self._node1_action,
            node2_action,
            self._node1_gt,
            self._node2_gt,
        )
        subenv2_score = self._run_subenv2_internally(
            reference_handoff.risk_profile,
            reference_handoff.estimated_drift_risk,
        )
        self._scores["subenv2"] = subenv2_score

        phoneme_obs = self._build_phoneme_risk_observation()
        self._phase = "node8"

        return Observation(
            done=False,
            reward=subenv1_score,
            metadata={
                "node": "node8_phoneme_risk_assessor",
                "step": 2,
                "subenv1_score": subenv1_score,
                "subenv2_score": subenv2_score,
                "instruction": (
                    "Assess phoneme-level behavioral risk from weight evidence and provide "
                    "mitigation guidance."
                ),
                "observation": phoneme_obs.model_dump(mode="json"),
                "expected_action_schema": "PhonemeRiskAction",
            },
        )

    def _handle_node8_action(self, action: Action) -> Observation:
        if self._subenv3_gt is None:
            raise RuntimeError("Behavioral ground truth is unavailable. Reset the environment.")

        node8_action = self._parse_action(action, PhonemeRiskAction)
        node8_score = grade_behavioral_audit(node8_action, self._subenv3_gt)

        self._scores["node8"] = node8_score
        self._scores["subenv3"] = node8_score

        s1 = self._scores.get("subenv1", 0.0)
        s2 = self._scores.get("subenv2", 0.0)
        s3 = self._scores.get("subenv3", 0.0)
        final_score = 0.25 * s1 + 0.35 * s2 + 0.40 * s3
        self._scores["final"] = final_score

        self._phase = "done"

        return Observation(
            done=True,
            reward=final_score,
            metadata={
                "node": "episode_complete",
                "step": 3,
                "reward_formula": "0.25 * subenv1 + 0.35 * subenv2 + 0.40 * subenv3",
                "scores": {
                    "subenv1_score": s1,
                    "subenv2_score": s2,
                    "subenv3_score": s3,
                    "final_score": final_score,
                    "breakdown": self._scores,
                },
            },
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
            proposed_config={"cfg": 7.5, "denoise_alt": 0.30, "eta": 0.10},
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

        # Current test sets do not include explicit Node 2 ground truth. For
        # deterministic grading, we synthesize a reference annotation from the
        # built-in Node 2 heuristic on the same observation.
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

    def _run_subenv2_internally(self, reference_risk_profile: str, estimated_drift_risk: float) -> float:
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
                    reference_risk_profile=reference_risk_profile,
                    estimated_drift_risk=estimated_drift_risk,
                    marginal_training_damage=marginal_damage,
                    marginal_coverage_gain=float(clip_obs.phoneme_coverage_new),
                )

                clip_action = recommend_clip_disposition(clip_disp_obs)
                scores.append(grade_clip_disposition(clip_action, clip_gt))
            except Exception as exc:  # noqa: BLE001
                log.warning("Sub-env 2 internal clip evaluation failed: %s", exc)

        return sum(scores) / len(scores) if scores else 0.5

    def _build_phoneme_risk_observation(self) -> PhonemeRiskObservation:
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