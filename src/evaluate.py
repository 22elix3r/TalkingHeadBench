"""
TalkingHeadBench evaluation harness.

Usage
-----
::

    # Validate all cases in a single file against their schemas (no node calls):
    python src/evaluate.py --test-set tests/test_set/subenv1_cases.json --subenv 1 --dry-run

    # Run Sub-env 1 graders on every case in a file:
    python src/evaluate.py --test-set tests/test_set/subenv1_cases.json --subenv 1

    # Run all sub-environments from a directory of JSON files:
    python src/evaluate.py --test-set tests/test_set/ --subenv all

    # Dry-run the entire directory:
    python src/evaluate.py --test-set tests/test_set/ --subenv all --dry-run

Exit codes
----------
- 0: success (all cases validated / scored)
- 1: schema validation failure or any other fatal error

Test-case JSON format
---------------------
Each file must be a JSON object with a ``"cases"`` array::

    {
      "cases": [
        {
          "id": "001",
          "observation": { ... },
          "ground_truth": { ... }
        }
      ]
    }

The schema for ``observation`` and ``ground_truth`` depends on ``--subenv``:

  --subenv 1
    observation keys:
      image_obs:        ImageDiagnosticsObservation  (all fields)
      proposed_config:  dict  (e.g. {"cfg": 7.5, "eta": 0.08})
    ground_truth keys:
      image:  GroundTruthImageAnnotation
      param:  GroundTruthParamAnnotation

  --subenv 2
        observation:  ClipSignalObservation  (all fields)
    ground_truth: GroundTruthClipAnnotation

  --subenv 3
    observation:  PhonemeRiskObservation  (all fields)
    ground_truth: GroundTruthBehavioralAnnotation

  --subenv all
    The file is expected to carry one of the three formats above; the harness
    auto-detects by checking which keys are present in ``observation``.
    Alternatively, supply a directory — each .json file is loaded independently
    and auto-detected.

Per-case output format (example for Sub-env 1)
-----------------------------------------------
::

    Case 001  score=0.745  [regime=1.00  risk=0.80  prompt=0.43]
    Case 002  score=0.512  [regime=0.70  risk=0.50  prompt=0.21]
    ─────────────────────────────────────────────────────────────
    Mean: 0.629   Std: 0.117   Min: 0.512   Max: 0.745
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------
from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import ImageDiagnosticsObservation, ParamAnomalyObservation
from src.schemas.subenv2 import (
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
)
from src.schemas.subenv3 import PhonemeRiskObservation

# ---------------------------------------------------------------------------
# Agent / grader imports
# ---------------------------------------------------------------------------
from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.envs.subenv1.node2_param_anomaly import detect_param_anomalies
from src.envs.subenv1.node3_grader import grade_anomaly_detection
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.envs.subenv2.node6_grader import grade_clip_disposition
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.envs.subenv3.node9_grader import grade_behavioral_audit
from src.pipeline import (
    _grade_image_diagnostics,
    _build_param_anomaly_obs,
)
from src.utils.grader_utils import jaccard_similarity, set_f1

# ─── Display constants ───────────────────────────────────────────────────────
_SEPARATOR = "─" * 61
_SAFETY_LEVELS = ["safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"]


# ---------------------------------------------------------------------------
# Auto-detection helper
# ---------------------------------------------------------------------------


def detect_subenv(observation: dict) -> int:
    """Detect sub-environment from observation field names."""
    # Sub-env 1: ImageDiagnosticsObservation
    if "face_occupancy_ratio" in observation:
        return 1
    # Sub-env 1 (wrapped): {"image_obs": {...}, "proposed_config": {...}}
    if "image_obs" in observation:
        return 1
    # Sub-env 2: ClipSignalObservation or ClipDispositionObservation
    if "face_embedding_variance" in observation or "evidence_dossier" in observation:
        return 2
    # Sub-env 3: WeightSignalObservation or PhonemeRiskObservation
    if "lora_rank" in observation or "weight_evidence" in observation:
        return 3
    raise ValueError(
        f"Cannot auto-detect sub-environment from observation keys. "
        f"Keys present: {list(observation.keys())}. "
        f"Expected 'face_occupancy_ratio' or 'image_obs' (sub-env 1), "
        f"'face_embedding_variance' (sub-env 2), or "
        f"'lora_rank' (sub-env 3)."
    )


def _coerce_subenv1_obs(obs_dict: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support both legacy wrapped and flat extractor Sub-env 1 observations."""
    if "image_obs" in obs_dict and isinstance(obs_dict["image_obs"], dict):
        image_obs = obs_dict["image_obs"]
        proposed_config = obs_dict.get("proposed_config", {})
    else:
        image_obs = obs_dict
        proposed_config = obs_dict.get("proposed_config", {})

    if not isinstance(proposed_config, dict):
        raise ValueError(
            f"'proposed_config' must be a dict, got {type(proposed_config).__name__}"
        )

    return image_obs, proposed_config


def _coerce_subenv1_gt(gt_dict: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support both legacy nested and flat annotation-ready Sub-env 1 ground truth."""
    if "image" in gt_dict and isinstance(gt_dict["image"], dict):
        image_gt = gt_dict["image"]
    else:
        image_gt = gt_dict

    if "param" in gt_dict and isinstance(gt_dict["param"], dict):
        param_gt = gt_dict["param"]
    else:
        # Flat extraction cases are image-first and do not include param GT yet.
        param_gt = {
            "config_risk_level": "safe",
            "anomalies": [],
            "predicted_failure_modes": [],
            "valid_fix_directions": [],
        }

    return image_gt, param_gt


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _validate_subenv1(obs_dict: dict, gt_dict: dict, case_id: str) -> None:
    """Validate a Sub-env 1 case against its Pydantic schemas."""
    try:
        image_obs_dict, proposed_config = _coerce_subenv1_obs(obs_dict)
        ImageDiagnosticsObservation(**image_obs_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: ImageDiagnosticsObservation validation failed\n{exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Case {case_id}: {exc}") from exc

    if not isinstance(proposed_config, dict):
        raise ValueError(f"Case {case_id}: 'proposed_config' must be a dict")

    try:
        image_gt_dict, param_gt_dict = _coerce_subenv1_gt(gt_dict)
        GroundTruthImageAnnotation(**image_gt_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: GroundTruthImageAnnotation validation failed\n{exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Case {case_id}: {exc}") from exc

    try:
        GroundTruthParamAnnotation(**param_gt_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: GroundTruthParamAnnotation validation failed\n{exc}"
        ) from exc


def _validate_subenv2(obs_dict: dict, gt_dict: dict, case_id: str) -> None:
    """Validate a Sub-env 2 case against its Pydantic schemas."""
    try:
        ClipSignalObservation(**obs_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: ClipSignalObservation validation failed\n{exc}"
        ) from exc

    try:
        gt_dict = _coerce_subenv2_gt(gt_dict)
    except ValueError as exc:
        raise ValueError(f"Case {case_id}: {exc}") from exc

    try:
        GroundTruthClipAnnotation(**gt_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: GroundTruthClipAnnotation validation failed\n{exc}"
        ) from exc


def _validate_subenv3(obs_dict: dict, gt_dict: dict, case_id: str) -> None:
    """Validate a Sub-env 3 case against its Pydantic schemas."""
    try:
        PhonemeRiskObservation(**obs_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: PhonemeRiskObservation validation failed\n{exc}"
        ) from exc

    try:
        GroundTruthBehavioralAnnotation(**gt_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: GroundTruthBehavioralAnnotation validation failed\n{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Grader runners — return (score, breakdown_str, dim_dict)
# ---------------------------------------------------------------------------


def _run_subenv1(obs_dict: dict, gt_dict: dict, case_id: str):
    """Run Sub-env 1 graders and return (score, breakdown, dims).

    Executes Node 1 → Node 2 → Node 3 graders.

    Breakdown dimensions:
      regime  — regime classification accuracy (0.35 weight in node1)
      risk    — risk factor recall (0.35 weight in node1)
      prompt  — prompt modification validity (0.30 weight in node1)
      node2   — full composite Node 2 score

    Final sub-env 1 score = 0.50 * node1_score + 0.50 * node2_score.
    """
    image_obs_dict, proposed_config = _coerce_subenv1_obs(obs_dict)
    image_gt_dict, param_gt_dict = _coerce_subenv1_gt(gt_dict)

    image_obs = ImageDiagnosticsObservation(**image_obs_dict)
    gt_image = GroundTruthImageAnnotation(**image_gt_dict)
    gt_param = GroundTruthParamAnnotation(**param_gt_dict)

    # Node 1
    node1_action = diagnose_image(image_obs)

    # Node 2
    node2_obs = _build_param_anomaly_obs(node1_action, image_obs, proposed_config)
    node2_action = detect_param_anomalies(node2_obs)

    # Node 3 — grade node1 (sub-dimensions)
    # Regime accuracy
    if node1_action.regime_classification == gt_image.regime_classification:
        regime_score = 1.0
    elif node1_action.regime_classification in gt_image.acceptable_regimes:
        regime_score = 0.7
    else:
        regime_score = 0.0

    # Risk factor recall
    pred_risks = set(node1_action.identified_risk_factors)
    true_risks = set(gt_image.identified_risk_factors)
    risk_score = (
        len(pred_risks & true_risks) / len(true_risks) if true_risks else 1.0
    )

    # Prompt modification validity
    valid_mods = set(gt_image.valid_prompt_modifications)
    agent_mods = set(node1_action.recommended_prompt_modifications)
    if agent_mods:
        prompt_score = len(agent_mods & valid_mods) / len(agent_mods)
    else:
        prompt_score = 0.0 if valid_mods else 1.0

    node1_score = (
        0.35 * regime_score
        + 0.35 * risk_score
        + 0.30 * prompt_score
    )

    # Node 3 — grade node2
    node2_score = grade_anomaly_detection(node2_action, gt_param)

    final_score = 0.50 * node1_score + 0.50 * node2_score

    dims = {
        "regime": regime_score,
        "risk": risk_score,
        "prompt": prompt_score,
        "node2": node2_score,
    }
    breakdown = (
        f"regime={regime_score:.2f}  risk={risk_score:.2f}  "
        f"prompt={prompt_score:.2f}  node2={node2_score:.2f}"
    )
    return final_score, breakdown, dims


def _coerce_subenv2_gt(gt_dict: dict) -> dict[str, Any]:
    """Support annotation-ready Sub-env 2 ground truth placeholders.

    If ``disposition`` is ``"ANNOTATE"``, this maps to a concrete
    ``GroundTruthClipAnnotation`` payload using ``_annotation_notes`` hints.
    """
    if not isinstance(gt_dict, dict):
        raise ValueError(
            f"Sub-env 2 ground_truth must be a dict, got {type(gt_dict).__name__}"
        )

    disposition = gt_dict.get("disposition")
    if not (isinstance(disposition, str) and disposition.upper() == "ANNOTATE"):
        return gt_dict

    notes = gt_dict.get("_annotation_notes", {})
    if not isinstance(notes, dict):
        notes = {}

    suggested = str(notes.get("suggested_disposition", "defer")).lower()
    if suggested not in {"accept", "reject", "fix", "defer"}:
        suggested = "defer"

    try:
        confidence = float(notes.get("suggested_confidence", gt_dict.get("confidence", 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    default_ambiguity = 0.5 if suggested == "defer" else 0.0
    try:
        ambiguity = float(gt_dict.get("disposition_ambiguity", default_ambiguity))
    except (TypeError, ValueError):
        ambiguity = default_ambiguity
    ambiguity = max(0.0, min(1.0, ambiguity))

    valid_fix_steps = gt_dict.get("valid_fix_steps", [])
    if not isinstance(valid_fix_steps, list):
        valid_fix_steps = []

    valid_override_justifications = gt_dict.get("valid_override_justifications", [])
    if not isinstance(valid_override_justifications, list):
        valid_override_justifications = []

    expected_reasoning_elements = gt_dict.get("expected_reasoning_elements", [])
    if not isinstance(expected_reasoning_elements, list):
        expected_reasoning_elements = []
    if not expected_reasoning_elements or all(
        str(v).upper() == "ANNOTATE" for v in expected_reasoning_elements
    ):
        expected_reasoning_elements = [
            "dataset phoneme gaps",
            "pose gaps",
            "critical gaps",
        ]

    return {
        "disposition": suggested,
        "confidence": confidence,
        "disposition_ambiguity": ambiguity,
        "valid_fix_steps": valid_fix_steps,
        "valid_override_justifications": valid_override_justifications,
        "expected_reasoning_elements": expected_reasoning_elements,
    }


def _resolve_subenv2_dossier_builder() -> Callable[[ClipSignalObservation], ClipEvidenceDossier]:
    """Resolve the Sub-env 2 evidence-dossier mapper from pipeline helpers."""
    from src import pipeline as pipeline_module

    fn = getattr(pipeline_module, "build_evidence_dossier", None)
    if callable(fn):
        return fn

    fn = getattr(pipeline_module, "_heuristic_clip_evidence_dossier", None)
    if callable(fn):
        return fn

    raise RuntimeError(
        "No Sub-env 2 dossier builder found in pipeline. "
        "Expected build_evidence_dossier or _heuristic_clip_evidence_dossier."
    )


def _run_subenv2(obs_dict: dict, gt_dict: dict, case_id: str):
    """Run Sub-env 2 graders and return (score, breakdown, dims).

    Executes Node 5 → Node 6 grader.

    Breakdown dimensions (mirroring node6 internal scoring):
      base        — base disposition score  (0.40 max)
      fix_quality — fix instruction quality (0.20 max)
      reasoning   — dataset impact reasoning (0.20 max)
      override    — override penalty (subtractive, 0.10 max)
    """
    clip_signal_obs = ClipSignalObservation(**obs_dict)
    gt_clip = GroundTruthClipAnnotation(**_coerce_subenv2_gt(gt_dict))

    dossier_builder = _resolve_subenv2_dossier_builder()
    dossier = dossier_builder(clip_signal_obs)
    if isinstance(dossier, ClipEvidenceDossier):
        evidence_dossier = dossier
    elif isinstance(dossier, dict):
        evidence_dossier = ClipEvidenceDossier(**dossier)
    else:
        raise ValueError(
            f"Sub-env 2 dossier builder returned unsupported type: {type(dossier).__name__}"
        )

    clip_obs = ClipDispositionObservation(
        evidence_dossier=evidence_dossier,
        minimum_clips_needed=20,
        phoneme_gap_severity={},
        pose_gap_severity={},
        budget_remaining=10,
        marginal_training_damage=0.2,
        marginal_coverage_gain=0.5,
    )

    # Node 5
    action = recommend_clip_disposition(clip_obs)

    # Node 6 — compute sub-dimensions manually for display, then get final
    score = grade_clip_disposition(action, gt_clip)

    # Reconstruct sub-dimension contributions for display
    base_score = 0.0
    if action.disposition == gt_clip.disposition:
        calibrated = abs(action.confidence - gt_clip.confidence) < 0.15
        base_score = 0.40 if calibrated else 0.28
    elif action.disposition == "fix" and gt_clip.disposition == "reject":
        base_score = 0.20
    elif action.disposition == "defer":
        if gt_clip.disposition_ambiguity >= 0.5:
            base_score = 0.15 if action.defer_reason else 0.10

    fix_score = 0.0
    if action.disposition == "fix" and action.fix_instructions:
        valid_steps = sum(
            1 for s in action.fix_instructions if s in gt_clip.valid_fix_steps
        )
        fp = valid_steps / len(action.fix_instructions)
        fix_score = 0.20 if fp >= 0.8 else (0.10 if fp >= 0.5 else 0.0)

    reasoning_score = 0.0
    kw_elements = gt_clip.expected_reasoning_elements
    agent_reasoning = action.dataset_impact_reasoning.lower()
    matched = sum(1 for kw in kw_elements if kw in agent_reasoning)
    reasoning_score = (
        0.20
        if matched >= len(kw_elements) * 0.8
        else 0.10 if matched >= 1 else 0.00
    )

    override_penalty = -0.0
    has_override_labels = (
        isinstance(gt_clip.valid_override_justifications, list)
        and len(gt_clip.valid_override_justifications) > 0
    )
    if has_override_labels and action.override_decision == "applied":
        if not action.override_justification:
            override_penalty = -0.10
        elif action.override_justification not in gt_clip.valid_override_justifications:
            override_penalty = -0.05

    dims = {
        "base": base_score,
        "fix": fix_score,
        "reasoning": reasoning_score,
        "override_penalty": override_penalty,
        "disposition": action.disposition,
    }
    breakdown = (
        f"base={base_score:.2f}  fix={fix_score:.2f}  "
        f"reasoning={reasoning_score:.2f}  override={override_penalty:+.2f}  "
        f"→{action.disposition}"
    )
    return score, breakdown, dims


def _run_subenv3(obs_dict: dict, gt_dict: dict, case_id: str):
    """Run Sub-env 3 graders and return (score, breakdown, dims).

    Executes Node 8 → Node 9 grader.

    Breakdown dimensions (node9 weights):
      ranking     — top-5 phoneme ranking overlap  (0.15)
      triggers    — behavior trigger set F1        (0.30)
      clusters    — phoneme cluster Jaccard        (0.20)
      safety      — safety level ordinal distance  (0.15)
      mitigation  — mitigation precision           (0.20)
    """
    phoneme_obs = PhonemeRiskObservation(**obs_dict)
    gt_behavioral = GroundTruthBehavioralAnnotation(**gt_dict)

    # Node 8
    action = assess_phoneme_risk(phoneme_obs)

    # Node 9
    score = grade_behavioral_audit(action, gt_behavioral)

    # Sub-dimension breakdown (mirrors node9 logic verbatim)
    agent_top5 = {e.phoneme for e in action.phoneme_risk_ranking[:5]}
    true_top5 = {e.phoneme for e in gt_behavioral.phoneme_risk_ranking[:5]}
    ranking_score = len(agent_top5 & true_top5) / 5

    agent_triggers = {
        (t.trigger_phoneme, t.triggered_behavior)
        for t in action.predicted_behavior_triggers
    }
    true_triggers = {
        (t.trigger_phoneme, t.triggered_behavior)
        for t in gt_behavioral.predicted_behavior_triggers
    }
    trigger_score = set_f1(agent_triggers, true_triggers)

    agent_clusters = {frozenset(c.phonemes) for c in action.risky_phoneme_clusters}
    true_clusters = {frozenset(c.phonemes) for c in gt_behavioral.risky_phoneme_clusters}
    cluster_score = jaccard_similarity(agent_clusters, true_clusters)

    try:
        ai = _SAFETY_LEVELS.index(action.model_behavioral_safety)
        ti = _SAFETY_LEVELS.index(gt_behavioral.model_behavioral_safety)
        safety_score = 1.0 - abs(ai - ti) / (len(_SAFETY_LEVELS) - 1)
    except ValueError:
        safety_score = 0.0

    agent_mits = {(m.target, m.action) for m in action.mitigation_recommendations}
    valid_mits = gt_behavioral.valid_mitigation_set
    mit_score = (
        len(agent_mits & valid_mits) / len(agent_mits) if agent_mits
        else (0.0 if valid_mits else 1.0)
    )

    dims = {
        "ranking": ranking_score,
        "triggers": trigger_score,
        "clusters": cluster_score,
        "safety": safety_score,
        "mitigation": mit_score,
    }
    breakdown = (
        f"ranking={ranking_score:.2f}  triggers={trigger_score:.2f}  "
        f"clusters={cluster_score:.2f}  safety={safety_score:.2f}  "
        f"mitigation={mit_score:.2f}"
    )
    return score, breakdown, dims


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _load_cases(path: Path) -> list[dict]:
    """Load and return the ``cases`` list from a JSON test-set file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"ERROR: Cannot read file '{path}': {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: Invalid JSON in '{path}': {exc}") from exc

    if not isinstance(data, dict) or "cases" not in data:
        raise SystemExit(
            f"ERROR: '{path}' must be a JSON object with a top-level 'cases' array. "
            f"Got keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    if not isinstance(data["cases"], list):
        raise SystemExit(
            f"ERROR: '{path}': 'cases' must be a list, "
            f"got {type(data['cases']).__name__}"
        )

    return data["cases"]


def _collect_json_files(test_set_path: Path) -> list[Path]:
    """Return a sorted list of .json files from a path (file or directory)."""
    if test_set_path.is_file():
        return [test_set_path]
    if test_set_path.is_dir():
        files = sorted(test_set_path.glob("*.json"))
        if not files:
            raise SystemExit(
                f"ERROR: No .json files found in directory '{test_set_path}'"
            )
        return files
    raise SystemExit(
        f"ERROR: '{test_set_path}' is neither a file nor a directory"
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(scores: list[float]) -> None:
    """Print the mean/std/min/max summary line."""
    if not scores:
        print("  (no cases scored)")
        return
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    lo = min(scores)
    hi = max(scores)
    print(_SEPARATOR)
    print(f"Mean: {mean:.3f}   Std: {std:.3f}   Min: {lo:.3f}   Max: {hi:.3f}")


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------


def _process_file(
    json_path: Path,
    subenv_arg: str,
    dry_run: bool,
) -> list[float]:
    """Process all cases in a single JSON file.

    Returns the list of per-case scores (empty in --dry-run mode).
    """
    cases = _load_cases(json_path)

    print(f"\nFile: {json_path}  ({len(cases)} case(s))")

    scores: list[float] = []
    total_validated = 0

    for raw_case in cases:
        case_id: str = str(raw_case.get("id", "???"))
        obs_dict: dict = raw_case.get("observation", {})
        gt_dict: dict = raw_case.get("ground_truth", {})

        # ── Determine sub-environment ────────────────────────────────────────
        if subenv_arg == "all":
            try:
                subenv = detect_subenv(obs_dict)
            except ValueError as exc:
                print(f"ERROR: Case {case_id}: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            subenv = int(subenv_arg)

        # ── Schema validation ────────────────────────────────────────────────
        try:
            if subenv == 1:
                _validate_subenv1(obs_dict, gt_dict, case_id)
            elif subenv == 2:
                _validate_subenv2(obs_dict, gt_dict, case_id)
            elif subenv == 3:
                _validate_subenv3(obs_dict, gt_dict, case_id)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        total_validated += 1

        if dry_run:
            continue

        # ── Run graders ──────────────────────────────────────────────────────
        try:
            if subenv == 1:
                score, breakdown, _ = _run_subenv1(obs_dict, gt_dict, case_id)
            elif subenv == 2:
                score, breakdown, _ = _run_subenv2(obs_dict, gt_dict, case_id)
            else:
                score, breakdown, _ = _run_subenv3(obs_dict, gt_dict, case_id)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: Case {case_id}: node/grader raised an exception: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        scores.append(score)
        print(f"Case {case_id:<6}  score={score:.3f}  [{breakdown}]")

    if dry_run:
        print(f"Schema OK: {total_validated} case(s)")
    else:
        _print_summary(scores)

    return scores


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "TalkingHeadBench evaluation harness. "
            "Runs graders for one or all sub-environments against a JSON test set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        required=True,
        metavar="PATH",
        help=(
            "Path to a JSON test-set file, or a directory of JSON files. "
            "Each file must be a JSON object with a top-level 'cases' array."
        ),
    )
    parser.add_argument(
        "--subenv",
        required=True,
        choices=["1", "2", "3", "all"],
        metavar="{1,2,3,all}",
        help=(
            "Which sub-environment to evaluate. "
            "Use 'all' to auto-detect from observation keys."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate all observation and ground-truth dicts against their "
            "Pydantic schemas, then exit 0.  No node calls are made."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the evaluation harness."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    test_set_path = Path(args.test_set)
    json_files = _collect_json_files(test_set_path)

    mode_label = "DRY-RUN (schema validation only)" if args.dry_run else f"sub-env {args.subenv}"
    print(f"TalkingHeadBench evaluate — mode: {mode_label}")
    print(f"Test set: {test_set_path}  ({len(json_files)} file(s))")

    all_scores: list[float] = []

    for json_path in json_files:
        file_scores = _process_file(json_path, args.subenv, args.dry_run)
        all_scores.extend(file_scores)

    # If multiple files were scored, print a cross-file summary
    if not args.dry_run and len(json_files) > 1 and all_scores:
        print(f"\nOverall summary across {len(json_files)} file(s):")
        _print_summary(all_scores)

    sys.exit(0)


if __name__ == "__main__":
    main()
