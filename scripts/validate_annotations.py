#!/usr/bin/env python3
"""Validate TalkingHeadBench annotation files against schema definitions.

Usage
-----
::

    python scripts/validate_annotations.py \
      --cases tests/test_set/subenv3_cases.json \
      --subenv 3

    python scripts/validate_annotations.py \
      --cases tests/test_set/ \
      --subenv all

Exit codes
----------
- 0: success (all cases validated)
- 1: validation failure or malformed input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.ground_truth import (
    GroundTruthBehavioralAnnotation,
    GroundTruthClipAnnotation,
    GroundTruthImageAnnotation,
    GroundTruthParamAnnotation,
)
from src.schemas.subenv1 import ImageDiagnosticsObservation
from src.schemas.subenv2 import ClipSignalObservation
from src.schemas.subenv3 import PhonemeRiskObservation


def detect_subenv(observation: dict[str, Any]) -> int:
    """Detect sub-environment from observation field names."""
    if "face_occupancy_ratio" in observation:
        return 1
    if "image_obs" in observation:
        return 1
    if "face_embedding_variance" in observation or "evidence_dossier" in observation:
        return 2
    if "lora_rank" in observation or "weight_evidence" in observation:
        return 3
    raise ValueError(
        "Cannot auto-detect sub-environment from observation keys. "
        f"Keys present: {list(observation.keys())}. "
        "Expected 'face_occupancy_ratio' or 'image_obs' (sub-env 1), "
        "'face_embedding_variance' (sub-env 2), or "
        "'lora_rank' (sub-env 3)."
    )


def _coerce_subenv1_obs(obs_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _coerce_subenv1_gt(gt_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _coerce_subenv2_gt(gt_dict: dict[str, Any]) -> dict[str, Any]:
    """Support annotation-ready Sub-env 2 placeholder payloads."""
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


def _validate_subenv1(obs_dict: dict[str, Any], gt_dict: dict[str, Any], case_id: str) -> None:
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


def _validate_subenv2(obs_dict: dict[str, Any], gt_dict: dict[str, Any], case_id: str) -> None:
    """Validate a Sub-env 2 case against its Pydantic schemas."""
    try:
        ClipSignalObservation(**obs_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: ClipSignalObservation validation failed\n{exc}"
        ) from exc

    try:
        coerced_gt = _coerce_subenv2_gt(gt_dict)
    except ValueError as exc:
        raise ValueError(f"Case {case_id}: {exc}") from exc

    try:
        GroundTruthClipAnnotation(**coerced_gt)
    except ValidationError as exc:
        raise ValueError(
            f"Case {case_id}: GroundTruthClipAnnotation validation failed\n{exc}"
        ) from exc


def _validate_subenv3(obs_dict: dict[str, Any], gt_dict: dict[str, Any], case_id: str) -> None:
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


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and return the cases array from a JSON test-set file."""
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
            f"ERROR: '{path}': 'cases' must be a list, got {type(data['cases']).__name__}"
        )

    return data["cases"]


def _collect_json_files(cases_path: Path) -> list[Path]:
    """Return a sorted list of JSON files from a file or directory path."""
    if cases_path.is_file():
        return [cases_path]

    if cases_path.is_dir():
        files = sorted(cases_path.glob("*.json"))
        if not files:
            raise SystemExit(f"ERROR: No .json files found in directory '{cases_path}'")
        return files

    raise SystemExit(f"ERROR: '{cases_path}' is neither a file nor a directory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_annotations",
        description=(
            "Validate TalkingHeadBench case annotations against observation and "
            "ground-truth Pydantic schemas."
        ),
    )
    parser.add_argument(
        "--cases",
        required=True,
        metavar="PATH",
        help=(
            "Path to a JSON cases file, or a directory containing JSON files. "
            "Each file must contain a top-level 'cases' array."
        ),
    )
    parser.add_argument(
        "--subenv",
        required=True,
        choices=["1", "2", "3", "all"],
        metavar="{1,2,3,all}",
        help="Sub-environment index to validate, or 'all' to auto-detect per case.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    json_files = _collect_json_files(cases_path)

    mode_label = f"sub-env {args.subenv}" if args.subenv != "all" else "sub-env auto-detect"
    print(f"TalkingHeadBench annotation validator — mode: {mode_label}")
    print(f"Cases: {cases_path}  ({len(json_files)} file(s))")

    total_validated = 0

    for json_path in json_files:
        cases = _load_cases(json_path)
        print(f"\nFile: {json_path}  ({len(cases)} case(s))")

        file_validated = 0
        for raw_case in cases:
            case_id = str(raw_case.get("id", "???"))
            obs_dict = raw_case.get("observation", {})
            gt_dict = raw_case.get("ground_truth", {})

            if not isinstance(obs_dict, dict):
                print(
                    f"ERROR: Case {case_id}: observation must be a dict, "
                    f"got {type(obs_dict).__name__}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not isinstance(gt_dict, dict):
                print(
                    f"ERROR: Case {case_id}: ground_truth must be a dict, "
                    f"got {type(gt_dict).__name__}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.subenv == "all":
                try:
                    subenv = detect_subenv(obs_dict)
                except ValueError as exc:
                    print(f"ERROR: Case {case_id}: {exc}", file=sys.stderr)
                    sys.exit(1)
            else:
                subenv = int(args.subenv)

            try:
                if subenv == 1:
                    _validate_subenv1(obs_dict, gt_dict, case_id)
                elif subenv == 2:
                    _validate_subenv2(obs_dict, gt_dict, case_id)
                else:
                    _validate_subenv3(obs_dict, gt_dict, case_id)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)

            file_validated += 1
            total_validated += 1

        print(f"Schema OK: {file_validated} case(s)")

    print(f"\nValidation complete: {total_validated} case(s) validated")
    sys.exit(0)


if __name__ == "__main__":
    main()