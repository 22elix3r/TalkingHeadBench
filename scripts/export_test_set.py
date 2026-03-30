#!/usr/bin/env python3
"""Export a clean TalkingHeadBench test set from annotation files.

This utility validates annotated cases, strips annotation-only metadata,
re-validates, writes a compact {"cases": [...]} payload, then verifies the
result with `src/evaluate.py --dry-run`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_annotations import (
    _collect_json_files,
    _load_cases,
    _validate_subenv1,
    _validate_subenv2,
    _validate_subenv3,
    detect_subenv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_test_set",
        description=(
            "Validate and export cleaned TalkingHeadBench test-set JSON "
            "from annotated case files."
        ),
    )
    parser.add_argument(
        "--cases",
        required=True,
        metavar="PATH",
        help=(
            "Path to a JSON cases file, or a directory containing JSON files. "
            "Each file must have a top-level 'cases' array."
        ),
    )
    parser.add_argument(
        "--subenv",
        required=True,
        choices=["1", "2", "3", "all"],
        metavar="{1,2,3,all}",
        help="Sub-environment index to export, or 'all' for per-case auto-detection.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output JSON path for the cleaned export payload.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=1,
        metavar="N",
        help="Minimum number of clean cases required for export (default: 1).",
    )
    return parser


def _validate_case(
    obs_dict: dict[str, Any],
    gt_dict: dict[str, Any],
    case_id: str,
    subenv: int,
) -> None:
    if subenv == 1:
        _validate_subenv1(obs_dict, gt_dict, case_id)
    elif subenv == 2:
        _validate_subenv2(obs_dict, gt_dict, case_id)
    elif subenv == 3:
        _validate_subenv3(obs_dict, gt_dict, case_id)
    else:
        raise ValueError(f"Unsupported subenv: {subenv}")


def _strip_annotation_metadata(case: dict[str, Any]) -> dict[str, Any]:
    clean_case = deepcopy(case)

    # Remove non-portable provenance fields from case-level payload.
    clean_case.pop("source_file", None)
    clean_case.pop("tokenizer_config_used", None)

    observation = clean_case.get("observation")
    if isinstance(observation, dict):
        observation.pop("tokenizer_config_used", None)

    ground_truth = clean_case.get("ground_truth")
    if isinstance(ground_truth, dict):
        ground_truth.pop("_annotation_notes", None)
        ground_truth.pop("tokenizer_config_used", None)

    return clean_case


def _update_distribution_summaries(
    ground_truth: dict[str, Any],
    regime_counts: Counter[str],
    disposition_counts: Counter[str],
    safety_counts: Counter[str],
) -> None:
    regime = ground_truth.get("regime_classification")
    if isinstance(regime, str) and regime:
        regime_counts[regime] += 1

    disposition = ground_truth.get("disposition")
    if isinstance(disposition, str) and disposition:
        disposition_counts[disposition] += 1

    safety = ground_truth.get("model_behavioral_safety")
    if isinstance(safety, str) and safety:
        safety_counts[safety] += 1


def _fmt_distribution(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=True)


def _run_evaluate_dry_run(export_path: Path, subenv_arg: str) -> None:
    cmd = [
        sys.executable,
        "src/evaluate.py",
        "--test-set",
        str(export_path),
        "--subenv",
        subenv_arg,
        "--dry-run",
    ]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print("\n[evaluate --dry-run]\n" + result.stdout.strip())

    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr output)"
        raise SystemExit(
            "ERROR: Final evaluate --dry-run verification failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{stderr}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.min_cases < 1:
        raise SystemExit("ERROR: --min-cases must be >= 1")

    cases_path = Path(args.cases)
    output_path = Path(args.output)
    json_files = _collect_json_files(cases_path)

    mode_label = f"sub-env {args.subenv}" if args.subenv != "all" else "sub-env auto-detect"
    print(f"TalkingHeadBench export_test_set - mode: {mode_label}")
    print(f"Input: {cases_path}  ({len(json_files)} file(s))")

    clean_cases: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()

    for json_path in json_files:
        cases = _load_cases(json_path)
        print(f"\nFile: {json_path}  ({len(cases)} case(s))")

        for raw_case in cases:
            case_id = str(raw_case.get("id", "???"))
            obs_dict = raw_case.get("observation", {})
            gt_dict = raw_case.get("ground_truth", {})

            if not isinstance(obs_dict, dict):
                raise SystemExit(
                    f"ERROR: Case {case_id}: observation must be a dict, "
                    f"got {type(obs_dict).__name__}"
                )
            if not isinstance(gt_dict, dict):
                raise SystemExit(
                    f"ERROR: Case {case_id}: ground_truth must be a dict, "
                    f"got {type(gt_dict).__name__}"
                )

            if args.subenv == "all":
                try:
                    subenv = detect_subenv(obs_dict)
                except ValueError as exc:
                    raise SystemExit(f"ERROR: Case {case_id}: {exc}") from exc
            else:
                subenv = int(args.subenv)

            # Step 1: Validate source annotations with the canonical validator logic.
            try:
                _validate_case(obs_dict, gt_dict, case_id, subenv)
            except ValueError as exc:
                raise SystemExit(f"ERROR: {exc}") from exc

            # Steps 2/3: Strip annotation-only and provenance fields.
            clean_case = _strip_annotation_metadata(raw_case)
            clean_obs = clean_case.get("observation", {})
            clean_gt = clean_case.get("ground_truth", {})

            if not isinstance(clean_obs, dict):
                raise SystemExit(
                    f"ERROR: Case {case_id}: cleaned observation must be a dict, "
                    f"got {type(clean_obs).__name__}"
                )
            if not isinstance(clean_gt, dict):
                raise SystemExit(
                    f"ERROR: Case {case_id}: cleaned ground_truth must be a dict, "
                    f"got {type(clean_gt).__name__}"
                )

            # Step 4: Final Pydantic validation after stripping.
            try:
                _validate_case(clean_obs, clean_gt, case_id, subenv)
            except ValueError as exc:
                raise SystemExit(
                    "ERROR: Post-strip validation failed for "
                    f"case {case_id}: {exc}"
                ) from exc

            _update_distribution_summaries(
                clean_gt,
                regime_counts,
                disposition_counts,
                safety_counts,
            )
            clean_cases.append(clean_case)

    # Step 5: Enforce minimum case threshold.
    if len(clean_cases) < args.min_cases:
        raise SystemExit(
            f"ERROR: Export aborted - {len(clean_cases)} case(s) is below "
            f"--min-cases={args.min_cases}."
        )

    # Step 6: Write clean payload.
    payload = {"cases": clean_cases}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote clean export: {output_path}  ({len(clean_cases)} case(s))")

    # Step 7: Final verifier pass through evaluate.py --dry-run.
    _run_evaluate_dry_run(output_path, args.subenv)

    # Step 8: Print key annotation distribution summaries.
    print("\nDistribution summary")
    print(f"  regime:      {_fmt_distribution(regime_counts)}")
    print(f"  disposition: {_fmt_distribution(disposition_counts)}")
    print(f"  safety:      {_fmt_distribution(safety_counts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
