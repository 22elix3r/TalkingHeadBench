#!/usr/bin/env python3
"""Generate human-readable annotation worksheets from sub-environment case JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

STEP_CONTEXT = {
    "step_00250": "very early - identity not yet established",
    "step_00500": "early - unstable identity",
    "step_00750": "mid - improving",
    "step_01000": "mid-late - approaching convergence",
    "step_01250": "good - identity preservation confirmed (your threshold)",
    "step_01500": "converged - minor residual artifacts only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown annotation worksheet from Sub-env case JSON "
            "for human review."
        )
    )
    parser.add_argument(
        "--cases",
        required=True,
        type=Path,
        help="Path to a cases JSON file (for example tests/test_set/subenv1_cases.json)",
    )
    parser.add_argument(
        "--subenv",
        required=True,
        type=int,
        choices=(1, 2, 3),
        help="Sub-environment index (1, 2, or 3)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output markdown path (for example docs/annotation_worksheet_subenv1.md)",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.3f}"


def _fmt_int(value: Any) -> str:
    parsed = _as_int(value)
    if parsed is None:
        return "N/A"
    return str(parsed)


def _fmt_distribution(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=True)


def _safe_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _case_label(case: dict[str, Any]) -> tuple[str, str]:
    case_id_raw = str(case.get("id", "")).strip()
    if case_id_raw.isdigit():
        case_id = case_id_raw.zfill(3)
    elif case_id_raw:
        case_id = case_id_raw
    else:
        case_id = "unknown"

    source_file = case.get("source_file")
    if not source_file:
        obs = case.get("observation") or {}
        source_file = (
            (obs.get("weight_file_id") if isinstance(obs, dict) else None)
            or (
                (obs.get("weight_evidence") or {}).get("weight_file_id")
                if isinstance(obs, dict)
                else None
            )
            or "unknown"
        )

    return case_id, str(source_file)


def _interp_face_occupancy(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.25:
        return "low - face too small for reliable identity encoding"
    if v < 0.50:
        return "moderate - face visible but background present"
    return "good - face dominant in frame"


def _interp_yaw(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    a = abs(v)
    if a > 45:
        return "extreme non-frontal - profile view"
    if a > 25:
        return "non-frontal - lateral pose"
    if a > 10:
        return "slight turn - borderline"
    return "frontal"


def _interp_bg_complexity(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.70:
        return "complex - busy background competes for attention"
    if v > 0.40:
        return "moderate complexity"
    return "simple background"


def _interp_lighting(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.40:
        return "uneven - shadow regions risk inconsistent rendering"
    if v < 0.60:
        return "moderate"
    return "good"


def _interp_sharpness(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.30:
        return "blurry - may cause degraded identity encoding"
    if v < 0.50:
        return "soft"
    return "acceptable"


def _interp_occlusion_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "occlusion present" if value else "clean"
    if value is None:
        return "unknown"
    return "occlusion present" if bool(value) else "clean"


def _interp_identity_drift(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.30:
        return "severe drift - identity unstable"
    if v > 0.15:
        return "moderate drift"
    if v > 0.05:
        return "minor drift"
    return "stable - identity consistent across frames"


def _interp_landmark_stability(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.70:
        return "high instability - jaw/face landmark jitter"
    if v > 0.40:
        return "moderate stability"
    return "stable"


def _interp_lip_sync(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.10:
        return "absent"
    if v < 0.40:
        return "poor"
    if v < 0.70:
        return "acceptable"
    return "good"


def _interp_phoneme_novelty(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if abs(v) < 1e-9:
        return "no new phonemes - redundant coverage"
    if v < 0.30:
        return "low novelty"
    if v < 0.60:
        return "moderate novelty"
    return "high novelty - valuable for coverage"


def _interp_blur(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.30:
        return "blurry"
    if v < 0.60:
        return "acceptable"
    return "sharp"


def _interp_exposure(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.30:
        return "underexposed"
    if v < 0.70:
        return "normal"
    return "overexposed"


def _interp_embedding_variance(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.020:
        return "high variance - identity inconsistent"
    if v > 0.005:
        return "moderate variance"
    return "low variance - consistent face"


def _interp_occlusion_frames(value: Any) -> str:
    n = _as_int(value)
    if n is None:
        return "unknown"
    if n == 0:
        return "none observed"
    if n < 5:
        return "occasional occlusion"
    return "frequent occlusion"


def _interp_gradient_noise(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.08:
        return "high noise - very early or unstable training"
    if v > 0.04:
        return "moderate noise - still converging"
    return "low noise - converged"


def _interp_overfitting(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v > 0.50:
        return "high - rank collapse detected"
    if v > 0.20:
        return "moderate"
    return "low - healthy rank utilization"


def _interp_entropy_mean(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "unknown"
    if v < 0.10:
        return "very low - untrained or collapsed"
    if v < 0.15:
        return "low - early training"
    if v < 0.20:
        return "moderate - converging"
    return "high - well trained"


def _json_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _extract_entropy_mean(observation: dict[str, Any]) -> float | None:
    canonical = observation.get("canonical_entropy_per_layer")
    if isinstance(canonical, dict) and canonical:
        vals = [_as_float(v) for v in canonical.values()]
        nums = [v for v in vals if v is not None]
        if nums:
            return float(sum(nums) / len(nums))

    flags = observation.get("high_entropy_token_flags")
    if isinstance(flags, list):
        entropy_values: list[float] = []
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            evidence = str(flag.get("evidence", ""))
            match = re.search(r"layer entropy mean\s+([0-9]*\.?[0-9]+)", evidence)
            if match:
                entropy_values.append(float(match.group(1)))
        if entropy_values:
            return float(sum(entropy_values) / len(entropy_values))

    return None


def _extract_flagged_positions(observation: dict[str, Any]) -> list[int]:
    direct = observation.get("high_entropy_token_positions")
    if isinstance(direct, list):
        parsed = [_as_int(v) for v in direct]
        return sorted({v for v in parsed if v is not None})

    flags = observation.get("high_entropy_token_flags")
    if isinstance(flags, list):
        positions: list[int] = []
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            pos = _as_int(flag.get("token_position"))
            if pos is not None:
                positions.append(pos)
        return sorted(set(positions))

    return []


def _extract_token_mapping(observation: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}

    direct = observation.get("token_position_to_phoneme")
    if isinstance(direct, dict):
        for key, value in direct.items():
            pos = _as_int(key)
            if pos is None:
                continue
            mapping[pos] = str(value)

    by_phoneme = observation.get("phoneme_to_token_indices")
    if isinstance(by_phoneme, dict):
        for phoneme, positions in by_phoneme.items():
            if not isinstance(positions, list):
                continue
            for pos_raw in positions:
                pos = _as_int(pos_raw)
                if pos is None:
                    continue
                mapping.setdefault(pos, str(phoneme))

    flags = observation.get("high_entropy_token_flags")
    if isinstance(flags, list):
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            pos = _as_int(flag.get("token_position"))
            mapped = flag.get("mapped_phoneme")
            if pos is None or not mapped:
                continue
            mapping.setdefault(pos, str(mapped))

    return mapping


def _render_subenv1_case(case: dict[str, Any]) -> tuple[str, str]:
    case_id, source_file = _case_label(case)
    observation = case.get("observation") or {}
    ground_truth = case.get("ground_truth") or {}
    notes = ground_truth.get("_annotation_notes") or {}

    suggested_regime = str(notes.get("suggested_regime", "unknown"))
    suggested_risk_factors = _safe_str_list(notes.get("suggested_risk_factors"))
    suggested_prompt_issues = _safe_str_list(
        notes.get("suggested_prompt_issues") or notes.get("prompt_issues")
    )

    yaw = _as_float(observation.get("estimated_yaw_degrees"))
    yaw_value = "N/A" if yaw is None else f"{yaw:.3f} deg"
    occlusion = observation.get("occlusion_detected")
    occlusion_value = "N/A" if occlusion is None else str(bool(occlusion))

    if not suggested_risk_factors:
        suggested_risk_factors = [
            "<suggested risk factors - confirm, add, remove, or rephrase>"
        ]

    annotation_template = {
        "regime_classification": (
            f"<{suggested_regime} - confirm or change>"
            if suggested_regime != "unknown"
            else "<suggested - confirm or change>"
        ),
        "acceptable_regimes": [],
        "identified_risk_factors": suggested_risk_factors,
        "valid_prompt_modifications": [
            "<write 1-3 actionable modifications for this specific image>"
        ],
    }

    lines = [
        "---",
        f"### Case {case_id} - `{source_file}`",
        "",
        "**Extracted signals:**",
        "| Signal | Value | Interpretation |",
        "|--------|-------|----------------|",
        (
            "| face_occupancy_ratio | "
            f"{_fmt_float(observation.get('face_occupancy_ratio'))} | "
            f"{_interp_face_occupancy(observation.get('face_occupancy_ratio'))} |"
        ),
        (
            "| estimated_yaw_degrees | "
            f"{yaw_value} | {_interp_yaw(observation.get('estimated_yaw_degrees'))} |"
        ),
        (
            "| background_complexity_score | "
            f"{_fmt_float(observation.get('background_complexity_score'))} | "
            f"{_interp_bg_complexity(observation.get('background_complexity_score'))} |"
        ),
        (
            "| lighting_uniformity_score | "
            f"{_fmt_float(observation.get('lighting_uniformity_score'))} | "
            f"{_interp_lighting(observation.get('lighting_uniformity_score'))} |"
        ),
        (
            "| estimated_sharpness | "
            f"{_fmt_float(observation.get('estimated_sharpness'))} | "
            f"{_interp_sharpness(observation.get('estimated_sharpness'))} |"
        ),
        (
            f"| occlusion_detected | {occlusion_value} | "
            f"{_interp_occlusion_bool(observation.get('occlusion_detected'))} |"
        ),
        "",
        "**Node 1 heuristic suggests:**",
        f"- Regime: `{suggested_regime}`",
        "- Risk factors:",
    ]

    for factor in suggested_risk_factors:
        lines.append(f"  - {factor}")

    prompt_issue_text = "none"
    if suggested_prompt_issues:
        prompt_issue_text = "; ".join(suggested_prompt_issues)
    lines.append(f"- Prompt issues: {prompt_issue_text}")

    lines.extend(
        [
            "",
            "**Your annotation (fill in):**",
            "```json",
            _json_block(annotation_template),
            "```",
            "",
        ]
    )

    return "\n".join(lines), suggested_regime


def _render_subenv2_case(case: dict[str, Any]) -> tuple[str, str]:
    case_id, source_file = _case_label(case)
    observation = case.get("observation") or {}
    ground_truth = case.get("ground_truth") or {}
    notes = ground_truth.get("_annotation_notes") or {}

    suggested_disposition = str(
        notes.get("suggested_disposition")
        or ground_truth.get("disposition")
        or "unknown"
    )
    suggested_confidence = _as_float(notes.get("suggested_confidence"))
    if suggested_confidence is None:
        suggested_confidence = _as_float(ground_truth.get("confidence"))
    if suggested_confidence is None:
        suggested_confidence = 0.8

    identity_drift_severity = str(notes.get("identity_drift_severity", "unknown"))
    lip_sync_quality = str(notes.get("lip_sync_quality", "unknown"))
    estimated_training_impact = str(notes.get("estimated_training_impact", "unknown"))
    primary_rejection_reason = notes.get("primary_rejection_reason")
    rejection_reason_text = "none" if primary_rejection_reason in (None, "") else str(primary_rejection_reason)

    annotation_template = {
        "disposition": (
            f"<{suggested_disposition} - confirm or change>"
            if suggested_disposition != "unknown"
            else "<accept|reject|fix|defer - confirm or change>"
        ),
        "confidence": round(float(suggested_confidence), 3),
        "disposition_ambiguity": 0.0,
        "valid_fix_steps": [],
        "valid_override_justifications": [],
        "expected_reasoning_elements": [
            "<keywords the agent's reasoning should mention>"
        ],
    }

    lines = [
        "---",
        f"### Case {case_id} - `{source_file}`",
        "",
        "**Extracted signals:**",
        "| Signal | Value | Interpretation |",
        "|--------|-------|----------------|",
        (
            "| identity_cosine_drift | "
            f"{_fmt_float(observation.get('identity_cosine_drift'))} | "
            f"{_interp_identity_drift(observation.get('identity_cosine_drift'))} |"
        ),
        (
            "| face_embedding_variance | "
            f"{_fmt_float(observation.get('face_embedding_variance'))} | "
            f"{_interp_embedding_variance(observation.get('face_embedding_variance'))} |"
        ),
        (
            "| landmark_stability_score | "
            f"{_fmt_float(observation.get('landmark_stability_score'))} | "
            f"{_interp_landmark_stability(observation.get('landmark_stability_score'))} |"
        ),
        (
            "| blur_score | "
            f"{_fmt_float(observation.get('blur_score'))} | "
            f"{_interp_blur(observation.get('blur_score'))} |"
        ),
        (
            "| exposure_score | "
            f"{_fmt_float(observation.get('exposure_score'))} | "
            f"{_interp_exposure(observation.get('exposure_score'))} |"
        ),
        (
            "| lip_sync_confidence | "
            f"{_fmt_float(observation.get('lip_sync_confidence'))} | "
            f"{_interp_lip_sync(observation.get('lip_sync_confidence'))} |"
        ),
        (
            "| phoneme_coverage_new | "
            f"{_fmt_float(observation.get('phoneme_coverage_new'))} | "
            f"{_interp_phoneme_novelty(observation.get('phoneme_coverage_new'))} |"
        ),
        (
            "| occlusion_frames | "
            f"{_fmt_int(observation.get('occlusion_frames'))} | "
            f"{_interp_occlusion_frames(observation.get('occlusion_frames'))} |"
        ),
        "",
        "**Node 5 heuristic suggests:**",
        f"- Disposition: `{suggested_disposition}`",
        f"- Confidence: {_fmt_float(suggested_confidence)}",
        f"- Identity drift severity: {identity_drift_severity}",
        f"- Lip sync quality: {lip_sync_quality}",
        f"- Training impact: {estimated_training_impact}",
        f"- Rejection reason: {rejection_reason_text}",
        "",
        "**Your annotation (fill in):**",
        "```json",
        _json_block(annotation_template),
        "```",
        "",
    ]

    return "\n".join(lines), suggested_disposition


def _render_subenv3_case(case: dict[str, Any]) -> tuple[str, str]:
    case_id, source_file = _case_label(case)
    source_stem = Path(source_file).stem if source_file and source_file != "unknown" else "unknown"

    observation = case.get("observation") or {}
    ground_truth = case.get("ground_truth") or {}
    notes = ground_truth.get("_annotation_notes") or {}

    lora_rank = observation.get("lora_rank")
    total_parameters = observation.get("total_parameters")
    overfitting_signature = observation.get("overfitting_signature")
    gradient_noise_estimate = observation.get("gradient_noise_estimate")

    canonical_entropy_mean = _extract_entropy_mean(observation)
    flagged_positions = _extract_flagged_positions(observation)
    token_mapping = _extract_token_mapping(observation)

    suggested_safety = str(
        notes.get("suggested_behavioral_safety")
        or ground_truth.get("model_behavioral_safety")
        or "unknown"
    )

    top_risk = notes.get("top_risk_phonemes")
    if isinstance(top_risk, list) and top_risk:
        top_risk_parts: list[str] = []
        for item in top_risk:
            if isinstance(item, dict):
                phoneme = str(item.get("phoneme", "?"))
                score = _as_float(item.get("risk_score"))
                if score is None:
                    top_risk_parts.append(phoneme)
                else:
                    top_risk_parts.append(f"{phoneme}({_fmt_float(score)})")
            else:
                top_risk_parts.append(str(item))
        top_risk_text = ", ".join(top_risk_parts)
    else:
        top_risk_text = "none"

    if flagged_positions:
        mapping_parts = [f"{pos}->{token_mapping.get(pos, '?')}" for pos in flagged_positions]
        mapping_text = ", ".join(mapping_parts)
    else:
        mapping_text = "none"

    step_match = re.search(r"(step_\d{5})", source_stem)
    step_key = step_match.group(1) if step_match else None
    detected_step = (
        f"{step_key}: {STEP_CONTEXT.get(step_key, 'not in predefined map')}"
        if step_key
        else "not detected from source file stem"
    )

    annotation_template = {
        "model_behavioral_safety": (
            f"<{suggested_safety} - confirm or change>"
            if suggested_safety != "unknown"
            else "<safe|minor_concerns|moderate_risk|high_risk|unsafe>"
        ),
        "phoneme_risk_ranking": [
            {
                "phoneme": "XX",
                "risk_score": 0.0,
                "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
                "confidence": 0.0,
                "evidence": "",
            }
        ],
        "predicted_behavior_triggers": [
            {
                "trigger_phoneme": "XX",
                "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
                "association_strength": 0.0,
                "is_intended": False,
                "concern_level": "low|medium|high",
            }
        ],
        "risky_phoneme_clusters": [],
        "valid_mitigation_set": [
            [
                "target phoneme or cluster",
                "add_counter_examples|flag_for_manual_review|retrain_with_more_data",
            ]
        ],
    }

    lines = [
        "---",
        f"### Case {case_id} - `{source_file}`",
        "",
        "**Extracted signals:**",
        "| Signal | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| lora_rank | {_fmt_int(lora_rank)} | - |",
        f"| total_parameters | {_fmt_int(total_parameters)} | - |",
        (
            "| overfitting_signature | "
            f"{_fmt_float(overfitting_signature)} | {_interp_overfitting(overfitting_signature)} |"
        ),
        (
            "| gradient_noise_estimate | "
            f"{_fmt_float(gradient_noise_estimate)} | "
            f"{_interp_gradient_noise(gradient_noise_estimate)} |"
        ),
        (
            "| canonical_entropy mean | "
            f"{_fmt_float(canonical_entropy_mean)} | {_interp_entropy_mean(canonical_entropy_mean)} |"
        ),
        (
            "| high_entropy_token_positions | "
            f"{json.dumps(flagged_positions, ensure_ascii=True)} | "
            "token positions with anomalous patterns |"
        ),
        "",
        "**Node 8 heuristic suggests:**",
        f"- Safety level: `{suggested_safety}`",
        f"- Top risk phonemes: {top_risk_text}",
        f"- Flagged token positions: {json.dumps(flagged_positions, ensure_ascii=True)}",
        f"- Token->phoneme mapping: {mapping_text}",
        "",
        "**Training step context:**",
        f"- Source stem: `{source_stem}`",
        f"- Detected step: {detected_step}",
        "- step_00250: very early - identity not yet established",
        "- step_00500: early - unstable identity",
        "- step_00750: mid - improving",
        "- step_01000: mid-late - approaching convergence",
        "- step_01250: good - identity preservation confirmed (your threshold)",
        "- step_01500+: converged - minor residual artifacts only",
        "",
        "**Your annotation (fill in):**",
        "```json",
        _json_block(annotation_template),
        "```",
        "",
    ]

    return "\n".join(lines), suggested_safety


def _render_summary(subenv: int, cases_count: int, distribution: Counter[str]) -> str:
    lines = [
        "## Summary",
        "",
        f"- Total cases: {cases_count}",
        f"- Suggested distribution: {_fmt_distribution(distribution)}",
        "- Fields always requiring manual input:",
    ]

    if subenv == 1:
        lines.append("  Sub-env 1: valid_prompt_modifications (always manual)")
    elif subenv == 2:
        lines.append("  Sub-env 2: expected_reasoning_elements, valid_fix_steps")
    else:
        lines.append(
            "  Sub-env 3: phoneme_risk_ranking, predicted_behavior_triggers, "
            "valid_mitigation_set"
        )

    return "\n".join(lines)


def _render_worksheet(subenv: int, cases: list[dict[str, Any]], cases_path: Path) -> str:
    lines = [
        f"# Annotation Worksheet - Sub-env {subenv}",
        "",
        f"Generated from `{cases_path.as_posix()}`",
        "",
    ]

    distribution: Counter[str] = Counter()

    for case in cases:
        if subenv == 1:
            section, label = _render_subenv1_case(case)
        elif subenv == 2:
            section, label = _render_subenv2_case(case)
        else:
            section, label = _render_subenv3_case(case)

        distribution[label] += 1
        lines.append(section)

    lines.append(_render_summary(subenv, len(cases), distribution))

    return "\n".join(lines).rstrip() + "\n"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"--cases must be an existing JSON file: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("Cases JSON must be an object with a 'cases' array")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("Cases JSON must contain a 'cases' array")

    validated: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise SystemExit(f"Case at index {idx} is not an object")
        validated.append(case)

    return validated


def main() -> int:
    args = parse_args()

    cases = _load_cases(args.cases)
    worksheet = _render_worksheet(args.subenv, cases, args.cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(worksheet, encoding="utf-8")

    print(f"Wrote {len(cases)} cases to {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
