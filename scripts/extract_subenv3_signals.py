#!/usr/bin/env python3
"""Extract Sub-env 3 signals and append annotation-ready cases.

This script runs the deterministic Sub-env 3 extraction path:
1. Ensure a token-position-to-phoneme mapping exists.
2. Run Node 7 weight extraction (canonical statistics).
3. Build deterministic WeightEvidenceDossier and Node 8 observation.
4. Run Node 8 risk assessment.
5. Append or merge a case into the target Sub-env 3 test-set JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.envs.subenv3.node8_phoneme_risk import assess_phoneme_risk
from src.schemas.ground_truth import GroundTruthBehavioralAnnotation
from src.schemas.subenv3 import (
    LayerAnomalyFlag,
    PhonemeRiskObservation,
    TokenAnomalyFlag,
    WeightEvidenceDossier,
    WeightSignalObservation,
)

DEFAULT_TOKENIZER_CONFIG = PROJECT_ROOT / "data" / "tokenizer_config.json"

HF_TOKENIZER_CANDIDATES = [
    "bookbot/wav2vec2-ljspeech-gruut",
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-base-960h",
]

ARPABET_BASE = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
ARPABET_SET = set(ARPABET_BASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Sub-env 3 canonical weight and phoneme risk signals, then "
            "append or merge an annotation-ready case into a JSON test set."
        )
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to a LoRA .safetensors file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Target Sub-env 3 JSON (for example tests/test_set/subenv3_cases.json)",
    )
    parser.add_argument(
        "--tokenizer-config",
        required=False,
        type=Path,
        default=DEFAULT_TOKENIZER_CONFIG,
        help=(
            "Optional tokenizer config path with token_position_to_phoneme. "
            "If missing, mapping is reconstructed and saved here."
        ),
    )
    return parser.parse_args()


def _normalize_to_arpabet(token: str) -> str | None:
    t = token.strip().upper()
    if not t:
        return None

    # Remove common tokenizer wrappers.
    t = t.replace("\u2581", "").replace("|", "")
    t = t.strip("<>[]{}")
    if not t:
        return None

    # Remove stress suffixes, for example AH0 -> AH.
    t = re.sub(r"[012]$", "", t)

    if t in ARPABET_SET:
        return t
    return None


def build_arpabet_char_mapping() -> dict[int, str]:
    """Build a stable sequential ARPAbet mapping for 39 phonemes."""
    return {idx: phoneme for idx, phoneme in enumerate(ARPABET_BASE)}


def _mapped_phoneme_count(token_map: dict[int, str]) -> int:
    return len(set(token_map.values()))


def _reconstruct_from_wav2vec2_hub() -> tuple[dict[int, str] | None, str]:
    try:
        import transformers  # type: ignore
    except Exception:
        return None, "transformers_not_available"

    loader_names = ["Wav2Vec2PhonemeCTCTokenizer", "Wav2Vec2CTCTokenizer", "AutoTokenizer"]

    for model_id in HF_TOKENIZER_CANDIDATES:
        for loader_name in loader_names:
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue

            try:
                tokenizer = loader.from_pretrained(model_id)
            except Exception:
                continue

            try:
                vocab = tokenizer.get_vocab()
            except Exception:
                continue

            token_map: dict[int, str] = {}
            for token, idx in vocab.items():
                normalized = _normalize_to_arpabet(token)
                if normalized is None:
                    continue
                token_map[int(idx)] = normalized

            # Require enough coverage to trust a hub-derived map.
            if len(set(token_map.values())) >= 10:
                return dict(sorted(token_map.items())), f"hub:{model_id}/{loader_name}"

    return None, "hub_candidates_failed"


def ensure_tokenizer_config(path: Path) -> tuple[Path, str]:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_map = raw.get("token_position_to_phoneme", {})
        if isinstance(raw_map, dict) and raw_map:
            existing = {int(k): str(v) for k, v in raw_map.items()}
            if _mapped_phoneme_count(existing) >= 30:
                source = str(raw.get("source", "existing"))
                return path, source

            print(
                f"  Existing mapping only covered {_mapped_phoneme_count(existing)} phonemes - "
                "insufficient for full ARPAbet coverage."
            )
            print("  Rebuilding mapping.")

    reconstructed, source = _reconstruct_from_wav2vec2_hub()
    if reconstructed is None:
        reconstructed = build_arpabet_char_mapping()
        source = "arpabet_sequential"
    elif _mapped_phoneme_count(reconstructed) < 30:
        print(
            f"  Hub mapping only covered {_mapped_phoneme_count(reconstructed)} phonemes - "
            "insufficient for full ARPAbet coverage."
        )
        print("  Falling back to sequential ARPAbet mapping (39 phonemes).")
        reconstructed = build_arpabet_char_mapping()
        source = "arpabet_sequential"

    payload = {
        "source": source,
        "token_position_to_phoneme": {str(k): v for k, v in reconstructed.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, source


def build_weight_evidence(obs: WeightSignalObservation) -> WeightEvidenceDossier:
    mean_util = (
        float(sum(obs.layer_rank_utilization.values()) / max(len(obs.layer_rank_utilization), 1))
        if obs.layer_rank_utilization
        else 0.5
    )

    if obs.overfitting_signature >= 0.6:
        training_quality = "overfit"
    elif obs.gradient_noise_estimate >= 0.5:
        training_quality = "unstable"
    elif mean_util <= 0.3:
        training_quality = "underfit"
    else:
        training_quality = "healthy"

    if mean_util <= 0.3:
        rank_assessment = "collapsed"
    elif mean_util <= 0.65:
        rank_assessment = "wasteful"
    else:
        rank_assessment = "efficient"

    sig = obs.overfitting_signature
    if sig >= 0.8:
        overall_risk = "critical"
    elif sig >= 0.6:
        overall_risk = "high"
    elif sig >= 0.3:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    token_map = obs.token_position_to_phoneme or {}
    mean_entropy = (
        float(sum(obs.canonical_entropy_per_layer.values()) /
              max(len(obs.canonical_entropy_per_layer), 1))
        if obs.canonical_entropy_per_layer
        else 0.0
    )
    token_flags = [
        TokenAnomalyFlag(
            token_position=pos,
            mapped_phoneme=token_map.get(pos),
            anomaly_type="excessive_influence",
            severity=min(1.0, mean_entropy * 1.5),
            evidence=(
                f"Token position {pos} flagged as high-entropy in canonical Vt analysis; "
                f"layer entropy mean {mean_entropy:.3f}"
            ),
        )
        for pos in obs.high_entropy_token_positions[:10]
    ]

    layer_flags: list[LayerAnomalyFlag] = []
    mean_norm = (
        float(sum(obs.layer_norms.values()) / max(len(obs.layer_norms), 1))
        if obs.layer_norms
        else 0.0
    )
    for layer_name, sparsity in obs.layer_sparsity.items():
        norm = obs.layer_norms.get(layer_name, 0.0)
        if sparsity > 0.5:
            layer_flags.append(
                LayerAnomalyFlag(
                    layer_name=layer_name,
                    anomaly_type="sparsity_anomaly",
                    severity=min(1.0, sparsity),
                    evidence=f"Sparsity {sparsity:.3f} - majority of canonical S near zero",
                )
            )
        elif mean_norm > 0 and norm > 3.0 * mean_norm:
            layer_flags.append(
                LayerAnomalyFlag(
                    layer_name=layer_name,
                    anomaly_type="norm_explosion",
                    severity=min(1.0, norm / (3.0 * mean_norm + 1e-8)),
                    evidence=(
                        f"Norm {norm:.3f} is {norm / (mean_norm + 1e-8):.1f}x "
                        f"the layer mean ({mean_norm:.3f})"
                    ),
                )
            )

    evidence_summary = (
        f"Training quality: {training_quality}. "
        f"Rank utilization: {rank_assessment} (mean {mean_util:.2f}). "
        f"{len(token_flags)} high-entropy token position(s) flagged. "
        f"{len(layer_flags)} layer anomaly/anomalies detected. "
        f"Overall risk: {overall_risk}."
    )

    return WeightEvidenceDossier(
        weight_file_id=obs.weight_file_id,
        training_quality=training_quality,
        rank_utilization_assessment=rank_assessment,
        high_entropy_token_flags=token_flags,
        layer_anomaly_flags=layer_flags,
        overall_behavioral_risk=overall_risk,
        evidence_summary=evidence_summary,
    )


def build_phoneme_risk_observation(
    obs: WeightSignalObservation,
    dossier: WeightEvidenceDossier,
) -> PhonemeRiskObservation:
    token_map: dict[int, str] = obs.token_position_to_phoneme or {}

    phoneme_to_token_indices: dict[str, list[int]] = {}
    for pos, phoneme in token_map.items():
        phoneme_to_token_indices.setdefault(phoneme, []).append(pos)

    phoneme_vocabulary = sorted(set(token_map.values()))
    high_entropy_positions = set(obs.high_entropy_token_positions)

    canonical_entropy = obs.canonical_entropy_per_layer
    attn_layers = {k: v for k, v in canonical_entropy.items() if "attn" in k.lower()}
    if not attn_layers:
        attn_layers = canonical_entropy

    mean_attn_entropy = (
        float(sum(attn_layers.values()) / len(attn_layers)) if attn_layers else 0.0
    )

    phoneme_entropy_scores: dict[str, float] = {}
    phoneme_influence_scores: dict[str, float] = {}
    for phoneme, positions in phoneme_to_token_indices.items():
        if not positions:
            phoneme_entropy_scores[phoneme] = 0.0
            phoneme_influence_scores[phoneme] = 0.0
            continue

        flagged = set(positions) & high_entropy_positions
        high_frac = len(flagged) / len(positions)
        phoneme_entropy_scores[phoneme] = float(high_frac)
        phoneme_influence_scores[phoneme] = float(
            min(1.0, mean_attn_entropy * (1.0 + high_frac * 0.5))
        )

    return PhonemeRiskObservation(
        weight_evidence=dossier,
        high_entropy_token_flags=dossier.high_entropy_token_flags,
        phoneme_vocabulary=phoneme_vocabulary,
        phoneme_to_token_indices=phoneme_to_token_indices,
        phoneme_entropy_scores=phoneme_entropy_scores,
        phoneme_influence_scores=phoneme_influence_scores,
        phoneme_cooccurrence_anomalies=[],
        behavior_vocabulary=["smile", "blink", "head_turn", "jaw_drift", "brow_raise"],
        training_data_phoneme_distribution=None,
        suspected_anomalous_phonemes_from_subenv2=obs.suspected_anomalous_phonemes,
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(raw).__name__}")

    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected 'cases' list in {path}")
    return cases


def _next_case_id(cases: list[dict[str, Any]]) -> str:
    numeric_ids: list[int] = []
    for case in cases:
        raw_id = str(case.get("id", "")).strip()
        if raw_id.isdigit():
            numeric_ids.append(int(raw_id))

    next_id = (max(numeric_ids) + 1) if numeric_ids else 1
    return f"{next_id:03d}"


def _find_existing_case_index(
    cases: list[dict[str, Any]],
    weight_file_id: str,
    source_file: str,
) -> int | None:
    for idx, case in enumerate(cases):
        if case.get("source_file") == source_file:
            return idx

        obs = case.get("observation")
        if not isinstance(obs, dict):
            continue

        evidence = obs.get("weight_evidence")
        if isinstance(evidence, dict) and evidence.get("weight_file_id") == weight_file_id:
            return idx

    return None


def build_case_entry(
    case_id: str,
    source_file: str,
    obs: PhonemeRiskObservation,
    tokenizer_source: str,
) -> dict[str, Any]:
    action = assess_phoneme_risk(obs)

    top_risks = [
        {
            "phoneme": entry.phoneme,
            "risk_score": round(float(entry.risk_score), 4),
            "risk_type": entry.risk_type,
            "confidence": round(float(entry.confidence), 4),
        }
        for entry in action.phoneme_risk_ranking[:8]
    ]

    suggested_triggers = [
        {
            "trigger_phoneme": trig.trigger_phoneme,
            "triggered_behavior": trig.triggered_behavior,
            "association_strength": round(float(trig.association_strength), 4),
            "concern_level": trig.concern_level,
        }
        for trig in action.predicted_behavior_triggers
    ]

    gt = {
        "phoneme_risk_ranking": [],
        "predicted_behavior_triggers": [],
        "risky_phoneme_clusters": [],
        "model_behavioral_safety": "ANNOTATE",
        "valid_mitigation_set": [],
        "_annotation_notes": {
            "tokenizer_source": tokenizer_source,
            "suggested_behavioral_safety": action.model_behavioral_safety,
            "suggested_training_quality": obs.weight_evidence.training_quality,
            "suggested_rank_assessment": obs.weight_evidence.rank_utilization_assessment,
            "suggested_overall_behavioral_risk": obs.weight_evidence.overall_behavioral_risk,
            "high_entropy_token_count": len(obs.high_entropy_token_flags),
            "top_risk_phonemes": top_risks,
            "suggested_behavior_triggers": suggested_triggers,
            "suggested_mitigations": [
                {
                    "target": m.target,
                    "action": m.action,
                    "priority": m.priority,
                }
                for m in action.mitigation_recommendations
            ],
            "summary": action.summary,
        },
    }

    # Validate placeholder ground truth shape for evaluate --dry-run compatibility.
    GroundTruthBehavioralAnnotation(**gt)

    return {
        "id": case_id,
        "source_file": source_file,
        "observation": obs.model_dump(),
        "ground_truth": gt,
    }


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    payload = {"cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.weights.exists() or not args.weights.is_file():
        raise SystemExit(f"--weights must be an existing file: {args.weights}")

    tokenizer_path, tokenizer_source = ensure_tokenizer_config(args.tokenizer_config)
    tokenizer_format = "hub-derived" if tokenizer_source.startswith("hub:") else "fallback"
    print(f"Detected format: {tokenizer_format}")

    try:
        weight_signal_obs = extract_weight_signals(
            args.weights, tokenizer_config_path=tokenizer_path
        )
        weight_signal_obs = WeightSignalObservation(**weight_signal_obs.model_dump())
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Node 7 extraction failed: {exc}") from exc

    obs = weight_signal_obs
    entropy_vals = list(obs.canonical_entropy_per_layer.values())
    entropy_mean = sum(entropy_vals) / len(entropy_vals)
    entropy_std = (sum((v - entropy_mean) ** 2 for v in entropy_vals) / len(entropy_vals)) ** 0.5
    print("  canonical_entropy_per_layer:")
    print(f"    count : {len(entropy_vals)}")
    print(f"    min   : {min(entropy_vals):.6f}")
    print(f"    max   : {max(entropy_vals):.6f}")
    print(f"    mean  : {entropy_mean:.6f}")
    print(f"    std   : {entropy_std:.6f}")

    dossier = build_weight_evidence(weight_signal_obs)

    try:
        phoneme_obs = build_phoneme_risk_observation(weight_signal_obs, dossier)
        phoneme_obs = PhonemeRiskObservation(**phoneme_obs.model_dump())
    except ValidationError as exc:
        raise SystemExit(f"Node 8 observation validation failed: {exc}") from exc

    pe_vals = list(phoneme_obs.phoneme_entropy_scores.values())
    pi_vals = list(phoneme_obs.phoneme_influence_scores.values())
    if pe_vals:
        print("  phoneme_entropy_scores:")
        print(f"    count : {len(pe_vals)}")
        print(f"    min   : {min(pe_vals):.6f}")
        print(f"    max   : {max(pe_vals):.6f}")
        print(f"    mean  : {sum(pe_vals)/len(pe_vals):.6f}")
    if pi_vals:
        print("  phoneme_influence_scores:")
        print(f"    count : {len(pi_vals)}")
        print(f"    min   : {min(pi_vals):.6f}")
        print(f"    max   : {max(pi_vals):.6f}")
        print(f"    mean  : {sum(pi_vals)/len(pi_vals):.6f}")

    print(f"  overfitting_signature : {obs.overfitting_signature:.4f}")
    print(f"  gradient_noise_estimate: {obs.gradient_noise_estimate:.4f}")
    print(f"  lora_rank             : {obs.lora_rank}")
    print(f"  total_parameters      : {obs.total_parameters}")
    print(f"  layers with adapters  : {len(obs.layer_norms)}")

    existing_cases = load_cases(args.output)
    existing_idx = _find_existing_case_index(
        existing_cases,
        weight_file_id=weight_signal_obs.weight_file_id,
        source_file=args.weights.name,
    )

    if existing_idx is not None:
        case_id = str(existing_cases[existing_idx].get("id", _next_case_id(existing_cases)))
        new_case = build_case_entry(case_id, args.weights.name, phoneme_obs, tokenizer_source)
        existing_cases[existing_idx] = new_case
        action_label = "merged"
    else:
        case_id = _next_case_id(existing_cases)
        new_case = build_case_entry(case_id, args.weights.name, phoneme_obs, tokenizer_source)
        existing_cases.append(new_case)
        action_label = "appended"

    write_cases(args.output, existing_cases)

    action_preview = assess_phoneme_risk(phoneme_obs)
    print(f"Tokenizer config: {tokenizer_path} ({tokenizer_source})")
    print(f"Weight file: {weight_signal_obs.weight_file_id}")
    print(f"Case {case_id} {action_label} in {args.output}")
    print(f"Model behavioral safety suggestion: {action_preview.model_behavioral_safety}")
    print(f"Flagged phonemes: {len(action_preview.phoneme_risk_ranking)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())