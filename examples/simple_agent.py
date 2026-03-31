#!/usr/bin/env python3
"""
TalkingHeadBench - Example Agent

Demonstrates a full episode loop using an LLM to generate diagnostic
actions for each sub-environment step. This script connects to a running
TalkingHeadBench server and walks through the complete pipeline:

  reset()  -> ImageDiagnosticsObservation  -> agent produces ImageDiagnosticsAction
  step(1)  -> ParamAnomalyObservation      -> agent produces ParamAnomalyAction
  step(2)  -> PhonemeRiskObservation       -> agent produces PhonemeRiskAction
  step(3)  -> done=True, final_score

Usage:
    # Start the server first:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Then run the agent:
    python examples/simple_agent.py \
        --base-url http://localhost:8000 \
        --model-id meta-llama/Llama-3.1-70B-Instruct \
        --api-key hf_YOUR_TOKEN_HERE
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import httpx
from pydantic import ValidationError

from client import TalkingHeadBenchEnv
from models import ImageDiagnosticsAction, ParamAnomalyAction, PhonemeRiskAction

HF_INFERENCE_CHAT_URL = (
    "https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"
)

SYSTEM_PROMPT = """You are a senior engineer reviewer for TalkingHeadBench.
You must output only strict JSON that matches the expected action schema.
You must NEVER output numerical parameter prescriptions (example forbidden: \"set CFG = 6.8\").
You may only provide directional recommendations (increase, decrease, enable, disable, reconsider).
Keep recommendations diagnostic, concrete, and grounded in the provided signals."""

ACTION_SCHEMA_SUMMARY = """Step 0 (ImageDiagnosticsAction):
  - regime_classification: one of \"frontal_simple\", \"non_frontal\", \"complex_background\", \"occluded\", \"low_quality\"
  - identified_risk_factors: list of strings
  - prompt_issues: list of strings
  - recommended_prompt_modifications: list of strings
  - image_usability_score: float 0-1
  - reasoning: string

Step 1 (ParamAnomalyAction):
  - config_risk_level: one of \"safe\", \"marginal\", \"risky\", \"dangerous\"
  - anomalies: list of {parameter, issue, severity, linked_failure_mode}
  - predicted_failure_modes: list of Literal strings
  - directional_fixes: list of {target, direction, rationale, priority}
  - summary: string

Step 2 (PhonemeRiskAction):
  - phoneme_risk_ranking: list of {phoneme, risk_score, risk_type, confidence, evidence}
  - predicted_behavior_triggers: list of {trigger_phoneme, triggered_behavior, association_strength, is_intended, concern_level}
  - risky_phoneme_clusters: list of {phonemes, cluster_risk_type, combined_risk_score, interaction_description}
  - model_behavioral_safety: one of \"safe\", \"minor_concerns\", \"moderate_risk\", \"high_risk\", \"unsafe\"
  - mitigation_recommendations: list of {target, action, rationale, priority}
  - summary: string
"""

SCHEMA_MODELS = {
    "ImageDiagnosticsAction": ImageDiagnosticsAction,
    "ParamAnomalyAction": ParamAnomalyAction,
    "PhonemeRiskAction": PhonemeRiskAction,
}

DEFAULT_STEP_SCHEMAS = [
    "ImageDiagnosticsAction",
    "ParamAnomalyAction",
    "PhonemeRiskAction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple TalkingHeadBench LLM agent.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the TalkingHeadBench OpenEnv server.",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model ID routed through HuggingFace Inference API.",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        help="HuggingFace API key (or set HF_API_KEY / HUGGINGFACEHUB_API_TOKEN).",
    )
    return parser.parse_args()


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return {
            k: v for k, v in vars(value).items() if not k.startswith("_")
        }
    return {"value": str(value)}


def unpack_step_result(result: Any) -> tuple[dict[str, Any], bool, float | None, dict[str, Any] | None]:
    raw_observation = getattr(result, "observation", result)
    obs = _coerce_to_dict(raw_observation)

    done = bool(getattr(result, "done", obs.get("done", False)))
    reward = getattr(result, "reward", obs.get("reward"))
    scores = getattr(result, "scores", obs.get("scores"))

    if not isinstance(scores, dict):
        scores = _coerce_to_dict(scores) if scores is not None else None

    if not obs.get("node") and hasattr(raw_observation, "node"):
        obs["node"] = getattr(raw_observation, "node")
    if obs.get("scores") is None and scores:
        obs["scores"] = scores

    return obs, done, reward, scores


def expected_schema_name(step_index: int, observation: dict[str, Any]) -> str:
    schema = observation.get("expected_action_schema")
    if isinstance(schema, str) and schema:
        return schema

    if hasattr(TalkingHeadBenchEnv, "expected_action_schema"):
        guessed = TalkingHeadBenchEnv.expected_action_schema(step_index)
        if guessed and guessed != "unknown":
            return guessed

    if 0 <= step_index < len(DEFAULT_STEP_SCHEMAS):
        return DEFAULT_STEP_SCHEMAS[step_index]
    return "unknown"


def minimal_action(step_index: int) -> dict[str, Any]:
    if hasattr(TalkingHeadBenchEnv, "make_minimal_action"):
        candidate = TalkingHeadBenchEnv.make_minimal_action(step_index)
        if isinstance(candidate, dict):
            return candidate

    if step_index == 0:
        return {
            "regime_classification": "frontal_simple",
            "identified_risk_factors": [],
            "prompt_issues": [],
            "recommended_prompt_modifications": [],
            "image_usability_score": 0.5,
            "reasoning": "Minimal fallback action.",
        }
    if step_index == 1:
        return {
            "config_risk_level": "marginal",
            "anomalies": [],
            "predicted_failure_modes": [],
            "directional_fixes": [],
            "summary": "Minimal fallback action.",
        }
    if step_index == 2:
        return {
            "phoneme_risk_ranking": [],
            "predicted_behavior_triggers": [],
            "risky_phoneme_clusters": [],
            "model_behavioral_safety": "minor_concerns",
            "mitigation_recommendations": [],
            "summary": "Minimal fallback action.",
        }
    return {}


def _extract_json_candidate(text: str) -> dict[str, Any]:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\\s*", "", stripped)
        stripped = re.sub(r"\\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON is not an object.")
    return payload


def validate_action_payload(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_cls = SCHEMA_MODELS.get(schema_name)
    if model_cls is None:
        return payload
    return model_cls.model_validate(payload).model_dump()


def call_hf_chat_completion(
    *,
    model_id: str,
    api_key: str,
    messages: list[dict[str, str]],
) -> str:
    url = HF_INFERENCE_CHAT_URL.format(model_id=model_id)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM response did not include any choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()

    return str(content).strip()


def build_user_prompt(
    *,
    step_index: int,
    schema_name: str,
    observation: dict[str, Any],
) -> str:
    instruction = observation.get("instruction", "")
    node = observation.get("node", "unknown")
    signals = observation.get("signals", {})

    return (
        f"Environment step index: {step_index}\\n"
        f"Current node: {node}\\n"
        f"Expected action schema: {schema_name}\\n\\n"
        f"Action schema summary:\\n{ACTION_SCHEMA_SUMMARY}\\n"
        f"Instruction:\\n{instruction}\\n\\n"
        f"Signals (JSON):\\n{json.dumps(signals, indent=2, sort_keys=True)}\\n\\n"
        f"Return only one JSON object that satisfies {schema_name}."
    )


def generate_action(
    *,
    step_index: int,
    schema_name: str,
    observation: dict[str, Any],
    model_id: str,
    api_key: str,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(
        step_index=step_index,
        schema_name=schema_name,
        observation=observation,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = "unknown"
    last_raw = ""

    for attempt in range(2):
        raw = call_hf_chat_completion(
            model_id=model_id,
            api_key=api_key,
            messages=messages,
        )
        last_raw = raw

        try:
            parsed = _extract_json_candidate(raw)
            return validate_action_payload(schema_name, parsed)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt == 0:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous output was invalid. "
                                f"Error: {last_error}. "
                                f"Return only corrected JSON for {schema_name}. "
                                "Do not include markdown fences or extra text."
                            ),
                        },
                    ]
                )

    print(
        "Warning: LLM JSON parsing failed after retry. "
        f"Using minimal fallback action for step {step_index}. "
        f"Last error: {last_error}"
    )
    if last_raw:
        print("Raw model output:")
        print(last_raw)
    return minimal_action(step_index)


def run_episode(base_url: str, model_id: str, api_key: str) -> None:
    with TalkingHeadBenchEnv(base_url=base_url).sync() as env:
        result = env.reset()
        step_index = 0

        while True:
            observation, done, reward, scores = unpack_step_result(result)
            node = observation.get("node", "unknown")
            schema_name = expected_schema_name(step_index, observation)

            print(f"\\nStep {step_index} | node={node} | expected={schema_name}")

            if done:
                print("Episode complete.")
                if scores:
                    print("Score breakdown:")
                    print(json.dumps(scores, indent=2, sort_keys=True))
                if reward is not None:
                    print(f"Final reward: {float(reward):.4f}")
                return

            action = generate_action(
                step_index=step_index,
                schema_name=schema_name,
                observation=observation,
                model_id=model_id,
                api_key=api_key,
            )

            print("Action payload:")
            print(json.dumps(action, indent=2, sort_keys=True))

            result = env.step(action)
            step_index += 1

            if step_index > 8:
                raise RuntimeError("Episode exceeded expected step count.")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing HuggingFace API key. Pass --api-key or set HF_API_KEY."
        )

    run_episode(
        base_url=args.base_url,
        model_id=args.model_id,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
