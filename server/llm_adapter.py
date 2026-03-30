"""Plug-and-play LLM adapter for TalkingHeadBench ingestion bundles."""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.request
from typing import Any, Literal
from urllib.parse import urlparse

Provider = Literal["openai", "anthropic", "huggingface", "local"]

_SYSTEM_PROMPT = (
    "You are a TalkingHeadBench diagnostic assistant. "
    "You receive only pre-extracted signals from a reference image, dataset clips, and LoRA weights. "
    "Do not invent generation outputs. Keep recommendations directional and evidence-backed."
)


class LLMAdapterError(Exception):
    """Structured error surfaced by the one-shot analysis adapter."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def analyze_ingested_bundle(
    bundle: dict[str, Any],
    *,
    model_id: str | None,
    api_key: str | None,
    provider: str | None,
    base_url: str | None,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Generate a one-shot natural-language report from an ingested signal bundle."""
    if not isinstance(bundle, dict):
        raise LLMAdapterError(
            code="invalid_ingestion_bundle",
            message="The ingested bundle payload is invalid.",
            status_code=400,
            retryable=False,
        )

    resolved_provider = _resolve_provider(provider=provider, model_id=model_id, api_key=api_key)
    resolved_model = _resolve_model_id(resolved_provider, model_id)
    resolved_base_url = _resolve_custom_base_url(
        provider=resolved_provider,
        base_url=base_url,
    )
    signal_digest = _build_signal_digest(bundle)
    prompt = _build_prompt(signal_digest)

    if resolved_provider == "openai":
        report = _call_openai(
            model_id=resolved_model,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            base_url=resolved_base_url,
        )
    elif resolved_provider == "anthropic":
        report = _call_anthropic(
            model_id=resolved_model,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            base_url=resolved_base_url,
        )
    elif resolved_provider == "huggingface":
        report = _call_huggingface(
            model_id=resolved_model,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            base_url=resolved_base_url,
        )
    else:
        report = _call_local(
            model_id=resolved_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            base_url=resolved_base_url,
        )

    text = (report or "").strip()
    if not text:
        raise LLMAdapterError(
            code="empty_model_response",
            message="The model provider returned an empty analysis response.",
            status_code=502,
            retryable=True,
        )

    return {
        "provider": resolved_provider,
        "model_id": resolved_model,
        "report": text,
        "signal_digest": signal_digest,
    }


def _resolve_provider(*, provider: str | None, model_id: str | None, api_key: str | None) -> Provider:
    requested = (provider or "auto").strip().lower()
    if requested == "hf":
        requested = "huggingface"

    if requested not in {"auto", "openai", "anthropic", "huggingface", "local"}:
        raise LLMAdapterError(
            code="unsupported_provider",
            message=(
                "provider must be one of: auto, openai, anthropic, huggingface, local"
            ),
            status_code=400,
            retryable=False,
        )

    if requested != "auto":
        return requested  # type: ignore[return-value]

    key = (api_key or "").strip()
    model = (model_id or "").strip().lower()

    if key.startswith("sk-ant-") or model.startswith("claude"):
        return "anthropic"
    if key.startswith("hf_"):
        return "huggingface"
    if key.startswith("sk-") or model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "/" in model:
        return "huggingface"
    return "local"


def _resolve_model_id(provider: Provider, model_id: str | None) -> str:
    cleaned = (model_id or "").strip()
    if cleaned:
        return cleaned

    if provider == "openai":
        return os.getenv("THB_OPENAI_MODEL", "gpt-4o-mini")
    if provider == "anthropic":
        return os.getenv("THB_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    if provider == "huggingface":
        env_model = os.getenv("THB_HF_MODEL", "").strip()
        if env_model:
            return env_model
        raise LLMAdapterError(
            code="missing_model_id",
            message="Provide model_id for huggingface provider.",
            status_code=400,
            retryable=False,
        )
    return os.getenv("THB_LOCAL_MODEL", "llama3.1:8b-instruct-q4_K_M")


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _allowed_base_url_prefixes() -> list[str]:
    raw = os.getenv("THB_ALLOWED_BASE_URL_PREFIXES", "")
    prefixes = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return prefixes


def _resolve_custom_base_url(*, provider: Provider, base_url: str | None) -> str | None:
    custom = (base_url or "").strip()
    if not custom:
        return None

    if not _env_truthy("THB_ALLOW_CUSTOM_BASE_URLS", default=False):
        raise LLMAdapterError(
            code="custom_base_url_disabled",
            message="Custom base_url is disabled for this deployment.",
            status_code=400,
            retryable=False,
        )

    _validate_custom_base_url(custom, provider=provider)

    allowed_prefixes = _allowed_base_url_prefixes()
    if allowed_prefixes:
        normalized = custom.rstrip("/")
        if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
            raise LLMAdapterError(
                code="base_url_not_allowed",
                message="base_url is not in THB_ALLOWED_BASE_URL_PREFIXES.",
                status_code=400,
                retryable=False,
            )

    return custom


def _validate_custom_base_url(base_url: str, *, provider: Provider) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMAdapterError(
            code="invalid_base_url",
            message="base_url must be a valid http(s) URL.",
            status_code=400,
            retryable=False,
        )

    if parsed.username or parsed.password:
        raise LLMAdapterError(
            code="invalid_base_url",
            message="base_url must not contain embedded credentials.",
            status_code=400,
            retryable=False,
        )

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise LLMAdapterError(
            code="invalid_base_url",
            message="base_url must include a hostname.",
            status_code=400,
            retryable=False,
        )

    if host in {"localhost"} or host.endswith(".local"):
        raise LLMAdapterError(
            code="unsafe_base_url",
            message=(
                f"base_url host is not allowed for public deployment (provider={provider})."
            ),
            status_code=400,
            retryable=False,
        )

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
    ):
        raise LLMAdapterError(
            code="unsafe_base_url",
            message=(
                f"base_url host is not allowed for public deployment (provider={provider})."
            ),
            status_code=400,
            retryable=False,
        )


def _require_api_key(provider: str, api_key: str | None) -> str:
    key = (api_key or "").strip()
    if key:
        return key
    raise LLMAdapterError(
        code="missing_api_key",
        message=f"Provider '{provider}' requires api_key.",
        status_code=400,
        retryable=False,
    )


def _build_signal_digest(bundle: dict[str, Any]) -> dict[str, Any]:
    image_obs = _as_dict(bundle.get("image_observation"))
    clip_obs = _as_list_of_dicts(bundle.get("clip_signal_observations"))
    weight_obs = _as_dict(bundle.get("weight_observation"))
    metadata = _as_dict(bundle.get("ingestion_metadata"))
    extractor_metadata = _as_dict(metadata.get("extractor_metadata"))

    drifts = [_as_float(item.get("identity_cosine_drift")) for item in clip_obs]
    blurs = [_as_float(item.get("blur_score")) for item in clip_obs]
    lips = [_as_float(item.get("lip_sync_confidence")) for item in clip_obs]

    high_risk_clip_ids: list[str] = []
    for item in clip_obs:
        drift = _as_float(item.get("identity_cosine_drift"), default=0.0)
        blur = _as_float(item.get("blur_score"), default=1.0)
        lip_sync = _as_float(item.get("lip_sync_confidence"), default=1.0)
        if drift >= 0.35 or blur <= 0.30 or lip_sync <= 0.35:
            high_risk_clip_ids.append(str(item.get("clip_id", "unknown")))

    layer_entropy = _as_dict(weight_obs.get("canonical_entropy_per_layer"))
    rank_util = _as_dict(weight_obs.get("layer_rank_utilization"))
    high_entropy_positions = _as_list(weight_obs.get("high_entropy_token_positions"))
    token_map = weight_obs.get("token_position_to_phoneme")

    return {
        "case_id": bundle.get("case_id"),
        "prompt": str(bundle.get("prompt", "")),
        "param_config": _as_dict(bundle.get("param_config")),
        "image_summary": {
            "face_occupancy_ratio": _as_float(image_obs.get("face_occupancy_ratio")),
            "estimated_sharpness": _as_float(image_obs.get("estimated_sharpness")),
            "lighting_uniformity_score": _as_float(image_obs.get("lighting_uniformity_score")),
            "background_complexity_score": _as_float(
                image_obs.get("background_complexity_score")
            ),
            "conflicting_descriptors": _as_list(image_obs.get("conflicting_descriptors")),
            "identity_anchoring_strength": _as_float(
                image_obs.get("identity_anchoring_strength")
            ),
        },
        "clip_summary": {
            "clip_count": len(clip_obs),
            "high_risk_clip_ids": high_risk_clip_ids,
            "mean_identity_drift": _safe_mean(drifts),
            "mean_blur_score": _safe_mean(blurs),
            "mean_lip_sync_confidence": _safe_mean(lips),
        },
        "weight_summary": {
            "available": bool(weight_obs),
            "weight_file_id": str(weight_obs.get("weight_file_id", "")),
            "lora_rank": _as_int(weight_obs.get("lora_rank")),
            "target_module_count": len(_as_list(weight_obs.get("target_modules"))),
            "max_canonical_entropy": _dict_max(layer_entropy),
            "min_rank_utilization": _dict_min(rank_util),
            "high_entropy_token_positions": high_entropy_positions[:24],
            "suspected_anomalous_phonemes": _phonemes_from_positions(
                high_entropy_positions,
                token_map,
            ),
            "overfitting_signature": _as_float(weight_obs.get("overfitting_signature")),
        },
        "ingestion_metadata": {
            "created_at_unix": metadata.get("created_at_unix"),
            "clip_extractor_fallback_count": extractor_metadata.get(
                "clip_extractor_fallback_count"
            ),
        },
    }


def _build_prompt(signal_digest: dict[str, Any]) -> str:
    return (
        "Analyze this TalkingHeadBench ingestion bundle and provide a concise report with "
        "exact headings: Overall Readiness, Critical Risks, Parameter Fixes, "
        "Data and Weight Concerns, Top 3 Next Actions, Confidence. "
        "Each risk and recommendation must reference explicit evidence from the digest. "
        "Avoid absolute prescriptions; use directional fixes.\n\n"
        f"Signal digest (JSON):\n{json.dumps(signal_digest, indent=2, sort_keys=True)}"
    )


def _call_openai(
    *,
    model_id: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    base_url: str | None,
) -> str:
    key = _require_api_key("openai", api_key)
    url = _resolve_openai_url(base_url)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = _http_post_json(
        provider="openai",
        url=url,
        payload=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout_s=timeout_s,
    )

    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        raise LLMAdapterError(
            code="invalid_provider_response",
            message="OpenAI response did not include choices.",
            status_code=502,
            retryable=True,
        )
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    return _extract_text_content(message.get("content") if isinstance(message, dict) else None)


def _call_anthropic(
    *,
    model_id: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    base_url: str | None,
) -> str:
    key = _require_api_key("anthropic", api_key)
    url = _resolve_anthropic_url(base_url)
    payload = {
        "model": model_id,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = _http_post_json(
        provider="anthropic",
        url=url,
        payload=payload,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        timeout_s=timeout_s,
    )

    content = response.get("content") if isinstance(response, dict) else None
    return _extract_text_content(content)


def _call_huggingface(
    *,
    model_id: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    base_url: str | None,
) -> str:
    # Use HuggingFace Inference API v2 (OpenAI-compatible chat completions).
    # The legacy /models/{model_id} pipeline endpoint is deprecated for modern models.
    token = (api_key or "").strip()

    if base_url:
        cleaned = base_url.rstrip("/")
        # Support explicit v1/chat/completions or v1 suffixes.
        if cleaned.endswith("/v1/chat/completions"):
            url = cleaned
        elif cleaned.endswith("/v1"):
            url = f"{cleaned}/chat/completions"
        else:
            url = f"{cleaned}/v1/chat/completions"
    else:
        # New HuggingFace Serverless Inference API v2 — OpenAI-compatible.
        url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = _http_post_json(
        provider="huggingface",
        url=url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )

    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        # Surface provider-side error message if present.
        if isinstance(response, dict) and "error" in response:
            err = response["error"]
            err_msg = err if isinstance(err, str) else (err.get("message") if isinstance(err, dict) else str(err))
            raise LLMAdapterError(
                code="provider_generation_error",
                message=f"Hugging Face provider returned an error: {err_msg}",
                status_code=502,
                retryable=True,
            )
        raise LLMAdapterError(
            code="invalid_provider_response",
            message="Hugging Face response did not include choices.",
            status_code=502,
            retryable=True,
        )
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    return _extract_text_content(message.get("content") if isinstance(message, dict) else None)


def _call_local(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    base_url: str | None,
) -> str:
    if base_url:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1/chat/completions") or cleaned.endswith("/v1"):
            url = cleaned if cleaned.endswith("/chat/completions") else f"{cleaned}/chat/completions"
            return _call_openai_compatible(
                provider_name="local",
                url=url,
                model_id=model_id,
                api_key=None,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )

    url = _resolve_local_url(base_url)
    payload = {
        "model": model_id,
        "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = _http_post_json(
        provider="local",
        url=url,
        payload=payload,
        headers={},
        timeout_s=timeout_s,
    )

    if not isinstance(response, dict):
        raise LLMAdapterError(
            code="invalid_provider_response",
            message="Local provider returned an invalid response payload.",
            status_code=502,
            retryable=True,
        )
    text = response.get("response")
    if not isinstance(text, str):
        raise LLMAdapterError(
            code="invalid_provider_response",
            message="Local provider response did not include text output.",
            status_code=502,
            retryable=True,
        )
    return text


def _call_openai_compatible(
    *,
    provider_name: str,
    url: str,
    model_id: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers: dict[str, str] = {}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = _http_post_json(
        provider=provider_name,
        url=url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        raise LLMAdapterError(
            code="invalid_provider_response",
            message="Provider response did not include choices.",
            status_code=502,
            retryable=True,
        )
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    return _extract_text_content(message.get("content") if isinstance(message, dict) else None)


def _resolve_openai_url(base_url: str | None) -> str:
    url = (base_url or os.getenv("THB_OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")).rstrip("/")
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return url


def _resolve_anthropic_url(base_url: str | None) -> str:
    url = (base_url or os.getenv("THB_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")).rstrip("/")
    if url.endswith("/v1"):
        return f"{url}/messages"
    return url


def _resolve_local_url(base_url: str | None) -> str:
    url = (base_url or os.getenv("THB_LOCAL_LLM_URL", "http://localhost:11434/api/generate")).rstrip("/")
    if url.endswith("/api/generate"):
        return url
    return f"{url}/api/generate"


def _http_post_json(
    *,
    provider: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> Any:
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        retryable = int(exc.code) >= 500 or int(exc.code) == 429
        message = _format_http_error(provider, int(exc.code))
        raise LLMAdapterError(
            code="provider_http_error",
            message=message,
            status_code=502,
            retryable=retryable,
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMAdapterError(
            code="provider_connection_error",
            message=f"Unable to reach {provider} provider endpoint.",
            status_code=502,
            retryable=True,
        ) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMAdapterError(
            code="provider_invalid_json",
            message=f"{provider} provider returned non-JSON output.",
            status_code=502,
            retryable=True,
        ) from exc


def _format_http_error(provider: str, status_code: int) -> str:
    if status_code in {401, 403}:
        return f"Authentication failed for {provider} provider."
    if status_code == 404:
        return f"Requested model or endpoint was not found for {provider}."
    if status_code == 429:
        return f"Rate limit exceeded for {provider} provider."
    if status_code >= 500:
        return f"{provider} provider is temporarily unavailable."
    return f"{provider} provider request failed."


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _safe_mean(values: list[float]) -> float:
    clean = [value for value in values if isinstance(value, (int, float))]
    if not clean:
        return 0.0
    return round(float(sum(clean) / len(clean)), 4)


def _dict_max(values: dict[str, Any]) -> float:
    if not values:
        return 0.0
    return max(_as_float(v) for v in values.values())


def _dict_min(values: dict[str, Any]) -> float:
    if not values:
        return 0.0
    return min(_as_float(v) for v in values.values())


def _phonemes_from_positions(
    positions: list[Any],
    token_position_to_phoneme: Any,
) -> list[str]:
    if not isinstance(token_position_to_phoneme, dict):
        return []

    mapped: list[str] = []
    for position in positions:
        idx = _as_int(position, default=-1)
        if idx < 0:
            continue
        key = str(idx)
        value = token_position_to_phoneme.get(key)
        if value is None and idx in token_position_to_phoneme:
            value = token_position_to_phoneme.get(idx)
        if isinstance(value, str) and value not in mapped:
            mapped.append(value)
        if len(mapped) >= 16:
            break
    return mapped