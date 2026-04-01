"""Unit tests for one-shot LLM adapter routing and validation."""

from __future__ import annotations

import pytest

import server.llm_adapter as adapter


def _sample_digest() -> dict:
    return {
        "case_id": "ing-1",
        "prompt": "frontal portrait",
        "param_config": {"cfg": 7.5},
        "image_summary": {
            "face_occupancy_ratio": 0.42,
            "estimated_sharpness": 0.71,
            "lighting_uniformity_score": 0.8,
            "background_complexity_score": 0.2,
            "conflicting_descriptors": [],
            "identity_anchoring_strength": 0.9,
        },
        "clip_summary": {
            "clip_count": 2,
            "high_risk_clip_ids": ["clip-1"],
            "mean_identity_drift": 0.25,
            "mean_blur_score": 0.4,
            "mean_lip_sync_confidence": 0.5,
        },
        "weight_summary": {
            "available": False,
            "weight_file_id": "",
            "lora_rank": 0,
            "target_module_count": 0,
            "max_canonical_entropy": 0.0,
            "min_rank_utilization": 0.0,
            "high_entropy_token_positions": [],
            "suspected_anomalous_phonemes": [],
            "overfitting_signature": 0.0,
        },
        "ingestion_metadata": {
            "created_at_unix": 1,
            "clip_extractor_fallback_count": 0,
        },
    }


def test_build_signal_digest_omits_absent_artifact_sections():
    digest = adapter._build_signal_digest(
        {
            "case_id": "ing-empty",
            "prompt": "",
            "param_config": {},
            "ingestion_metadata": {},
        }
    )

    assert digest["case_id"] == "ing-empty"
    assert "image_summary" not in digest
    assert "clip_summary" not in digest
    assert "weight_summary" not in digest


def test_resolve_provider_from_openai_key():
    assert (
        adapter._resolve_provider(provider="auto", model_id=None, api_key="sk-test") == "openai"
    )


def test_resolve_provider_from_huggingface_model_id():
    assert (
        adapter._resolve_provider(
            provider=None,
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            api_key=None,
        )
        == "huggingface"
    )


def test_analyze_ingested_bundle_local_success(monkeypatch):
    monkeypatch.setattr(adapter, "_call_local", lambda **kwargs: "Overall Readiness\nStable")

    result = adapter.analyze_ingested_bundle(
        {
            "case_id": "ing-1",
            "prompt": "frontal reference",
            "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
        },
        model_id="llama3.1",
        api_key=None,
        provider="local",
        base_url=None,
        max_tokens=256,
        temperature=0.2,
        timeout_s=30.0,
    )

    assert result["provider"] == "local"
    assert result["model_id"] == "llama3.1"
    assert "Overall Readiness" in result["report"]
    assert result["signal_digest"]["case_id"] == "ing-1"


def test_analyze_ingested_bundle_openai_missing_key_raises():
    with pytest.raises(adapter.LLMAdapterError) as exc:
        adapter.analyze_ingested_bundle(
            {"case_id": "ing-2"},
            model_id="gpt-4o-mini",
            api_key=None,
            provider="openai",
            base_url=None,
            max_tokens=256,
            temperature=0.2,
            timeout_s=30.0,
        )

    assert exc.value.code == "missing_api_key"


def test_analyze_ingested_bundle_rejects_custom_base_url_by_default(monkeypatch):
    monkeypatch.delenv("THB_ALLOW_CUSTOM_BASE_URLS", raising=False)

    with pytest.raises(adapter.LLMAdapterError) as exc:
        adapter.analyze_ingested_bundle(
            {"case_id": "ing-3"},
            model_id="llama3.1",
            api_key=None,
            provider="local",
            base_url="https://api.openai.com/v1",
            max_tokens=128,
            temperature=0.2,
            timeout_s=20.0,
        )

    assert exc.value.code == "custom_base_url_disabled"


def test_analyze_ingested_bundle_rejects_private_custom_base_url(monkeypatch):
    monkeypatch.setenv("THB_ALLOW_CUSTOM_BASE_URLS", "1")

    with pytest.raises(adapter.LLMAdapterError) as exc:
        adapter.analyze_ingested_bundle(
            {"case_id": "ing-4"},
            model_id="llama3.1",
            api_key=None,
            provider="local",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=128,
            temperature=0.2,
            timeout_s=20.0,
        )

    assert exc.value.code == "unsafe_base_url"


def test_analyze_ingested_bundle_allows_custom_base_url_when_allowlisted(monkeypatch):
    monkeypatch.setenv("THB_ALLOW_CUSTOM_BASE_URLS", "1")
    monkeypatch.setenv(
        "THB_ALLOWED_BASE_URL_PREFIXES",
        "https://api-inference.huggingface.co",
    )
    monkeypatch.setattr(adapter, "_call_huggingface", lambda **kwargs: "Overall Readiness\nStable")

    result = adapter.analyze_ingested_bundle(
        {"case_id": "ing-5", "param_config": {}},
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        api_key="hf_test",
        provider="huggingface",
        base_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=128,
        temperature=0.2,
        timeout_s=20.0,
    )

    assert result["provider"] == "huggingface"
    assert "Overall Readiness" in result["report"]


def test_build_prompt_image_audit_scopes_to_image_and_params():
    prompt = adapter._build_prompt(_sample_digest(), task_tier="image_audit")

    assert "### Data and Weight Concerns" not in prompt
    assert "### Clip and Weight Status" in prompt
    assert "image_audit mode" in prompt
    assert "image_summary fields only when image_summary is present" in prompt
    assert "param_config anomalies only" in prompt


def test_build_prompt_clip_audit_scopes_to_clips_only():
    prompt = adapter._build_prompt(_sample_digest(), task_tier="clip_audit")

    assert "### Data and Weight Concerns" in prompt
    assert "### Weight Status" in prompt
    assert "clip_summary is available for analysis" in prompt
    assert "clip_audit mode" in prompt
    assert "mean_identity_drift" in prompt
    assert "mean_blur_score" in prompt
    assert "mean_lip_sync_confidence" in prompt
    assert "high_risk_clip_ids" in prompt
    assert "clip_extractor_fallback_count" in prompt
    assert "Omit all weight_summary interpretation for this tier." in prompt


def test_build_prompt_weight_audit_keeps_all_headings():
    prompt = adapter._build_prompt(_sample_digest(), task_tier="weight_audit")

    assert "### Overall Readiness" in prompt
    assert "### Critical Risks" in prompt
    assert "### Parameter Fixes" in prompt
    assert "### Data and Weight Concerns" in prompt
    assert "### Weight Status" in prompt
    assert "### Top 3 Next Actions" in prompt
    assert "### Confidence" in prompt
    assert "weight_summary interpretation only when weight_summary is present" in prompt
    assert "state 'not uploaded - no data'" in prompt