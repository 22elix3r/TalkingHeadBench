"""Smoke tests for artifact ingestion API contracts and error semantics."""

from __future__ import annotations

from fastapi.testclient import TestClient

import server.app as app_module
from server.llm_adapter import LLMAdapterError


client = TestClient(app_module.app)


async def _fake_bundle(**kwargs):
    return {
        "case_id": "case-123",
        "prompt": kwargs.get("prompt", ""),
        "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
        "source_files": {"reference_image": "reference_image_abc123.png"},
        "ingestion_metadata": {
            "api_version": "1.0",
            "created_at_unix": 1,
            "limits": {},
            "extractor_metadata": {"clip_extractor_fallback_count": 0},
        },
    }


def test_health_endpoint_returns_status():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "healthy"}


def test_healthz_endpoint_returns_versioned_ok():
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["api_version"] == "1.0"


def test_ingest_openapi_clips_schema_is_binary_array():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    request_schema = (
        schema["paths"]["/ingest-artifacts"]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]
    )
    if "$ref" in request_schema:
        ref_name = request_schema["$ref"].rsplit("/", 1)[-1]
        body_schema = schema["components"]["schemas"][ref_name]
    else:
        body_schema = request_schema

    clips_schema = body_schema["properties"]["clips"]
    param_config_schema = body_schema["properties"]["param_config_json"]

    assert clips_schema["type"] == "array"
    assert clips_schema["items"]["type"] == "string"
    assert clips_schema["items"]["format"] == "binary"
    assert clips_schema["default"] == []
    assert param_config_schema["type"] == "string"
    assert param_config_schema["default"] == ""


def test_analyze_ingestion_openapi_schema_is_simplified():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    request_schema = schema["components"]["schemas"]["AnalyzeIngestionRequest"]
    props = request_schema["properties"]

    assert set(props.keys()) == {
        "ingestion_id",
        "model_id",
        "api_key",
        "provider",
        "task_tier",
    }
    assert request_schema["required"] == ["ingestion_id"]
    assert props["task_tier"]["default"] == "weight_audit"


def test_ingest_artifacts_success_response(monkeypatch):
    stored_bundle = {
        "case_id": "ing-1",
        "prompt": "demo",
        "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
        "source_files": {"reference_image": "reference_image_abc123.png"},
        "ingestion_metadata": {
            "api_version": "1.0",
            "created_at_unix": 1,
            "limits": {},
            "extractor_metadata": {"clip_extractor_fallback_count": 0},
        },
    }

    monkeypatch.setattr(app_module, "ingest_artifacts_to_bundle", _fake_bundle)
    monkeypatch.setattr(app_module, "store_ingested_bundle", lambda bundle: "ing-1")
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: stored_bundle)

    response = client.post(
        "/ingest-artifacts",
        data={"prompt": "frontal reference image"},
        files={"reference_image": ("ref.png", b"fake", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "1.0"
    assert payload["ingestion_id"] == "ing-1"
    assert payload["next_step"]["reset_payload"] == {"ingestion_id": "ing-1"}


def test_ingest_artifacts_multipart_uploads_are_detected(monkeypatch):
    captured: dict[str, object] = {}

    async def _capture_bundle(**kwargs):
        captured.update(kwargs)
        return {
            "case_id": "ing-multipart",
            "prompt": kwargs.get("prompt", ""),
            "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
            "source_files": {"reference_image": "reference_image_abc123.png"},
            "ingestion_metadata": {
                "api_version": "1.0",
                "created_at_unix": 1,
                "limits": {},
                "extractor_metadata": {"clip_extractor_fallback_count": 0},
            },
        }

    monkeypatch.setattr(app_module, "ingest_artifacts_to_bundle", _capture_bundle)
    monkeypatch.setattr(app_module, "store_ingested_bundle", lambda bundle: "ing-multipart")
    monkeypatch.setattr(
        app_module,
        "get_ingested_bundle",
        lambda ingestion_id: {
            "case_id": "ing-multipart",
            "prompt": "",
            "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
            "source_files": {"reference_image": "reference_image_abc123.png"},
            "ingestion_metadata": {
                "api_version": "1.0",
                "created_at_unix": 1,
                "limits": {},
                "extractor_metadata": {"clip_extractor_fallback_count": 0},
            },
        },
    )

    response = client.post(
        "/ingest-artifacts",
        data={
            "lora_weights": "",
            "tokenizer_config": "",
            "prompt": "",
            "param_config_json": "",
        },
        files={"reference_image": ("ref.png", b"fake", "image/png")},
    )

    assert response.status_code == 200
    assert captured["reference_image"] is not None


def test_ingest_artifacts_no_files_returns_structured_400():
    response = client.post(
        "/ingest-artifacts",
        data={"prompt": "no files attached"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_ingestion_request"
    assert detail["retryable"] is False


def test_ingest_artifacts_internal_error_hides_exception(monkeypatch):
    async def _broken_bundle(**kwargs):
        raise RuntimeError("internal stack detail should not leak")

    monkeypatch.setattr(app_module, "ingest_artifacts_to_bundle", _broken_bundle)

    response = client.post(
        "/ingest-artifacts",
        data={"prompt": "x"},
        files={"reference_image": ("ref.png", b"fake", "image/png")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "internal_ingestion_error"
    assert detail["message"] == "Artifact ingestion failed."
    assert "internal stack detail" not in str(detail)


def test_get_ingestion_unknown_returns_structured_404(monkeypatch):
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: None)

    response = client.get("/ingestions/missing")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "unknown_ingestion_id"


def test_list_ingestions_has_api_version(monkeypatch):
    monkeypatch.setattr(app_module, "list_ingested_bundle_ids", lambda: ["a", "b"])

    response = client.get("/ingestions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "1.0"
    assert payload["ingestion_ids"] == ["a", "b"]


def test_analyze_ingestion_success(monkeypatch):
    bundle = {
        "case_id": "ing-1",
        "prompt": "frontal portrait",
        "param_config": {"cfg": 7.5, "denoise_alt": 0.3, "eta": 0.1},
    }

    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: bundle)
    monkeypatch.setattr(
        app_module,
        "analyze_ingested_bundle",
        lambda _bundle, **kwargs: {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "report": "Overall Readiness\nReady with minor risks.",
            "signal_digest": {"case_id": "ing-1"},
        },
    )

    response = client.post(
        "/analyze-ingestion",
        json={
            "ingestion_id": "ing-1",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "api_key": "sk-test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "1.0"
    assert payload["ingestion_id"] == "ing-1"
    assert payload["provider"] == "openai"
    assert payload["model_id"] == "gpt-4o-mini"
    assert "Overall Readiness" in payload["report"]


def test_analyze_ingestion_task_tier_forwarding(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: {"case_id": "ing-1"})

    def _fake_analyze(_bundle, **kwargs):
        captured.update(kwargs)
        return {
            "provider": "local",
            "model_id": "llama3.1",
            "report": "Overall Readiness\nScoped by tier.",
            "signal_digest": {"case_id": "ing-1"},
        }

    monkeypatch.setattr(app_module, "analyze_ingested_bundle", _fake_analyze)

    response = client.post(
        "/analyze-ingestion",
        json={
            "ingestion_id": "ing-1",
            "provider": "local",
            "model_id": "llama3.1",
            "task_tier": "image_audit",
        },
    )

    assert response.status_code == 200
    assert captured["task_tier"] == "image_audit"

    response_default = client.post(
        "/analyze-ingestion",
        json={
            "ingestion_id": "ing-1",
            "provider": "local",
            "model_id": "llama3.1",
        },
    )

    assert response_default.status_code == 200
    assert captured["task_tier"] == "weight_audit"


def test_analyze_ingestion_unknown_returns_structured_404(monkeypatch):
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: None)

    response = client.post(
        "/analyze-ingestion",
        json={"ingestion_id": "missing", "provider": "local", "model_id": "llama3.1"},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "unknown_ingestion_id"
    assert detail["retryable"] is False


def test_analyze_ingestion_adapter_error_is_structured(monkeypatch):
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: {"case_id": "ing-1"})

    def _raise_adapter_error(_bundle, **kwargs):
        raise LLMAdapterError(
            code="missing_api_key",
            message="Provider 'openai' requires api_key.",
            status_code=400,
            retryable=False,
        )

    monkeypatch.setattr(app_module, "analyze_ingested_bundle", _raise_adapter_error)

    response = client.post(
        "/analyze-ingestion",
        json={"ingestion_id": "ing-1", "provider": "openai", "model_id": "gpt-4o-mini"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "missing_api_key"
    assert detail["message"] == "Provider 'openai' requires api_key."


def test_analyze_ingestion_custom_base_url_disabled_is_structured(monkeypatch):
    monkeypatch.setattr(app_module, "get_ingested_bundle", lambda ingestion_id: {"case_id": "ing-1"})

    response = client.post(
        "/analyze-ingestion",
        json={
            "ingestion_id": "ing-1",
            "provider": "local",
            "model_id": "llama3.1",
            "base_url": "https://api.openai.com/v1",
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "custom_base_url_disabled"
    assert detail["message"] == "Custom base_url is disabled for this deployment."
