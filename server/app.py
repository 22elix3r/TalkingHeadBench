"""FastAPI app entrypoint for TalkingHeadBench OpenEnv server."""

from __future__ import annotations

import logging
import sys
from typing import Any, List, Literal
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import TalkingHeadObservation
from server.artifact_ingest import (
    delete_ingested_bundle,
    get_ingested_bundle,
    ingest_artifacts_to_bundle,
    list_ingested_bundle_ids,
    store_ingested_bundle,
)
from server.llm_adapter import LLMAdapterError, analyze_ingested_bundle
from server.talking_head_environment import TalkingHeadEnvironment

log = logging.getLogger(__name__)

API_VERSION = "1.0"


class APIErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class NextStepHint(BaseModel):
    reset_payload: dict[str, Any]
    description: str


class IngestArtifactsResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    ingestion_id: str
    bundle: dict[str, Any]
    next_step: NextStepHint


class ListIngestionsResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    ingestion_ids: list[str]


class IngestionResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    ingestion_id: str
    bundle: dict[str, Any]


class DeleteIngestionResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    ingestion_id: str
    deleted: bool


class AnalyzeIngestionRequest(BaseModel):
    ingestion_id: str
    model_id: str | None = None
    api_key: str | None = None
    provider: Literal["auto", "openai", "anthropic", "huggingface", "local"] = "auto"
    base_url: str | None = None
    max_tokens: int = Field(default=700, ge=64, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    timeout_s: float = Field(default=45.0, ge=5.0, le=180.0)


class AnalyzeIngestionResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    ingestion_id: str
    provider: str
    model_id: str
    report: str
    signal_digest: dict[str, Any]


class HealthResponse(BaseModel):
    api_version: str = Field(default=API_VERSION)
    status: Literal["ok"] = "ok"


def _raise_http_error(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=APIErrorDetail(code=code, message=message, retryable=retryable).model_dump(
            mode="json"
        ),
    )

app = create_app(
    TalkingHeadEnvironment,
    Action,
    TalkingHeadObservation,
    env_name="talking_head_bench",
)


# ---------------------------------------------------------------------------
# OpenAPI schema patch: make `clips` render as file pickers in Swagger UI.
# FastAPI generates array<string> for list[UploadFile] which is wrong.
# ---------------------------------------------------------------------------
_openapi_cache: dict | None = None


def _patched_openapi() -> dict:
    global _openapi_cache
    if _openapi_cache is not None:
        return _openapi_cache

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    # Walk every requestBody and fix the clips field
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            try:
                content = operation["requestBody"]["content"]
                form_schema = content["multipart/form-data"]["schema"]
                props = form_schema.get("properties", {})
                if "clips" in props:
                    props["clips"] = {
                        "type": "array",
                        "items": {"type": "string", "format": "binary"},
                        "description": "Optional list of dataset clip files (.mp4 / .mov / .avi / .mkv / .webm)",
                    }
            except (KeyError, TypeError):
                continue

    _openapi_cache = schema
    return schema


app.openapi = _patched_openapi  # type: ignore[method-assign]


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health endpoint for deployment probes."""
    return HealthResponse()


@app.post("/ingest-artifacts", response_model=IngestArtifactsResponse)
async def ingest_artifacts(
    reference_image: UploadFile | None = File(default=None, description="Optional reference image file."),
    clips: List[UploadFile] = File(default=[], description="Optional list of dataset clip files (.mp4 / .mov / .avi / .mkv / .webm)"),
    lora_weights: UploadFile | None = File(default=None, description="Optional LoRA weights file (.safetensors)."),
    tokenizer_config: UploadFile | None = File(default=None, description="Optional tokenizer config (.json) for phoneme mapping."),
    prompt: str = Form(default="", description="Text prompt used for generation."),
    param_config_json: str | None = Form(default=None, description='Generation params JSON, e.g. {"cfg": 7.5, "eta": 0.1}'),
) -> IngestArtifactsResponse:
    """Upload artifacts, extract signals, and return a reusable ingestion id."""
    try:
        bundle = await ingest_artifacts_to_bundle(
            reference_image=reference_image,
            clips=clips,
            lora_weights=lora_weights,
            tokenizer_config=tokenizer_config,
            prompt=prompt,
            param_config_json=param_config_json,
        )
    except ValueError as exc:
        _raise_http_error(
            status_code=400,
            code="invalid_ingestion_request",
            message=str(exc),
        )
    except Exception:  # noqa: BLE001
        log.exception("Artifact ingestion failed")
        _raise_http_error(
            status_code=500,
            code="internal_ingestion_error",
            message="Artifact ingestion failed.",
            retryable=True,
        )

    ingestion_id = store_ingested_bundle(bundle)
    stored_bundle = get_ingested_bundle(ingestion_id)
    if stored_bundle is None:
        _raise_http_error(
            status_code=500,
            code="ingestion_persistence_failed",
            message="Failed to persist ingested bundle.",
            retryable=True,
        )

    return IngestArtifactsResponse(
        ingestion_id=ingestion_id,
        bundle=stored_bundle,
        next_step=NextStepHint(
            reset_payload={"ingestion_id": ingestion_id},
            description="Pass this payload to env.reset(...) over OpenEnv WebSocket.",
        ),
    )


@app.get("/ingestions", response_model=ListIngestionsResponse)
async def list_ingestions() -> ListIngestionsResponse:
    """List available ingestion ids currently held in memory."""
    return ListIngestionsResponse(ingestion_ids=list_ingested_bundle_ids())


@app.get("/ingestions/{ingestion_id}", response_model=IngestionResponse)
async def get_ingestion(ingestion_id: str) -> IngestionResponse:
    """Retrieve a previously ingested signal bundle by id."""
    bundle = get_ingested_bundle(ingestion_id)
    if bundle is None:
        _raise_http_error(
            status_code=404,
            code="unknown_ingestion_id",
            message=f"Unknown ingestion id: {ingestion_id}",
        )
    return IngestionResponse(ingestion_id=ingestion_id, bundle=bundle)


@app.delete("/ingestions/{ingestion_id}", response_model=DeleteIngestionResponse)
async def delete_ingestion(ingestion_id: str) -> DeleteIngestionResponse:
    """Delete an ingested bundle from in-memory storage."""
    removed = delete_ingested_bundle(ingestion_id)
    if not removed:
        _raise_http_error(
            status_code=404,
            code="unknown_ingestion_id",
            message=f"Unknown ingestion id: {ingestion_id}",
        )
    return DeleteIngestionResponse(ingestion_id=ingestion_id, deleted=True)


@app.post("/analyze-ingestion", response_model=AnalyzeIngestionResponse)
async def analyze_ingestion(request: AnalyzeIngestionRequest) -> AnalyzeIngestionResponse:
    """Generate an LLM report from a stored ingestion bundle."""
    bundle = get_ingested_bundle(request.ingestion_id)
    if bundle is None:
        _raise_http_error(
            status_code=404,
            code="unknown_ingestion_id",
            message=f"Unknown ingestion id: {request.ingestion_id}",
        )

    try:
        result = analyze_ingested_bundle(
            bundle,
            model_id=request.model_id,
            api_key=request.api_key,
            provider=request.provider,
            base_url=request.base_url,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout_s=request.timeout_s,
        )
    except LLMAdapterError as exc:
        _raise_http_error(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
    except Exception:  # noqa: BLE001
        log.exception("Ingestion analysis failed")
        _raise_http_error(
            status_code=502,
            code="analysis_provider_error",
            message="Failed to generate analysis report.",
            retryable=True,
        )

    return AnalyzeIngestionResponse(
        ingestion_id=request.ingestion_id,
        provider=str(result["provider"]),
        model_id=str(result["model_id"]),
        report=str(result["report"]),
        signal_digest=dict(result.get("signal_digest") or {}),
    )


def main() -> None:
    """Run the TalkingHeadBench environment server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()