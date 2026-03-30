"""FastAPI app entrypoint for TalkingHeadBench OpenEnv server."""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal
from pathlib import Path

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
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
# Custom OpenAPI schema: ensure clips is typed as array-of-binary files and
# param_config_json has no spurious string default.
# ---------------------------------------------------------------------------

def _patch_ingest_artifacts_props(props: dict[str, Any]) -> None:
    if "clips" in props:
        props["clips"] = {
            "title": "Clips",
            "description": "One or more video clips (.mp4 / .mov / .avi / .mkv / .webm)",
            "type": "array",
            "items": {"type": "string", "format": "binary"},
            "default": [],
        }

    # Convert OpenAPI 3.1 contentMediaType encoding into Swagger-friendly binary format.
    for key in ("reference_image", "lora_weights", "tokenizer_config"):
        field = props.get(key)
        if not isinstance(field, dict):
            continue
        variants = field.get("anyOf")
        if not isinstance(variants, list):
            continue
        for item in variants:
            if isinstance(item, dict) and item.get("type") == "string":
                item.pop("contentMediaType", None)
                item["format"] = "binary"

    if "param_config_json" in props and isinstance(props["param_config_json"], dict):
        field = props["param_config_json"]
        field.pop("anyOf", None)
        field["type"] = "string"
        field["default"] = ""
        field["example"] = ""
        field.pop("nullable", None)


def _patch_analyze_ingestion_props(props: dict[str, Any]) -> None:
    allowed_keys = {"ingestion_id", "model_id", "api_key", "provider"}
    for key in list(props.keys()):
        if key not in allowed_keys:
            props.pop(key, None)


def _patch_analyze_ingestion_schema(schema_obj: dict[str, Any]) -> None:
    props = schema_obj.get("properties")
    if isinstance(props, dict):
        _patch_analyze_ingestion_props(props)

    required = schema_obj.get("required")
    if isinstance(required, list):
        schema_obj["required"] = [
            field_name
            for field_name in required
            if field_name in {"ingestion_id", "model_id", "api_key", "provider"}
        ]

def _patched_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    components = schema.get("components", {}).get("schemas", {})

    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue

            request_schema = (
                operation
                .get("requestBody", {})
                .get("content", {})
                .get("multipart/form-data", {})
                .get("schema", {})
            )
            if not isinstance(request_schema, dict):
                continue

            props: dict[str, Any] | None = None
            if isinstance(request_schema.get("properties"), dict):
                props = request_schema["properties"]
            elif isinstance(request_schema.get("$ref"), str):
                ref_name = str(request_schema["$ref"]).rsplit("/", 1)[-1]
                ref_schema = components.get(ref_name)
                if isinstance(ref_schema, dict) and isinstance(ref_schema.get("properties"), dict):
                    props = ref_schema["properties"]

            if props:
                _patch_ingest_artifacts_props(props)

    # Safety net: patch ingest body component directly.
    for schema_name, schema_obj in components.items():
        if "ingest_artifacts" not in schema_name.lower():
            continue
        if isinstance(schema_obj, dict) and isinstance(schema_obj.get("properties"), dict):
            _patch_ingest_artifacts_props(schema_obj["properties"])

    analyze_request_schema = (
        schema
        .get("paths", {})
        .get("/analyze-ingestion", {})
        .get("post", {})
        .get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    if isinstance(analyze_request_schema, dict):
        if isinstance(analyze_request_schema.get("properties"), dict):
            _patch_analyze_ingestion_schema(analyze_request_schema)
        elif isinstance(analyze_request_schema.get("$ref"), str):
            ref_name = str(analyze_request_schema["$ref"]).rsplit("/", 1)[-1]
            ref_schema = components.get(ref_name)
            if isinstance(ref_schema, dict):
                _patch_analyze_ingestion_schema(ref_schema)

    # Safety net: patch analyze request body component directly.
    for schema_name, schema_obj in components.items():
        if "analyzeingestionrequest" not in schema_name.lower():
            continue
        if isinstance(schema_obj, dict):
            _patch_analyze_ingestion_schema(schema_obj)

    # ---------------------------------------------------------------------------
    # /ingest-artifacts: the endpoint now takes a raw Request (to filter out the
    # empty-string placeholders Swagger sends for unfilled file fields), so FastAPI
    # no longer auto-generates its requestBody.  Inject it manually here.
    # ---------------------------------------------------------------------------
    _INGEST_REQUEST_BODY: dict[str, Any] = {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "reference_image": {
                            "title": "Reference Image",
                            "description": "Reference portrait image (jpg / png / webp)",
                            "anyOf": [
                                {"type": "string", "format": "binary"},
                                {"type": "null"},
                            ],
                        },
                        "clips": {
                            "title": "Clips",
                            "description": "One or more video clips (.mp4 / .mov / .avi / .mkv / .webm)",
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "default": [],
                        },
                        "lora_weights": {
                            "title": "LoRA Weights",
                            "description": "LoRA weight file (.safetensors / .bin / .pt)",
                            "anyOf": [
                                {"type": "string", "format": "binary"},
                                {"type": "null"},
                            ],
                        },
                        "tokenizer_config": {
                            "title": "Tokenizer Config",
                            "description": "Tokenizer config JSON file",
                            "anyOf": [
                                {"type": "string", "format": "binary"},
                                {"type": "null"},
                            ],
                        },
                        "prompt": {
                            "title": "Prompt",
                            "description": "Text prompt describing the talking-head generation task",
                            "type": "string",
                            "default": "",
                        },
                        "param_config_json": {
                            "title": "Param Config JSON",
                            "description": "Optional JSON string with generation parameter overrides",
                            "type": "string",
                            "default": "",
                        },
                    },
                }
            }
        },
    }
    ingest_post = (
        schema.get("paths", {}).get("/ingest-artifacts", {}).get("post")
    )
    if isinstance(ingest_post, dict):
        ingest_post["requestBody"] = _INGEST_REQUEST_BODY

    app.openapi_schema = schema
    return app.openapi_schema



app.openapi = _patched_openapi  # type: ignore[method-assign]


# Serve Scalar as the API reference UI (replaces the default Swagger UI).
_SCALAR_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TalkingHeadBench API Reference</title>
    <style>body { margin: 0; }</style>
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{
        "theme": "purple",
        "layout": "modern",
        "defaultHttpClient": {"targetKey": "python", "clientKey": "requests"}
      }'
    ></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>
"""


@app.get("/docs", include_in_schema=False)
async def scalar_ui(request: Request) -> HTMLResponse:  # noqa: ARG001
    return HTMLResponse(_SCALAR_HTML)


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health endpoint for deployment probes."""
    return HealthResponse()


@app.post("/ingest-artifacts", response_model=IngestArtifactsResponse)
async def ingest_artifacts(request: Request) -> IngestArtifactsResponse:
    """Upload artifacts, extract signals, and return a reusable ingestion id."""

    # Parse the raw multipart form ourselves so we can filter out the empty-string
    # placeholders that Swagger UI sends for optional file fields that aren't filled in.
    # FastAPI's automatic File() injection fails with "Expected UploadFile, received str"
    # whenever Swagger submits an empty string for clips / reference_image / etc.
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        _raise_http_error(
            status_code=400,
            code="invalid_multipart_form",
            message=f"Could not parse multipart form: {exc}",
        )

    def _as_upload(value: object) -> UploadFile | None:
        """Return value only if it is a real uploaded file (non-empty filename)."""
        if isinstance(value, UploadFile) and value.filename:
            return value
        return None

    reference_image: UploadFile | None = _as_upload(form.get("reference_image"))
    lora_weights: UploadFile | None = _as_upload(form.get("lora_weights"))
    tokenizer_config: UploadFile | None = _as_upload(form.get("tokenizer_config"))

    # clips may be sent as a single value or a list; filter out empty-string entries.
    raw_clips = form.getlist("clips")
    clips: list[UploadFile] = [f for f in raw_clips if isinstance(f, UploadFile) and f.filename]

    prompt: str = str(form.get("prompt") or "")
    param_config_json: str = str(form.get("param_config_json") or "")

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