"""Artifact ingestion and signal extraction helpers for TalkingHeadBench.

This module enables user-provided files (reference images, clips, and LoRA
weights) to be uploaded once, converted into pre-extracted signal JSON, and
reused in OpenEnv episodes through an ingestion ID.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

import cv2
from fastapi import UploadFile

from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
from src.envs.subenv3.node7_weight_extractor import extract_weight_signals
from src.schemas.subenv1 import ImageDiagnosticsObservation
from src.schemas.subenv2 import ClipSignalObservation
from src.schemas.subenv3 import WeightSignalObservation

log = logging.getLogger(__name__)

_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "talkingheadbench_uploads"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

_BUNDLE_STORE: dict[str, dict[str, Any]] = {}
_BUNDLE_STORE_LOCK = Lock()


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


API_VERSION = "1.0"
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_UPLOAD_BYTES = _env_int("THB_MAX_UPLOAD_BYTES", 512 * 1024 * 1024, minimum=1)
_MAX_CLIPS_PER_REQUEST = _env_int("THB_MAX_CLIPS_PER_REQUEST", 12, minimum=1)
_MAX_BUNDLES_IN_MEMORY = _env_int("THB_MAX_BUNDLES_IN_MEMORY", 128, minimum=1)
_UPLOAD_TTL_SECONDS = _env_int("THB_UPLOAD_TTL_SECONDS", 24 * 60 * 60, minimum=300)

_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_ALLOWED_CLIP_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_ALLOWED_WEIGHT_EXTS = {".safetensors"}
_ALLOWED_TOKENIZER_EXTS = {".json"}

_DEFAULT_PARAM_CONFIG: dict[str, float] = {
    "cfg": 7.5,
    "denoise_alt": 0.30,
    "eta": 0.10,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _detect_conflicting_descriptors(prompt: str) -> list[str]:
    text = prompt.lower()
    pairs = [
        ("young", "old"),
        ("male", "female"),
        ("smiling", "serious"),
        ("frontal", "profile"),
        ("bright", "dark"),
    ]
    conflicts: list[str] = []
    for left, right in pairs:
        if left in text and right in text:
            conflicts.append(f"{left}|{right}")
    return conflicts


def _estimate_face_occupancy_ratio(image_bgr) -> float:
    try:
        import mediapipe as mp

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        ) as detector:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

        if result.detections:
            bbox = result.detections[0].location_data.relative_bounding_box
            return float(_clamp(float(bbox.width * bbox.height), 0.0, 1.0))
    except Exception:
        pass

    return 0.35


def extract_image_signals(image_path: Path, prompt: str) -> ImageDiagnosticsObservation:
    """Extract a Node 1-compatible observation from a reference image file."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"Failed to read reference image: {image_path}")

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_occupancy_ratio = _estimate_face_occupancy_ratio(image_bgr)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clamp((lap_var / float(max(width * height, 1))) / 0.001, 0.0, 1.0)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lighting_uniformity = _clamp(1.0 - float(lab[:, :, 0].astype("float32").std() / 80.0), 0.0, 1.0)

    edges = cv2.Canny(gray, 50, 150)
    background_complexity = _clamp(float((edges.mean() / 255.0) * 3.0), 0.0, 1.0)

    tokens = [tok for tok in prompt.split() if tok.strip()]
    token_count = len(tokens)
    density = len({tok.lower() for tok in tokens}) / max(1, token_count)
    conflicts = _detect_conflicting_descriptors(prompt)

    anchoring = 0.9 if "ohwx" in prompt.lower() else _clamp(0.35 + 0.50 * density, 0.0, 1.0)

    return ImageDiagnosticsObservation(
        face_occupancy_ratio=round(face_occupancy_ratio, 4),
        estimated_yaw_degrees=0.0,
        estimated_pitch_degrees=0.0,
        background_complexity_score=round(background_complexity, 4),
        lighting_uniformity_score=round(lighting_uniformity, 4),
        skin_tone_bucket=3,
        occlusion_detected=False,
        image_resolution=(width, height),
        estimated_sharpness=round(sharpness, 4),
        prompt_token_count=token_count,
        prompt_semantic_density=round(float(density), 4),
        conflicting_descriptors=conflicts,
        identity_anchoring_strength=round(float(anchoring), 4),
    )


def _parse_param_config(param_config_json: Optional[str]) -> dict[str, float]:
    config = dict(_DEFAULT_PARAM_CONFIG)
    if not param_config_json:
        return config

    try:
        payload = json.loads(param_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("param_config_json must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("param_config_json must decode to an object")

    for key, value in payload.items():
        if isinstance(value, (int, float)):
            config[key] = float(value)

    return config


def _validate_upload_extension(
    upload: UploadFile,
    *,
    fallback_name: str,
    allowed_extensions: set[str],
    field_name: str,
) -> None:
    suffix = Path(upload.filename or fallback_name).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"{field_name} must use one of the following extensions: {allowed}")


def _remove_upload_dir(ingestion_id: str) -> None:
    upload_dir = _UPLOAD_ROOT / str(ingestion_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)


def _prune_expired_assets() -> None:
    now = time.time()
    expired_ids: list[str] = []
    active_ids: set[str] = set()

    with _BUNDLE_STORE_LOCK:
        for ingestion_id, bundle in list(_BUNDLE_STORE.items()):
            active_ids.add(ingestion_id)
            metadata = bundle.get("ingestion_metadata", {})
            created_at = metadata.get("created_at_unix")
            if isinstance(created_at, (int, float)) and now - float(created_at) > _UPLOAD_TTL_SECONDS:
                expired_ids.append(ingestion_id)

        for ingestion_id in expired_ids:
            _BUNDLE_STORE.pop(ingestion_id, None)
            active_ids.discard(ingestion_id)

    for ingestion_id in expired_ids:
        _remove_upload_dir(ingestion_id)

    for child in _UPLOAD_ROOT.iterdir():
        if not child.is_dir() or child.name in active_ids:
            continue
        try:
            age_seconds = now - child.stat().st_mtime
        except OSError:
            continue
        if age_seconds > _UPLOAD_TTL_SECONDS:
            shutil.rmtree(child, ignore_errors=True)


def _update_phoneme_coverage(coverage: dict[str, int], clip_obs: ClipSignalObservation) -> None:
    for phoneme in clip_obs.phoneme_sequence:
        coverage[phoneme] = int(coverage.get(phoneme, 0)) + 1


def _extract_clip_signals_fallback(
    clip_path: Path,
    dataset_context: dict[str, Any],
) -> ClipSignalObservation:
    """Deterministic OpenCV-only fallback when full extractor deps are unavailable."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open clip: {clip_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) >= 180:
            break
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Clip has too few frames: {clip_path}")

    blur_scores: list[float] = []
    exposure_scores: list[float] = []
    diffs: list[float] = []

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        pixel_count = max(1, gray.shape[0] * gray.shape[1])
        blur_scores.append(_clamp((lap_var / pixel_count) / 0.001, 0.0, 1.0))

        brightness = float(gray.mean() / 255.0)
        exposure_scores.append(_clamp(1.0 - abs(brightness - 0.5) * 2.0, 0.0, 1.0))

        if idx > 0:
            prev = frames[idx - 1].astype("float32")
            cur = frame.astype("float32")
            diffs.append(float(abs(cur - prev).mean()))

    frame_difference_mean = float(sum(diffs) / max(1, len(diffs)))
    identity_drift_proxy = _clamp(frame_difference_mean / 45.0, 0.0, 1.0)
    landmark_jitter_proxy = _clamp(frame_difference_mean / 500.0, 0.0, 1.0)

    return ClipSignalObservation(
        clip_id=clip_path.stem,
        face_embedding_variance=round(identity_drift_proxy, 4),
        landmark_stability_score=round(landmark_jitter_proxy, 4),
        identity_cosine_drift=round(identity_drift_proxy, 4),
        frame_difference_mean=round(frame_difference_mean, 4),
        optical_flow_magnitude=1.0,
        blink_count=max(0, len(frames) // 60),
        lip_sync_confidence=0.5,
        phoneme_sequence=[],
        phoneme_coverage_new=0.0,
        blur_score=round(float(sum(blur_scores) / max(1, len(blur_scores))), 4),
        exposure_score=round(float(sum(exposure_scores) / max(1, len(exposure_scores))), 4),
        occlusion_frames=0,
        clips_audited_so_far=int(dataset_context.get("clips_audited_so_far", 0)),
        current_phoneme_coverage=dataset_context.get("current_phoneme_coverage", {}),
        current_pose_distribution=dataset_context.get("current_pose_distribution", {}),
        similar_clips_accepted=int(dataset_context.get("similar_clips_accepted", 0)),
    )


def _extract_clip_signal_observations(
    clip_paths: list[Path],
) -> tuple[list[ClipSignalObservation], int]:
    observations: list[ClipSignalObservation] = []
    phoneme_coverage: dict[str, int] = {}
    fallback_count = 0

    for idx, clip_path in enumerate(clip_paths):
        dataset_context = {
            "clips_audited_so_far": idx,
            "current_phoneme_coverage": phoneme_coverage,
            "current_pose_distribution": {},
            "similar_clips_accepted": 0,
        }

        try:
            clip_obs = extract_clip_signals(
                clip_path=clip_path,
                dataset_context=dataset_context,
                aligner_output=None,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Full clip extractor failed for %s; using fallback extractor: %s",
                clip_path,
                exc,
            )
            fallback_count += 1
            clip_obs = _extract_clip_signals_fallback(clip_path, dataset_context)
        observations.append(clip_obs)
        _update_phoneme_coverage(phoneme_coverage, clip_obs)

    return observations, fallback_count


async def _save_upload(
    upload: UploadFile,
    output_dir: Path,
    fallback_name: str,
    *,
    max_upload_bytes: int = _MAX_UPLOAD_BYTES,
) -> Path:
    original_name = Path(upload.filename or fallback_name).name
    suffix = Path(original_name).suffix.lower() or Path(fallback_name).suffix.lower() or ".bin"
    destination = output_dir / f"{Path(fallback_name).stem}_{uuid4().hex[:8]}{suffix}"

    bytes_written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise ValueError(
                        f"Uploaded file exceeds the size limit of {max_upload_bytes} bytes"
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        await upload.close()
        raise

    await upload.close()
    return destination


async def ingest_artifacts_to_bundle(
    *,
    reference_image: UploadFile | None,
    clips: list[UploadFile] | None,
    lora_weights: UploadFile | None,
    tokenizer_config: UploadFile | None,
    prompt: str,
    param_config_json: str,
) -> dict[str, Any]:
    """Extract a custom signal bundle from uploaded user artifacts."""
    clips = clips or []
    _prune_expired_assets()

    if reference_image is None and not clips and lora_weights is None:
        raise ValueError("At least one artifact is required (image, clips, or lora_weights).")

    if len(clips) > _MAX_CLIPS_PER_REQUEST:
        raise ValueError(f"A maximum of {_MAX_CLIPS_PER_REQUEST} clips is allowed per request")

    if reference_image is not None:
        _validate_upload_extension(
            reference_image,
            fallback_name="reference_image.png",
            allowed_extensions=_ALLOWED_IMAGE_EXTS,
            field_name="reference_image",
        )

    for clip_upload in clips:
        _validate_upload_extension(
            clip_upload,
            fallback_name="clip.mp4",
            allowed_extensions=_ALLOWED_CLIP_EXTS,
            field_name="clips",
        )

    if lora_weights is not None:
        _validate_upload_extension(
            lora_weights,
            fallback_name="weights.safetensors",
            allowed_extensions=_ALLOWED_WEIGHT_EXTS,
            field_name="lora_weights",
        )

    if tokenizer_config is not None:
        _validate_upload_extension(
            tokenizer_config,
            fallback_name="tokenizer_config.json",
            allowed_extensions=_ALLOWED_TOKENIZER_EXTS,
            field_name="tokenizer_config",
        )

    ingestion_id = str(uuid4())
    created_at = int(time.time())
    output_dir = _UPLOAD_ROOT / ingestion_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {
        "case_id": ingestion_id,
        "prompt": prompt,
        "param_config": _parse_param_config(param_config_json),
        "source_files": {},
        "ingestion_metadata": {
            "api_version": API_VERSION,
            "created_at_unix": created_at,
            "limits": {
                "max_upload_bytes": _MAX_UPLOAD_BYTES,
                "max_clips_per_request": _MAX_CLIPS_PER_REQUEST,
                "upload_ttl_seconds": _UPLOAD_TTL_SECONDS,
                "max_bundles_in_memory": _MAX_BUNDLES_IN_MEMORY,
            },
            "extractor_metadata": {
                "opencv_version": cv2.__version__,
                "clip_extractor_fallback_count": 0,
            },
        },
    }

    if reference_image is not None:
        image_path = await _save_upload(reference_image, output_dir, "reference_image.png")
        image_obs = extract_image_signals(image_path, prompt)
        bundle["image_observation"] = image_obs.model_dump(mode="json")
        bundle["source_files"]["reference_image"] = image_path.name

    if clips:
        clip_paths: list[Path] = []
        for idx, clip_upload in enumerate(clips):
            clip_path = await _save_upload(clip_upload, output_dir, f"clip_{idx:03d}.mp4")
            clip_paths.append(clip_path)

        clip_obs_list, fallback_count = _extract_clip_signal_observations(clip_paths)
        bundle["clip_signal_observations"] = [
            clip_obs.model_dump(mode="json") for clip_obs in clip_obs_list
        ]
        bundle["source_files"]["clips"] = [p.name for p in clip_paths]
        bundle["ingestion_metadata"]["extractor_metadata"][
            "clip_extractor_fallback_count"
        ] = fallback_count

    if lora_weights is not None:
        weight_path = await _save_upload(lora_weights, output_dir, "weights.safetensors")
        tokenizer_path: Path | None = None
        if tokenizer_config is not None:
            tokenizer_path = await _save_upload(tokenizer_config, output_dir, "tokenizer_config.json")

        weight_obs: WeightSignalObservation = extract_weight_signals(
            weight_path=weight_path,
            tokenizer_config_path=tokenizer_path,
        )
        bundle["weight_observation"] = weight_obs.model_dump(mode="json")
        bundle["source_files"]["lora_weights"] = weight_path.name
        if tokenizer_path is not None:
            bundle["source_files"]["tokenizer_config"] = tokenizer_path.name

    return bundle


def store_ingested_bundle(bundle: dict[str, Any]) -> str:
    _prune_expired_assets()

    ingestion_id = str(bundle.get("case_id") or uuid4())
    saved = copy.deepcopy(bundle)
    saved["case_id"] = ingestion_id

    evicted_ids: list[str] = []

    with _BUNDLE_STORE_LOCK:
        _BUNDLE_STORE[ingestion_id] = saved

        while len(_BUNDLE_STORE) > _MAX_BUNDLES_IN_MEMORY:
            oldest_id = next(iter(_BUNDLE_STORE.keys()))
            _BUNDLE_STORE.pop(oldest_id, None)
            evicted_ids.append(oldest_id)

    for evicted_id in evicted_ids:
        _remove_upload_dir(evicted_id)

    return ingestion_id


def get_ingested_bundle(ingestion_id: str) -> dict[str, Any] | None:
    _prune_expired_assets()

    with _BUNDLE_STORE_LOCK:
        bundle = _BUNDLE_STORE.get(str(ingestion_id))
    if bundle is None:
        return None
    return copy.deepcopy(bundle)


def list_ingested_bundle_ids() -> list[str]:
    _prune_expired_assets()

    with _BUNDLE_STORE_LOCK:
        return sorted(_BUNDLE_STORE.keys())


def delete_ingested_bundle(ingestion_id: str) -> bool:
    with _BUNDLE_STORE_LOCK:
        removed = _BUNDLE_STORE.pop(str(ingestion_id), None)
    if removed is not None:
        _remove_upload_dir(str(ingestion_id))
    return removed is not None
