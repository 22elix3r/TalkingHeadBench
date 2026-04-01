#!/usr/bin/env python3
"""Extract Sub-env 2 clip signals and write annotation-ready cases JSON.

Workflow per clip:
1. Build dataset context from accumulated coverage state.
2. Run Node 4 clip signal extraction.
3. Build a ClipEvidenceDossier (pipeline helper if available, else inline mapping).
4. Run Node 5 disposition recommendation.
5. Emit a case with ClipSignalObservation as observation and ANNOTATE ground truth.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from numpy.linalg import norm
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
from src.envs.subenv2.node5_disposition import recommend_clip_disposition
from src.schemas.subenv2 import (
    ClipDispositionAction,
    ClipDispositionObservation,
    ClipEvidenceDossier,
    ClipSignalObservation,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)

EAR_BLINK_THRESHOLD = 0.2
BLUR_CALIBRATION_CEILING = 0.001
MIN_FRAMES = 24

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
            "Extract Sub-env 2 clip signals for all videos in a directory and "
            "write annotation-ready JSON test cases."
        )
    )
    parser.add_argument(
        "--clips",
        required=True,
        type=Path,
        help="Directory containing clip files (.mp4, .mov, .avi)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path (for example tests/test_set/subenv2_cases.json)",
    )
    parser.add_argument(
        "--align-dir",
        required=False,
        type=Path,
        default=None,
        help="Optional directory containing forced-alignment JSON files by clip stem",
    )
    parser.add_argument(
        "--landmarker-model",
        required=False,
        type=Path,
        default=Path("data/models/face_landmarker.task"),
        help="Optional .task model path for MediaPipe Tasks API.",
    )
    return parser.parse_args()


def detect_mediapipe_api() -> str:
    try:
        importlib.import_module("mediapipe.python.solutions.face_mesh")
        return "solutions"
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass
    try:
        importlib.import_module("mediapipe.tasks.python.vision")
        return "tasks"
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass
    raise ImportError(
        "No usable MediaPipe API found. Install with:\n"
        "  pip install mediapipe==0.10.9  # solutions API\n"
        "or provide a face landmarker model file via --landmarker-model"
    )


def download_landmarker_model(dest: Path) -> Path:
    import urllib.request

    url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")
    return dest


def make_face_mesh_tasks(model_path: str) -> Any:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def get_landmarks_tasks(
    landmarker: Any,
    rgb_image: np.ndarray,
) -> tuple[np.ndarray | None, float, Any]:
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image.copy())
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, 0.0, None

    lm = result.face_landmarks[0]
    h, w = rgb_image.shape[:2]
    landmarks_array = np.array(
        [[l.x * w, l.y * h, l.z * w] for l in lm],
        dtype=np.float32,
    )

    confidence = 1.0
    transform_matrix = None
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if matrices:
        transform_matrix = matrices[0]

    return landmarks_array, confidence, transform_matrix


def extract_pose_from_transform_matrix(matrix: Any) -> tuple[float, float]:
    m = np.array(matrix.data, dtype=np.float64).reshape(4, 4)

    yaw = math.degrees(math.atan2(m[0, 2], m[2, 2]))
    pitch = math.degrees(math.atan2(-m[1, 2], math.sqrt(m[0, 2] ** 2 + m[2, 2] ** 2)))
    return float(yaw), float(pitch)


def process_image(
    landmarker: Any,
    rgb_image: np.ndarray,
) -> tuple[np.ndarray | None, float, float, float]:
    landmarks_array, confidence, transform_matrix = get_landmarks_tasks(landmarker, rgb_image)
    if landmarks_array is None:
        return None, 0.0, 0.0, 0.0

    yaw_deg = 0.0
    pitch_deg = 0.0
    if transform_matrix is not None:
        try:
            yaw_deg, pitch_deg = extract_pose_from_transform_matrix(transform_matrix)
        except Exception:
            yaw_deg, pitch_deg = 0.0, 0.0

    return landmarks_array, confidence, yaw_deg, pitch_deg


def process_frame(
    landmarker: Any,
    rgb_image: np.ndarray,
) -> tuple[np.ndarray | None, float, float, float]:
    return process_image(landmarker, rgb_image)


def make_arcface_analyzer() -> Any | None:
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        return None

    try:
        analyzer = FaceAnalysis(allowed_modules=["detection", "recognition"])
        analyzer.prepare(ctx_id=-1)
        return analyzer
    except Exception:
        return None


def list_video_files(clips_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in clips_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def load_alignment_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in alignment file {path}: {exc}") from exc


def parse_aligner_phonemes(aligner_output: dict[str, Any] | None) -> list[str]:
    if aligner_output is None:
        return []
    if "phonemes" in aligner_output:
        return [str(p) for p in aligner_output.get("phonemes", [])]
    try:
        entries = aligner_output["tiers"]["phones"]["entries"]
        return [str(entry[2]) for entry in entries]
    except (KeyError, IndexError, TypeError):
        return []


def phoneme_coverage_new(
    phoneme_sequence: list[str],
    current_phoneme_coverage: dict[str, int],
) -> float:
    unique_in_clip = set(phoneme_sequence)
    if not unique_in_clip:
        return 0.0
    new_count = sum(1 for p in unique_in_clip if current_phoneme_coverage.get(p, 0) == 0)
    return new_count / len(unique_in_clip)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def eye_aspect_ratio(landmarks_px: np.ndarray, indices: tuple[int, ...]) -> float:
    if landmarks_px.shape[0] <= max(indices):
        return 1.0

    p1 = landmarks_px[indices[0], :2]
    p2 = landmarks_px[indices[1], :2]
    p3 = landmarks_px[indices[2], :2]
    p4 = landmarks_px[indices[3], :2]
    p5 = landmarks_px[indices[4], :2]
    p6 = landmarks_px[indices[5], :2]

    return float((norm(p2 - p6) + norm(p3 - p5)) / (2.0 * norm(p1 - p4) + 1e-6))


def face_bbox_from_landmarks(
    landmarks_px: np.ndarray,
    width: int,
    height: int,
    padding_ratio: float = 0.2,
) -> tuple[int, int, int, int]:
    x_coords = landmarks_px[:, 0]
    y_coords = landmarks_px[:, 1]

    x0 = int(np.clip(np.floor(np.min(x_coords)), 0, width - 1))
    x1 = int(np.clip(np.ceil(np.max(x_coords)), 1, width))
    y0 = int(np.clip(np.floor(np.min(y_coords)), 0, height - 1))
    y1 = int(np.clip(np.ceil(np.max(y_coords)), 1, height))

    pad_x = int((x1 - x0) * padding_ratio)
    pad_y = int((y1 - y0) * padding_ratio)

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)

    return x0, y0, x1, y1


def extract_clip_signals_tasks(
    clip_path: Path,
    dataset_context: dict[str, Any],
    aligner_output: dict[str, Any] | None,
    landmarker: Any,
    arcface_analyzer: Any | None,
) -> ClipSignalObservation:
    clip_path = Path(clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if landmarker is None:
        raise ValueError("MediaPipe Tasks landmarker is not initialized")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {clip_path}")

    frames_bgr: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
    finally:
        cap.release()

    if len(frames_bgr) < MIN_FRAMES:
        raise ValueError(
            f"Clip '{clip_path.stem}' has only {len(frames_bgr)} frames; at least {MIN_FRAMES} are required."
        )

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr]

    landmarks_per_frame: list[np.ndarray | None] = []
    landmark_vectors: list[np.ndarray] = []
    embedding_proxies: list[np.ndarray] = []
    arcface_embeddings: list[np.ndarray] = []
    blur_scores: list[float] = []
    exposure_scores: list[float] = []

    blink_count = 0
    occlusion_frames = 0

    for frame_bgr, gray in zip(frames_bgr, gray_frames):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks_px, confidence, _yaw, _pitch = process_frame(landmarker, rgb)

        if landmarks_px is None:
            landmarks_per_frame.append(None)
            occlusion_frames += 1
            continue

        landmarks_per_frame.append(landmarks_px)
        landmark_vec = landmarks_px.flatten().astype(np.float32)
        landmark_vectors.append(landmark_vec)
        embedding_proxies.append(landmark_vec / 1000.0)

        if confidence < 0.6:
            occlusion_frames += 1

        x0, y0, x1, y1 = face_bbox_from_landmarks(
            landmarks_px,
            frame_bgr.shape[1],
            frame_bgr.shape[0],
            padding_ratio=0.2,
        )
        face_gray = gray[y0:y1, x0:x1]
        if face_gray.size == 0:
            face_gray = gray

        lap_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        blur_raw = lap_var / float(max(face_gray.size, 1))
        blur_scores.append(float(np.clip(blur_raw / BLUR_CALIBRATION_CEILING, 0.0, 1.0)))
        exposure_scores.append(float(np.clip(face_gray.mean() / 255.0, 0.0, 1.0)))

        left_ear = eye_aspect_ratio(landmarks_px, LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(landmarks_px, RIGHT_EYE_IDX)
        if 0.5 * (left_ear + right_ear) < EAR_BLINK_THRESHOLD:
            blink_count += 1

        if arcface_analyzer is not None:
            try:
                faces = arcface_analyzer.get(frame_bgr)
            except Exception:
                faces = []
            if faces:
                arcface_embeddings.append(faces[0].normed_embedding.astype(np.float32))

    if not landmark_vectors:
        raise ValueError("No face detected in clip")

    if len(gray_frames) >= 2:
        diffs = [
            float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            for a, b in zip(frames_bgr, frames_bgr[1:])
        ]
        frame_difference_mean = float(np.mean(diffs))
    else:
        frame_difference_mean = 0.0

    stability_dists: list[float] = []
    for prev, curr in zip(landmarks_per_frame, landmarks_per_frame[1:]):
        if prev is None or curr is None:
            continue
        stability_dists.append(float(norm(curr - prev)))

    if stability_dists:
        landmark_stability_score = float(np.clip(np.mean(stability_dists) / 100.0, 0.0, 1.0))
    else:
        landmark_stability_score = 1.0

    if len(landmark_vectors) >= 2:
        v1 = landmark_vectors[0]
        v2 = landmark_vectors[-1]
        drift = 1.0 - float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8))
        identity_cosine_drift = float(np.clip(drift, 0.0, 1.0))
    elif len(landmark_vectors) == 1:
        identity_cosine_drift = 0.0
    else:
        identity_cosine_drift = 1.0

    if len(arcface_embeddings) >= 2:
        emb_matrix = np.stack(arcface_embeddings, axis=0)
        face_embedding_variance = float(np.var(emb_matrix, axis=0).mean())
    elif len(embedding_proxies) >= 2:
        proxy_matrix = np.stack(embedding_proxies, axis=0)
        face_embedding_variance = float(np.var(proxy_matrix, axis=0).mean())
    elif len(embedding_proxies) == 1:
        face_embedding_variance = 0.0
    else:
        face_embedding_variance = 1.0

    flow_ratios: list[float] = []
    for i in range(min(len(gray_frames) - 1, 30)):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i],
            gray_frames[i + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        landmarks = landmarks_per_frame[i] if landmarks_per_frame[i] is not None else landmarks_per_frame[i + 1]
        if landmarks is None:
            continue

        x0, y0, x1, y1 = face_bbox_from_landmarks(
            landmarks,
            gray_frames[i].shape[1],
            gray_frames[i].shape[0],
            padding_ratio=0.2,
        )
        face_mag = mag[y0:y1, x0:x1]
        if face_mag.size == 0:
            continue

        face_mean = float(face_mag.mean())
        full_mean = float(mag.mean() + 1e-6)
        flow_ratios.append(face_mean / full_mean)

    optical_flow_magnitude = float(np.mean(flow_ratios)) if flow_ratios else 1.0

    phoneme_sequence = parse_aligner_phonemes(aligner_output)
    current_cov = dataset_context.get("current_phoneme_coverage", {})

    return ClipSignalObservation(
        clip_id=clip_path.stem,
        face_embedding_variance=face_embedding_variance,
        landmark_stability_score=landmark_stability_score,
        identity_cosine_drift=identity_cosine_drift,
        frame_difference_mean=frame_difference_mean,
        optical_flow_magnitude=optical_flow_magnitude,
        blink_count=blink_count,
        lip_sync_confidence=0.0,
        phoneme_sequence=phoneme_sequence,
        phoneme_coverage_new=phoneme_coverage_new(phoneme_sequence, current_cov),
        blur_score=float(np.mean(blur_scores)) if blur_scores else 0.0,
        exposure_score=float(np.mean(exposure_scores)) if exposure_scores else 0.0,
        occlusion_frames=occlusion_frames,
        clips_audited_so_far=int(dataset_context.get("clips_audited_so_far", 0)),
        current_phoneme_coverage=current_cov,
        current_pose_distribution=dataset_context.get("current_pose_distribution", {}),
        similar_clips_accepted=int(dataset_context.get("similar_clips_accepted", 0)),
    )


def is_no_face_detected(obs: ClipSignalObservation) -> bool:
    # Node4 sets these to worst-case sentinels when no faces are detected.
    return (
        obs.face_embedding_variance >= 0.999
        and obs.identity_cosine_drift >= 0.999
        and obs.landmark_stability_score >= 0.999
    )


def call_extract_clip_signals(
    clip_path: Path,
    dataset_context: dict[str, Any],
    forced_align_path: Path | None,
    mediapipe_api: str,
    tasks_landmarker: Any | None,
    arcface_analyzer: Any | None,
) -> ClipSignalObservation:
    aligner_output = load_alignment_json(forced_align_path)

    if mediapipe_api == "tasks":
        return extract_clip_signals_tasks(
            clip_path=clip_path,
            dataset_context=dataset_context,
            aligner_output=aligner_output,
            landmarker=tasks_landmarker,
            arcface_analyzer=arcface_analyzer,
        )

    sig = inspect.signature(extract_clip_signals)

    try:
        if "forced_align_path" in sig.parameters:
            raw_obs = extract_clip_signals(
                clip_path=clip_path,
                dataset_context=dataset_context,
                forced_align_path=forced_align_path,
            )
        elif "aligner_output" in sig.parameters:
            raw_obs = extract_clip_signals(
                clip_path=clip_path,
                dataset_context=dataset_context,
                aligner_output=aligner_output,
            )
        else:
            raw_obs = extract_clip_signals(
                clip_path=clip_path,
                dataset_context=dataset_context,
            )
    except AttributeError as exc:
        msg = str(exc).lower()
        if "mediapipe" not in msg or "solutions" not in msg:
            raise
        raw_obs = extract_clip_signals_tasks(
            clip_path=clip_path,
            dataset_context=dataset_context,
            aligner_output=aligner_output,
            landmarker=tasks_landmarker,
            arcface_analyzer=arcface_analyzer,
        )

    if isinstance(raw_obs, ClipSignalObservation):
        return ClipSignalObservation(**raw_obs.model_dump())
    if isinstance(raw_obs, dict):
        return ClipSignalObservation(**raw_obs)
    raise ValueError(
        "extract_clip_signals returned unsupported type: "
        f"{type(raw_obs).__name__}"
    )


def build_evidence_dossier(obs: ClipSignalObservation) -> ClipEvidenceDossier:
    drift_map = {
        (0.0, 0.05): "none",
        (0.05, 0.15): "minor",
        (0.15, 0.30): "moderate",
        (0.30, 1.0): "severe",
    }
    drift_severity = "none"
    for (lo, hi), label in drift_map.items():
        if lo <= obs.identity_cosine_drift < hi or (
            label == "severe" and obs.identity_cosine_drift >= lo
        ):
            drift_severity = label
            break

    sync_map = {
        (0.7, 1.0): "good",
        (0.4, 0.7): "acceptable",
        (0.1, 0.4): "poor",
        (0.0, 0.1): "absent",
    }
    lip_sync = "absent"
    for (lo, hi), label in sync_map.items():
        if lo <= obs.lip_sync_confidence < hi or (
            label == "good" and obs.lip_sync_confidence >= lo
        ):
            lip_sync = label
            break

    redundancy = min(obs.similar_clips_accepted / 5.0, 1.0)
    impact = (
        "negative"
        if drift_severity in ["moderate", "severe"]
        else "positive"
        if obs.phoneme_coverage_new > 0.3
        else "neutral"
    )

    rejection_reason = None
    if drift_severity in ["moderate", "severe"]:
        rejection_reason = (
            f"identity cosine drift {obs.identity_cosine_drift:.3f} "
            f"exceeds threshold ({drift_severity})"
        )

    return ClipEvidenceDossier(
        clip_id=obs.clip_id,
        identity_drift_severity=drift_severity,
        temporal_instability_flag=obs.landmark_stability_score > 0.6,
        lip_sync_quality=lip_sync,
        unique_phoneme_value=obs.phoneme_coverage_new,
        dataset_redundancy_score=redundancy,
        estimated_training_impact=impact,
        primary_rejection_reason=rejection_reason,
        evidence_summary=(
            f"drift={drift_severity}, sync={lip_sync}, "
            f"phoneme_new={obs.phoneme_coverage_new:.2f}, "
            f"redundancy={redundancy:.2f}"
        ),
    )


def resolve_dossier_builder() -> Callable[[ClipSignalObservation], ClipEvidenceDossier]:
    from src import pipeline as pipeline_module

    fn = getattr(pipeline_module, "build_evidence_dossier", None)
    if callable(fn):
        return fn
    return build_evidence_dossier


def make_disposition_observation(
    dossier: ClipEvidenceDossier,
    accumulated_coverage: dict[str, int],
    accumulated_pose_dist: dict[str, int],
    clip_index: int,
    total_clips: int,
) -> ClipDispositionObservation:
    phoneme_gap_severity = {
        p: 1.0 for p in ARPABET_BASE if accumulated_coverage.get(p, 0) == 0
    }

    return ClipDispositionObservation(
        evidence_dossier=dossier,
        minimum_clips_needed=total_clips,
        phoneme_gap_severity=phoneme_gap_severity,
        pose_gap_severity=accumulated_pose_dist,
        budget_remaining=max(total_clips - clip_index - 1, 0),
        marginal_training_damage=0.0,
        marginal_coverage_gain=float(dossier.unique_phoneme_value),
    )


def build_case_entry(
    case_id: str,
    source_file: str,
    observation: ClipSignalObservation,
    dossier: ClipEvidenceDossier,
    action: ClipDispositionAction,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "source_file": source_file,
        "observation": observation.model_dump(),
        "ground_truth": {
            "disposition": "ANNOTATE",
            "confidence": 0.8,
            "disposition_ambiguity": 0.0,
            "valid_fix_steps": [],
            "valid_override_justifications": [],
            "expected_reasoning_elements": ["ANNOTATE"],
            "_annotation_notes": {
                "suggested_disposition": action.disposition,
                "suggested_confidence": float(action.confidence),
                "identity_drift_severity": dossier.identity_drift_severity,
                "lip_sync_quality": dossier.lip_sync_quality,
                "unique_phoneme_value": float(dossier.unique_phoneme_value),
                "dataset_redundancy_score": float(dossier.dataset_redundancy_score),
                "estimated_training_impact": dossier.estimated_training_impact,
                "primary_rejection_reason": dossier.primary_rejection_reason,
                "face_embedding_variance": float(observation.face_embedding_variance),
                "identity_cosine_drift": float(observation.identity_cosine_drift),
                "blur_score": float(observation.blur_score),
                "phoneme_sequence": list(observation.phoneme_sequence),
            },
        },
    }


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.clips.exists() or not args.clips.is_dir():
        raise SystemExit(f"--clips must be an existing directory: {args.clips}")

    if args.align_dir is not None and (not args.align_dir.exists() or not args.align_dir.is_dir()):
        print(
            f"Warning: --align-dir not found, continuing without alignments: {args.align_dir}",
            file=sys.stderr,
        )
        args.align_dir = None

    try:
        mediapipe_api = detect_mediapipe_api()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    tasks_landmarker: Any | None = None
    arcface_analyzer: Any | None = None

    if mediapipe_api == "tasks":
        model_path = args.landmarker_model
        if model_path is None:
            model_path = Path("data/models/face_landmarker.task")
        if not model_path.exists():
            try:
                model_path = download_landmarker_model(model_path)
            except Exception as exc:
                raise SystemExit(
                    "Failed to download face landmarker model. "
                    "Use --landmarker-model to provide a local .task file."
                ) from exc

        print(f"MediaPipe API: tasks (model: {model_path})")
        tasks_landmarker = make_face_mesh_tasks(str(model_path))
        arcface_analyzer = make_arcface_analyzer()
    else:
        print("MediaPipe API: solutions")

    video_files = list_video_files(args.clips)
    if not video_files:
        raise SystemExit(f"No video files found in {args.clips} (expected .mp4/.mov/.avi)")

    dossier_builder = resolve_dossier_builder()

    accumulated_coverage: dict[str, int] = {}
    accumulated_pose_dist: dict[str, int] = {}

    cases: list[dict[str, Any]] = []
    skipped = 0
    disposition_counts: Counter[str] = Counter({
        "accept": 0,
        "reject": 0,
        "fix": 0,
        "defer": 0,
    })

    for clip_index, clip_path in enumerate(video_files):
        dataset_context = {
            "current_phoneme_coverage": accumulated_coverage,
            "current_pose_distribution": accumulated_pose_dist,
            "clips_audited_so_far": clip_index,
            "similar_clips_accepted": 0,
        }

        forced_align_path = None
        if args.align_dir is not None:
            candidate = args.align_dir / f"{clip_path.stem}.json"
            if candidate.exists() and candidate.is_file():
                forced_align_path = candidate

        try:
            observation = call_extract_clip_signals(
                clip_path=clip_path,
                dataset_context=dataset_context,
                forced_align_path=forced_align_path,
                mediapipe_api=mediapipe_api,
                tasks_landmarker=tasks_landmarker,
                arcface_analyzer=arcface_analyzer,
            )
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            print(f"Warning: skipping {clip_path.name} - extraction failed: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if is_no_face_detected(observation):
            print(
                f"Warning: skipping {clip_path.name} - no face detected",
                file=sys.stderr,
            )
            skipped += 1
            continue

        try:
            dossier = dossier_builder(observation)
            if isinstance(dossier, ClipEvidenceDossier):
                dossier = ClipEvidenceDossier(**dossier.model_dump())
            elif isinstance(dossier, dict):
                dossier = ClipEvidenceDossier(**dossier)
            else:
                raise ValueError(
                    "dossier builder returned unsupported type: "
                    f"{type(dossier).__name__}"
                )

            disposition_obs = make_disposition_observation(
                dossier=dossier,
                accumulated_coverage=accumulated_coverage,
                accumulated_pose_dist=accumulated_pose_dist,
                clip_index=clip_index,
                total_clips=len(video_files),
            )
            action = recommend_clip_disposition(disposition_obs)
        except (ValidationError, ValueError) as exc:
            print(f"Warning: skipping {clip_path.name} - disposition failed: {exc}", file=sys.stderr)
            skipped += 1
            continue

        disposition_counts[action.disposition] += 1

        for phoneme in observation.phoneme_sequence:
            key = str(phoneme)
            accumulated_coverage[key] = accumulated_coverage.get(key, 0) + 1

        case_id = f"{len(cases) + 1:03d}"
        cases.append(
            build_case_entry(
                case_id=case_id,
                source_file=clip_path.name,
                observation=observation,
                dossier=dossier,
                action=action,
            )
        )

    if tasks_landmarker is not None and hasattr(tasks_landmarker, "close"):
        try:
            tasks_landmarker.close()
        except Exception:
            pass

    covered_arpabet = len({p for p in accumulated_coverage if p in ARPABET_SET})
    coverage_pct = (covered_arpabet / len(ARPABET_BASE) * 100.0) if ARPABET_BASE else 0.0

    payload = {
        "dataset_summary": {
            "total_clips": len(cases),
            "phoneme_coverage": accumulated_coverage,
            "suggested_accept": int(disposition_counts["accept"]),
            "suggested_reject": int(disposition_counts["reject"]),
            "suggested_fix": int(disposition_counts["fix"]),
            "suggested_defer": int(disposition_counts["defer"]),
        },
        "cases": cases,
    }

    write_output(args.output, payload)

    print(f"{len(cases)} cases extracted, {skipped} skipped")
    print(
        "Suggested dispositions: "
        f"accept={disposition_counts['accept']} "
        f"reject={disposition_counts['reject']} "
        f"fix={disposition_counts['fix']} "
        f"defer={disposition_counts['defer']}"
    )
    print(f"Phoneme coverage: {coverage_pct:.1f}% of ARPAbet")
    print(f"Wrote: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
