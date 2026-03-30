#!/usr/bin/env python3
"""Extract Sub-env 1 image signals and build annotation-ready case JSON.

This script scans a directory of reference images, computes all
ImageDiagnosticsObservation fields expected by Node 1, runs the Node 1
heuristic for suggested annotations, and writes a partially populated
subenv1_cases.json for human review.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.subenv1.node1_image_diagnostician import diagnose_image
from src.schemas.subenv1 import ImageDiagnosticsObservation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

POSE_LANDMARK_IDS = (1, 4, 454, 234)

LEFT_IRIS_INDICES = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES = (473, 474, 475, 476, 477)

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Sub-env 1 observation signals from reference images and "
            "emit annotation-ready JSON cases."
        )
    )
    parser.add_argument("--images", required=True, type=Path, help="Folder of images")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path (for example tests/test_set/subenv1_cases.json)",
    )
    parser.add_argument(
        "--prompt-file",
        required=False,
        type=Path,
        help="Optional prompt file; supports 'stem: prompt' lines.",
    )
    parser.add_argument(
        "--landmarker-model",
        required=False,
        type=Path,
        default=None,
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


def get_landmarks_tasks(landmarker: Any, rgb_image: np.ndarray) -> tuple[np.ndarray | None, float, Any]:
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image.copy())
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, 0.0, None

    lm = result.face_landmarks[0]
    h, w = rgb_image.shape[:2]
    landmarks_array = np.array([[l.x * w, l.y * h, l.z * w] for l in lm], dtype=np.float32)

    if not getattr(get_landmarks_tasks, "_debug_printed", False) and len(lm) > 0:
        print(f"  Image shape: {rgb_image.shape} -> w={w}, h={h}")
        print(f"  Landmark 0 raw: ({lm[0].x:.4f}, {lm[0].y:.4f})")
        print(f"  Landmark 0 px:  ({lm[0].x * w:.1f}, {lm[0].y * h:.1f})")
        setattr(get_landmarks_tasks, "_debug_printed", True)

    confidence = 1.0

    transform_matrix = None
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if matrices:
        transform_matrix = matrices[0]

    return landmarks_array, confidence, transform_matrix


def extract_pose_from_transform_matrix(matrix: Any) -> tuple[float, float]:
    """Extract yaw and pitch in degrees from a 4x4 transform matrix."""
    m = np.array(matrix.data, dtype=np.float64).reshape(4, 4)

    yaw = math.degrees(math.atan2(m[0, 2], m[2, 2]))
    pitch = math.degrees(
        math.atan2(-m[1, 2], math.sqrt(m[0, 2] ** 2 + m[2, 2] ** 2))
    )
    return float(yaw), float(pitch)


def make_face_mesh_solutions() -> Any:
    face_mesh_mod = importlib.import_module("mediapipe.python.solutions.face_mesh")

    return face_mesh_mod.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def estimate_face_confidence_from_solution_landmarks(landmarks: list[Any]) -> float:
    values: list[float] = []
    for idx in POSE_LANDMARK_IDS:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        for attr in ("visibility", "presence"):
            if hasattr(lm, attr):
                value = getattr(lm, attr)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))

    if values and any(v > 0.0 for v in values):
        return float(clamp(float(np.mean(values)), 0.0, 1.0))
    return 1.0


def get_landmarks_solutions(face_mesh: Any, rgb_image: np.ndarray) -> tuple[np.ndarray | None, float, Any]:
    result = face_mesh.process(rgb_image)
    if not result.multi_face_landmarks:
        return None, 0.0, None

    landmarks = list(result.multi_face_landmarks[0].landmark)
    h, w = rgb_image.shape[:2]
    landmarks_array = np.array([[l.x * w, l.y * h, l.z * w] for l in landmarks], dtype=np.float32)
    confidence = estimate_face_confidence_from_solution_landmarks(landmarks)
    return landmarks_array, confidence, None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def list_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def load_prompt_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Load optional prompt file.

    Supported formats:
    - "stem: prompt text"
    - plain prompt lines (fallback to sequential matching)
    """
    stem_map: dict[str, str] = {}
    sequential: list[str] = []

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            stem, text = line.split(":", 1)
            stem = stem.strip().lower()
            text = text.strip()
            if stem and text:
                stem_map[stem] = text
                continue

        sequential.append(line)

    return stem_map, sequential


def get_prompt_for_stem(
    stem: str,
    index: int,
    stem_map: dict[str, str],
    sequential_prompts: list[str],
) -> str:
    stem_l = stem.lower()
    candidates = [stem_l]

    if stem_l.startswith("ref_"):
        candidates.append(f"clip_{stem_l[4:]}")
    elif stem_l.startswith("clip_"):
        candidates.append(f"ref_{stem_l[5:]}")

    for candidate in candidates:
        if candidate in stem_map:
            return stem_map[candidate]

    if index < len(sequential_prompts):
        return sequential_prompts[index]

    return ""


def maybe_load_clip_tokenizer() -> Any:
    try:
        transformers_mod = importlib.import_module("transformers")
    except Exception:
        return None

    clip_tokenizer_cls = getattr(transformers_mod, "CLIPTokenizer", None)
    if clip_tokenizer_cls is None:
        return None

    try:
        return clip_tokenizer_cls.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
    except Exception:
        return None


def maybe_load_spacy_model() -> Any:
    try:
        spacy_mod = importlib.import_module("spacy")
    except Exception:
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy_mod.load(model_name)
        except Exception:
            continue

    return None


def prompt_token_count(prompt: str, clip_tokenizer: Any) -> int:
    if not prompt:
        return 0

    if clip_tokenizer is not None:
        try:
            return int(len(clip_tokenizer.encode(prompt, add_special_tokens=False)))
        except Exception:
            pass

    approx = int(round(len(prompt.split()) * 1.3))
    return max(0, approx)


def prompt_semantic_density(prompt: str, nlp: Any) -> float:
    if not prompt:
        return 0.0

    if nlp is not None:
        try:
            doc = nlp(prompt)
            tokens = [t for t in doc if not t.is_space and not t.is_punct]
            if not tokens:
                return 0.0
            concept_set = {
                t.lemma_.lower()
                for t in tokens
                if t.pos_ in {"NOUN", "ADJ", "PROPN"} and t.lemma_
            }
            return float(clamp(len(concept_set) / len(tokens), 0.0, 1.0))
        except Exception:
            pass

    words = re.findall(r"[A-Za-z0-9']+", prompt.lower())
    if not words:
        return 0.0
    return float(clamp(len(set(words)) / len(words), 0.0, 1.0))


def compute_face_bbox(landmarks_px: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if landmarks_px.size == 0:
        return 0, 0, width, height

    x_coords = landmarks_px[:, 0]
    y_coords = landmarks_px[:, 1]

    w = width
    assert x_coords.max() > 1.5, (
        f"Landmarks still appear normalized - max x={x_coords.max():.4f}. "
        f"Expected pixel coords for a {w}px wide image."
    )

    xs = x_coords
    ys = y_coords

    x0 = int(clamp(math.floor(min(xs)), 0, width - 1))
    y0 = int(clamp(math.floor(min(ys)), 0, height - 1))
    x1 = int(clamp(math.ceil(max(xs)), 1, width))
    y1 = int(clamp(math.ceil(max(ys)), 1, height))

    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)

    return x0, y0, x1, y1


def compute_face_occupancy(
    landmarks_px: np.ndarray,
    img_shape: tuple,
) -> float:
    h, w = img_shape[:2]

    # Tight face bbox from landmarks
    x_min = landmarks_px[:, 0].min()
    x_max = landmarks_px[:, 0].max()
    y_min = landmarks_px[:, 1].min()
    y_max = landmarks_px[:, 1].max()

    face_w = x_max - x_min
    face_h = y_max - y_min

    # Use the larger of face width or height as the head diameter
    # then compute a head-region area with 1.4x expansion to
    # approximate the full head including hair and chin
    head_diameter = max(face_w, face_h) * 1.4
    head_area = head_diameter ** 2
    image_area = w * h

    return float(min(head_area / image_area, 1.0))


def normalise_occupancy_for_spec(raw_occupancy: float) -> float:
    """Map real occupancy values into the scale expected by spec thresholds."""
    if raw_occupancy <= 0.08:
        return raw_occupancy * (0.25 / 0.08)
    if raw_occupancy <= 0.30:
        t = (raw_occupancy - 0.08) / (0.30 - 0.08)
        return 0.25 + t * (0.60 - 0.25)
    t = (raw_occupancy - 0.30) / 0.70
    return min(0.60 + t * 0.40, 1.0)


def rotation_matrix_to_euler_xyz(rot_mat: np.ndarray) -> tuple[float, float, float]:
    """Return Euler angles (x=pitch, y=yaw, z=roll) in radians."""
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch_x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw_y = math.atan2(-rot_mat[2, 0], sy)
        roll_z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        pitch_x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
        yaw_y = math.atan2(-rot_mat[2, 0], sy)
        roll_z = 0.0

    return pitch_x, yaw_y, roll_z


def estimate_pose_from_landmarks(landmarks_array: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Estimate yaw and pitch from FaceMesh landmarks via solvePnP."""
    if landmarks_array.size == 0 or any(idx >= len(landmarks_array) for idx in POSE_LANDMARK_IDS):
        return 0.0, 0.0

    image_points = np.array(
        [landmarks_array[idx, :2] for idx in POSE_LANDMARK_IDS],
        dtype=np.float64,
    )

    # Approximate canonical face model points for indices 1, 4, 454, 234.
    model_points = np.array(
        [
            [0.0, 0.0, 0.0],      # 1
            [0.0, -63.6, -12.5],  # 4
            [43.3, 32.7, -26.0],  # 454
            [-43.3, 32.7, -26.0], # 234
        ],
        dtype=np.float64,
    )

    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        success, rot_vec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
    if not success:
        return 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pitch_x, yaw_y, _ = rotation_matrix_to_euler_xyz(rot_mat)

    yaw_deg = float(np.degrees(yaw_y))
    pitch_deg = float(np.degrees(pitch_x))
    return yaw_deg, pitch_deg


def compute_background_complexity(
    img_bgr: np.ndarray,
    landmarks_px: np.ndarray,
) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x_min = max(0, int(landmarks_px[:, 0].min()))
    x_max = min(w, int(landmarks_px[:, 0].max()))
    y_min = max(0, int(landmarks_px[:, 1].min()))
    y_max = min(h, int(landmarks_px[:, 1].max()))

    if not getattr(compute_background_complexity, "_bbox_debug_printed", False):
        bbox_area = (x_max - x_min) * (y_max - y_min)
        image_area = img_bgr.shape[1] * img_bgr.shape[0]
        print(f"  Total landmarks: {len(landmarks_px)}")
        print(
            f"  x range: {landmarks_px[:,0].min():.1f} -> {landmarks_px[:,0].max():.1f}"
        )
        print(
            f"  y range: {landmarks_px[:,1].min():.1f} -> {landmarks_px[:,1].max():.1f}"
        )
        print(f"  bbox: ({x_min},{y_min}) -> ({x_max},{y_max})")
        print(f"  bbox area: {bbox_area} px²")
        print(f"  image area: {image_area} px²")
        print(f"  occupancy ratio: {bbox_area / image_area:.4f}")
        setattr(compute_background_complexity, "_bbox_debug_printed", True)

    # Expand bbox by 20% for margin
    pad_x = int((x_max - x_min) * 0.2)
    pad_y = int((y_max - y_min) * 0.2)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    # Mask out face region - background is everything outside
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y_min:y_max, x_min:x_max] = 0

    edges = cv2.Canny(gray, 50, 150)
    background_edges = cv2.bitwise_and(edges, edges, mask=mask)
    background_pixels = int(mask.sum()) // 255
    if background_pixels == 0:
        return 0.0

    raw_density = float(int(background_edges.sum()) // 255) / background_pixels
    raw_print_count = getattr(compute_background_complexity, "_raw_density_print_count", 0)
    if raw_print_count < 5:
        print(f"  Raw background edge density: {raw_density:.6f}")
        setattr(compute_background_complexity, "_raw_density_print_count", raw_print_count + 1)

    return raw_density


def normalise_background_complexity(raw: float) -> float:
    """Scale real edge-density values into the spec threshold range."""
    return min(raw * 23.3, 1.0)


def lighting_uniformity_in_face(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0

    std_dev = float(region.std())
    normalized = clamp(std_dev / 64.0, 0.0, 1.0)
    return float(1.0 - normalized)


def skin_tone_bucket_from_face(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    face = image_bgr[y0:y1, x0:x1]
    if face.size == 0:
        return 3

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    skin_like = (s_chan > 20) & (v_chan > 40)
    if np.any(skin_like):
        hue_vals = h_chan[skin_like].reshape(-1)
        val_vals = v_chan[skin_like].reshape(-1)
    else:
        hue_vals = h_chan.reshape(-1)
        val_vals = v_chan.reshape(-1)

    if hue_vals.size == 0:
        return 3

    hist = np.bincount(hue_vals.astype(np.int32), minlength=180)
    dominant_hue = int(hist.argmax())
    mean_v = float(val_vals.mean()) if val_vals.size else 128.0

    # Fitzpatrick-ish approximation using brightness with a hue sanity adjustment.
    if mean_v >= 200:
        bucket = 1
    elif mean_v >= 170:
        bucket = 2
    elif mean_v >= 140:
        bucket = 3
    elif mean_v >= 110:
        bucket = 4
    elif mean_v >= 80:
        bucket = 5
    else:
        bucket = 6

    if dominant_hue < 5 or dominant_hue > 35:
        bucket = min(6, bucket + 1)

    return int(clamp(bucket, 1, 6))


def iris_obscured(landmarks_array: np.ndarray) -> bool:
    max_iris_idx = max(RIGHT_IRIS_INDICES)
    if len(landmarks_array) <= max_iris_idx:
        return True

    def pix(idx: int) -> np.ndarray:
        return landmarks_array[idx, :2].astype(np.float32)

    left_iris = np.array([pix(i) for i in LEFT_IRIS_INDICES], dtype=np.float32)
    right_iris = np.array([pix(i) for i in RIGHT_IRIS_INDICES], dtype=np.float32)

    left_area = float(cv2.contourArea(left_iris))
    right_area = float(cv2.contourArea(right_iris))

    left_eye_w = float(np.linalg.norm(pix(LEFT_EYE_CORNERS[0]) - pix(LEFT_EYE_CORNERS[1])))
    right_eye_w = float(np.linalg.norm(pix(RIGHT_EYE_CORNERS[0]) - pix(RIGHT_EYE_CORNERS[1])))

    if left_eye_w < 1.0 or right_eye_w < 1.0:
        return True

    left_min_area = 0.01 * (left_eye_w ** 2)
    right_min_area = 0.01 * (right_eye_w ** 2)

    return left_area < left_min_area or right_area < right_min_area


def estimated_sharpness(gray: np.ndarray, width: int, height: int) -> float:
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    normalized = (lap_var / float(width * height)) / 0.001
    return float(clamp(normalized, 0.0, 1.0))


def build_case_entry(
    case_id: int,
    image_path: Path,
    observation: ImageDiagnosticsObservation,
) -> dict[str, Any]:
    suggestion = diagnose_image(observation)

    return {
        "id": f"{case_id:03d}",
        "source_file": image_path.name,
        "observation": observation.model_dump(),
        "ground_truth": {
            "regime_classification": "ANNOTATE",
            "acceptable_regimes": [],
            "identified_risk_factors": ["ANNOTATE"],
            "valid_prompt_modifications": ["ANNOTATE"],
            "_annotation_notes": {
                "computed_yaw": round(float(observation.estimated_yaw_degrees), 4),
                "computed_occupancy": round(float(observation.face_occupancy_ratio), 4),
                "computed_sharpness": round(float(observation.estimated_sharpness), 4),
                "suggested_regime": suggestion.regime_classification,
                "suggested_risk_factors": suggestion.identified_risk_factors,
                "image_usability_score": suggestion.image_usability_score,
            },
        },
    }


def main() -> int:
    args = parse_args()

    if not args.images.exists() or not args.images.is_dir():
        raise SystemExit(f"--images must be an existing directory: {args.images}")

    image_files = list_image_files(args.images)
    if not image_files:
        raise SystemExit(f"No image files found in {args.images}")

    prompt_map: dict[str, str] = {}
    sequential_prompts: list[str] = []
    if args.prompt_file is not None:
        if not args.prompt_file.exists():
            raise SystemExit(f"Prompt file not found: {args.prompt_file}")
        prompt_map, sequential_prompts = load_prompt_file(args.prompt_file)

    clip_tokenizer = maybe_load_clip_tokenizer()
    nlp = maybe_load_spacy_model()

    try:
        api = detect_mediapipe_api()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    detector: Any = None
    model_path: Path | None = None

    if api == "solutions":
        detector = make_face_mesh_solutions()
        print("MediaPipe API: solutions")
    else:
        if args.landmarker_model is None:
            model_path = Path("data/models/face_landmarker.task")
            if not model_path.exists():
                try:
                    model_path = download_landmarker_model(model_path)
                except Exception as exc:
                    raise SystemExit(
                        "Failed to download face landmarker model. "
                        "Use --landmarker-model to provide a local .task file."
                    ) from exc
        else:
            model_path = args.landmarker_model

        if model_path is None or not model_path.exists():
            raise SystemExit(f"Landmarker model not found: {model_path}")

        print(f"Using MediaPipe Tasks API with model: {model_path}")
        print(f"MediaPipe API: tasks (model: {model_path})")
        detector = make_face_mesh_tasks(str(model_path))

    def process_image(rgb_image: np.ndarray) -> tuple[np.ndarray | None, float, float, float]:
        if api == "solutions":
            landmarks_array, confidence, transform_matrix = get_landmarks_solutions(
                detector, rgb_image
            )
        else:
            landmarks_array, confidence, transform_matrix = get_landmarks_tasks(
                detector, rgb_image
            )

        if landmarks_array is None:
            return None, 0.0, 0.0, 0.0

        h, w = rgb_image.shape[:2]

        if transform_matrix is not None:
            try:
                yaw_deg, pitch_deg = extract_pose_from_transform_matrix(transform_matrix)
            except Exception:
                yaw_deg, pitch_deg = estimate_pose_from_landmarks(landmarks_array, w, h)
        else:
            yaw_deg, pitch_deg = estimate_pose_from_landmarks(landmarks_array, w, h)

        return landmarks_array, confidence, yaw_deg, pitch_deg

    extracted_cases: list[dict[str, Any]] = []
    skipped_no_face = 0
    landmark_debug_printed = False
    shape_debug_print_count = 0

    try:
        for file_index, image_path in enumerate(image_files):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            assert image_bgr is not None and image_bgr.size > 0, f"Failed to load {image_path}"

            if shape_debug_print_count < 3:
                print(f"Processing {image_path.name}: shape={image_bgr.shape}")
                shape_debug_print_count += 1

            height, width = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            landmarks_array, confidence, yaw_deg, pitch_deg = process_image(image_rgb)

            if landmarks_array is None:
                skipped_no_face += 1
                print(
                    f"Warning: no face detected, skipping: {image_path.name}",
                    file=sys.stderr,
                )
                continue

            if not landmark_debug_printed:
                l0x = float(landmarks_array[0, 0])
                l0y = float(landmarks_array[0, 1])
                print(f"Landmark 0: ({l0x:.1f}, {l0y:.1f}) px")
                landmark_debug_printed = True

            bbox = compute_face_bbox(landmarks_array, width, height)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            raw_occupancy = compute_face_occupancy(landmarks_array, image_bgr.shape)
            raw_bg_complexity = compute_background_complexity(image_bgr, landmarks_array)
            occupancy = normalise_occupancy_for_spec(raw_occupancy)
            bg_complexity = normalise_background_complexity(raw_bg_complexity)
            light_uniformity = lighting_uniformity_in_face(gray, bbox)
            skin_bucket = skin_tone_bucket_from_face(image_bgr, bbox)

            occluded = iris_obscured(landmarks_array) or confidence < 0.7

            sharpness = estimated_sharpness(gray, width, height)

            prompt = get_prompt_for_stem(
                image_path.stem,
                file_index,
                prompt_map,
                sequential_prompts,
            )
            token_count = prompt_token_count(prompt, clip_tokenizer)
            semantic_density = prompt_semantic_density(prompt, nlp)

            try:
                observation = ImageDiagnosticsObservation(
                    face_occupancy_ratio=float(occupancy),
                    estimated_yaw_degrees=float(yaw_deg),
                    estimated_pitch_degrees=float(pitch_deg),
                    background_complexity_score=float(bg_complexity),
                    lighting_uniformity_score=float(light_uniformity),
                    skin_tone_bucket=int(skin_bucket),
                    occlusion_detected=bool(occluded),
                    image_resolution=(int(width), int(height)),
                    estimated_sharpness=float(sharpness),
                    prompt_token_count=int(token_count),
                    prompt_semantic_density=float(semantic_density),
                    conflicting_descriptors=[],
                    identity_anchoring_strength=0.5,
                )
            except ValidationError as exc:
                raise SystemExit(
                    f"Validation failed for {image_path.name}: {exc}"
                ) from exc

            case_entry = build_case_entry(
                case_id=len(extracted_cases) + 1,
                image_path=image_path,
                observation=observation,
            )
            extracted_cases.append(case_entry)
    finally:
        if detector is not None and hasattr(detector, "close"):
            detector.close()

    payload = {"cases": extracted_cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{len(extracted_cases)} cases extracted, {skipped_no_face} skipped (no face detected)")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
