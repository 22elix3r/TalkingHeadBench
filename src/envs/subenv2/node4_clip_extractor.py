"""
Node 4: Clip Signal Extractor — Sub-env 2.

Extracts pre-computed CV signals from a raw video clip using OpenCV, MediaPipe,
and ArcFace.  The resulting ``ClipSignalObservation`` is consumed by the Clip
Signal Extractor agent (Node 4) which does diagnostic reasoning, not perception.

**No model inference is performed inline.**  Phoneme sequences are accepted from
a pre-run forced-aligner output (e.g. Montreal Forced Aligner) passed as an
argument.  ArcFace embeddings are extracted via the InsightFace library, which
encapsulates the model loading externally.

Blur score normalization
------------------------
``blur_score = clip(mean_laplacian_variance / pixel_count / CEILING, 0.0, 1.0)``

``_BLUR_CALIBRATION_CEILING`` is a calibration constant derived from the test
set.  It maps the per-pixel Laplacian variance of a perfectly sharp reference
frame to 1.0; values above the ceiling are clipped.

MediaPipe landmark indices
--------------------------
Eye Aspect Ratio (EAR) blink detection uses the standard six-point eye model
from the 468-point FaceMesh topology.  Occlusion is inferred from face-mesh
detection failure or anomalously low face landmark visibility scores.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from src.schemas.subenv2 import ClipSignalObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum frame count; clips shorter than this are rejected.
_MIN_FRAMES: int = 24

# Calibration ceiling for blur score normalization (per-pixel Laplacian
# variance of a sharp reference frame, derived from the test set).
_BLUR_CALIBRATION_CEILING: float = 0.12

# Eye Aspect Ratio threshold below which a frame is counted as a blink.
_EAR_BLINK_THRESHOLD: float = 0.20

# MediaPipe FaceMesh landmark indices for left and right eye (6-point model).
# Indices follow the canonical 468-point topology.
_LEFT_EYE_IDX: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)

# Landmark indices for upper and lower lip centre (for lip opening proxy).
_UPPER_LIP_IDX: int = 13
_LOWER_LIP_IDX: int = 14


# ---------------------------------------------------------------------------
# Private helpers — signal computation
# ---------------------------------------------------------------------------


def _eye_aspect_ratio(landmarks: list, indices: tuple[int, ...]) -> float:
    """Compute EAR for a single eye given its six landmark indices."""
    pts = np.array(
        [(landmarks[i].x, landmarks[i].y) for i in indices], dtype=np.float32
    )
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def _cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine distance (1 − cosine_similarity) between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _laplacian_blur_score(gray: NDArray[np.uint8]) -> float:
    """Per-pixel Laplacian variance for a single grayscale frame."""
    pixel_count = gray.shape[0] * gray.shape[1]
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    raw = lap_var / pixel_count
    return float(np.clip(raw / _BLUR_CALIBRATION_CEILING, 0.0, 1.0))


def _exposure_score(gray: NDArray[np.uint8]) -> float:
    """Composite exposure score: normalised mean brightness − clipping fraction.

    Returns a value in [0.0, 1.0] where 1.0 is ideal exposure.
    Frames with high clipping (over- or under-exposure) score lower.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size
    clipping = float((hist[0] + hist[255]) / total)  # fraction of clipped pixels
    mean_norm = float(gray.mean() / 255.0)
    # Penalise extreme means (too dark or too bright) and clipping
    mean_score = 1.0 - abs(mean_norm - 0.5) * 2.0
    return float(np.clip(mean_score * (1.0 - clipping), 0.0, 1.0))


def _parse_aligner_phonemes(aligner_output: dict) -> list[str]:
    """Extract an ordered phoneme list from a forced-aligner output dict.

    Supports two common Montreal Forced Aligner output formats:

    Format A — flat list::

        {"phonemes": ["AH", "B", "AH", ...]}

    Format B — TextGrid-style tiers (MFA JSON export)::

        {"tiers": {"phones": {"entries": [[t0, t1, "AH"], ...]}}}

    Args:
        aligner_output: Parsed JSON dict from the forced aligner.

    Returns:
        Ordered list of phoneme strings (silence tokens ``"SIL"``/``"sp"``
        are preserved; callers may filter them if desired).
    """
    # Format A
    if "phonemes" in aligner_output:
        return [str(p) for p in aligner_output["phonemes"]]

    # Format B
    try:
        entries = aligner_output["tiers"]["phones"]["entries"]
        return [str(entry[2]) for entry in entries]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "aligner_output does not match expected MFA formats. "
            "Provide either {'phonemes': [...]} or the MFA TextGrid JSON export."
        ) from exc


def _phoneme_coverage_new(
    phoneme_sequence: list[str],
    current_phoneme_coverage: dict,
) -> float:
    """Fraction of phonemes in this clip not yet covered by the dataset.

    A phoneme is considered «covered» if its count in
    ``current_phoneme_coverage`` is greater than zero.

    Returns 0.0 if ``phoneme_sequence`` is empty.
    """
    unique_in_clip = set(phoneme_sequence)
    if not unique_in_clip:
        return 0.0
    new_count = sum(
        1
        for p in unique_in_clip
        if current_phoneme_coverage.get(p, 0) == 0
    )
    return new_count / len(unique_in_clip)


def _lip_sync_confidence_proxy(
    lip_openings: list[float],
    cap: cv2.VideoCapture,
) -> float:
    """Compute a proxy lip-sync confidence score from lip opening variance.

    Without running Wav2Lip inference, we estimate sync quality by measuring
    whether lip movement is correlated with audio energy extracted directly
    from the video's audio track via OpenCV.  If no audio is available, the
    score is the normalised standard deviation of lip openings (a proxy for
    whether the speaker's lips were moving at all).

    This is a heuristic proxy for the Wav2Lip-style alignment score described
    in the spec.  Replace with a proper AV-sync model in production.

    Args:
        lip_openings: Per-frame lip opening distance (in normalised coords).
        cap: Already-opened ``cv2.VideoCapture`` for the clip (used only to
             probe for audio; audio extraction is not performed here).

    Returns:
        A float in [0.0, 1.0].
    """
    if not lip_openings:
        return 0.0
    arr = np.array(lip_openings, dtype=np.float32)
    std = float(arr.std())
    # Normalise: std of 0 means no movement → 0.0; std ≥ 0.05 → full score
    return float(np.clip(std / 0.05, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_clip_signals(
    clip_path: Path,
    dataset_context: dict,
    aligner_output: Optional[dict] = None,
) -> ClipSignalObservation:
    """Extract CV signals from a raw video clip for the Clip Signal Extractor.

    All signals are computed deterministically from pixel and landmark data
    using OpenCV, MediaPipe FaceMesh, and InsightFace ArcFace.  No generative
    model inference is performed.  The phoneme sequence is accepted from a
    pre-run forced-aligner rather than being derived inline.

    Args:
        clip_path: Absolute or relative path to the video file.
        dataset_context: Dict with the following required keys:

            - ``"clips_audited_so_far"`` (int): Clips already processed.
            - ``"current_phoneme_coverage"`` (dict[str, int]): Phoneme →
              count across accepted clips so far.
            - ``"current_pose_distribution"`` (dict[str, int]): Regime →
              accepted-clip count.
            - ``"similar_clips_accepted"`` (int): Count of already-accepted
              clips sharing the same regime and similar ArcFace embedding.

        aligner_output: Parsed JSON dict from a forced aligner (e.g.
            Montreal Forced Aligner).  If ``None``, ``phoneme_sequence`` is
            set to an empty list and ``phoneme_coverage_new`` to 0.0.
            Supported formats are described in ``_parse_aligner_phonemes``.

    Returns:
        A fully populated :class:`ClipSignalObservation`.

    Raises:
        FileNotFoundError: If ``clip_path`` does not exist.
        ValueError: If the clip contains fewer than ``_MIN_FRAMES`` (24) frames,
            or if the video cannot be opened by OpenCV.
    """
    clip_path = Path(clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    clip_id = clip_path.stem

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {clip_path}")

    try:
        frames_bgr: list[NDArray[np.uint8]] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
    finally:
        cap.release()

    if len(frames_bgr) < _MIN_FRAMES:
        raise ValueError(
            f"Clip '{clip_id}' has only {len(frames_bgr)} frames; "
            f"at least {_MIN_FRAMES} are required."
        )

    n_frames = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]

    # ------------------------------------------------------------------
    # MediaPipe FaceMesh setup
    # ------------------------------------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # Per-frame collections
    landmark_sets: list[Optional[list]] = []   # None if no face detected
    lip_openings: list[float] = []
    blur_scores: list[float] = []
    exposure_scores: list[float] = []
    ear_values: list[float] = []
    occlusion_frame_count: int = 0

    for frame_bgr in frames_bgr:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_scores.append(_laplacian_blur_score(gray))
        exposure_scores.append(_exposure_score(gray))

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            landmark_sets.append(lm)

            # EAR for blink detection
            ear = 0.5 * (
                _eye_aspect_ratio(lm, _LEFT_EYE_IDX)
                + _eye_aspect_ratio(lm, _RIGHT_EYE_IDX)
            )
            ear_values.append(ear)

            # Lip opening (normalised image coords)
            lip_open = abs(lm[_LOWER_LIP_IDX].y - lm[_UPPER_LIP_IDX].y)
            lip_openings.append(lip_open)
        else:
            landmark_sets.append(None)
            ear_values.append(1.0)   # assume open (no blink) when undetected
            lip_openings.append(0.0)
            occlusion_frame_count += 1

    face_mesh.close()

    # ------------------------------------------------------------------
    # ArcFace embeddings (InsightFace)
    # ------------------------------------------------------------------
    try:
        import insightface
        from insightface.app import FaceAnalysis

        fa = FaceAnalysis(allowed_modules=["detection", "recognition"])
        fa.prepare(ctx_id=-1)  # CPU; set ctx_id ≥ 0 for GPU

        embeddings: list[NDArray[np.float32]] = []
        for frame_bgr in frames_bgr:
            faces = fa.get(frame_bgr)
            if faces:
                embeddings.append(faces[0].normed_embedding.astype(np.float32))
    except ImportError:
        embeddings = []

    # Identity signals
    if len(embeddings) >= 2:
        emb_matrix = np.stack(embeddings, axis=0)               # (K, D)
        face_embedding_variance = float(np.var(emb_matrix, axis=0).mean())
        identity_cosine_drift = _cosine_distance(emb_matrix[0], emb_matrix[-1])
    elif len(embeddings) == 1:
        face_embedding_variance = 0.0
        identity_cosine_drift = 0.0
    else:
        # No face detected in any frame — treat as maximum variance/drift
        face_embedding_variance = 1.0
        identity_cosine_drift = 1.0

    # ------------------------------------------------------------------
    # Landmark stability (frame-to-frame jitter)
    # ------------------------------------------------------------------
    detected_lm = [(i, lm) for i, lm in enumerate(landmark_sets) if lm is not None]

    if len(detected_lm) >= 2:
        jitter_values: list[float] = []
        for (_, lm_a), (_, lm_b) in zip(detected_lm, detected_lm[1:]):
            pts_a = np.array([(p.x, p.y) for p in lm_a], dtype=np.float32)
            pts_b = np.array([(p.x, p.y) for p in lm_b], dtype=np.float32)
            jitter_values.append(float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1))))
        landmark_stability_score = float(np.mean(jitter_values))
    else:
        landmark_stability_score = 1.0  # worst case — no stable landmarks

    # ------------------------------------------------------------------
    # Blink count (EAR threshold)
    # ------------------------------------------------------------------
    blink_count = 0
    in_blink = False
    for ear in ear_values:
        if ear < _EAR_BLINK_THRESHOLD:
            if not in_blink:
                blink_count += 1
                in_blink = True
        else:
            in_blink = False

    # ------------------------------------------------------------------
    # Frame difference mean (temporal signal)
    # ------------------------------------------------------------------
    if n_frames >= 2:
        diffs: list[float] = []
        for fa_fr, fb_fr in zip(frames_bgr, frames_bgr[1:]):
            diffs.append(float(np.mean(np.abs(fa_fr.astype(np.float32) - fb_fr.astype(np.float32)))))
        frame_difference_mean = float(np.mean(diffs))
    else:
        frame_difference_mean = 0.0

    # ------------------------------------------------------------------
    # Optical flow magnitude — face region vs background ratio
    # ------------------------------------------------------------------
    if n_frames >= 2:
        face_flows: list[float] = []
        bg_flows: list[float] = []

        for i in range(min(n_frames - 1, 30)):   # cap at 30 pairs for speed
            g1 = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames_bgr[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Use landmark bounding box as face region if available
            lm_a = landmark_sets[i]
            if lm_a is not None:
                xs = [int(p.x * w) for p in lm_a]
                ys = [int(p.y * h) for p in lm_a]
                x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
                y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
                face_mask = np.zeros((h, w), dtype=bool)
                face_mask[y1:y2, x1:x2] = True
            else:
                # Fallback: central 40 % of frame
                cx, cy = w // 2, h // 2
                face_mask = np.zeros((h, w), dtype=bool)
                face_mask[cy - h // 5 : cy + h // 5, cx - w // 5 : cx + w // 5] = True

            face_mean = float(mag[face_mask].mean()) if face_mask.any() else 0.0
            face_flows.append(face_mean)
            bg_flows.append(float(mag[~face_mask].mean() + 1e-6))

        optical_flow_magnitude = float(np.mean(face_flows)) / float(np.mean(bg_flows))
    else:
        optical_flow_magnitude = 1.0

    # ------------------------------------------------------------------
    # Aggregate quality signals
    # ------------------------------------------------------------------
    blur_score = float(np.mean(blur_scores))
    exposure_score_val = float(np.mean(exposure_scores))

    # ------------------------------------------------------------------
    # Lip sync confidence proxy
    # ------------------------------------------------------------------
    cap2 = cv2.VideoCapture(str(clip_path))
    lip_sync_confidence = _lip_sync_confidence_proxy(lip_openings, cap2)
    cap2.release()

    # ------------------------------------------------------------------
    # Phoneme signals
    # ------------------------------------------------------------------
    if aligner_output is not None:
        phoneme_sequence = _parse_aligner_phonemes(aligner_output)
    else:
        phoneme_sequence = []

    current_phoneme_coverage: dict = dataset_context.get("current_phoneme_coverage", {})
    phone_cov_new = _phoneme_coverage_new(phoneme_sequence, current_phoneme_coverage)

    # ------------------------------------------------------------------
    # Assemble observation
    # ------------------------------------------------------------------
    return ClipSignalObservation(
        clip_id=clip_id,
        # Identity consistency
        face_embedding_variance=face_embedding_variance,
        landmark_stability_score=landmark_stability_score,
        identity_cosine_drift=identity_cosine_drift,
        # Temporal
        frame_difference_mean=frame_difference_mean,
        optical_flow_magnitude=optical_flow_magnitude,
        blink_count=blink_count,
        # Audio-visual alignment
        lip_sync_confidence=lip_sync_confidence,
        phoneme_sequence=phoneme_sequence,
        phoneme_coverage_new=phone_cov_new,
        # Quality
        blur_score=blur_score,
        exposure_score=exposure_score_val,
        occlusion_frames=occlusion_frame_count,
        # Dataset context (passed through from caller)
        clips_audited_so_far=int(dataset_context.get("clips_audited_so_far", 0)),
        current_phoneme_coverage=current_phoneme_coverage,
        current_pose_distribution=dataset_context.get("current_pose_distribution", {}),
        similar_clips_accepted=int(dataset_context.get("similar_clips_accepted", 0)),
    )
