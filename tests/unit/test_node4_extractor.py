"""
Unit tests for Node 4: Clip Signal Extractor
(src/envs/subenv2/node4_clip_extractor.py)

All heavy dependencies — OpenCV VideoCapture, InsightFace, and MediaPipe
FaceMesh — are fully mocked so no real video files or GPU/model weights
are required.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schemas.subenv2 import ClipSignalObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _random_frame(height: int = 72, width: int = 128) -> np.ndarray:
    """Return a random uint8 BGR frame."""
    return _RNG.integers(0, 256, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixture: mock_capture
# ---------------------------------------------------------------------------


def mock_capture(frame_count: int, height: int = 72, width: int = 128) -> MagicMock:
    """Return a MagicMock that behaves like cv2.VideoCapture.

    * ``get(cv2.CAP_PROP_FRAME_COUNT)`` → *frame_count*
    * ``get(cv2.CAP_PROP_FPS)``         → 24.0
    * ``read()`` cycles through *frame_count* random numpy frames then
      returns ``(False, None)``
    * ``isOpened()``                     → True
    * ``release()``                      → no-op
    """
    import cv2  # noqa: PLC0415

    frames = [_random_frame(height, width) for _ in range(frame_count)]
    read_returns = [(True, f) for f in frames] + [(False, None)]
    read_iter = iter(read_returns)

    cap = MagicMock()
    cap.isOpened.return_value = True

    def _get(prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(frame_count)
        if prop_id == cv2.CAP_PROP_FPS:
            return 24.0
        return 0.0

    cap.get.side_effect = _get
    cap.read.side_effect = lambda: next(read_iter)
    cap.release.return_value = None
    return cap


# ---------------------------------------------------------------------------
# Shared MediaPipe / InsightFace mock builders
# ---------------------------------------------------------------------------


def _make_face_mesh_mock() -> MagicMock:
    """Return a MagicMock mimicking mediapipe FaceMesh.

    Each call to ``process()`` returns a result with a single detected face
    carrying 478 stable landmarks (x=0.5, y=0.5, z=0.0).
    """
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    lm.z = 0.0

    face_landmark = MagicMock()
    face_landmark.landmark = [lm] * 478

    mp_result = MagicMock()
    mp_result.multi_face_landmarks = [face_landmark]

    face_mesh = MagicMock()
    face_mesh.process.return_value = mp_result
    face_mesh.close.return_value = None
    return face_mesh


def _make_mediapipe_mock(face_mesh_instance: MagicMock) -> MagicMock:
    """Build a fake ``mp`` module alias as imported in node4_clip_extractor."""
    mp_mock = MagicMock()
    mp_mock.solutions.face_mesh.FaceMesh.return_value = face_mesh_instance
    return mp_mock


def _make_insightface_modules() -> dict:
    """Return a sys.modules patch dict for insightface.

    ``FaceAnalysis.get()`` returns a single face with a unit 512-D embedding.
    """
    embedding = np.ones(512, dtype=np.float32)

    face = MagicMock()
    face.normed_embedding = embedding

    fa_instance = MagicMock()
    fa_instance.get.return_value = [face]
    fa_instance.prepare.return_value = None

    FaceAnalysis = MagicMock(return_value=fa_instance)

    insightface_mod = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = FaceAnalysis
    insightface_mod.app = app_mod

    return {
        "insightface": insightface_mod,
        "insightface.app": app_mod,
    }


# ---------------------------------------------------------------------------
# Common dataset context skeleton
# ---------------------------------------------------------------------------

_EMPTY_CTX: dict = {
    "current_phoneme_coverage": {},
    "current_pose_distribution": {},
    "clips_audited_so_far": 0,
    "similar_clips_accepted": 0,
}


# ---------------------------------------------------------------------------
# Context manager: full env patch
# ---------------------------------------------------------------------------


def _full_patch(cap_mock: MagicMock, mp_mock: MagicMock):
    """Return a combined context manager that patches cv2, insightface, and mp."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("cv2.VideoCapture", return_value=cap_mock))
    stack.enter_context(patch.dict("sys.modules", _make_insightface_modules()))
    stack.enter_context(
        patch("src.envs.subenv2.node4_clip_extractor.mp", mp_mock)
    )
    return stack


# ---------------------------------------------------------------------------
# Test 1 — short clip raises ValueError mentioning the minimum frame count
# ---------------------------------------------------------------------------


def test_raises_on_short_clip(tmp_path):
    """Clips with fewer than 24 frames must raise ValueError containing '24'."""
    dummy = tmp_path / "dummy.mp4"
    dummy.touch()

    cap = mock_capture(10)

    with patch("cv2.VideoCapture", return_value=cap):
        with pytest.raises(ValueError, match="24"):
            from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
            extract_clip_signals(dummy, {})


# ---------------------------------------------------------------------------
# Test 2 — blur_score is in [0, 1] and result is ClipSignalObservation
# ---------------------------------------------------------------------------


def test_blur_score_in_range(tmp_path):
    """blur_score must be in [0.0, 1.0] and return type must be ClipSignalObservation."""
    dummy = tmp_path / "dummy.mp4"
    dummy.touch()

    cap30 = mock_capture(30)
    mp_mock = _make_mediapipe_mock(_make_face_mesh_mock())

    with _full_patch(cap30, mp_mock):
        from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
        result = extract_clip_signals(dummy, _EMPTY_CTX)

    assert isinstance(result, ClipSignalObservation)
    assert 0.0 <= result.blur_score <= 1.0


# ---------------------------------------------------------------------------
# Test 3 — phoneme_coverage_new == 1.0 when dataset phoneme coverage is empty
# ---------------------------------------------------------------------------


def test_phoneme_coverage_new_empty_dataset(tmp_path):
    """All phonemes in the clip are new when the dataset coverage is empty."""
    dummy = tmp_path / "dummy.mp4"
    dummy.touch()

    aligner_data = {"phonemes": ["AH", "EE", "OW"]}
    aligner_json = tmp_path / "align.json"
    aligner_json.write_text(json.dumps(aligner_data))

    cap30 = mock_capture(30)
    mp_mock = _make_mediapipe_mock(_make_face_mesh_mock())
    ctx = {**_EMPTY_CTX, "current_phoneme_coverage": {}}

    with _full_patch(cap30, mp_mock):
        from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
        result = extract_clip_signals(
            dummy,
            ctx,
            aligner_output=json.loads(aligner_json.read_text()),
        )

    assert result.phoneme_coverage_new == 1.0


# ---------------------------------------------------------------------------
# Test 4 — phoneme_coverage_new ≈ 2/3 when one phoneme already covered
# ---------------------------------------------------------------------------


def test_phoneme_coverage_new_partial(tmp_path):
    """When one of three unique phonemes is already covered, new coverage = 2/3."""
    dummy = tmp_path / "dummy.mp4"
    dummy.touch()

    aligner_data = {"phonemes": ["AH", "EE", "OW"]}
    aligner_json = tmp_path / "align.json"
    aligner_json.write_text(json.dumps(aligner_data))

    cap30 = mock_capture(30)
    mp_mock = _make_mediapipe_mock(_make_face_mesh_mock())
    ctx = {**_EMPTY_CTX, "current_phoneme_coverage": {"AH": 3}}

    with _full_patch(cap30, mp_mock):
        from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
        result = extract_clip_signals(
            dummy,
            ctx,
            aligner_output=json.loads(aligner_json.read_text()),
        )

    assert abs(result.phoneme_coverage_new - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# Test 5 — no forced-align path → empty phoneme sequence and zero lip sync
# ---------------------------------------------------------------------------


def test_no_forced_align_path(tmp_path):
    """When aligner_output is None, phoneme_sequence=[] and lip_sync_confidence=0.0."""
    dummy = tmp_path / "dummy.mp4"
    dummy.touch()

    cap30 = mock_capture(30)
    mp_mock = _make_mediapipe_mock(_make_face_mesh_mock())

    with _full_patch(cap30, mp_mock):
        from src.envs.subenv2.node4_clip_extractor import extract_clip_signals
        result = extract_clip_signals(dummy, _EMPTY_CTX, aligner_output=None)

    assert result.phoneme_sequence == []
    assert result.lip_sync_confidence == 0.0
