"""
CLI smoke tests for src/evaluate.py.

Exercises the evaluate harness as a real subprocess to verify:
  - Correct exit codes for valid and invalid inputs.
  - Expected strings appear in stdout/stderr.
  - Schema-invalid JSON is correctly rejected with exit code 1.
  - Invalid --subenv argument is rejected by argparse.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> str:
    return str(Path(__file__).parent.parent.parent)


def _run(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Helper to run evaluate.py as a subprocess.

    Injects PYTHONPATH=cwd so ``from src.schemas...`` absolute imports resolve
    correctly — evaluate.py is invoked as a script, not as a module, so it
    does not inherit the ``sys.path`` modifications that conftest.py makes.
    """
    env = {**os.environ, "PYTHONPATH": cwd}
    return subprocess.run(
        [sys.executable, "src/evaluate.py"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


# ===========================================================================
# Dry-run tests
# ===========================================================================

def test_dry_run_subenv1_exits_zero(project_root):
    """--dry-run on a valid subenv1 file must exit 0 and print 'Schema OK'."""
    result = _run(
        ["--test-set", "tests/test_set/subenv1_cases.json", "--subenv", "1", "--dry-run"],
        cwd=project_root,
    )
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Schema OK" in result.stdout


def test_dry_run_subenv2_exits_zero(project_root):
    """--dry-run on a valid subenv2 file must exit 0 and print 'Schema OK'."""
    result = _run(
        ["--test-set", "tests/test_set/subenv2_cases.json", "--subenv", "2", "--dry-run"],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Schema OK" in result.stdout


def test_dry_run_subenv3_exits_zero(project_root):
    """--dry-run on a valid subenv3 file must exit 0 and print 'Schema OK'."""
    result = _run(
        ["--test-set", "tests/test_set/subenv3_cases.json", "--subenv", "3", "--dry-run"],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Schema OK" in result.stdout


def test_dry_run_all_exits_zero(project_root):
    """--test-set <directory> --subenv all --dry-run must exit 0."""
    result = _run(
        ["--test-set", "tests/test_set/", "--subenv", "all", "--dry-run"],
        cwd=project_root,
    )
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # All three files → three "Schema OK" lines
    assert result.stdout.count("Schema OK") >= 3


def test_dry_run_all_wrapped_subenv1_exits_zero(project_root):
    """--subenv all must auto-detect wrapped Sub-env 1 observations via image_obs."""
    wrapped_cases = {
        "cases": [
            {
                "id": "wrapped_001",
                "observation": {
                    "image_obs": {
                        "face_occupancy_ratio": 0.6,
                        "estimated_yaw_degrees": 5.0,
                        "estimated_pitch_degrees": 2.0,
                        "background_complexity_score": 0.3,
                        "lighting_uniformity_score": 0.7,
                        "skin_tone_bucket": 3,
                        "occlusion_detected": False,
                        "image_resolution": [1280, 720],
                        "estimated_sharpness": 0.8,
                        "prompt_token_count": 40,
                        "prompt_semantic_density": 0.5,
                        "conflicting_descriptors": [],
                        "identity_anchoring_strength": 0.8,
                    },
                    "proposed_config": {"cfg": 7.0, "eta": 0.08, "denoise_alt": 0.5},
                },
                "ground_truth": {
                    "image": {
                        "regime_classification": "frontal_simple",
                        "acceptable_regimes": [],
                        "identified_risk_factors": [],
                        "valid_prompt_modifications": [],
                    },
                    "param": {
                        "config_risk_level": "safe",
                        "anomalies": [],
                        "predicted_failure_modes": [],
                        "valid_fix_directions": [],
                    },
                },
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(wrapped_cases, f)
        tmp_path = f.name

    try:
        result = _run(
            ["--test-set", tmp_path, "--subenv", "all", "--dry-run"],
            cwd=project_root,
        )
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Schema OK" in result.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ===========================================================================
# Scoring (non-dry-run) tests
# ===========================================================================

def test_subenv1_produces_scores(project_root):
    """Full scoring run on subenv1 must print case IDs and the Mean stats line."""
    result = _run(
        ["--test-set", "tests/test_set/subenv1_cases.json", "--subenv", "1"],
        cwd=project_root,
    )
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Mean:" in result.stdout
    assert "Case 001" in result.stdout


def test_subenv2_produces_scores(project_root):
    """Full scoring run on subenv2 must print case IDs and Mean."""
    result = _run(
        ["--test-set", "tests/test_set/subenv2_cases.json", "--subenv", "2"],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Mean:" in result.stdout


def test_subenv3_produces_scores(project_root):
    """Full scoring run on subenv3 must print Mean."""
    result = _run(
        ["--test-set", "tests/test_set/subenv3_cases.json", "--subenv", "3"],
        cwd=project_root,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Mean:" in result.stdout


def test_score_values_in_range(project_root):
    """All score values printed to stdout must fall in [0.000, 1.000]."""
    result = _run(
        ["--test-set", "tests/test_set/subenv1_cases.json", "--subenv", "1"],
        cwd=project_root,
    )
    assert result.returncode == 0
    for line in result.stdout.splitlines():
        if not line.startswith("Case"):
            continue
        # Line format: "Case 001   score=0.745  [...]"
        token = next((t for t in line.split() if t.startswith("score=")), None)
        assert token is not None, f"No score token found in: {line!r}"
        score = float(token.split("=")[1])
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ===========================================================================
# Error-path tests
# ===========================================================================

def test_missing_required_field_exits_one(project_root):
    """A case with a missing required 'image_obs' field must cause exit 1."""
    bad_cases = {
        "cases": [
            {
                "id": "bad",
                "observation": {
                    # 'image_obs' key is missing; only proposed_config present
                    "proposed_config": {"cfg": 7.0}
                },
                "ground_truth": {
                    "image": {
                        "regime_classification": "frontal_simple",
                        "acceptable_regimes": [],
                        "identified_risk_factors": [],
                        "valid_prompt_modifications": [],
                    },
                    "param": {
                        "config_risk_level": "safe",
                        "anomalies": [],
                        "predicted_failure_modes": [],
                        "valid_fix_directions": [],
                    },
                },
            }
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad_cases, f)
        tmp_path = f.name

    try:
        result = _run(
            ["--test-set", tmp_path, "--subenv", "1", "--dry-run"],
            cwd=project_root,
        )
        assert result.returncode == 1, (
            f"Expected exit 1 for missing field, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "ERROR" in combined or "missing" in combined.lower()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_invalid_observation_field_exits_one(project_root):
    """A case where image_obs has a wrong-type field must cause exit 1."""
    bad_cases = {
        "cases": [
            {
                "id": "bad_type",
                "observation": {
                    "image_obs": {
                        "face_occupancy_ratio": "not_a_float",   # wrong type
                        "estimated_yaw_degrees": 5.0,
                        "estimated_pitch_degrees": 2.0,
                        "background_complexity_score": 0.3,
                        "lighting_uniformity_score": 0.7,
                        "skin_tone_bucket": 3,
                        "occlusion_detected": False,
                        "image_resolution": [1280, 720],
                        "estimated_sharpness": 0.75,
                        "prompt_token_count": 40,
                        "prompt_semantic_density": 0.5,
                        "conflicting_descriptors": [],
                        "identity_anchoring_strength": 0.8,
                    },
                    "proposed_config": {},
                },
                "ground_truth": {
                    "image": {
                        "regime_classification": "frontal_simple",
                        "acceptable_regimes": [],
                        "identified_risk_factors": [],
                        "valid_prompt_modifications": [],
                    },
                    "param": {
                        "config_risk_level": "safe",
                        "anomalies": [],
                        "predicted_failure_modes": [],
                        "valid_fix_directions": [],
                    },
                },
            }
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad_cases, f)
        tmp_path = f.name

    try:
        result = _run(
            ["--test-set", tmp_path, "--subenv", "1", "--dry-run"],
            cwd=project_root,
        )
        assert result.returncode == 1, (
            f"Expected exit 1 for wrong type, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_invalid_json_exits_nonzero(project_root):
    """A file with invalid JSON must cause a non-zero exit."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("{this is not json}")
        tmp_path = f.name

    try:
        result = _run(
            ["--test-set", tmp_path, "--subenv", "1", "--dry-run"],
            cwd=project_root,
        )
        assert result.returncode != 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_invalid_subenv_argument(project_root):
    """--subenv 9 must be rejected by argparse (it's not in choices)."""
    result = _run(
        ["--test-set", "tests/test_set/subenv1_cases.json", "--subenv", "9"],
        cwd=project_root,
    )
    assert result.returncode != 0
    # argparse writes the error to stderr
    assert "9" in result.stderr or "invalid choice" in result.stderr.lower()


def test_missing_test_set_argument(project_root):
    """Omitting --test-set must exit non-zero."""
    result = _run(["--subenv", "1"], cwd=project_root)
    assert result.returncode != 0
