"""CLI smoke tests for scripts/validate_annotations.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> str:
    return str(Path(__file__).parent.parent.parent)


def _run(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run validate_annotations.py as a subprocess."""
    env = {**os.environ, "PYTHONPATH": cwd}
    return subprocess.run(
        [sys.executable, "scripts/validate_annotations.py"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def test_validate_all_wrapped_subenv1_exits_zero(project_root):
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
        result = _run(["--cases", tmp_path, "--subenv", "all"], cwd=project_root)
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Schema OK: 1 case(s)" in result.stdout
        assert "Validation complete: 1 case(s) validated" in result.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)
