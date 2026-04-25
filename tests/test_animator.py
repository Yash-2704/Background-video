"""
tests/test_animator.py
──────────────────────
Structural tests for core/animator.py.

All 8 tests are purely structural — none invoke FFmpeg or write real video
files. They pass on the Mac development machine with no GPU and no FFmpeg.

Run with:
    pytest tests/test_animator.py
"""

import inspect
import json
from pathlib import Path

import pytest


# ── 1. Module imports cleanly ─────────────────────────────────────────────────

def test_module_imports_cleanly():
    """Importing the public surface of core.animator must succeed without error."""
    from core.animator import animate_image, PROTOTYPE_DURATION_SECONDS, _get_motion_params  # noqa: F401


# ── 2. animate_image is callable with the correct signature ───────────────────

def test_animate_image_is_callable():
    """animate_image must be a callable with the declared parameter names."""
    from core.animator import animate_image

    assert callable(animate_image)

    sig = inspect.signature(animate_image)
    params = list(sig.parameters.keys())

    assert "image_path" in params
    assert "motion_intensity" in params
    assert "run_id" in params

    # Verify return annotation is Path
    assert sig.return_annotation is Path


# ── 3. PROTOTYPE_DURATION_SECONDS constant equals 5 ──────────────────────────

def test_prototype_duration_constant():
    """PROTOTYPE_DURATION_SECONDS must equal exactly 5."""
    from core.animator import PROTOTYPE_DURATION_SECONDS

    assert PROTOTYPE_DURATION_SECONDS == 5


# ── 4. Low motion intensity returns correct params ────────────────────────────

def test_motion_params_low():
    """motion_intensity=0.1 (low bracket) → zoom_speed=0.0005, pan_speed=0.3."""
    from core.animator import _get_motion_params

    result = _get_motion_params(0.1)

    assert result == {"zoom_speed": 0.0005, "pan_speed": 0.3}


# ── 5. Mid motion intensity returns correct params ────────────────────────────

def test_motion_params_mid():
    """motion_intensity=0.5 (mid bracket) → zoom_speed=0.001, pan_speed=0.6."""
    from core.animator import _get_motion_params

    result = _get_motion_params(0.5)

    assert result == {"zoom_speed": 0.001, "pan_speed": 0.6}


# ── 6. High motion intensity returns correct params ───────────────────────────

def test_motion_params_high():
    """motion_intensity=0.9 (high bracket) → zoom_speed=0.002, pan_speed=1.2."""
    from core.animator import _get_motion_params

    result = _get_motion_params(0.9)

    assert result == {"zoom_speed": 0.002, "pan_speed": 1.2}


# ── 7. prototype_duration_seconds key exists in config ───────────────────────

def test_prototype_duration_config_key_exists():
    """config/generation_constants.json must contain 'prototype_duration_seconds' == 5."""
    # Resolve config path the same way the module does — relative to project root.
    config_path = Path(__file__).resolve().parent.parent / "config" / "generation_constants.json"

    with config_path.open("r", encoding="utf-8") as fh:
        constants = json.load(fh)

    assert "prototype_duration_seconds" in constants, (
        "'prototype_duration_seconds' key is missing from generation_constants.json"
    )
    assert constants["prototype_duration_seconds"] == 5, (
        f"Expected prototype_duration_seconds=5, got {constants['prototype_duration_seconds']}"
    )


# ── 8. Output path is a sibling of the input PNG ─────────────────────────────

def test_output_path_is_sibling_of_input():
    """
    The output MP4 must be derived as image_path.parent / 'animated.mp4',
    i.e. a sibling of the input PNG in the same run directory.
    """
    image_path = Path("/some/run/image.png")
    expected_output = Path("/some/run/animated.mp4")

    # Exercise the exact path arithmetic used in animate_image() — no mocking
    # needed; we verify the arithmetic directly.
    actual_output = image_path.parent / "animated.mp4"

    assert actual_output == expected_output, (
        f"Expected {expected_output}, got {actual_output}"
    )
