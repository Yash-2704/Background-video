"""
tests/test_upscale_live.py
──────────────────────────
Tests for the live upscale path in upscale_clip() (Prompt 3).

tests 01–06: dry_run=True (no GPU required — always run)
tests 07–10: dry_run=False (GPU required — always skipped in CI)
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

from core.post_processor import upscale_clip


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_720p(tmp_path_factory):
    """
    10-frame 1280×720 synthetic MP4. Fast. No run_generation().
    """
    out = tmp_path_factory.mktemp("input") / "tiny_720p.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 24.0, (1280, 720))
    for i in range(10):
        frame = np.full((720, 1280, 3), (50 + i*5, 80, 120),
                        dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out


@pytest.fixture(scope="module")
def passing_decode():
    return {
        "mean_luminance":       0.46,
        "luminance_range":      [0.14, 0.79],
        "dominant_hue_degrees": 212.0,
        "saturation_mean":      0.38,
        "luminance_gate_min":   0.30,
        "luminance_gate_max":   0.70,
        "dry_run":              False,
    }


# ── Dry-run tests (01–06) ────────────────────────────────────────────────────────

def test_01_upscale_dry_run_returns_path(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    result = upscale_clip(tiny_720p, out, passing_decode, dry_run=True)
    assert isinstance(result, Path)


def test_02_upscale_dry_run_output_exists(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(tiny_720p, out, passing_decode, dry_run=True)
    assert out.exists()


def test_03_upscale_dry_run_dimensions_1920x1080(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(tiny_720p, out, passing_decode, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 1920
    assert h == 1080


def test_04_upscale_dry_run_readable_by_cv2(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(tiny_720p, out, passing_decode, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ret, frame = cap.read()
    cap.release()
    assert ret
    assert frame is not None


def test_05_upscale_dry_run_low_luminance_clamp(tiny_720p, tmp_path):
    """mean_luminance=0.10 → lum_factor clamped to 2.0 — must not crash."""
    probe_low = {"mean_luminance": 0.10}
    out = tmp_path / "up_low.mp4"
    result = upscale_clip(tiny_720p, out, probe_low, dry_run=True)
    assert out.exists()
    assert isinstance(result, Path)


def test_06_upscale_dry_run_high_luminance_clamp(tiny_720p, tmp_path):
    """mean_luminance=0.99 → lum_factor clamped to 0.5 — must not crash."""
    probe_high = {"mean_luminance": 0.99}
    out = tmp_path / "up_high.mp4"
    result = upscale_clip(tiny_720p, out, probe_high, dry_run=True)
    assert out.exists()
    assert isinstance(result, Path)


# ── Live-path tests (07–10) — skipped: GPU required ─────────────────────────────

@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_07_upscale_live_returns_path(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up_live.mp4"
    result = upscale_clip(tiny_720p, out, passing_decode, dry_run=False)
    assert isinstance(result, Path)


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_08_upscale_live_dimensions_1920x1080(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up_live.mp4"
    upscale_clip(tiny_720p, out, passing_decode, dry_run=False)
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 1920
    assert h == 1080


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_09_upscale_live_readable_by_cv2(tiny_720p, passing_decode, tmp_path):
    out = tmp_path / "up_live.mp4"
    upscale_clip(tiny_720p, out, passing_decode, dry_run=False)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ret, frame = cap.read()
    cap.release()
    assert ret
    assert frame is not None


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_10_upscale_live_raises_if_binary_missing(tiny_720p, passing_decode, tmp_path):
    """RuntimeError when binary path does not exist (patched PROJECT_ROOT → /nonexistent)."""
    out = tmp_path / "up_live_missing.mp4"
    with patch("core.post_processor.PROJECT_ROOT", Path("/nonexistent/nowhere")):
        with pytest.raises(RuntimeError, match="Real-ESRGAN binary not found"):
            upscale_clip(tiny_720p, out, passing_decode, dry_run=False)
