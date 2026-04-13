"""
tests/test_composite_preview.py
────────────────────────────────
12 tests for composite_final() and export_preview() in core/post_processor.py.

Tests 01–08: dry_run=True (no FFmpeg, no GPU required).
Tests 09–12: skipped (require GPU machine / FFmpeg live pipeline).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.post_processor import composite_final, export_preview

# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_1080p_graded(tmp_path_factory):
    """10-frame 1920×1080 MP4. Simulates graded video input."""
    out = tmp_path_factory.mktemp("comp") / "graded.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 24.0, (1920, 1080))
    for i in range(10):
        frame = np.full((1080, 1920, 3),
                        (80, 100 + i*2, 120), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out


@pytest.fixture(scope="module")
def tiny_mask(tmp_path_factory):
    """Static 1920×1080 grayscale PNG mask."""
    out = tmp_path_factory.mktemp("mask") / "mask.png"
    mask = np.full((1080, 1920), 200, dtype=np.uint8)
    cv2.imwrite(str(out), mask)
    return out


@pytest.fixture(scope="module")
def seam_frames():
    return [138, 269]


# ── Seam math fix verification (dry_run=True) ─────────────────────────────────────

def test_01_seam1_start_s(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """Segment 2 start_s must equal (seam_frames[0] / 30.0) - 1.0."""
    tmp = tmp_path_factory.mktemp("t01")
    gif = tmp / "preview.gif"
    manifest_path = tmp / "manifest.json"
    result = export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=True
    )
    expected = (seam_frames[0] / 30.0) - 1.0
    actual = result["segments"][1]["start_s"]
    assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"


def test_02_seam2_start_s(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """Segment 3 start_s must equal (seam_frames[1] / 30.0) - 1.0."""
    tmp = tmp_path_factory.mktemp("t02")
    gif = tmp / "preview.gif"
    manifest_path = tmp / "manifest.json"
    result = export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=True
    )
    expected = (seam_frames[1] / 30.0) - 1.0
    actual = result["segments"][2]["start_s"]
    assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"


def test_03_manifest_has_3_segments(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """Manifest must have exactly 3 segments."""
    tmp = tmp_path_factory.mktemp("t03")
    gif = tmp / "preview.gif"
    manifest_path = tmp / "manifest.json"
    result = export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=True
    )
    assert "segments" in result
    assert len(result["segments"]) == 3


def test_04_gif_exists_after_dry_run(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """GIF output file must exist on disk after dry_run=True call."""
    tmp = tmp_path_factory.mktemp("t04")
    gif = tmp / "preview.gif"
    manifest_path = tmp / "manifest.json"
    export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=True
    )
    assert gif.exists()


# ── Composite tests (dry_run=True) ───────────────────────────────────────────────

def test_05_composite_returns_path(tiny_1080p_graded, tiny_mask, tmp_path_factory):
    """composite_final dry_run=True must return a Path."""
    tmp = tmp_path_factory.mktemp("t05")
    out = tmp / "final.mp4"
    result = composite_final(tiny_1080p_graded, tiny_mask, out, dry_run=True)
    assert isinstance(result, Path)


def test_06_composite_output_exists(tiny_1080p_graded, tiny_mask, tmp_path_factory):
    """Composite output file must exist on disk."""
    tmp = tmp_path_factory.mktemp("t06")
    out = tmp / "final.mp4"
    composite_final(tiny_1080p_graded, tiny_mask, out, dry_run=True)
    assert out.exists()


def test_07_composite_readable_by_cv2(tiny_1080p_graded, tiny_mask, tmp_path_factory):
    """Composite output must be openable by cv2."""
    tmp = tmp_path_factory.mktemp("t07")
    out = tmp / "final.mp4"
    composite_final(tiny_1080p_graded, tiny_mask, out, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_08_composite_dimensions(tiny_1080p_graded, tiny_mask, tmp_path_factory):
    """Composite output dimensions must be 1920×1080."""
    tmp = tmp_path_factory.mktemp("t08")
    out = tmp / "final.mp4"
    composite_final(tiny_1080p_graded, tiny_mask, out, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 1920
    assert h == 1080


# ── Live-path tests (skipped — require GPU machine) ──────────────────────────────

@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_09_composite_live_returns_path(tiny_1080p_graded, tiny_mask, tmp_path_factory):
    """composite_final dry_run=False must return a Path."""
    tmp = tmp_path_factory.mktemp("t09")
    out = tmp / "final_live.mp4"
    result = composite_final(tiny_1080p_graded, tiny_mask, out, dry_run=False)
    assert isinstance(result, Path)


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_10_export_preview_live_returns_dict(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """export_preview dry_run=False must return a dict with 4 required keys."""
    tmp = tmp_path_factory.mktemp("t10")
    gif = tmp / "preview_live.gif"
    manifest_path = tmp / "manifest_live.json"
    result = export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=False
    )
    assert isinstance(result, dict)
    for key in ("segments", "preview_fps", "preview_resolution", "source_seam_frames_playable"):
        assert key in result, f"Missing key: {key}"


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_11_export_preview_live_gif_exists(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """export_preview dry_run=False GIF file must exist on disk."""
    tmp = tmp_path_factory.mktemp("t11")
    gif = tmp / "preview_live.gif"
    manifest_path = tmp / "manifest_live.json"
    export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=False
    )
    assert gif.exists()


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_12_export_preview_live_manifest_parseable(tiny_1080p_graded, tiny_mask, seam_frames, tmp_path_factory):
    """export_preview dry_run=False manifest JSON must exist and parse correctly."""
    tmp = tmp_path_factory.mktemp("t12")
    gif = tmp / "preview_live.gif"
    manifest_path = tmp / "manifest_live.json"
    export_preview(
        tiny_1080p_graded, gif, manifest_path, seam_frames, dry_run=False
    )
    assert manifest_path.exists()
    with manifest_path.open("r") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
    assert "segments" in data
