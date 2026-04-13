"""
tests/test_lut_grade.py
───────────────────────
Tests for LUT .cube file generation (Deliverable A) and
apply_lut_grade() dry_run paths (Deliverable B).

Tests 01–12: no GPU required — run in CI.
Tests 13–16: skipped (require RTX 4090 + FFmpeg lut3d on GPU machine).

Never calls run_generation(). Never writes more than 30 frames.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.post_processor import apply_lut_grade

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LUTS_DIR     = PROJECT_ROOT / "luts"
LUT_NAMES    = ["cool_authority", "neutral", "warm_tension"]
LUT_SIZE     = 33
EXPECTED_DATA_LINES = LUT_SIZE ** 3  # 35937

HEADER_PREFIXES = ("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_1080p(tmp_path_factory):
    """
    10-frame 1920×1080 synthetic MP4. Fast. No run_generation().
    Simulates upscaled source — apply_lut_grade input.
    """
    out = tmp_path_factory.mktemp("lut_input") / "tiny_1080p.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 24.0, (1920, 1080))
    for i in range(10):
        frame = np.full((1080, 1920, 3),
                        (100 + i * 5, 120, 140), dtype=np.uint8)
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


# ── Helper ────────────────────────────────────────────────────────────────────────

def _data_lines(cube_path: Path):
    """Return only the numeric data lines from a .cube file."""
    return [
        ln for ln in cube_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith(HEADER_PREFIXES)
    ]


# ── .cube file existence tests (01–03) ───────────────────────────────────────────

def test_01_cool_authority_cube_exists():
    assert (LUTS_DIR / "cool_authority.cube").exists()


def test_02_neutral_cube_exists():
    assert (LUTS_DIR / "neutral.cube").exists()


def test_03_warm_tension_cube_exists():
    assert (LUTS_DIR / "warm_tension.cube").exists()


# ── .cube file data-line count (04) ──────────────────────────────────────────────

def test_04_each_cube_has_35937_data_lines():
    for name in LUT_NAMES:
        cube_path = LUTS_DIR / f"{name}.cube"
        lines = _data_lines(cube_path)
        assert len(lines) == EXPECTED_DATA_LINES, (
            f"{name}.cube: expected {EXPECTED_DATA_LINES} data lines, "
            f"got {len(lines)}"
        )


# ── .cube file header (05) ────────────────────────────────────────────────────────

def test_05_each_cube_header_contains_lut_3d_size_33():
    for name in LUT_NAMES:
        cube_path = LUTS_DIR / f"{name}.cube"
        text = cube_path.read_text(encoding="utf-8")
        assert "LUT_3D_SIZE 33" in text, (
            f"{name}.cube does not contain 'LUT_3D_SIZE 33'"
        )


# ── neutral values within [0, 1] (06) ────────────────────────────────────────────

def test_06_neutral_all_values_in_range():
    lines = _data_lines(LUTS_DIR / "neutral.cube")
    for i, ln in enumerate(lines):
        parts = ln.split()
        assert len(parts) == 3, f"Line {i}: expected 3 values, got {len(parts)}"
        for val in parts:
            f = float(val)
            assert 0.0 <= f <= 1.0, f"Line {i}: value {f} out of [0,1]"


# ── blue-shift verification (07) ─────────────────────────────────────────────────

def test_07_cool_authority_blue_mean_gt_neutral():
    def blue_mean(name):
        lines = _data_lines(LUTS_DIR / f"{name}.cube")
        return sum(float(ln.split()[2]) for ln in lines) / len(lines)

    assert blue_mean("cool_authority") > blue_mean("neutral"), (
        "cool_authority blue channel mean should exceed neutral's (blue shift)"
    )


# ── warm-shift verification (08) ─────────────────────────────────────────────────

def test_08_warm_tension_red_mean_gt_neutral():
    """
    The warm shift (R*=1.06, B*=0.94) widens the R-B spread even after
    saturation compression and shadow lift are applied. Comparing global R
    means is unreliable because shadow lift (-0.04) depresses the whole
    lattice. The canonical verification is mean(R-B) in warm_tension >
    mean(R-B) in neutral.
    """
    def rb_diff_mean(name):
        lines = _data_lines(LUTS_DIR / f"{name}.cube")
        return sum(
            float(ln.split()[0]) - float(ln.split()[2]) for ln in lines
        ) / len(lines)

    assert rb_diff_mean("warm_tension") > rb_diff_mean("neutral"), (
        "warm_tension mean(R-B) should exceed neutral's — warm R/B shift not applied"
    )


# ── dry_run=True path tests (09–12) ──────────────────────────────────────────────

def test_09_dry_run_returns_path(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "graded.mp4"
    result = apply_lut_grade(tiny_1080p, out, "neutral", passing_decode, dry_run=True)
    assert isinstance(result, Path)


def test_10_dry_run_output_file_exists(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "graded.mp4"
    apply_lut_grade(tiny_1080p, out, "neutral", passing_decode, dry_run=True)
    assert out.exists()


def test_11_dry_run_output_readable_by_cv2(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "graded.mp4"
    apply_lut_grade(tiny_1080p, out, "neutral", passing_decode, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_12_dry_run_works_for_all_three_lut_names(tiny_1080p, passing_decode, tmp_path):
    for lut_name in LUT_NAMES:
        out = tmp_path / f"graded_{lut_name}.mp4"
        result = apply_lut_grade(tiny_1080p, out, lut_name, passing_decode, dry_run=True)
        assert isinstance(result, Path)
        assert out.exists()


# ── Live-path tests (13–16) — skipped (GPU machine only) ─────────────────────────

@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_13_live_returns_path(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "live_graded.mp4"
    result = apply_lut_grade(tiny_1080p, out, "neutral", passing_decode, dry_run=False)
    assert isinstance(result, Path)


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_14_live_output_readable_by_cv2(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "live_graded.mp4"
    apply_lut_grade(tiny_1080p, out, "neutral", passing_decode, dry_run=False)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_15_live_raises_file_not_found_if_cube_missing(tiny_1080p, passing_decode, tmp_path):
    out = tmp_path / "live_graded.mp4"
    with pytest.raises(FileNotFoundError):
        apply_lut_grade(tiny_1080p, out, "nonexistent_lut", passing_decode, dry_run=False)


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_16_live_raises_runtime_error_if_ffmpeg_fails(tiny_1080p, passing_decode, tmp_path):
    """Corrupt the .cube file path to force FFmpeg failure."""
    out = tmp_path / "live_graded.mp4"
    with pytest.raises((RuntimeError, FileNotFoundError)):
        apply_lut_grade(
            Path("/nonexistent/input.mp4"),
            out,
            "neutral",
            passing_decode,
            dry_run=False,
        )
