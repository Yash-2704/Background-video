"""
tests/test_generator.py
────────────────────────
15 tests covering core/generator.py dry-run behavior.

Isolation: the compiled dict fixture is hardcoded here using the canonical
Economy / Urban / Dusk / Cool / Serious / Gentle example so this test module
has no runtime dependency on prompt_compiler.py.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from core.generator import (
    GENERATION_CONSTANTS,
    crossfade_join,
    generate_clip,
    interpolate_clip,
    run_generation,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

COMPILED_DICT = {
    "positive": (
        "financial district architecture, economic infrastructure, "
        "institutional scale, implied market activity, "
        "dense urban environment, glass and steel architecture, "
        "city grid geometry, "
        "fading amber and blue light, long shadows, golden hour gradient, "
        "muted desaturated tones, cinematic restraint, no warmth, composed gravity, "
        "wide establishing shot, no people visible, "
        "professional broadcast aesthetic, photorealistic"
    ),
    "motion": (
        "slow lateral camera drift, gentle parallax, 0.2x speed, "
        "smooth and continuous motion, no cuts, no camera shake, "
        "no zoom, no sudden movements, temporally stable"
    ),
    "negative": (
        "text, titles, watermarks, logos, people, faces, hands, bodies, "
        "news tickers, clocks, timestamps, flags, explicit content, cartoon, "
        "CGI artifacts, lens flare, overexposed regions, flickering, strobing, "
        "fast motion, shaky camera, jump cuts, split screen, overlays, "
        "UI elements, animated graphics, lower thirds, chyrons"
    ),
    "positive_hash":     "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
    "motion_hash":       "bcd234efg567bcd234efg567bcd234efg567bcd234efg567bcd234efg567bcd2",
    "negative_hash":     "cde345fgh678cde345fgh678cde345fgh678cde345fgh678cde345fgh678cde3",
    "input_hash_short":  "a1b2c3",
    "selected_lut":      "cool_authority",
    "lower_third_style": "minimal_dark_bar",
    "compiler_version":  "1.0.0",
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_run_generation_completes_and_returns_required_keys(tmp_path):
    """Test 1: run_generation() with dev_mode=True returns all required keys."""
    result = run_generation(COMPILED_DICT, "run001", tmp_path, seed=42000)
    required_keys = {
        "run_id", "status", "raw_loop_path", "seed",
        "seam_frames_raw", "seam_frames_playable", "generation_log",
    }
    assert required_keys.issubset(result.keys())
    assert result["status"] == "complete"
    assert result["run_id"] == "run001"


def test_run_generation_raw_loop_file_exists(tmp_path):
    """Test 2: raw_loop_path file exists on disk after run."""
    result = run_generation(COMPILED_DICT, "run002", tmp_path, seed=42001)
    assert Path(result["raw_loop_path"]).exists()


def test_run_generation_raw_loop_is_valid_mp4(tmp_path):
    """Test 3: raw_loop_path is a valid MP4 readable by cv2.VideoCapture."""
    result = run_generation(COMPILED_DICT, "run003", tmp_path, seed=42002)
    cap = cv2.VideoCapture(result["raw_loop_path"])
    assert cap.isOpened()
    cap.release()


def test_run_generation_seam_frames_raw_are_positive_ints(tmp_path):
    """Test 4: seam_frames_raw is a list of 2 positive integers."""
    result = run_generation(COMPILED_DICT, "run004", tmp_path, seed=42003)
    sfr = result["seam_frames_raw"]
    assert isinstance(sfr, list)
    assert len(sfr) == 2
    assert all(isinstance(x, int) and x > 0 for x in sfr)


def test_run_generation_playable_seam_less_than_raw_seam(tmp_path):
    """Test 5: seam_frames_playable[0] < seam_frames_raw[0]."""
    result = run_generation(COMPILED_DICT, "run005", tmp_path, seed=42004)
    assert result["seam_frames_playable"][0] < result["seam_frames_raw"][0]


def test_run_generation_seeds_used_in_log(tmp_path):
    """Test 6: generation_log seeds_used == [seed, seed+1, seed+2]."""
    result = run_generation(COMPILED_DICT, "run006", tmp_path, seed=50000)
    seed   = result["seed"]
    assert result["generation_log"]["seeds_used"] == [seed, seed + 1, seed + 2]


def test_run_generation_clips_generated_in_log(tmp_path):
    """Test 7: generation_log clips_generated == 3."""
    result = run_generation(COMPILED_DICT, "run007", tmp_path, seed=42006)
    assert result["generation_log"]["clips_generated"] == 3


def test_run_generation_determinism(tmp_path):
    """Test 8: same explicit seed produces raw_loop files with identical frame counts."""
    result_a = run_generation(COMPILED_DICT, "run008a", tmp_path, seed=55555)
    result_b = run_generation(COMPILED_DICT, "run008b", tmp_path, seed=55555)
    fc_a = _frame_count(Path(result_a["raw_loop_path"]))
    fc_b = _frame_count(Path(result_b["raw_loop_path"]))
    assert fc_a == fc_b


def test_generate_clip_dry_run_creates_file(tmp_path):
    """Test 9: generate_clip() dry-run produces a file at the given path."""
    out = tmp_path / "test_clip.mp4"
    result = generate_clip(
        positive="test positive",
        motion="test motion",
        negative="test negative",
        seed=12345,
        clip_index=0,
        output_path=out,
        dry_run=True,
    )
    assert result == out
    assert out.exists()


def test_generate_clip_dry_run_exact_frame_count(tmp_path):
    """Test 10: generate_clip() dry-run produces exactly base_clip_frames_native frames."""
    out = tmp_path / "test_clip_frames.mp4"
    generate_clip(
        positive="test positive",
        motion="test motion",
        negative="test negative",
        seed=12345,
        clip_index=0,
        output_path=out,
        dry_run=True,
    )
    expected = GENERATION_CONSTANTS["base_clip_frames_native"]
    assert _frame_count(out) == expected


def test_interpolate_clip_dry_run_increases_frame_count(tmp_path):
    """Test 11: interpolate_clip() dry-run produces more frames than its input."""
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    generate_clip(
        positive="p", motion="m", negative="n",
        seed=11111, clip_index=0, output_path=src, dry_run=True,
    )
    src_count = _frame_count(src)
    interpolate_clip(src, dst, dry_run=True)
    assert _frame_count(dst) > src_count


def test_crossfade_join_returns_required_keys(tmp_path):
    """Test 12: crossfade_join() returns dict with all 5 required keys."""
    clips = []
    for i in range(3):
        raw = tmp_path / f"c{i}.mp4"
        interp = tmp_path / f"c{i}_30fps.mp4"
        generate_clip("p", "m", "n", seed=20000 + i, clip_index=i, output_path=raw, dry_run=True)
        interpolate_clip(raw, interp, dry_run=True)
        clips.append(interp)

    out   = tmp_path / "loop.mp4"
    result = crossfade_join(clips, out, dry_run=True)
    required = {"raw_loop_path", "seam_frames_raw", "seam_frames_playable",
                "total_frames_raw", "playable_frames"}
    assert required.issubset(result.keys())


def test_crossfade_join_seam_math(tmp_path):
    """Test 13: seam_frames_playable[0] == seam_frames_raw[0] - crossfade_frames."""
    clips = []
    for i in range(3):
        raw   = tmp_path / f"d{i}.mp4"
        interp = tmp_path / f"d{i}_30fps.mp4"
        generate_clip("p", "m", "n", seed=30000 + i, clip_index=i, output_path=raw, dry_run=True)
        interpolate_clip(raw, interp, dry_run=True)
        clips.append(interp)

    out    = tmp_path / "loop2.mp4"
    result = crossfade_join(clips, out, dry_run=True)
    cf     = GENERATION_CONSTANTS["crossfade_frames"]
    assert result["seam_frames_playable"][0] == result["seam_frames_raw"][0] - cf


def test_run_generation_cleans_up_on_failure(tmp_path):
    """Test 14: on generate_clip() exception, run directory is deleted."""
    with patch("core.generator.generate_clip", side_effect=RuntimeError("mock failure")):
        with pytest.raises(RuntimeError, match="Generation failed for run"):
            run_generation(COMPILED_DICT, "run_fail", tmp_path, seed=99999)

    assert not (tmp_path / "run_fail").exists()


def test_run_generation_assigns_seed_in_valid_range(tmp_path):
    """Test 15: run_generation() with no seed argument assigns a seed in [10000, 99999]."""
    result = run_generation(COMPILED_DICT, "run015", tmp_path)
    assert 10000 <= result["seed"] <= 99999
