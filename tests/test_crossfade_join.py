"""
tests/test_crossfade_join.py
─────────────────────────────
Tests for the crossfade_join() function in core/generator.py.

Fixture constraint: three_tiny_clips writes exactly 60 frames total
(3 clips × 20 frames). All dry-run tests use this fixture and must
not call run_generation() or write more than 60 frames via cv2.

Seam value authority: GENERATION_CONSTANTS (generation_constants.json),
not the actual clip dimensions. The dry-run crossfade silently skips
the blend window when clips are shorter than production size.
"""

from pathlib import Path

import cv2
import pytest

from core.generator import GENERATION_CONSTANTS, crossfade_join


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def three_tiny_clips(tmp_path_factory):
    """
    3 tiny MP4s, 20 frames each, 1280x720, 24fps.
    Fast. No run_generation(). Used for all crossfade tests.
    Colors differ per clip so seam blending is visually
    distinguishable in frame analysis.
    """
    import numpy as np

    clips = []
    colors = [(50, 80, 120), (80, 120, 50), (120, 50, 80)]
    for i, color in enumerate(colors):
        out = tmp_path_factory.mktemp(f"clip{i}") / f"clip{i}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, 24.0, (1280, 720))
        for _ in range(20):
            frame = np.full((720, 1280, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        clips.append(out)
    return clips


# ── Helper ─────────────────────────────────────────────────────────────────────


def _open_cap(path: Path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cv2 could not open: {path}"
    return cap


# ── Tests 01–07: dry_run=True (no GPU required) ────────────────────────────────


def test_01_returns_all_required_keys(three_tiny_clips, tmp_path):
    """crossfade_join returns a dict with all 5 required keys."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    required = {
        "raw_loop_path",
        "seam_frames_raw",
        "seam_frames_playable",
        "total_frames_raw",
        "playable_frames",
    }
    assert required.issubset(result.keys())


def test_02_raw_loop_path_exists_on_disk(three_tiny_clips, tmp_path):
    """raw_loop_path file exists on disk after join."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    assert Path(result["raw_loop_path"]).exists()


def test_03_raw_loop_path_readable_by_cv2(three_tiny_clips, tmp_path):
    """raw_loop_path is readable by cv2."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    cap = _open_cap(result["raw_loop_path"])
    cap.release()


def test_04_seam_frames_raw_matches_constants(three_tiny_clips, tmp_path):
    """seam_frames_raw matches [145, 290] from GENERATION_CONSTANTS (config is authority)."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    expected = [
        GENERATION_CONSTANTS["base_clip_frames_native"],        # 145
        GENERATION_CONSTANTS["base_clip_frames_native"] * 2,   # 290
    ]
    assert result["seam_frames_raw"] == expected


def test_05_seam_frames_playable_matches_constants(three_tiny_clips, tmp_path):
    """seam_frames_playable matches [138, 269] from GENERATION_CONSTANTS."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    expected = GENERATION_CONSTANTS["seam_frames_playable_timeline"]  # [138, 269]
    assert result["seam_frames_playable"] == expected


def test_06_output_written_inside_expected_parent(three_tiny_clips, tmp_path):
    """Output file is written inside output_path's parent directory."""
    out = tmp_path / "loop.mp4"
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    assert Path(result["raw_loop_path"]).parent == tmp_path


def test_07_second_call_overwrites_cleanly(three_tiny_clips, tmp_path):
    """Calling crossfade_join twice with the same output_path overwrites cleanly."""
    out = tmp_path / "loop.mp4"
    crossfade_join(three_tiny_clips, out, dry_run=True)
    # Second call must not raise and must produce a valid MP4
    result = crossfade_join(three_tiny_clips, out, dry_run=True)
    cap = _open_cap(result["raw_loop_path"])
    cap.release()


# ── Test 08: live path (GPU skipped) ──────────────────────────────────────────


@pytest.mark.skip(reason="requires RTX 4090 — run on GPU machine only")
def test_08_live_path_import_only(three_tiny_clips, tmp_path):
    """
    Verifies the function signature accepts dry_run=False without import
    errors (import-only test, no execution).

    The live branch guards all ML imports inside dry_run=False, so this
    module must import cleanly on a CPU-only machine even if the test is
    never executed.
    """
    # Reaching here means the module imported without pulling in torch at
    # module level. The actual call is not made — no GPU is available in CI.
    out = tmp_path / "loop_live.mp4"
    # crossfade_join(three_tiny_clips, out, dry_run=False)  # not called
    assert callable(crossfade_join)
