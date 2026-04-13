"""
tests/test_generation_live.py
──────────────────────────────
Live-inference tests for core/generator.py — Wan2.2-TI2V-5B pathway.

ALL tests except test_00 require a CUDA-capable RTX 4090 with:
  • diffusers >= 0.27.0 (WanPipeline + WanImageToVideoPipeline)
  • torch 2.1.0 + CUDA 12.1
  • Wan2.2-TI2V-5B-Diffusers weights at <project_root>/Wan2.2/Wan2.2-TI2V-5B-Diffusers/

These tests serve as documentation and a runnable spec for the GPU machine.
Do NOT remove the skip decorators — they exist to keep CI green on this Mac.
Do NOT call run_generation() or generate_clip(dry_run=False) in any test
that is not already marked skip — doing so will trigger full model inference.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Non-skipped: import-safety gate (runs on this Mac) ────────────────────────

def test_00_import_does_not_pull_torch():
    """
    Importing core.generator must not cause torch or diffusers to be loaded
    at module level.  The live branch guard-imports them inside the else block
    only — if this test fails, a module-level import was added accidentally.
    """
    # Remove any previously cached module (in case another test imported it)
    for mod in list(sys.modules.keys()):
        if mod == "core.generator" or mod.startswith("core.generator."):
            del sys.modules[mod]

    # Also ensure torch is not already in sys.modules from a prior test
    torch_was_present    = "torch" in sys.modules
    diffusers_was_present = "diffusers" in sys.modules

    import core.generator  # noqa: F401 — import under test

    # torch / diffusers must not have been imported as a side-effect
    if not torch_was_present:
        assert "torch" not in sys.modules, (
            "core.generator imported torch at module level — "
            "move the import inside the dry_run=False branch."
        )
    if not diffusers_was_present:
        assert "diffusers" not in sys.modules, (
            "core.generator imported diffusers at module level — "
            "move the import inside the dry_run=False branch."
        )


# ── Skipped live tests (require RTX 4090) ─────────────────────────────────────

COMPILED_DICT = {
    "positive": (
        "financial district architecture, economic infrastructure, "
        "dense urban environment, glass and steel, city grid geometry"
    ),
    "motion": "slow lateral camera drift, gentle parallax, 0.2x speed",
    "negative": "text, watermarks, people, faces, flickering, fast motion",
    "positive_hash":     "abc123" * 10,
    "motion_hash":       "bcd234" * 10,
    "negative_hash":     "cde345" * 10,
    "input_hash_short":  "a1b2c3",
    "selected_lut":      "neutral",
    "lower_third_style": "minimal_dark_bar",
    "compiler_version":  "1.0.0",
}

_SKIP = pytest.mark.skip(reason="requires RTX 4090 with CUDA 12.1 and Wan2.2 weights")


@_SKIP
def test_01_t2v_generate_clip_returns_path(tmp_path):
    """T2V pathway (clip_index=0, conditioning_frame=None) returns a Path."""
    from core.generator import generate_clip

    out = tmp_path / "clip0.mp4"
    result = generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42000,
        clip_index=0,
        output_path=out,
        dry_run=False,
        conditioning_frame=None,
    )
    assert isinstance(result, Path)


@_SKIP
def test_02_t2v_output_file_exists_and_cv2_readable(tmp_path):
    """T2V output file exists on disk and cv2.VideoCapture can open it."""
    from core.generator import generate_clip

    out = tmp_path / "clip0.mp4"
    generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42001,
        clip_index=0,
        output_path=out,
        dry_run=False,
    )
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


@_SKIP
def test_03_t2v_output_has_145_frames(tmp_path):
    """T2V output has exactly 145 frames (base_clip_frames_native)."""
    from core.generator import GENERATION_CONSTANTS, generate_clip

    out = tmp_path / "clip0.mp4"
    generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42002,
        clip_index=0,
        output_path=out,
        dry_run=False,
    )
    cap   = cv2.VideoCapture(str(out))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == GENERATION_CONSTANTS["base_clip_frames_native"]  # 145


@_SKIP
def test_04_t2v_output_dimensions_1280x720(tmp_path):
    """T2V output frame dimensions are 1280×720 (generate_resolution)."""
    from core.generator import GENERATION_CONSTANTS, generate_clip

    out = tmp_path / "clip0.mp4"
    generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42003,
        clip_index=0,
        output_path=out,
        dry_run=False,
    )
    cap    = cv2.VideoCapture(str(out))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    expected_w, expected_h = GENERATION_CONSTANTS["generate_resolution"]
    assert width  == expected_w   # 1280
    assert height == expected_h   # 720


@_SKIP
def test_05_i2v_generate_clip_returns_path(tmp_path):
    """I2V pathway (clip_index=1, conditioning_frame=np.ndarray) returns a Path."""
    from core.generator import generate_clip

    # Synthesise a minimal BGR conditioning frame (720×1280)
    conditioning = np.full((720, 1280, 3), (80, 100, 120), dtype=np.uint8)

    out = tmp_path / "clip1.mp4"
    result = generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42004,
        clip_index=1,
        output_path=out,
        dry_run=False,
        conditioning_frame=conditioning,
    )
    assert isinstance(result, Path)


@_SKIP
def test_06_i2v_output_has_145_frames(tmp_path):
    """I2V output has exactly 145 frames."""
    from core.generator import GENERATION_CONSTANTS, generate_clip

    conditioning = np.full((720, 1280, 3), (80, 100, 120), dtype=np.uint8)
    out = tmp_path / "clip1.mp4"
    generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42005,
        clip_index=1,
        output_path=out,
        dry_run=False,
        conditioning_frame=conditioning,
    )
    cap   = cv2.VideoCapture(str(out))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == GENERATION_CONSTANTS["base_clip_frames_native"]  # 145


@_SKIP
def test_07_i2v_output_dimensions_1280x720(tmp_path):
    """I2V output frame dimensions are 1280×720."""
    from core.generator import GENERATION_CONSTANTS, generate_clip

    conditioning = np.full((720, 1280, 3), (80, 100, 120), dtype=np.uint8)
    out = tmp_path / "clip1.mp4"
    generate_clip(
        positive=COMPILED_DICT["positive"],
        motion=COMPILED_DICT["motion"],
        negative=COMPILED_DICT["negative"],
        seed=42006,
        clip_index=1,
        output_path=out,
        dry_run=False,
        conditioning_frame=conditioning,
    )
    cap    = cv2.VideoCapture(str(out))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    expected_w, expected_h = GENERATION_CONSTANTS["generate_resolution"]
    assert width  == expected_w
    assert height == expected_h


@_SKIP
def test_08_t2v_determinism_same_seed(tmp_path):
    """Same seed produces identical frame 0 pixel mean on two T2V runs."""
    from core.generator import generate_clip

    def _frame0_mean(path):
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        assert ret, f"Could not read frame 0 from {path}"
        return float(np.mean(frame))

    out_a = tmp_path / "det_a.mp4"
    out_b = tmp_path / "det_b.mp4"

    for out in (out_a, out_b):
        generate_clip(
            positive=COMPILED_DICT["positive"],
            motion=COMPILED_DICT["motion"],
            negative=COMPILED_DICT["negative"],
            seed=77777,
            clip_index=0,
            output_path=out,
            dry_run=False,
        )

    assert _frame0_mean(out_a) == pytest.approx(_frame0_mean(out_b), abs=0.5), (
        "Frame 0 pixel means differ between identical-seed T2V runs — "
        "check that torch.Generator is seeded correctly."
    )


@_SKIP
def test_09_vae_compression_rule_satisfied():
    """(base_clip_frames_native - 1) % 4 == 0 — VAE temporal compression rule."""
    from core.generator import GENERATION_CONSTANTS

    n = GENERATION_CONSTANTS["base_clip_frames_native"]  # 145
    assert (n - 1) % 4 == 0, (
        f"base_clip_frames_native={n} violates VAE rule (n-1) % 4 == 0. "
        "Changing this value will break Wan2.2 VAE temporal compression."
    )


@_SKIP
def test_10_full_run_generation_produces_raw_loop(tmp_path):
    """
    Full run_generation() with dev_mode=False produces raw_loop_path
    that exists on disk and is readable by cv2.VideoCapture.
    """
    import json
    from pathlib import Path as _Path
    from unittest.mock import patch

    from core.generator import run_generation

    # Temporarily override dev_mode in GENERATION_CONSTANTS for this test
    gc_path = _Path(__file__).resolve().parent.parent / "config" / "generation_constants.json"
    with gc_path.open() as fh:
        gc = json.load(fh)

    gc_live = {**gc, "dev_mode": False}

    with patch("core.generator.GENERATION_CONSTANTS", gc_live):
        result = run_generation(COMPILED_DICT, "run_live_test", tmp_path, seed=42000)

    raw_loop = _Path(result["raw_loop_path"])
    assert raw_loop.exists(), f"raw_loop_path not found on disk: {raw_loop}"
    cap = cv2.VideoCapture(str(raw_loop))
    assert cap.isOpened(), f"cv2.VideoCapture could not open: {raw_loop}"
    cap.release()
