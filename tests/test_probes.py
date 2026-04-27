"""
tests/test_probes.py
────────────────────
22 tests for core/probes.py (Prompt 5).

Fixtures:
  - synthetic_clip_path: a real MP4 produced by run_generation() in dry-run mode.
    Used for all OpenCV-path (live mode) tests.

Test count breakdown:
  Decode probe dry-run:  5 tests  (1–5)
  Decode probe live:     3 tests  (6–8)
  Temporal probe dry-run: 6 tests (9–14)
  Temporal probe live:    4 tests (15–18)
  Schema:                 3 tests (19–21)
  Regression:             1 test  (22)
"""

import sys
import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.probes import (
    run_decode_probe,
    run_temporal_probe,
    get_probe_schema,
    GENERATION_CONSTANTS,
    PROBE_SAMPLE_EVERY_N,
)
from core.generator import run_generation


# ── Fixture ─────────────────────────────────────────────────────────────────────

# Canonical Economy/Urban/Dusk/Cool/Serious/Gentle compiled dict
_COMPILED = {
    "positive": (
        "Urban environment at dusk, cool color temperature, serious atmosphere, "
        "gentle motion, cinematic background, broadcast safe, looping video"
    ),
    "motion": "gentle drift",
    "negative": "people, faces, text, logos, artifacts, flickering, fast motion",
    "input_hash_short": "ab12cd34",
    "compiler_version": "1.0.0",
}


@pytest.fixture(scope="module")
def synthetic_clip_path(tmp_path_factory):
    """
    Generate a synthetic raw-loop MP4 using run_generation() in dry-run mode.
    Scoped to module so the (moderately expensive) generation runs once.
    """
    out_dir = tmp_path_factory.mktemp("gen_output")
    result = run_generation(
        compiled=_COMPILED,
        run_id="test_probe_run",
        output_dir=out_dir,
        seed=42000,
        dry_run=True,
    )
    return Path(result["raw_loop_path"])


# ── Decode probe — dry-run ───────────────────────────────────────────────────────

class TestDecodeProbe_DryRun:

    def test_01_returns_all_required_keys(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        schema = get_probe_schema()["decode_probe_keys"]
        for key in schema:
            assert key in result, f"Missing key: {key}"

    def test_02_mean_luminance_is_float_in_range(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert isinstance(result["mean_luminance"], float)
        assert 0.0 <= result["mean_luminance"] <= 1.0

    def test_03_luminance_range_is_valid_pair(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        lr = result["luminance_range"]
        assert isinstance(lr, list)
        assert len(lr) == 2
        assert isinstance(lr[0], float) and isinstance(lr[1], float)
        assert lr[0] < lr[1]

    def test_04_gate_values_match_constants(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        gates = GENERATION_CONSTANTS["quality_gates"]
        assert result["luminance_gate_min"] == gates["luminance_gate_min"]
        assert result["luminance_gate_max"] == gates["luminance_gate_max"]

    def test_05_dry_run_key_is_true(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert result["dry_run"] is True


# ── Decode probe — live ──────────────────────────────────────────────────────────

class TestDecodeProbe_Live:

    def test_06_returns_all_required_keys(self, synthetic_clip_path):
        result = run_decode_probe(synthetic_clip_path, dry_run=False)
        schema = get_probe_schema()["decode_probe_keys"]
        for key in schema:
            assert key in result, f"Missing key: {key}"

    def test_07_mean_luminance_is_float_in_range(self, synthetic_clip_path):
        result = run_decode_probe(synthetic_clip_path, dry_run=False)
        assert isinstance(result["mean_luminance"], float)
        assert 0.0 <= result["mean_luminance"] <= 1.0

    def test_08_dominant_hue_degrees_in_range(self, synthetic_clip_path):
        result = run_decode_probe(synthetic_clip_path, dry_run=False)
        assert isinstance(result["dominant_hue_degrees"], float)
        assert 0.0 <= result["dominant_hue_degrees"] <= 360.0


# ── Temporal probe — dry-run ─────────────────────────────────────────────────────

class TestTemporalProbe_DryRun:

    def test_09_returns_all_required_keys(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        schema = get_probe_schema()["temporal_probe_keys"]
        for key in schema:
            assert key in result, f"Missing key: {key}"

    def test_10_flicker_index_is_non_negative_float(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert isinstance(result["flicker_index"], float)
        assert result["flicker_index"] >= 0.0

    def test_11_warping_artifact_score_is_non_negative_float(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert isinstance(result["warping_artifact_score"], float)
        assert result["warping_artifact_score"] >= 0.0

    def test_12_scene_cut_detected_is_bool(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert isinstance(result["scene_cut_detected"], bool)

    def test_13_perceptual_loop_score_in_range(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert isinstance(result["perceptual_loop_score"], float)
        assert 0.0 <= result["perceptual_loop_score"] <= 1.0

    def test_14_dry_run_key_is_true(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert result["dry_run"] is True


# ── Temporal probe — live ────────────────────────────────────────────────────────

class TestTemporalProbe_Live:

    def test_15_returns_all_required_keys(self, synthetic_clip_path):
        result = run_temporal_probe(synthetic_clip_path, dry_run=False)
        schema = get_probe_schema()["temporal_probe_keys"]
        for key in schema:
            assert key in result, f"Missing key: {key}"

    def test_16_frame_count_matches_actual_clip(self, synthetic_clip_path):
        result = run_temporal_probe(synthetic_clip_path, dry_run=False)
        cap = cv2.VideoCapture(str(synthetic_clip_path))
        actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert result["frame_count"] == actual

    def test_17_flicker_index_very_low_for_solid_color_clip(self, synthetic_clip_path):
        # Synthetic clip is near-solid color — frame-to-frame luminance variance is tiny
        result = run_temporal_probe(synthetic_clip_path, dry_run=False)
        assert result["flicker_index"] < 0.05, (
            f"Expected flicker_index < 0.05 for solid-color clip, "
            f"got {result['flicker_index']}"
        )

    def test_18_scene_cut_not_detected_in_solid_color_clip(self, synthetic_clip_path):
        result = run_temporal_probe(synthetic_clip_path, dry_run=False)
        assert result["scene_cut_detected"] is False


# ── Schema ───────────────────────────────────────────────────────────────────────

class TestProbeSchema:

    def test_19_get_probe_schema_returns_both_key_lists(self):
        schema = get_probe_schema()
        assert "decode_probe_keys" in schema
        assert "temporal_probe_keys" in schema
        assert isinstance(schema["decode_probe_keys"], list)
        assert isinstance(schema["temporal_probe_keys"], list)

    def test_20_dry_run_decode_keys_all_in_schema(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        schema_keys = get_probe_schema()["decode_probe_keys"]
        for key in result:
            assert key in schema_keys, f"Key '{key}' from probe not in schema"

    def test_21_dry_run_temporal_keys_all_in_schema(self, tmp_path):
        result = run_temporal_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        schema_keys = get_probe_schema()["temporal_probe_keys"]
        for key in result:
            assert key in schema_keys, f"Key '{key}' from probe not in schema"


# ── Regression ───────────────────────────────────────────────────────────────────

class TestRegression:

    def test_22_import_does_not_pull_in_torch_or_diffusers(self):
        # core.probes must import cleanly on a machine with zero ML packages
        assert "torch" not in sys.modules, "torch must not be imported by core.probes"
        assert "diffusers" not in sys.modules, (
            "diffusers must not be imported by core.probes"
        )


# ── NEW TESTS — sampling optimisation ────────────────────────────────────────────

def _make_synthetic_video(path: Path, num_frames: int, pixel_value: int = 128) -> None:
    """Write a solid-colour MP4 using cv2.VideoWriter. Width/height chosen to be
    small so tests run fast; pixel_value sets all BGR channels identically."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (64, 64))
    frame = np.full((64, 64, 3), pixel_value, dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


class TestSamplingConstant:

    def test_A_probe_sample_every_n_equals_6(self):
        assert PROBE_SAMPLE_EVERY_N == 6


class TestSampledFramesKey_DryRun:

    def test_B_sampled_frames_key_present_and_positive_in_dry_run(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        assert "sampled_frames" in result
        assert isinstance(result["sampled_frames"], int)
        assert result["sampled_frames"] > 0


class TestSampledFramesKey_Live:

    def test_C_sampled_frames_less_than_total_in_live_path(self, tmp_path):
        video_path = tmp_path / "short_clip.mp4"
        total_frames = 12
        _make_synthetic_video(video_path, total_frames)

        # Force the cv2 fallback by making FFmpeg appear unavailable.
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_decode_probe(video_path, dry_run=False)

        assert result["sampled_frames"] < total_frames, (
            f"Expected sampled_frames ({result['sampled_frames']}) < "
            f"total_frames ({total_frames})"
        )

    def test_D_all_required_keys_present_in_dry_run(self, tmp_path):
        result = run_decode_probe(tmp_path / "nonexistent.mp4", dry_run=True)
        required = [
            "mean_luminance",
            "luminance_range",
            "dominant_hue_degrees",
            "saturation_mean",
            "luminance_gate_min",
            "luminance_gate_max",
            "dry_run",
            "sampled_frames",
        ]
        for key in required:
            assert key in result, f"Missing key in dry_run result: {key}"

    def test_E_empty_video_fallback_does_not_crash(self, tmp_path):
        video_path = tmp_path / "empty_clip.mp4"
        # Create a minimal but unreadable video by writing 0 frames.
        _make_synthetic_video(video_path, num_frames=0)

        # Patch cv2.VideoCapture so read() always returns (False, None),
        # simulating a video with 0 readable frames.
        original_vc = cv2.VideoCapture

        class _AlwaysFailCapture:
            def __init__(self, *args, **kwargs):
                self._delegate = original_vc(*args, **kwargs)

            def read(self):
                return False, None

            def release(self):
                self._delegate.release()

            def isOpened(self):
                return True

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with patch("cv2.VideoCapture", _AlwaysFailCapture):
                result = run_decode_probe(video_path, dry_run=False)

        assert isinstance(result, dict), "Expected a dict even for empty video"
        assert result["sampled_frames"] == 0

    def test_F_mean_luminance_numerically_stable_under_sampling(self, tmp_path):
        pixel_value = 128
        video_path = tmp_path / "uniform_clip.mp4"
        _make_synthetic_video(video_path, num_frames=30, pixel_value=pixel_value)

        expected_luminance = pixel_value / 255.0

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_decode_probe(video_path, dry_run=False)

        assert abs(result["mean_luminance"] - expected_luminance) <= 0.05, (
            f"mean_luminance {result['mean_luminance']:.4f} deviates more than 0.05 "
            f"from expected {expected_luminance:.4f}"
        )
