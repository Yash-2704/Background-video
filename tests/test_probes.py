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
import pytest
from pathlib import Path

from core.probes import (
    run_decode_probe,
    run_temporal_probe,
    get_probe_schema,
    GENERATION_CONSTANTS,
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
