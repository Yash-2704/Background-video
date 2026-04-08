"""
tests/test_post_processor.py
────────────────────────────
39 tests for core/post_processor.py (Prompt 7).

All tests run in dry_run=True mode. No FFmpeg, no Real-ESRGAN,
no ML packages required.

Fixtures:
  synthetic_raw_loop — real MP4 via run_generation(dry_run=True)
  passing_decode     — dry-run decode probe result (mean_luminance=0.46)
  compiled_dict      — canonical Economy/Urban/Dusk/Cool/Serious/Gentle dict
  seam_frames        — [169, 338]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.generator import run_generation
from core.probes import run_decode_probe
from core.post_processor import (
    UPSCALE_TARGET,
    upscale_clip,
    generate_anchor_mask,
    assess_content_risk,
    apply_lut_grade,
    composite_final,
    export_preview,
    run_post_processing,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def compiled_dict():
    """Canonical Economy/Urban/Dusk/Cool/Serious/Gentle compiled dict."""
    return {
        "positive":          "Urban cityscape at dusk, cool blue tones, serious mood",
        "motion":            "gentle drift",
        "negative":          "text, watermark, faces, logos",
        "selected_lut":      "cool_authority",
        "input_hash_short":  "a1b2c3d4",
        "compiler_version":  "1.0.0",
        "scene":             "Urban",
        "time_of_day":       "Dusk",
        "color_grade":       "Cool",
        "mood":              "Serious",
        "motion_speed":      "Gentle",
        "budget":            "Economy",
    }


@pytest.fixture(scope="module")
def seam_frames():
    return [169, 338]


@pytest.fixture(scope="module")
def passing_decode():
    """Dry-run decode probe result — mean_luminance=0.46."""
    return run_decode_probe(Path("nonexistent.mp4"), dry_run=True)


@pytest.fixture(scope="module")
def synthetic_raw_loop(tmp_path_factory, compiled_dict):
    """Real MP4 file produced by run_generation() in dry_run mode."""
    tmp = tmp_path_factory.mktemp("gen")
    result = run_generation(compiled_dict, "test_clip_001", tmp)
    return Path(result["raw_loop_path"])


# ── upscale_clip tests (1–4) ────────────────────────────────────────────────────

def test_upscale_returns_path(synthetic_raw_loop, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    result = upscale_clip(synthetic_raw_loop, out, passing_decode, dry_run=True)
    assert isinstance(result, Path)


def test_upscale_file_exists(synthetic_raw_loop, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(synthetic_raw_loop, out, passing_decode, dry_run=True)
    assert out.exists()


def test_upscale_readable_by_cv2(synthetic_raw_loop, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(synthetic_raw_loop, out, passing_decode, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_upscale_frame_dimensions(synthetic_raw_loop, passing_decode, tmp_path):
    out = tmp_path / "up.mp4"
    upscale_clip(synthetic_raw_loop, out, passing_decode, dry_run=True)
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 1920
    assert h == 1080


# ── generate_anchor_mask tests (5–11) ───────────────────────────────────────────

def test_mask_center_returns_path(tmp_path):
    out = tmp_path / "mask_center.png"
    result = generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    assert isinstance(result, Path)


def test_mask_center_file_exists(tmp_path):
    out = tmp_path / "mask_center.png"
    generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    assert out.exists()


def test_mask_center_readable_by_cv2(tmp_path):
    out = tmp_path / "mask_center.png"
    generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    img = cv2.imread(str(out))
    assert img is not None


def test_mask_center_shape(tmp_path):
    out = tmp_path / "mask_center.png"
    generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (1080, 1920)


def test_mask_center_zone_darkened(tmp_path):
    out = tmp_path / "mask_center.png"
    generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # Center zone: x_frac=0.25, y_frac=0.35, w=0.50, h=0.55
    # Zone center: approx pixel (960, 648)
    zone_cx = int(0.25 * 1920 + 0.5 * 0.50 * 1920)  # 720
    zone_cy = int(0.35 * 1080 + 0.5 * 0.55 * 1080)  # 675
    assert img[zone_cy, zone_cx] < 255


def test_mask_lower_third_bottom_darkened(tmp_path):
    out = tmp_path / "mask_lower.png"
    generate_anchor_mask("lower_third", out, (1920, 1080), dry_run=True)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # lower_third: y=0.65, h=0.35 → rows 702–1080
    # Check row 900 (well inside zone)
    bottom_vals = img[900, :]
    assert np.mean(bottom_vals) < 255


def test_mask_upper_third_top_darkened(tmp_path):
    out = tmp_path / "mask_upper.png"
    generate_anchor_mask("upper_third", out, (1920, 1080), dry_run=True)
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # upper_third: y=0.0, h=0.20 → rows 0–216
    # Check row 100 (well inside zone)
    top_vals = img[100, :]
    assert np.mean(top_vals) < 255


# ── assess_content_risk tests (12–15) ───────────────────────────────────────────

def test_risk_center_has_all_keys(tmp_path, synthetic_raw_loop):
    mask_path = tmp_path / "m.png"
    generate_anchor_mask("center", mask_path, (1920, 1080), dry_run=True)
    risk = assess_content_risk(mask_path, synthetic_raw_loop, "center", dry_run=True)
    required = {
        "luminance_variance_at_zone",
        "edge_density_at_boundary",
        "bright_intrusion_risk",
        "flag",
        "dev_phase_behavior",
        "production_behavior_planned",
    }
    assert required.issubset(set(risk.keys()))


def test_risk_flag_valid_values(tmp_path, synthetic_raw_loop):
    mask_path = tmp_path / "m.png"
    generate_anchor_mask("center", mask_path, (1920, 1080), dry_run=True)
    for pos in ["center", "lower_third", "upper_third"]:
        risk = assess_content_risk(mask_path, synthetic_raw_loop, pos, dry_run=True)
        assert risk["flag"] in ("clear", "review_recommended")


def test_risk_dev_phase_behavior_log_only(tmp_path, synthetic_raw_loop):
    mask_path = tmp_path / "m.png"
    generate_anchor_mask("center", mask_path, (1920, 1080), dry_run=True)
    for pos in ["center", "lower_third", "upper_third"]:
        risk = assess_content_risk(mask_path, synthetic_raw_loop, pos, dry_run=True)
        assert risk["dev_phase_behavior"] == "log_only"


def test_risk_dry_run_positions_different_flags(tmp_path, synthetic_raw_loop):
    mask_path = tmp_path / "m.png"
    generate_anchor_mask("center", mask_path, (1920, 1080), dry_run=True)
    risk_center = assess_content_risk(mask_path, synthetic_raw_loop, "center", dry_run=True)
    risk_lower  = assess_content_risk(mask_path, synthetic_raw_loop, "lower_third", dry_run=True)
    risk_upper  = assess_content_risk(mask_path, synthetic_raw_loop, "upper_third", dry_run=True)
    assert risk_center["flag"] == "clear"
    assert risk_upper["flag"]  == "clear"
    assert risk_lower["flag"]  == "review_recommended"


# ── apply_lut_grade tests (16–20) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def upscaled_clip(synthetic_raw_loop, passing_decode, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("up")
    out = tmp / "up.mp4"
    upscale_clip(synthetic_raw_loop, out, passing_decode, dry_run=True)
    return out


def test_lut_cool_returns_path(upscaled_clip, passing_decode, tmp_path):
    out = tmp_path / "cool.mp4"
    result = apply_lut_grade(upscaled_clip, out, "cool_authority", passing_decode, dry_run=True)
    assert isinstance(result, Path)


def test_lut_cool_file_exists_and_readable(upscaled_clip, passing_decode, tmp_path):
    out = tmp_path / "cool.mp4"
    apply_lut_grade(upscaled_clip, out, "cool_authority", passing_decode, dry_run=True)
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_lut_neutral_produces_file(upscaled_clip, passing_decode, tmp_path):
    out = tmp_path / "neutral.mp4"
    apply_lut_grade(upscaled_clip, out, "neutral", passing_decode, dry_run=True)
    assert out.exists()


def test_lut_warm_produces_file(upscaled_clip, passing_decode, tmp_path):
    out = tmp_path / "warm.mp4"
    apply_lut_grade(upscaled_clip, out, "warm_tension", passing_decode, dry_run=True)
    assert out.exists()


def test_lut_frame_count_matches_input(upscaled_clip, passing_decode, tmp_path):
    out = tmp_path / "cool_fc.mp4"
    apply_lut_grade(upscaled_clip, out, "cool_authority", passing_decode, dry_run=True)
    cap_in  = cv2.VideoCapture(str(upscaled_clip))
    cap_out = cv2.VideoCapture(str(out))
    fc_in  = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fc_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_in.release()
    cap_out.release()
    assert fc_in > 0
    assert fc_out == fc_in


# ── composite_final tests (21–24) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def center_mask_for_composite(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mask")
    out = tmp / "mask_center.png"
    generate_anchor_mask("center", out, (1920, 1080), dry_run=True)
    return out


@pytest.fixture(scope="module")
def graded_clip(upscaled_clip, passing_decode, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lut")
    out = tmp / "graded.mp4"
    apply_lut_grade(upscaled_clip, out, "cool_authority", passing_decode, dry_run=True)
    return out


def test_composite_returns_path(graded_clip, center_mask_for_composite, tmp_path):
    out = tmp_path / "final.mp4"
    result = composite_final(graded_clip, center_mask_for_composite, out, dry_run=True)
    assert isinstance(result, Path)


def test_composite_file_exists_and_readable(graded_clip, center_mask_for_composite, tmp_path):
    out = tmp_path / "final.mp4"
    composite_final(graded_clip, center_mask_for_composite, out, dry_run=True)
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()


def test_composite_frame_count_matches_graded(graded_clip, center_mask_for_composite, tmp_path):
    out = tmp_path / "final_fc.mp4"
    composite_final(graded_clip, center_mask_for_composite, out, dry_run=True)
    cap_g = cv2.VideoCapture(str(graded_clip))
    cap_o = cv2.VideoCapture(str(out))
    fc_g = int(cap_g.get(cv2.CAP_PROP_FRAME_COUNT))
    fc_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_g.release()
    cap_o.release()
    assert fc_g > 0
    assert fc_o == fc_g


def test_composite_mask_darkening_applied(graded_clip, center_mask_for_composite, tmp_path):
    """Pixels in the anchor zone of the composite must be darker than in the unmasked source."""
    out = tmp_path / "final_dark.mp4"
    composite_final(graded_clip, center_mask_for_composite, out, dry_run=True)

    # Read first frame from graded and composited
    cap_g = cv2.VideoCapture(str(graded_clip))
    ret_g, frame_g = cap_g.read()
    cap_g.release()

    cap_o = cv2.VideoCapture(str(out))
    ret_o, frame_o = cap_o.read()
    cap_o.release()

    assert ret_g and ret_o

    # Sample zone center (~pixel 720, 675 for center zone at 1920x1080)
    zone_cx = int(0.25 * 1920 + 0.5 * 0.50 * 1920)   # 720
    zone_cy = int(0.35 * 1080 + 0.5 * 0.55 * 1080)   # 675

    sample_g = int(np.mean(frame_g[zone_cy, zone_cx].astype(float)))
    sample_o = int(np.mean(frame_o[zone_cy, zone_cx].astype(float)))

    assert sample_o < sample_g or sample_g == 0, (
        f"Expected composite zone pixel ({sample_o}) < graded pixel ({sample_g})"
    )


# ── export_preview tests (25–31) ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def composite_for_preview(graded_clip, center_mask_for_composite, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("comp")
    out = tmp / "final.mp4"
    composite_final(graded_clip, center_mask_for_composite, out, dry_run=True)
    return out


def test_preview_returns_dict(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest = tmp_path / "manifest.json"
    result = export_preview(composite_for_preview, gif, manifest, seam_frames, dry_run=True)
    assert isinstance(result, dict)


def test_preview_gif_file_exists(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest = tmp_path / "manifest.json"
    export_preview(composite_for_preview, gif, manifest, seam_frames, dry_run=True)
    assert gif.exists()


def test_preview_manifest_file_exists(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest = tmp_path / "manifest.json"
    export_preview(composite_for_preview, gif, manifest, seam_frames, dry_run=True)
    assert manifest.exists()


def test_preview_manifest_has_3_segments(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest_path = tmp_path / "manifest.json"
    result = export_preview(composite_for_preview, gif, manifest_path, seam_frames, dry_run=True)
    assert "segments" in result
    assert len(result["segments"]) == 3


def test_preview_segment_labels(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest_path = tmp_path / "manifest.json"
    result = export_preview(composite_for_preview, gif, manifest_path, seam_frames, dry_run=True)
    labels = [s["label"] for s in result["segments"]]
    assert labels == ["base_clip_character", "seam_1_window", "seam_2_window"]


def test_preview_seam1_start_s(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest_path = tmp_path / "manifest.json"
    result = export_preview(composite_for_preview, gif, manifest_path, seam_frames, dry_run=True)
    seam_1_s = seam_frames[0] / 30.0  # 169/30 ≈ 5.6333
    expected_start = seam_1_s - 1     # ≈ 4.6333
    actual_start = result["segments"][1]["start_s"]
    assert abs(actual_start - expected_start) < 1e-6


def test_preview_seam_frames_in_manifest(composite_for_preview, seam_frames, tmp_path):
    gif = tmp_path / "preview.gif"
    manifest_path = tmp_path / "manifest.json"
    result = export_preview(composite_for_preview, gif, manifest_path, seam_frames, dry_run=True)
    assert result["source_seam_frames_playable"] == seam_frames


# ── run_post_processing integration tests (32–38) ───────────────────────────────

@pytest.fixture(scope="module")
def post_result(synthetic_raw_loop, passing_decode, compiled_dict, seam_frames, tmp_path_factory):
    """Full dry-run run_post_processing() result."""
    tmp = tmp_path_factory.mktemp("post")
    return run_post_processing(
        clip_id="test_clip_001",
        raw_loop_path=synthetic_raw_loop,
        decode_probe=passing_decode,
        compiled=compiled_dict,
        seam_frames_playable=seam_frames,
        output_dir=tmp,
        dry_run=True,
    )


def test_post_processing_completes_with_all_keys(post_result):
    required = {
        "clip_id", "upscaled", "masks", "risks",
        "graded_variants", "selected_lut", "luts_generated",
        "final", "preview_gif", "preview_manifest",
    }
    assert required.issubset(set(post_result.keys()))


def test_post_processing_all_file_paths_exist(post_result):
    # upscaled
    assert Path(post_result["upscaled"]).exists()
    # masks
    for path_str in post_result["masks"].values():
        assert Path(path_str).exists(), f"Missing mask: {path_str}"
    # graded variants
    for path_str in post_result["graded_variants"].values():
        assert Path(path_str).exists(), f"Missing graded variant: {path_str}"
    # final
    assert Path(post_result["final"]).exists()
    # preview gif
    assert Path(post_result["preview_gif"]).exists()


def test_luts_generated_always_contains_neutral(
    synthetic_raw_loop, passing_decode, seam_frames, tmp_path
):
    """neutral must be generated even when selected_lut is also neutral."""
    compiled_neutral = {
        "positive":         "test",
        "motion":           "drift",
        "negative":         "",
        "selected_lut":     "neutral",
        "input_hash_short": "deadbeef",
        "compiler_version": "1.0.0",
    }
    result = run_post_processing(
        clip_id="neutral_test",
        raw_loop_path=synthetic_raw_loop,
        decode_probe=passing_decode,
        compiled=compiled_neutral,
        seam_frames_playable=seam_frames,
        output_dir=tmp_path,
        dry_run=True,
    )
    assert "neutral" in result["luts_generated"]
    # No duplicates
    assert len(result["luts_generated"]) == len(set(result["luts_generated"]))


def test_masks_dict_has_exactly_3_positions(post_result):
    assert set(post_result["masks"].keys()) == {"center", "lower_third", "upper_third"}


def test_risks_dict_has_exactly_3_positions(post_result):
    assert set(post_result["risks"].keys()) == {"center", "lower_third", "upper_third"}


def test_missing_raw_loop_raises_file_not_found(
    passing_decode, compiled_dict, seam_frames, tmp_path
):
    with pytest.raises(FileNotFoundError, match="raw_loop not found"):
        run_post_processing(
            clip_id="bad_clip",
            raw_loop_path=Path("/nonexistent/path/raw_loop.mp4"),
            decode_probe=passing_decode,
            compiled=compiled_dict,
            seam_frames_playable=seam_frames,
            output_dir=tmp_path,
            dry_run=True,
        )


def test_risk_json_files_written_as_valid_json(
    synthetic_raw_loop, passing_decode, compiled_dict, seam_frames, tmp_path
):
    result = run_post_processing(
        clip_id="risk_json_test",
        raw_loop_path=synthetic_raw_loop,
        decode_probe=passing_decode,
        compiled=compiled_dict,
        seam_frames_playable=seam_frames,
        output_dir=tmp_path,
        dry_run=True,
    )
    # Risk JSON files live in masks/
    upscaled_path = Path(result["upscaled"])
    masks_dir = upscaled_path.parent.parent / "masks"
    for position in ["center", "lower_third", "upper_third"]:
        risk_file = masks_dir / f"risk_json_test_risk_{position}.json"
        assert risk_file.exists(), f"Missing risk file: {risk_file}"
        with risk_file.open("r") as fh:
            data = json.load(fh)
        assert "flag" in data
        assert "dev_phase_behavior" in data


# ── Regression test (39) ────────────────────────────────────────────────────────

def test_import_does_not_load_torch_or_diffusers():
    """Importing core.post_processor must not pull in torch or diffusers."""
    import importlib
    import sys as _sys

    # Module is already imported — check sys.modules
    assert "torch"     not in _sys.modules, "torch must not be imported by post_processor"
    assert "diffusers" not in _sys.modules, "diffusers must not be imported by post_processor"
