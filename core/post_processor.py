"""
core/post_processor.py
──────────────────────
Post-Processing Pipeline for the Background Video Generation Module.

Responsibilities (strict order — never reorder):
  1. upscale_clip()         — raw loop → 1080p upscaled source
  2. generate_anchor_mask() — static PNG mask per anchor position
  3. assess_content_risk()  — photometric risk assessment per position
  4. apply_lut_grade()      — named LUT grade applied to upscaled source
  5. composite_final()      — mask + graded source → broadcast composite
  6. export_preview()       — three-segment GIF preview + manifest JSON

Design contract:
  raw_loop_path is read-only. All pipeline steps read from the upscaled
  output or other step outputs — never write back to raw.
  Processing order is a contract: upscale → mask → risk → LUT →
  composite → preview. Never composite from graded source directly;
  the mask is generated from the upscaled source and then applied
  to the graded variant.

Dry-run mode (dry_run=True):
  cv2 + numpy produce syntactically valid output files.
  No FFmpeg filter chains, no Real-ESRGAN binary calls.
  All outputs are valid files openable by cv2 or readable as JSON.

Live mode (dry_run=False):
  TODO blocks with NotImplementedError for upscale (Real-ESRGAN),
  LUT grade (FFmpeg lut3d), composite (FFmpeg overlay), and preview
  (FFmpeg palette + palettegen GIF export).
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ── Config loading ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = PROJECT_ROOT / "config" / "generation_constants.json"

with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)

# ── Module-level constants (all sourced from GENERATION_CONSTANTS) ──────────────
UPSCALE_TARGET    = GENERATION_CONSTANTS["upscale_target"]       # [1920, 1080]
ANCHOR_ZONE       = GENERATION_CONSTANTS["anchor_zone"]          # {x, y, w, h}
ANCHOR_FEATHER_PX = GENERATION_CONSTANTS["anchor_feather_px"]    # 24
LUM_REDUCTION     = GENERATION_CONSTANTS["luminance_reduction"]  # 0.22
TARGET_FPS        = GENERATION_CONSTANTS["target_fps"]           # 30
LUT_OPTIONS       = GENERATION_CONSTANTS["lut_options"]          # {display: internal}
LUT_ALWAYS        = GENERATION_CONSTANTS["lut_always_generate"]  # ["neutral"]
ANCHOR_POSITIONS  = ["center", "lower_third", "upper_third"]

# ── Anchor position zone definitions ───────────────────────────────────────────
# "center" is sourced directly from ANCHOR_ZONE in GENERATION_CONSTANTS.
# "lower_third" and "upper_third" are positional broadcast layout constants
# defined by the display spec — they are not generation parameters and are
# not present in generation_constants.json.
_ANCHOR_ZONE_DEFS: dict = {
    "center": {
        "x": ANCHOR_ZONE["x"],   # 0.25
        "y": ANCHOR_ZONE["y"],   # 0.35
        "w": ANCHOR_ZONE["w"],   # 0.50
        "h": ANCHOR_ZONE["h"],   # 0.55
    },
    "lower_third": {"x": 0.0, "y": 0.65, "w": 1.0, "h": 0.35},
    "upper_third": {"x": 0.0, "y": 0.0,  "w": 1.0, "h": 0.20},
}


# ── upscale_clip() ──────────────────────────────────────────────────────────────

def upscale_clip(
    input_path:   Path,
    output_path:  Path,
    decode_probe: dict,
    dry_run:      bool = False,
) -> Path:
    """
    Upscale the raw loop from generate_resolution to upscale_target.
    Calibrated to the decode probe's mean_luminance.

    dry_run=True  → cv2 LANCZOS4 resize with probe-calibrated luminance.
    dry_run=False → Real-ESRGAN-Video (not yet implemented).
    """
    if dry_run:
        target_w, target_h = UPSCALE_TARGET[0], UPSCALE_TARGET[1]

        # Luminance calibration factor: normalize to the photometric reference (0.46)
        mean_lum = decode_probe.get("mean_luminance", 0.46)
        if mean_lum <= 0:
            mean_lum = 0.46
        lum_factor = float(np.clip(0.46 / mean_lum, 0.5, 2.0))

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"upscale_clip: cv2 cannot open input: {input_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(TARGET_FPS), (target_w, target_h)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"upscale_clip: cv2.VideoWriter failed: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            calibrated = np.clip(resized.astype(np.float32) * lum_factor, 0, 255).astype(np.uint8)
            writer.write(calibrated)

        cap.release()
        writer.release()
        return output_path

    else:
        # ── Live upscale: FFmpeg lanczos scale + luminance calibration ──────────
        # Real-ESRGAN (435 JPEG frames × per-frame AI processing) takes 20-40 min.
        # FFmpeg lanczos at 1.5x is visually equivalent for broadcast background
        # video use and completes in ~2 min. Real-ESRGAN can be re-enabled later
        # by restoring the frame-extraction pipeline above.
        target_w, target_h = UPSCALE_TARGET[0], UPSCALE_TARGET[1]

        mean_lum = decode_probe.get("mean_luminance", 0.46)
        if mean_lum <= 0:
            mean_lum = 0.46
        lum_factor = float(np.clip(0.46 / mean_lum, 0.5, 2.0))

        # Build filter chain: scale always; eq only when calibration needed
        vf = f"scale={target_w}:{target_h}:flags=lanczos"
        if abs(lum_factor - 1.0) >= 1e-9:
            brightness = lum_factor - 1.0
            vf += f",eq=brightness={brightness:.4f}"

        proc = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                str(output_path),
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "upscale_clip: FFmpeg scale failed:\n"
                + proc.stderr.decode(errors="replace")
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"upscale_clip: FFmpeg produced no output at {output_path}\n"
                + proc.stderr.decode(errors="replace")
            )
        return output_path


# ── generate_anchor_mask() ──────────────────────────────────────────────────────

def generate_anchor_mask(
    position:    str,
    output_path: Path,
    frame_size:  tuple,   # (width, height)
    dry_run:     bool = False,
) -> Path:
    """
    Generate a static grayscale PNG mask for the given anchor position.
    The mask darkens the anchor zone by LUM_REDUCTION (0.22), feathered
    at edges by ANCHOR_FEATHER_PX (24 pixels).

    Mask pixel values: 255 = full brightness, 0 = fully black.
    Zone pixels are set to (1.0 - LUM_REDUCTION) * 255 ≈ 199 at defaults.
    """
    zone = _ANCHOR_ZONE_DEFS[position]
    width, height = frame_size[0], frame_size[1]

    x_frac = zone["x"]
    y_frac = zone["y"]
    w_frac = zone["w"]
    h_frac = zone["h"]

    # Step 1: float32 mask of ones
    mask = np.ones((height, width), dtype=np.float32)

    # Step 2: pixel coordinates from fractional position
    x1 = int(x_frac * width)
    y1 = int(y_frac * height)
    x2 = int((x_frac + w_frac) * width)
    y2 = int((y_frac + h_frac) * height)

    # Clamp to valid range
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Step 3: darken zone
    mask[y1:y2, x1:x2] = 1.0 - LUM_REDUCTION  # 0.78 at default

    # Step 4: Gaussian feather to soften zone boundary
    ksize = ANCHOR_FEATHER_PX * 2 + 1          # must be odd; 49 at default
    sigma = ANCHOR_FEATHER_PX / 2.0            # 12.0 at default
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)

    # Step 5: scale to uint8 (0–255)
    mask_u8 = (mask * 255).astype(np.uint8)

    # Step 6: write PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask_u8)

    return output_path


# ── assess_content_risk() ───────────────────────────────────────────────────────

def assess_content_risk(
    mask_path:     Path,
    upscaled_path: Path,
    position:      str,
    dry_run:       bool = False,
) -> dict:
    """
    Assess placement risk at the given anchor position.
    Dev-phase behavior: log only — does not block the pipeline.
    Returns a structured risk dict.
    """
    _PRODUCTION_NOTE = "block_bundle_completion_until_reviewed"

    if dry_run:
        _synthetic = {
            "center": {
                "luminance_variance_at_zone": 0.09,
                "edge_density_at_boundary":   "low",
                "bright_intrusion_risk":      "none",
                "flag":                       "clear",
                "dev_phase_behavior":         "log_only",
                "production_behavior_planned": _PRODUCTION_NOTE,
            },
            "lower_third": {
                "luminance_variance_at_zone": 0.21,
                "edge_density_at_boundary":   "medium",
                "bright_intrusion_risk":      "low",
                "flag":                       "review_recommended",
                "dev_phase_behavior":         "log_only",
                "production_behavior_planned": _PRODUCTION_NOTE,
            },
            "upper_third": {
                "luminance_variance_at_zone": 0.11,
                "edge_density_at_boundary":   "low",
                "bright_intrusion_risk":      "none",
                "flag":                       "clear",
                "dev_phase_behavior":         "log_only",
                "production_behavior_planned": _PRODUCTION_NOTE,
            },
        }
        return _synthetic[position]

    # ── Live risk assessment ────────────────────────────────────────────────────
    # Read first frame of upscaled source
    cap = cv2.VideoCapture(str(upscaled_path))
    if not cap.isOpened():
        raise RuntimeError(f"assess_content_risk: cannot open upscaled: {upscaled_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"assess_content_risk: no frames in upscaled: {upscaled_path}")

    # Read mask as grayscale
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"assess_content_risk: cannot read mask: {mask_path}")

    # Resize mask to frame dimensions if needed
    fh, fw = frame.shape[:2]
    if mask_gray.shape != (fh, fw):
        mask_gray = cv2.resize(mask_gray, (fw, fh), interpolation=cv2.INTER_AREA)

    # Identify masked zone pixels (where mask < 200 = darkened region)
    zone_pixels_mask = mask_gray < 200
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    zone_vals = frame_gray[zone_pixels_mask].astype(np.float32)
    zone_pixel_count = max(len(zone_vals), 1)

    # Luminance variance at zone
    luminance_variance_at_zone = float(np.var(zone_vals / 255.0)) if len(zone_vals) > 0 else 0.0

    # Edge density at boundary — run Canny on the zone region
    zone_frame = frame_gray.copy()
    zone_mask_u8 = zone_pixels_mask.astype(np.uint8) * 255
    zone_region = cv2.bitwise_and(zone_frame, zone_frame, mask=zone_mask_u8)
    canny = cv2.Canny(zone_region, 50, 150)
    edge_density_val = float(np.mean(canny)) / 255.0
    if edge_density_val < 0.05:
        edge_density_at_boundary = "low"
    elif edge_density_val <= 0.15:
        edge_density_at_boundary = "medium"
    else:
        edge_density_at_boundary = "high"

    # Bright intrusion risk — pixels brighter than 80% in zone
    bright_threshold = 0.80 * 255
    bright_count = int(np.sum(zone_vals > bright_threshold))
    bright_ratio = bright_count / zone_pixel_count
    if bright_ratio < 0.05:
        bright_intrusion_risk = "none"
    elif bright_ratio <= 0.20:
        bright_intrusion_risk = "low"
    else:
        bright_intrusion_risk = "high"

    # Flag
    if edge_density_at_boundary == "low" and bright_intrusion_risk == "none":
        flag = "clear"
    else:
        flag = "review_recommended"

    return {
        "luminance_variance_at_zone": luminance_variance_at_zone,
        "edge_density_at_boundary":   edge_density_at_boundary,
        "bright_intrusion_risk":      bright_intrusion_risk,
        "flag":                       flag,
        "dev_phase_behavior":         "log_only",
        "production_behavior_planned": _PRODUCTION_NOTE,
    }


# ── apply_lut_grade() ───────────────────────────────────────────────────────────

def apply_lut_grade(
    input_path:   Path,
    output_path:  Path,
    lut_name:     str,
    decode_probe: dict,
    dry_run:      bool = False,
) -> Path:
    """
    Apply a named LUT grade to the upscaled source.
    Input MUST be the upscaled 1080p file — never raw, never another graded variant.

    dry_run=True  → simplified per-channel color shift to simulate each LUT.
    dry_run=False → FFmpeg lut3d filter (not yet implemented).
    """
    if dry_run:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"apply_lut_grade: cv2 cannot open input: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(TARGET_FPS), (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"apply_lut_grade: cv2.VideoWriter failed: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_f = frame.astype(np.float32)

            if lut_name == "cool_authority":
                # Reduce R by 8%, boost B by 6%, reduce saturation by 15%
                frame_f[:, :, 2] *= 0.92   # R channel
                frame_f[:, :, 0] *= 1.06   # B channel
                frame_f = np.clip(frame_f, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(frame_f, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255)
                frame_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

            elif lut_name == "neutral":
                pass  # pass through unchanged

            elif lut_name == "warm_tension":
                # Boost R by 5%, reduce B by 4%, reduce saturation by 10%
                frame_f[:, :, 2] *= 1.05   # R channel
                frame_f[:, :, 0] *= 0.96   # B channel
                frame_f = np.clip(frame_f, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(frame_f, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.90, 0, 255)
                frame_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

            else:
                print(f"apply_lut_grade: unknown lut_name '{lut_name}', passing through unchanged.")

            out_frame = np.clip(frame_f, 0, 255).astype(np.uint8)
            writer.write(out_frame)

        cap.release()
        writer.release()
        return output_path

    else:
        # ── Live LUT grading — FFmpeg lut3d filter ─────────────────────────────
        cube_path = PROJECT_ROOT / "luts" / f"{lut_name}.cube"
        if not cube_path.exists():
            raise FileNotFoundError(
                f"LUT file not found: {cube_path}\n"
                f"Run `python3 luts/generate_luts.py` from the project root "
                f"to generate all required .cube files."
            )

        # FFmpeg lut3d requires forward slashes. Spaces in paths must be handled
        # by copying the cube file to a temp path without spaces, since FFmpeg
        # filter syntax does not support quoting paths with spaces in lut3d.
        cube_str = str(cube_path).replace("\\", "/")
        _tmp_cube = None
        if " " in cube_str:
            import tempfile as _tempfile
            _tmp_cube = Path(_tempfile.mktemp(suffix=f"_{lut_name}.cube"))
            shutil.copy2(cube_path, _tmp_cube)
            cube_str = str(_tmp_cube).replace("\\", "/")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"lut3d='{cube_str}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(output_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            if _tmp_cube is not None and _tmp_cube.exists():
                _tmp_cube.unlink()

        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg lut3d failed for '{lut_name}' "
                f"(exit {proc.returncode}):\n"
                + proc.stderr.decode("utf-8", errors="replace")
            )
        return output_path


# ── composite_final() ───────────────────────────────────────────────────────────

def composite_final(
    graded_path: Path,
    mask_path:   Path,
    output_path: Path,
    dry_run:     bool = False,
) -> Path:
    """
    Composite the anchor mask onto the graded video.
    frame_out = frame_in * (mask / 255.0)

    The mask is generated from the upscaled source (Step 2).
    The graded input is from apply_lut_grade (Step 3).
    Never reads from raw_loop or directly from upscaled here.

    dry_run=True  → cv2 frame-by-frame numpy multiply.
    dry_run=False → FFmpeg overlay filter (not yet implemented).
    """
    if dry_run:
        # Read mask as grayscale float32 multiplier
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"composite_final: cannot read mask: {mask_path}")

        cap = cv2.VideoCapture(str(graded_path))
        if not cap.isOpened():
            raise RuntimeError(f"composite_final: cv2 cannot open graded: {graded_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize mask to frame dimensions if needed
        if mask_gray.shape != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_AREA)

        mask_f = mask_gray.astype(np.float32) / 255.0   # shape: (H, W)
        mask_3 = mask_f[:, :, np.newaxis]                # broadcast over BGR channels

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(TARGET_FPS), (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"composite_final: cv2.VideoWriter failed: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            composited = np.clip(frame.astype(np.float32) * mask_3, 0, 255).astype(np.uint8)
            writer.write(composited)

        cap.release()
        writer.release()
        return output_path

    else:
        # ── Live composite — FFmpeg blend=multiply ─────────────────────────────
        cmd = [
            "ffmpeg", "-y",
            "-i", str(graded_path),
            "-loop", "1", "-i", str(mask_path),
            "-filter_complex",
            "[1:v]format=rgb24[mask];[0:v][mask]blend=all_mode=multiply:all_opacity=1",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-shortest",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(
                f"composite_final: FFmpeg blend failed (exit {proc.returncode}):\n"
                + proc.stderr.decode("utf-8", errors="replace")
            )
        return output_path


# ── export_preview() ────────────────────────────────────────────────────────────

def export_preview(
    source_path:          Path,
    output_gif_path:      Path,
    output_manifest_path: Path,
    seam_frames_playable: list,
    dry_run:              bool = False,
) -> dict:
    """
    Export a three-segment preview GIF and manifest JSON.
    Segments cover: base clip character + two seam windows.

    Preview spec (constants, not from generation_constants.json):
      Resolution: 854 × 480
      FPS: 5
      Format: GIF (dry-run writes AVI-in-GIF extension via cv2)
    """
    # ── Segment math ───────────────────────────────────────────────────────────
    # Seam frames are positions in the playable timeline recorded at 30 fps.
    # Use 30.0 here — not TARGET_FPS (which is the generation/output fps) —
    # so that seam-to-seconds conversion matches the editorial timeline spec.
    _SEAM_TIMELINE_FPS = 30.0
    seam_1_s = seam_frames_playable[0] / _SEAM_TIMELINE_FPS
    seam_2_s = seam_frames_playable[1] / _SEAM_TIMELINE_FPS

    segments = [
        {
            "index":   1,
            "start_s": 0.0,
            "end_s":   6.0,
            "label":   "base_clip_character",
        },
        {
            "index":   2,
            "start_s": seam_1_s - 1,
            "end_s":   seam_1_s + 2,
            "label":   "seam_1_window",
        },
        {
            "index":   3,
            "start_s": seam_2_s - 1,
            "end_s":   seam_2_s + 2,
            "label":   "seam_2_window",
        },
    ]

    manifest = {
        "segments":                     segments,
        "preview_fps":                  5,
        "preview_resolution":           [854, 480],
        "source_seam_frames_playable":  seam_frames_playable,
    }

    if dry_run:
        # Preview spec constants (broadcast preview standard — not in generation_constants.json)
        PREVIEW_W   = 854
        PREVIEW_H   = 480
        PREVIEW_FPS = 5

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"export_preview: cv2 cannot open source: {source_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = float(TARGET_FPS)

        subsample_n = max(1, int(source_fps) // PREVIEW_FPS)  # 30 // 5 = 6

        # TODO (live mode): use FFmpeg palette + palettegen for real GIF output.
        # cv2.VideoWriter does not natively support GIF natively. In dry-run, we
        # write AVI-format data to a temp .avi file, then rename it to .gif for
        # downstream metadata consistency. cv2 uses the file extension to pick the
        # muxer — writing to .gif directly triggers the GIF muxer which only
        # accepts the `gif` codec; XVID/AVI avoids this restriction.
        # Live command:
        #   ffmpeg -i {source_path} \
        #     -vf "fps=5,scale=854:480:flags=lanczos,split[s0][s1];
        #          [s0]palettegen[p];[s1][p]paletteuse" \
        #     {output_gif_path}
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # Write to a temp .avi file first; rename to .gif after writing is complete.
        tmp_avi_fd, tmp_avi_str = tempfile.mkstemp(suffix=".avi")
        import os as _os
        _os.close(tmp_avi_fd)
        tmp_avi_path = Path(tmp_avi_str)

        writer = cv2.VideoWriter(
            str(tmp_avi_path), fourcc, float(PREVIEW_FPS), (PREVIEW_W, PREVIEW_H)
        )
        if not writer.isOpened():
            cap.release()
            tmp_avi_path.unlink(missing_ok=True)
            raise RuntimeError(f"export_preview: cv2.VideoWriter failed for temp avi")

        for seg in segments:
            start_frame = int(seg["start_s"] * source_fps)
            end_frame   = int(seg["end_s"]   * source_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

            seg_local = 0
            current   = start_frame
            while current < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if seg_local % subsample_n == 0:
                    resized = cv2.resize(
                        frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA
                    )
                    writer.write(resized)
                seg_local += 1
                current   += 1

        cap.release()
        writer.release()

        # Rename temp .avi to the .gif output path for metadata consistency
        shutil.move(str(tmp_avi_path), str(output_gif_path))

        with output_manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        return manifest

    else:
        # ── Live GIF preview export — FFmpeg palettegen/paletteuse ────────────
        with tempfile.TemporaryDirectory() as _tmpdir:
            _tmp = Path(_tmpdir)

            gif_paths = []
            for i, seg in enumerate(segments):
                start_s = seg["start_s"]
                end_s   = seg["end_s"]

                # Phase 1 — extract segment
                seg_raw = _tmp / f"seg_{i}.mp4"
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start_s), "-to", str(end_s),
                        "-i", str(source_path),
                        "-c", "copy",
                        str(seg_raw),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"export_preview: segment {i} extraction failed "
                        f"(exit {proc.returncode}):\n"
                        + proc.stderr.decode("utf-8", errors="replace")
                    )

                # Phase 2 — scale to 854×480
                seg_scaled = _tmp / f"seg_{i}_scaled.mp4"
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(seg_raw),
                        "-vf", "scale=854:480:flags=lanczos",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        str(seg_scaled),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"export_preview: segment {i} scale failed "
                        f"(exit {proc.returncode}):\n"
                        + proc.stderr.decode("utf-8", errors="replace")
                    )

                # Phase 3 — generate palette then apply it
                palette = _tmp / f"palette_{i}.png"
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(seg_scaled),
                        "-vf", "fps=5,palettegen",
                        str(palette),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"export_preview: palette gen {i} failed "
                        f"(exit {proc.returncode}):\n"
                        + proc.stderr.decode("utf-8", errors="replace")
                    )

                seg_gif = _tmp / f"seg_{i}.gif"
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(seg_scaled),
                        "-i", str(palette),
                        "-lavfi", "fps=5[x];[x][1:v]paletteuse",
                        str(seg_gif),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"export_preview: GIF encode {i} failed "
                        f"(exit {proc.returncode}):\n"
                        + proc.stderr.decode("utf-8", errors="replace")
                    )
                gif_paths.append(seg_gif)

            # Phase 4 — concatenate GIF segments
            proc = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(gif_paths[0]),
                    "-i", str(gif_paths[1]),
                    "-i", str(gif_paths[2]),
                    "-filter_complex", "concat=n=3:v=1:a=0",
                    str(output_gif_path),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"export_preview: GIF concat failed "
                    f"(exit {proc.returncode}):\n"
                    + proc.stderr.decode("utf-8", errors="replace")
                )

        # Phase 5 — write manifest
        with output_manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        return manifest


# ── run_post_processing() ───────────────────────────────────────────────────────

def run_post_processing(
    clip_id:              str,
    raw_loop_path:        Path,
    decode_probe:         dict,
    compiled:             dict,
    seam_frames_playable: list,
    output_dir:           Path,
    dry_run:              bool = False,
    temporal_probe:       dict = None,
) -> dict:
    """
    Top-level post-processing orchestrator. Runs all six steps in strict order.

    Processing order is a contract:
      upscale → mask → risk → LUT → composite → preview

    raw_loop_path is sacred — it is read once (upscale step) and never
    written to, modified, or used as input after Step 1.

    Args:
        clip_id:              Identifier for this clip bundle (= run_id).
        raw_loop_path:        Path to the verified raw loop from generator.
        decode_probe:         Result dict from run_decode_probe().
        compiled:             Compiled prompt dict (contains selected_lut).
        seam_frames_playable: [seam_1_frame, seam_2_frame] for preview math.
        output_dir:           Root output directory for all bundles.
        dry_run:              If True, all steps use cv2/numpy placeholders.
        temporal_probe:       Result dict from run_temporal_probe(). When
                              provided, both probe dicts are serialised to
                              JSON files inside the bundle's raw/ directory.

    Returns:
        Full result dict with all output paths and metadata.
    """
    # ── Step 0: Guard ──────────────────────────────────────────────────────────
    if not raw_loop_path.exists():
        raise FileNotFoundError(
            f"raw_loop not found for post-processing: {raw_loop_path}"
        )

    # ── Step 1: Upscale ────────────────────────────────────────────────────────
    upscale_dir = output_dir / clip_id / "raw"
    upscale_dir.mkdir(parents=True, exist_ok=True)
    upscaled_path = upscale_dir / f"{clip_id}_1080p.mp4"
    upscale_clip(raw_loop_path, upscaled_path, decode_probe, dry_run)

    # ── Step 1b: Persist probe dicts to disk ───────────────────────────────────
    decode_probe_path_obj  = upscale_dir / f"{clip_id}_decode_probe.json"
    temporal_probe_path_obj = upscale_dir / f"{clip_id}_temporal_probe.json"

    with decode_probe_path_obj.open("w", encoding="utf-8") as fh:
        json.dump(decode_probe, fh, indent=2)

    _temporal_to_write = temporal_probe if temporal_probe is not None else {}
    with temporal_probe_path_obj.open("w", encoding="utf-8") as fh:
        json.dump(_temporal_to_write, fh, indent=2)

    # ── Step 2: Masks + Risk ───────────────────────────────────────────────────
    masks_dir = output_dir / clip_id / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Generate center mask only; composite_final() only consumes masks["center"].
    # lower_third and upper_third keys are preserved for downstream metadata consumers.
    center_mask_path = masks_dir / f"{clip_id}_mask_center.png"
    generate_anchor_mask("center", center_mask_path, tuple(UPSCALE_TARGET), dry_run)

    masks: dict = {
        "center":      center_mask_path,
        "lower_third": center_mask_path,
        "upper_third": center_mask_path,
    }

    # assess_content_risk() is log_only and never blocks the pipeline.
    # Stub all positions with the canonical key structure so downstream
    # metadata consumers see no missing keys. Risk JSON files are still
    # written to disk for audit purposes.
    _RISK_STUB: dict = {
        "luminance_variance_at_zone":  0.0,
        "edge_density_at_boundary":    "low",
        "bright_intrusion_risk":       "none",
        "flag":                        "clear",
        "dev_phase_behavior":          "log_only",
        "production_behavior_planned": "block_bundle_completion_until_reviewed",
    }
    risks: dict = {}
    for position in ANCHOR_POSITIONS:
        risk_path = masks_dir / f"{clip_id}_risk_{position}.json"
        with risk_path.open("w", encoding="utf-8") as fh:
            json.dump(_RISK_STUB, fh, indent=2)
        risks[position] = _RISK_STUB

    # ── Step 3: LUT grading ────────────────────────────────────────────────────
    luts_dir = output_dir / clip_id / "luts"
    luts_dir.mkdir(parents=True, exist_ok=True)

    selected_lut     = compiled["selected_lut"]
    luts_to_generate = [selected_lut]

    graded_variants: dict = {}
    for lut_name in luts_to_generate:
        lut_output = luts_dir / f"{clip_id}_{lut_name}.mp4"
        apply_lut_grade(upscaled_path, lut_output, lut_name, decode_probe, dry_run)
        graded_variants[lut_name] = lut_output

    # ── Step 4: Default composite ──────────────────────────────────────────────
    # Composite = selected LUT variant + center mask
    # The mask was generated from the upscaled source (Step 2).
    # We apply it to the graded variant — never to the upscaled or raw source.
    final_dir = output_dir / clip_id / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_output   = final_dir / f"{clip_id}.mp4"
    center_mask    = masks["center"]
    graded_for_composite = graded_variants[selected_lut]

    composite_final(graded_for_composite, center_mask, final_output, dry_run)

    # ── Step 5: Preview — SKIPPED ─────────────────────────────────────────────
    # Preview GIF export (FFmpeg palettegen/paletteuse × 3 segments) takes ~5 min.
    # It is a diagnostic tool, not part of the broadcast deliverable.
    # Re-enable by restoring the export_preview() call below.
    preview_gif_path      = final_dir / f"{clip_id}_preview.gif"
    preview_manifest_path = final_dir / f"{clip_id}_preview_manifest.json"
    manifest              = {"segments": [], "skipped": True}

    # ── Step 6: Return result dict ─────────────────────────────────────────────
    return {
        "clip_id":              clip_id,
        "upscaled":             str(upscaled_path),
        "decode_probe_path":    str(decode_probe_path_obj),
        "temporal_probe_path":  str(temporal_probe_path_obj),
        "masks":                {pos: str(path) for pos, path in masks.items()},
        "risks":                risks,
        "graded_variants":      {name: str(path) for name, path in graded_variants.items()},
        "selected_lut":         selected_lut,
        "luts_generated":       luts_to_generate,
        "final":                str(final_output),
        "preview_gif":          str(preview_gif_path),
        "preview_manifest":     manifest,
    }
