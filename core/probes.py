"""
core/probes.py
──────────────
Probe Stage for the Background Video Generation Module.

Responsibilities:
  1. run_decode_probe()   — photometric analysis via FFmpeg (luminance + color)
  2. run_temporal_probe() — motion + consistency analysis via OpenCV + scikit-image
  3. get_probe_schema()   — returns expected key sets for shape validation

Design contract:
  Probes are pure measurement functions. They return numbers and facts.
  They never make pass/fail decisions — that is the gate evaluator's job
  (Prompt 6). No threshold value from quality_gates is evaluated here.

Dry-run mode (dry_run=True):
  Both probes return fully-shaped synthetic dicts with plausible values.
  No FFmpeg or OpenCV calls are made. This allows downstream stages
  (Prompt 6 gate evaluation, Prompt 7 post-processing) to be tested
  without real video assets.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ── Config loading ──────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
_CONFIG_PATH    = PROJECT_ROOT / "config" / "generation_constants.json"

with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)


# ── run_decode_probe() ──────────────────────────────────────────────────────────

def run_decode_probe(clip_path: Path, dry_run: bool = False) -> dict:
    """
    Photometric analysis of the clip.

    Measures luminance properties and color characteristics.
    Returns numbers only — no pass/fail evaluation.

    Args:
        clip_path: Path to the MP4 to analyse.
        dry_run:   If True, return a synthetic result dict without any
                   FFmpeg or OpenCV call.

    Returns:
        {
          "mean_luminance":       float,         # normalized 0.0–1.0
          "luminance_range":      [float, float],# [min, max] normalized
          "dominant_hue_degrees": float,         # 0–360
          "saturation_mean":      float,         # 0.0–1.0
          "luminance_gate_min":   float,         # from GENERATION_CONSTANTS
          "luminance_gate_max":   float,         # from GENERATION_CONSTANTS
          "dry_run":              bool,
        }
    """
    gates = GENERATION_CONSTANTS["quality_gates"]
    lum_gate_min = gates["luminance_gate_min"]
    lum_gate_max = gates["luminance_gate_max"]

    if dry_run:
        return {
            "mean_luminance":       0.46,
            "luminance_range":      [0.14, 0.79],
            "dominant_hue_degrees": 212.0,
            "saturation_mean":      0.38,
            "luminance_gate_min":   lum_gate_min,
            "luminance_gate_max":   lum_gate_max,
            "dry_run":              True,
        }

    # ── Step 1: Luminance via FFmpeg signalstats ────────────────────────────────
    # Primary path: FFmpeg signalstats for YAVG-based luminance.
    # Fallback path: cv2 grayscale when FFmpeg is not on PATH.
    # The cv2 fallback is authoritative in test environments where FFmpeg may
    # not be installed; the FFmpeg path is preferred in production.
    cmd_lum = [
        "ffmpeg", "-i", str(clip_path),
        "-vf", "signalstats=stat=tout+vrep+brng",
        "-f", "null", "-",
    ]
    _ffmpeg_available = True
    try:
        ffmpeg_result = subprocess.run(
            cmd_lum,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,  # we check manually to capture stderr
        )
    except FileNotFoundError:
        _ffmpeg_available = False
        ffmpeg_result = None

    if _ffmpeg_available and ffmpeg_result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg decode probe failed for {clip_path}: "
            f"{ffmpeg_result.stderr.decode('utf-8', errors='replace')}"
        )

    yavg_values = []
    if _ffmpeg_available and ffmpeg_result is not None:
        # Parse YAVG values from FFmpeg stderr output
        # Lines look like: [Parsed_signalstats_0 @ 0x...] YAVG:123.45 ...
        stderr_text = ffmpeg_result.stderr.decode("utf-8", errors="replace")
        for line in stderr_text.splitlines():
            if "YAVG:" in line:
                try:
                    segment = line.split("YAVG:")[1].split()[0]
                    yavg_values.append(float(segment))
                except (IndexError, ValueError):
                    continue

    if yavg_values:
        mean_luminance   = float(np.mean(yavg_values)) / 255.0
        luminance_range  = [float(np.min(yavg_values)) / 255.0,
                            float(np.max(yavg_values)) / 255.0]
    else:
        # FFmpeg not available or returned no YAVG lines — fall back to cv2 grayscale
        cap = cv2.VideoCapture(str(clip_path))
        gray_means = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_means.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
        cap.release()
        if gray_means:
            mean_luminance  = float(np.mean(gray_means)) / 255.0
            luminance_range = [float(np.min(gray_means)) / 255.0,
                               float(np.max(gray_means)) / 255.0]
        else:
            mean_luminance  = 0.0
            luminance_range = [0.0, 0.0]

    # ── Step 2: Hue + saturation via cv2 HSV (first frame) ─────────────────────
    # The FFmpeg hue filter output format varies across platforms and versions.
    # The cv2 HSV path is authoritative for hue/saturation measurement.
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"FFmpeg decode probe: cv2 cannot open {clip_path} for HSV analysis."
        )
    ret, first_frame = cap.read()
    cap.release()

    if ret and first_frame is not None:
        hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
        # cv2 H channel: 0–179 (degrees/2); S channel: 0–255
        dominant_hue_degrees = float(np.mean(hsv[:, :, 0])) * 2.0   # scale to 0–360
        saturation_mean      = float(np.mean(hsv[:, :, 1])) / 255.0  # normalize to 0–1
    else:
        dominant_hue_degrees = 0.0
        saturation_mean      = 0.0

    return {
        "mean_luminance":       mean_luminance,
        "luminance_range":      luminance_range,
        "dominant_hue_degrees": dominant_hue_degrees,
        "saturation_mean":      saturation_mean,
        "luminance_gate_min":   lum_gate_min,
        "luminance_gate_max":   lum_gate_max,
        "dry_run":              False,
    }


# ── run_temporal_probe() ────────────────────────────────────────────────────────

def run_temporal_probe(clip_path: Path, dry_run: bool = False) -> dict:
    """
    Temporal consistency analysis of the clip.

    Measures motion stability, flicker, scene cuts, and loop quality.
    Returns numbers and facts only — no pass/fail evaluation.

    Args:
        clip_path: Path to the MP4 to analyse.
        dry_run:   If True, return a synthetic result dict without any
                   OpenCV call.

    Returns:
        {
          "flicker_index":          float,  # mean abs frame-to-frame lum diff, 0–1
          "warping_artifact_score": float,  # mean Farneback flow variance
          "scene_cut_detected":     bool,   # True if Bhattacharyya dist > 0.40
          "perceptual_loop_score":  float,  # SSIM between first and last frame
          "frame_count":            int,
          "dry_run":                bool,
        }
    """
    if dry_run:
        return {
            "flicker_index":          0.003,
            "warping_artifact_score": 0.018,
            "scene_cut_detected":     False,
            "perceptual_loop_score":  0.94,
            "frame_count":            521,   # matches playable_frames from generator seam math
            "dry_run":                True,
        }

    # ── Step 1: Load all frames ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video file for temporal probe: {clip_path}"
        )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frame_count = len(frames)

    if frame_count < 2:
        raise RuntimeError(
            f"Temporal probe requires at least 2 frames. "
            f"{clip_path} has {frame_count} frame(s)."
        )

    # ── Step 2: Flicker index (IES RP-16 adapted) ──────────────────────────────
    lums = [
        float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) / 255.0
        for f in frames
    ]
    flicker_index = float(
        np.mean([abs(lums[n] - lums[n - 1]) for n in range(1, len(lums))])
    )

    # ── Step 3: Warping artifact score (Farneback dense optical flow) ───────────
    # TODO (production): consider GPU-accelerated flow or sampling every Nth frame;
    # computing 520 Farneback flows on a 521-frame clip is intentionally thorough
    # but will be slow on CPU. For now correctness over speed.
    flow_variances = []
    for n in range(1, frame_count):
        prev_gray = cv2.cvtColor(frames[n - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[n],     cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_variances.append(float(np.var(magnitude)))

    warping_artifact_score = float(np.mean(flow_variances))

    # ── Step 4: Scene cut detection (Bhattacharyya histogram distance) ──────────
    # 0.40 is a signal-processing constant for the Bhattacharyya distance
    # threshold that reliably separates hard cuts from gradual transitions.
    # It is NOT a quality gate threshold — do not move it to generation_constants.json.
    BHATTACHARYYA_CUT_THRESHOLD = 0.40  # signal-processing constant, not a quality gate

    scene_cut_detected = False
    for n in range(1, frame_count):
        prev_gray = cv2.cvtColor(frames[n - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[n],     cv2.COLOR_BGR2GRAY)
        hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256]).flatten()
        hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256]).flatten()
        # Gaussian smoothing before comparison prevents near-delta-function
        # histograms (solid-color synthetic frames) from producing spurious
        # maximum-distance values when adjacent bins shift by 1.  Real scene
        # cuts (disjoint intensity distributions) still produce distance > 0.40.
        _kernel = np.exp(-0.5 * np.arange(-6, 7) ** 2 / 9.0)   # sigma=3
        _kernel /= _kernel.sum()
        hist_prev = np.convolve(hist_prev, _kernel, mode="same").reshape(-1, 1).astype(np.float32)
        hist_curr = np.convolve(hist_curr, _kernel, mode="same").reshape(-1, 1).astype(np.float32)
        # Normalization is required by OpenCV for HISTCMP_BHATTACHARYYA to be well-defined
        cv2.normalize(hist_prev, hist_prev, alpha=1, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_curr, hist_curr, alpha=1, norm_type=cv2.NORM_L1)
        dist = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
        if dist > BHATTACHARYYA_CUT_THRESHOLD:
            scene_cut_detected = True
            break

    # ── Step 5: Perceptual loop score (SSIM between first and last frame) ───────
    from skimage.metrics import structural_similarity as ssim  # already pinned: scikit-image==0.22.0

    first = cv2.cvtColor(frames[0],  cv2.COLOR_BGR2GRAY)
    last  = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    perceptual_loop_score = float(ssim(first, last))

    return {
        "flicker_index":          flicker_index,
        "warping_artifact_score": warping_artifact_score,
        "scene_cut_detected":     scene_cut_detected,
        "perceptual_loop_score":  perceptual_loop_score,
        "frame_count":            frame_count,
        "dry_run":                False,
    }


# ── get_probe_schema() ──────────────────────────────────────────────────────────

def get_probe_schema() -> dict:
    """
    Returns the expected key sets for both probes.

    Used by the FastAPI layer (Prompt 3 stubs) and the gate evaluator
    (Prompt 6) to validate probe output shape before consuming it.
    """
    return {
        "decode_probe_keys": [
            "mean_luminance",
            "luminance_range",
            "dominant_hue_degrees",
            "saturation_mean",
            "luminance_gate_min",
            "luminance_gate_max",
            "dry_run",
        ],
        "temporal_probe_keys": [
            "flicker_index",
            "warping_artifact_score",
            "scene_cut_detected",
            "perceptual_loop_score",
            "frame_count",
            "dry_run",
        ],
    }
