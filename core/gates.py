"""
core/gates.py
─────────────
Gate Evaluation Layer for the Background Video Generation Module.

Responsibilities:
  1. evaluate_gates() — pure function: reads probe results, applies all 5
     quality thresholds, returns a structured verdict dict.
  2. get_gate_schema() — returns the expected output shape of evaluate_gates().

Design contract:
  This module is the judgment layer. It reads numbers (probe outputs) and
  emits structured decisions. It never runs probes, never touches the
  filesystem, and never calls any generation function.
  evaluate_gates() is referentially transparent: same inputs → same output,
  always. Zero side effects.

Threshold sourcing:
  All 5 gate thresholds are loaded from GENERATION_CONSTANTS["quality_gates"]
  at module level. No threshold value is hardcoded anywhere in this file.
"""

import json
from pathlib import Path

from core.probes import get_probe_schema

# ── Config loading ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = PROJECT_ROOT / "config" / "generation_constants.json"

with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)

# ── Gate thresholds — sourced from config, never hardcoded ──────────────────────
FLICKER_REJECT         = GENERATION_CONSTANTS["quality_gates"]["flicker_index_reject"]
WARPING_REJECT         = GENERATION_CONSTANTS["quality_gates"]["warping_artifact_reject"]
LOOP_QUALITY_THRESHOLD = GENERATION_CONSTANTS["quality_gates"]["loop_quality_threshold"]
LUMINANCE_GATE_MIN     = GENERATION_CONSTANTS["quality_gates"]["luminance_gate_min"]
LUMINANCE_GATE_MAX     = GENERATION_CONSTANTS["quality_gates"]["luminance_gate_max"]


# ── evaluate_gates() ────────────────────────────────────────────────────────────

def evaluate_gates(decode_probe: dict, temporal_probe: dict) -> dict:
    """
    Pure gate evaluation. Reads two probe result dicts, applies all 5
    quality thresholds, and returns a structured verdict dict.

    No I/O. No side effects. Referentially transparent.

    Args:
        decode_probe:   Output dict from run_decode_probe().
        temporal_probe: Output dict from run_temporal_probe().

    Returns:
        {
          "overall":       "pass" | "fail" | "human_review",
          "failures":      [ ...auto_reject entries... ],
          "human_flags":   [ ...human_review entries... ],
          "gates_checked": 5,
          "thresholds_used": { ...all 5 threshold values... }
        }

    Raises:
        ValueError: If either probe dict is missing a required key.
    """
    schema = get_probe_schema()

    # ── Input validation ────────────────────────────────────────────────────────
    decode_keys   = schema["decode_probe_keys"]
    temporal_keys = schema["temporal_probe_keys"]

    for key in decode_keys:
        if key not in decode_probe:
            raise ValueError(
                f"Gate evaluation received malformed probe dict. "
                f"Missing key: {key}. "
                f"Expected keys: {decode_keys}"
            )

    for key in temporal_keys:
        if key not in temporal_probe:
            raise ValueError(
                f"Gate evaluation received malformed probe dict. "
                f"Missing key: {key}. "
                f"Expected keys: {temporal_keys}"
            )

    failures:    list = []
    human_flags: list = []

    # ── Gate 1: Luminance range (decode probe) ──────────────────────────────────
    mean_lum = decode_probe["mean_luminance"]
    if mean_lum < LUMINANCE_GATE_MIN or mean_lum > LUMINANCE_GATE_MAX:
        failures.append({
            "gate":      "luminance_range",
            "metric":    "mean_luminance",
            "value":     mean_lum,
            "threshold": f"{LUMINANCE_GATE_MIN}–{LUMINANCE_GATE_MAX}",
            "verdict":   "auto_reject",
        })

    # ── Gate 2: Flicker index (temporal probe) ──────────────────────────────────
    flicker = temporal_probe["flicker_index"]
    if flicker > FLICKER_REJECT:
        failures.append({
            "gate":      "flicker_index",
            "metric":    "flicker_index",
            "value":     flicker,
            "threshold": FLICKER_REJECT,
            "verdict":   "auto_reject",
        })

    # ── Gate 3: Warping artifact score (temporal probe) ─────────────────────────
    warp = temporal_probe["warping_artifact_score"]
    if warp > WARPING_REJECT:
        failures.append({
            "gate":      "warping_artifact_score",
            "metric":    "warping_artifact_score",
            "value":     warp,
            "threshold": WARPING_REJECT,
            "verdict":   "auto_reject",
        })

    # ── Gate 4: Scene cut detection (temporal probe) ────────────────────────────
    if temporal_probe["scene_cut_detected"] is True:
        failures.append({
            "gate":      "scene_cut",
            "metric":    "scene_cut_detected",
            "value":     True,
            "threshold": False,
            "verdict":   "auto_reject",
        })

    # ── Gate 5: Perceptual loop score (temporal probe) — human flag only ────────
    loop_score = temporal_probe["perceptual_loop_score"]
    if loop_score < LOOP_QUALITY_THRESHOLD:
        human_flags.append({
            "gate":      "perceptual_loop_score",
            "metric":    "perceptual_loop_score",
            "value":     loop_score,
            "threshold": LOOP_QUALITY_THRESHOLD,
            "verdict":   "human_review",
        })

    # ── Overall verdict ─────────────────────────────────────────────────────────
    if failures:
        overall = "fail"
    elif human_flags:
        overall = "human_review"
    else:
        overall = "pass"

    return {
        "overall":       overall,
        "failures":      failures,
        "human_flags":   human_flags,
        "gates_checked": 5,
        "thresholds_used": {
            "flicker_index_reject":    FLICKER_REJECT,
            "warping_artifact_reject": WARPING_REJECT,
            "loop_quality_threshold":  LOOP_QUALITY_THRESHOLD,
            "luminance_gate_min":      LUMINANCE_GATE_MIN,
            "luminance_gate_max":      LUMINANCE_GATE_MAX,
        },
    }


# ── get_gate_schema() ───────────────────────────────────────────────────────────

def get_gate_schema() -> dict:
    """
    Returns the expected output shape of evaluate_gates().

    Used by regenerator.py and the FastAPI layer to validate gate results
    before consuming them.
    """
    return {
        "required_keys": [
            "overall",
            "failures",
            "human_flags",
            "gates_checked",
            "thresholds_used",
        ],
        "overall_values": ["pass", "fail", "human_review"],
        "auto_reject_gates": [
            "luminance_range",
            "flicker_index",
            "warping_artifact_score",
            "scene_cut",
        ],
        "human_review_gates": ["perceptual_loop_score"],
    }
