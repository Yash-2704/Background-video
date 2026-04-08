"""
tests/test_constants.py

Pytest tests for the constants layer.
Verifies structural integrity and logical invariants of both JSON config files.
Run with: pytest tests/
"""

import json
from pathlib import Path

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
ENV_CONSTANTS_PATH = BASE_DIR / "config" / "environment_constants.json"
GEN_CONSTANTS_PATH = BASE_DIR / "config" / "generation_constants.json"

ENV_REQUIRED_KEYS = [
    "python_version",
    "cuda_version",
    "torch_version",
    "diffusers_version",
    "cogvideox_checkpoint",
    "cogvideox_commit_hash",
    "upscaler",
    "upscaler_model_weights",
    "interpolation_method",
    "temporal_probe_library",
    "ssim_library",
    "ffmpeg_version",
]

GEN_REQUIRED_KEYS = [
    "model",
    "tokenizer",
    "sampler",
    "steps",
    "cfg_scale",
    "native_fps",
    "target_fps",
    "generate_resolution",
    "upscale_target",
    "base_clip_duration_s",
    "base_clip_frames_native",
    "total_anchor_screen_time_s",
    "extensions_per_clip",
    "total_loop_duration_s",
    "playable_duration_s",
    "crossfade_frames",
    "crossfade_duration_s",
    "seam_count",
    "total_crossfade_consumed_s",
    "anchor_position_default",
    "anchor_zone",
    "anchor_feather_px",
    "luminance_reduction",
    "max_regeneration_retries",
    "quality_gates",
    "lut_options",
    "lut_specifications",
    "dev_mode",
    "production_note",
]


@pytest.fixture(scope="module")
def env_cfg() -> dict:
    with open(ENV_CONSTANTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gen_cfg() -> dict:
    with open(GEN_CONSTANTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Test 1: Both JSON files load without error ────────────────────────────────
def test_environment_constants_loads():
    """environment_constants.json must parse as valid JSON."""
    with open(ENV_CONSTANTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_generation_constants_loads():
    """generation_constants.json must parse as valid JSON."""
    with open(GEN_CONSTANTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


# ── Test 2: All required keys in environment_constants.json ──────────────────
def test_environment_constants_required_keys(env_cfg):
    """All required top-level keys must be present in environment_constants.json."""
    missing = [k for k in ENV_REQUIRED_KEYS if k not in env_cfg]
    assert missing == [], f"Missing keys in environment_constants.json: {missing}"


# ── Test 3: All required keys in generation_constants.json ───────────────────
def test_generation_constants_required_keys(gen_cfg):
    """All required top-level keys must be present in generation_constants.json."""
    missing = [k for k in GEN_REQUIRED_KEYS if k not in gen_cfg]
    assert missing == [], f"Missing keys in generation_constants.json: {missing}"


# ── Test 4: playable_duration_s < total_loop_duration_s ──────────────────────
def test_playable_duration_less_than_total_loop(gen_cfg):
    """Playable duration must be less than the total loop duration."""
    assert gen_cfg["playable_duration_s"] < gen_cfg["total_loop_duration_s"], (
        f"playable_duration_s ({gen_cfg['playable_duration_s']}) must be < "
        f"total_loop_duration_s ({gen_cfg['total_loop_duration_s']})"
    )


# ── Test 5: luminance_gate_min < luminance_gate_max ──────────────────────────
def test_luminance_gate_range(gen_cfg):
    """quality_gates.luminance_gate_min must be strictly less than luminance_gate_max."""
    gates = gen_cfg["quality_gates"]
    assert gates["luminance_gate_min"] < gates["luminance_gate_max"], (
        f"luminance_gate_min ({gates['luminance_gate_min']}) must be < "
        f"luminance_gate_max ({gates['luminance_gate_max']})"
    )


# ── Test 6: extensions_per_clip == 2 when dev_mode is true ───────────────────
def test_dev_mode_extensions_cap(gen_cfg):
    """When dev_mode is true, extensions_per_clip must be capped at 2."""
    if gen_cfg["dev_mode"]:
        assert gen_cfg["extensions_per_clip"] == 2, (
            f"dev_mode is true but extensions_per_clip is {gen_cfg['extensions_per_clip']} "
            "(expected 2 — production cap is 4)"
        )


# ── Test 7: generate_resolution has exactly 2 integer elements ───────────────
def test_generate_resolution_shape(gen_cfg):
    """generate_resolution must be a list of exactly 2 integers."""
    res = gen_cfg["generate_resolution"]
    assert isinstance(res, list), "generate_resolution must be a list"
    assert len(res) == 2, f"generate_resolution must have exactly 2 elements, got {len(res)}"
    assert all(isinstance(v, int) for v in res), (
        f"generate_resolution elements must be integers, got {res}"
    )


# ── Test 8: upscale_target elements are both larger than generate_resolution ──
def test_upscale_target_larger_than_generate_resolution(gen_cfg):
    """Each dimension of upscale_target must be strictly larger than the corresponding generate_resolution dimension."""
    src = gen_cfg["generate_resolution"]
    dst = gen_cfg["upscale_target"]
    assert isinstance(dst, list) and len(dst) == 2, "upscale_target must be a list of 2 elements"
    assert dst[0] > src[0], (
        f"upscale_target width ({dst[0]}) must be > generate_resolution width ({src[0]})"
    )
    assert dst[1] > src[1], (
        f"upscale_target height ({dst[1]}) must be > generate_resolution height ({src[1]})"
    )
