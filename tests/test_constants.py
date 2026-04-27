"""
tests/test_constants.py

Pytest tests for the constants layer.
Verifies structural integrity and logical invariants of both JSON config files.
Run with: pytest tests/
"""

import json
import subprocess
import sys
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
    "model_id",
    "model_checkpoint",
    "model_commit_hash",
    "model_architecture",
    "model_task",
    "upscaler",
    "upscaler_model_weights",
    "interpolation_method",
    "interpolation_status",
    "temporal_probe_library",
    "ssim_library",
    "ffmpeg_version",
]

GEN_REQUIRED_KEYS = [
    "model",
    # "tokenizer" removed — not applicable to Wan2.2-TI2V-5B-Diffusers
    "sampler",
    "steps",
    "cfg_scale",
    "native_fps",
    "target_fps",
    "generate_resolution",
    "upscale_target",
    "upscale_factor",
    "vae_compression",
    "generation_modes",
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


# ── Test 6: extensions_per_clip == 4 (dev_mode removed, production value) ─────
def test_dev_mode_extensions_cap(gen_cfg):
    """extensions_per_clip must be 4 now that dev_mode is removed."""
    assert gen_cfg["extensions_per_clip"] == 4, (
        f"extensions_per_clip must be 4, got {gen_cfg['extensions_per_clip']}"
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


# ── Test 9: Model identity updated to Wan2.2-TI2V-5B-Diffusers ───────────────
def test_model_name(gen_cfg, env_cfg):
    """model field in generation_constants and model_id in environment_constants must reference Wan2.2."""
    assert gen_cfg["model"] == "Wan2.2-TI2V-5B-Diffusers", (
        f"generation_constants model must be 'Wan2.2-TI2V-5B-Diffusers', got {gen_cfg['model']!r}"
    )
    assert env_cfg["model_id"] == "Wan2.2-TI2V-5B-Diffusers", (
        f"environment_constants model_id must be 'Wan2.2-TI2V-5B-Diffusers', got {env_cfg['model_id']!r}"
    )


# ── Test 10: native_fps is 24 ────────────────────────────────────────────────
def test_native_fps(gen_cfg):
    """native_fps must be 24 (Wan2.2-TI2V-5B native rate, not CogVideoX 8fps)."""
    assert gen_cfg["native_fps"] == 24, (
        f"native_fps must be 24, got {gen_cfg['native_fps']} — stale config?"
    )


# ── Test 11: generate_resolution is [1280, 720] ───────────────────────────────
def test_generate_resolution_values(gen_cfg):
    """generate_resolution must be [1280, 720] for Wan2.2-TI2V-5B."""
    assert gen_cfg["generate_resolution"] == [1280, 720], (
        f"generate_resolution must be [1280, 720], got {gen_cfg['generate_resolution']}"
    )


# ── Test 12: base_clip_duration_s and base_clip_frames_native ────────────────
def test_base_clip_values(gen_cfg):
    """base_clip_duration_s must be 6.04 and base_clip_frames_native must be 145."""
    assert gen_cfg["base_clip_duration_s"] == 6.04, (
        f"base_clip_duration_s must be 6.04, got {gen_cfg['base_clip_duration_s']}"
    )
    assert gen_cfg["base_clip_frames_native"] == 145, (
        f"base_clip_frames_native must be 145, got {gen_cfg['base_clip_frames_native']}"
    )


# ── Test 13: playable_duration_s and total_loop_duration_s values ─────────────
def test_duration_values(gen_cfg):
    """playable_duration_s must be 16.917 and total_loop_duration_s must be 18.12."""
    assert gen_cfg["playable_duration_s"] == 16.917, (
        f"playable_duration_s must be 16.917, got {gen_cfg['playable_duration_s']}"
    )
    assert gen_cfg["total_loop_duration_s"] == 18.12, (
        f"total_loop_duration_s must be 18.12, got {gen_cfg['total_loop_duration_s']}"
    )


# ── Test 14: model_checkpoint key present in environment_constants ─────────────
def test_model_checkpoint_key_present(env_cfg):
    """environment_constants.json must have model_checkpoint, not cogvideox_checkpoint."""
    assert "model_checkpoint" in env_cfg, "model_checkpoint key missing from environment_constants.json"
    assert "cogvideox_checkpoint" not in env_cfg, (
        "cogvideox_checkpoint must be removed from environment_constants.json"
    )


# ── Test A: vae_compression block exists and has correct keys and values ───────
def test_vae_compression_block(gen_cfg):
    """vae_compression must exist and contain temporal=4, spatial_h=16, spatial_w=16."""
    assert "vae_compression" in gen_cfg, "vae_compression block missing from generation_constants.json"
    block = gen_cfg["vae_compression"]
    assert block["temporal"] == 4, f"vae_compression.temporal must be 4, got {block['temporal']}"
    assert block["spatial_h"] == 16, f"vae_compression.spatial_h must be 16, got {block['spatial_h']}"
    assert block["spatial_w"] == 16, f"vae_compression.spatial_w must be 16, got {block['spatial_w']}"


# ── Test B: generation_modes block exists and has correct values ───────────────
def test_generation_modes_block(gen_cfg):
    """generation_modes must have base_clip=T2V and extension_1=I2V."""
    assert "generation_modes" in gen_cfg, "generation_modes block missing from generation_constants.json"
    modes = gen_cfg["generation_modes"]
    assert modes["base_clip"] == "T2V", f"generation_modes.base_clip must be 'T2V', got {modes['base_clip']!r}"
    assert modes["extension_1"] == "I2V", f"generation_modes.extension_1 must be 'I2V', got {modes['extension_1']!r}"


# ── Test C: seam_frames_raw_timeline ──────────────────────────────────────────
def test_seam_frames_raw_timeline(gen_cfg):
    """seam_frames_raw_timeline must be [145, 290]."""
    assert gen_cfg["seam_frames_raw_timeline"] == [145, 290], (
        f"seam_frames_raw_timeline must be [145, 290], got {gen_cfg['seam_frames_raw_timeline']}"
    )


# ── Test D: seam_frames_playable_timeline ─────────────────────────────────────
def test_seam_frames_playable_timeline(gen_cfg):
    """seam_frames_playable_timeline must be [138, 269]."""
    assert gen_cfg["seam_frames_playable_timeline"] == [138, 269], (
        f"seam_frames_playable_timeline must be [138, 269], got {gen_cfg['seam_frames_playable_timeline']}"
    )


# ── Test E: upscale_factor is 1.5 ─────────────────────────────────────────────
def test_upscale_factor(gen_cfg):
    """upscale_factor must be 1.5 (1280→1920, cleaner than old 2.67×)."""
    assert gen_cfg["upscale_factor"] == 1.5, (
        f"upscale_factor must be 1.5, got {gen_cfg['upscale_factor']}"
    )


# ── Test F: validate_environment.py exits 0 on valid config ───────────────────
def test_validate_environment_exits_zero():
    """validate_environment.py must exit with code 0 when config is valid."""
    validate_script = Path(__file__).parent.parent / "validate_environment.py"
    result = subprocess.run(
        [sys.executable, str(validate_script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"validate_environment.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── Test G: quality_gates block is unchanged ──────────────────────────────────
def test_quality_gates_unchanged(gen_cfg):
    """quality_gates spot-check: flicker_index_reject and luminance_gate_min must be unchanged."""
    gates = gen_cfg["quality_gates"]
    assert gates["flicker_index_reject"] == 0.01, (
        f"quality_gates.flicker_index_reject must be 0.01, got {gates['flicker_index_reject']}"
    )
    assert gates["luminance_gate_min"] == 0.30, (
        f"quality_gates.luminance_gate_min must be 0.30, got {gates['luminance_gate_min']}"
    )
