"""
tests/test_integration_wiring.py
─────────────────────────────────
Integration wiring verification.

Verifies that orchestrator stage keys, config structure, LUT files, and
the GPU readiness script are all correctly wired. These tests do NOT call
run_generation() and do NOT write video frames.

Run with:
    pytest tests/test_integration_wiring.py -v
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 11 canonical stage keys (design-doc source of truth) ──────────────────────
CANONICAL_STAGE_KEYS = [
    "prompt_compilation",
    "generation",
    "probe_decode",
    "probe_temporal",
    "gate_evaluation",
    "upscale",
    "mask_generation",
    "lut_grading",
    "composite",
    "preview_export",
    "metadata_assembly",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_gen_constants() -> dict:
    path = PROJECT_ROOT / "config" / "generation_constants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _orchestrator_source() -> str:
    return (PROJECT_ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")


# ── Tests ──────────────────────────────────────────────────────────────────────

# ── REPLACEMENT TESTS (replace deleted test_01, test_02, test_03) ──────────────

def test_01_dev_mode_key_absent():
    """dev_mode key must NOT be present in generation_constants.json."""
    gc = _load_gen_constants()
    assert "dev_mode" not in gc, (
        "dev_mode key found in generation_constants.json — it must be removed"
    )


def test_02_dry_run_constant_absent_from_orchestrator():
    """DRY_RUN constant must NOT appear anywhere in orchestrator.py source."""
    source = _orchestrator_source()
    assert "DRY_RUN" not in source, (
        "DRY_RUN found in orchestrator.py — the global flag must be fully removed"
    )


def test_03_orchestrator_passes_dry_run_false_explicitly():
    """
    orchestrator.py must pass dry_run=False explicitly to pipeline functions.
    dry_run=DRY_RUN must not appear anywhere in the source.
    """
    source = _orchestrator_source()
    assert "dry_run=False" in source, (
        "dry_run=False not found in orchestrator.py — pipeline functions must receive explicit False"
    )
    assert "dry_run=DRY_RUN" not in source, (
        "dry_run=DRY_RUN still present in orchestrator.py — remove the DRY_RUN constant reference"
    )


# ── REPLACEMENT TEST (replaces test_06 in test_constants.py, added here too) ──

def test_extensions_per_clip_is_4():
    """extensions_per_clip must be 4 now that dev_mode cap is removed."""
    gc = _load_gen_constants()
    assert gc["extensions_per_clip"] == 4, (
        f"extensions_per_clip must be 4, got {gc['extensions_per_clip']}"
    )


def test_04_all_11_stage_keys_present_in_orchestrator():
    """Every canonical stage key must appear in orchestrator STAGE_KEYS."""
    from core.orchestrator import STAGE_KEYS
    for key in CANONICAL_STAGE_KEYS:
        assert key in STAGE_KEYS, f"Stage key {key!r} missing from orchestrator STAGE_KEYS"


def test_05_stage_keys_exact_match():
    """
    orchestrator STAGE_KEYS must match the 11 canonical keys exactly —
    no extras, no missing, regardless of order.
    """
    from core.orchestrator import STAGE_KEYS
    assert sorted(STAGE_KEYS) == sorted(CANONICAL_STAGE_KEYS), (
        f"STAGE_KEYS mismatch.\n"
        f"  Orchestrator: {sorted(STAGE_KEYS)}\n"
        f"  Canonical:    {sorted(CANONICAL_STAGE_KEYS)}"
    )
    assert len(STAGE_KEYS) == 11, f"Expected 11 stage keys, got {len(STAGE_KEYS)}"


def test_06_lut_cool_authority_exists():
    """luts/cool_authority.cube must exist at project root."""
    lut = PROJECT_ROOT / "luts" / "cool_authority.cube"
    assert lut.exists(), f"Missing LUT file: {lut}"


def test_07_lut_neutral_exists():
    """luts/neutral.cube must exist at project root."""
    lut = PROJECT_ROOT / "luts" / "neutral.cube"
    assert lut.exists(), f"Missing LUT file: {lut}"


def test_08_lut_warm_tension_exists():
    """luts/warm_tension.cube must exist at project root."""
    lut = PROJECT_ROOT / "luts" / "warm_tension.cube"
    assert lut.exists(), f"Missing LUT file: {lut}"


def test_09_gpu_readiness_script_exists():
    """scripts/gpu_readiness_check.py must exist."""
    script = PROJECT_ROOT / "scripts" / "gpu_readiness_check.py"
    assert script.exists(), f"Missing script: {script}"


def test_10_gpu_readiness_script_importable_no_torch_at_top_level():
    """
    gpu_readiness_check.py must be importable without error on a machine
    that may not have torch installed. torch must NOT be imported at the
    top level.
    """
    script_path = PROJECT_ROOT / "scripts" / "gpu_readiness_check.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Check that torch is not imported at module level (top-level statements only)
    for node in tree.body:
        # Top-level bare import
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch", (
                    "torch imported at top level in gpu_readiness_check.py — must be guarded"
                )
        # Top-level from-import
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch", (
                "torch imported at top level in gpu_readiness_check.py — must be guarded"
            )

    # Script must also be importable (i.e., no syntax or import errors at module level)
    import importlib.util
    spec = importlib.util.spec_from_file_location("gpu_readiness_check", script_path)
    mod = importlib.util.module_from_spec(spec)
    # This executes module-level code — must not raise
    spec.loader.exec_module(mod)


def test_11_practical_rife_folder_exists():
    """Practical-RIFE/ folder must exist at project root."""
    folder = PROJECT_ROOT / "Practical-RIFE"
    assert folder.exists() and folder.is_dir(), f"Missing folder: {folder}"


def test_12_realesrgan_folder_exists():
    """realesrgan/ folder must exist at project root."""
    folder = PROJECT_ROOT / "realesrgan"
    assert folder.exists() and folder.is_dir(), f"Missing folder: {folder}"


def test_13_wan22_folder_exists():
    """Wan2.2/ folder must exist at project root."""
    folder = PROJECT_ROOT / "Wan2.2"
    assert folder.exists() and folder.is_dir(), f"Missing folder: {folder}"


def test_14_output_directory_exists():
    """output/ directory must exist at project root."""
    output = PROJECT_ROOT / "output"
    assert output.exists() and output.is_dir(), f"Missing directory: {output}"
