"""
tests/test_integration_wiring.py
─────────────────────────────────
Integration wiring verification for Prompt 7 (Final integration checkpoint).

Verifies that dev_mode toggle, orchestrator stage keys, LUT files, and
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

def test_01_dev_mode_key_exists():
    """dev_mode key must be present in generation_constants.json."""
    gc = _load_gen_constants()
    assert "dev_mode" in gc, "dev_mode key missing from generation_constants.json"


def test_02_dev_mode_is_true():
    """
    dev_mode must be true on the dev machine.

    This test intentionally fails on the GPU machine after the operator flips
    dev_mode to false — that failure is expected and correct.
    """
    gc = _load_gen_constants()
    assert gc["dev_mode"] is True, (
        "dev_mode is not true in generation_constants.json. "
        "On a GPU machine this is expected after flipping the switch."
    )


def test_03_dry_run_reads_from_config_not_hardcoded():
    """
    DRY_RUN in orchestrator must be assigned from GENERATION_CONSTANTS['dev_mode'],
    not from a hardcoded True or False literal.

    Uses AST inspection so it works without importing torch/diffusers.
    """
    source = _orchestrator_source()
    tree = ast.parse(source)

    # Find the assignment: DRY_RUN = ... or DRY_RUN: bool = ...
    # orchestrator uses an annotated assignment (ast.AnnAssign), not plain ast.Assign
    dry_run_assignment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DRY_RUN":
                    dry_run_assignment = node.value
                    break
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "DRY_RUN":
                dry_run_assignment = node.value
                break

    assert dry_run_assignment is not None, "DRY_RUN assignment not found in orchestrator.py"

    # The RHS must NOT be a bare True/False constant
    assert not isinstance(dry_run_assignment, ast.Constant), (
        "DRY_RUN is assigned a hardcoded constant — must read from GENERATION_CONSTANTS"
    )

    # The RHS source text must reference GENERATION_CONSTANTS and dev_mode
    rhs_src = ast.get_source_segment(source, dry_run_assignment) or ""
    assert "GENERATION_CONSTANTS" in rhs_src, (
        f"DRY_RUN RHS does not reference GENERATION_CONSTANTS: {rhs_src!r}"
    )
    assert "dev_mode" in rhs_src, (
        f"DRY_RUN RHS does not reference dev_mode: {rhs_src!r}"
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
