"""
tests/test_regenerator.py
──────────────────────────
Tests for core/regenerator.py — regeneration loop and escalation logic.

All generation, decode-probe, and temporal-probe callables are injected
as mocks so no real video pipeline is required.
"""

import copy
import json
import pytest
from pathlib import Path

from core.regenerator import (
    regeneration_loop,
    PipelineEscalationError,
    GENERATION_CONSTANTS,
    BASE_CFG,
    MAX_RETRIES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_generation_fn(tmp_path):
    """
    Returns a callable that simulates a successful generation result.
    Creates the output path directory so downstream log writing works.
    """
    def _fn(compiled, run_id, output_dir, seed):
        out_dir = output_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "raw_loop.mp4"
        raw_path.touch()
        return {
            "run_id":               run_id,
            "status":               "complete",
            "raw_loop_path":        str(raw_path),
            "seed":                 seed,
            "seam_frames_raw":      [183, 366],
            "seam_frames_playable": [169, 338],
            "generation_log":       {},
        }
    return _fn


@pytest.fixture
def mock_probe_decode_fn():
    """Always returns the dry-run decode probe result (all gates passing)."""
    def _fn(clip_path):
        return {
            "mean_luminance":       0.46,
            "luminance_range":      [0.14, 0.79],
            "dominant_hue_degrees": 212.0,
            "saturation_mean":      0.38,
            "luminance_gate_min":   0.30,
            "luminance_gate_max":   0.70,
            "dry_run":              True,
        }
    return _fn


@pytest.fixture
def mock_probe_temporal_fn():
    """Always returns the dry-run temporal probe result (all gates passing)."""
    def _fn(clip_path):
        return {
            "flicker_index":          0.003,
            "warping_artifact_score": 0.018,
            "scene_cut_detected":     False,
            "perceptual_loop_score":  0.94,
            "frame_count":            521,
            "dry_run":                True,
        }
    return _fn


@pytest.fixture
def mock_probe_temporal_flicker_fail():
    """Always returns temporal probe with flicker_index=0.025 (fails gate 2)."""
    def _fn(clip_path):
        return {
            "flicker_index":          0.025,
            "warping_artifact_score": 0.018,
            "scene_cut_detected":     False,
            "perceptual_loop_score":  0.94,
            "frame_count":            521,
            "dry_run":                True,
        }
    return _fn


def _run(tmp_path, generation_fn, probe_decode_fn, probe_temporal_fn,
         compiled=None, run_id="test_run", base_seed=42000):
    """Helper to call regeneration_loop with standard tmp_path dirs."""
    if compiled is None:
        compiled = {
            "positive":          "serene cityscape at dusk",
            "motion":            "slow pan left",
            "negative":          "blur, noise",
            "input_hash_short":  "abc123",
            "compiler_version":  "1.0.0",
        }
    return regeneration_loop(
        compiled          = compiled,
        run_id            = run_id,
        output_dir        = tmp_path / "output",
        base_seed         = base_seed,
        generation_fn     = generation_fn,
        probe_decode_fn   = probe_decode_fn,
        probe_temporal_fn = probe_temporal_fn,
        log_dir           = tmp_path / "logs",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────────

def test_regeneration_loop_returns_required_keys(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 1: All-passing mocks return dict with all required keys."""
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    for key in (
        "run_id", "status", "raw_loop_path", "seed_used", "attempts_used",
        "gate_result", "failure_log", "decode_probe", "temporal_probe",
        "gate_evaluation", "seam_frames_raw", "seam_frames_playable",
    ):
        assert key in result, f"Missing key: {key}"


def test_regeneration_loop_gate_result_pass(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 2: gate_result == 'pass' when all probes return clean values."""
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    assert result["gate_result"] == "pass"


def test_regeneration_loop_attempts_used_first(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 3: attempts_used == 1 on first-attempt pass."""
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    assert result["attempts_used"] == 1


def test_regeneration_loop_seed_used_attempt1(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 4: seed_used == base_seed + 100 (attempt 1 seed strategy)."""
    base_seed = 50000
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn,
                  base_seed=base_seed)
    assert result["seed_used"] == base_seed + 100


def test_regeneration_loop_failure_log_one_entry(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 5: failure_log has exactly 1 entry on first-attempt pass."""
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    assert len(result["failure_log"]) == 1


def test_regeneration_loop_writes_log_file(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 6: A generation_log JSON file is written to log_dir."""
    log_dir = tmp_path / "logs"
    result = _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    log_files = list(log_dir.glob("*_generation_log.json"))
    assert len(log_files) == 1


def test_regeneration_loop_log_file_valid_json(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 7: The written log file is valid JSON and contains 'run_id'."""
    log_dir = tmp_path / "logs"
    _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn,
         run_id="json_test_run")
    log_file = log_dir / "json_test_run_generation_log.json"
    assert log_file.exists()
    with log_file.open() as fh:
        data = json.load(fh)
    assert "run_id" in data


def test_regeneration_loop_escalation_raised(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 8: When probe always fails, all 3 attempts exhausted and PipelineEscalationError raised."""
    with pytest.raises(PipelineEscalationError):
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail)


def test_regeneration_loop_escalation_failure_log_count(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 9: PipelineEscalationError carries failure_log with 3 entries."""
    with pytest.raises(PipelineEscalationError) as exc_info:
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail)
    assert len(exc_info.value.failure_log) == 3


def test_regeneration_loop_escalation_run_id(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 10: PipelineEscalationError.run_id matches the run_id argument."""
    run_id = "escalation_test_run"
    with pytest.raises(PipelineEscalationError) as exc_info:
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail,
             run_id=run_id)
    assert exc_info.value.run_id == run_id


def test_regeneration_loop_log_written_on_escalation(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 11: Generation log is written to disk even when escalation occurs."""
    log_dir = tmp_path / "logs"
    run_id  = "escalation_log_test"
    with pytest.raises(PipelineEscalationError):
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail,
             run_id=run_id)
    log_file = log_dir / f"{run_id}_generation_log.json"
    assert log_file.exists()


def test_regeneration_loop_attempt3_cfg(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 12: Attempt 3 uses cfg = BASE_CFG - 0.5."""
    with pytest.raises(PipelineEscalationError) as exc_info:
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail)
    attempt3_entry = exc_info.value.failure_log[2]
    assert attempt3_entry["cfg_used"] == pytest.approx(BASE_CFG - 0.5)


def test_regeneration_loop_attempt3_motion_mutation(tmp_path, mock_probe_decode_fn, mock_probe_temporal_flicker_fail):
    """Test 13: Attempt 3 injects stability suffix into motion prompt."""
    recorded_calls = []

    def recording_generation_fn(compiled, run_id, output_dir, seed):
        recorded_calls.append({"attempt_motion": compiled["motion"]})
        out_dir = output_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "raw_loop.mp4"
        raw_path.touch()
        return {
            "run_id":               run_id,
            "status":               "complete",
            "raw_loop_path":        str(raw_path),
            "seed":                 seed,
            "seam_frames_raw":      [183, 366],
            "seam_frames_playable": [169, 338],
            "generation_log":       {},
        }

    with pytest.raises(PipelineEscalationError):
        _run(tmp_path, recording_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail)

    assert len(recorded_calls) == 3
    assert "extra temporal stability" in recorded_calls[2]["attempt_motion"]


def test_regeneration_loop_compiled_not_mutated(
    tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail
):
    """Test 14: The compiled dict passed in is not mutated after 3-attempt exhausted run."""
    compiled = {
        "positive":         "serene cityscape at dusk",
        "motion":           "slow pan left",
        "negative":         "blur, noise",
        "input_hash_short": "abc123",
        "compiler_version": "1.0.0",
    }
    original_motion = compiled["motion"]
    with pytest.raises(PipelineEscalationError):
        _run(tmp_path, mock_generation_fn, mock_probe_decode_fn, mock_probe_temporal_flicker_fail,
             compiled=compiled)
    assert compiled["motion"] == original_motion


def test_regeneration_loop_generation_error_continues(
    tmp_path, mock_probe_decode_fn, mock_probe_temporal_fn
):
    """Test 15: If generation_fn raises on attempt 1, loop continues to attempt 2."""
    call_count = {"n": 0}

    def flaky_generation_fn(compiled, run_id, output_dir, seed):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Simulated generation failure on attempt 1")
        out_dir = output_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "raw_loop.mp4"
        raw_path.touch()
        return {
            "run_id":               run_id,
            "status":               "complete",
            "raw_loop_path":        str(raw_path),
            "seed":                 seed,
            "seam_frames_raw":      [183, 366],
            "seam_frames_playable": [169, 338],
            "generation_log":       {},
        }

    result = _run(tmp_path, flaky_generation_fn, mock_probe_decode_fn, mock_probe_temporal_fn)
    assert result["attempts_used"] == 2
    assert call_count["n"] == 2


def test_pipeline_escalation_error_importable():
    """Test 16: PipelineEscalationError is importable from core.regenerator."""
    from core.regenerator import PipelineEscalationError as PSE
    assert PSE is not None


# ── Regression: no heavy ML imports ─────────────────────────────────────────────

def test_gates_no_torch_import():
    """Test 17: Importing core.gates does not import torch or diffusers."""
    import sys
    import importlib
    # Ensure fresh import
    if "core.gates" in sys.modules:
        importlib.reload(sys.modules["core.gates"])
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules


def test_regenerator_no_torch_import():
    """Test 18: Importing core.regenerator does not import torch or diffusers."""
    import sys
    import importlib
    if "core.regenerator" in sys.modules:
        importlib.reload(sys.modules["core.regenerator"])
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules
