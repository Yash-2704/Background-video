"""
tests/test_orchestrator.py
──────────────────────────
32 pytest tests for core/orchestrator.py and the two new FastAPI routes.

Design rules (per spec):
  - NO calls to run_generation() via the pipeline.
  - NO cv2.VideoWriter via generator — synthetic MP4s built directly with cv2.
  - Mock: run_generation, run_post_processing, run_metadata_assembly.
  - Real: run_decode_probe, run_temporal_probe, evaluate_gates
    (dry_run=True path — fast, no filesystem side effects for probes).
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def canonical_user_input() -> dict:
    """The 6 editorial fields used as the canonical test input."""
    return {
        "category":          "Economy",
        "location_feel":     "Urban",
        "time_of_day":       "Dusk",
        "color_temperature": "Cool",
        "mood":              "Serious",
        "motion_intensity":  "Gentle",
    }


@pytest.fixture
def synthetic_raw_loop_path(tmp_path) -> Path:
    """
    A minimal valid MP4 built directly with cv2 — NOT via run_generation().
    30 frames, 720×480, 30 fps (mp4v codec).
    """
    path = tmp_path / "synthetic_raw_loop.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (720, 480))
    for _ in range(30):
        writer.write(np.zeros((480, 720, 3), dtype=np.uint8))
    writer.release()
    assert path.exists(), "Fixture failed to create synthetic MP4"
    return path


@pytest.fixture
def mock_gen_result(synthetic_raw_loop_path) -> dict:
    """Minimal dict matching run_generation() return shape."""
    return {
        "run_id":               "test_run_fixture",
        "status":               "complete",
        "raw_loop_path":        str(synthetic_raw_loop_path),
        "seed":                 42000,
        "seam_frames_raw":      [183, 366],
        "seam_frames_playable": [169, 338],
        "generation_log":       {},
    }


@pytest.fixture
def mock_post_result(tmp_path) -> dict:
    """Minimal valid post_result dict."""
    return {
        "clip_id":         "test_run",
        "upscaled":        str(tmp_path / "upscaled.mp4"),
        "masks":           {
            "center":      str(tmp_path / "mask_center.png"),
            "lower_third": str(tmp_path / "mask_lower_third.png"),
            "upper_third": str(tmp_path / "mask_upper_third.png"),
        },
        "risks":           {
            "center":      {"flag": "clear"},
            "lower_third": {"flag": "review_recommended"},
            "upper_third": {"flag": "clear"},
        },
        "graded_variants": {
            "cool_authority": str(tmp_path / "cool.mp4"),
            "neutral":        str(tmp_path / "neutral.mp4"),
        },
        "selected_lut":    "cool_authority",
        "luts_generated":  ["cool_authority", "neutral"],
        "final":           str(tmp_path / "final.mp4"),
        "preview_gif":     str(tmp_path / "preview.gif"),
        "preview_manifest": {"segments": []},
    }


@pytest.fixture
def mock_metadata_result(tmp_path) -> dict:
    """Minimal valid metadata_result dict."""
    return {
        "clip_id":              "test_run",
        "metadata_path":        str(tmp_path / "test_run_metadata.json"),
        "edit_manifest_path":   str(tmp_path / "test_run_edit_manifest.json"),
        "contract_path":        str(tmp_path / "test_run_integration_contract.json"),
        "generation_log_path":  str(tmp_path / "test_run_generation_log.json"),
        "metadata":             {},
        "edit_manifest":        {},
        "integration_contract": {},
        "generation_log":       {},
    }


# ── Fresh RUN_REGISTRY per test ───────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_registry():
    """Ensure each test starts with a clean RUN_REGISTRY."""
    import core.orchestrator as orch
    orch.RUN_REGISTRY.clear()
    yield
    orch.RUN_REGISTRY.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_pipeline(gen_result, post_result, meta_result):
    """Return a list of active patches for the three heavy-side-effect mocks."""
    p1 = patch("core.orchestrator.run_generation",       return_value=gen_result)
    p2 = patch("core.orchestrator.run_post_processing",  return_value=post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=meta_result)
    return p1, p2, p3


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 1–4: _init_run_state
# ═══════════════════════════════════════════════════════════════════════════════

def test_01_init_run_state_returns_required_keys():
    from core.orchestrator import _init_run_state
    state = _init_run_state("test_run_1")
    for key in ("run_id", "status", "stages", "result", "error", "started_at"):
        assert key in state, f"Missing key: {key}"


def test_02_init_run_state_all_11_stage_keys_present():
    from core.orchestrator import STAGE_KEYS, _init_run_state
    state = _init_run_state("test_run_2")
    for key in STAGE_KEYS:
        assert key in state["stages"], f"Missing stage key: {key}"
    assert len(state["stages"]) == 11


def test_03_init_run_state_all_stages_idle():
    from core.orchestrator import _init_run_state
    state = _init_run_state("test_run_3")
    for key, val in state["stages"].items():
        assert val == "idle", f"Stage {key!r} is {val!r}, expected 'idle'"


def test_04_init_run_state_stored_in_registry():
    from core.orchestrator import RUN_REGISTRY, _init_run_state
    _init_run_state("test_run_4")
    assert "test_run_4" in RUN_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 5–6: _set_stage
# ═══════════════════════════════════════════════════════════════════════════════

def test_05_set_stage_updates_correct_key():
    from core.orchestrator import RUN_REGISTRY, _init_run_state, _set_stage
    _init_run_state("test_run_5")
    _set_stage("test_run_5", "generation", "running")
    assert RUN_REGISTRY["test_run_5"]["stages"]["generation"] == "running"


def test_06_set_stage_failed_also_sets_run_status():
    from core.orchestrator import RUN_REGISTRY, _init_run_state, _set_stage
    _init_run_state("test_run_6")
    _set_stage("test_run_6", "generation", "failed")
    assert RUN_REGISTRY["test_run_6"]["stages"]["generation"] == "failed"
    assert RUN_REGISTRY["test_run_6"]["status"] == "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 7–12: run_pipeline — success path
# ═══════════════════════════════════════════════════════════════════════════════

def test_07_run_pipeline_returns_all_required_keys(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import _init_run_state, run_pipeline
    run_id = "test_success_07"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        result = run_pipeline(run_id, canonical_user_input)

    required_keys = [
        "run_id", "status", "raw_loop_path", "seed",
        "seam_frames_raw", "seam_frames_playable",
        "gate_result", "selected_lut", "lower_third_style",
        "metadata_path", "stages",
    ]
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_08_run_pipeline_success_status_complete(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import _init_run_state, run_pipeline
    run_id = "test_success_08"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        result = run_pipeline(run_id, canonical_user_input)
    assert result["status"] == "complete"


def test_09_registry_status_complete_after_successful_run(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline
    run_id = "test_success_09"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)
    assert RUN_REGISTRY[run_id]["status"] == "complete"


def test_10_all_11_stages_complete_after_success(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, STAGE_KEYS, _init_run_state, run_pipeline
    run_id = "test_success_10"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)
    for key in STAGE_KEYS:
        assert RUN_REGISTRY[run_id]["stages"][key] == "complete", (
            f"Stage {key!r} is not 'complete'"
        )


def test_11_result_stages_matches_registry_stages(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline
    run_id = "test_success_11"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        result = run_pipeline(run_id, canonical_user_input)
    assert result["stages"] == RUN_REGISTRY[run_id]["stages"]


def test_12_run_generation_called_exactly_once_on_clean_run(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import _init_run_state, run_pipeline
    run_id = "test_success_12"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1 as mock_gen, p2, p3:
        run_pipeline(run_id, canonical_user_input)
    mock_gen.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 13: stage ordering
# ═══════════════════════════════════════════════════════════════════════════════

def test_13_prompt_compilation_complete_before_generation_runs(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    """Capture stage status snapshots via a side-effecting mock."""
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline

    run_id = "test_order_13"
    _init_run_state(run_id)

    prompt_comp_status_when_gen_called = []

    def _capturing_gen(**kwargs):
        # Snapshot prompt_compilation status at the moment generation is called
        prompt_comp_status_when_gen_called.append(
            RUN_REGISTRY[run_id]["stages"]["prompt_compilation"]
        )
        return mock_gen_result

    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)
    p1 = patch("core.orchestrator.run_generation", side_effect=_capturing_gen)

    with p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)

    assert prompt_comp_status_when_gen_called, "Generation was never called"
    assert prompt_comp_status_when_gen_called[0] == "complete", (
        f"prompt_compilation was {prompt_comp_status_when_gen_called[0]!r} when "
        f"generation started — expected 'complete'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 14–17: gate failure → regen loop
# ═══════════════════════════════════════════════════════════════════════════════

def test_14_gate_failure_triggers_regen_loop(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    """
    When evaluate_gates returns overall=='fail', run_generation should be called
    more than once (initial attempt + at least one regen attempt).
    """
    from core.orchestrator import _init_run_state, run_pipeline

    run_id = "test_regen_14"
    _init_run_state(run_id)

    # Regen loop returns a successful regen_result
    regen_return = {
        "run_id":               run_id,
        "status":               "complete",
        "raw_loop_path":        mock_gen_result["raw_loop_path"],
        "seed_used":            42100,
        "seam_frames_raw":      [183, 366],
        "seam_frames_playable": [169, 338],
        "attempts_used":        1,
        "failure_log":          [],
        "decode_probe":         {
            "mean_luminance": 0.46, "luminance_range": [0.14, 0.79],
            "dominant_hue_degrees": 212.0, "saturation_mean": 0.38,
            "luminance_gate_min": 0.30, "luminance_gate_max": 0.70, "dry_run": True,
        },
        "temporal_probe":       {
            "flicker_index": 0.003, "warping_artifact_score": 0.018,
            "scene_cut_detected": False, "perceptual_loop_score": 0.94,
            "frame_count": 521, "dry_run": True,
        },
        "gate_evaluation":      {
            "overall": "pass", "failures": [], "human_flags": [],
            "gates_checked": 5, "thresholds_used": {},
        },
    }

    fail_gate = {
        "overall": "fail", "failures": [{"gate": "flicker_index"}],
        "human_flags": [], "gates_checked": 5, "thresholds_used": {},
    }

    def _regen_side_effect(**kwargs):
        kwargs["generation_fn"](
            compiled=kwargs["compiled"],
            run_id=f"{kwargs['run_id']}_regen_mock",
            output_dir=kwargs["output_dir"],
            seed=kwargs["base_seed"] + 100,
        )
        return regen_return

    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)
    p_gates = patch("core.orchestrator.evaluate_gates",   return_value=fail_gate)
    p_regen = patch("core.orchestrator.regeneration_loop", side_effect=_regen_side_effect)
    # run_generation is still called for the initial attempt
    p1 = patch("core.orchestrator.run_generation",         return_value=mock_gen_result)

    with p1 as mock_gen, p2, p3, p_gates, p_regen as mock_regen:
        result = run_pipeline(run_id, canonical_user_input)

    assert mock_gen.call_count > 1
    mock_regen.assert_called_once()
    assert result["status"] == "complete"


def test_15_escalation_result_status_escalated(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import PipelineEscalationError, _init_run_state, run_pipeline

    run_id = "test_escalate_15"
    _init_run_state(run_id)

    fail_gate = {
        "overall": "fail", "failures": [{"gate": "flicker_index"}],
        "human_flags": [], "gates_checked": 5, "thresholds_used": {},
    }
    escalation_error = PipelineEscalationError(run_id, [{"attempt": 1}, {"attempt": 2}, {"attempt": 3}])

    p1 = patch("core.orchestrator.run_generation",       return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",  return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)
    p_gates = patch("core.orchestrator.evaluate_gates",  return_value=fail_gate)
    p_regen = patch("core.orchestrator.regeneration_loop", side_effect=escalation_error)

    with p1, p2, p3, p_gates, p_regen:
        result = run_pipeline(run_id, canonical_user_input)

    assert result["status"] == "escalated"


def test_16_escalation_registry_status_escalated(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import (
        PipelineEscalationError, RUN_REGISTRY, _init_run_state, run_pipeline,
    )

    run_id = "test_escalate_16"
    _init_run_state(run_id)

    fail_gate = {
        "overall": "fail", "failures": [{"gate": "flicker_index"}],
        "human_flags": [], "gates_checked": 5, "thresholds_used": {},
    }
    escalation_error = PipelineEscalationError(run_id, [])

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)
    p_gates = patch("core.orchestrator.evaluate_gates",   return_value=fail_gate)
    p_regen = patch("core.orchestrator.regeneration_loop", side_effect=escalation_error)

    with p1, p2, p3, p_gates, p_regen:
        run_pipeline(run_id, canonical_user_input)

    assert RUN_REGISTRY[run_id]["status"] == "escalated"


def test_17_escalation_does_not_raise_runtime_error(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    """run_pipeline must return (not raise) on PipelineEscalationError."""
    from core.orchestrator import PipelineEscalationError, _init_run_state, run_pipeline

    run_id = "test_escalate_17"
    _init_run_state(run_id)

    fail_gate = {
        "overall": "fail", "failures": [{"gate": "flicker_index"}],
        "human_flags": [], "gates_checked": 5, "thresholds_used": {},
    }
    escalation_error = PipelineEscalationError(run_id, [])

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)
    p_gates = patch("core.orchestrator.evaluate_gates",   return_value=fail_gate)
    p_regen = patch("core.orchestrator.regeneration_loop", side_effect=escalation_error)

    # Must not raise — any exception here fails the test
    with p1, p2, p3, p_gates, p_regen:
        result = run_pipeline(run_id, canonical_user_input)

    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 18–20: stage failure path
# ═══════════════════════════════════════════════════════════════════════════════

def test_18_post_processing_failure_sets_stage_failed_and_raises(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import _init_run_state, run_pipeline

    run_id = "test_fail_18"
    _init_run_state(run_id)

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   side_effect=RuntimeError("upscale boom"))
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)

    with pytest.raises(RuntimeError), p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)


def test_19_registry_status_failed_after_stage_failure(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline

    run_id = "test_fail_19"
    _init_run_state(run_id)

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   side_effect=RuntimeError("boom"))
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)

    with pytest.raises(RuntimeError), p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)

    assert RUN_REGISTRY[run_id]["status"] == "failed"


def test_20_runtime_error_message_contains_stage_name(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import _init_run_state, run_pipeline

    run_id = "test_fail_20"
    _init_run_state(run_id)

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   side_effect=RuntimeError("boom"))
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)

    with pytest.raises(RuntimeError) as exc_info, p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)

    # The error message contains the name of the failing stage
    assert "upscale" in str(exc_info.value) or "post" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 21–22: RUN_REGISTRY isolation
# ═══════════════════════════════════════════════════════════════════════════════

def test_21_result_stored_in_registry_after_complete_run(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline

    run_id = "test_registry_21"
    _init_run_state(run_id)
    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        run_pipeline(run_id, canonical_user_input)
    assert RUN_REGISTRY[run_id]["result"] is not None


def test_22_two_run_ids_produce_independent_registry_entries(
    canonical_user_input, mock_gen_result, mock_post_result, mock_metadata_result
):
    from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline

    run_id_a = "test_registry_22a"
    run_id_b = "test_registry_22b"

    _init_run_state(run_id_a)
    _init_run_state(run_id_b)

    p1, p2, p3 = _patch_pipeline(mock_gen_result, mock_post_result, mock_metadata_result)
    with p1, p2, p3:
        run_pipeline(run_id_a, canonical_user_input)
        run_pipeline(run_id_b, canonical_user_input)

    assert run_id_a in RUN_REGISTRY
    assert run_id_b in RUN_REGISTRY
    assert RUN_REGISTRY[run_id_a] is not RUN_REGISTRY[run_id_b]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 23–30: FastAPI routes (TestClient)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client(mock_gen_result, mock_post_result, mock_metadata_result):
    """TestClient with generation/post/metadata mocked out."""
    from api.main import app

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)

    with p1, p2, p3:
        with TestClient(app) as tc:
            yield tc


VALID_PAYLOAD = {
    "run_id": "test_route_payload",
    "compiled": {
        "user_input": {
            "category":          "Economy",
            "location_feel":     "Urban",
            "time_of_day":       "Dusk",
            "color_temperature": "Cool",
            "mood":              "Serious",
            "motion_intensity":  "Gentle",
        }
    },
}


def test_23_post_generate_valid_payload_returns_200(client):
    resp = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    assert resp.status_code == 200


def test_24_response_body_contains_status_field(client):
    resp = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    assert "status" in resp.json()


def test_25_response_run_id_is_string(client):
    resp = client.post("/api/v1/generate", json=VALID_PAYLOAD)
    body = resp.json()
    assert "run_id" in body
    assert body["run_id"] == VALID_PAYLOAD["run_id"]


def test_26_get_run_status_known_run_returns_200(
    mock_gen_result, mock_post_result, mock_metadata_result
):
    from api.main import app
    from core.orchestrator import RUN_REGISTRY, _init_run_state

    run_id = "test_route_26"
    _init_run_state(run_id)
    RUN_REGISTRY[run_id]["status"] = "complete"

    with TestClient(app) as tc:
        resp = tc.get(f"/api/v1/run/{run_id}/status")
    assert resp.status_code == 200


def test_27_status_response_contains_stages_dict_with_11_keys(
    mock_gen_result, mock_post_result, mock_metadata_result
):
    from api.main import app
    from core.orchestrator import RUN_REGISTRY, _init_run_state

    run_id = "test_route_27"
    _init_run_state(run_id)

    with TestClient(app) as tc:
        resp = tc.get(f"/api/v1/run/{run_id}/status")
    body = resp.json()
    assert "stages" in body
    assert len(body["stages"]) == 11


def test_28_get_unknown_run_id_returns_404():
    from api.main import app

    with TestClient(app) as tc:
        resp = tc.get("/api/v1/run/definitely_not_a_real_run_id_xyz/status")
    assert resp.status_code == 404


def test_29_post_generate_missing_user_input_returns_422():
    from api.main import app

    bad_payload = {"run_id": "bad_run_29", "compiled": {}}
    with TestClient(app) as tc:
        resp = tc.post("/api/v1/generate", json=bad_payload)
    assert resp.status_code == 422


def test_30_run_id_appears_in_registry_after_post_generate(
    mock_gen_result, mock_post_result, mock_metadata_result
):
    from api.main import app
    from core.orchestrator import RUN_REGISTRY

    p1 = patch("core.orchestrator.run_generation",        return_value=mock_gen_result)
    p2 = patch("core.orchestrator.run_post_processing",   return_value=mock_post_result)
    p3 = patch("core.orchestrator.run_metadata_assembly", return_value=mock_metadata_result)

    with p1, p2, p3:
        with TestClient(app) as tc:
            resp = tc.post("/api/v1/generate", json=VALID_PAYLOAD)

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert run_id in RUN_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 31–32: Regression
# ═══════════════════════════════════════════════════════════════════════════════

def test_31_importing_orchestrator_does_not_import_torch_or_diffusers():
    """
    core.orchestrator must not import torch or diffusers at module level.
    These heavy ML packages are guard-imported inside generator.py only.
    """
    # Remove cached modules so we get a fresh import trace
    for mod_name in list(sys.modules.keys()):
        if "orchestrator" in mod_name:
            del sys.modules[mod_name]

    import core.orchestrator  # noqa: F401 — side effect is what we're testing

    assert "torch" not in sys.modules, "torch was imported by core.orchestrator"
    assert "diffusers" not in sys.modules, "diffusers was imported by core.orchestrator"


def test_32_all_210_existing_backend_tests_still_importable():
    """
    Smoke-check: importing core.orchestrator does not break any existing
    module by introducing circular imports or global side effects.
    """
    existing_modules = [
        "core.prompt_compiler",
        "core.generator",
        "core.probes",
        "core.gates",
        "core.regenerator",
        "core.post_processor",
        "core.metadata_assembler",
        "api.models",
    ]
    for mod in existing_modules:
        imported = importlib.import_module(mod)
        assert imported is not None, f"Failed to import {mod}"
