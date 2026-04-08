import json
import re
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import core.orchestrator as orchestrator
from core.regenerator import PipelineEscalationError


@pytest.fixture
def canonical_user_input():
    return {
        "category": "Economy",
        "location_feel": "Urban",
        "time_of_day": "Dusk",
        "color_temperature": "Cool",
        "mood": "Serious",
        "motion_intensity": "Gentle",
    }


@pytest.fixture
def synthetic_raw_loop_path(tmp_path):
    raw_path = tmp_path / "raw_loop.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(raw_path), fourcc, 30, (720, 480))
    for _ in range(30):
        out.write(np.zeros((480, 720, 3), dtype=np.uint8))
    out.release()
    return raw_path


@pytest.fixture
def mock_gen_result(synthetic_raw_loop_path):
    return {
        "run_id": "e2e_test_001",
        "status": "complete",
        "raw_loop_path": str(synthetic_raw_loop_path),
        "seed": 42819,
        "attempts_used": 1,
        "seam_frames_raw": [183, 366],
        "seam_frames_playable": [169, 338],
        "generation_log": {},
        "failure_log": [],
    }


@pytest.fixture
def mock_post_result(tmp_path):
    final_path = tmp_path / "final" / "e2e_test_001.mp4"
    upscaled_path = tmp_path / "raw" / "e2e_test_001_1080p.mp4"
    preview_path = tmp_path / "final" / "e2e_test_001_preview.gif"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.touch()
    upscaled_path.touch()
    preview_path.touch()
    return {
        "upscaled": str(upscaled_path),
        "masks": {},
        "graded_variants": {},
        "final": str(final_path),
        "preview_gif": str(preview_path),
        "selected_lut": "cool_authority",
        "luts_generated": [],
        "risks": {},
    }


@pytest.fixture(autouse=True)
def clear_run_registry():
    orchestrator.RUN_REGISTRY.clear()
    yield
    orchestrator.RUN_REGISTRY.clear()


@pytest.fixture
def successful_result(tmp_path, canonical_user_input, mock_gen_result, mock_post_result, monkeypatch):
    run_id = "e2e_test_001"
    monkeypatch.setattr(orchestrator, "OUTPUT_DIR", tmp_path)
    orchestrator._init_run_state(run_id)
    with (
        patch.object(orchestrator, "run_generation", return_value=mock_gen_result),
        patch.object(orchestrator, "run_post_processing", return_value=mock_post_result),
    ):
        result = orchestrator.run_pipeline(run_id, canonical_user_input)
    return result


def test_pipeline_returns_dict(successful_result):
    assert isinstance(successful_result, dict)


def test_pipeline_status_complete(successful_result):
    assert successful_result["status"] == "complete"


def test_pipeline_run_id_matches(successful_result):
    assert successful_result["run_id"] == "e2e_test_001"


def test_pipeline_all_stage_keys_complete(successful_result):
    assert set(successful_result["stages"].keys()) == set(orchestrator.STAGE_KEYS)
    assert all(v == "complete" for v in successful_result["stages"].values())


def test_pipeline_seed_threads_through(successful_result):
    assert successful_result["seed"] == 42819


def test_pipeline_selected_lut_from_compiler(successful_result):
    assert successful_result["selected_lut"] == "cool_authority"


def test_pipeline_lower_third_from_compiler(successful_result):
    assert successful_result["lower_third_style"] == "minimal_dark_bar"


def test_pipeline_gate_result_is_valid_terminal(successful_result):
    assert successful_result["gate_result"]["overall"] in ("pass", "human_review")


def test_pipeline_run_registry_status_complete(successful_result):
    _ = successful_result
    assert orchestrator.RUN_REGISTRY["e2e_test_001"]["status"] == "complete"


def test_metadata_path_exists_and_valid_json(successful_result):
    metadata_path = Path(successful_result["metadata_path"])
    assert metadata_path.exists()
    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


def test_metadata_contains_clip_id(successful_result):
    metadata_path = Path(successful_result["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["clip_id"] == "e2e_test_001"


def test_metadata_contains_hash_short(successful_result):
    metadata_path = Path(successful_result["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hash_short = metadata["prompt_provenance"]["input_hash_short"]
    assert re.fullmatch(r"[0-9a-f]{6}", hash_short)


def test_edit_manifest_exists_and_valid_json(successful_result):
    run_id = successful_result["run_id"]
    out_dir = Path(successful_result["metadata_path"]).parent
    edit_manifest_path = out_dir / f"{run_id}_edit_manifest.json"
    assert edit_manifest_path.exists()
    parsed = json.loads(edit_manifest_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


def test_integration_contract_exists_and_audience_keys_present(successful_result):
    run_id = successful_result["run_id"]
    out_dir = Path(successful_result["metadata_path"]).parent
    contract_path = out_dir / f"{run_id}_integration_contract.json"
    assert contract_path.exists()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert "for_human_editor" in contract
    assert "for_downstream_modules" in contract
    assert "for_compositor_process" in contract


def test_generation_log_exists(successful_result):
    run_id = successful_result["run_id"]
    out_dir = Path(successful_result["metadata_path"]).parent
    generation_log_path = out_dir / f"{run_id}_generation_log.json"
    assert generation_log_path.exists()


def test_post_processing_failure_sets_failed_stage_and_raises(
    tmp_path, canonical_user_input, mock_gen_result, monkeypatch
):
    run_id = "e2e_test_fail_001"
    monkeypatch.setattr(orchestrator, "OUTPUT_DIR", tmp_path)
    orchestrator._init_run_state(run_id)
    with (
        patch.object(orchestrator, "run_generation", return_value=mock_gen_result),
        patch.object(orchestrator, "run_post_processing", side_effect=RuntimeError("post failed")),
    ):
        with pytest.raises(RuntimeError):
            orchestrator.run_pipeline(run_id, canonical_user_input)
    post_stages = ("upscale", "mask_generation", "lut_grading", "composite", "preview_export")
    assert any(orchestrator.RUN_REGISTRY[run_id]["stages"][k] == "failed" for k in post_stages)


def test_post_processing_failure_sets_registry_failed(
    tmp_path, canonical_user_input, mock_gen_result, monkeypatch
):
    run_id = "e2e_test_fail_002"
    monkeypatch.setattr(orchestrator, "OUTPUT_DIR", tmp_path)
    orchestrator._init_run_state(run_id)
    with (
        patch.object(orchestrator, "run_generation", return_value=mock_gen_result),
        patch.object(orchestrator, "run_post_processing", side_effect=RuntimeError("post failed")),
    ):
        with pytest.raises(RuntimeError):
            orchestrator.run_pipeline(run_id, canonical_user_input)
    assert orchestrator.RUN_REGISTRY[run_id]["status"] == "failed"


def test_escalation_returns_escalated_dict(
    tmp_path, canonical_user_input, mock_gen_result, monkeypatch
):
    run_id = "e2e_test_escalate_001"
    monkeypatch.setattr(orchestrator, "OUTPUT_DIR", tmp_path)
    orchestrator._init_run_state(run_id)
    escalation_error = PipelineEscalationError(
        run_id=run_id,
        failure_log=[{"attempt": 1, "gate_result": "fail"}],
    )
    with (
        patch.object(orchestrator, "run_generation", return_value=mock_gen_result),
        patch.object(orchestrator, "evaluate_gates", return_value={"overall": "fail"}),
        patch.object(orchestrator, "regeneration_loop", side_effect=escalation_error),
    ):
        result = orchestrator.run_pipeline(run_id, canonical_user_input)
    assert result["status"] == "escalated"


def test_escalation_sets_registry_escalated(
    tmp_path, canonical_user_input, mock_gen_result, monkeypatch
):
    run_id = "e2e_test_escalate_002"
    monkeypatch.setattr(orchestrator, "OUTPUT_DIR", tmp_path)
    orchestrator._init_run_state(run_id)
    escalation_error = PipelineEscalationError(
        run_id=run_id,
        failure_log=[{"attempt": 1, "gate_result": "fail"}],
    )
    with (
        patch.object(orchestrator, "run_generation", return_value=mock_gen_result),
        patch.object(orchestrator, "evaluate_gates", return_value={"overall": "fail"}),
        patch.object(orchestrator, "regeneration_loop", side_effect=escalation_error),
    ):
        _ = orchestrator.run_pipeline(run_id, canonical_user_input)
    assert orchestrator.RUN_REGISTRY[run_id]["status"] == "escalated"
