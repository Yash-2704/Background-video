"""
tests/test_upscale_exit_path.py
────────────────────────────────
6 tests for the upscale-only exit path added in Prompt 1 of 2.

All tests run without a GPU (upscale_clip and pipeline stages are mocked).
No test calls run_generation() directly or writes more than 30 frames.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline
from api.models import RunStatusResponse
from api.routes.bundle import router as bundle_router


# ── Shared helpers ─────────────────────────────────────────────────────────────

_RUN_ID = "test01"

_USER_INPUT = {
    "mode": "i2v",
    "selected_lut": "neutral",
    "lower_third_style": "minimal_dark_bar",
}

_GEN_RESULT = {
    "raw_loop_path": "/tmp/fake_raw_loop.mp4",
    "seed": 42,
    "seam_frames_raw": [145, 290],
    "seam_frames_playable": [138, 269],
}

_DECODE_PROBE = {"mean_luminance": 0.46, "frame_count": 145}

_POST_RESULT = {
    "upscaled_path": "/tmp/fake_1080p.mp4",
    "final_path": "/tmp/fake_final.mp4",
    "preview_path": "/tmp/fake_preview.gif",
    "mask_paths": [],
    "lut_paths": [],
}

_METADATA_RESULT = {
    "metadata_path": "/tmp/fake_metadata.json",
}


def _setup_registry(run_id: str) -> None:
    """Populate RUN_REGISTRY for a run_id before calling run_pipeline."""
    _init_run_state(run_id)


# ── test_01 ────────────────────────────────────────────────────────────────────

def test_01_upscale_clip_called_once_when_verify_raw_only(tmp_path):
    """upscale_clip is called exactly once when verify_raw_only=True."""
    run_id = "t01abc"
    _setup_registry(run_id)

    with (
        patch.dict("core.orchestrator.GENERATION_CONSTANTS", {"verify_raw_only": True}),
        patch("core.orchestrator.OUTPUT_DIR", tmp_path),
        patch("core.orchestrator.run_generation", return_value=_GEN_RESULT),
        patch("core.orchestrator.run_decode_probe", return_value=_DECODE_PROBE),
        patch("core.orchestrator.upscale_clip") as mock_upscale,
    ):
        run_pipeline(run_id, _USER_INPUT)

    assert mock_upscale.call_count == 1


# ── test_02 ────────────────────────────────────────────────────────────────────

def test_02_result_contains_upscaled_loop_path_key(tmp_path):
    """result dict has key 'upscaled_loop_path' when verify_raw_only=True."""
    run_id = "t02abc"
    _setup_registry(run_id)

    with (
        patch.dict("core.orchestrator.GENERATION_CONSTANTS", {"verify_raw_only": True}),
        patch("core.orchestrator.OUTPUT_DIR", tmp_path),
        patch("core.orchestrator.run_generation", return_value=_GEN_RESULT),
        patch("core.orchestrator.run_decode_probe", return_value=_DECODE_PROBE),
        patch("core.orchestrator.upscale_clip"),
    ):
        result = run_pipeline(run_id, _USER_INPUT)

    assert "upscaled_loop_path" in result


# ── test_03 ────────────────────────────────────────────────────────────────────

def test_03_upscaled_loop_path_ends_with_1080p(tmp_path):
    """result['upscaled_loop_path'] ends with '_1080p.mp4' when verify_raw_only=True."""
    run_id = "t03abc"
    _setup_registry(run_id)

    with (
        patch.dict("core.orchestrator.GENERATION_CONSTANTS", {"verify_raw_only": True}),
        patch("core.orchestrator.OUTPUT_DIR", tmp_path),
        patch("core.orchestrator.run_generation", return_value=_GEN_RESULT),
        patch("core.orchestrator.run_decode_probe", return_value=_DECODE_PROBE),
        patch("core.orchestrator.upscale_clip"),
    ):
        result = run_pipeline(run_id, _USER_INPUT)

    assert result["upscaled_loop_path"].endswith("_1080p.mp4")


# ── test_04 ────────────────────────────────────────────────────────────────────

def test_04_upscale_clip_not_called_when_verify_raw_only_false(tmp_path):
    """upscale_clip is NOT called when verify_raw_only=False."""
    run_id = "t04abc"
    _setup_registry(run_id)

    with (
        patch.dict("core.orchestrator.GENERATION_CONSTANTS", {"verify_raw_only": False}),
        patch("core.orchestrator.OUTPUT_DIR", tmp_path),
        patch("core.orchestrator.run_generation", return_value=_GEN_RESULT),
        patch("core.orchestrator.run_decode_probe", return_value=_DECODE_PROBE),
        patch("core.orchestrator.run_post_processing", return_value=_POST_RESULT),
        patch("core.orchestrator.run_metadata_assembly", return_value=_METADATA_RESULT),
        patch("core.orchestrator.upscale_clip") as mock_upscale,
    ):
        run_pipeline(run_id, _USER_INPUT)

    assert mock_upscale.call_count == 0


# ── test_05 ────────────────────────────────────────────────────────────────────

def test_05_run_status_response_accepts_upscaled_loop_path():
    """RunStatusResponse can be instantiated with upscaled_loop_path without error."""
    resp = RunStatusResponse(
        run_id="t05abc",
        status="complete",
        stages={s: "complete" for s in ["prompt_compilation", "generation",
                                         "probe_decode", "probe_temporal",
                                         "gate_evaluation", "upscale",
                                         "mask_generation", "lut_grading",
                                         "composite", "preview_export",
                                         "metadata_assembly"]},
        upscaled_loop_path="output/t05abc/raw/t05abc_1080p.mp4",
    )
    assert resp.upscaled_loop_path == "output/t05abc/raw/t05abc_1080p.mp4"


# ── test_06 ────────────────────────────────────────────────────────────────────

def test_06_media_endpoint_serves_1080p_and_rejects_unknown(tmp_path):
    """
    GET /api/v1/media/{clip_id}/{clip_id}_1080p.mp4 → 200 when file exists.
    GET /api/v1/media/{clip_id}/unknown_file.mp4    → 400 (whitelist guard).
    """
    clip_id = "t06abc"

    # Create the expected file in a raw/ subdirectory under tmp_path/output/
    raw_dir = tmp_path / "output" / clip_id / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / f"{clip_id}_1080p.mp4").write_bytes(b"\x00" * 32)

    app = FastAPI()
    app.include_router(bundle_router, prefix="/api/v1")
    client = TestClient(app, raise_server_exceptions=False)

    with patch("api.routes.bundle._PROJECT_ROOT", tmp_path):
        # Valid 1080p file — should be 200
        resp_ok = client.get(f"/api/v1/media/{clip_id}/{clip_id}_1080p.mp4")
        assert resp_ok.status_code == 200

        # Unknown filename — should be 400
        resp_bad = client.get(f"/api/v1/media/{clip_id}/unknown_file.mp4")
        assert resp_bad.status_code == 400
