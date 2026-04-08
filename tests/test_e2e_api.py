import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.orchestrator import RUN_REGISTRY, STAGE_KEYS


@pytest.fixture(autouse=True)
def clear_registry():
    RUN_REGISTRY.clear()
    yield
    RUN_REGISTRY.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_pipeline_result():
    return {
        "run_id": "api_test_001",
        "status": "complete",
        "raw_loop_path": "raw/api_test_001_raw_loop.mp4",
        "seed": 55123,
        "seam_frames_raw": [183, 366],
        "seam_frames_playable": [169, 338],
        "gate_result": {"overall": "pass"},
        "selected_lut": "cool_authority",
        "lower_third_style": "minimal_dark_bar",
        "metadata_path": "output/api_test_001/metadata.json",
        "stages": {s: "complete" for s in STAGE_KEYS},
    }


@pytest.fixture
def compile_payload():
    return {
        "category": "Economy",
        "location_feel": "Urban",
        "time_of_day": "Dusk",
        "color_temperature": "Cool",
        "mood": "Serious",
        "motion_intensity": "Gentle",
    }


@pytest.fixture
def compile_response(client, compile_payload):
    response = client.post("/api/v1/compile", json=compile_payload)
    assert response.status_code == 200
    return response.json()


def _generate_payload(compiled_response):
    return {
        "run_id": compiled_response["input_hash_short"],
        "compiled": compiled_response,
    }


def test_compile_returns_200(client, compile_payload):
    response = client.post("/api/v1/compile", json=compile_payload)
    assert response.status_code == 200


def test_compile_contains_input_hash_short(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    assert re.fullmatch(r"[0-9a-f]{6}", body["input_hash_short"])


def test_compile_contains_selected_lut(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    assert body["selected_lut"] == "cool_authority"


def test_compile_contains_lower_third_style(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    assert body["lower_third_style"] == "minimal_dark_bar"


def test_compile_contains_hashes(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    assert isinstance(body["positive_hash"], str) and body["positive_hash"]
    assert isinstance(body["motion_hash"], str) and body["motion_hash"]
    assert isinstance(body["negative_hash"], str) and body["negative_hash"]


def test_compile_contains_user_input_with_all_fields(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    assert "user_input" in body
    assert set(body["user_input"].keys()) == set(compile_payload.keys())


def test_generate_with_valid_payload_returns_200(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200


def test_generate_response_contains_status(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert "status" in body


def test_generate_response_run_id_matches_payload(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert body["run_id"] == payload["run_id"]


def test_generate_response_seed_is_int(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert isinstance(body["seed"], int)


def test_generate_response_gate_result_overall_present(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert "gate_result" in body and "overall" in body["gate_result"]


def test_generate_response_stages_contains_all_keys(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert set(body["stages"].keys()) == set(STAGE_KEYS)


def test_status_for_registered_run_returns_200(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        client.post("/api/v1/generate", json=payload)
    status_response = client.get(f"/api/v1/run/{payload['run_id']}/status")
    assert status_response.status_code == 200


def test_status_contains_stages_with_11_keys(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        client.post("/api/v1/generate", json=payload)
    body = client.get(f"/api/v1/run/{payload['run_id']}/status").json()
    assert len(body["stages"]) == 11


def test_status_contains_valid_state_value(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        client.post("/api/v1/generate", json=payload)
    body = client.get(f"/api/v1/run/{payload['run_id']}/status").json()
    assert body["status"] in {"queued", "running", "complete", "failed", "escalated"}


def test_status_nonexistent_returns_404(client):
    response = client.get("/api/v1/run/nonexistent_id/status")
    assert response.status_code == 404


def test_bundle_non_whitelisted_filename_returns_400(client):
    response = client.get("/api/v1/bundle/abc123/not_allowed.json")
    assert response.status_code == 400


def test_bundle_whitelisted_filename_nonexistent_clip_returns_404(client):
    clip_id = "does_not_exist"
    response = client.get(f"/api/v1/bundle/{clip_id}/{clip_id}_metadata.json")
    assert response.status_code == 404


def test_compile_contract_keys_present_subset(client, compile_payload):
    body = client.post("/api/v1/compile", json=compile_payload).json()
    expected = {
        "input_hash_short",
        "selected_lut",
        "lower_third_style",
        "positive_hash",
        "motion_hash",
        "negative_hash",
        "user_input",
    }
    assert expected.issubset(set(body.keys()))


def test_generate_stages_is_dict(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        body = client.post("/api/v1/generate", json=payload).json()
    assert isinstance(body["stages"], dict)


def test_status_stages_dict_has_string_values(client, compile_response, mock_pipeline_result):
    payload = _generate_payload(compile_response)
    with patch("api.routes.generate.run_pipeline", return_value=mock_pipeline_result):
        client.post("/api/v1/generate", json=payload)
    body = client.get(f"/api/v1/run/{payload['run_id']}/status").json()
    assert isinstance(body["stages"], dict)
    assert all(isinstance(v, str) for v in body["stages"].values())


def test_generate_missing_compiled_user_input_returns_422(client, compile_response):
    payload = {
        "run_id": compile_response["input_hash_short"],
        "compiled": {k: v for k, v in compile_response.items() if k != "user_input"},
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422
