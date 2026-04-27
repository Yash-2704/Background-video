"""
tests/test_prompt_parser.py
────────────────────────────
Tests for core/prompt_parser.py and POST /api/v1/parse-prompt.

All tests mock the Groq client — no real API calls are made.
No test imports torch or calls run_generation().
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_groq_response(content: str):
    """Build a fake Groq completion response object."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


_VALID_PAYLOAD = {
    "category":          "Tech",
    "location_feel":     "Data",
    "time_of_day":       "Night",
    "color_temperature": "Cool",
    "mood":              "Serious",
    "motion_intensity":  "Minimal",
    "inference_notes":   "Chosen Tech and Data for server room context.",
}


# ── TEST A — parse_free_prompt returns all 6 field keys ───────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_parse_free_prompt_returns_all_six_fields():
    """Mock Groq returns valid JSON; assert all 6 field keys are present and correct."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)  # pick up the patched env var

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        json.dumps(_VALID_PAYLOAD)
    )

    # Groq is guard-imported inside parse_free_prompt via `from groq import Groq`.
    # Inject a mock groq module so the import resolves to our fake client.
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        result = pp.parse_free_prompt("server room at night, cool and serious")

    assert result["category"]          == "Tech"
    assert result["location_feel"]     == "Data"
    assert result["time_of_day"]       == "Night"
    assert result["color_temperature"] == "Cool"
    assert result["mood"]              == "Serious"
    assert result["motion_intensity"]  == "Minimal"
    assert "inference_notes" in result


# ── TEST B — invalid LLM field value falls back to default ────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_invalid_field_value_falls_back_to_default():
    """One field has an invalid value; that field must be replaced by its default."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)

    bad_payload = {**_VALID_PAYLOAD, "category": "InvalidCategory"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        json.dumps(bad_payload)
    )

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        result = pp.parse_free_prompt("some video description")

    assert result["category"] == pp._DEFAULTS["category"]
    # Valid fields are still preserved
    assert result["location_feel"] == "Data"


# ── TEST C — malformed JSON from LLM returns defaults ────────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_malformed_json_returns_defaults():
    """Non-JSON response from LLM must not raise — return _DEFAULTS for all fields."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "Sorry, I cannot help with that."
    )

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        result = pp.parse_free_prompt("some prompt")

    for field in ("category", "location_feel", "time_of_day",
                  "color_temperature", "mood", "motion_intensity"):
        assert field in result
        assert result[field] == pp._DEFAULTS[field]


# ── TEST D — markdown fence stripping works ───────────────────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_markdown_fence_stripping():
    """Response wrapped in ```json fences must be parsed correctly, not crash."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)

    fenced = "```json\n" + json.dumps(_VALID_PAYLOAD) + "\n```"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(fenced)

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        result = pp.parse_free_prompt("night time server room")

    assert result["category"]      == "Tech"
    assert result["mood"]          == "Serious"
    assert result["time_of_day"]   == "Night"


# ── Shared TestClient fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    # Set a fake key so prompt_parser module-level code doesn't block startup
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"}):
        from api.main import app
        return TestClient(app)


# ── TEST E — empty prompt raises 422 via API route ────────────────────────────

def test_empty_prompt_returns_422(client):
    """POST with empty prompt string must return HTTP 422."""
    response = client.post("/api/v1/parse-prompt", json={"prompt": ""})
    assert response.status_code == 422


# ── TEST F — parse endpoint returns all required response keys ────────────────

def test_parse_endpoint_returns_required_keys(client):
    """Mock parse_free_prompt; assert ParseResponse contains all expected keys."""
    mock_result = {
        "category":          "Economy",
        "location_feel":     "Urban",
        "time_of_day":       "Dusk",
        "color_temperature": "Warm",
        "mood":              "Tense",
        "motion_intensity":  "Dynamic",
        "inference_notes":   "City financial district at dusk matches Economy/Urban.",
    }

    with patch("core.prompt_parser.parse_free_prompt", return_value=mock_result):
        response = client.post(
            "/api/v1/parse-prompt",
            json={"prompt": "financial district at golden hour, dynamic motion"},
        )

    assert response.status_code == 200
    body = response.json()

    required_keys = {
        "category", "location_feel", "time_of_day", "color_temperature",
        "mood", "motion_intensity", "original_prompt", "inference_notes",
    }
    assert required_keys.issubset(body.keys())
    assert body["original_prompt"] == "financial district at golden hour, dynamic motion"
    assert body["category"]        == "Economy"
    assert body["inference_notes"] == "City financial district at dusk matches Economy/Urban."
