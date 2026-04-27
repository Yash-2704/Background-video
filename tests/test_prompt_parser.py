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


# ══════════════════════════════════════════════════════════════════════════════
# NEW TESTS — prompt enrichment (enrich_prompt_for_wan + generator integration)
# ══════════════════════════════════════════════════════════════════════════════

# ── TEST A — enrich_prompt_for_wan returns an expanded string ─────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_enrich_prompt_for_wan_returns_expanded_string():
    """Mock Groq returns a long response; assert result is a string longer than 50 chars."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)

    expanded = (
        "A dense financial district at dusk, glass and steel towers reflecting "
        "amber and gold light. The camera performs a slow dolly forward at street level. "
        "Warm directional light casts long shadows across empty pavement. "
        "Cinematic depth of field with shallow focus on foreground geometry. "
        "Muted desaturated tones with cool blue shadow detail. "
        "A gentle haze softens distant architecture. "
        "Stable textures, smooth motion, broadcast quality, no artifacts, no text, "
        "no people, no faces."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(expanded)

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client

    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        result = pp.enrich_prompt_for_wan(
            "financial district, urban, dusk, muted tones",
            "slow lateral drift, gentle parallax",
        )

    assert isinstance(result, str)
    assert len(result) > 50


# ── TEST B — enrich_prompt_for_wan raises on Groq failure ────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-testing"})
def test_enrich_prompt_for_wan_raises_on_groq_failure():
    """Groq raises Exception; assert enrich_prompt_for_wan propagates it (does not swallow)."""
    import importlib
    import core.prompt_parser as pp
    importlib.reload(pp)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client

    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        with pytest.raises(Exception):
            pp.enrich_prompt_for_wan("some short prompt", "some motion")


# ── TEST C — generator falls back to original prompt on enrichment failure ────

try:
    from core.generator import _ML_AVAILABLE as _GEN_ML_AVAILABLE
except ImportError:
    _GEN_ML_AVAILABLE = False


@pytest.mark.skipif(not _GEN_ML_AVAILABLE, reason="requires CUDA GPU and ML packages")
def test_generator_falls_back_on_enrichment_failure(tmp_path):
    """
    enrich_prompt_for_wan raises; run_generation must not raise and must store
    the original compiled positive prompt as enriched_positive_prompt in the log.
    """
    import numpy as np
    from core.generator import GENERATION_CONSTANTS, run_generation

    _COMPILED = {
        "positive":          "financial district, urban, dusk",
        "motion":            "slow lateral drift",
        "negative":          "text, people, faces",
        "positive_hash":     "abc123" * 10,
        "motion_hash":       "bcd234" * 10,
        "negative_hash":     "cde345" * 10,
        "input_hash_short":  "a1b2c3",
        "selected_lut":      "neutral",
        "lower_third_style": "standard_bar",
        "compiler_version":  "1.0.0",
    }

    _join_result = {
        "raw_loop_path":        tmp_path / "loop.mp4",
        "seam_frames_raw":      [145, 290],
        "seam_frames_playable": [138, 269],
        "total_frames_raw":     435,
        "playable_frames":      406,
    }

    _blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    with patch("core.prompt_parser.enrich_prompt_for_wan", side_effect=Exception("API error")):
        with patch("core.generator.generate_clip", side_effect=lambda **kw: kw["output_path"]):
            with patch("core.generator._extract_last_frame", return_value=_blank_frame):
                with patch("core.generator.crossfade_join", return_value=_join_result):
                    result = run_generation(
                        _COMPILED, "run_enrich_fallback", tmp_path, seed=42, dry_run=False
                    )

    assert result["generation_log"]["enriched_positive_prompt"] == _COMPILED["positive"]


# ── TEST D — enrichment skipped when enrich_prompts=false ────────────────────

def test_enrichment_skipped_when_enrich_prompts_false(tmp_path):
    """
    GENERATION_CONSTANTS["enrich_prompts"]=False; assert enrich_prompt_for_wan
    is never called (dry_run=True also gates it, verifying the config check path).
    """
    from unittest.mock import MagicMock, patch
    from core.generator import GENERATION_CONSTANTS, run_generation

    mock_enrich = MagicMock()
    patched_gc = {**GENERATION_CONSTANTS, "enrich_prompts": False}

    _COMPILED = {
        "positive":          "financial district, urban, dusk",
        "motion":            "slow lateral drift",
        "negative":          "text, people, faces",
        "positive_hash":     "abc123" * 10,
        "motion_hash":       "bcd234" * 10,
        "negative_hash":     "cde345" * 10,
        "input_hash_short":  "a1b2c3",
        "selected_lut":      "neutral",
        "lower_third_style": "standard_bar",
        "compiler_version":  "1.0.0",
    }

    with patch("core.generator.GENERATION_CONSTANTS", patched_gc):
        with patch("core.prompt_parser.enrich_prompt_for_wan", mock_enrich):
            run_generation(_COMPILED, "run_skip_enrich", tmp_path, seed=42, dry_run=True)

    assert mock_enrich.call_count == 0


# ── TEST E — _WAN_ENRICHMENT_SYSTEM_PROMPT contains all required terms ────────

def test_wan_enrichment_system_prompt_contains_required_terms():
    """_WAN_ENRICHMENT_SYSTEM_PROMPT must be non-empty and contain key structural terms."""
    from core.prompt_parser import _WAN_ENRICHMENT_SYSTEM_PROMPT

    assert isinstance(_WAN_ENRICHMENT_SYSTEM_PROMPT, str)
    assert len(_WAN_ENRICHMENT_SYSTEM_PROMPT) > 0

    required_terms = ["80", "camera", "broadcast", "no people", "no faces", "stable textures"]
    for term in required_terms:
        assert term in _WAN_ENRICHMENT_SYSTEM_PROMPT, (
            f"_WAN_ENRICHMENT_SYSTEM_PROMPT is missing required term: {term!r}"
        )
