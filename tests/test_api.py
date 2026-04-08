"""
tests/test_api.py
─────────────────
FastAPI layer contract tests — 23 cases covering health, inputs,
compile, generate, status, and bundle endpoints.

Uses FastAPI's TestClient (backed by httpx/starlette).
"""

import re

import pytest
from starlette.testclient import TestClient

from api.main import app

client = TestClient(app)

# ── Canonical input used across multiple tests ─────────────────────────────
CANONICAL = {
    "category": "Economy",
    "location_feel": "Urban",
    "time_of_day": "Dusk",
    "color_temperature": "Cool",
    "mood": "Serious",
    "motion_intensity": "Gentle",
}


# ── Health ─────────────────────────────────────────────────────────────────

def test_health_200():
    """1. GET /api/v1/health → 200"""
    r = client.get("/api/v1/health")
    assert r.status_code == 200


def test_health_status_ok():
    """1b. status == "ok" """
    r = client.get("/api/v1/health")
    assert r.json()["status"] == "ok"


def test_health_module_bg_video():
    """1c. module == "bg_video" """
    r = client.get("/api/v1/health")
    assert r.json()["module"] == "bg_video"


# ── Inputs ─────────────────────────────────────────────────────────────────

def test_inputs_200():
    """2. GET /api/v1/inputs → 200, response contains all 6 field keys"""
    r = client.get("/api/v1/inputs")
    assert r.status_code == 200
    body = r.json()
    for key in ("category", "location_feel", "time_of_day",
                "color_temperature", "mood", "motion_intensity"):
        assert key in body


def test_inputs_category_contains_economy():
    """3. category list contains "Economy" """
    r = client.get("/api/v1/inputs")
    assert "Economy" in r.json()["category"]


def test_inputs_motion_intensity_contains_gentle():
    """4. motion_intensity list contains "Gentle" """
    r = client.get("/api/v1/inputs")
    assert "Gentle" in r.json()["motion_intensity"]


# ── Compile ────────────────────────────────────────────────────────────────

def test_compile_canonical_200():
    """5. POST /api/v1/compile with canonical input → 200"""
    r = client.post("/api/v1/compile", json=CANONICAL)
    assert r.status_code == 200


def test_compile_canonical_selected_lut():
    """6. selected_lut == "cool_authority" """
    r = client.post("/api/v1/compile", json=CANONICAL)
    assert r.json()["selected_lut"] == "cool_authority"


def test_compile_canonical_lower_third_style():
    """7. lower_third_style == "minimal_dark_bar" """
    r = client.post("/api/v1/compile", json=CANONICAL)
    assert r.json()["lower_third_style"] == "minimal_dark_bar"


def test_compile_canonical_positive_hash_format():
    """8. positive_hash is a 64-char hex string"""
    r = client.post("/api/v1/compile", json=CANONICAL)
    h = r.json()["positive_hash"]
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_compile_missing_field_422():
    """9. POST /api/v1/compile with missing field → 422"""
    bad = {k: v for k, v in CANONICAL.items() if k != "category"}
    r = client.post("/api/v1/compile", json=bad)
    assert r.status_code == 422


def test_compile_invalid_field_value_422():
    """10. POST /api/v1/compile with invalid field value → 422
    (category: "Weather" is not a valid option)"""
    bad = {**CANONICAL, "category": "Weather"}
    r = client.post("/api/v1/compile", json=bad)
    assert r.status_code == 422


# ── Generate ───────────────────────────────────────────────────────────────

def test_generate_canonical_200():
    """11. POST /api/v1/generate with canonical input + dry_run:false → 200"""
    r = client.post("/api/v1/generate",
                    json={"editorial_input": CANONICAL, "dry_run": False})
    assert r.status_code == 200


def test_generate_status_queued():
    """12. status == "queued" """
    r = client.post("/api/v1/generate",
                    json={"editorial_input": CANONICAL, "dry_run": False})
    assert r.json()["status"] == "queued"


def test_generate_stages_count():
    """13. stages list has exactly 11 items"""
    r = client.post("/api/v1/generate",
                    json={"editorial_input": CANONICAL, "dry_run": False})
    assert len(r.json()["stages"]) == 11


def test_generate_input_hash_short_length():
    """14. compiled.input_hash_short is 6 chars"""
    r = client.post("/api/v1/generate",
                    json={"editorial_input": CANONICAL, "dry_run": False})
    assert len(r.json()["compiled"]["input_hash_short"]) == 6


def test_generate_all_stages_pending():
    """15. all stages have status "pending" """
    r = client.post("/api/v1/generate",
                    json={"editorial_input": CANONICAL, "dry_run": False})
    statuses = [s["status"] for s in r.json()["stages"]]
    assert all(s == "pending" for s in statuses)


# ── Status ─────────────────────────────────────────────────────────────────

def test_status_200():
    """16. GET /api/v1/status/bg_001_b2e7f3 → 200, status == "running" """
    r = client.get("/api/v1/status/bg_001_b2e7f3")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_status_first_stage_complete():
    """17. first stage status == "complete" """
    r = client.get("/api/v1/status/bg_001_b2e7f3")
    assert r.json()["stages"][0]["status"] == "complete"


# ── Bundle ─────────────────────────────────────────────────────────────────

def test_bundle_200():
    """18. GET /api/v1/bundle/bg_001_b2e7f3 → 200, status == "complete" """
    r = client.get("/api/v1/bundle/bg_001_b2e7f3")
    assert r.status_code == 200
    assert r.json()["status"] == "complete"


def test_bundle_quality_gates_overall_pass():
    """19. quality_gates.overall == "pass" """
    r = client.get("/api/v1/bundle/bg_001_b2e7f3")
    assert r.json()["quality_gates"]["overall"] == "pass"


def test_bundle_files_final_contains_run_id():
    """20. files.final contains run_id"""
    run_id = "bg_001_b2e7f3"
    r = client.get(f"/api/v1/bundle/{run_id}")
    assert run_id in r.json()["files"]["final"]


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_generate_invalid_editorial_input_422():
    """21. POST /api/v1/generate with invalid editorial_input field → 422"""
    bad_input = {**CANONICAL, "mood": "Anxious"}  # not a valid mood
    r = client.post("/api/v1/generate",
                    json={"editorial_input": bad_input, "dry_run": False})
    assert r.status_code == 422


# ── Regression guards ──────────────────────────────────────────────────────
# Tests 22 and 23 are covered by running the full test suite with:
#   pytest tests/test_prompt_compiler.py
#   pytest tests/test_constants.py
# They are not re-implemented here to avoid duplication, but the suite
# must be run as a group to satisfy the regression requirement.
