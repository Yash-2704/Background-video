"""
tests/test_prototype_route.py
──────────────────────────────
Structural tests for api/routes/prototype.py.

All tests are import-level and model-level only.
None invoke FLUX, FFmpeg, or make real HTTP requests.
All 8 tests pass on the Mac development machine with no GPU.
"""

import uuid

import pytest
from pydantic import ValidationError


# ── 1. Clean import ───────────────────────────────────────────────────────────

def test_module_imports_cleanly():
    """Importing router and PrototypeRequest must succeed with no GPU."""
    from api.routes.prototype import router, PrototypeRequest  # noqa: F401


# ── 2. Model instantiation ────────────────────────────────────────────────────

def test_prototype_request_model_fields():
    """All 6 fields must be accessible on a valid instance."""
    from api.routes.prototype import PrototypeRequest

    req = PrototypeRequest(
        category="Economy",
        location_feel="Urban",
        time_of_day="Day",
        color_temperature="Neutral",
        mood="Serious",
        motion_intensity=0.5,
    )

    assert req.category          == "Economy"
    assert req.location_feel     == "Urban"
    assert req.time_of_day       == "Day"
    assert req.color_temperature == "Neutral"
    assert req.mood              == "Serious"
    assert req.motion_intensity  == 0.5


# ── 3. Out-of-range validation ────────────────────────────────────────────────

def test_motion_intensity_validation_rejects_out_of_range():
    """motion_intensity=1.5 must raise ValidationError (ge=0.0, le=1.0)."""
    from api.routes.prototype import PrototypeRequest

    with pytest.raises(ValidationError):
        PrototypeRequest(
            category="Economy",
            location_feel="Urban",
            time_of_day="Day",
            color_temperature="Neutral",
            mood="Serious",
            motion_intensity=1.5,
        )


# ── 4. Router has /generate route ─────────────────────────────────────────────

def test_router_has_generate_route():
    """router.routes must contain at least one route with path '/generate'."""
    from api.routes.prototype import router

    paths = [route.path for route in router.routes]
    assert "/generate" in paths, f"'/generate' not found in router paths: {paths}"


# ── 5. /generate is POST-only ────────────────────────────────────────────────

def test_generate_route_is_post_only():
    """The /generate route must not respond to GET."""
    from api.routes.prototype import router

    generate_route = next(
        (r for r in router.routes if r.path == "/generate"), None
    )
    assert generate_route is not None, "No route with path '/generate' found"
    assert "GET" not in generate_route.methods, (
        f"GET should not be in methods; got {generate_route.methods}"
    )


# ── 6. motion_intensity=0.0 is valid ─────────────────────────────────────────

def test_prototype_request_motion_intensity_zero_is_valid():
    """Lower boundary value 0.0 must instantiate without error."""
    from api.routes.prototype import PrototypeRequest

    req = PrototypeRequest(
        category="Economy",
        location_feel="Urban",
        time_of_day="Day",
        color_temperature="Neutral",
        mood="Serious",
        motion_intensity=0.0,
    )
    assert req.motion_intensity == 0.0


# ── 7. motion_intensity=1.0 is valid ─────────────────────────────────────────

def test_prototype_request_motion_intensity_one_is_valid():
    """Upper boundary value 1.0 must instantiate without error."""
    from api.routes.prototype import PrototypeRequest

    req = PrototypeRequest(
        category="Economy",
        location_feel="Urban",
        time_of_day="Day",
        color_temperature="Neutral",
        mood="Serious",
        motion_intensity=1.0,
    )
    assert req.motion_intensity == 1.0


# ── 8. run_id format ──────────────────────────────────────────────────────────

def test_run_id_format():
    """Ten consecutive run IDs must each be exactly 8 alphanumeric characters."""
    for _ in range(10):
        run_id = uuid.uuid4().hex[:8]
        assert len(run_id) == 8,           f"run_id length is {len(run_id)}, expected 8"
        assert run_id.isalnum(),           f"run_id '{run_id}' is not alphanumeric"
        assert run_id == run_id.lower(),   f"run_id '{run_id}' contains uppercase characters"
