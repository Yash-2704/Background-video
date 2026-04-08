"""
tests/test_gates.py
───────────────────
Tests for core/gates.py — gate evaluation layer.

All fixtures are built from the dry-run probe values defined in
core/probes.py, ensuring test data stays consistent with the probe
output shape without requiring a real video file.
"""

import copy
import pytest

from core.gates import (
    evaluate_gates,
    get_gate_schema,
    GENERATION_CONSTANTS,
    FLICKER_REJECT,
    WARPING_REJECT,
    LOOP_QUALITY_THRESHOLD,
    LUMINANCE_GATE_MIN,
    LUMINANCE_GATE_MAX,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def passing_decode():
    """Dry-run decode probe result — all gates passing."""
    return {
        "mean_luminance":       0.46,
        "luminance_range":      [0.14, 0.79],
        "dominant_hue_degrees": 212.0,
        "saturation_mean":      0.38,
        "luminance_gate_min":   0.30,
        "luminance_gate_max":   0.70,
        "dry_run":              True,
    }


@pytest.fixture
def passing_temporal():
    """Dry-run temporal probe result — all gates passing."""
    return {
        "flicker_index":          0.003,
        "warping_artifact_score": 0.018,
        "scene_cut_detected":     False,
        "perceptual_loop_score":  0.94,
        "frame_count":            521,
        "dry_run":                True,
    }


@pytest.fixture
def failing_decode(passing_decode):
    """Decode probe with mean_luminance above gate max (0.70)."""
    d = copy.deepcopy(passing_decode)
    d["mean_luminance"] = 0.85
    return d


@pytest.fixture
def failing_temporal_flicker(passing_temporal):
    """Temporal probe with flicker_index above threshold (0.01)."""
    d = copy.deepcopy(passing_temporal)
    d["flicker_index"] = 0.025
    return d


@pytest.fixture
def failing_temporal_warp(passing_temporal):
    """Temporal probe with warping_artifact_score above threshold (0.05)."""
    d = copy.deepcopy(passing_temporal)
    d["warping_artifact_score"] = 0.08
    return d


@pytest.fixture
def failing_temporal_scenecut(passing_temporal):
    """Temporal probe with scene_cut_detected = True."""
    d = copy.deepcopy(passing_temporal)
    d["scene_cut_detected"] = True
    return d


@pytest.fixture
def human_flag_temporal(passing_temporal):
    """Temporal probe with perceptual_loop_score below threshold (0.80)."""
    d = copy.deepcopy(passing_temporal)
    d["perceptual_loop_score"] = 0.72
    return d


# ── evaluate_gates — pass cases ──────────────────────────────────────────────────

def test_evaluate_gates_pass_overall(passing_decode, passing_temporal):
    """Test 1: Clean probes return overall == 'pass'."""
    result = evaluate_gates(passing_decode, passing_temporal)
    assert result["overall"] == "pass"


def test_evaluate_gates_pass_has_required_keys(passing_decode, passing_temporal):
    """Test 2: Result contains all required keys from get_gate_schema()."""
    result = evaluate_gates(passing_decode, passing_temporal)
    schema = get_gate_schema()
    for key in schema["required_keys"]:
        assert key in result, f"Missing required key: {key}"


def test_evaluate_gates_pass_failures_empty(passing_decode, passing_temporal):
    """Test 3: failures list is empty on clean pass."""
    result = evaluate_gates(passing_decode, passing_temporal)
    assert result["failures"] == []


def test_evaluate_gates_pass_human_flags_empty(passing_decode, passing_temporal):
    """Test 4: human_flags list is empty on clean pass."""
    result = evaluate_gates(passing_decode, passing_temporal)
    assert result["human_flags"] == []


def test_evaluate_gates_pass_gates_checked(passing_decode, passing_temporal):
    """Test 5: gates_checked == 5."""
    result = evaluate_gates(passing_decode, passing_temporal)
    assert result["gates_checked"] == 5


# ── evaluate_gates — auto_reject cases ──────────────────────────────────────────

def test_evaluate_gates_luminance_fail_overall(failing_decode, passing_temporal):
    """Test 6: High luminance returns overall == 'fail'."""
    result = evaluate_gates(failing_decode, passing_temporal)
    assert result["overall"] == "fail"


def test_evaluate_gates_luminance_fail_entry(failing_decode, passing_temporal):
    """Test 7: failures contains exactly 1 entry with gate == 'luminance_range'."""
    result = evaluate_gates(failing_decode, passing_temporal)
    assert len(result["failures"]) == 1
    assert result["failures"][0]["gate"] == "luminance_range"


def test_evaluate_gates_flicker_fail_overall(passing_decode, failing_temporal_flicker):
    """Test 8: High flicker returns overall == 'fail'."""
    result = evaluate_gates(passing_decode, failing_temporal_flicker)
    assert result["overall"] == "fail"


def test_evaluate_gates_flicker_fail_entry(passing_decode, failing_temporal_flicker):
    """Test 9: failures contains entry with gate == 'flicker_index'."""
    result = evaluate_gates(passing_decode, failing_temporal_flicker)
    gates = [f["gate"] for f in result["failures"]]
    assert "flicker_index" in gates


def test_evaluate_gates_warp_fail_overall(passing_decode, failing_temporal_warp):
    """Test 10: High warp score returns overall == 'fail'."""
    result = evaluate_gates(passing_decode, failing_temporal_warp)
    assert result["overall"] == "fail"


def test_evaluate_gates_scenecut_fail_overall(passing_decode, failing_temporal_scenecut):
    """Test 11: Scene cut detected returns overall == 'fail'."""
    result = evaluate_gates(passing_decode, failing_temporal_scenecut)
    assert result["overall"] == "fail"


def test_evaluate_gates_multiple_failures(failing_decode, failing_temporal_flicker):
    """Test 12: Multiple simultaneous gate failures — both decode + flicker."""
    result = evaluate_gates(failing_decode, failing_temporal_flicker)
    assert len(result["failures"]) == 2


# ── evaluate_gates — human_review cases ─────────────────────────────────────────

def test_evaluate_gates_human_review_overall(passing_decode, human_flag_temporal):
    """Test 13: Low loop score returns overall == 'human_review'."""
    result = evaluate_gates(passing_decode, human_flag_temporal)
    assert result["overall"] == "human_review"


def test_evaluate_gates_human_review_flag_entry(passing_decode, human_flag_temporal):
    """Test 14: human_flags contains exactly 1 entry with gate == 'perceptual_loop_score'."""
    result = evaluate_gates(passing_decode, human_flag_temporal)
    assert len(result["human_flags"]) == 1
    assert result["human_flags"][0]["gate"] == "perceptual_loop_score"


def test_evaluate_gates_human_review_no_failures(passing_decode, human_flag_temporal):
    """Test 15: failures list is empty when only human_review flags exist."""
    result = evaluate_gates(passing_decode, human_flag_temporal)
    assert result["failures"] == []


# ── evaluate_gates — human_review + failure coexist ─────────────────────────────

def test_evaluate_gates_auto_reject_priority(failing_decode, human_flag_temporal):
    """Test 16: auto_reject takes priority over human_review — overall == 'fail'."""
    result = evaluate_gates(failing_decode, human_flag_temporal)
    assert result["overall"] == "fail"


def test_evaluate_gates_both_lists_populated(failing_decode, human_flag_temporal):
    """Test 17: Both failures and human_flags lists are populated when both conditions met."""
    result = evaluate_gates(failing_decode, human_flag_temporal)
    assert len(result["failures"]) >= 1
    assert len(result["human_flags"]) >= 1


# ── evaluate_gates — threshold sourcing ─────────────────────────────────────────

def test_evaluate_gates_thresholds_from_constants(passing_decode, passing_temporal):
    """Test 18: thresholds_used matches GENERATION_CONSTANTS['quality_gates'] exactly."""
    result = evaluate_gates(passing_decode, passing_temporal)
    qg = GENERATION_CONSTANTS["quality_gates"]
    assert result["thresholds_used"]["flicker_index_reject"]    == qg["flicker_index_reject"]
    assert result["thresholds_used"]["warping_artifact_reject"] == qg["warping_artifact_reject"]
    assert result["thresholds_used"]["loop_quality_threshold"]  == qg["loop_quality_threshold"]
    assert result["thresholds_used"]["luminance_gate_min"]      == qg["luminance_gate_min"]
    assert result["thresholds_used"]["luminance_gate_max"]      == qg["luminance_gate_max"]


# ── evaluate_gates — input validation ───────────────────────────────────────────

def test_evaluate_gates_missing_decode_key(passing_temporal):
    """Test 19: Missing 'mean_luminance' in decode dict raises ValueError."""
    bad_decode = {
        "luminance_range":      [0.14, 0.79],
        "dominant_hue_degrees": 212.0,
        "saturation_mean":      0.38,
        "luminance_gate_min":   0.30,
        "luminance_gate_max":   0.70,
        "dry_run":              True,
        # "mean_luminance" deliberately omitted
    }
    with pytest.raises(ValueError, match="Missing key"):
        evaluate_gates(bad_decode, passing_temporal)


def test_evaluate_gates_missing_temporal_key(passing_decode):
    """Test 20: Missing 'flicker_index' in temporal dict raises ValueError."""
    bad_temporal = {
        "warping_artifact_score": 0.018,
        "scene_cut_detected":     False,
        "perceptual_loop_score":  0.94,
        "frame_count":            521,
        "dry_run":                True,
        # "flicker_index" deliberately omitted
    }
    with pytest.raises(ValueError, match="Missing key"):
        evaluate_gates(passing_decode, bad_temporal)


# ── get_gate_schema ──────────────────────────────────────────────────────────────

def test_get_gate_schema_required_keys():
    """Test 21: get_gate_schema() returns dict with all 4 required top-level keys."""
    schema = get_gate_schema()
    for key in ("required_keys", "overall_values", "auto_reject_gates", "human_review_gates"):
        assert key in schema, f"Missing key in gate schema: {key}"


def test_get_gate_schema_overall_values():
    """Test 22: 'overall_values' list contains exactly the 3 valid state strings."""
    schema = get_gate_schema()
    assert set(schema["overall_values"]) == {"pass", "fail", "human_review"}
    assert len(schema["overall_values"]) == 3
