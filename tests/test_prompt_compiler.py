"""
tests/test_prompt_compiler.py
─────────────────────────────
15 pytest tests for core/prompt_compiler.py.

All tests are deterministic — no random values, no time-dependent logic.
"""

import copy
import pytest

from core.prompt_compiler import (
    CATEGORY_FRAGMENTS,
    LOCATION_FRAGMENTS,
    TIME_FRAGMENTS,
    MOOD_FRAGMENTS,
    MOTION_FRAGMENTS,
    LOWER_THIRD_STYLE_RULES,
    compile_prompts,
    get_all_valid_inputs,
)

# ── Canonical example fixture ─────────────────────────────────────────────────

CANONICAL_INPUT = {
    "category":          "Economy",
    "location_feel":     "Urban",
    "time_of_day":       "Dusk",
    "color_temperature": "Cool",
    "mood":              "Serious",
    "motion_intensity":  "Gentle",
}

CANONICAL_POSITIVE = (
    "financial district architecture, economic infrastructure, "
    "institutional scale, implied market activity, "
    "dense urban environment, glass and steel architecture, "
    "city grid geometry, "
    "fading amber and blue light, long shadows, "
    "golden hour gradient, "
    "muted desaturated tones, cinematic restraint, "
    "no warmth, composed gravity, "
    "wide establishing shot, no people visible, "
    "professional broadcast aesthetic, photorealistic"
)

COMPILER_VERSION = "v2.0"


# ── Test 1: canonical positive prompt — exact string match ────────────────────

def test_canonical_positive_prompt():
    result = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert result["positive"] == CANONICAL_POSITIVE


# ── Test 2: canonical selected_lut ───────────────────────────────────────────

def test_canonical_selected_lut():
    result = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert result["selected_lut"] == "cool_authority"


# ── Test 3: canonical lower_third_style ──────────────────────────────────────

def test_canonical_lower_third_style():
    result = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert result["lower_third_style"] == "minimal_dark_bar"


# ── Test 4: input_hash_short is exactly 6 characters ─────────────────────────

def test_input_hash_short_length():
    result = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert len(result["input_hash_short"]) == 6


# ── Test 5: determinism — identical input produces identical input_hash_short ─

def test_input_hash_short_determinism():
    result_a = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    result_b = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert result_a["input_hash_short"] == result_b["input_hash_short"]


# ── Test 6: hash sensitivity — changing any single field changes the hash ─────

def test_input_hash_short_sensitivity():
    base = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)["input_hash_short"]

    alt_category = {**CANONICAL_INPUT, "category": "Politics"}
    assert compile_prompts(alt_category, COMPILER_VERSION)["input_hash_short"] != base

    alt_location = {**CANONICAL_INPUT, "location_feel": "Nature"}
    assert compile_prompts(alt_location, COMPILER_VERSION)["input_hash_short"] != base

    alt_motion = {**CANONICAL_INPUT, "motion_intensity": "Dynamic"}
    assert compile_prompts(alt_motion, COMPILER_VERSION)["input_hash_short"] != base


# ── Test 7: all 3 prompt hashes are 64-character hex strings ─────────────────

def test_prompt_hashes_are_64_char_hex():
    result = compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    for key in ("positive_hash", "motion_hash", "negative_hash"):
        h = result[key]
        assert len(h) == 64, f"{key} length is {len(h)}, expected 64"
        assert all(c in "0123456789abcdef" for c in h), f"{key} is not hex"


# ── Test 8: missing input field raises ValueError ─────────────────────────────

def test_missing_field_raises_value_error():
    incomplete = {k: v for k, v in CANONICAL_INPUT.items() if k != "mood"}
    with pytest.raises(ValueError) as exc_info:
        compile_prompts(incomplete, COMPILER_VERSION)
    assert "mood" in str(exc_info.value)


# ── Test 9: invalid field value raises ValueError naming the bad field ─────────

def test_invalid_field_value_raises_value_error():
    bad_input = {**CANONICAL_INPUT, "category": "Astrology"}
    with pytest.raises(ValueError) as exc_info:
        compile_prompts(bad_input, COMPILER_VERSION)
    assert "category" in str(exc_info.value)
    assert "Astrology" in str(exc_info.value)


# ── Test 10: get_all_valid_inputs returns dict with all 6 keys ────────────────

def test_get_all_valid_inputs_has_all_keys():
    valid = get_all_valid_inputs()
    expected_keys = {
        "category", "location_feel", "time_of_day",
        "color_temperature", "mood", "motion_intensity",
    }
    assert set(valid.keys()) == expected_keys


# ── Test 11: get_all_valid_inputs lists match lookup table keys ───────────────

def test_get_all_valid_inputs_matches_table_keys():
    valid = get_all_valid_inputs()
    assert set(valid["category"])       == set(CATEGORY_FRAGMENTS.keys())
    assert set(valid["location_feel"])  == set(LOCATION_FRAGMENTS.keys())
    assert set(valid["time_of_day"])    == set(TIME_FRAGMENTS.keys())
    assert set(valid["mood"])           == set(MOOD_FRAGMENTS.keys())
    assert set(valid["motion_intensity"]) == set(MOTION_FRAGMENTS.keys())


# ── Test 12: all 12 lower_third style combinations are reachable ──────────────

def test_all_lower_third_combinations_reachable():
    """
    Iterate every key in LOWER_THIRD_STYLE_RULES, build a minimal valid
    user_input around it, and call compile_prompts. No KeyError must occur.
    """
    # Fixed valid values for the non-key fields
    base_fields = {
        "category":      "General",
        "location_feel": "Urban",
        "time_of_day":   "Day",
        "motion_intensity": "Minimal",
    }
    for (mood, color_temperature) in LOWER_THIRD_STYLE_RULES:
        user_input = {
            **base_fields,
            "mood":              mood,
            "color_temperature": color_temperature,
        }
        result = compile_prompts(user_input, COMPILER_VERSION)
        expected_style = LOWER_THIRD_STYLE_RULES[(mood, color_temperature)]
        assert result["lower_third_style"] == expected_style, (
            f"Combination ({mood}, {color_temperature}) returned wrong style"
        )


# ── Test 13: compile_prompts does not mutate its input dict ───────────────────

def test_compile_prompts_does_not_mutate_input():
    original = copy.deepcopy(CANONICAL_INPUT)
    compile_prompts(CANONICAL_INPUT, COMPILER_VERSION)
    assert CANONICAL_INPUT == original


# ── Test 14: Warm color_temperature maps to "warm_tension" LUT ───────────────

def test_warm_color_temperature_lut():
    warm_input = {**CANONICAL_INPUT, "color_temperature": "Warm"}
    result = compile_prompts(warm_input, COMPILER_VERSION)
    assert result["selected_lut"] == "warm_tension"


# ── Test 15: Neutral color_temperature maps to "neutral" LUT ─────────────────

def test_neutral_color_temperature_lut():
    neutral_input = {**CANONICAL_INPUT, "color_temperature": "Neutral"}
    result = compile_prompts(neutral_input, COMPILER_VERSION)
    assert result["selected_lut"] == "neutral"
