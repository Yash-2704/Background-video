"""
core/prompt_compiler.py
────────────────────────
Deterministic lookup-and-assembly compiler.

Converts 6 editorial inputs into 3 hashed prompts plus derived metadata.
No LLM. No external calls. Pure Python lookup tables and string assembly.

Config is loaded once at module import time from:
  <project_root>/config/generation_constants.json

All fragment tables are module-level constants. Any change to a table
string invalidates cross-run hash comparisons — treat them as locked.
"""

import hashlib
import json
from pathlib import Path

# ── Config loading ────────────────────────────────────────────────────────────
# Resolve project root relative to THIS file's location, not cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "generation_constants.json"

with CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    _GENERATION_CONSTANTS = json.load(_fh)

# lut_options maps color_temperature → LUT name
# e.g. {"Cool": "cool_authority", "Neutral": "neutral", "Warm": "warm_tension"}
_LUT_OPTIONS: dict = _GENERATION_CONSTANTS["lut_options"]


# ── Fragment lookup tables ────────────────────────────────────────────────────
# These strings are hashed. Do NOT alter spacing, punctuation, or casing
# without regenerating all stored hashes in downstream systems.

CATEGORY_FRAGMENTS: dict = {
    "Economy":  (
        "financial district architecture, economic infrastructure, "
        "institutional scale, implied market activity"
    ),
    "Politics": (
        "civic plaza, government building facades, "
        "formal institutional geometry, ceremonial space"
    ),
    "Tech": (
        "data center environment, server infrastructure aesthetic, "
        "clean technical lines, engineered precision"
    ),
    "Climate": (
        "natural landscape under atmospheric pressure, "
        "open sky with weather presence, environmental scale"
    ),
    "Crime": (
        "urban back streets, low-key ambient lighting, "
        "compressed shadowed atmosphere, city infrastructure"
    ),
    "Sports": (
        "open stadium environment, spatial venue scale, "
        "crowd infrastructure implied, arena geometry"
    ),
    "General": (
        "neutral architectural environment, "
        "non-specific professional setting"
    ),
}

LOCATION_FRAGMENTS: dict = {
    "Urban": (
        "dense urban environment, glass and steel architecture, "
        "city grid geometry"
    ),
    "Government": (
        "wide institutional plaza, stone and concrete facades, "
        "formal civic architecture"
    ),
    "Nature": (
        "open natural landscape, trees and horizon, "
        "no man-made structures"
    ),
    "Abstract": (
        "abstract light forms, soft bokeh, "
        "non-representational depth"
    ),
    "Data": (
        "data center corridors, server rack arrays, "
        "blue ambient light, technical infrastructure"
    ),
}

TIME_FRAGMENTS: dict = {
    "Day":   "midday even lighting, clear sky, neutral daylight",
    "Dusk":  (
        "fading amber and blue light, long shadows, "
        "golden hour gradient"
    ),
    "Night": "city lights, dark sky, artificial illumination pools",
    "N/A":   "indeterminate lighting, ambient fill, no time reference",
}

MOOD_FRAGMENTS: dict = {
    "Serious": (
        "muted desaturated tones, cinematic restraint, "
        "no warmth, composed gravity"
    ),
    "Tense": (
        "high contrast shadows, compressed dynamic range, "
        "heavy atmosphere, visual pressure"
    ),
    "Neutral": (
        "balanced exposure, moderate saturation, "
        "clean broadcast aesthetic"
    ),
    "Uplifting": (
        "open sky, elevated vantage point, "
        "warm diffuse light, expansive composition"
    ),
}

MOTION_FRAGMENTS: dict = {
    "Minimal": (
        "completely static camera, locked frame, "
        "imperceptible drift only"
    ),
    "Gentle":  "slow lateral camera drift, gentle parallax, 0.2x speed",
    "Dynamic": (
        "controlled push-in, moderate motion, "
        "architectural tracking shot"
    ),
}

# Fixed suffixes — appended verbatim; never assembled conditionally
FIXED_POSITIVE_SUFFIX: str = (
    "wide establishing shot, no people visible, "
    "professional broadcast aesthetic, photorealistic"
)

FIXED_MOTION_SUFFIX: str = (
    "smooth and continuous motion, no cuts, no camera shake, "
    "no zoom, no sudden movements, temporally stable"
)

FIXED_NEGATIVE_PROMPT: str = (
    "text, titles, watermarks, logos, people, faces, hands, bodies, "
    "news tickers, clocks, timestamps, flags, explicit content, cartoon, "
    "CGI artifacts, lens flare, overexposed regions, flickering, strobing, "
    "fast motion, shaky camera, jump cuts, split screen, overlays, "
    "UI elements, animated graphics, lower thirds, chyrons"
)


# ── Lower-third style rule table ──────────────────────────────────────────────
# Key: (mood, color_temperature) — all 12 combinations are present.
# Value: style token consumed by the graphics renderer (Prompt 9+).

LOWER_THIRD_STYLE_RULES: dict = {
    ("Serious",   "Cool"):    "minimal_dark_bar",
    ("Serious",   "Neutral"): "minimal_dark_bar",
    ("Serious",   "Warm"):    "minimal_dark_bar",
    ("Tense",     "Cool"):    "high_contrast_black",
    ("Tense",     "Neutral"): "high_contrast_black",
    ("Tense",     "Warm"):    "high_contrast_black",
    ("Neutral",   "Cool"):    "standard_bar",
    ("Neutral",   "Neutral"): "standard_bar",
    ("Neutral",   "Warm"):    "standard_bar",
    ("Uplifting", "Cool"):    "light_transparent_bar",
    ("Uplifting", "Neutral"): "light_transparent_bar",
    ("Uplifting", "Warm"):    "warm_gradient_bar",
}


# ── Allowed value sets (derived from table keys — single source of truth) ────
# These are referenced both by the validator and by get_all_valid_inputs().
_VALID_CATEGORIES         = set(CATEGORY_FRAGMENTS.keys())
_VALID_LOCATION_FEELS     = set(LOCATION_FRAGMENTS.keys())
_VALID_TIMES_OF_DAY       = set(TIME_FRAGMENTS.keys())
_VALID_COLOR_TEMPERATURES = set(_LUT_OPTIONS.keys())          # from config
_VALID_MOODS              = set(MOOD_FRAGMENTS.keys())
_VALID_MOTION_INTENSITIES = set(MOTION_FRAGMENTS.keys())


# ── Private helper ────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    """Return the lowercase hex SHA256 digest of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def compile_prompts(user_input: dict, compiler_version: str) -> dict:
    """
    Convert 6 editorial inputs into 3 hashed prompts plus derived metadata.

    Parameters
    ----------
    user_input : dict
        Must contain exactly the 6 keys listed in _REQUIRED_FIELDS.
        The dict is never mutated.
    compiler_version : str
        Caller-supplied version token; passed through verbatim into the
        return dict for provenance tracking.

    Returns
    -------
    dict
        10-key result dict. See module docstring for full schema.

    Raises
    ------
    ValueError
        If any required key is missing or any value is not in its allowed set.
    """

    _REQUIRED_FIELDS = [
        "category",
        "location_feel",
        "time_of_day",
        "color_temperature",
        "mood",
        "motion_intensity",
    ]

    # ── 1. Presence validation ────────────────────────────────────────────────
    for field in _REQUIRED_FIELDS:
        if field not in user_input:
            raise ValueError(
                f"compile_prompts: required field '{field}' is missing "
                f"from user_input."
            )

    # Read values into locals (never mutate user_input)
    category          = user_input["category"]
    location_feel     = user_input["location_feel"]
    time_of_day       = user_input["time_of_day"]
    color_temperature = user_input["color_temperature"]
    mood              = user_input["mood"]
    motion_intensity  = user_input["motion_intensity"]

    # ── 2. Value validation ───────────────────────────────────────────────────
    _checks = [
        ("category",          category,          _VALID_CATEGORIES),
        ("location_feel",     location_feel,     _VALID_LOCATION_FEELS),
        ("time_of_day",       time_of_day,       _VALID_TIMES_OF_DAY),
        ("color_temperature", color_temperature, _VALID_COLOR_TEMPERATURES),
        ("mood",              mood,              _VALID_MOODS),
        ("motion_intensity",  motion_intensity,  _VALID_MOTION_INTENSITIES),
    ]
    for field_name, value, valid_set in _checks:
        if value not in valid_set:
            raise ValueError(
                f"compile_prompts: field '{field_name}' has invalid value "
                f"'{value}'. Allowed: {sorted(valid_set)}"
            )

    # ── 3. Positive prompt assembly ───────────────────────────────────────────
    # Canonical join order (do not reorder — hashes depend on this):
    #   category_fragment, location_fragment, time_fragment,
    #   mood_fragment, FIXED_POSITIVE_SUFFIX
    # Each fragment is joined with ", " — no trailing separator.
    positive_prompt: str = ", ".join([
        CATEGORY_FRAGMENTS[category],
        LOCATION_FRAGMENTS[location_feel],
        TIME_FRAGMENTS[time_of_day],
        MOOD_FRAGMENTS[mood],
        FIXED_POSITIVE_SUFFIX,
    ])

    # ── 4. Motion prompt assembly ─────────────────────────────────────────────
    # Canonical join order: motion_fragment, FIXED_MOTION_SUFFIX
    motion_prompt: str = ", ".join([
        MOTION_FRAGMENTS[motion_intensity],
        FIXED_MOTION_SUFFIX,
    ])

    # ── 5. Negative prompt ────────────────────────────────────────────────────
    negative_prompt: str = FIXED_NEGATIVE_PROMPT  # verbatim, no assembly

    # ── 6. Individual prompt hashes ───────────────────────────────────────────
    positive_hash: str = _sha256(positive_prompt)
    motion_hash:   str = _sha256(motion_prompt)
    negative_hash: str = _sha256(negative_prompt)

    # ── 7. Input combination hash ─────────────────────────────────────────────
    # Canonical field order for concatenation:
    #   category + location_feel + time_of_day +
    #   color_temperature + mood + motion_intensity
    input_concat: str = (
        category
        + location_feel
        + time_of_day
        + color_temperature
        + mood
        + motion_intensity
    )
    input_hash_short: str = _sha256(input_concat)[:6]

    # ── 8. LUT lookup (from loaded config, not hardcoded) ─────────────────────
    selected_lut: str = _LUT_OPTIONS[color_temperature]

    # ── 9. Lower-third style lookup ───────────────────────────────────────────
    lower_third_style: str = LOWER_THIRD_STYLE_RULES[(mood, color_temperature)]

    return {
        "positive":          positive_prompt,
        "motion":            motion_prompt,
        "negative":          negative_prompt,
        "positive_hash":     positive_hash,
        "motion_hash":       motion_hash,
        "negative_hash":     negative_hash,
        "input_hash_short":  input_hash_short,
        "selected_lut":      selected_lut,
        "lower_third_style": lower_third_style,
        "compiler_version":  compiler_version,
        # Backward-compatible addition (Prompt 8): downstream assembler needs the
        # six original input values for metadata provenance and integration contract
        # derivation without requiring caller to re-pass user_input separately.
        "user_input": {
            "category":          category,
            "location_feel":     location_feel,
            "time_of_day":       time_of_day,
            "color_temperature": color_temperature,
            "mood":              mood,
            "motion_intensity":  motion_intensity,
        },
    }


def get_all_valid_inputs() -> dict:
    """
    Return all valid options per field, derived from lookup table keys.

    The lists are built at call time from the module-level tables so any
    addition to a table is automatically reflected here.

    Returns
    -------
    dict
        Keys are the 6 user_input field names; values are sorted lists of
        valid string options.
    """
    return {
        "category":          sorted(_VALID_CATEGORIES),
        "location_feel":     sorted(_VALID_LOCATION_FEELS),
        "time_of_day":       sorted(_VALID_TIMES_OF_DAY),
        "color_temperature": sorted(_VALID_COLOR_TEMPERATURES),
        "mood":              sorted(_VALID_MOODS),
        "motion_intensity":  sorted(_VALID_MOTION_INTENSITIES),
    }
