"""
core/prompt_parser.py
──────────────────────
LLM-backed free-text parser for the Background Video Generation pipeline.

Accepts a natural language description of a desired video background and
returns the same 6 structured fields expected by compile_prompts().  Uses
Groq (llama-3.3-70b-versatile) at temperature=0.0 for maximum determinism.
Invalid LLM output is silently replaced by per-field safe defaults — no
exception is ever raised for bad model output.
"""

import hashlib
import json
import os
import re

from core.prompt_compiler import get_all_valid_inputs

# ── API key ───────────────────────────────────────────────────────────────────
# Read once at module level so the error message is emitted at import time
# on machines where the key is set.  If absent, a clear RuntimeError is raised
# inside parse_free_prompt() so the module still imports cleanly on Mac/CI
# environments that run only the non-LLM parts of the pipeline.
_GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# ── Safe fallback values ──────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "category":          "General",
    "location_feel":     "Urban",
    "time_of_day":       "Day",
    "color_temperature": "Neutral",
    "mood":              "Neutral",
    "motion_intensity":  "Gentle",
}


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    valid = get_all_valid_inputs()
    return (
        "You are a structured data extractor for a broadcast video generation system.\n\n"
        "Given a natural language description of a desired video background, extract "
        "exactly these 6 fields and return them as a JSON object.\n\n"
        "Valid values per field:\n"
        f"  category:          {valid['category']}\n"
        f"  location_feel:     {valid['location_feel']}\n"
        f"  time_of_day:       {valid['time_of_day']}\n"
        f"  color_temperature: {valid['color_temperature']}\n"
        f"  mood:              {valid['mood']}\n"
        f"  motion_intensity:  {valid['motion_intensity']}\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON. No markdown fences, no preamble, no explanation.\n"
        "- Always return all 7 keys (the 6 fields above plus inference_notes).\n"
        "- If a field is ambiguous, use the most neutral valid value.\n"
        "- inference_notes must be a brief 1-2 sentence explanation of your choices.\n\n"
        "Output schema (return this exact structure with no surrounding text):\n"
        '{\n'
        '  "category": "...",\n'
        '  "location_feel": "...",\n'
        '  "time_of_day": "...",\n'
        '  "color_temperature": "...",\n'
        '  "mood": "...",\n'
        '  "motion_intensity": "...",\n'
        '  "inference_notes": "brief explanation of choices"\n'
        '}'
    )


_SYSTEM_PROMPT: str = _build_system_prompt()
_SYSTEM_PROMPT_HASH: str = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
print(f"[prompt_parser] system_prompt_hash={_SYSTEM_PROMPT_HASH}", flush=True)


# ── Wan2.2 enrichment system prompt ──────────────────────────────────────────

_WAN_ENRICHMENT_SYSTEM_PROMPT: str = (
    "You are a prompt engineer for Wan2.2-TI2V-5B, a text-to-video diffusion model "
    "used for broadcast background video production.\n\n"
    "Your task: Expand a short editorial prompt into a structured 80-120 word prompt "
    "that Wan2.2 can use to generate high-quality, coherent video output.\n\n"
    "Required structure (write as continuous prose in this order):\n"
    "1. Scene description — specific architecture, environment, or landscape with "
    "concrete visual details\n"
    "2. camera movement — one explicit movement such as: slow dolly forward, "
    "static locked shot, gentle pan right, subtle crane up, slow push-in\n"
    "3. Lighting — color, direction, intensity, and shadow quality\n"
    "4. Visual style — cinematic depth of field, lens characteristics, photorealistic rendering\n"
    "5. Color grade — specific tone, saturation, and contrast\n"
    "6. Atmosphere — ambient conditions such as wind, haze, humidity, or stillness\n"
    "7. Quality anchors — end with exactly: "
    "\"stable textures, smooth motion, broadcast quality, no artifacts, no text, "
    "no people, no faces\"\n\n"
    "Hard rules:\n"
    "- NEVER use these words: evokes, suggests, conveys, feels, emotional\n"
    "- NO people, NO faces, NO hands, NO bodies, NO text overlays, NO logos, "
    "NO news graphics\n"
    "- This is a BACKGROUND video — all foreground must be empty space, architecture, "
    "or nature\n"
    "- Use only concrete visual and cinematographic language\n"
    "- Return ONLY the expanded prompt. No preamble, no explanation, no quotation marks.\n"
    "- Target length: 80-120 words"
)


# ── Wan2.2 I2V enrichment system prompt ──────────────────────────────────────

_WAN_I2V_ENRICHMENT_SYSTEM_PROMPT: str = (
    "You are a motion prompt engineer for Wan2.2-TI2V-5B, an image-to-video diffusion model "
    "used for broadcast background video production.\n\n"
    "Your task: Expand a short animation description into a focused 60-80 word motion prompt. "
    "The scene is already defined by the input image — do NOT describe it.\n\n"
    "Focus ONLY on motion language:\n"
    "- Element movement speeds (e.g. 'clouds drift left at 0.2x speed')\n"
    "- Camera behaviour (e.g. 'static locked shot', 'subtle parallax drift of 3px')\n"
    "- Light behaviour (e.g. 'sunlight pulses with 4-second interval', "
    "'shadows lengthen slowly')\n"
    "- Temporal rhythm — favour loop-friendly patterns with even, repeating cycles\n\n"
    "Use concrete, measurable motion language. Avoid abstract adjectives.\n"
    "End with exactly: "
    "\"smooth looping motion, broadcast quality, no cuts, no people, no text appearing\"\n\n"
    "Hard rules:\n"
    "- NEVER describe the scene, environment, colours, or objects — only HOW things move\n"
    "- NO people, NO faces, NO text overlays\n"
    "- Return ONLY the motion prompt. No preamble, no explanation.\n"
    "- Target length: 60-80 words"
)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_free_prompt(user_prompt: str) -> dict:
    """
    Parse a natural language video description into 6 structured fields.

    Returns a dict with keys: category, location_feel, time_of_day,
    color_temperature, mood, motion_intensity, inference_notes.
    Never raises on bad LLM output — falls back to _DEFAULTS per field.
    """
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to .env at the project root."
        )

    from groq import Groq  # guard-imported so module loads without groq on Mac

    client = Groq(api_key=_GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw: str = response.choices[0].message.content

    # Strip markdown fences that the model may add despite instructions
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed: dict = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {**_DEFAULTS, "inference_notes": ""}

    valid = get_all_valid_inputs()
    result: dict = {}

    for field in ("category", "location_feel", "time_of_day",
                  "color_temperature", "mood", "motion_intensity"):
        value = parsed.get(field, "")
        if value in valid[field]:
            result[field] = value
        else:
            result[field] = _DEFAULTS[field]

    result["inference_notes"] = parsed.get("inference_notes", "")
    return result


def enrich_prompt_for_wan(positive_prompt: str, motion_prompt: str) -> str:
    """
    Expand a short compiled positive prompt into an 80-120 word Wan2.2-structured prompt.

    Only the positive prompt is expanded; motion_prompt is passed as context only.
    Raises on any Groq failure — the caller is responsible for catching and falling back.
    """
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to .env at the project root."
        )

    from groq import Groq  # guard-imported so module loads without groq on Mac

    client = Groq(api_key=_GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _WAN_ENRICHMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Editorial prompt: {positive_prompt}\n"
                    f"Motion intent: {motion_prompt}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def enrich_prompt_for_i2v(animation_prompt: str) -> str:
    """
    Expand a short animation description into a 60-80 word motion-only prompt for Wan2.2 I2V.

    Focuses exclusively on motion language — camera, element, and light behaviour.
    Raises on any Groq failure — the caller is responsible for catching and falling back.
    """
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to .env at the project root."
        )

    from groq import Groq  # guard-imported so module loads without groq on Mac

    client = Groq(api_key=_GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _WAN_I2V_ENRICHMENT_SYSTEM_PROMPT},
            {"role": "user",   "content": animation_prompt},
        ],
        temperature=0.4,
        max_tokens=256,
    )

    return response.choices[0].message.content.strip()


# ── Wan2.2 T2V direct-compile system prompt ───────────────────────────────────

_WAN_T2V_COMPILE_SYSTEM_PROMPT: str = (
    "You are a prompt engineer for Wan2.2-TI2V-5B, a text-to-video diffusion model "
    "used for broadcast background video production.\n\n"
    "Your task: Rewrite the user's prompt into a structured Wan2.2-optimised video prompt "
    "and extract metadata. Return ONLY valid JSON with no markdown, no preamble.\n\n"
    "Output schema (return this exact structure):\n"
    "{\n"
    '  "positive_prompt": "80-120 word Wan2.2-optimised prompt",\n'
    '  "motion_prompt": "one sentence describing camera/scene motion only",\n'
    '  "color_temperature": "Cool | Neutral | Warm",\n'
    '  "mood": "Serious | Tense | Neutral | Calm | Uplifting",\n'
    '  "inference_notes": "brief 1-2 sentence explanation of choices"\n'
    "}\n\n"
    "positive_prompt structure (write as continuous prose in this order):\n"
    "1. Scene — specific architecture, environment, or landscape with concrete visual details\n"
    "2. Camera — explicitly state one of: camera locked static, slow dolly forward, "
    "gentle pan right, subtle crane up, slow push-in\n"
    "3. Lighting — colour, direction, intensity, shadow quality\n"
    "4. Visual style — cinematic depth of field, lens characteristics, photorealistic rendering\n"
    "5. Colour grade — specific tone, saturation, contrast\n"
    "6. Atmosphere — ambient conditions such as wind, haze, humidity, stillness\n"
    "7. Quality anchors — end with exactly: "
    "\"stable textures, smooth motion, broadcast quality, no artifacts, no text, "
    "no people, no faces\"\n\n"
    "Hard rules:\n"
    "- NEVER use: evokes, suggests, conveys, feels, emotional\n"
    "- NO people, NO faces, NO hands, NO bodies, NO text overlays, NO logos, "
    "NO news graphics\n"
    "- This is a BACKGROUND video — all foreground must be empty space, "
    "architecture, or nature\n"
    "- motion_prompt: one sentence only, camera/scene motion language, no scene description\n"
    "- color_temperature must be exactly one of: Cool, Neutral, Warm\n"
    "- mood must be exactly one of: Serious, Tense, Neutral, Calm, Uplifting\n"
    "- Return ONLY valid JSON. No markdown fences, no explanation outside the JSON."
)

_VALID_COLOR_TEMPERATURES = {"Cool", "Neutral", "Warm"}
_VALID_MOODS = {"Serious", "Tense", "Neutral", "Calm", "Uplifting"}

_COMPILE_DEFAULTS: dict = {
    "positive_prompt":   "Wide architectural establishing shot, camera locked static, "
                         "neutral daylight, cinematic depth of field, balanced tones, "
                         "still atmosphere, stable textures, smooth motion, broadcast quality, "
                         "no artifacts, no text, no people, no faces",
    "motion_prompt":     "Camera locked static, no movement, imperceptible drift only.",
    "color_temperature": "Neutral",
    "mood":              "Neutral",
    "inference_notes":   "",
}


def compile_prompt_from_text(raw_prompt: str) -> dict:
    """
    Rewrite a free-text user prompt directly into a Wan2.2-optimised compiled dict.

    Makes a single Groq call. Returns dict with keys:
    positive_prompt, motion_prompt, color_temperature, mood, inference_notes.
    Never raises on bad LLM output — falls back to _COMPILE_DEFAULTS.
    """
    if not _GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to .env at the project root."
        )

    from groq import Groq  # guard-imported so module loads without groq on Mac

    client = Groq(api_key=_GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _WAN_T2V_COMPILE_SYSTEM_PROMPT},
            {"role": "user",   "content": raw_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    raw: str = response.choices[0].message.content
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed: dict = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return dict(_COMPILE_DEFAULTS)

    color_temperature = parsed.get("color_temperature", "")
    if color_temperature not in _VALID_COLOR_TEMPERATURES:
        color_temperature = _COMPILE_DEFAULTS["color_temperature"]

    mood = parsed.get("mood", "")
    if mood not in _VALID_MOODS:
        mood = _COMPILE_DEFAULTS["mood"]

    return {
        "positive_prompt":   parsed.get("positive_prompt", "") or _COMPILE_DEFAULTS["positive_prompt"],
        "motion_prompt":     parsed.get("motion_prompt", "")   or _COMPILE_DEFAULTS["motion_prompt"],
        "color_temperature": color_temperature,
        "mood":              mood,
        "inference_notes":   parsed.get("inference_notes", ""),
    }
