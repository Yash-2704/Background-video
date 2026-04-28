"""
api/routes/compile.py
─────────────────────
POST /api/v1/compile

Accepts a free-text prompt, calls compile_prompt_from_text() once,
and returns a fully-assembled compiled dict ready for the generation route.
"""

import hashlib

from fastapi import APIRouter, HTTPException

from api.models import CompileRequest, CompileResponse
from core.prompt_compiler import FIXED_NEGATIVE_PROMPT
from core.prompt_parser import compile_prompt_from_text

router = APIRouter()

COMPILER_VERSION = "2.0.0"

_LUT_MAP: dict = {
    "Cool":    "cool_authority",
    "Neutral": "neutral",
    "Warm":    "warm_tension",
}

_LOWER_THIRD_MAP: dict = {
    "Serious":   "minimal_dark_bar",
    "Tense":     "high_contrast_black",
    "Neutral":   "minimal_dark_bar",
    "Calm":      "minimal_dark_bar",
    "Uplifting": "warm_lower_bar",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@router.post("/compile", response_model=CompileResponse)
def compile_route(request: CompileRequest) -> CompileResponse:
    try:
        result = compile_prompt_from_text(request.prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt compilation error ({type(exc).__name__}): {exc}",
        ) from exc

    positive  = result["positive_prompt"]
    motion    = result["motion_prompt"]
    negative  = FIXED_NEGATIVE_PROMPT

    compiled = {
        "positive":          positive,
        "motion":            motion,
        "negative":          negative,
        "positive_hash":     _sha256(positive),
        "motion_hash":       _sha256(motion),
        "negative_hash":     _sha256(negative),
        "input_hash_short":  _sha256(request.prompt)[:6],
        "selected_lut":      _LUT_MAP.get(result["color_temperature"], "neutral"),
        "lower_third_style": _LOWER_THIRD_MAP.get(result["mood"], "minimal_dark_bar"),
        "compiler_version":  COMPILER_VERSION,
        "user_input":        {"raw_prompt": request.prompt},
    }

    return CompileResponse(**compiled)
