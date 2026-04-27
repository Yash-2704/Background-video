"""
api/routes/parse.py
────────────────────
POST /api/v1/parse-prompt

Accepts a free-text natural language description and returns the 6 structured
editorial fields inferred by the Groq LLM parser.  The compile step remains
separate — this endpoint only infers; it does not compile prompts.
"""

from fastapi import APIRouter, HTTPException

from api.models import ParseRequest, ParseResponse

router = APIRouter()


@router.post("/parse-prompt", response_model=ParseResponse)
def parse_prompt_route(request: ParseRequest) -> ParseResponse:
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    # Guard import: keeps this module loadable on Mac where GROQ_API_KEY may
    # not be present during test runs that don't exercise this route.
    from core.prompt_parser import parse_free_prompt

    inferred = parse_free_prompt(request.prompt)

    return ParseResponse(
        category=inferred["category"],
        location_feel=inferred["location_feel"],
        time_of_day=inferred["time_of_day"],
        color_temperature=inferred["color_temperature"],
        mood=inferred["mood"],
        motion_intensity=inferred["motion_intensity"],
        original_prompt=request.prompt,
        inference_notes=inferred.get("inference_notes", ""),
    )
