"""
api/routes/compile.py
─────────────────────
POST /api/v1/compile

The one real route in Prompt 3. Delegates entirely to
core/prompt_compiler.compile_prompts(). No business logic here.
"""

from fastapi import APIRouter, HTTPException

from api.models import CompileResponse, EditorialInput
from core.prompt_compiler import compile_prompts

router = APIRouter()

COMPILER_VERSION = "1.0.0"


@router.post("/compile", response_model=CompileResponse)
def compile_route(editorial_input: EditorialInput) -> CompileResponse:
    try:
        result = compile_prompts(
            editorial_input.model_dump(),
            compiler_version=COMPILER_VERSION,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal compiler error ({type(exc).__name__})",
        ) from exc

    return CompileResponse(**result)
