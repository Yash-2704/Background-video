"""POST /api/v1/generate full orchestration route."""

import asyncio

from fastapi import APIRouter, HTTPException

from api.models import CompileResponse, GenerateRequest, GenerateResponse, PipelineStage
from core.orchestrator import RUN_REGISTRY, _init_run_state, run_pipeline
from core.prompt_compiler import compile_prompts

router = APIRouter()
COMPILER_VERSION = "1.0.0"
_PIPELINE_STAGES = [
    "prompt_compilation",
    "generation",
    "probe_decode",
    "probe_temporal",
    "gate_evaluation",
    "upscale",
    "mask_generation",
    "lut_grading",
    "composite",
    "preview_export",
    "metadata_assembly",
]


def _build_pending_stages() -> list[PipelineStage]:
    return [
        PipelineStage(
            stage=stage,
            status="pending",
            message="",
            started_at=None,
            ended_at=None,
        )
        for stage in _PIPELINE_STAGES
    ]

@router.post("/generate", response_model=GenerateResponse)
async def generate_route(
    request: GenerateRequest,
) -> GenerateResponse:
    # Backward-compatible request shape used by legacy API tests.
    if request.editorial_input is not None:
        compiled_dict = compile_prompts(
            request.editorial_input.model_dump(),
            compiler_version=COMPILER_VERSION,
        )
        run_id = f"bg_001_{compiled_dict['input_hash_short']}"
        _init_run_state(run_id)
        return GenerateResponse(
            run_id=run_id,
            status="queued",
            dry_run=request.dry_run,
            stages=_build_pending_stages(),
            editorial_input=request.editorial_input,
            compiled=CompileResponse(**{
                k: v for k, v in compiled_dict.items()
                if k in CompileResponse.model_fields
            }),
            message="Pipeline queued. Poll GET /api/v1/run/{run_id}/status for progress.",
        )

    compiled = request.compiled or {}
    if not request.run_id:
        raise HTTPException(status_code=422, detail="Field 'run_id' is required.")
    if "user_input" not in compiled:
        raise HTTPException(
            status_code=422,
            detail=(
                "Field 'compiled.user_input' is required; this should come from "
                "compile_prompts() output."
            ),
        )

    _init_run_state(request.run_id)

    try:
        result = await asyncio.to_thread(
            run_pipeline,
            request.run_id,
            compiled["user_input"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("status") == "escalated":
        return GenerateResponse(
            run_id=request.run_id,
            status="escalated",
            stages=RUN_REGISTRY[request.run_id]["stages"],
            failure_log=result.get("failure_log"),
            error=RUN_REGISTRY[request.run_id].get("error"),
        )

    return GenerateResponse(**result)
