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

    # Reject duplicate concurrent requests for the same run_id.
    # React StrictMode (dev) runs effects twice — this prevents two concurrent
    # pipeline runs that race each other and corrupt shared output directories.
    if request.run_id in RUN_REGISTRY and RUN_REGISTRY[request.run_id].get("status") == "running":
        raise HTTPException(status_code=409, detail="Run already in progress.")

    _init_run_state(request.run_id)

    # Fire-and-forget: start pipeline in background, return immediately.
    # The entire pipeline takes ~19 minutes — holding the HTTP connection
    # that long causes browser timeouts ("Failed to fetch"). Instead we
    # return status="running" now and let the frontend poll
    # GET /api/v1/run/{run_id}/status every 2 seconds for live progress.
    # The orchestrator stores the full result in RUN_REGISTRY on completion.
    async def _run_and_store():
        try:
            await asyncio.to_thread(run_pipeline, request.run_id, compiled["user_input"])
        except Exception as exc:
            import traceback
            print(f"\n[PIPELINE ERROR] {exc}", flush=True)
            traceback.print_exc()
            # run_pipeline's except block already writes error + failed status
            # to RUN_REGISTRY — nothing more to do here.

    asyncio.ensure_future(_run_and_store())

    return GenerateResponse(
        run_id=request.run_id,
        status="running",
        stages=RUN_REGISTRY[request.run_id]["stages"],
        message="Pipeline started. Poll GET /api/v1/run/{run_id}/status for progress.",
    )
