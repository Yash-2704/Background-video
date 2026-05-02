"""
api/routes/status.py
────────────────────
Two status endpoints:

  GET /api/v1/status/{run_id}         — legacy stub (preserved for
                                        backward compatibility with the
                                        existing 210-test suite)

  GET /api/v1/run/{run_id}/status     — live orchestrator status from
                                        RUN_REGISTRY (Prompt 12)
"""

from fastapi import APIRouter, HTTPException

from api.models import PipelineStage, RunStatusResponse, StatusResponse
from core.orchestrator import RUN_REGISTRY

router = APIRouter()

_PIPELINE_STAGES = [
    "generation",
    "interpolation",
    "crossfade",
    "probe",
    "gate",
    "upscale",
    "mask",
    "lut_grade",
    "composite",
    "preview",
    "metadata",
]

# First two stages are shown as complete in the legacy stub.
_COMPLETE_STAGES = {"generation", "interpolation"}


def _build_stub_stages() -> list[PipelineStage]:
    return [
        PipelineStage(
            stage=s,
            status="complete" if s in _COMPLETE_STAGES else "pending",
            message="",
            started_at=None,
            ended_at=None,
        )
        for s in _PIPELINE_STAGES
    ]


# ── Legacy stub endpoint (kept for backward compatibility) ─────────────────────

@router.get("/status/{run_id}", response_model=StatusResponse)
def status_route(run_id: str) -> StatusResponse:
    """Original stub — preserved so existing tests continue to pass."""
    return StatusResponse(
        run_id=run_id,
        status="running",
        stages=_build_stub_stages(),
        message=(
            "Legacy status stub. Use /api/v1/run/{run_id}/status for live "
            "orchestrator status."
        ),
    )


# ── Live orchestrator status endpoint ─────────────────────────────────────────

@router.get("/run/{run_id}/status", response_model=RunStatusResponse)
def run_status_route(run_id: str) -> RunStatusResponse:
    """
    Return live pipeline stage statuses from the in-memory RUN_REGISTRY.

    Returns 404 if the run_id is not known (never started or registry cleared).
    """
    if run_id not in RUN_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    entry  = RUN_REGISTRY[run_id]
    result = entry.get("result") or {}

    return RunStatusResponse(
        run_id=run_id,
        status=entry["status"],
        stages=entry["stages"],
        error=entry.get("error"),
        # Spread result fields so the frontend can use the final data
        # without making a separate request after polling detects completion.
        raw_loop_path=result.get("raw_loop_path"),
        upscaled_loop_path=result.get("upscaled_loop_path"),
        seed=result.get("seed"),
        seam_frames_raw=result.get("seam_frames_raw"),
        seam_frames_playable=result.get("seam_frames_playable"),
        gate_result=result.get("gate_result"),
        selected_lut=result.get("selected_lut"),
        lower_third_style=result.get("lower_third_style"),
        metadata_path=result.get("metadata_path"),
    )
