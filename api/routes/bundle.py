"""
api/routes/bundle.py
────────────────────
GET /api/v1/bundle/{run_id}

STUB. Returns a deterministic fake bundle response representing a
complete, quality-passed run.

playable_duration_s is loaded from generation_constants.json — not
hardcoded — so it stays in sync with the generation spec.

seam_frames_playable [159, 332] are derived from the system design
example run, not from constants.json. They represent frame indices at
which the two crossfade seams land in the playable timeline at 30 fps:
  seam 1: ~5.3 s × 30 fps ≈ 159 frames
  seam 2: ~11.1 s × 30 fps ≈ 332 frames

Real bundle is written by Prompt 7 (Post-Processing) and Prompt 8
(Metadata Assembly).

GET /api/v1/bundle/{clip_id}/{filename}
Serves JSON metadata files from output/{clip_id}/{filename}.
Only the four canonical metadata filenames are permitted.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models import BundleFiles, BundleResponse, QualityGates

router = APIRouter()

# Load playable_duration_s from generation_constants.json at startup.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GEN_CONSTANTS = json.loads(
    (_PROJECT_ROOT / "config" / "generation_constants.json").read_text(encoding="utf-8")
)
_PLAYABLE_DURATION_S: float = _GEN_CONSTANTS["playable_duration_s"]

_STUB_QUALITY_GATES = QualityGates(
    mean_luminance=0.46,
    luminance_gate="pass",
    flicker_index=0.003,
    warping_artifact_score=0.018,
    scene_cut_detected=False,
    perceptual_loop_score=0.94,
    overall="pass",
    attempts_used=1,
)


def _build_stub_files(run_id: str) -> BundleFiles:
    return BundleFiles(
        raw_loop=f"raw/{run_id}_raw_loop.mp4",
        upscaled=f"raw/{run_id}_1080p.mp4",
        decode_probe=f"raw/{run_id}_decode_probe.json",
        temporal_probe=f"raw/{run_id}_temporal_probe.json",
        masks=[
            f"masks/{run_id}_mask_center.png",
            f"masks/{run_id}_mask_lower_third.png",
            f"masks/{run_id}_mask_upper_third.png",
        ],
        luts=[
            f"luts/{run_id}_cool_authority.mp4",
            f"luts/{run_id}_neutral.mp4",
        ],
        final=f"final/{run_id}.mp4",
        preview=f"final/{run_id}_preview.gif",
    )


@router.get("/bundle/{run_id}", response_model=BundleResponse)
def bundle_route(run_id: str) -> BundleResponse:
    return BundleResponse(
        run_id=run_id,
        status="complete",
        clip_id=run_id,
        quality_gates=_STUB_QUALITY_GATES,
        files=_build_stub_files(run_id),
        selected_lut="cool_authority",
        lower_third_style="minimal_dark_bar",
        playable_duration_s=_PLAYABLE_DURATION_S,
        seam_frames_playable=[159, 332],
        message=(
            "Bundle stub. Real bundle written by Prompt 7 "
            "(Post-Processing) and Prompt 8 (Metadata Assembly)."
        ),
    )


# ── File-serving endpoint ──────────────────────────────────────────────────────

def _allowed_filenames(clip_id: str) -> frozenset:
    """Return the set of four permitted JSON filenames for a given clip_id."""
    return frozenset({
        f"{clip_id}_metadata.json",
        f"{clip_id}_edit_manifest.json",
        f"{clip_id}_integration_contract.json",
        f"{clip_id}_generation_log.json",
    })


@router.get("/bundle/{clip_id}/{filename}")
def serve_bundle_file(clip_id: str, filename: str) -> FileResponse:
    """
    Serve one of the four JSON metadata files for a completed run.

    Permitted filenames (whitelist):
      {clip_id}_metadata.json
      {clip_id}_edit_manifest.json
      {clip_id}_integration_contract.json
      {clip_id}_generation_log.json

    Files are served from:  output/{clip_id}/{filename}
    Returns 400 if filename is not whitelisted.
    Returns 404 if the file does not exist on disk.
    """
    if filename not in _allowed_filenames(clip_id):
        raise HTTPException(status_code=400, detail="Filename not permitted")

    file_path = _PROJECT_ROOT / "output" / clip_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(path=str(file_path), media_type="application/json")
