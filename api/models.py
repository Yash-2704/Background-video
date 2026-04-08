"""
api/models.py
─────────────
All Pydantic v2 request and response models for the Background Video
Generation API. No models are defined anywhere else in the API layer.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


# ── Request models ─────────────────────────────────────────────────────────────

class EditorialInput(BaseModel):
    category:          Literal["Economy", "Politics", "Tech", "Climate",
                                "Crime", "Sports", "General"]
    location_feel:     Literal["Urban", "Government", "Nature",
                                "Abstract", "Data"]
    time_of_day:       Literal["Day", "Dusk", "Night", "N/A"]
    color_temperature: Literal["Cool", "Neutral", "Warm"]
    mood:              Literal["Serious", "Tense", "Neutral", "Uplifting"]
    motion_intensity:  Literal["Minimal", "Gentle", "Dynamic"]


class GenerateRequest(BaseModel):
    run_id: Optional[str] = None
    compiled: Optional[dict] = None
    editorial_input: Optional[EditorialInput] = None
    dry_run: bool = False


# ── Response models ────────────────────────────────────────────────────────────

class CompileResponse(BaseModel):
    positive:          str
    motion:            str
    negative:          str
    positive_hash:     str
    motion_hash:       str
    negative_hash:     str
    input_hash_short:  str
    selected_lut:      str
    lower_third_style: str
    compiler_version:  str
    user_input:        EditorialInput


class PipelineStage(BaseModel):
    # Used inside GenerateResponse and StatusResponse
    stage:      str
    status:     Literal["pending", "running", "complete", "failed", "skipped"]
    message:    str
    started_at: Optional[str]  # ISO 8601 UTC, None if not started
    ended_at:   Optional[str]  # ISO 8601 UTC, None if not ended


class GenerateResponse(BaseModel):
    run_id:               str
    status:               Literal["queued", "running", "complete", "failed", "escalated"]
    raw_loop_path:        Optional[str] = None
    seed:                 Optional[int] = None
    seam_frames_raw:      Optional[list[int]] = None
    seam_frames_playable: Optional[list[int]] = None
    gate_result:          Optional[dict] = None
    selected_lut:         Optional[str] = None
    lower_third_style:    Optional[str] = None
    metadata_path:        Optional[str] = None
    stages:               list[PipelineStage] | dict[str, str]
    failure_log:          Optional[list] = None
    error:                Optional[str] = None
    dry_run:              Optional[bool] = None
    editorial_input:      Optional[EditorialInput] = None
    compiled:             Optional[CompileResponse] = None
    message:              Optional[str] = None


class QualityGates(BaseModel):
    mean_luminance:         float
    luminance_gate:         Literal["pass", "fail"]
    flicker_index:          float
    warping_artifact_score: float
    scene_cut_detected:     bool
    perceptual_loop_score:  float
    overall:                Literal["pass", "fail", "human_review"]
    attempts_used:          int


class BundleFiles(BaseModel):
    raw_loop:       str
    upscaled:       str
    decode_probe:   str
    temporal_probe: str
    masks:          list[str]
    luts:           list[str]
    final:          str
    preview:        str


class StatusResponse(BaseModel):
    run_id:  str
    status:  Literal["queued", "running", "complete", "failed", "escalated"]
    stages:  list[PipelineStage]
    message: str


class RunStatusResponse(BaseModel):
    """Returned by GET /api/v1/run/{run_id}/status — live stage data from RUN_REGISTRY."""
    run_id:  str
    status:  str
    stages:  dict[str, str]
    error:   Optional[str] = None


class BundleResponse(BaseModel):
    run_id:               str
    status:               Literal["complete", "failed", "escalated"]
    clip_id:              str
    quality_gates:        QualityGates
    files:                BundleFiles
    selected_lut:         str
    lower_third_style:    str
    playable_duration_s:  float
    seam_frames_playable: list[int]
    message:              str


class HealthResponse(BaseModel):
    status:           Literal["ok"]
    module:           str   # "bg_video"
    module_version:   str   # "1.1"
    dev_mode:         bool
    compiler_version: str


class ValidInputsResponse(BaseModel):
    category:          list[str]
    location_feel:     list[str]
    time_of_day:       list[str]
    color_temperature: list[str]
    mood:              list[str]
    motion_intensity:  list[str]
