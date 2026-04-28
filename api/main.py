"""
api/main.py
───────────
FastAPI application entry point for the Background Video Generation API.

Start with:
  uvicorn api.main:app --reload

CORS is open to all origins for dev. Lock this down before production.
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.models import HealthResponse, ValidInputsResponse
from api.routes import bundle, compile, generate, parse, prototype, status, upload
from core.prompt_compiler import get_all_valid_inputs

# ── Config ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GEN_CONSTANTS = json.loads(
    (_PROJECT_ROOT / "config" / "generation_constants.json").read_text(encoding="utf-8")
)
COMPILER_VERSION = "1.0.0"

_UPLOADS_DIR = _PROJECT_ROOT / "output" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Background Video Generation API",
    version="1.1",
    description=(
        "Broadcast background video pipeline API. "
        "Dev phase — generation stubs active."
    ),
)

# ── CORS — dev only, open to all origins ─────────────────────────────────────
# TODO: restrict allow_origins to specific frontend URL before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
_PREFIX = "/api/v1"
app.include_router(compile.router, prefix=_PREFIX)
app.include_router(parse.router, prefix=_PREFIX)
app.include_router(generate.router, prefix=_PREFIX)
app.include_router(status.router, prefix=_PREFIX)
app.include_router(bundle.router, prefix=_PREFIX)
app.include_router(upload.router, prefix=_PREFIX)
app.include_router(prototype.router, prefix=f"{_PREFIX}/prototype")


# ── Standalone routes ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/api/v1/health")


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # This route must NEVER fail if the server is running.
    return HealthResponse(
        status="ok",
        module="bg_video",
        module_version="1.1",
        compiler_version=COMPILER_VERSION,
    )


@app.get("/api/v1/inputs", response_model=ValidInputsResponse)
def inputs() -> ValidInputsResponse:
    # Never hardcoded — always reflects the live state of lookup tables.
    return ValidInputsResponse(**get_all_valid_inputs())
