"""
api/routes/prototype.py
───────────────────────
Prototype generation endpoint.

Wires the three core pipeline functions together behind a single HTTP route:

    POST /api/v1/prototype/generate

    compile_prompts() → generate_image() → animate_image() → JSON response

Design decisions
----------------
- Synchronous (``def``, not ``async def``) — FLUX inference blocks for ~30 s.
  Wrapping blocking calls in asyncio adds complexity with no benefit for a
  single-GPU, single-user prototype.
- Heavy ML imports (``generate_image``, ``animate_image``) are guard-imported
  *inside* the route function body so the module loads cleanly on any machine
  that does not have torch/diffusers/ffmpeg installed (e.g. the Mac dev box).
- ``compile_prompts`` has no heavy dependencies — imported at module level.
- ``motion_intensity`` arrives as a float (0.0–1.0) in ``PrototypeRequest``
  but ``compile_prompts()`` validates against string labels
  {"Minimal", "Gentle", "Dynamic"}.  The float is mapped to a label *before*
  the compile call, using the same brackets defined in ``_get_motion_params()``.
- All Path objects are converted to ``str`` before being included in the JSON
  response; returning a ``Path`` raises a serialisation error.
"""

import random
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.prompt_compiler import compile_prompts

# ── Project root — resolved relative to this file, never cwd ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Version token — passed to compile_prompts() for provenance tracking ───────
_COMPILER_VERSION = "prototype-1.0"


# ── Request model ─────────────────────────────────────────────────────────────

class PrototypeRequest(BaseModel):
    """
    Six editorial fields that drive the full generation chain.

    ``motion_intensity`` is a normalised float (0.0–1.0).  It is mapped to
    the string label expected by ``compile_prompts()`` inside the endpoint,
    and passed directly as a float to ``animate_image()``.
    """

    category:          str
    location_feel:     str
    time_of_day:       str
    color_temperature: str
    mood:              str
    motion_intensity:  float = Field(..., ge=0.0, le=1.0)


# ── Router ────────────────────────────────────────────────────────────────────
# No prefix here — prefix is set in api/main.py via include_router().

router = APIRouter()


# ── Private helper ────────────────────────────────────────────────────────────

def _float_to_motion_label(intensity: float) -> str:
    """
    Map a 0.0–1.0 float to the string label accepted by compile_prompts().

    Brackets mirror ``_get_motion_params()`` in ``core/animator.py`` so the
    two functions always agree on the intensity tier.

    0.00 – 0.33  →  "Minimal"
    0.33 – 0.66  →  "Gentle"
    0.66 – 1.00  →  "Dynamic"
    """
    if intensity >= 0.66:
        return "Dynamic"
    elif intensity >= 0.33:
        return "Gentle"
    else:
        return "Minimal"


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/generate")
def generate(body: PrototypeRequest) -> dict:
    """
    Run the full prototype generation chain for a single clip.

    Pipeline
    --------
    1. Map ``motion_intensity`` float → string label for ``compile_prompts()``.
    2. ``compile_prompts()``  — assemble the positive prompt and validate inputs.
    3. ``generate_image()``   — run FLUX.1-schnell (~30 s); write image.png.
    4. ``animate_image()``    — run FFmpeg zoompan (~5 s); write animated.mp4.
    5. Return relative paths and prompt string as JSON.

    Blocks until animated.mp4 is written — no background tasks, no job queue.

    Returns
    -------
    dict
        {
            "run_id":      "<8-char hex>",
            "image_path":  "output/prototype/<run_id>/image.png",
            "video_path":  "output/prototype/<run_id>/animated.mp4",
            "prompt_used": "<compiled positive prompt string>"
        }

    Raises
    ------
    HTTPException(500)
        On any exception in the pipeline chain.
    """
    # ── Guard-import ML/FFmpeg dependencies ───────────────────────────────────
    # Imported here, not at module level, so the FastAPI app starts cleanly on
    # machines without torch/diffusers/ffmpeg (e.g. the Mac dev box).
    from core.image_generator import generate_image
    from core.animator import animate_image

    try:
        # ── 1. Generate run identifiers ───────────────────────────────────────
        run_id: str = uuid.uuid4().hex[:8]
        seed:   int = random.randint(0, 2**32 - 1)

        # ── 2. Resolve output directory ───────────────────────────────────────
        # Always PROJECT_ROOT / "output" so .relative_to(PROJECT_ROOT) never
        # raises ValueError (both paths share the same root).
        output_dir: Path = PROJECT_ROOT / "output"

        # ── 3. Build user_input dict for compile_prompts() ────────────────────
        # compile_prompts() validates motion_intensity as a string label
        # {"Minimal", "Gentle", "Dynamic"}, not as a float.
        # We map the float now and keep the original float for animate_image().
        motion_label: str   = _float_to_motion_label(body.motion_intensity)
        motion_float: float = body.motion_intensity

        user_input: dict = body.model_dump()
        user_input["motion_intensity"] = motion_label  # swap float → label

        # ── 4. Compile prompts ────────────────────────────────────────────────
        compiled: dict = compile_prompts(user_input, _COMPILER_VERSION)
        positive: str  = compiled["positive"]

        # ── 5. Generate image ─────────────────────────────────────────────────
        image_path: Path = generate_image(positive, run_id, output_dir, seed)

        # ── 6. Animate image ──────────────────────────────────────────────────
        video_path: Path = animate_image(image_path, motion_float, run_id)

        # ── 7. Build and return response ──────────────────────────────────────
        # Convert Path objects to str; returning Path raises a JSON
        # serialisation error.  Use .relative_to(PROJECT_ROOT) for clean,
        # portable paths in the response body.
        return {
            "run_id":      run_id,
            "image_path":  str(image_path.relative_to(PROJECT_ROOT)),
            "video_path":  str(video_path.relative_to(PROJECT_ROOT)),
            "prompt_used": positive,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
