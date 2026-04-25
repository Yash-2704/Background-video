"""
core/animator.py
────────────────
FFmpeg-based still-image animator for the prototype preview chain.

Responsibilities:
  1. animate_image()       — take the PNG written by generate_image() and
                             produce a 5-second MP4 at the same path with
                             a slow Ken-Burns (zoompan) motion effect.
  2. _get_motion_params()  — map a 0.0–1.0 motion_intensity float to
                             concrete FFmpeg zoompan speed parameters.

Pipeline position:
  compile_prompts() → generate_image() → animate_image() → MP4 out

No GPU is used. All work is CPU/FFmpeg. FFmpeg must be on PATH.
Designed for: Windows, RTX 4090 host, Python 3.11, FFmpeg ≥ 5.x.

FFmpeg is always invoked via subprocess.run() with a Python list of args
so that multiline shell commands are never required over Windows SSH.
"""

import json
import subprocess
from pathlib import Path

# ── Project root — resolved relative to this file, never cwd ─────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "generation_constants.json"

# ── Module-level constants ────────────────────────────────────────────────────
PROTOTYPE_DURATION_SECONDS: int = 5  # seconds — hardcoded for prototype previews


# ── Private helpers ──────────────────────────────────────────────────────────

def _get_motion_params(motion_intensity: float) -> dict:
    """
    Map a normalised motion intensity to FFmpeg zoompan speed parameters.

    Parameters
    ----------
    motion_intensity : float
        Value in the range 0.0–1.0, as produced by the upstream pipeline.
        Anything below 0.33 is treated as low; 0.33–0.66 as mid; above as high.

    Returns
    -------
    dict
        ``{"zoom_speed": float, "pan_speed": float}``

        zoom_speed  — increment added to the zoom factor each frame
                      (used as: ``z='min(zoom+{zoom_speed},1.5)'``)
        pan_speed   — pixels of x/y drift per frame
                      (used as: ``x='...+{pan_speed}*on'``)

    Motion intensity brackets
    -------------------------
    Low  (0.00–0.33)  →  zoom_speed=0.0005, pan_speed=0.3
    Mid  (0.33–0.66)  →  zoom_speed=0.001,  pan_speed=0.6
    High (0.66–1.00)  →  zoom_speed=0.002,  pan_speed=1.2
    """
    if motion_intensity >= 0.66:
        return {"zoom_speed": 0.002, "pan_speed": 1.2}
    elif motion_intensity >= 0.33:
        return {"zoom_speed": 0.001, "pan_speed": 0.6}
    else:
        return {"zoom_speed": 0.0005, "pan_speed": 0.3}


# ── Public API ───────────────────────────────────────────────────────────────

def animate_image(
    image_path: Path,
    motion_intensity: float,
    run_id: str,
) -> Path:
    """
    Animate a static PNG with a Ken-Burns zoompan effect and write an MP4.

    Reads ``native_fps`` from ``config/generation_constants.json`` at call
    time so that any config update is picked up without restarting the server.

    Parameters
    ----------
    image_path : Path
        Absolute path to the source PNG produced by ``generate_image()``.
        Expected location: ``output/prototype/{run_id}/image.png``.
    motion_intensity : float
        Normalised motion intensity in 0.0–1.0.  Sourced from the compiled
        dict returned by ``compile_prompts()``.  Controls zoom and pan speed.
    run_id : str
        Unique run identifier — used for logging context; the output path is
        derived entirely from ``image_path``, not from ``run_id`` directly.

    Returns
    -------
    Path
        Absolute path to the written MP4 file:
        ``output/prototype/{run_id}/animated.mp4``
        (always a sibling of the input PNG).

    Raises
    ------
    subprocess.CalledProcessError
        If FFmpeg exits with a non-zero return code.
    FileNotFoundError
        If FFmpeg is not found on PATH (surfaces as CalledProcessError or
        FileNotFoundError depending on OS).
    """
    # ── 1. Resolve output path ────────────────────────────────────────────────
    # Output is always a sibling of the input PNG in the same run directory.
    output_path: Path = image_path.parent / "animated.mp4"

    # ── 2. Load native_fps from config at call time ───────────────────────────
    # Loading at call time (not module level) means an in-flight config edit
    # takes effect on the next call without a server restart.
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        constants = json.load(fh)
    native_fps: int = constants["native_fps"]

    # ── 3. Resolve motion parameters ─────────────────────────────────────────
    motion_params = _get_motion_params(motion_intensity)
    zoom_speed: float = motion_params["zoom_speed"]
    pan_speed:  float = motion_params["pan_speed"]

    # ── 4. Compute total frame count ──────────────────────────────────────────
    # Only used to inform the zoompan filter duration implicitly via -t;
    # kept explicit here for clarity and future logging.
    total_frames: int = int(native_fps * PROTOTYPE_DURATION_SECONDS)  # e.g. 120

    # ── 5. Build the zoompan filter string ────────────────────────────────────
    # Rules:
    #   z    — starts at 1.0, increments by zoom_speed each frame, capped at 1.5
    #   x/y  — centred, then drifted by pan_speed pixels per output frame (on)
    #   d=1  — one output frame per input frame; avoids duplication artifacts
    #          on still-image input
    #   s    — must match output resolution exactly (1280x736)
    zoompan_filter: str = (
        f"zoompan="
        f"z='min(zoom+{zoom_speed},1.5)':"
        f"x='iw/2-(iw/zoom/2)+{pan_speed}*on':"
        f"y='ih/2-(ih/zoom/2)+{pan_speed}*on':"
        f"d=1:"
        f"s=1280x736"
    )

    # ── 6. Build the full FFmpeg argument list ────────────────────────────────
    # arg list, not a shell string — safe over Windows SSH where multiline
    # shell commands fail.
    #
    # Flag notes:
    #   -y            overwrite output without prompting
    #   -loop 1       loop the single input PNG to feed frames into the filter
    #   -i            input file
    #   -vf           video filter chain (zoompan only — no audio filter needed)
    #   -c:v libx264  CPU software encoder (no GPU encoding)
    #   -pix_fmt yuv420p  broadcast-safe pixel format; required by many players
    #   -r            output frame rate
    #   -t 5          encode exactly 5 seconds (do NOT combine with -vframes)
    ffmpeg_args = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", zoompan_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(native_fps),
        "-t", str(PROTOTYPE_DURATION_SECONDS),
        str(output_path),
    ]

    # ── 7. Execute FFmpeg ─────────────────────────────────────────────────────
    # check=True raises CalledProcessError on non-zero exit — never silently
    # produces a corrupt file.
    subprocess.run(ffmpeg_args, check=True)

    # ── 8. Return output path ─────────────────────────────────────────────────
    return output_path
