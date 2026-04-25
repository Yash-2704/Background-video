"""
core/image_generator.py
────────────────────────
Prototype image generation using FLUX.1-schnell.

Responsibilities:
  1. generate_image()       — produce a single PNG from a positive prompt
                              using FLUX.1-schnell at 1280×736 in bfloat16.
  2. download_flux_model()  — one-shot helper to pull model weights from
                              HuggingFace Hub into Wan2.2/FLUX.1-schnell/.
                              Call once manually; never called by generate_image().

Live mode only — no dev_mode branching. This module is designed exclusively
for the GPU machine (RTX 4090, 24 GB VRAM, Windows, Python 3.11).

Device strategy:
  - low_cpu_mem_usage=False  prevents AccelerateDeviceHook from silently
    overriding .to("cuda") and routing compute through CPU RAM.
  - .to("cuda") moves all pipeline components to GPU in one step.
  - enable_model_cpu_offload() is never used — it routes compute through
    CPU RAM (~16s/step) and must not be called.

VRAM budget for FLUX.1-schnell at bfloat16:
  transformer (~12 GB) + text encoders (~3 GB) + VAE (~1 GB) ≈ 16 GB
  Fits within the 24 GB RTX 4090 with no sub-module offloading required.
"""

import subprocess
from pathlib import Path

# ── Project root (resolved relative to this file, not cwd) ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Model weights path ─────────────────────────────────────────────────────────
# Must be populated by calling download_flux_model() once before first use.
FLUX_WEIGHTS_PATH = PROJECT_ROOT / "Wan2.2" / "FLUX.1-schnell"


# ── download_flux_model() ──────────────────────────────────────────────────────

def download_flux_model() -> None:
    """
    Pull FLUX.1-schnell weights from HuggingFace Hub into FLUX_WEIGHTS_PATH.

    Calls:
        huggingface-cli download black-forest-labs/FLUX.1-schnell
            --local-dir <FLUX_WEIGHTS_PATH>

    Prerequisites:
      - huggingface_hub must be installed (pip install huggingface_hub).
      - HF_TOKEN env var must be set if the repo requires authentication.
        FLUX.1-schnell is gated — you must accept the licence on HF first.

    This function is called once manually from a terminal, not inside
    generate_image(). Running it again is safe (HF CLI skips cached blobs).

    Raises:
      subprocess.CalledProcessError  if huggingface-cli exits non-zero.
    """
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "black-forest-labs/FLUX.1-schnell",
            "--local-dir",
            str(FLUX_WEIGHTS_PATH),
        ],
        check=True,
    )


# ── generate_image() ───────────────────────────────────────────────────────────

def generate_image(
    positive: str,
    run_id: str,
    output_dir: Path,
    seed: int,
) -> Path:
    """
    Generate a single 1280×736 PNG using FLUX.1-schnell.

    Parameters
    ----------
    positive : str
        The positive image-generation prompt produced by compile_prompts()
        (the ``compiled["positive"]`` field).
    run_id : str
        Unique identifier for this generation run (e.g. "run_abc123").
        Used to build the output sub-directory.
    output_dir : Path
        Root output directory (typically the project ``output/`` folder).
        The image is written to ``output_dir / "prototype" / run_id / "image.png"``.
    seed : int
        RNG seed for reproducible generation. Passed to a CUDA torch.Generator.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.

    Notes
    -----
    - Requires: torch, diffusers >= 0.28 (FluxPipeline), and an RTX 4090.
    - FLUX.1-schnell is a distilled model: 4 steps, guidance_scale=0.0.
    - VRAM is freed via ``del pipe`` + ``torch.cuda.empty_cache()`` before
      returning, so the caller can load another model immediately.
    - Output resolution 1280×736 is required (both dimensions divisible by 32).
      Do NOT use 720 — FLUX VAE requires multiples of 32.
    """
    # ── 1. Resolve output path and create directory ───────────────────────────
    image_path: Path = output_dir / "prototype" / run_id / "image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 2. Guard-import ML dependencies ───────────────────────────────────────
    # Imported here (not at module level) so the module loads cleanly on any
    # machine without torch/diffusers installed — matching the pattern used
    # throughout core/generator.py.
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as exc:
        raise RuntimeError(
            "torch and/or diffusers are not installed. "
            "Cannot run generate_image() without ML dependencies."
        ) from exc

    # ── 3. Load FLUX.1-schnell pipeline ───────────────────────────────────────
    # low_cpu_mem_usage=False: diffusers default True attaches AlignDevicesHook
    # which silently overrides .to("cuda") and routes compute through CPU RAM.
    # Always pass False to guarantee clean GPU placement.
    pipe = FluxPipeline.from_pretrained(
        str(FLUX_WEIGHTS_PATH),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    # ── 4. Move all pipeline components to CUDA ───────────────────────────────
    pipe.to("cuda")

    # ── 5. Run inference ──────────────────────────────────────────────────────
    # FLUX.1-schnell is a distilled (guidance-distilled) model:
    #   num_inference_steps=4  — correct for schnell; more steps waste time.
    #   guidance_scale=0.0     — CFG is disabled for the schnell variant.
    generator = torch.Generator("cuda").manual_seed(seed)

    result = pipe(
        prompt=positive,
        width=1280,
        height=736,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    )

    # ── 6. Save output image ──────────────────────────────────────────────────
    # result.images[0] is a PIL.Image in RGB mode.
    pil_image = result.images[0]
    pil_image.save(str(image_path), format="PNG")

    # ── 7. Free VRAM before returning ─────────────────────────────────────────
    # Explicit deletion + cache flush ensures the next pipeline (e.g. WanPipeline)
    # can load without hitting the 24 GB VRAM ceiling.
    del pipe
    torch.cuda.empty_cache()

    return image_path
