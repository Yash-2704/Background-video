"""
tests/test_image_generator.py
──────────────────────────────
Structural tests for core/image_generator.py.

All tests are purely structural — none invoke FLUX, write real images,
or require a GPU. They are designed to pass on the Mac development machine
with no ML packages installed. Run with:

    pytest tests/test_image_generator.py
"""

import inspect
import json
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

# Locate config from the tests/ directory (one level up is project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "generation_constants.json"


# ── Test 1 ─────────────────────────────────────────────────────────────────────

def test_module_imports_cleanly():
    """
    Importing generate_image, download_flux_model, and FLUX_WEIGHTS_PATH
    from core.image_generator must succeed without error, even on a machine
    with no torch or diffusers installed (ML deps are guard-imported).
    """
    from core.image_generator import generate_image, download_flux_model, FLUX_WEIGHTS_PATH  # noqa: F401


# ── Test 2 ─────────────────────────────────────────────────────────────────────

def test_generate_image_is_callable():
    """
    generate_image must be callable and its signature must match exactly:
        (positive: str, run_id: str, output_dir: Path, seed: int) -> Path
    """
    from core.image_generator import generate_image

    assert callable(generate_image), "generate_image is not callable"

    sig = inspect.signature(generate_image)
    params = sig.parameters

    # Check parameter names and order
    assert list(params.keys()) == ["positive", "run_id", "output_dir", "seed"], (
        f"Unexpected parameter list: {list(params.keys())}"
    )

    # Check annotations
    assert params["positive"].annotation  is str,  "positive must be annotated str"
    assert params["run_id"].annotation    is str,  "run_id must be annotated str"
    assert params["output_dir"].annotation is Path, "output_dir must be annotated Path"
    assert params["seed"].annotation      is int,  "seed must be annotated int"

    # Check return annotation
    assert sig.return_annotation is Path, (
        f"Return annotation must be Path, got {sig.return_annotation}"
    )


# ── Test 3 ─────────────────────────────────────────────────────────────────────

def test_flux_weights_path_is_inside_project():
    """
    FLUX_WEIGHTS_PATH must be a Path object whose string representation
    contains 'FLUX.1-schnell'.
    """
    from core.image_generator import FLUX_WEIGHTS_PATH

    assert isinstance(FLUX_WEIGHTS_PATH, Path), (
        f"FLUX_WEIGHTS_PATH must be a Path, got {type(FLUX_WEIGHTS_PATH)}"
    )
    assert "FLUX.1-schnell" in str(FLUX_WEIGHTS_PATH), (
        f"'FLUX.1-schnell' not found in FLUX_WEIGHTS_PATH: {FLUX_WEIGHTS_PATH}"
    )


# ── Test 4 ─────────────────────────────────────────────────────────────────────

def test_prototype_config_keys_exist():
    """
    config/generation_constants.json must contain all three prototype keys
    added for this module.
    """
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    assert "prototype_image_steps" in config, (
        "'prototype_image_steps' key missing from generation_constants.json"
    )
    assert "prototype_image_guidance_scale" in config, (
        "'prototype_image_guidance_scale' key missing from generation_constants.json"
    )
    assert "prototype_output_resolution" in config, (
        "'prototype_output_resolution' key missing from generation_constants.json"
    )


# ── Test 5 ─────────────────────────────────────────────────────────────────────

def test_prototype_output_resolution_divisible_by_32():
    """
    Both dimensions in prototype_output_resolution must be divisible by 32.
    FLUX VAE requires spatial dimensions to be multiples of 32.
    """
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    resolution = config["prototype_output_resolution"]
    width, height = resolution[0], resolution[1]

    assert width % 32 == 0, (
        f"prototype_output_resolution width {width} is not divisible by 32"
    )
    assert height % 32 == 0, (
        f"prototype_output_resolution height {height} is not divisible by 32"
    )


# ── Test 6 ─────────────────────────────────────────────────────────────────────

def test_download_flux_model_is_callable():
    """
    download_flux_model must be callable and must accept zero arguments.
    """
    from core.image_generator import download_flux_model

    assert callable(download_flux_model), "download_flux_model is not callable"

    sig = inspect.signature(download_flux_model)
    assert len(sig.parameters) == 0, (
        f"download_flux_model must take zero arguments, "
        f"got: {list(sig.parameters.keys())}"
    )
