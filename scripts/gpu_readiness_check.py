"""
scripts/gpu_readiness_check.py
───────────────────────────────
GPU environment readiness checker for RTX 4090 deployment.

Run ONCE on the target machine before first live generation:
    python scripts/gpu_readiness_check.py

Prints [PASS] or [FAIL] for each of 14 checks.
Exits with code 0 if all 14 pass, code 1 if any fail.

On a Mac dev machine the GPU checks (1-3) will FAIL — that is expected.
All non-GPU checks must PASS on both Mac and the GPU machine.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── torch: guarded import (never at top level) ─────────────────────────────────
try:
    import torch as _torch
except ImportError:
    _torch = None

# ── diffusers: guarded import ──────────────────────────────────────────────────
try:
    import diffusers as _diffusers
except ImportError:
    _diffusers = None


def _load_env_constants() -> dict:
    path = PROJECT_ROOT / "config" / "environment_constants.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_gen_constants() -> dict:
    path = PROJECT_ROOT / "config" / "generation_constants.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _ffmpeg_version() -> str | None:
    """Return the FFmpeg version string, or None if FFmpeg is not on PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        # e.g. "ffmpeg version 6.1.1 Copyright..."
        parts = first_line.split()
        if len(parts) >= 3:
            return parts[2]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def run_checks() -> int:
    """Run all 14 checks. Returns count of passing checks."""
    results: list[tuple[bool, str]] = []

    env = _load_env_constants()
    gen = _load_gen_constants()

    # 1. CUDA available
    cuda_ok = _torch is not None and _torch.cuda.is_available()
    results.append((cuda_ok, "CUDA available (torch.cuda.is_available())"))

    # 2. CUDA device name contains "4090"
    if cuda_ok:
        device_name = _torch.cuda.get_device_name(0)
        device_ok = "4090" in device_name
    else:
        device_name = "N/A"
        device_ok = False
    results.append((device_ok, f"CUDA device name contains '4090' (found: {device_name!r})"))

    # 3. torch version matches environment_constants.json
    expected_torch = env.get("torch_version", "")
    if _torch is not None and expected_torch:
        torch_ver_ok = _torch.__version__.startswith(expected_torch.lstrip(">=").split(",")[0].strip())
    else:
        torch_ver_ok = False
    actual_torch = _torch.__version__ if _torch is not None else "torch not installed"
    results.append((torch_ver_ok, f"torch version matches environment_constants.json "
                                  f"(expected: {expected_torch!r}, found: {actual_torch!r})"))

    # 4. diffusers importable
    diffusers_ok = _diffusers is not None
    results.append((diffusers_ok, "diffusers importable"))

    # 5. Wan2.2 weights folder exists
    wan_folder = PROJECT_ROOT / "Wan2.2" / "Wan2.2-TI2V-5B-Diffusers"
    results.append((wan_folder.exists(), f"Wan2.2 weights folder exists at {wan_folder}"))

    # 6. Wan2.2 model_index.json exists (confirms full download)
    model_index = wan_folder / "model_index.json"
    results.append((model_index.exists(), f"Wan2.2 model_index.json exists at {model_index}"))

    # 7. Practical-RIFE inference_video.py exists
    rife_script = PROJECT_ROOT / "Practical-RIFE" / "inference_video.py"
    results.append((rife_script.exists(), f"Practical-RIFE/inference_video.py exists at {rife_script}"))

    # 8. realesrgan ncnn-vulkan binary exists
    ncnn_binary = None
    realesrgan_dir = PROJECT_ROOT / "realesrgan"
    if realesrgan_dir.exists():
        candidates = list(realesrgan_dir.rglob("realesrgan-ncnn-vulkan*"))
        if candidates:
            ncnn_binary = candidates[0]
    binary_ok = ncnn_binary is not None
    results.append((binary_ok,
                    f"realesrgan ncnn-vulkan binary exists"
                    + (f" at {ncnn_binary}" if ncnn_binary else " (not found under realesrgan/)")))

    # 9. FFmpeg available and version is 6.x
    ffmpeg_ver = _ffmpeg_version()
    ffmpeg_ok = ffmpeg_ver is not None and ffmpeg_ver.startswith("6.")
    results.append((ffmpeg_ok,
                    f"FFmpeg on PATH and version is 6.x (found: {ffmpeg_ver!r})"))

    # 10. luts/cool_authority.cube exists
    lut_cool = PROJECT_ROOT / "luts" / "cool_authority.cube"
    results.append((lut_cool.exists(), f"luts/cool_authority.cube exists"))

    # 11. luts/neutral.cube exists
    lut_neutral = PROJECT_ROOT / "luts" / "neutral.cube"
    results.append((lut_neutral.exists(), f"luts/neutral.cube exists"))

    # 12. luts/warm_tension.cube exists
    lut_warm = PROJECT_ROOT / "luts" / "warm_tension.cube"
    results.append((lut_warm.exists(), f"luts/warm_tension.cube exists"))

    # 13. generation_constants.json dev_mode is false
    dev_mode = gen.get("dev_mode", True)
    dev_mode_ok = dev_mode is False
    results.append((dev_mode_ok,
                    f"generation_constants.json dev_mode is false (currently: {dev_mode!r})"))

    # 14. output/ directory exists and is writable
    output_dir = PROJECT_ROOT / "output"
    output_ok = output_dir.exists() and output_dir.is_dir()
    if output_ok:
        try:
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError:
            output_ok = False
    results.append((output_ok, f"output/ directory exists and is writable at {output_dir}"))

    # ── Print results ──────────────────────────────────────────────────────────
    passed = 0
    for ok, description in results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {description}")
        if ok:
            passed += 1

    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return passed


def main() -> None:
    passed = run_checks()
    sys.exit(0 if passed == 14 else 1)


if __name__ == "__main__":
    main()
