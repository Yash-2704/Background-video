"""
validate_environment.py

Standalone environment validation script for the Background Video Generation Module.
Run with: python validate_environment.py

Checks all required environment conditions against the values locked in
config/environment_constants.json and config/generation_constants.json.
Prints a structured pass/fail/warn report. Exits with code 1 if any FAILs exist.
"""

import json
import subprocess
import sys
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path
from typing import Optional, Tuple

# ── Path resolution ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = BASE_DIR / "config"
ENV_CONSTANTS_PATH = CONFIG_DIR / "environment_constants.json"
GEN_CONSTANTS_PATH = CONFIG_DIR / "generation_constants.json"
OUTPUT_DIR = BASE_DIR / "output"

# ── Required keys (used by check 6) ─────────────────────────────────────────
ENV_REQUIRED_KEYS = [
    "python_version",
    "cuda_version",
    "torch_version",
    "diffusers_version",
    "cogvideox_checkpoint",
    "cogvideox_commit_hash",
    "upscaler",
    "upscaler_model_weights",
    "interpolation_method",
    "temporal_probe_library",
    "ssim_library",
    "ffmpeg_version",
]

GEN_REQUIRED_KEYS = [
    "model",
    "tokenizer",
    "sampler",
    "steps",
    "cfg_scale",
    "native_fps",
    "target_fps",
    "generate_resolution",
    "upscale_target",
    "base_clip_duration_s",
    "base_clip_frames_native",
    "total_anchor_screen_time_s",
    "extensions_per_clip",
    "total_loop_duration_s",
    "playable_duration_s",
    "crossfade_frames",
    "crossfade_duration_s",
    "seam_count",
    "total_crossfade_consumed_s",
    "anchor_position_default",
    "anchor_zone",
    "anchor_feather_px",
    "luminance_reduction",
    "max_regeneration_retries",
    "quality_gates",
    "lut_options",
    "lut_specifications",
    "dev_mode",
    "production_note",
]

COGVIDEOX_UNSET_PLACEHOLDER = "UNSET \u2014 lock at first successful run"

# ── Result accumulator ───────────────────────────────────────────────────────
results = []  # list of (status, label, detail) tuples


def record(status: str, label: str, detail: str) -> None:
    results.append((status, label, detail))


# ── Helper: parse 'package==version' strings ────────────────────────────────
def parse_pinned_version(pinned: str) -> tuple[str, str]:
    """Split 'package==1.2.3' into ('package', '1.2.3')."""
    parts = pinned.split("==", 1)
    return parts[0].strip(), parts[1].strip() if len(parts) == 2 else ("", "")


# ── Load config files (needed by multiple checks) ───────────────────────────
def load_configs() -> Tuple[Optional[dict], Optional[dict]]:
    env_cfg, gen_cfg = None, None
    try:
        with open(ENV_CONSTANTS_PATH, "r", encoding="utf-8") as f:
            env_cfg = json.load(f)
    except FileNotFoundError:
        pass  # handled in check 6
    except json.JSONDecodeError:
        pass  # handled in check 6
    try:
        with open(GEN_CONSTANTS_PATH, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return env_cfg, gen_cfg


# ── CHECK 1: Python version ──────────────────────────────────────────────────
def check_python_version(env_cfg: Optional[dict]) -> None:
    if env_cfg is None:
        record("FAIL", "Python version", "Cannot check — environment_constants.json not loaded")
        return

    required = env_cfg.get("python_version", "")
    major_s, minor_s = required.split(".", 1) if "." in required else ("", "")
    try:
        req_major, req_minor = int(major_s), int(minor_s)
    except ValueError:
        record("FAIL", "Python version", f"Cannot parse required version '{required}'")
        return

    actual = sys.version_info
    actual_str = f"{actual.major}.{actual.minor}.{actual.micro}"

    if actual.major != req_major or actual.minor != req_minor:
        record(
            "FAIL",
            "Python version",
            f"expected {required}.x, found {actual_str}",
        )
    else:
        record("PASS", "Python version", actual_str)


# ── CHECK 2: Package versions ────────────────────────────────────────────────
def check_package_versions(env_cfg: Optional[dict]) -> None:
    if env_cfg is None:
        for label in ("torch", "diffusers", "opencv-python", "scikit-image"):
            record("FAIL", label, "Cannot check — environment_constants.json not loaded")
        return

    checks = [
        ("torch", env_cfg.get("torch_version", ""), "torch"),
        ("diffusers", env_cfg.get("diffusers_version", ""), "diffusers"),
    ]

    # temporal_probe_library and ssim_library are stored as 'pkg==ver'
    probe_pkg, probe_ver = parse_pinned_version(env_cfg.get("temporal_probe_library", ""))
    ssim_pkg, ssim_ver = parse_pinned_version(env_cfg.get("ssim_library", ""))
    checks.append((probe_pkg, probe_ver, probe_pkg))
    checks.append((ssim_pkg, ssim_ver, ssim_pkg))

    for display_name, required_ver, import_name in checks:
        if not required_ver:
            record("FAIL", display_name, "Required version not defined in constants")
            continue
        try:
            installed = pkg_version(import_name)
        except PackageNotFoundError:
            record("FAIL", display_name, "NOT INSTALLED")
            continue

        # opencv-python reports e.g. 4.9.0.80 but constant pins 4.9.0 —
        # treat as prefix match so patch sub-versions don't cause false FAILs.
        if display_name in (probe_pkg,):
            if not installed.startswith(required_ver):
                record(
                    "FAIL",
                    display_name,
                    f"expected {required_ver}.x, found {installed}",
                )
            else:
                record("PASS", display_name, f"{installed} (satisfies {required_ver})")
        else:
            if installed != required_ver:
                record(
                    "FAIL",
                    display_name,
                    f"expected {required_ver}, found {installed}",
                )
            else:
                record("PASS", display_name, installed)


# ── CHECK 3: ffmpeg version ──────────────────────────────────────────────────
def check_ffmpeg(env_cfg: Optional[dict]) -> None:
    required_ver = env_cfg.get("ffmpeg_version", "6.0") if env_cfg else "6.0"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
        # First line is typically: "ffmpeg version 6.0 Copyright ..."
        tokens = first_line.split()
        if len(tokens) >= 3 and tokens[0] == "ffmpeg" and tokens[1] == "version":
            found_ver = tokens[2]
            if found_ver.startswith(required_ver):
                record("PASS", "ffmpeg", f"{found_ver}")
            else:
                record(
                    "FAIL",
                    "ffmpeg",
                    f"expected {required_ver}, found {found_ver}",
                )
        else:
            record("FAIL", "ffmpeg", f"could not parse version from output: {first_line!r}")
    except FileNotFoundError:
        record("WARN", "ffmpeg", "not found on PATH")
    except subprocess.TimeoutExpired:
        record("WARN", "ffmpeg", "timed out running ffmpeg -version")


# ── CHECK 4: CUDA availability ───────────────────────────────────────────────
def check_cuda(env_cfg: Optional[dict]) -> None:
    required_cuda = env_cfg.get("cuda_version", "") if env_cfg else ""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            found_cuda = torch.version.cuda or ""
            if found_cuda == required_cuda:
                record("PASS", "CUDA", f"available, version {found_cuda}")
            else:
                record(
                    "FAIL",
                    "CUDA",
                    f"available but expected {required_cuda}, found {found_cuda}",
                )
        else:
            record("WARN", "CUDA", "not available (CPU-only mode)")
    except ImportError:
        record("WARN", "CUDA", "torch not installed — cannot check CUDA")


# ── CHECK 5: CogVideoX commit hash ───────────────────────────────────────────
def check_cogvideox_hash(env_cfg: Optional[dict]) -> None:
    if env_cfg is None:
        record("FAIL", "CogVideoX commit hash", "Cannot check — environment_constants.json not loaded")
        return

    commit_hash = env_cfg.get("cogvideox_commit_hash", "")
    if commit_hash == COGVIDEOX_UNSET_PLACEHOLDER:
        record(
            "WARN",
            "CogVideoX commit hash",
            "not yet locked. Lock after first successful run.",
        )
    elif commit_hash:
        record("PASS", "CogVideoX commit hash", "set")
    else:
        record("FAIL", "CogVideoX commit hash", "key missing or empty")


# ── CHECK 6: Config files integrity ─────────────────────────────────────────
def check_config_integrity() -> Tuple[Optional[dict], Optional[dict]]:
    env_cfg, gen_cfg = None, None

    # environment_constants.json
    try:
        with open(ENV_CONSTANTS_PATH, "r", encoding="utf-8") as f:
            env_cfg = json.load(f)
        missing = [k for k in ENV_REQUIRED_KEYS if k not in env_cfg]
        if missing:
            record("FAIL", "Config: environment_constants.json", f"missing keys: {missing}")
        else:
            record("PASS", "Config: environment_constants.json", "all keys present")
    except FileNotFoundError:
        record("FAIL", "Config: environment_constants.json", f"file not found at {ENV_CONSTANTS_PATH}")
    except json.JSONDecodeError as exc:
        record("FAIL", "Config: environment_constants.json", f"malformed JSON: {exc}")

    # generation_constants.json
    try:
        with open(GEN_CONSTANTS_PATH, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        missing = [k for k in GEN_REQUIRED_KEYS if k not in gen_cfg]
        if missing:
            record("FAIL", "Config: generation_constants.json", f"missing keys: {missing}")
        else:
            record("PASS", "Config: generation_constants.json", "all keys present")
    except FileNotFoundError:
        record("FAIL", "Config: generation_constants.json", f"file not found at {GEN_CONSTANTS_PATH}")
    except json.JSONDecodeError as exc:
        record("FAIL", "Config: generation_constants.json", f"malformed JSON: {exc}")

    return env_cfg, gen_cfg


# ── CHECK 7: Output directory ────────────────────────────────────────────────
def check_output_directory() -> None:
    if OUTPUT_DIR.exists() and OUTPUT_DIR.is_dir():
        record("PASS", "Output directory", f"exists at {OUTPUT_DIR}")
    else:
        record("WARN", "Output directory", f"missing at {OUTPUT_DIR} (create it before running pipeline)")


# ── Report printer ───────────────────────────────────────────────────────────
def print_report() -> int:
    width = 46
    print()
    print("  \u2554" + "\u2550" * width + "\u2557")
    print("  \u2551   BACKGROUND VIDEO \u2014 ENVIRONMENT CHECK   \u2551")
    print("  \u255a" + "\u2550" * width + "\u255d")
    print()

    passes = fails = warns = 0
    for status, label, detail in results:
        tag = f"[{status}]"
        if status == "PASS":
            passes += 1
        elif status == "FAIL":
            fails += 1
        else:
            warns += 1
        print(f"  {tag:<6} {label}: {detail}")

    print()
    print("  " + "\u2500" * width)
    summary = f"Result: {fails} FAILURE{'S' if fails != 1 else ''} \u00b7 {warns} WARNING{'S' if warns != 1 else ''} \u00b7 {passes} PASSED"
    print(f"  {summary}")
    print("  " + "\u2500" * width)

    if fails > 0:
        print("  Environment is NOT ready. Resolve failures before running pipeline.")
    else:
        print("  Environment is READY. Warnings are advisory only.")
    print()

    return 1 if fails > 0 else 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Check 6 first — we need the loaded configs for all other checks.
    env_cfg, gen_cfg = check_config_integrity()

    check_python_version(env_cfg)
    check_package_versions(env_cfg)
    check_ffmpeg(env_cfg)
    check_cuda(env_cfg)
    check_cogvideox_hash(env_cfg)
    check_output_directory()

    exit_code = print_report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
