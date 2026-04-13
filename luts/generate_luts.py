"""
luts/generate_luts.py
─────────────────────
Generates the three broadcast LUT .cube files required by the pipeline.

Run from any directory:
    python luts/generate_luts.py

Output:
    luts/cool_authority.cube
    luts/neutral.cube
    luts/warm_tension.cube

Each file is a valid CUBE v1.0 3D LUT (LUT_3D_SIZE 33, 35937 data lines).

Math applied per lattice point (in order):
    1. Color temperature shift  (RGB channel scale)
    2. Saturation adjustment    (luminance lerp)
    3. Shadow lift              (uniform offset)
    4. Clamp to [0.0, 1.0]
"""

from pathlib import Path

# ── LUT specs (must match generation_constants.json) ────────────────────────────
LUT_SPECS = {
    "cool_authority": {
        "r_scale":               0.92,
        "b_scale":               1.08,
        "saturation_multiplier": 0.85,
        "lift_shadows":         -0.02,
    },
    "neutral": {
        "r_scale":               1.00,
        "b_scale":               1.00,
        "saturation_multiplier": 1.00,
        "lift_shadows":          0.00,
    },
    "warm_tension": {
        "r_scale":               1.06,
        "b_scale":               0.94,
        "saturation_multiplier": 0.90,
        "lift_shadows":         -0.04,
    },
}

LUT_SIZE   = 33
LUTS_DIR   = Path(__file__).resolve().parent


def _apply_lut_math(r: float, g: float, b: float, spec: dict) -> tuple:
    """Apply the 4-step LUT math to a single RGB lattice point."""
    # Step 1: Color temperature shift
    r *= spec["r_scale"]
    b *= spec["b_scale"]

    # Step 2: Saturation (lerp each channel toward luminance)
    sat = spec["saturation_multiplier"]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    r = lum + (r - lum) * sat
    g = lum + (g - lum) * sat
    b = lum + (b - lum) * sat

    # Step 3: Shadow lift
    lift = spec["lift_shadows"]
    r += lift
    g += lift
    b += lift

    # Step 4: Clamp
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))

    return r, g, b


def generate_cube(lut_name: str, spec: dict, output_dir: Path) -> Path:
    """
    Generate a CUBE v1.0 file for the given LUT spec.

    Lattice iteration order follows the CUBE spec:
        R is the innermost (fastest-varying) index,
        G is middle, B is outermost (slowest-varying).
    """
    out_path = output_dir / f"{lut_name}.cube"
    n = LUT_SIZE
    step = 1.0 / (n - 1)

    lines = [
        f'TITLE "{lut_name}"',
        f"LUT_3D_SIZE {n}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]

    # B outermost, G middle, R innermost — CUBE spec §3.3
    for bi in range(n):
        b_in = bi * step
        for gi in range(n):
            g_in = gi * step
            for ri in range(n):
                r_in = ri * step
                ro, go, bo = _apply_lut_math(r_in, g_in, b_in, spec)
                lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    LUTS_DIR.mkdir(parents=True, exist_ok=True)
    expected_data_lines = LUT_SIZE ** 3  # 35937

    for lut_name, spec in LUT_SPECS.items():
        path = generate_cube(lut_name, spec, LUTS_DIR)

        # Verify line count
        all_lines = path.read_text(encoding="utf-8").splitlines()
        data_lines = [ln for ln in all_lines if ln and not ln.startswith(
            ("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")
        )]
        assert len(data_lines) == expected_data_lines, (
            f"{path.name}: expected {expected_data_lines} data lines, "
            f"got {len(data_lines)}"
        )
        print(f"  [OK] {path.name}  ({len(data_lines)} data lines)")

    print("All LUTs generated successfully.")


if __name__ == "__main__":
    main()
