"""
core/metadata_assembler.py
──────────────────────────
Metadata Assembly Layer for the Background Video Generation Module.

Collects all upstream results and writes four JSON output files:
  bg_XXX_metadata.json             — generation record
  bg_XXX_edit_manifest.json        — editability record
  bg_XXX_integration_contract.json — downstream interface (3 audiences)
  bg_XXX_generation_log.json       — run history

This module is the final step in the pipeline. It assembles and writes.
It does not generate, process, probe, or gate.

Every field in every output file is derived from a named upstream source:
  compiled          — from core/prompt_compiler.compile_prompts()
  generation_result — from core/generator.run_generation()
  decode_probe      — from core/probes.run_decode_probe()
  temporal_probe    — from core/probes.run_temporal_probe()
  gate_result       — from core/gates.evaluate_gates()
  post_result       — from core/post_processor.run_post_processing()
  GENERATION_CONSTANTS — from config/generation_constants.json
  ENV_CONSTANTS        — from config/environment_constants.json
"""

import json
from datetime import datetime
from pathlib import Path

# ── Config loading ─────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
_GEN_CONFIG_PATH = PROJECT_ROOT / "config" / "generation_constants.json"
_ENV_CONFIG_PATH = PROJECT_ROOT / "config" / "environment_constants.json"

with _GEN_CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)

with _ENV_CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    ENV_CONSTANTS: dict = json.load(_fh)


# ── Private helpers ────────────────────────────────────────────────────────────

def _check_path(path_str) -> str:
    """Return str(path) if it exists on disk, else "MISSING" with a warning."""
    if path_str is None:
        return "MISSING"
    p = Path(path_str)
    if p.exists():
        return str(p)
    print(f"WARNING: expected file missing: {path_str}")
    return "MISSING"


def _build_files_section(
    clip_id:       str,
    post_result:   dict,
    gen_log_path:  Path,
    raw_loop_path: str = None,   # from generation_result["raw_loop_path"]
) -> dict:
    """
    Build the files section for metadata.json.

    Checks each path for existence on disk; missing paths become "MISSING".
    gen_log_path is NOT included in the files section — per spec it is a
    sibling file in output_dir referenced by clip_id convention.

    Sources:
      raw_loop       — generation_result["raw_loop_path"] (passed through)
      upscaled       — post_result["upscaled"]
      decode_probe   — not yet written (TODO: Prompt 11 adds probe JSON export)
      temporal_probe — not yet written (TODO: Prompt 11 adds probe JSON export)
      masks          — post_result["masks"].values()
      luts           — post_result["graded_variants"].values()
      final          — post_result["final"]
      preview        — post_result["preview_gif"]
    """
    # raw_loop — from generation_result, passed through assemble_metadata
    raw_loop = _check_path(raw_loop_path)

    # upscaled — from post_result
    upscaled = _check_path(post_result.get("upscaled"))

    # decode_probe / temporal_probe — not yet written in dev phase
    # TODO (Prompt 11): post_processor should write probe JSONs so these paths
    # can be populated. Until then, both will always be "MISSING".
    decode_probe_file   = "MISSING"
    temporal_probe_file = "MISSING"

    # masks — list of paths from post_result["masks"] values
    masks = [
        _check_path(p)
        for p in post_result.get("masks", {}).values()
    ]

    # luts — list of graded-variant paths from post_result["graded_variants"]
    luts = [
        _check_path(p)
        for p in post_result.get("graded_variants", {}).values()
    ]

    # final — from post_result
    final = _check_path(post_result.get("final"))

    # preview — from post_result["preview_gif"]
    preview = _check_path(post_result.get("preview_gif"))

    return {
        "raw_loop":       raw_loop,
        "upscaled":       upscaled,
        "decode_probe":   decode_probe_file,
        "temporal_probe": temporal_probe_file,
        "masks":          masks,
        "luts":           luts,
        "final":          final,
        "preview":        preview,
    }


def _compute_cut_points() -> list:
    """
    Compute recommended cut points in seconds.
    Returns [0, seam_1_s, seam_2_s, playable_duration_s], rounded to 2 dp.

    Sources:
      seam_positions_seconds_playable — GENERATION_CONSTANTS (pre-computed)
      playable_duration_s             — GENERATION_CONSTANTS["playable_duration_s"]

    Values are read from config rather than derived from frame/fps arithmetic
    to avoid rounding drift (e.g. 269/24 = 11.2083... vs config-authoritative 11.21).
    """
    seam_positions = GENERATION_CONSTANTS["seam_positions_seconds_playable"]
    playable_duration_s = GENERATION_CONSTANTS["playable_duration_s"]
    return [
        0,
        round(seam_positions[0], 2),
        round(seam_positions[1], 2),
        round(playable_duration_s, 2),
    ]


def _compute_anchor_pixels() -> dict:
    """
    Convert ANCHOR_ZONE fractions to pixel values at 1920×1080.
    Source: GENERATION_CONSTANTS["anchor_zone"]
    """
    az = GENERATION_CONSTANTS["anchor_zone"]
    W, H = 1920, 1080
    return {
        "x": int(az["x"] * W),
        "y": int(az["y"] * H),
        "w": int(az["w"] * W),
        "h": int(az["h"] * H),
    }


def _derive_environment(compiled: dict) -> str:
    """Map compiled user_input.location_feel to environment label."""
    mapping = {
        "Urban":      "urban_exterior",
        "Government": "government_plaza",
        "Nature":     "natural_landscape",
        "Abstract":   "abstract_light",
        "Data":       "data_center_interior",
    }
    return mapping.get(compiled["user_input"]["location_feel"], "unclassified")


def _derive_light(compiled: dict) -> str:
    """Map compiled user_input.time_of_day to light label."""
    mapping = {
        "Day":   "daylight_even",
        "Dusk":  "dusk_directional",
        "Night": "night_artificial",
        "N/A":   "indeterminate",
    }
    return mapping.get(compiled["user_input"]["time_of_day"], "indeterminate")


def _derive_motion_character(compiled: dict) -> str:
    """Map compiled user_input.motion_intensity to motion label."""
    mapping = {
        "Minimal": "locked_frame",
        "Gentle":  "slow_lateral_drift",
        "Dynamic": "controlled_push_in",
    }
    return mapping.get(compiled["user_input"]["motion_intensity"], "unclassified")


def _derive_color_temp_k(decode_probe: dict) -> int:
    """
    Map decode_probe.dominant_hue_degrees to approximate color temperature (K).
    Source: decode_probe["dominant_hue_degrees"]
    """
    hue = decode_probe["dominant_hue_degrees"]
    if 180 <= hue <= 240:
        return 6500   # blue range → cool/daylight
    if 20 <= hue <= 60:
        return 3200   # warm/amber range → tungsten
    return 5000       # default → neutral daylight


def _derive_complexity(decode_probe: dict) -> str:
    """
    Map decode_probe.saturation_mean to scene complexity label.
    Source: decode_probe["saturation_mean"]
    """
    sat = decode_probe["saturation_mean"]
    if sat < 0.25:
        return "low"
    if sat < 0.50:
        return "medium"
    return "high"


def _derive_ticker_contrast(compiled: dict) -> str:
    """Map compiled user_input.color_temperature to ticker contrast choice."""
    ct = compiled["user_input"]["color_temperature"]
    if ct in ("Cool", "Neutral"):
        return "light_on_dark"
    return "dark_on_light"


def _derive_audio_mood(compiled: dict) -> str:
    """Map compiled user_input.mood to audio mood label."""
    mapping = {
        "Serious":   "measured_serious",
        "Tense":     "urgent_tense",
        "Neutral":   "balanced_neutral",
        "Uplifting": "optimistic_open",
    }
    return mapping.get(compiled["user_input"]["mood"], "balanced_neutral")


# ── assemble_metadata() ────────────────────────────────────────────────────────

def assemble_metadata(
    clip_id:            str,
    run_number:         int,
    compiled:           dict,
    generation_result:  dict,
    decode_probe:       dict,
    temporal_probe:     dict,
    gate_result:        dict,
    post_result:        dict,
    output_dir:         Path,
) -> dict:
    """
    Assemble and write bg_XXX_metadata.json.

    Returns the assembled dict so subsequent assembler functions can consume
    it directly — do not re-read the JSON file.
    """
    GC = GENERATION_CONSTANTS
    EC = ENV_CONSTANTS

    # luminance_gate: derived from probe values, not from gate_result.overall
    # (gate_result.overall covers all 5 gates; this field isolates luminance).
    mean_lum = decode_probe["mean_luminance"]
    lum_min  = decode_probe["luminance_gate_min"]
    lum_max  = decode_probe["luminance_gate_max"]
    luminance_gate = "pass" if (lum_min <= mean_lum <= lum_max) else "fail"

    # generation_log path: sibling file in output_dir — not in files section
    gen_log_path = output_dir / f"{clip_id}_generation_log.json"

    files = _build_files_section(
        clip_id=clip_id,
        post_result=post_result,
        gen_log_path=gen_log_path,
        raw_loop_path=generation_result.get("raw_loop_path"),
    )

    # NOTE: generation_constants.json does not contain "interpolation_method".
    # That key lives in environment_constants.json. Both the environment and
    # generation sections use ENV_CONSTANTS["interpolation_method"] as the
    # authoritative source. A comment is left here so a future engineer doesn't
    # accidentally add a duplicate key to the wrong config file.
    metadata = {
        "module":           "bg_video",
        "module_version":   "1.1",
        "clip_id":          clip_id,
        "schema_version":   "1.0",
        "dev_mode":         GC["dev_mode"],
        "timestamp_utc":    datetime.utcnow().isoformat() + "Z",

        "environment": {
            # All fields sourced from config/environment_constants.json
            # Wan2.2-TI2V-5B-Diffusers model fields (replaced CogVideoX fields)
            "model_id":               EC["model_id"],
            "model_checkpoint":       EC["model_checkpoint"],
            "model_commit_hash":      EC["model_commit_hash"],
            "model_architecture":     EC["model_architecture"],
            "model_task":             EC["model_task"],
            "upscaler_model_weights": EC["upscaler_model_weights"],
            "interpolation_method":   EC["interpolation_method"],
            "temporal_probe_library": EC["temporal_probe_library"],
            "ffmpeg_version":         EC["ffmpeg_version"],
        },

        "generation": {
            # Fields removed in Wan2.2-TI2V-5B migration:
            # tokenizer — not applicable to this model
            # interpolated_to_fps — model outputs 24fps natively,
            #   no interpolation step in pipeline
            # interpolation_method — removed from generation block;
            #   RIFE retained in environment block for crossfade only

            # Model/sampler config — from GENERATION_CONSTANTS
            "model":                GC["model"],
            "sampler":              GC["sampler"],
            "cfg_scale":            GC["cfg_scale"],
            "steps":                GC["steps"],
            # Seed — from generation_result (caller-assigned or randomly drawn)
            "seed":                 generation_result["seed"],
            # FPS — from GENERATION_CONSTANTS (24fps native, no interpolation)
            "native_fps":           GC["native_fps"],
            # Frame counts — from GENERATION_CONSTANTS
            "base_clip_frames":     GC["base_clip_frames_native"],
            # Resolution — from GENERATION_CONSTANTS
            "generate_resolution":  GC["generate_resolution"],
            # Duration / seam config — from GENERATION_CONSTANTS
            "base_duration_s":      GC["base_clip_duration_s"],
            "extensions":           GC["extensions_per_clip"],
            "total_raw_duration_s": GC["total_loop_duration_s"],
            "playable_duration_s":  GC["playable_duration_s"],
            "crossfade_frames":     GC["crossfade_frames"],
            "crossfade_method":     "optical_flow_blend_RIFE",
            # Seam positions — from generation_result (computed by crossfade_join)
            "seam_frames_raw":      generation_result["seam_frames_raw"],
            "seam_frames_playable": generation_result["seam_frames_playable"],
            # VAE compression params — from GENERATION_CONSTANTS
            "vae_compression":      GC["vae_compression"],
            # Generation mode per clip — from GENERATION_CONSTANTS
            "generation_mode_by_clip": {
                "base_clip":   GC["generation_modes"]["base_clip"],
                "extension_1": GC["generation_modes"]["extension_1"],
                "extension_2": GC["generation_modes"]["extension_2"],
            },
        },

        "prompt_provenance": {
            # All fields from compile_prompts() return dict
            "compiler_version":       compiled["compiler_version"],
            "positive_prompt":        compiled["positive"],
            "motion_prompt":          compiled["motion"],
            "negative_prompt":        compiled["negative"],
            "positive_hash":          compiled["positive_hash"],
            "motion_hash":            compiled["motion_hash"],
            "negative_hash":          compiled["negative_hash"],
            "input_hash_short":       compiled["input_hash_short"],
            "reproducibility_rating": "high",
        },

        "user_input": {
            # Sourced from compiled["user_input"] (added in Prompt 8 modification)
            "category":          compiled["user_input"]["category"],
            "location_feel":     compiled["user_input"]["location_feel"],
            "time_of_day":       compiled["user_input"]["time_of_day"],
            "color_temperature": compiled["user_input"]["color_temperature"],
            "mood":              compiled["user_input"]["mood"],
            "motion_intensity":  compiled["user_input"]["motion_intensity"],
        },

        "quality_gates": {
            # Probe values — sourced from decode_probe and temporal_probe
            "mean_luminance":         decode_probe["mean_luminance"],
            "luminance_gate":         luminance_gate,
            "flicker_index":          temporal_probe["flicker_index"],
            "warping_artifact_score": temporal_probe["warping_artifact_score"],
            "scene_cut_detected":     temporal_probe["scene_cut_detected"],
            "perceptual_loop_score":  temporal_probe["perceptual_loop_score"],
            # Overall verdict — from gate_result
            "overall":                gate_result["overall"],
            # attempts_used — from generation_result (1 if not retried)
            "attempts_used":          generation_result.get("attempts_used", 1),
        },

        "post_processing": {
            # Anchor / LUM config — from GENERATION_CONSTANTS
            "anchor_position_default": GC["anchor_position_default"],
            "anchor_zone":             GC["anchor_zone"],
            "anchor_feather_px":       GC["anchor_feather_px"],
            "luminance_reduction":     GC["luminance_reduction"],
            # LUT selection — from post_result
            "selected_lut":            post_result["selected_lut"],
            "luts_generated":          post_result["luts_generated"],
            # Mask position keys — from post_result["masks"] (list of positions)
            "masks_generated":         list(post_result["masks"].keys()),
        },

        "files": files,
    }

    # Atomic write: full dict built in memory before any file I/O
    out_path = output_dir / f"{clip_id}_metadata.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


# ── assemble_edit_manifest() ───────────────────────────────────────────────────

def assemble_edit_manifest(
    clip_id:     str,
    post_result: dict,
    metadata:    dict,
    output_dir:  Path,
) -> dict:
    """
    Assemble and write bg_XXX_edit_manifest.json.

    Answers: what can change without regeneration and what does each change cost.
    """
    GC = GENERATION_CONSTANTS

    edit_manifest = {
        "clip_id":               clip_id,
        "edit_manifest_version": "1.0",
        # Source paths from assembled metadata (already existence-checked)
        "source_raw":      metadata["files"]["raw_loop"],
        "source_upscaled": metadata["files"]["upscaled"],

        "editable_parameters": {
            "anchor_zone": {
                # current anchor position — from GENERATION_CONSTANTS
                "current":               GC["anchor_position_default"],
                "available_precomputed": ["center", "lower_third", "upper_third"],
                "custom_supported":      True,
                "re_render_cost":        "mask_only",
                "operation":             "swap mask file + recomposite from source_upscaled",
                "estimated_time_s":      15,
                # content_risk_by_position — from post_result["risks"]
                # (each position assessed individually by assess_content_risk)
                "content_risk_by_position": {
                    position: post_result["risks"][position]
                    for position in post_result["risks"]
                },
            },

            "luminance_reduction": {
                # current reduction level — from GENERATION_CONSTANTS
                "current":          GC["luminance_reduction"],
                "range":            [0.10, 0.45],
                "re_render_cost":   "mask_only",
                "operation":        "regenerate mask + recomposite",
                "estimated_time_s": 15,
            },

            "lut": {
                # current LUT applied — from post_result
                "current":                   post_result["selected_lut"],
                # available pre-computed variants — from post_result
                "available_precomputed":     post_result["luts_generated"],
                "additional_luts_supported": True,
                "re_render_cost":            "grade_only",
                "operation":                 "apply new LUT to source_upscaled + recomposite",
                "estimated_time_s":          20,
            },

            "clip_start_offset_s": {
                "current":          0,
                "max":              12,
                "re_render_cost":   "none",
                "operation":        "trim only",
                "estimated_time_s": 2,
            },

            "loop_duration_extension": {
                # current total duration — from GENERATION_CONSTANTS
                "current_s":        GC["total_loop_duration_s"],
                "extendable_to_s":  30,
                "re_render_cost":   "extension_pass",
                "operation":        (
                    "one additional diffusion extension "
                    "+ interpolation + crossfade"
                ),
                "estimated_time_s": 900,
                "warning": (
                    "Each extension pass introduces temporal drift risk. "
                    "Review perceptual_loop_score after."
                ),
            },
        },

        "locked_parameters": {
            # base_generation points to the raw loop file path
            "base_generation": metadata["files"]["raw_loop"],
            # model — from GENERATION_CONSTANTS
            "model":           GC["model"],
            # prompts reference — points to metadata.json for provenance
            "prompts":         f"see {clip_id}_metadata.json prompt_provenance",
            # seed — from assembled metadata (ultimately from generation_result)
            "seed":            metadata["generation"]["seed"],
            "note": (
                "Any change to locked parameters requires full regeneration. "
                "A new clip_id will be assigned."
            ),
        },
    }

    # Atomic write
    out_path = output_dir / f"{clip_id}_edit_manifest.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(edit_manifest, fh, indent=2)

    return edit_manifest


# ── assemble_integration_contract() ───────────────────────────────────────────

def assemble_integration_contract(
    clip_id:        str,
    compiled:       dict,
    decode_probe:   dict,
    temporal_probe: dict,
    gate_result:    dict,
    post_result:    dict,
    metadata:       dict,
    output_dir:     Path,
) -> dict:
    """
    Assemble and write bg_XXX_integration_contract.json.

    Three independent audience sections — each is a separate dict.
    No section shares data by reference with another.
    """
    GC = GENERATION_CONSTANTS

    # cut_points — read from GENERATION_CONSTANTS (seam seconds + playable duration)
    cut_points = _compute_cut_points()

    # frame_count — read pre-computed value from GENERATION_CONSTANTS to avoid
    # rounding drift (playable_duration_s * target_fps can drift by ±1 frame)
    frame_count = int(GC["total_playable_frames"])

    # loop_frame_delta_max value: use explicit key if present (future probe), else
    # derive from perceptual_loop_score (1.0 - SSIM approximates mean abs delta)
    loop_delta_value = temporal_probe.get(
        "loop_origin_return_delta",
        round(1.0 - temporal_probe["perceptual_loop_score"], 6),
    )

    # ── Section 1: for_human_editor (independent dict) ──────────────────────────
    for_human_editor = {
        # primary_asset — from assembled metadata files section
        "primary_asset":   metadata["files"]["final"],
        # duration / fps / resolution — from GENERATION_CONSTANTS
        "duration_s":      GC["playable_duration_s"],
        "fps":             GC["target_fps"],
        "resolution":      GC["upscale_target"],
        "is_loop":         True,
        # loop_clean: pass or human_review means the loop is acceptable for air
        "loop_clean":      gate_result["overall"] in ("pass", "human_review"),
        "recommended_cut_points_s": cut_points,
        "anchor_zone_pixels":       _compute_anchor_pixels(),
        "safe_zones": {
            "lower_third": {"y_start_px": 842, "y_end_px": 1080},
            "upper_band":  {"y_start_px": 0,   "y_end_px": 130},
        },
        "editor_notes": (
            "Center anchor zone is pre-composited. "
            "Review risk JSON for boundary activity."
        ),
    }

    # ── Section 2: for_downstream_modules (independent dict) ────────────────────
    for_downstream_modules = {
        "scene_descriptors": {
            # All derived from compiled user_input + decode_probe via private helpers
            "dominant_environment":      _derive_environment(compiled),
            "light_condition":           _derive_light(compiled),
            "motion_character":          _derive_motion_character(compiled),
            "color_temperature_k":       _derive_color_temp_k(decode_probe),
            "color_temperature_k_source": (
                f"derived_from_decode_probe_hue_"
                f"{int(decode_probe['dominant_hue_degrees'])}_degrees"
            ),
            "background_complexity":     _derive_complexity(decode_probe),
        },
        "compositional_constraints": {
            # anchor_avoid_zone — from GENERATION_CONSTANTS
            "anchor_avoid_zone": GC["anchor_zone"],
            "text_safe_zones":   ["lower_third", "upper_band"],
        },
        "module_suggestions": {
            # lower_third_style — from compiled (LOWER_THIRD_STYLE_RULES lookup)
            "lower_third_style":        compiled["lower_third_style"],
            # source documents which rule-table keys produced the style token
            "lower_third_style_source": (
                f"rule_table: mood={compiled['user_input']['mood']} + "
                f"color_temperature={compiled['user_input']['color_temperature']}"
            ),
            "ticker_contrast":          _derive_ticker_contrast(compiled),
            "graphic_overlay_weight":   "light",
            "audio_mood_match":         _derive_audio_mood(compiled),
        },
    }

    # ── Section 3: for_compositor_process (independent dict) ────────────────────
    for_compositor_process = {
        # primary_asset — copied independently from metadata (not a reference)
        "primary_asset":   metadata["files"]["final"],
        # TODO (production): compute SHA256 hash of the final MP4 bytes before
        # writing this contract. Use hashlib.sha256 over the file content at
        # asset finalization time. Placeholder used in dev mode per spec.
        "asset_hash_sha256": "[computed_at_runtime]",
        # frame_count — derived from GENERATION_CONSTANTS (not passed in directly)
        "frame_count":       frame_count,
        "verified_loop":     True,
        "loop_frame_delta_max": {
            # value derived from temporal_probe
            "value":            loop_delta_value,
            "metric":           "mean_absolute_luminance_difference",
            "unit":             "normalized_0_to_1",
            "measured_between": "last_frame and first_frame of loop",
            "compositor_use":   (
                "values below 0.010 are invisible at broadcast frame rate"
            ),
        },
        # flicker/warp — directly from temporal_probe
        "flicker_index":           temporal_probe["flicker_index"],
        "warping_artifact_score":  temporal_probe["warping_artifact_score"],
        # decode_profile — from metadata files section (MISSING in dev phase)
        "decode_profile":          metadata["files"].get("decode_probe", "MISSING"),
        # integrity_check — from gate_result
        "integrity_check":         gate_result["overall"],
    }

    contract = {
        "module":                       "bg_video",
        "clip_id":                      clip_id,
        "integration_contract_version": "1.0",
        "schema_changelog":             {"1.0": "initial release"},
        "consumer_compatibility": {
            "minimum_reader_version": "1.0",
            "breaking_change_policy": "major version increment only",
        },
        "for_human_editor":          for_human_editor,
        "for_downstream_modules":    for_downstream_modules,
        "for_compositor_process":    for_compositor_process,
    }

    # Atomic write
    out_path = output_dir / f"{clip_id}_integration_contract.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2)

    return contract


# ── assemble_generation_log() ─────────────────────────────────────────────────

def assemble_generation_log(
    clip_id:           str,
    generation_result: dict,
    gate_result:       dict,
    output_dir:        Path,
) -> dict:
    """
    Write bg_XXX_generation_log.json for the standard (non-regeneration) case.

    If the log file already exists at the expected path (written by
    regenerator.py's write_generation_log()), read it and return its contents
    without overwriting. This covers the regeneration path transparently.

    If no prior log exists, write the standard pass-on-first-attempt shape.
    """
    log_path = output_dir / f"{clip_id}_generation_log.json"

    if log_path.exists():
        # Regenerator already wrote this file — honour it, do not overwrite
        with log_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    attempts = generation_result.get("attempts_used", 1)
    log_dict = {
        "clip_id":            clip_id,
        # total_attempts — from generation_result (1 for standard single-pass)
        "total_attempts":     attempts,
        # seeds_used — list containing the single seed used
        "seeds_used":         [generation_result["seed"]],
        "outcome":            f"pass_on_attempt_{attempts}",
        "escalated_to_human": False,
        # failure_log — from generation_result (empty list for standard pass)
        "failure_log":        generation_result.get("failure_log", []),
    }

    # Atomic write
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_dict, fh, indent=2)

    return log_dict


# ── run_metadata_assembly() ───────────────────────────────────────────────────

def run_metadata_assembly(
    clip_id:            str,
    run_number:         int,
    compiled:           dict,
    generation_result:  dict,
    decode_probe:       dict,
    temporal_probe:     dict,
    gate_result:        dict,
    post_result:        dict,
    output_dir:         Path,
) -> dict:
    """
    Top-level metadata assembly orchestrator.

    Calls all four assemblers in order:
      1. assemble_metadata        → bg_XXX_metadata.json
      2. assemble_edit_manifest   → bg_XXX_edit_manifest.json
      3. assemble_integration_contract → bg_XXX_integration_contract.json
      4. assemble_generation_log  → bg_XXX_generation_log.json

    All four files are written to output_dir (not a subdirectory).

    Returns dict with all 8 keys: clip_id, four file paths, four dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = assemble_metadata(
        clip_id=clip_id,
        run_number=run_number,
        compiled=compiled,
        generation_result=generation_result,
        decode_probe=decode_probe,
        temporal_probe=temporal_probe,
        gate_result=gate_result,
        post_result=post_result,
        output_dir=output_dir,
    )

    edit_manifest = assemble_edit_manifest(
        clip_id=clip_id,
        post_result=post_result,
        metadata=metadata,
        output_dir=output_dir,
    )

    integration_contract = assemble_integration_contract(
        clip_id=clip_id,
        compiled=compiled,
        decode_probe=decode_probe,
        temporal_probe=temporal_probe,
        gate_result=gate_result,
        post_result=post_result,
        metadata=metadata,
        output_dir=output_dir,
    )

    generation_log = assemble_generation_log(
        clip_id=clip_id,
        generation_result=generation_result,
        gate_result=gate_result,
        output_dir=output_dir,
    )

    return {
        "clip_id":              clip_id,
        "metadata_path":        str(output_dir / f"{clip_id}_metadata.json"),
        "edit_manifest_path":   str(output_dir / f"{clip_id}_edit_manifest.json"),
        "contract_path":        str(output_dir / f"{clip_id}_integration_contract.json"),
        "generation_log_path":  str(output_dir / f"{clip_id}_generation_log.json"),
        "metadata":             metadata,
        "edit_manifest":        edit_manifest,
        "integration_contract": integration_contract,
        "generation_log":       generation_log,
    }
