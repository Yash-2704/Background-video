"""
tests/test_metadata_assembler.py
─────────────────────────────────
47 tests for core/metadata_assembler.py and the prompt_compiler.py modification.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.prompt_compiler import compile_prompts
from core.probes import run_decode_probe, run_temporal_probe
from core.gates import evaluate_gates
from core.generator import run_generation
from core.post_processor import run_post_processing
from core.metadata_assembler import (
    GENERATION_CONSTANTS,
    assemble_metadata,
    assemble_edit_manifest,
    assemble_integration_contract,
    assemble_generation_log,
    run_metadata_assembly,
    _compute_anchor_pixels,
    _derive_environment,
    _derive_light,
    _derive_motion_character,
    _derive_color_temp_k,
    _derive_complexity,
    _derive_ticker_contrast,
    _derive_audio_mood,
)

# ── Canonical user input ───────────────────────────────────────────────────────
_CANONICAL_INPUT = {
    "category":          "Economy",
    "location_feel":     "Urban",
    "time_of_day":       "Dusk",
    "color_temperature": "Cool",
    "mood":              "Serious",
    "motion_intensity":  "Gentle",
}

_CLIP_ID = "bg_001"


# ── Session-scoped fixtures (no file I/O) ─────────────────────────────────────

@pytest.fixture(scope="session")
def canonical_compiled():
    return compile_prompts(_CANONICAL_INPUT, "v1.0")


@pytest.fixture(scope="session")
def dry_decode():
    # dry_run=True: no clip_path needed but argument is required
    return run_decode_probe(Path("/dev/null"), dry_run=True)


@pytest.fixture(scope="session")
def dry_temporal():
    return run_temporal_probe(Path("/dev/null"), dry_run=True)


@pytest.fixture(scope="session")
def dry_gates(dry_decode, dry_temporal):
    return evaluate_gates(dry_decode, dry_temporal)


# ── Function-scoped fixtures (use tmp_path for real file creation) ─────────────

@pytest.fixture
def dry_run_generation_result(tmp_path):
    """Minimal synthetic generation result dict as specified."""
    return {
        "run_id":               "bg_001_b2e7f3",
        "status":               "complete",
        "raw_loop_path":        str(tmp_path / "raw" / "raw_loop.mp4"),
        "seed":                 42819,
        "attempts_used":        1,
        "seam_frames_raw":      [183, 366],
        "seam_frames_playable": [169, 338],
        "generation_log":       {},
        "failure_log":          [],
    }


@pytest.fixture
def dry_post_result(tmp_path, canonical_compiled, dry_decode):
    """
    Integration fixture: real files on disk from run_generation() + run_post_processing().
    The assembler will be able to verify these paths exist.
    """
    gen_result = run_generation(canonical_compiled, _CLIP_ID, tmp_path)
    post = run_post_processing(
        clip_id=_CLIP_ID,
        raw_loop_path=Path(gen_result["raw_loop_path"]),
        decode_probe=dry_decode,
        compiled=canonical_compiled,
        seam_frames_playable=gen_result["seam_frames_playable"],
        output_dir=tmp_path,
        dry_run=True,
    )
    return post


@pytest.fixture
def integration_bundle(tmp_path, canonical_compiled, dry_decode, dry_temporal, dry_gates):
    """
    Full integration bundle: consistent paths for run_metadata_assembly tests.
    """
    gen_result = run_generation(canonical_compiled, _CLIP_ID, tmp_path)
    post = run_post_processing(
        clip_id=_CLIP_ID,
        raw_loop_path=Path(gen_result["raw_loop_path"]),
        decode_probe=dry_decode,
        compiled=canonical_compiled,
        seam_frames_playable=gen_result["seam_frames_playable"],
        output_dir=tmp_path,
        dry_run=True,
    )
    return {
        "clip_id":           _CLIP_ID,
        "compiled":          canonical_compiled,
        "gen_result":        gen_result,
        "decode":            dry_decode,
        "temporal":          dry_temporal,
        "gates":             dry_gates,
        "post":              post,
        "tmp_path":          tmp_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tests 1–4: prompt_compiler modification
# ══════════════════════════════════════════════════════════════════════════════

def test_01_compiled_contains_user_input_key(canonical_compiled):
    """compile_prompts() result must contain 'user_input' key."""
    assert "user_input" in canonical_compiled


def test_02_user_input_category_value(canonical_compiled):
    """user_input['category'] == 'Economy' for canonical input."""
    assert canonical_compiled["user_input"]["category"] == "Economy"


def test_03_all_six_user_input_fields_present(canonical_compiled):
    """All 6 user input fields present in user_input sub-dict."""
    expected = {"category", "location_feel", "time_of_day",
                "color_temperature", "mood", "motion_intensity"}
    assert set(canonical_compiled["user_input"].keys()) == expected


def test_04_existing_compiler_keys_unchanged(canonical_compiled):
    """Existing compile_prompts() behavior unchanged — hashes still present."""
    for key in ("positive_hash", "motion_hash", "negative_hash", "input_hash_short"):
        assert key in canonical_compiled
        assert isinstance(canonical_compiled[key], str)
        assert len(canonical_compiled[key]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Tests 5–14: assemble_metadata
# ══════════════════════════════════════════════════════════════════════════════

def test_05_assemble_metadata_top_level_keys(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """assemble_metadata() returns dict with all required top-level keys."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode,
        temporal_probe=dry_temporal,
        gate_result=dry_gates,
        post_result=dry_post_result,
        output_dir=tmp_path,
    )
    expected_keys = {
        "module", "module_version", "clip_id", "schema_version",
        "dev_mode", "timestamp_utc", "environment", "generation",
        "prompt_provenance", "user_input", "quality_gates",
        "post_processing", "files",
    }
    assert expected_keys.issubset(set(meta.keys()))


def test_06_metadata_json_written(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """bg_XXX_metadata.json is written to output_dir."""
    assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert (tmp_path / f"{_CLIP_ID}_metadata.json").exists()


def test_07_metadata_json_valid(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """Written metadata.json is valid JSON and loads correctly."""
    assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    path = tmp_path / f"{_CLIP_ID}_metadata.json"
    with path.open() as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict)
    assert "clip_id" in loaded


def test_08_clip_id_matches(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.clip_id matches the argument."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["clip_id"] == _CLIP_ID


def test_09_generation_seed_matches(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.generation.seed matches generation_result['seed']."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["generation"]["seed"] == dry_run_generation_result["seed"]


def test_10_quality_gates_overall_matches(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.quality_gates.overall matches gate_result['overall']."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["quality_gates"]["overall"] == dry_gates["overall"]


def test_11_input_hash_short_matches(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.prompt_provenance.input_hash_short matches compiled['input_hash_short']."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["prompt_provenance"]["input_hash_short"] == canonical_compiled["input_hash_short"]


def test_12_selected_lut_matches(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.post_processing.selected_lut matches post_result['selected_lut']."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["post_processing"]["selected_lut"] == dry_post_result["selected_lut"]


def test_13_files_section_has_required_keys(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """files section contains all required keys."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    for key in ("raw_loop", "upscaled", "masks", "luts", "final", "preview"):
        assert key in meta["files"], f"missing files key: {key}"


def test_14_timestamp_utc_ends_with_z(
    tmp_path, canonical_compiled, dry_run_generation_result,
    dry_decode, dry_temporal, dry_gates, dry_post_result
):
    """metadata.timestamp_utc ends with 'Z'."""
    meta = assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )
    assert meta["timestamp_utc"].endswith("Z")


# ══════════════════════════════════════════════════════════════════════════════
# Tests 15–20: assemble_edit_manifest
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_metadata(tmp_path, canonical_compiled, dry_run_generation_result,
                    dry_decode, dry_temporal, dry_gates, dry_post_result):
    return assemble_metadata(
        clip_id=_CLIP_ID, run_number=1,
        compiled=canonical_compiled,
        generation_result=dry_run_generation_result,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        output_dir=tmp_path,
    )


def test_15_edit_manifest_top_level_keys(
    tmp_path, dry_post_result, sample_metadata
):
    """assemble_edit_manifest() returns dict with editable_parameters and locked_parameters."""
    em = assemble_edit_manifest(
        clip_id=_CLIP_ID,
        post_result=dry_post_result,
        metadata=sample_metadata,
        output_dir=tmp_path,
    )
    assert "editable_parameters" in em
    assert "locked_parameters" in em


def test_16_edit_manifest_json_written(
    tmp_path, dry_post_result, sample_metadata
):
    """bg_XXX_edit_manifest.json is written to output_dir."""
    assemble_edit_manifest(
        clip_id=_CLIP_ID, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    assert (tmp_path / f"{_CLIP_ID}_edit_manifest.json").exists()


def test_17_editable_parameters_has_five_keys(
    tmp_path, dry_post_result, sample_metadata
):
    """editable_parameters contains all 5 expected keys."""
    em = assemble_edit_manifest(
        clip_id=_CLIP_ID, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    expected = {
        "anchor_zone", "luminance_reduction", "lut",
        "clip_start_offset_s", "loop_duration_extension",
    }
    assert expected == set(em["editable_parameters"].keys())


def test_18_lut_current_matches_post_result(
    tmp_path, dry_post_result, sample_metadata
):
    """editable_parameters.lut.current matches post_result['selected_lut']."""
    em = assemble_edit_manifest(
        clip_id=_CLIP_ID, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    assert em["editable_parameters"]["lut"]["current"] == dry_post_result["selected_lut"]


def test_19_content_risk_has_three_entries(
    tmp_path, dry_post_result, sample_metadata
):
    """anchor_zone.content_risk_by_position has 3 entries."""
    em = assemble_edit_manifest(
        clip_id=_CLIP_ID, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    risk = em["editable_parameters"]["anchor_zone"]["content_risk_by_position"]
    assert len(risk) == 3


def test_20_locked_parameters_seed_matches(
    tmp_path, dry_run_generation_result, dry_post_result, sample_metadata
):
    """locked_parameters.seed matches generation_result['seed']."""
    em = assemble_edit_manifest(
        clip_id=_CLIP_ID, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    assert em["locked_parameters"]["seed"] == dry_run_generation_result["seed"]


# ══════════════════════════════════════════════════════════════════════════════
# Tests 21–30: assemble_integration_contract
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_contract(
    tmp_path, canonical_compiled, dry_decode, dry_temporal,
    dry_gates, dry_post_result, sample_metadata
):
    return assemble_integration_contract(
        clip_id=_CLIP_ID,
        compiled=canonical_compiled,
        decode_probe=dry_decode,
        temporal_probe=dry_temporal,
        gate_result=dry_gates,
        post_result=dry_post_result,
        metadata=sample_metadata,
        output_dir=tmp_path,
    )


def test_21_contract_has_three_audience_keys(sample_contract):
    """assemble_integration_contract() returns dict with 3 audience keys."""
    assert "for_human_editor" in sample_contract
    assert "for_downstream_modules" in sample_contract
    assert "for_compositor_process" in sample_contract


def test_22_integration_contract_json_written(
    tmp_path, canonical_compiled, dry_decode, dry_temporal,
    dry_gates, dry_post_result, sample_metadata
):
    """bg_XXX_integration_contract.json is written to output_dir."""
    assemble_integration_contract(
        clip_id=_CLIP_ID,
        compiled=canonical_compiled,
        decode_probe=dry_decode, temporal_probe=dry_temporal,
        gate_result=dry_gates, post_result=dry_post_result,
        metadata=sample_metadata, output_dir=tmp_path,
    )
    assert (tmp_path / f"{_CLIP_ID}_integration_contract.json").exists()


def test_23_human_editor_duration_matches_constants(sample_contract):
    """for_human_editor.duration_s == GENERATION_CONSTANTS['playable_duration_s']."""
    assert (
        sample_contract["for_human_editor"]["duration_s"]
        == GENERATION_CONSTANTS["playable_duration_s"]
    )


def test_24_lower_third_style_is_minimal_dark_bar(sample_contract):
    """for_downstream_modules.module_suggestions.lower_third_style == 'minimal_dark_bar'
    for Serious + Cool inputs."""
    style = sample_contract["for_downstream_modules"]["module_suggestions"]["lower_third_style"]
    assert style == "minimal_dark_bar"


def test_25_dominant_environment_is_urban_exterior(sample_contract):
    """for_downstream_modules.scene_descriptors.dominant_environment == 'urban_exterior'."""
    env = sample_contract["for_downstream_modules"]["scene_descriptors"]["dominant_environment"]
    assert env == "urban_exterior"


def test_26_light_condition_is_dusk_directional(sample_contract):
    """for_downstream_modules.scene_descriptors.light_condition == 'dusk_directional'."""
    light = sample_contract["for_downstream_modules"]["scene_descriptors"]["light_condition"]
    assert light == "dusk_directional"


def test_27_compositor_flicker_index_matches_probe(sample_contract, dry_temporal):
    """for_compositor_process.flicker_index == dry_temporal['flicker_index']."""
    assert (
        sample_contract["for_compositor_process"]["flicker_index"]
        == dry_temporal["flicker_index"]
    )


def test_28_anchor_pixels_returns_correct_types():
    """_compute_anchor_pixels() returns dict with x, y, w, h all as integers."""
    pixels = _compute_anchor_pixels()
    for key in ("x", "y", "w", "h"):
        assert key in pixels
        assert isinstance(pixels[key], int), f"{key} should be int"


def test_29_anchor_pixels_x_value():
    """_compute_anchor_pixels().x == int(0.25 * 1920) == 480."""
    pixels = _compute_anchor_pixels()
    assert pixels["x"] == 480


def test_30_recommended_cut_points_structure(sample_contract):
    """recommended_cut_points_s has 4 entries, first is 0, last matches playable_duration_s."""
    points = sample_contract["for_human_editor"]["recommended_cut_points_s"]
    assert len(points) == 4
    assert points[0] == 0
    assert points[-1] == round(GENERATION_CONSTANTS["playable_duration_s"], 2)


# ══════════════════════════════════════════════════════════════════════════════
# Tests 31–34: assemble_generation_log
# ══════════════════════════════════════════════════════════════════════════════

def test_31_generation_log_has_required_keys(
    tmp_path, dry_run_generation_result, dry_gates
):
    """assemble_generation_log() returns dict with all required keys."""
    log = assemble_generation_log(
        clip_id=_CLIP_ID,
        generation_result=dry_run_generation_result,
        gate_result=dry_gates,
        output_dir=tmp_path,
    )
    for key in ("clip_id", "total_attempts", "seeds_used",
                "outcome", "escalated_to_human", "failure_log"):
        assert key in log, f"missing key: {key}"


def test_32_generation_log_json_written(
    tmp_path, dry_run_generation_result, dry_gates
):
    """bg_XXX_generation_log.json written to output_dir when no prior log exists."""
    assemble_generation_log(
        clip_id=_CLIP_ID,
        generation_result=dry_run_generation_result,
        gate_result=dry_gates,
        output_dir=tmp_path,
    )
    assert (tmp_path / f"{_CLIP_ID}_generation_log.json").exists()


def test_33_existing_log_not_overwritten(
    tmp_path, dry_run_generation_result, dry_gates
):
    """If log file already exists, it is not overwritten — returned dict matches file."""
    # Write a pre-existing log (simulating regenerator.py having written it)
    log_path = tmp_path / f"{_CLIP_ID}_generation_log.json"
    prior_content = {
        "clip_id": _CLIP_ID,
        "total_attempts": 3,
        "seeds_used": [100, 200, 300],
        "outcome": "pass_on_attempt_3",
        "escalated_to_human": False,
        "failure_log": [{"attempt": 1}, {"attempt": 2}],
    }
    with log_path.open("w") as f:
        json.dump(prior_content, f)

    returned = assemble_generation_log(
        clip_id=_CLIP_ID,
        generation_result=dry_run_generation_result,
        gate_result=dry_gates,
        output_dir=tmp_path,
    )

    # File should still contain the prior content
    with log_path.open() as f:
        on_disk = json.load(f)
    assert on_disk["total_attempts"] == 3
    assert returned["total_attempts"] == 3


def test_34_outcome_pass_on_attempt_1(
    tmp_path, dry_run_generation_result, dry_gates
):
    """outcome == 'pass_on_attempt_1' for standard single-attempt case."""
    log = assemble_generation_log(
        clip_id=_CLIP_ID,
        generation_result=dry_run_generation_result,
        gate_result=dry_gates,
        output_dir=tmp_path,
    )
    assert log["outcome"] == "pass_on_attempt_1"


# ══════════════════════════════════════════════════════════════════════════════
# Tests 35–38: run_metadata_assembly (integration)
# ══════════════════════════════════════════════════════════════════════════════

def test_35_run_metadata_assembly_returns_eight_keys(integration_bundle):
    b = integration_bundle
    result = run_metadata_assembly(
        clip_id=b["clip_id"], run_number=1,
        compiled=b["compiled"],
        generation_result=b["gen_result"],
        decode_probe=b["decode"],
        temporal_probe=b["temporal"],
        gate_result=b["gates"],
        post_result=b["post"],
        output_dir=b["tmp_path"],
    )
    expected = {
        "clip_id", "metadata_path", "edit_manifest_path",
        "contract_path", "generation_log_path",
        "metadata", "edit_manifest", "integration_contract", "generation_log",
    }
    assert expected.issubset(set(result.keys()))


def test_36_all_four_file_paths_exist(integration_bundle):
    """All 4 file paths in result dict exist on disk."""
    b = integration_bundle
    result = run_metadata_assembly(
        clip_id=b["clip_id"], run_number=1,
        compiled=b["compiled"],
        generation_result=b["gen_result"],
        decode_probe=b["decode"],
        temporal_probe=b["temporal"],
        gate_result=b["gates"],
        post_result=b["post"],
        output_dir=b["tmp_path"],
    )
    for key in ("metadata_path", "edit_manifest_path",
                "contract_path", "generation_log_path"):
        assert Path(result[key]).exists(), f"file missing: {key} = {result[key]}"


def test_37_all_four_files_are_valid_json(integration_bundle):
    """All 4 output files are valid JSON."""
    b = integration_bundle
    result = run_metadata_assembly(
        clip_id=b["clip_id"], run_number=1,
        compiled=b["compiled"],
        generation_result=b["gen_result"],
        decode_probe=b["decode"],
        temporal_probe=b["temporal"],
        gate_result=b["gates"],
        post_result=b["post"],
        output_dir=b["tmp_path"],
    )
    for key in ("metadata_path", "edit_manifest_path",
                "contract_path", "generation_log_path"):
        path = Path(result[key])
        with path.open() as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict), f"not a dict: {key}"


def test_38_no_file_written_outside_output_dir(integration_bundle):
    """All 4 files are written inside output_dir."""
    b = integration_bundle
    result = run_metadata_assembly(
        clip_id=b["clip_id"], run_number=1,
        compiled=b["compiled"],
        generation_result=b["gen_result"],
        decode_probe=b["decode"],
        temporal_probe=b["temporal"],
        gate_result=b["gates"],
        post_result=b["post"],
        output_dir=b["tmp_path"],
    )
    out = b["tmp_path"].resolve()
    for key in ("metadata_path", "edit_manifest_path",
                "contract_path", "generation_log_path"):
        p = Path(result[key]).resolve()
        assert str(p).startswith(str(out)), f"{key} written outside output_dir"


# ══════════════════════════════════════════════════════════════════════════════
# Tests 39–45: private helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_39_derive_environment_urban(canonical_compiled):
    assert _derive_environment(canonical_compiled) == "urban_exterior"


def test_40_derive_light_dusk(canonical_compiled):
    assert _derive_light(canonical_compiled) == "dusk_directional"


def test_41_derive_motion_gentle(canonical_compiled):
    assert _derive_motion_character(canonical_compiled) == "slow_lateral_drift"


def test_42_derive_color_temp_k_blue_hue():
    """dominant_hue_degrees=212 (blue range) → 6500K."""
    probe = {"dominant_hue_degrees": 212.0}
    assert _derive_color_temp_k(probe) == 6500


def test_43_derive_complexity_medium():
    """saturation_mean=0.38 → 'medium'."""
    probe = {"saturation_mean": 0.38}
    assert _derive_complexity(probe) == "medium"


def test_44_derive_ticker_contrast_cool():
    """color_temperature='Cool' → 'light_on_dark'."""
    compiled = {"user_input": {"color_temperature": "Cool"}}
    assert _derive_ticker_contrast(compiled) == "light_on_dark"


def test_45_derive_audio_mood_serious():
    """mood='Serious' → 'measured_serious'."""
    compiled = {"user_input": {"mood": "Serious"}}
    assert _derive_audio_mood(compiled) == "measured_serious"


# ══════════════════════════════════════════════════════════════════════════════
# Tests 46–47: regression
# ══════════════════════════════════════════════════════════════════════════════

def test_46_no_ml_imports():
    """Importing core.metadata_assembler must not import torch or diffusers."""
    import importlib
    import sys

    # Remove cached module if present so we get a clean import trace
    mods_to_remove = [k for k in sys.modules if k in (
        "core.metadata_assembler", "torch", "diffusers"
    )]
    for m in mods_to_remove:
        del sys.modules[m]

    import core.metadata_assembler  # noqa: F401

    assert "torch" not in sys.modules, "torch was imported by metadata_assembler"
    assert "diffusers" not in sys.modules, "diffusers was imported by metadata_assembler"


def test_47_prompt_compiler_tests_still_pass():
    """All prior prompt_compiler tests still pass — zero regressions."""
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_prompt_compiler.py", "-q", "--tb=short"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"prompt_compiler tests failed:\n{result.stdout}\n{result.stderr}"
    )
