"""
core/regenerator.py
───────────────────
Regeneration Loop for the Background Video Generation Module.

Responsibilities:
  1. regeneration_loop() — retry policy engine: attempts up to MAX_RETRIES
     generations, probes each result, evaluates gates, returns on first
     pass/human_review, or escalates via PipelineEscalationError.
  2. write_generation_log() — writes a JSON audit log to disk.
  3. PipelineEscalationError — raised when all retries are exhausted.

Design contract:
  This module is the orchestration layer. It does not generate video, run
  probes, or evaluate gates directly — all three are received as injected
  callables. This keeps the regeneration loop independently testable without
  a real video pipeline. The compiled dict passed in is never mutated.

Dependency direction:
  regenerator.py → gates.py (imported directly)
  regenerator.py → generation_fn, probe_decode_fn, probe_temporal_fn (injected)
  regenerator.py does NOT import core.generator or core.probes.
"""

import json
from pathlib import Path

from core.gates import evaluate_gates, GENERATION_CONSTANTS

# ── Module-level constants ──────────────────────────────────────────────────────
MAX_RETRIES = GENERATION_CONSTANTS["max_regeneration_retries"]   # 3
BASE_CFG    = GENERATION_CONSTANTS["cfg_scale"]                  # 6.0


# ── PipelineEscalationError ─────────────────────────────────────────────────────

class PipelineEscalationError(Exception):
    """
    Raised when all regeneration attempts are exhausted without a passing
    or human_review gate result. Carries the full failure log for upstream
    handling.
    """
    def __init__(self, run_id: str, failure_log: list):
        self.run_id      = run_id
        self.failure_log = failure_log
        super().__init__(
            f"Run {run_id}: all {len(failure_log)} regeneration "
            f"attempts failed. Pipeline escalated to human review. "
            f"See failure_log attribute for attempt details."
        )


# ── write_generation_log() ──────────────────────────────────────────────────────

def write_generation_log(
    log_dir:     Path,
    run_id:      str,
    failure_log: list,
    escalated:   bool,
) -> Path:
    """
    Writes a JSON audit log to log_dir / f"{run_id}_generation_log.json".
    Creates log_dir if it does not exist.

    Returns the path of the written log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine outcome string
    if escalated:
        outcome = "escalated"
    else:
        # Derive passing attempt from the last entry with a passing gate_result
        passing_attempt = None
        for entry in failure_log:
            if entry.get("gate_result") in ("pass", "human_review"):
                passing_attempt = entry["attempt"]
        outcome = f"pass_on_attempt_{passing_attempt}"

    log_content = {
        "run_id":             run_id,
        "total_attempts":     len(failure_log),
        "seeds_used":         [entry["seed"] for entry in failure_log],
        "outcome":            outcome,
        "escalated_to_human": escalated,
        "failure_log":        failure_log,
    }

    log_path = log_dir / f"{run_id}_generation_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_content, fh, indent=2)

    return log_path


# ── regeneration_loop() ─────────────────────────────────────────────────────────

def regeneration_loop(
    compiled:          dict,
    run_id:            str,
    output_dir:        Path,
    base_seed:         int,
    generation_fn:     callable,
    probe_decode_fn:   callable,
    probe_temporal_fn: callable,
    log_dir:           Path,
) -> dict:
    """
    Retry policy engine. Attempts up to MAX_RETRIES generations, runs probes
    on each result, evaluates gates, and either returns on the first
    pass/human_review or raises PipelineEscalationError after all attempts
    are exhausted.

    Args:
        compiled:          Prompt-compiled input dict (never mutated).
        run_id:            Unique run identifier.
        output_dir:        Directory for generation output.
        base_seed:         Base seed; each attempt offsets by +100, +200, +300.
        generation_fn:     Injected callable matching run_generation() signature.
        probe_decode_fn:   Injected callable matching run_decode_probe() signature.
        probe_temporal_fn: Injected callable matching run_temporal_probe() signature.
        log_dir:           Directory where the generation log JSON is written.

    Returns:
        Dict with run metadata, probe results, gate evaluation, and seam data.

    Raises:
        PipelineEscalationError: After all MAX_RETRIES attempts fail.
    """
    _STABILITY_SUFFIX = (
        ", extra temporal stability, "
        "minimal motion, static background"
    )

    failure_log: list = []

    for attempt in range(1, MAX_RETRIES + 1):
        current_seed = base_seed + attempt * 100

        # Build per-attempt cfg and motion (attempt 3 mutates both)
        if attempt == 3:
            current_cfg    = BASE_CFG - 0.5
            attempt_motion = compiled["motion"] + _STABILITY_SUFFIX
        else:
            current_cfg    = BASE_CFG
            attempt_motion = compiled["motion"]

        # Build a modified copy of compiled — never mutate the original
        attempt_compiled = {**compiled, "motion": attempt_motion}

        gen_result     = None
        decode_result  = None
        temporal_result = None
        gates          = None

        try:
            gen_result = generation_fn(
                compiled   = attempt_compiled,
                run_id     = f"{run_id}_regen_attempt_{attempt}",
                output_dir = output_dir,
                seed       = current_seed,
            )

            raw_loop_path = Path(gen_result["raw_loop_path"])

            decode_result  = probe_decode_fn(raw_loop_path)
            temporal_result = probe_temporal_fn(raw_loop_path)
            gates          = evaluate_gates(decode_result, temporal_result)

        except PipelineEscalationError:
            # Never swallow escalation exceptions
            raise
        except Exception:
            # Generation or probe failure — log and continue to next attempt
            failure_log.append({
                "attempt":     attempt,
                "seed":        current_seed,
                "cfg_used":    current_cfg,
                "gate_result": "generation_error",
                "failures":    [],
                "human_flags": [],
            })
            continue

        # Append attempt record to failure_log
        failure_log.append({
            "attempt":     attempt,
            "seed":        current_seed,
            "cfg_used":    current_cfg,
            "gate_result": gates["overall"],
            "failures":    gates["failures"],
            "human_flags": gates["human_flags"],
        })

        # Return on first pass or human_review
        if gates["overall"] in ("pass", "human_review"):
            write_generation_log(log_dir, run_id, failure_log, escalated=False)
            return {
                "run_id":               run_id,
                "status":               "complete",
                "raw_loop_path":        str(gen_result["raw_loop_path"]),
                "seed_used":            current_seed,
                "attempts_used":        attempt,
                "gate_result":          gates["overall"],
                "failure_log":          failure_log,
                "decode_probe":         decode_result,
                "temporal_probe":       temporal_result,
                "gate_evaluation":      gates,
                "seam_frames_raw":      gen_result["seam_frames_raw"],
                "seam_frames_playable": gen_result["seam_frames_playable"],
            }

    # All attempts exhausted without pass/human_review
    write_generation_log(log_dir, run_id, failure_log, escalated=True)
    raise PipelineEscalationError(run_id, failure_log)
