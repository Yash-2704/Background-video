"""
core/generator.py
─────────────────
Generation Stage for the Background Video Generation Module.

Responsibilities:
  1. generate_clip()    — produce a raw native-fps clip (dry-run or live)
  2. interpolate_clip() — upsample native_fps → target_fps  (dry-run or live)
  3. crossfade_join()   — join 3 interpolated clips into a single raw loop
  4. run_generation()   — top-level orchestrator called by the FastAPI layer

Dry-run mode (dev_mode=True in generation_constants.json):
  All model calls are bypassed. cv2 writes synthetic placeholder MP4 files
  that are syntactically valid so every downstream stage can run today.

Live mode (dev_mode=False):
  torch and diffusers are imported only inside the live branch so this
  module imports cleanly on a machine with zero ML packages installed.
"""

import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Config loading ─────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
_CONFIG_PATH     = PROJECT_ROOT / "config" / "generation_constants.json"

with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)


# ── generate_clip() ────────────────────────────────────────────────────────────

def generate_clip(
    positive:    str,
    motion:      str,
    negative:    str,
    seed:        int,
    clip_index:  int,    # 0 = base, 1 = ext_1, 2 = ext_2
    output_path: Path,
    dry_run:     bool,
) -> Path:
    """
    Produce a single raw clip at native_fps resolution.

    dry_run=True  → writes a synthetic solid-color MP4 using cv2.
    dry_run=False → calls CogVideoX (not yet implemented; raises NotImplementedError).
    """
    if dry_run:
        GC          = GENERATION_CONSTANTS
        width, height = GC["generate_resolution"]   # [720, 480]
        fps         = GC["native_fps"]              # 8
        n_frames    = GC["base_clip_frames_native"] # 49

        # Solid color varies by clip_index so tests can distinguish clips
        _colors = [
            (50,  80,  120),   # clip 0 — blue-ish
            (40,  70,  110),   # clip 1 — slightly darker
            (45,  75,  115),   # clip 2 — between
        ]
        color = _colors[clip_index % len(_colors)]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

        blank = np.full((height, width, 3), color, dtype=np.uint8)

        for i in range(n_frames):
            frame = blank.copy()
            if i == 0:
                # Burn seed into frame 0 so tests can verify which clip was written
                cv2.putText(
                    frame,
                    f"seed={seed}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)

        writer.release()
        return output_path

    else:
        # ── LIVE GENERATION ───────────────────────────────────────────────────
        # Guard-import heavy dependencies so dry-run never requires them.
        import torch                        # noqa: F401  (used by pipeline below)
        from diffusers import CogVideoXPipeline  # noqa: F401

        # TODO: LIVE GENERATION — slot real CogVideoX call here
        # Required parameters (all from GENERATION_CONSTANTS):
        #   model:      GENERATION_CONSTANTS["model"]
        #   steps:      GENERATION_CONSTANTS["steps"]
        #   cfg_scale:  GENERATION_CONSTANTS["cfg_scale"]
        #   sampler:    GENERATION_CONSTANTS["sampler"]
        #   resolution: GENERATION_CONSTANTS["generate_resolution"]
        #   fps:        GENERATION_CONSTANTS["native_fps"]
        #   frames:     GENERATION_CONSTANTS["base_clip_frames_native"]
        #   seed:       seed  (passed in — not from constants)
        # Prompt inputs: positive, motion, negative (passed in)
        raise NotImplementedError(
            "Live CogVideoX generation not yet implemented. "
            "Set dev_mode=true in generation_constants.json to use dry-run."
        )


# ── interpolate_clip() ─────────────────────────────────────────────────────────

def interpolate_clip(
    input_path:  Path,
    output_path: Path,
    dry_run:     bool,
) -> Path:
    """
    Upsample a clip from native_fps (8) to target_fps (30).

    dry_run=True  → frame-repeat placeholder (no RIFE model required).
    dry_run=False → calls RIFE 4.6 optical-flow interpolation (not yet implemented).
    """
    if dry_run:
        GC           = GENERATION_CONSTANTS
        native_fps   = GC["native_fps"]    # 8
        target_fps   = GC["target_fps"]    # 30
        target_frames = GC["base_clip_frames_native"] * target_fps // native_fps  # 183

        ratio   = target_fps / native_fps      # 3.75
        repeats = math.ceil(ratio)             # 4

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(target_fps), (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

        out_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for _ in range(repeats):
                if out_count >= target_frames:
                    break
                writer.write(frame)
                out_count += 1
            if out_count >= target_frames:
                break

        cap.release()
        writer.release()
        return output_path

    else:
        # TODO: LIVE INTERPOLATION — slot real RIFE 4.6 call here
        # Required parameters (all from GENERATION_CONSTANTS):
        #   model:       "RIFE 4.6"
        #   input_fps:   GENERATION_CONSTANTS["native_fps"]
        #   output_fps:  GENERATION_CONSTANTS["target_fps"]
        #   input_path:  input_path (passed in)
        #   output_path: output_path (passed in)
        raise NotImplementedError(
            "Live RIFE interpolation not yet implemented. "
            "Set dev_mode=true in generation_constants.json to use dry-run."
        )


# ── crossfade_join() ───────────────────────────────────────────────────────────

def crossfade_join(
    clip_paths:  list,    # list[Path] — 3 interpolated 30fps clips
    output_path: Path,
    dry_run:     bool,
) -> dict:
    """
    Join 3 interpolated clips into a single raw loop with linear crossfade
    blending at each seam. Computes and returns seam frame indices in both
    the raw and playable timelines.

    Returns:
        {
            "raw_loop_path":        Path,
            "seam_frames_raw":      [int, int],
            "seam_frames_playable": [int, int],
            "total_frames_raw":     int,
            "playable_frames":      int,
        }

    Seam math (all values from GENERATION_CONSTANTS):
        fpc              = base_clip_frames_native * target_fps // native_fps = 183
        crossfade_frames = 14
        seam_1_raw       = fpc       = 183
        seam_2_raw       = fpc * 2   = 366
        total_frames_raw = fpc * 3   = 549
        playable_frames  = 549 - (14 * 2) = 521
        seam_1_playable  = 183 - 14  = 169
        seam_2_playable  = 366 - 28  = 338

    Crossfade window:
        For seam at frame S, the blend replaces frames [S-cf : S-1]
        (the last cf frames of the outgoing clip) with a linear blend
        using the first cf frames of the incoming clip.
        alpha goes from 0.0 at j=0 to 1.0 at j=cf-1.
    """
    if dry_run:
        GC          = GENERATION_CONSTANTS
        native_fps  = GC["native_fps"]              # 8
        target_fps  = GC["target_fps"]              # 30
        cf          = GC["crossfade_frames"]        # 14
        seam_count  = GC["seam_count"]              # 2

        fpc = GC["base_clip_frames_native"] * target_fps // native_fps  # 183

        # ── Load all frames from all clips ────────────────────────────────────
        all_clip_frames: list = []   # list of lists of uint8 frames
        width = height = None

        for p in clip_paths:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture failed to open: {p}")
            if width is None:
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            all_clip_frames.append(frames)

        # ── Concatenate into a mutable output list ────────────────────────────
        output_frames: list = []
        for clip_frames in all_clip_frames:
            for f in clip_frames:
                output_frames.append(f.copy())

        total_frames_raw = len(output_frames)  # should be fpc*3 = 549

        # ── Apply crossfade at each seam ──────────────────────────────────────
        # Seam i is at frame index (i+1)*fpc (the first frame of the incoming clip).
        # Blend window: frames [seam - cf : seam] (inclusive start, exclusive end).
        # We replace these frames in output_frames with a blend of:
        #   outgoing: the same position in output_frames (already from outgoing clip)
        #   incoming: all_clip_frames[i+1][0 : cf]
        seams_raw = [fpc * (i + 1) for i in range(seam_count)]  # [183, 366]

        for seam_idx, seam in enumerate(seams_raw):
            incoming_clip = all_clip_frames[seam_idx + 1]
            window_start  = seam - cf   # 169 for seam=183, cf=14

            for j in range(cf):
                frame_pos = window_start + j               # index in output_frames
                alpha     = j / (cf - 1) if cf > 1 else 1.0

                frame_out = output_frames[frame_pos].astype(np.float32)
                frame_in  = incoming_clip[j].astype(np.float32)

                blended = ((1.0 - alpha) * frame_out + alpha * frame_in)
                output_frames[frame_pos] = np.clip(blended, 0, 255).astype(np.uint8)

        # ── Write joined file ─────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(target_fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

        for frame in output_frames:
            writer.write(frame)
        writer.release()

        # ── Seam math ─────────────────────────────────────────────────────────
        playable_frames    = total_frames_raw - (cf * seam_count)
        seam_frames_raw    = [seams_raw[0], seams_raw[1]]
        seam_frames_playable = [
            seams_raw[0] - cf,
            seams_raw[1] - cf * 2,
        ]

        return {
            "raw_loop_path":        output_path,
            "seam_frames_raw":      seam_frames_raw,
            "seam_frames_playable": seam_frames_playable,
            "total_frames_raw":     total_frames_raw,
            "playable_frames":      playable_frames,
        }

    else:
        # TODO: LIVE CROSSFADE JOIN — slot real RIFE 4.6 + FFmpeg 6.0 call here
        # Required parameters (all from GENERATION_CONSTANTS):
        #   crossfade_frames:  GENERATION_CONSTANTS["crossfade_frames"]
        #   target_fps:        GENERATION_CONSTANTS["target_fps"]
        #   seam_count:        GENERATION_CONSTANTS["seam_count"]
        #   clip_paths:        clip_paths (passed in — 3 interpolated 30fps clips)
        #   output_path:       output_path (passed in)
        # Approach:
        #   1. RIFE 4.6 optical-flow blend for crossfade windows
        #   2. FFmpeg 6.0 concat + overlay filter for final assembly
        raise NotImplementedError(
            "Live RIFE/FFmpeg crossfade join not yet implemented. "
            "Set dev_mode=true in generation_constants.json to use dry-run."
        )


# ── run_generation() ───────────────────────────────────────────────────────────

def run_generation(
    compiled:   dict,
    run_id:     str,
    output_dir: Path,
    seed:       int = None,
) -> dict:
    """
    Top-level orchestrator called by the FastAPI layer.

    Steps:
      1. Assign seed (random 10000–99999) if not provided.
      2. Create output_dir / run_id / "raw" directory.
      3. Generate 3 clips (seeds: seed, seed+1, seed+2).
      4. Interpolate all 3 clips from native_fps to target_fps.
      5. Join with crossfade → single raw loop file.
      6. Assemble and return the result + generation_log dict.

    On any failure: the run directory is deleted before re-raising.
    """
    if seed is None:
        seed = random.randint(10000, 99999)

    GC       = GENERATION_CONSTANTS
    dev_mode = GC["dev_mode"]

    run_dir = output_dir / run_id
    raw_dir = run_dir / "raw"

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 3: Generate 3 clips ──────────────────────────────────────────
        clip_seeds   = [seed, seed + 1, seed + 2]
        clip_names   = ["base_clip", "ext_1", "ext_2"]
        raw_clip_paths = []

        for idx, (clip_seed, clip_name) in enumerate(zip(clip_seeds, clip_names)):
            out_path = raw_dir / f"{clip_name}.mp4"
            generate_clip(
                positive=compiled["positive"],
                motion=compiled["motion"],
                negative=compiled["negative"],
                seed=clip_seed,
                clip_index=idx,
                output_path=out_path,
                dry_run=dev_mode,
            )
            raw_clip_paths.append(out_path)

        # ── Step 4: Interpolate all 3 clips ──────────────────────────────────
        interp_paths = []
        for raw_path, clip_name in zip(raw_clip_paths, clip_names):
            out_path = raw_dir / f"{clip_name}_30fps.mp4"
            interpolate_clip(
                input_path=raw_path,
                output_path=out_path,
                dry_run=dev_mode,
            )
            interp_paths.append(out_path)

        # ── Step 5: Crossfade join ────────────────────────────────────────────
        raw_loop_path = raw_dir / f"bg_{run_id}_raw_loop.mp4"
        join_result   = crossfade_join(
            clip_paths=interp_paths,
            output_path=raw_loop_path,
            dry_run=dev_mode,
        )

        # ── Step 6: Assemble return dict ──────────────────────────────────────
        generation_log = {
            "run_id":                   run_id,
            "seed":                     seed,
            "dev_mode":                 dev_mode,
            "clips_generated":          3,
            "seeds_used":               clip_seeds,
            "native_fps":               GC["native_fps"],
            "target_fps":               GC["target_fps"],
            "crossfade_frames":         GC["crossfade_frames"],
            "seam_frames_raw":          join_result["seam_frames_raw"],
            "seam_frames_playable":     join_result["seam_frames_playable"],
            "total_frames_raw":         join_result["total_frames_raw"],
            "playable_frames":          join_result["playable_frames"],
            "compiled_input_hash_short": compiled["input_hash_short"],
            "compiler_version":         compiled["compiler_version"],
            "status":                   "complete",
        }

        return {
            "run_id":               run_id,
            "status":               "complete",
            "raw_loop_path":        str(join_result["raw_loop_path"]),
            "seed":                 seed,
            "seam_frames_raw":      join_result["seam_frames_raw"],
            "seam_frames_playable": join_result["seam_frames_playable"],
            "generation_log":       generation_log,
        }

    except Exception as exc:
        # Clean up partial output — never leave a broken run directory on disk
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise RuntimeError(
            f"Generation failed for run {run_id}: {exc}"
        ) from exc
