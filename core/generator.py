"""
core/generator.py
─────────────────
Generation Stage for the Background Video Generation Module.

Responsibilities:
  1. generate_clip()    — produce a raw native-fps clip (dry-run or live)
                          T2V pathway for base clip (clip_index==0)
                          I2V pathway for extension clips (clip_index>0)
  2. crossfade_join()   — join 3 native-fps clips into a single raw loop
  3. run_generation()   — top-level orchestrator called by the FastAPI layer
  4. interpolate_clip() — OPTIONAL: upsample native_fps → target_fps
                          Not called in default 24fps pipeline. See function
                          docstring for re-enable instructions.

Dry-run mode (dev_mode=True in generation_constants.json):
  All model calls are bypassed. cv2 writes synthetic placeholder MP4 files
  that are syntactically valid so every downstream stage can run today.

Live mode (dev_mode=False):
  torch and diffusers are imported only inside the live branch so this
  module imports cleanly on a machine with zero ML packages installed.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Config loading ─────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
_CONFIG_PATH     = PROJECT_ROOT / "config" / "generation_constants.json"

with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    GENERATION_CONSTANTS: dict = json.load(_fh)


# ── Internal helper ────────────────────────────────────────────────────────────

def _extract_last_frame(video_path: Path) -> np.ndarray:
    """
    Read the last frame of a video file using cv2.VideoCapture.
    Returns a BGR uint8 ndarray.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read last frame from: {video_path}")
    return frame


# ── generate_clip() ────────────────────────────────────────────────────────────

def generate_clip(
    positive:    str,
    motion:      str,
    negative:    str,
    seed:        int,
    clip_index:  int,              # 0 = base (T2V), 1 = ext_1 (I2V), 2 = ext_2 (I2V)
    output_path: Path,
    dry_run:     bool,
    conditioning_frame: Optional[np.ndarray] = None,
                                   # None  → T2V pathway (clip_index == 0)
                                   # array → I2V pathway (clip_index > 0),
                                   #         last frame of previous clip, BGR format
) -> Path:
    """
    Produce a single raw clip at native_fps resolution.

    T2V pathway (conditioning_frame is None):
        Used for the base clip (clip_index == 0).
        dry_run → solid color at new resolution/fps, seed burned on frame 0.

    I2V pathway (conditioning_frame is np.ndarray):
        Used for extension clips (clip_index > 0).
        dry_run → color derived from mean of conditioning_frame + slight shift,
        simulating visual continuity. "I2V seed=N" burned on frame 0.
        (DRY-RUN STUB — not real I2V conditioning.)

    dry_run=True  → writes a synthetic MP4 using cv2.
    dry_run=False → calls Wan2.2-TI2V-5B (not yet implemented; raises NotImplementedError).
    """
    if dry_run:
        GC              = GENERATION_CONSTANTS
        width, height   = GC["generate_resolution"]        # [1280, 720]
        fps             = GC["native_fps"]                  # 24
        n_frames        = GC["base_clip_frames_native"]     # 145

        if conditioning_frame is None:
            # ── T2V pathway (base clip) ───────────────────────────────────────
            # Solid color varies by clip_index so tests can distinguish clips.
            _colors = [
                (50, 80, 120),   # clip 0 — blue-ish
            ]
            color = _colors[clip_index % len(_colors)]
            label = f"seed={seed}"
        else:
            # ── I2V pathway (extension clip) ──────────────────────────────────
            # DRY-RUN STUB: derive base color from mean BGR of conditioning_frame,
            # then apply a slight hue shift to simulate visual continuity.
            mean_bgr = np.mean(conditioning_frame.reshape(-1, 3), axis=0)  # shape (3,)
            shift    = np.array([5, 3, -3], dtype=np.float32)
            shifted  = np.clip(mean_bgr + shift, 0, 255)
            color    = tuple(int(v) for v in shifted)
            label    = f"I2V seed={seed}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

        blank = np.full((height, width, 3), color, dtype=np.uint8)

        for i in range(n_frames):
            frame = blank.copy()
            if i == 0:
                cv2.putText(
                    frame,
                    label,
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
        import torch                                     # noqa: F401
        from diffusers import AutoencoderKLWan           # noqa: F401

        # TODO: LIVE GENERATION — slot real Wan2.2-TI2V call here
        # T2V pathway (clip_index == 0, conditioning_frame is None):
        #   Use standard text-to-video pipeline
        #   Required parameters (all from GENERATION_CONSTANTS):
        #     model:           GENERATION_CONSTANTS["model"]
        #     num_frames:      GENERATION_CONSTANTS["base_clip_frames_native"]  # 145
        #     fps:             GENERATION_CONSTANTS["native_fps"]               # 24
        #     resolution:      GENERATION_CONSTANTS["generate_resolution"]      # [1280, 720]
        #     cfg_scale:       GENERATION_CONSTANTS["cfg_scale"]
        #     steps:           GENERATION_CONSTANTS["steps"]
        #     sampler:         GENERATION_CONSTANTS["sampler"]
        #     seed:            seed  (passed in — not from constants)
        #     vae_compression: GENERATION_CONSTANTS["vae_compression"]
        #       → (frames - 1) % 4 == 0 must be satisfied: 145 ✓
        #
        # I2V pathway (clip_index > 0, conditioning_frame provided):
        #   Use image-to-video pipeline
        #   Additional parameter:
        #     conditioning_frame: np.ndarray (last frame of previous clip, BGR format)
        #   The model conditions on this frame natively —
        #   no external workaround needed for Wan2.2-TI2V
        raise NotImplementedError(
            "Live Wan2.2-TI2V-5B generation not yet implemented. "
            "Set dev_mode=true in generation_constants.json to use dry-run."
        )


# ── crossfade_join() ───────────────────────────────────────────────────────────

def crossfade_join(
    clip_paths:  list,    # list[Path] — 3 native-fps clips
    output_path: Path,
    dry_run:     bool,
) -> dict:
    """
    Join 3 native-fps clips into a single raw loop with linear crossfade
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
        fpc              = base_clip_frames_native = 145
        crossfade_frames = 14
        seam_1_raw       = 145
        seam_2_raw       = 290
        total_frames_raw = 435
        seam_frames_playable_timeline = [138, 269]  ← config is authority
        total_playable_frames         = 406          ← config is authority

    Crossfade window:
        For seam at frame S, the blend replaces frames [S-cf : S-1]
        (the last cf frames of the outgoing clip) with a linear blend
        using the first cf frames of the incoming clip.
        alpha goes from 0.0 at j=0 to 1.0 at j=cf-1.
    """
    if dry_run:
        GC          = GENERATION_CONSTANTS
        native_fps  = GC["native_fps"]                        # 24
        cf          = GC["crossfade_frames"]                  # 14
        seam_count  = GC["seam_count"]                        # 2
        fpc         = GC["base_clip_frames_native"]           # 145

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

        total_frames_raw = len(output_frames)  # should be fpc*3 = 435

        # ── Apply crossfade at each seam ──────────────────────────────────────
        # Seam i is at frame index (i+1)*fpc (the first frame of the incoming clip).
        # Blend window: frames [seam - cf : seam] (inclusive start, exclusive end).
        # We replace these frames in output_frames with a blend of:
        #   outgoing: the same position in output_frames (already from outgoing clip)
        #   incoming: all_clip_frames[i+1][0 : cf]
        seams_raw = [fpc * (i + 1) for i in range(seam_count)]  # [145, 290]

        for seam_idx, seam in enumerate(seams_raw):
            incoming_clip = all_clip_frames[seam_idx + 1]
            window_start  = seam - cf   # 131 for seam=145, cf=14

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
            str(output_path), fourcc, float(native_fps), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

        for frame in output_frames:
            writer.write(frame)
        writer.release()

        # ── Seam math — config is authority for playable values ───────────────
        seam_frames_playable = GC["seam_frames_playable_timeline"]   # [138, 269]
        playable_frames      = GC["total_playable_frames"]            # 406

        return {
            "raw_loop_path":        output_path,
            "seam_frames_raw":      [seams_raw[0], seams_raw[1]],
            "seam_frames_playable": seam_frames_playable,
            "total_frames_raw":     total_frames_raw,
            "playable_frames":      playable_frames,
        }

    else:
        # TODO: LIVE CROSSFADE JOIN — slot real Wan2.2-TI2V + FFmpeg 6.0 call here
        # Required parameters (all from GENERATION_CONSTANTS):
        #   crossfade_frames:  GENERATION_CONSTANTS["crossfade_frames"]
        #   native_fps:        GENERATION_CONSTANTS["native_fps"]
        #   seam_count:        GENERATION_CONSTANTS["seam_count"]
        #   clip_paths:        clip_paths (passed in — 3 native 24fps clips)
        #   output_path:       output_path (passed in)
        # Approach:
        #   1. RIFE 4.6 optical-flow blend for crossfade windows
        #   2. FFmpeg 6.0 concat + overlay filter for final assembly
        raise NotImplementedError(
            "Live crossfade join not yet implemented. "
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
      3. Generate clip 0 (T2V — base clip, no conditioning).
      4. Extract last frame of clip 0 → conditioning input for clip 1.
      5. Generate clip 1 (I2V — ext_1, conditioned on last frame of clip 0).
      6. Extract last frame of clip 1 → conditioning input for clip 2.
      7. Generate clip 2 (I2V — ext_2, conditioned on last frame of clip 1).
      8. No interpolation — 24fps native output used directly.
         See interpolate_clip() if target_fps ever changes to 30.
      9. Join with crossfade → single raw loop file.
      10. Assemble and return the result + generation_log dict.

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

        clip_seeds = [seed, seed + 1, seed + 2]
        clip_names = ["base_clip", "ext_1", "ext_2"]
        generation_modes_map = GC["generation_modes"]
        generation_modes = [
            generation_modes_map["base_clip"],      # "T2V"
            generation_modes_map["extension_1"],    # "I2V"
            generation_modes_map["extension_2"],    # "I2V"
        ]

        raw_clip_paths = []
        conditioning_frame = None  # None for T2V base clip

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
                conditioning_frame=conditioning_frame,
            )
            raw_clip_paths.append(out_path)

            # Extract last frame for I2V conditioning of the next clip
            if idx < len(clip_seeds) - 1:
                conditioning_frame = _extract_last_frame(out_path)

        # ── No interpolation — 24fps native output used directly ──────────────
        # See interpolate_clip() if target_fps ever changes to 30.

        # ── Crossfade join ────────────────────────────────────────────────────
        raw_loop_path = raw_dir / f"bg_{run_id}_raw_loop.mp4"
        join_result   = crossfade_join(
            clip_paths=raw_clip_paths,
            output_path=raw_loop_path,
            dry_run=dev_mode,
        )

        # ── Assemble return dict ──────────────────────────────────────────────
        generation_log = {
            "run_id":                    run_id,
            "seed":                      seed,
            "dev_mode":                  dev_mode,
            "clips_generated":           3,
            "seeds_used":                clip_seeds,
            "generation_modes":          generation_modes,
            "native_fps":                GC["native_fps"],
            "target_fps":                GC["target_fps"],
            "interpolation":             "none",
            "crossfade_frames":          GC["crossfade_frames"],
            "seam_frames_raw":           join_result["seam_frames_raw"],
            "seam_frames_playable":      join_result["seam_frames_playable"],
            "total_frames_raw":          join_result["total_frames_raw"],
            "playable_frames":           join_result["playable_frames"],
            "compiled_input_hash_short": compiled["input_hash_short"],
            "compiler_version":          compiled["compiler_version"],
            "status":                    "complete",
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


# ── interpolate_clip() — OPTIONAL ─────────────────────────────────────────────

def interpolate_clip(
    input_path:  Path,
    output_path: Path,
    dry_run:     bool,
) -> Path:
    """
    STATUS: OPTIONAL — not called in default 24fps pipeline.

    Wan2.2-TI2V-5B outputs 24fps natively. Interpolation
    to a higher fps target is not required. This function
    is retained for the contingency case where target_fps
    is changed to 30fps in generation_constants.json.

    If called, it performs the same frame-duplication stub
    in dry_run mode as before. The live TODO block
    references RIFE 4.6 as before.

    To re-enable: call it in run_generation() between
    generate_clip() and crossfade_join().
    """
    import math

    if dry_run:
        GC           = GENERATION_CONSTANTS
        native_fps   = GC["native_fps"]
        target_fps   = GC["target_fps"]
        target_frames = GC["base_clip_frames_native"] * target_fps // native_fps

        ratio   = target_fps / native_fps
        repeats = math.ceil(ratio)

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
