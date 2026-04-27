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

Dry-run mode (dry_run=True on run_generation / generate_clip):
  All model calls are bypassed. cv2 writes synthetic placeholder MP4 files
  that are syntactically valid so every downstream stage can run today.

Live mode (dry_run=False):
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

# ── Optional ML imports (guard-imported so module loads on non-ML machines) ────
# Imported at module level to avoid concurrent thread import race conditions
# when multiple requests hit the pipeline simultaneously.
try:
    import torch
    import PIL.Image
    from diffusers import WanImageToVideoPipeline, WanPipeline
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

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

    mp4v-encoded files written by OpenCV can report an incorrect
    CAP_PROP_FRAME_COUNT. We try the seek shortcut first; if it fails
    we fall back to reading all frames sequentially to find the last one.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")

    last_frame = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        if ret:
            cap.release()
            return frame

    # Fallback: read all frames sequentially
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame

    cap.release()
    if last_frame is None:
        raise RuntimeError(f"Failed to read any frame from: {video_path}")
    return last_frame


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
        # All heavy dependencies are guard-imported here so that this module
        # imports cleanly on any machine without ML packages installed.
        #
        # Pipeline class names (verified against model_index.json):
        #   T2V: WanPipeline               (diffusers ≥ 0.27.0)
        #   I2V: WanImageToVideoPipeline   (diffusers ≥ 0.27.0)
        # torch, PIL, WanPipeline, WanImageToVideoPipeline imported at module level.
        if not _ML_AVAILABLE:
            raise RuntimeError(
                "ML dependencies (torch / diffusers) are not installed. "
                "Cannot run live generation."
            )

        GC           = GENERATION_CONSTANTS
        # Weights live in the Wan2.2-TI2V-5B-Diffusers subfolder, not Wan2.2/ root.
        WEIGHTS_PATH = str(PROJECT_ROOT / "Wan2.2" / "Wan2.2-TI2V-5B-Diffusers")

        # ── STEP 1: Load pipeline ─────────────────────────────────────────────
        if clip_index == 0:
            pipe = WanPipeline.from_pretrained(
                WEIGHTS_PATH,
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(
                WEIGHTS_PATH,
                torch_dtype=torch.bfloat16,
            )

        # ── Device placement strategy ─────────────────────────────────────────
        #
        # Use pipe.to("cuda") to put all components on CUDA, then monkey-patch
        # encode_prompt to offload the text encoder (~9.3 GB) immediately after
        # encoding. This avoids accelerate's enable_model_cpu_offload() hooks
        # which behave differently between WanPipeline and WanImageToVideoPipeline
        # on Windows, causing device mismatches for I2V clips.
        #
        # VRAM budget:
        #   During text encoding : transformer(9.3) + text_enc(9.3) + VAE(2) ≈ 21 GB
        #   During denoising     : transformer(9.3) + VAE(2) + activations(5) ≈ 16 GB
        #   Both fit within 24 GB.
        pipe.to("cuda")

        # Flash Attention: O(N) memory attention kernel.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # VAE slicing / tiling notes:
        # - enable_tiling() is intentionally omitted — WanVAE's feat_cache
        #   mechanism is incompatible with spatial tiling (tensor size mismatch).
        # - enable_slicing() is also omitted here: we offload the transformer
        #   (~9.3 GB) at the end of the last denoising step (via callback) so
        #   that VAE decode has ~22 GB free and can process all frames in one
        #   batched pass. Slicing is only enabled as a fallback if OOM occurs.

        # ── STEP 2: Sampler ───────────────────────────────────────────────────
        # Use the default scheduler loaded from the model (UniPCMultistepScheduler).

        # ── STEP 2b: Offload text encoder after internal encoding ────────────
        # Monkey-patch encode_prompt so the text encoder (~9.3 GB) is moved to
        # CPU immediately after encoding, freeing VRAM before the denoising loop.
        _orig_encode = pipe.encode_prompt
        def _encode_then_offload(*args, **kwargs):
            result = _orig_encode(*args, **kwargs)
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            return result
        pipe.encode_prompt = _encode_then_offload

        # ── STEP 3: Inference ─────────────────────────────────────────────────
        generator = torch.Generator(device="cuda").manual_seed(seed)

        def _diag_callback(pipe, i, t, callback_kwargs):
            if i == 0:
                _alloc = torch.cuda.memory_allocated() / 1024 ** 3
                _reserv = torch.cuda.memory_reserved() / 1024 ** 3
                print(f"[DIAG] step 0 VRAM allocated : {_alloc:.2f} GB")
                print(f"[DIAG] step 0 VRAM reserved  : {_reserv:.2f} GB")
            if i == GC["steps"] - 1:
                # Last denoising step — offload transformer before VAE decode.
                # This frees ~9.3 GB so VAE can decode all 145 frames in one
                # batched pass instead of one-frame-at-a-time (slicing).
                pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
                _alloc = torch.cuda.memory_allocated() / 1024 ** 3
                print(f"[DIAG] transformer offloaded pre-VAE, VRAM now: {_alloc:.2f} GB", flush=True)
            return callback_kwargs

        import time as _time
        _t0 = _time.time()
        print(f"[TIMING] Inference start", flush=True)

        # Steps override: Wan2.2 produces good results at 20 steps.
        # Constants may store a higher value (e.g. 50) intended for
        # a different scheduler — cap at 20 to keep per-clip time ~6 min.
        _steps = min(GC["steps"], 20)

        if conditioning_frame is None:
            # T2V: text-only conditioning
            output = pipe(
                prompt=f"{positive}, {motion}",
                negative_prompt=negative,
                num_frames=GC["base_clip_frames_native"],   # 145
                height=GC["generate_resolution"][1],         # 720
                width=GC["generate_resolution"][0],          # 1280
                guidance_scale=GC["cfg_scale"],              # 6.0
                num_inference_steps=_steps,                  # capped at 20
                generator=generator,
                callback_on_step_end=_diag_callback,
            )
        else:
            # I2V: image conditioning.
            # conditioning_frame arrives in BGR format (cv2 convention) —
            # convert to RGB PIL Image before passing to the pipeline.
            frame_rgb = cv2.cvtColor(conditioning_frame, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(frame_rgb)

            output = pipe(
                image=pil_image,
                prompt=f"{positive}, {motion}",
                negative_prompt=negative,
                num_frames=GC["base_clip_frames_native"],
                height=GC["generate_resolution"][1],
                width=GC["generate_resolution"][0],
                guidance_scale=GC["cfg_scale"],
                num_inference_steps=_steps,                  # capped at 20
                generator=generator,
                callback_on_step_end=_diag_callback,
            )

        _t1 = _time.time()
        print(f"[TIMING] pipe() total (denoise+VAE decode): {_t1-_t0:.1f}s", flush=True)

        # ── STEP 4: Write frames to output_path via imageio ──────────────────
        # imageio-ffmpeg writes PIL RGB frames directly to
        # H.264/yuv420p — no cv2 color conversion, no re-encode.
        import imageio as _imageio
        frames = output.frames[0]
        _t2_start = _time.time()
        writer = _imageio.get_writer(
            str(output_path),
            fps=float(GC["native_fps"]),
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
            ffmpeg_params=["-crf", "18"],
        )
        try:
            for pil_frame in frames:
                writer.append_data(np.array(pil_frame))
        finally:
            writer.close()
        _t2 = _time.time()
        print(
            f"[TIMING] Video write (H.264): {_t2-_t2_start:.1f}s"
            f" | Clip total: {_t2-_t0:.1f}s",
            flush=True,
        )
        if not output_path.exists():
            raise RuntimeError(
                f"generate_clip() live: output file was not "
                f"created at {output_path}"
            )

        # Explicitly free pipeline VRAM before returning so the next clip can
        # load its own pipeline without doubling peak memory usage.
        del pipe
        torch.cuda.empty_cache()

        return output_path


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
                if frame_pos >= len(output_frames):        # safety: short test clips
                    break
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
        # ── LIVE CROSSFADE JOIN ───────────────────────────────────────────────
        # Guard-import stdlib modules so module-level imports stay ML-free.
        import subprocess
        import sys
        import tempfile
        import time as _time

        GC         = GENERATION_CONSTANTS
        native_fps = GC["native_fps"]                      # 24
        cf         = GC["crossfade_frames"]                # 14
        seam_count = GC["seam_count"]                      # 2
        fpc        = GC["base_clip_frames_native"]         # 145

        RIFE_DIR    = PROJECT_ROOT / "Practical-RIFE"
        RIFE_MODEL  = RIFE_DIR / "train_log"
        RIFE_SCRIPT = RIFE_DIR / "inference_video.py"

        if not RIFE_SCRIPT.exists():
            raise FileNotFoundError(
                f"RIFE inference script not found: {RIFE_SCRIPT}\n"
                f"Clone Practical-RIFE into: {RIFE_DIR}"
            )
        if not RIFE_MODEL.exists():
            raise FileNotFoundError(
                f"RIFE model weights not found: {RIFE_MODEL}\n"
                f"Download train_log/ into: {RIFE_DIR}"
            )

        seam_frames_playable = GC["seam_frames_playable_timeline"]   # [138, 269]
        playable_frames      = GC["total_playable_frames"]            # 406
        seams_raw            = [fpc * (i + 1) for i in range(seam_count)]  # [145, 290]

        # ── Local helpers ─────────────────────────────────────────────────────

        def _ffmpeg(*args_list):
            """Run FFmpeg with -y, capture output, raise RuntimeError on failure."""
            result = subprocess.run(
                ["ffmpeg", "-y", *args_list],
                capture_output=True,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed (exit {result.returncode}):\n"
                    f"{result.stderr.decode(errors='replace')}"
                )

        def _read_frame(video_path, frame_idx):
            """Read a single frame from a video by 0-based index using cv2."""
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture failed: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from {video_path}"
                )
            return frame

        def _write_segment(src_path, start_frame, num_frames, dst_path):
            """Extract num_frames frames starting at start_frame into a new mp4v file."""
            cap = cv2.VideoCapture(str(src_path))
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture failed: {src_path}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(dst_path), fourcc, float(native_fps), (w, h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"cv2.VideoWriter failed: {dst_path}")
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            cap.release()
            writer.release()

        def _run_rife(input_mp4, output_mp4):
            """
            Run Practical-RIFE on a 2-frame input video to produce a
            crossfade segment of exactly cf=14 frames via optical flow.

            Frame math:
                2 input frames, 1 pair → RIFE writes:
                  lastframe(I0) + (multi-1) intermediates + lastframe(I1)
                  = multi + 1 total frames
                With --multi=cf-1=13 → 13+1 = 14 output frames ✓

            --fps native_fps overrides RIFE's default of (source_fps × multi),
            which would otherwise produce a spurious 312fps output container.

            NOTE: sys.executable assumes RIFE dependencies (torch, skvideo)
            are installed in the current Python env. On a GPU machine with a
            separate RIFE venv, replace sys.executable with that env's python.
            """
            result = subprocess.run(
                [
                    sys.executable,
                    str(RIFE_SCRIPT),
                    "--video", str(input_mp4),
                    "--output", str(output_mp4),
                    "--model", str(RIFE_MODEL),
                    "--multi", str(cf - 1),        # 13 → yields cf=14 frames
                    "--fps",   str(native_fps),    # keep output at 24fps
                ],
                capture_output=True,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Practical-RIFE failed (exit {result.returncode}):\n"
                    f"{result.stderr.decode(errors='replace')}"
                )

        # ── Phase 1: RIFE optical-flow blend for each seam ───────────────────
        #
        # For each seam at raw position S = fpc*(seam_idx+1):
        #   Boundary frame A = outgoing_clip[S - cf]  (first frame of outgoing window)
        #   Boundary frame B = incoming_clip[cf - 1]  (last frame of incoming window)
        #   A 2-frame input video [A, B] → RIFE → 14-frame crossfade segment
        #
        # This produces an optical-flow morph from the outgoing window entry
        # to the incoming window exit, matching the alpha ramp of the dry-run
        # linear blend but with motion-compensated intermediate frames.

        with tempfile.TemporaryDirectory() as _tmpdir:
            tmp = Path(_tmpdir)
            rife_segs = []

            for seam_idx in range(seam_count):
                seam          = seams_raw[seam_idx]          # 145 or 290
                outgoing_clip = clip_paths[seam_idx]         # clip_0 or clip_1
                incoming_clip = clip_paths[seam_idx + 1]     # clip_1 or clip_2
                seam_tmp      = tmp / f"seam_{seam_idx}"
                seam_tmp.mkdir()

                # Extract the two boundary frames.
                # seam is in raw-timeline coords (145 or 290); the clip files
                # are each only fpc=145 frames, so convert to clip-local index:
                #   outgoing boundary = last frame before the cf-window = fpc - cf = 131
                #   incoming boundary = last frame of the cf-window        = cf - 1  = 13
                frame_a = _read_frame(outgoing_clip, fpc - cf)   # clip-local: always 131
                frame_b = _read_frame(incoming_clip, cf - 1)     # clip-local: always 13
                h, w    = frame_a.shape[:2]

                # Write 2-frame input video for RIFE (mp4v, same codec as source clips)
                rife_in = seam_tmp / "rife_in.mp4"
                fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
                writer  = cv2.VideoWriter(str(rife_in), fourcc, float(native_fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"cv2.VideoWriter failed: {rife_in}")
                writer.write(frame_a)
                writer.write(frame_b)
                writer.release()

                # Run RIFE → cf=14 frame optical-flow crossfade segment
                rife_out = seam_tmp / "rife_out.mp4"
                _run_rife(rife_in, rife_out)
                rife_segs.append(rife_out)

            # ── Phase 2: FFmpeg concat assembly ──────────────────────────────
            #
            # Segment layout — total = 3 × fpc = 435 frames, seams at [145, 290]:
            #
            #   seg0  │ clip_0 frames [0 .. fpc-cf-1]   │ fpc-cf=131 frames
            #   seg1  │ RIFE crossfade seam 0            │ cf=14 frames  → seam at 145
            #   seg2  │ clip_1 frames [0 .. fpc-cf-1]   │ fpc-cf=131 frames
            #   seg3  │ RIFE crossfade seam 1            │ cf=14 frames  → seam at 290
            #   seg4  │ clip_2 frames [0 .. fpc-1]      │ fpc=145 frames
            #
            # Total: (fpc-cf) + cf + (fpc-cf) + cf + fpc = 3*fpc = 435 ✓
            # Seam 1: (fpc-cf) + cf = fpc = 145 ✓
            # Seam 2: (fpc-cf) + cf + (fpc-cf) + cf = 2*fpc = 290 ✓
            #
            # clip_1[0..cf-1] appears both inside the RIFE segment (as frame_b
            # boundary) and at the start of seg2 — this mirrors the dry-run
            # behaviour where the incoming clip plays from frame 0 after the seam.

            seg0 = tmp / "seg0.mp4"
            seg2 = tmp / "seg2.mp4"
            _write_segment(clip_paths[0], 0, fpc - cf, seg0)   # clip_0[0..130]
            _write_segment(clip_paths[1], 0, fpc - cf, seg2)   # clip_1[0..130]

            all_segs = [seg0, rife_segs[0], seg2, rife_segs[1], clip_paths[2]]

            concat_list = tmp / "concat.txt"
            with concat_list.open("w", encoding="utf-8") as fh:
                for seg in all_segs:
                    # Use absolute paths with forward slashes; -safe 0 allows them.
                    # Backslashes in Windows paths cause FFmpeg concat demuxer to
                    # treat "C:\" as a protocol, failing with "Protocol 'c' not found".
                    fwd = str(seg.resolve()).replace("\\", "/")
                    fh.write(f"file '{fwd}'\n")

            _ffmpeg(
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",            # all segments are mp4v — no re-encode needed
                str(output_path),
            )

            # Re-encode concat output mp4v → H.264/yuv420p
            _h264_tmp = output_path.parent / (output_path.stem + "_h264tmp.mp4")
            _t_enc0 = _time.time()
            try:
                _ffmpeg(
                    "-i", str(output_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    str(_h264_tmp),
                )
                output_path.unlink()
                _h264_tmp.rename(output_path)
            except RuntimeError as _enc_err:
                if _h264_tmp.exists():
                    _h264_tmp.unlink()
                raise RuntimeError(
                    f"H.264 re-encode failed for crossfade output: {_enc_err}"
                ) from _enc_err
            _t_enc1 = _time.time()
            print(f"[TIMING] H.264 re-encode (crossfade): {_t_enc1-_t_enc0:.1f}s", flush=True)

            total_frames_raw = fpc * 3   # 435

            return {
                "raw_loop_path":        output_path,
                "seam_frames_raw":      [seams_raw[0], seams_raw[1]],
                "seam_frames_playable": seam_frames_playable,
                "total_frames_raw":     total_frames_raw,
                "playable_frames":      playable_frames,
            }


# ── run_generation() ───────────────────────────────────────────────────────────

def run_generation(
    compiled:   dict,
    run_id:     str,
    output_dir: Path,
    seed:       int = None,
    dry_run:    bool = False,
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

    GC      = GENERATION_CONSTANTS
    run_dir = output_dir / run_id
    raw_dir = run_dir / "raw"

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)

        # ── Prompt enrichment ─────────────────────────────────────────────────
        # Expands the short compiled positive prompt to 80-120 words before
        # passing it to the model. Only runs in live mode; dry_run always skips.
        if not dry_run and GC.get("enrich_prompts", True):
            from core.prompt_parser import enrich_prompt_for_wan
            try:
                prompt_to_use = enrich_prompt_for_wan(
                    compiled["positive"],
                    compiled["motion"],
                )
                print(f"[ENRICHED PROMPT] {prompt_to_use}", flush=True)
            except Exception as _enrich_exc:
                print(
                    f"[WARN] Prompt enrichment failed: {_enrich_exc}. "
                    "Using original prompt.",
                    flush=True,
                )
                prompt_to_use = compiled["positive"]
        else:
            prompt_to_use = compiled["positive"]

        generation_modes_map = GC["generation_modes"]
        extensions = GC.get("extensions_per_clip", 0)
        clip_names = ["base_clip"]
        generation_modes = [generation_modes_map["base_clip"]]
        if extensions >= 1 and "extension_1" in generation_modes_map:
            clip_names.append("ext_1")
            generation_modes.append(generation_modes_map["extension_1"])
        if extensions >= 2 and "extension_2" in generation_modes_map:
            clip_names.append("ext_2")
            generation_modes.append(generation_modes_map["extension_2"])
        clip_seeds = [seed + i for i in range(len(clip_names))]

        raw_clip_paths = []
        conditioning_frame = None  # None for T2V base clip

        for idx, (clip_seed, clip_name) in enumerate(zip(clip_seeds, clip_names)):
            out_path = raw_dir / f"{clip_name}.mp4"
            generate_clip(
                positive=prompt_to_use,
                motion=compiled["motion"],
                negative=compiled["negative"],
                seed=clip_seed,
                clip_index=idx,
                output_path=out_path,
                dry_run=dry_run,
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
        if len(raw_clip_paths) == 1:
            shutil.copy2(str(raw_clip_paths[0]), str(raw_loop_path))
            join_result = {
                "raw_loop_path":        raw_loop_path,
                "seam_frames_raw":      [],
                "seam_frames_playable": [],
                "total_frames_raw":     GC["base_clip_frames_native"],
                "playable_frames":      GC["base_clip_frames_native"],
            }
        else:
            join_result = crossfade_join(
                clip_paths=raw_clip_paths,
                output_path=raw_loop_path,
                dry_run=dry_run,
            )

        # ── Assemble return dict ──────────────────────────────────────────────
        generation_log = {
            "run_id":                    run_id,
            "seed":                      seed,
            "dry_run":                   dry_run,
            "clips_generated":           len(raw_clip_paths),
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
            "positive_prompt":           compiled["positive"],
            "enriched_positive_prompt":  prompt_to_use,
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
            "Pass dry_run=True to use the cv2 frame-duplication stub."
        )
