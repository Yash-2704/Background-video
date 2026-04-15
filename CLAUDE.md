# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A 13-prompt automated broadcast background video generation pipeline. Given 6 editorial inputs (category, location, mood, etc.), it compiles prompts, generates 3 raw clips using Wan2.2-TI2V-5B-Diffusers on a Windows RTX 4090, crossfade-joins them into a looping raw loop, runs quality probes/gates, upscales to 1080p, applies LUT grading, composites, and outputs a broadcast-ready asset with metadata.

**Split architecture**: Backend runs on a Windows GPU machine (`100.86.96.47:8000`). Frontend runs locally on Mac (`localhost:5173`). Only `core/generator.py` ever gets SCP'd to the Windows machine for backend changes.

## Commands

### Backend (run on Windows GPU machine via SSH)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Frontend (run on Mac)
```bash
cd frontend && npm run dev       # dev server on :5173
cd frontend && npm run build     # production build
cd frontend && npm run test      # vitest
```

### Tests (run on Mac, no GPU needed)
```bash
pytest tests/                                        # all tests
pytest tests/test_constants.py                       # single file
pytest tests/test_generator.py::test_generate_clip   # single test
```

### Environment validation (run on Windows GPU machine)
```bash
python validate_environment.py          # checks config keys and values
python scripts/gpu_readiness_check.py   # checks CUDA, weights, FFmpeg, RIFE, ESRGAN
```

### Patch skvideo numpy incompatibility (Windows, one-time fix)
```bash
python -c "import skvideo, pathlib, re; f = pathlib.Path(skvideo.__file__).parent / 'io' / 'abstract.py'; txt = f.read_text(); fixed = re.sub(r'\bnp\.float\b', 'float', re.sub(r'\bnp\.bool\b', 'bool', re.sub(r'\bnp\.complex\b', 'complex', re.sub(r'\bnp\.int\b', 'int', txt)))); f.write_text(fixed); print('Patched:', f)"
```

## Architecture

### Pipeline stages (11, in strict order)
`prompt_compilation` → `generation` → `probe_decode` → `probe_temporal` → `gate_evaluation` → `upscale` → `mask_generation` → `lut_grading` → `composite` → `preview_export` → `metadata_assembly`

The orchestrator (`core/orchestrator.py`) runs all stages, writing live status to `RUN_REGISTRY` (in-memory dict). The FastAPI layer fires the pipeline in a background task and returns immediately; the frontend polls `GET /api/v1/run/{run_id}/status` every 2s.

### Core modules (7)
| Module | Responsibility |
|---|---|
| `core/prompt_compiler.py` | 6 editorial inputs → 3 hashed prompts via pure Python lookup tables (no LLM). Fragment strings are hashed — do **not** alter spacing/punctuation/casing. |
| `core/generator.py` | T2V base clip + 2× I2V extensions via Wan2.2-TI2V-5B. Crossfade join via RIFE optical flow. This is the only file SCP'd to Windows. |
| `core/probes.py` | Decode probe (luminance, frame count) + temporal probe (flicker, warp score, scene cut, loop score). |
| `core/gates.py` | Pure gate evaluation — reads probe dicts, applies 5 thresholds, returns `pass`/`fail`/`human_review`. All thresholds from config, never hardcoded. |
| `core/regenerator.py` | Retry policy engine. `MAX_RETRIES = 0` (disabled — each retry costs ~19 min). |
| `core/post_processor.py` | Upscale (Real-ESRGAN-Video 1.5×), masking, LUT grading, composite, preview export. |
| `core/metadata_assembler.py` | Generates metadata JSON, edit manifest, integration contract. |

### Config system
All pipeline parameters live in two locked JSON files:
- `config/generation_constants.json` — model params, resolution, frame counts, quality gate thresholds, LUT assignments, `dev_mode`
- `config/environment_constants.json` — Python/CUDA/package versions, model commit hash (set on first live run)

Constants are loaded **at module level** on import. Any code reading `GENERATION_CONSTANTS["x"]` is reading from these files. Never hardcode threshold values — always reference the config.

### dev_mode / dry_run
`generation_constants.json` → `"dev_mode": false`. When `true`:
- `generate_clip()` writes synthetic placeholder MP4s using cv2 (no torch/diffusers)
- `run_post_processing()` uses cv2/numpy stubs
- torch/diffusers are **never imported at module level** — only inside the `else` branch of `if dry_run`. This lets the module load on Mac (no ML packages).

### VAE / VRAM constraints (Wan2.2-TI2V-5B)

**The critical VRAM fix: transformer offload before VAE decode**

**Problem**: Decoding 145 frames at 1280×720 requires slicing (one frame at a time) when transformer (~9.3 GB) is pinned in VRAM. This takes **515 seconds per clip** (8.5 min) — 62% of total generation time.

**Root cause**: After denoising completes, the transformer sits idle in VRAM, blocking VAE from batching all frames at once.

**The fix** (`core/generator.py` lines 244–260): In the `_diag_callback` on the last denoising step (`i == GC["steps"] - 1`), move transformer to CPU:
```python
pipe.transformer.to("cpu")
torch.cuda.empty_cache()
```
This frees ~9.3 GB, allowing VAE to decode all 145 frames **in one batched pass** instead of slicing → **reduces VAE decode to ~60 seconds** (7x faster).

**Why the constraints**:
- `enable_tiling()` is **incompatible** with WanVAE — causes `RuntimeError: tensor size mismatch` in `feat_cache`. Never add it.
- `enable_slicing()` is **intentionally omitted** — the transformer offload above makes it unnecessary and would waste the freed VRAM.
- If you revert this offload, VAE decode goes back to 515s/clip and the pipeline takes 40+ min instead of 19 min.

**Don't change this without understanding the impact**: Any modification to VRAM management (re-adding slicing, enabling tiling, removing the offload callback) will either (a) crash with tensor mismatch, or (b) destroy performance.

### Frame math
- 3 clips × 145 frames = 435 frames raw
- Crossfade: 14 frames per seam (`crossfade_frames`), 2 seams
- Seams at raw frames [145, 290]; playable seams at [138, 269]
- Total playable: 406 frames (~16.9s at 24fps)

### Run ID
Derived from `input_hash_short` (first 6 chars of SHA-256 of compiled prompts). Same editorial inputs always produce the same run_id — the 409 deduplication check in `generate_route` relies on this.

## Key Files
- `api/routes/generate.py` — fire-and-forget POST; returns immediately with `status:"running"`
- `api/routes/status.py` — live poll endpoint; spreads full result fields into response when `status=="complete"`
- `api/models.py` — all Pydantic models; `RunStatusResponse` carries result fields for polling
- `frontend/src/api/client.js` — `API_BASE` hardcoded to Windows IP; 409 treated as "already running"
- `frontend/src/components/RunMonitor.jsx` — polls status every 2s; result comes from poll not POST

## Constraints
- **Never add `pipe.vae.enable_tiling()`** — incompatible with WanVAE.
- **Do not alter fragment strings in `prompt_compiler.py`** without regenerating all hashes.
- **`generation_constants.json` and `environment_constants.json` are locked** — changing values requires full pipeline re-validation.
- **`RUN_REGISTRY` is in-memory** — state is lost on uvicorn restart. Don't design features that assume persistence across restarts.
- **Only `core/generator.py` is SCP'd to Windows** — all other backend changes require SCP of the specific changed file.
