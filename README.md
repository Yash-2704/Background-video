# Background Video Generation Module

This module is one component of an automated broadcast production system. Given 6 editorial inputs, it generates a broadcast-ready looping background video asset — an 18-second seamlessly looping clip designed to play behind a news anchor at 1920×1080. Every generation parameter is locked in version-controlled config files so that any two runs with identical editorial inputs produce deterministically comparable outputs; all variation in output is attributable to editorial input, not pipeline noise.

## Setup

```bash
# 1. Clone the repository and enter the module directory
cd "Background Video"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set BASE_DIR to the absolute path of this directory
```

## Validate your environment

Before running any pipeline step, confirm your environment matches the locked constants:

```bash
python validate_environment.py
```

The script checks Python version, package versions, ffmpeg, CUDA availability, the CogVideoX commit hash lock, config file integrity, and the output directory. It exits with code 0 if there are no failures (warnings are advisory), or code 1 if any hard requirement is unmet.

## Run tests

```bash
pytest tests/
```

All 8 tests verify the structural integrity and logical invariants of the JSON config files. They must pass before any pipeline code is executed.

## Dev phase cap

During development, `extensions_per_clip` is set to **2** (see `config/generation_constants.json`). This limits generation time and compute cost while the pipeline is being validated. For production, set `extensions_per_clip` to **4** and `dev_mode` to **false** as noted in the `production_note` field.
