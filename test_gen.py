import sys
import traceback
import tempfile
from pathlib import Path

sys.path.insert(0, r"C:\Users\Admin\BackgroundVideo")

from core.prompt_compiler import compile_prompts
from core.generator import generate_clip

compiled = compile_prompts({
    'category': 'Economy',
    'location_feel': 'Urban',
    'time_of_day': 'Dusk',
    'color_temperature': 'Cool',
    'mood': 'Serious',
    'motion_intensity': 'Gentle'
}, compiler_version='1.0.0')

print('Compile OK:', compiled['input_hash_short'])

tmp = Path(tempfile.mkdtemp())
print('Output dir:', tmp)

try:
    generate_clip(
        positive=compiled['positive'],
        motion=compiled['motion'],
        negative=compiled['negative'],
        seed=42819,
        clip_index=0,
        output_path=tmp / 'test_clip.mp4',
        dry_run=False,
    )
    print('SUCCESS')
except Exception:
    traceback.print_exc()
