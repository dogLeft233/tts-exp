"""Serialization-only repair: preserve frozen analysis; convert NumPy scalars for JSON."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_fixed_natural_holdout80_20260927'
SOURCE = ROOT / 'scripts/experiments/tts_fixed_natural_holdout80_20260927.py'

def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    return value

if __name__ == '__main__':
    scope = runpy.run_path(str(SOURCE), run_name='frozen_analysis')
    scope['protocol']()
    original = scope['write']
    before = hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest()
    scope['analyze'].__globals__['write'] = lambda path, value: original(path, native(value))
    scope['analyze']()
    after = hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest()
    assert before == after, 'Frozen computed summary changed during serialization repair'
    (OUT / 'serialization_repair_result.json').write_text(json.dumps({
        'status': 'PASS', 'summary_before_sha256': before,
        'summary_after_sha256': after, 'summary_bytes_unchanged': True,
        'scope': 'numpy int64 best_lag effect rows converted to builtin integer only',
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
