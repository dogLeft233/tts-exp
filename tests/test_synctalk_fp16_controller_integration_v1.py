"""Explicit CUDA integration runner; safe to import during ordinary collection."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/experiments/synctalk_multispeaker_full_20261017/qualify_fp16_controller_fixture_v1.py'), run_name='__main__')