#!/usr/bin/env python3
"""Run the current SyncNet evaluator with the current local helpers."""

import importlib.util
import runpy
import sys
from pathlib import Path


script_dir = Path(__file__).resolve().parent
helper_path = script_dir / "stableavatar_utils.py"
spec = importlib.util.spec_from_file_location("utils", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load helper module: {helper_path}")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
sys.modules["utils"] = helpers
runpy.run_path(str(script_dir / "04_eval_stableavatar.py"), run_name="__main__")
