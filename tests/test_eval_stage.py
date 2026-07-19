"""Tests for the parameterized SyncNet evaluation stage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "04_eval.py"
sys.path.insert(0, str(_SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("_eval_stage", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_run_pipeline_overwrites_existing_face_tracks():
    with patch.object(
        _mod.subprocess,
        "run",
        side_effect=[None, SimpleNamespace(stdout="Confidence: 1\nMin dist: 2", stderr="")],
    ) as run:
        _mod.run_syncnet_pipeline(
            Path("syncnet"), "python", "bin", Path("video.mp4"),
            Path("data"), "reference", Path("model"),
        )
    assert "--overwrite" in run.call_args_list[0].args[0]
