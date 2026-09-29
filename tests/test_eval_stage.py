"""Tests for the parameterized SyncNet evaluation stage."""

from __future__ import annotations

import importlib.util
import json
import sys
from threading import Barrier
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


def test_main_scores_distinct_videos_concurrently_without_mixing_results(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(_mod, "__file__", str(tmp_path / "scripts" / "04_eval.py"))
    monkeypatch.setattr(
        _mod,
        "load_config",
        lambda *_: {
            "paths": {"syncnet_repo": "third_party/syncnet_python", "envs_dir": str(tmp_path / "envs")},
            "ditto": {"conditions": ["natural_raw"]},
        },
    )
    monkeypatch.setattr(_mod, "detect_sample_ids", lambda *_, **__: [1, 2])
    monkeypatch.setattr(
        _mod,
        "find_videos",
        lambda *_: {"natural_raw": {1: tmp_path / "1.mp4", 2: tmp_path / "2.mp4"}},
    )
    simultaneous = Barrier(2, timeout=2)
    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append((args[3], args[4], args[5], kwargs["min_track"]))
        simultaneous.wait()
        sid = int(args[3].stem)
        return f"Confidence: {sid}.125\nMin dist: 8.250\nAV offset: 1", ""

    monkeypatch.setattr(_mod, "run_syncnet_pipeline", fake_pipeline)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--run_id", "parallel_test", "--workers", "2"])

    _mod.main()

    output = tmp_path / "runs" / "parallel_test" / "04_eval"
    meta = json.loads((output / "eval_meta.json").read_text())
    assert meta["workers"] == 2
    assert meta["samples_ok"] == 2
    assert meta["samples_failed"] == 0
    assert meta["complete_case_ids"] == [1, 2]
    assert [meta["results"][f"natural_raw:{sid}"]["sync_c"] for sid in (1, 2)] == [1.125, 2.125]
    assert {call[1] for call in calls} == {
        output / "natural_raw" / str(sid) / "syncnet_data" for sid in (1, 2)
    }
    assert {call[2] for call in calls} == {"natural_raw_1", "natural_raw_2"}
    assert {call[3] for call in calls} == {100}
