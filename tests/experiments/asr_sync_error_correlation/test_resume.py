import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.asr_sync_error_correlation.config import RuntimeOptions
from scripts.experiments.asr_sync_error_correlation.io import (
    append_failure,
    atomic_write_json,
    atomic_write_npz,
    read_json,
    read_npz,
    success_marker_valid,
    write_success_marker,
)
from scripts.experiments.asr_sync_error_correlation.run import _run_dir_guard


def test_finite_json_and_npz_contract(tmp_path: Path):
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"x": float("nan")})
    path = tmp_path / "data.npz"
    atomic_write_npz(path, {"x": np.array([1.0, 2.0])})
    assert read_npz(path, required=("x",))["x"].tolist() == [1.0, 2.0]
    with pytest.raises(TypeError):
        atomic_write_npz(tmp_path / "obj.npz", {"x": np.array([object()], dtype=object)})


def test_marker_rejects_stale_binding_and_corrupt_output(tmp_path: Path):
    output = tmp_path / "out.json"
    marker = tmp_path / "cell.json"
    atomic_write_json(output, {"ok": True})
    write_success_marker(marker, stage="x", sample_id="s", arm="natural", bindings={"config": "a"}, outputs={"record": output})
    assert success_marker_valid(marker, stage="x", sample_id="s", arm="natural", bindings={"config": "a"}, outputs={"record": output})
    assert not success_marker_valid(marker, bindings={"config": "b"}, outputs={"record": output})
    output.write_text("corrupt", encoding="utf-8")
    assert not success_marker_valid(marker, bindings={"config": "a"}, outputs={"record": output})


def test_nonempty_run_requires_resume(tmp_path: Path):
    (tmp_path / "existing").write_text("keep", encoding="utf-8")
    options = RuntimeOptions(tmp_path, tmp_path, None, "cpu", False, False)
    with pytest.raises(RuntimeError):
        _run_dir_guard(options)
    _run_dir_guard(RuntimeOptions(tmp_path, tmp_path, None, "cpu", False, True))


def test_failure_ledger_is_structured(tmp_path: Path):
    path = tmp_path / "failures.json"
    row = append_failure(path, stage="03_asr", sample_id="x", arm="tts", exception=RuntimeError("bad"))
    assert row["exception_type"] == "RuntimeError"
    assert read_json(path)[0]["sample_id"] == "x"
