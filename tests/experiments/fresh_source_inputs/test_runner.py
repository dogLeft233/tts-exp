from __future__ import annotations

import pytest

from scripts.experiments.fresh_source_inputs import runner
from scripts.experiments.fresh_source_inputs.protocol import write_json
from scripts.experiments.fresh_source_inputs.runner import run_freeze


def test_freeze_rejects_preliminary_pending_inputs(tmp_path) -> None:
    run_root = tmp_path / "run"
    write_json(run_root / "cohort.json", {"schema_version": 1, "status": "GO"})
    write_json(run_root / "inputs.json", {
        "schema_version": 1,
        "status": "PENDING_CANDIDATE",
        "formal_count": 0,
        "smoke_count": 0,
        "records": [],
    })
    assert run_freeze(run_root) == 2
    assert not (run_root / "run" / "shared" / "freeze.json").exists()


def test_freeze_rejects_zero_record_go_inputs(tmp_path) -> None:
    run_root = tmp_path / "run"
    write_json(run_root / "cohort.json", {"schema_version": 1, "status": "GO"})
    write_json(run_root / "inputs.json", {
        "schema_version": 1,
        "status": "GO",
        "formal_count": 12,
        "smoke_count": 2,
        "records": [],
    })
    assert run_freeze(run_root) == 2
    assert not (run_root / "run" / "shared" / "freeze.json").exists()


def test_resume_requires_execution_contract(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    write_json(run_root / "cohort.json", {"schema_version": 1, "status": "BLOCKED_NEW_SOURCE"})
    write_json(run_root / "inputs.json", {"schema_version": 1, "status": "BLOCKED_NEW_SOURCE"})
    monkeypatch.setattr(runner, "candidate_rows", lambda *args, **kwargs: ([], {"groups": [], "coverage_blockers": []}))
    monkeypatch.setattr(runner, "_execution_contract", lambda *args, **kwargs: {"schema_version": 1, "contract_sha256": "x"})
    with pytest.raises(RuntimeError, match="without an execution contract"):
        runner.run_prepare(run_root, resume=True)
