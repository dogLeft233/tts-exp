from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fresh_source_timing.runner import run_blocked


def test_timing_blocked_does_not_create_cells(tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_text(json.dumps({"status": "BLOCKED_NEW_SOURCE", "blockers": ["short"]}))
    assert run_blocked(tmp_path) == 0
    delay = json.loads((tmp_path / "run/C/delay_inputs.json").read_text())
    assert delay["delay_samples"] == 3200
    assert delay["groups"] == []
