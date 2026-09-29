from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fresh_source_tts_increment.runner import run_deferred


def test_tts_is_deferred_when_measurement_gate_is_unavailable(tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_text(json.dumps({"status": "BLOCKED_NEW_SOURCE", "blockers": ["short"]}))
    assert run_deferred(tmp_path) == 0
    final = json.loads((tmp_path / "run/D/final.json").read_text())
    assert final["status"] == "DEFERRED_MEASUREMENT"
    assert final["formal_cells"] == 0
