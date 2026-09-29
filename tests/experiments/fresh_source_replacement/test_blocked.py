from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fresh_source_replacement.runner import run_blocked


def test_blocked_branch_never_confirms_replacement(tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_text(json.dumps({"status": "BLOCKED_NEW_SOURCE", "blockers": ["short"]}))
    assert run_blocked(tmp_path) == 0
    final = json.loads((tmp_path / "run/A/final.json").read_text())
    assert final["replacement_confirmed"] is False
    assert final["formal_cells"] == 0
