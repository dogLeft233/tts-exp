from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fresh_source_visual.runner import run_blocked


def test_visual_blocked_keeps_human_pending(tmp_path: Path) -> None:
    (tmp_path / "cohort.json").write_text(json.dumps({"status": "BLOCKED_NEW_SOURCE", "blockers": ["short"]}))
    assert run_blocked(tmp_path) == 0
    final = json.loads((tmp_path / "run/B/final.json").read_text())
    assert final["human_status"] == "pending"
    assert final["visual_verified"] is False
