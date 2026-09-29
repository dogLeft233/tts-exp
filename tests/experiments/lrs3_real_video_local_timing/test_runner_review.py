from pathlib import Path

import pytest

from scripts.experiments.lrs3_real_video_local_timing import config
from scripts.experiments.lrs3_real_video_local_timing.common import (
    DiagnosticError,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_real_video_local_timing.review import (
    create_review_package,
)
from scripts.experiments.lrs3_real_video_local_timing.runner import run


def test_review_order_is_reproducible_and_display_has_no_condition_names(tmp_path: Path) -> None:
    protocol = {"records": [{"sample_id": f"s{index}", "source_group": f"g{index}"} for index in range(4)]}
    media_rows = []
    for index in range(4):
        real = tmp_path / f"real{index}.mkv"
        warp = tmp_path / f"warp{index}.mkv"
        real.write_bytes(f"real-{index}".encode())
        warp.write_bytes(f"warp-{index}".encode())
        media_rows.append({"sample_id": f"s{index}", "arms": {config.REAL_ARM: {"output": str(real)}, config.WARP_ARM: {"output": str(warp)}}})
    result = create_review_package(protocol, {"rows": media_rows}, config.RunPaths(tmp_path / "run"))
    assert result["status"] == "PENDING"
    names = {path.name for path in (tmp_path / "run" / "review").iterdir()}
    assert "form.md" in names
    assert not any(config.REAL_ARM in name or config.WARP_ARM in name for name in names)
    assert config.REAL_ARM not in (tmp_path / "run" / "review" / "form.md").read_text(encoding="utf-8")


def test_runner_preserves_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REPO", tmp_path)
    root = config.run_root_for("terminal")
    write_self_hashed_json(root / "final.json", {"protocol_id": config.PROTOCOL_ID, "status": "complete"})
    before = (root / "final.json").read_bytes()
    with pytest.raises(DiagnosticError, match="terminal run"):
        run("terminal", "all")
    assert (root / "final.json").read_bytes() == before


def test_runner_records_blocked_terminal_without_overwriting_existing_final(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REPO", tmp_path)
    from scripts.experiments.lrs3_real_video_local_timing import runner

    def fail(_stage: str, _paths: config.RunPaths) -> dict[str, object]:
        raise DiagnosticError("synthetic blocked input")

    monkeypatch.setattr(runner, "_run_stage", fail)
    with pytest.raises(DiagnosticError, match="synthetic blocked input"):
        run("blocked", "prepare")
    final = verify_self_hashed_json(config.run_root_for("blocked") / "final.json")
    assert final["status"] == "blocked"
    assert final["scientific_decision"] is None
