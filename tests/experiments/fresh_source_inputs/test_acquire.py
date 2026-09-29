from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fresh_source_inputs import acquire


def _seed_local(root: Path, group: str, clip_ids: list[str]) -> None:
    group_root = root / "data/dataset_samples/lrs3/pretrain" / group
    group_root.mkdir(parents=True)
    for clip_id in clip_ids:
        (group_root / f"{clip_id}.mp4").write_bytes(f"video-{group}-{clip_id}".encode())
        (group_root / f"{clip_id}.txt").write_text(f"Text: sample {clip_id}\n", encoding="utf-8")


def test_acquisition_excludes_history_and_freezes_first_eight(tmp_path: Path, monkeypatch) -> None:
    _seed_local(tmp_path, "ABCDEFGHIJK", [f"{index:05d}" for index in range(1, 10)])
    _seed_local(tmp_path, "BCDEFGHIJKL", ["00003", "00001", "00002"])
    monkeypatch.setattr(acquire, "scan_history", lambda *args, **kwargs: {"groups": ["ABCDEFGHIJK"], "scan_file_manifest_sha256": "history", "coverage": {}, "coverage_blockers": []})
    pool = acquire.acquire_pool(tmp_path / "run", repo_root=tmp_path, max_groups=24, min_free_bytes=0)
    assert pool["status"] == "BLOCKED_SOURCE_ACCESS"
    assert pool["group_order"] == ["BCDEFGHIJKL"]
    assert pool["groups"][0]["clip_ids"] == ["00001", "00002", "00003"]
    plan = json.loads((tmp_path / "run/acquisition/plan.json").read_text())
    assert plan["selection"]["clips_per_group"] == 8
    assert "ABCDEFGHIJK" not in pool["group_order"]


def test_acquisition_honors_group_cap(tmp_path: Path, monkeypatch) -> None:
    for index in range(30):
        group = f"{index:011d}"[-11:]
        _seed_local(tmp_path, group, ["00001"])
    monkeypatch.setattr(acquire, "scan_history", lambda *args, **kwargs: {"groups": [], "scan_file_manifest_sha256": "history", "coverage": {}, "coverage_blockers": []})
    pool = acquire.acquire_pool(tmp_path / "run", repo_root=tmp_path, max_groups=24, min_free_bytes=0)
    assert len(pool["groups"]) == 24
    assert pool["group_order"] == sorted(pool["group_order"])


def test_acquisition_plan_is_self_hashed(tmp_path: Path, monkeypatch) -> None:
    _seed_local(tmp_path, "ABCDEFGHIJK", ["00001"])
    monkeypatch.setattr(acquire, "scan_history", lambda *args, **kwargs: {"groups": [], "scan_file_manifest_sha256": "history", "coverage": {}, "coverage_blockers": []})
    acquire.acquire_pool(tmp_path / "run", repo_root=tmp_path, min_free_bytes=0)
    plan = json.loads((tmp_path / "run/acquisition/plan.json").read_text())
    assert plan["artifact_sha256"]
    assert (tmp_path / "run/acquisition/ledger.jsonl").is_file()
