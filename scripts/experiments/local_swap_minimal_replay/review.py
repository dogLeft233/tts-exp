from __future__ import annotations

import html
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import file_sha256, write_self_hashed_json


def create_playback(paths: config.RunPaths, media_manifest: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    paths.playback.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    counter = 0
    truth_map: dict[str, Any] = {}
    wanted = {("A", "G_N"), ("A", "G_S"), ("B", "R_N0"), ("B", "R_S0"), ("B", "Rs_N0"), ("B", "Rs_S0")}
    for row in media_manifest.get("cells", []):
        key = (str(row.get("family")), str(row.get("cell")))
        if key not in wanted or row.get("status") != "ready":
            continue
        counter += 1
        blind_id = f"clip_{counter:03d}"
        relative_media = os.path.relpath(str(row["media"]), start=str(paths.playback))
        entry = {
            "blind_id": blind_id,
            "sample_id": str(row["sample_id"]),
            "family": str(row["family"]),
            "cell": str(row["cell"]),
            "media": relative_media,
            "media_sha256": str(row["media_sha256"]),
            "time_support": support.get("samples", {}).get(str(row["sample_id"]), {}).get("B", {}).get("common_support"),
            "human_observation": None,
        }
        entries.append(entry)
        truth_map[blind_id] = {key: value for key, value in entry.items() if key not in {"human_observation"}}
    (paths.review / "blind_labels.json").parent.mkdir(parents=True, exist_ok=True)
    write_self_hashed_json(paths.review / "blind_labels.json", {"schema_version": 1, "status": "NOT_HUMAN_REVIEWED", "blind_seed": 20260913, "entries": [{"blind_id": row["blind_id"], "human_observation": None} for row in entries]})
    write_self_hashed_json(paths.review / "condition_map.json", {"schema_version": 1, "status": "REVEAL_AFTER_REVIEW", "entries": list(truth_map.values())})
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>LOCAL_SWAP blind playback</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}article{display:inline-block;vertical-align:top;width:31%;margin:1%;background:#f5f5f5;padding:.5rem}video{width:100%}.note{color:#666}</style>",
        "<h1>LOCAL_SWAP blind playback</h1>",
        "<p class='note'>本页只显示中性编号；人工观看状态：未人工核验。条件映射保存在 review/condition_map.json，复核后再揭示。</p>",
    ]
    for entry in entries:
        note = "B 对照：交换段为中间两段；A 对照：历史生成视频。"
        lines.extend([
            "<article>",
            f"<h2>{html.escape(str(entry['blind_id']))}</h2>",
            f"<p>{html.escape(note)}</p>",
            f"<video controls preload='metadata' src='{html.escape(str(entry['media']))}'></video>",
            "<p>观察记录：未填写</p>",
            "</article>",
        ])
    (paths.playback / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete" if entries else "blocked",
        "human_review_status": "NOT_HUMAN_REVIEWED",
        "entry_count": len(entries),
        "expected_minimum_entries": len(config.SAMPLE_IDS) * 6,
        "index": str((paths.playback / "index.html").resolve()),
        "index_sha256": file_sha256(paths.playback / "index.html"),
        "blind_labels": str((paths.review / "blind_labels.json").resolve()),
        "condition_map": str((paths.review / "condition_map.json").resolve()),
        "entries": entries,
    }
    write_self_hashed_json(paths.review / "manifest.json", manifest)
    return manifest
