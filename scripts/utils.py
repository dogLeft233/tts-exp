"""Shared helpers for pipeline scripts."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(repo_root: Path, override_path: str | Path | None = None) -> dict[str, Any]:
    base_path = repo_root / "scripts" / "config.yaml"
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
    if not override_path:
        return base or {}
    override = Path(override_path)
    if not override.is_absolute():
        override = repo_root / override
    if not override.exists():
        raise FileNotFoundError(f"config override not found: {override}")
    return deep_merge(base or {}, yaml.safe_load(override.read_text()) or {})


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def detect_sample_ids(
    repo_root: Path,
    smoke: bool = False,
    cfg: dict[str, Any] | None = None,
    audio_dir: str | Path | None = None,
) -> list[int]:
    samples_cfg = (cfg or {}).get("samples", {})
    configured = samples_cfg.get("ids")
    if configured:
        ids = sorted({int(sample_id) for sample_id in configured})
    else:
        configured_audio = audio_dir or (cfg or {}).get("paths", {}).get(
            "audio_dir", "data/data/audio"
        )
        source_dir = resolve_repo_path(repo_root, configured_audio)
        ids = []
        for wav_path in source_dir.glob("*.wav"):
            match = re.match(r"^(\d+)\.wav$", wav_path.name)
            if match:
                ids.append(int(match.group(1)))
        ids.sort()
    if smoke:
        smoke_id = samples_cfg.get("smoke_id")
        if smoke_id is not None and int(smoke_id) in ids:
            return [int(smoke_id)]
        return ids[:1]
    return ids
