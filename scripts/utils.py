"""Shared helpers for pipeline scripts."""

import re
from pathlib import Path


def detect_sample_ids(repo_root: Path, smoke: bool = False) -> list[int]:
    """Auto-detect sample IDs from data/data/audio/*.wav files."""
    audio_dir = repo_root / "data" / "data" / "audio"
    ids: list[int] = []
    for f in audio_dir.glob("*.wav"):
        m = re.match(r"^(\d+)\.wav$", f.name)
        if m:
            ids.append(int(m.group(1)))
    ids.sort()
    if smoke:
        return ids[:1]
    return ids
