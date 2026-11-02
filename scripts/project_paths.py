"""Resolve paths from historical manifests against the current checkout."""

from __future__ import annotations

from pathlib import Path


LEGACY_LOCAL_ROOTS = (
    "/mnt/e/Documents/tts-audio/tts-exp",
    "E:/Documents/tts-audio/tts-exp",
)


def is_legacy_project_path(value: str | Path) -> bool:
    raw = str(value).replace("\\", "/").casefold()
    return any(raw == root.replace("\\", "/").casefold() or raw.startswith(root.replace("\\", "/").casefold() + "/")
               for root in LEGACY_LOCAL_ROOTS)


def resolve_project_path(value: str | Path, repo_root: Path) -> Path:
    """Rebase old local-checkout paths, even when the old file still exists.

    Unrelated absolute paths (including the Ubuntu execution host) retain
    their meaning. Historical manifests are never modified in place.
    """
    raw = str(value).replace("\\", "/")
    root = Path(repo_root).resolve()
    for legacy in LEGACY_LOCAL_ROOTS:
        legacy = legacy.replace("\\", "/")
        if raw.casefold() == legacy.casefold():
            return root
        if raw.casefold().startswith(legacy.casefold() + "/"):
            candidate = (root / raw[len(legacy) + 1:]).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("legacy path escapes the current project")
            return candidate
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()
