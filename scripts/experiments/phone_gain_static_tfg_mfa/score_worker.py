"""Strict SyncNet scoring request boundary (no target-video input)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .assets import FORBIDDEN_FIELDS

ALLOWED_FIELDS = frozenset({"video_path", "video_arm", "sample_id", "score_box", "audio_paths", "model", "output_dir", "vshift", "batch_size", "python"})
BATCH_ALLOWED_FIELDS = frozenset({"video_paths", "video_sha256", "sample_id", "score_box", "score_box_sha256", "audio_paths", "audio_sha256", "model", "model_sha256", "output_root", "vshift", "batch_size", "python"})


def validate_score_request(request: Mapping[str, Any]) -> None:
    # A generated video is allowed here; the forbidden object is the original
    # LRS3 target video or any path/feature derived from it.
    forbidden_target = {"target_video", "source_video", "lrs3_video", "landmarks", "frame_sequence", "video_features"}
    if set(request).intersection(forbidden_target) or set(request) - ALLOWED_FIELDS:
        raise ValueError("SyncNet request contains forbidden/unknown fields")
    required = {"video_path", "video_arm", "sample_id", "score_box", "audio_paths", "model", "output_dir", "python"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"SyncNet request missing fields: {sorted(missing)}")
    if not Path(str(request["video_path"])).is_file():
        raise FileNotFoundError("generated scoring video is missing")
    if not Path(str(request["model"])).is_file():
        raise FileNotFoundError("SyncNet checkpoint is missing")
    if not Path(str(request["python"])).is_file():
        raise FileNotFoundError("SyncNet Python executable is missing")
    score_box = request["score_box"]
    if not isinstance(score_box, Mapping) or not isinstance(score_box.get("box"), (list, tuple)) or len(score_box["box"]) != 4:
        raise ValueError("SyncNet request must contain a frozen score_box")
    for path in dict(request["audio_paths"]).values():
        if not Path(str(path)).is_file():
            raise FileNotFoundError(f"scoring audio is missing: {path}")


def validate_batch_score_request(request: Mapping[str, Any]) -> None:
    forbidden_target = {"target_video", "source_video", "lrs3_video", "landmarks", "frame_sequence", "video_features"}
    if set(request).intersection(forbidden_target) or set(request) - BATCH_ALLOWED_FIELDS:
        raise ValueError("batch SyncNet request contains forbidden/unknown fields")
    required = {"video_paths", "sample_id", "score_box", "audio_paths", "model", "output_root", "python"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"batch SyncNet request missing fields: {sorted(missing)}")
    for path in dict(request["video_paths"]).values():
        if not Path(str(path)).is_file():
            raise FileNotFoundError(f"generated scoring video is missing: {path}")
    if isinstance(request.get("video_sha256"), Mapping):
        for arm, path in dict(request["video_paths"]).items():
            expected = request["video_sha256"].get(arm)
            if expected is not None:
                from .config import file_sha256

                if file_sha256(Path(str(path))) != str(expected):
                    raise ValueError(f"generated scoring video hash mismatch: {arm}")
    if isinstance(request.get("audio_sha256"), Mapping):
        from .config import file_sha256

        for arm, path in dict(request["audio_paths"]).items():
            expected = request["audio_sha256"].get(arm)
            if expected is not None and file_sha256(Path(str(path))) != str(expected):
                raise ValueError(f"scoring audio hash mismatch: {arm}")
    if request.get("model_sha256") is not None:
        from .config import file_sha256

        if file_sha256(Path(str(request["model"]))) != str(request["model_sha256"]):
            raise ValueError("SyncNet model hash mismatch")
    if request.get("score_box_sha256") is not None:
        from .config import canonical_hash

        if canonical_hash(request["score_box"]) != str(request["score_box_sha256"]):
            raise ValueError("SyncNet score box hash mismatch")
    validate_score_request({
        "video_path": next(iter(dict(request["video_paths"]).values())),
        "video_arm": "BATCH",
        "sample_id": request["sample_id"],
        "score_box": request["score_box"],
        "audio_paths": request["audio_paths"],
        "model": request["model"],
        "output_dir": request["output_root"],
        "vshift": request.get("vshift", 15),
        "batch_size": request.get("batch_size", 20),
        "python": request["python"],
    })


__all__ = ["ALLOWED_FIELDS", "BATCH_ALLOWED_FIELDS", "validate_batch_score_request", "validate_score_request"]
