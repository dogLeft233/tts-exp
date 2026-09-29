"""Static-only Wav2Lip request validator and subprocess boundary."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .assets import FORBIDDEN_FIELDS
from .config import REPO_ROOT, canonical_hash, file_sha256, write_json

ALLOWED_FIELDS = frozenset({"portrait_path", "portrait_rgb_sha256", "audio_path", "audio_sha256", "box_xyxy", "checkpoint", "checkpoint_sha256", "ffmpeg", "ffmpeg_sha256", "outfile", "result", "batch_size", "seed", "device", "source_frame_indices", "python"})


def validate_request(request: Mapping[str, Any]) -> None:
    if set(request).intersection(FORBIDDEN_FIELDS) or set(request) - ALLOWED_FIELDS:
        raise ValueError("static render request contains unknown or video-derived fields")
    required = {"portrait_path", "portrait_rgb_sha256", "audio_path", "audio_sha256", "box_xyxy", "checkpoint", "checkpoint_sha256", "ffmpeg", "ffmpeg_sha256", "outfile", "result"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"static render request missing fields: {sorted(missing)}")
    if request.get("source_frame_indices") not in (None, [0]):
        raise ValueError("static renderer accepts only repeated source frame 0")
    portrait = Path(str(request["portrait_path"])).resolve()
    audio = Path(str(request["audio_path"])).resolve()
    if not portrait.is_file() or not audio.is_file():
        raise FileNotFoundError("static portrait/audio input missing")
    if not Path(str(request.get("checkpoint", ""))).is_file() or not Path(str(request.get("ffmpeg", ""))).is_file():
        raise FileNotFoundError("static render checkpoint/ffmpeg missing")
    if len(request["box_xyxy"]) != 4:
        raise ValueError("box_xyxy must contain four coordinates")
    if str(request["audio_sha256"]) != file_sha256(audio) or str(request["checkpoint_sha256"]) != file_sha256(Path(str(request["checkpoint"])).resolve()) or str(request["ffmpeg_sha256"]) != file_sha256(Path(str(request["ffmpeg"])).resolve()):
        raise ValueError("static render dependency hash mismatch")
    values = [int(value) for value in request["box_xyxy"]]
    if not (0 <= values[0] < values[2] and 0 <= values[1] < values[3]):
        raise ValueError("generation box must be a positive rectangle")


def render_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validate_request(request)
    portrait = Path(str(request["portrait_path"])).resolve()
    audio = Path(str(request["audio_path"])).resolve()
    worker = REPO_ROOT / "scripts/experiments/phone_gain_static_tfg_mfa/static_render_worker.py"
    command = [str(Path(request.get("python", REPO_ROOT / ".venv/bin/python"))), str(worker), "--image", str(portrait), "--image-rgb-sha256", str(request["portrait_rgb_sha256"]), "--audio", str(audio), "--box", *[str(int(value)) for value in request["box_xyxy"]], "--checkpoint", str(Path(request["checkpoint"]).resolve()), "--ffmpeg", str(Path(request["ffmpeg"]).resolve()), "--outfile", str(Path(request["outfile"]).resolve()), "--result", str(Path(request["result"]).resolve()), "--batch-size", str(int(request.get("batch_size", 4))), "--seed", str(int(request.get("seed", 42))), "--device", str(request.get("device", "cuda"))]
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    sidecar = Path(str(request["result"]))
    if result.returncode != 0 or not sidecar.is_file():
        raise RuntimeError(f"static Wav2Lip failed: {result.stderr[-2000:]}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("input_mode") != "one_png_only" or any(int(value) != 0 for value in payload.get("source_frame_indices", [])):
        raise RuntimeError("static worker did not prove repeated frame 0")
    payload["request_sha256"] = canonical_hash(dict(request))
    payload["request_inputs"] = {"portrait_sha256": file_sha256(portrait), "audio_sha256": file_sha256(audio), "checkpoint_sha256": file_sha256(Path(str(request["checkpoint"])).resolve())}
    # The worker's sidecar is the cache receipt.  Persist the parent request
    # binding there as well as in the returned object so a later invocation
    # cannot mistake a stale worker result for a valid cell.
    write_json(sidecar, payload)
    return payload


__all__ = ["ALLOWED_FIELDS", "render_request", "validate_request"]
