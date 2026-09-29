"""Manifest-driven static rendering; all cells consume only PNG and PCM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import canonical_hash, file_sha256, read_json, write_json
from .render_worker import render_request


def render_cell(row: Mapping[str, Any], *, portrait_id: str, video_arm: str, audio_path: str, output_dir: str | Path, checkpoint: str | Path, ffmpeg: str | Path, device: str, python: str | Path, seed: int = 42) -> dict[str, Any]:
    portrait = row["portraits"][portrait_id]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    request = {"portrait_path": portrait["path"], "portrait_rgb_sha256": portrait["rgb_pixel_sha256"], "audio_path": audio_path, "audio_sha256": file_sha256(Path(audio_path).resolve()), "box_xyxy": portrait["generation_box_xyxy"], "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(Path(checkpoint).resolve()), "ffmpeg": str(ffmpeg), "ffmpeg_sha256": file_sha256(Path(ffmpeg).resolve()), "outfile": str((target / f"{row['pair_id']}__{portrait_id}__{video_arm}.mkv").resolve()), "result": str((target / f"{row['pair_id']}__{portrait_id}__{video_arm}.json").resolve()), "batch_size": 4, "seed": int(seed), "device": device, "source_frame_indices": [0], "python": str(python)}
    request_hash = canonical_hash(request)
    result_path = Path(str(request["result"]))
    output_path = Path(str(request["outfile"]))
    if result_path.is_file() and output_path.is_file():
        cached = read_json(result_path)
        if cached.get("request_sha256") == request_hash and cached.get("output_sha256") == file_sha256(output_path) and cached.get("image_rgb_sha256") == portrait["rgb_pixel_sha256"] and cached.get("request_inputs", {}).get("audio_sha256") == request["audio_sha256"] and cached.get("request_inputs", {}).get("checkpoint_sha256") == request["checkpoint_sha256"] and cached.get("outside_generation_box_identity") is True and int(cached.get("decoded_frame_count", 0)) > 0:
            result = cached
        else:
            raise RuntimeError(f"render cache binding mismatch; refusing reuse: {output_path}")
    elif result_path.exists() or output_path.exists():
        raise RuntimeError(f"partial render artifact cannot be resumed: {output_path}")
    else:
        result = render_request(request)
    if result.get("request_sha256") != request_hash:
        raise RuntimeError("render worker request hash mismatch")
    result.update({"pair_id": row["pair_id"], "portrait_id": portrait_id, "video_arm": video_arm, "audio_arm": video_arm, "request_fields": sorted(request)})
    write_json(target / f"{row['pair_id']}__{portrait_id}__{video_arm}__cell.json", result)
    return result


__all__ = ["render_cell"]
