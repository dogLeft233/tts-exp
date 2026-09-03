"""Dedicated SyncNet-environment adapter and subprocess command construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .io import atomic_write_json, atomic_write_npz, file_sha256
    from .local_sync import select_track
except ImportError:
    def file_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _atomic(path: Path, writer: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
        _atomic(Path(path), lambda handle: handle.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))

    def atomic_write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
        _atomic(Path(path), lambda handle: np.savez_compressed(handle, **arrays))

    def select_track(tracks: Any, *, min_track: int = 50) -> dict[str, Any]:
        raw_pairs = [(int(k), v) for k, v in tracks.items()] if isinstance(tracks, Mapping) else list(enumerate(tracks))
        pairs = []
        for index, value in raw_pairs:
            track = value.get("track") if isinstance(value, Mapping) and "frame" not in value else value
            if not isinstance(track, Mapping) or "frame" not in track:
                raise ValueError(f"track {index} has no frame list")
            pairs.append((index, track))
        eligible = [(k, v) for k, v in pairs if len(v["frame"]) > min_track]
        if not eligible:
            raise ValueError(f"no face track longer than min_track={min_track}")
        index, selected = min(eligible, key=lambda pair: (-len(pair[1]["frame"]), pair[0]))
        ranges = []
        for track_index, track in sorted(pairs):
            frames = list(track["frame"])
            ranges.append({"track_index": track_index, "frame_count": len(frames), "start_frame": int(frames[0]), "end_frame": int(frames[-1]), "eligible": len(frames) > min_track})
        frames = list(selected["frame"])
        return {"selected_track_index": index, "selected_frame_count": len(frames), "track_start_frame": int(frames[0]), "track_end_frame": int(frames[-1]), "track_ranges": ranges, "selection_reason": "max_frame_count_then_min_track_index"}


def build_pipeline_command(
    *,
    syncnet_python: str | Path,
    syncnet_dir: str | Path,
    video_path: str | Path,
    reference: str,
    data_dir: str | Path,
    min_track: int = 50,
) -> list[str]:
    return [
        str(syncnet_python), "run_pipeline.py", "--videofile", str(video_path),
        "--reference", reference, "--data_dir", str(data_dir), "--min_track", str(min_track), "--overwrite",
    ]


def build_adapter_command(
    *,
    syncnet_python: str | Path,
    adapter_path: str | Path,
    video_path: str | Path,
    reference: str,
    data_dir: str | Path,
    model_path: str | Path,
    output_npz: str | Path,
    output_json: str | Path,
    batch_size: int = 20,
    vshift: int = 15,
    device: str = "auto",
) -> list[str]:
    return [
        str(syncnet_python), str(adapter_path), "--videofile", str(video_path), "--reference", reference,
        "--data-dir", str(data_dir), "--initial-model", str(model_path), "--output-npz", str(output_npz),
        "--output-json", str(output_json), "--batch-size", str(batch_size), "--vshift", str(vshift), "--device", device,
    ]


def _load_tracks(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _find_crop(data_dir: Path, reference: str) -> Path:
    candidates = sorted((data_dir / "pycrop" / reference).glob("0*.avi"))
    if not candidates:
        raise FileNotFoundError(f"no SyncNet crop AVI in {data_dir / 'pycrop' / reference}")
    return candidates[0]


def export_distances(
    *,
    videofile: Path,
    reference: str,
    data_dir: Path,
    initial_model: Path,
    output_npz: Path,
    output_json: Path,
    batch_size: int = 20,
    vshift: int = 15,
    device: str = "auto",
) -> dict[str, Any]:
    """Run the unchanged vendor evaluator once and export its complete distance matrix."""
    import sys
    import torch
    vendor_dir = Path(__file__).resolve().parents[3] / "third_party" / "syncnet_python"
    if vendor_dir.is_dir() and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    from SyncNetInstance import SyncNetInstance

    track_path = data_dir / "pywork" / reference / "tracks.pckl"
    tracks = _load_tracks(track_path)
    track = select_track(tracks, min_track=50)
    chosen_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if chosen_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SyncNet CUDA requested but unavailable")
    crop_path = _find_crop(data_dir, reference)
    options = SimpleNamespace(
        tmp_dir=str(data_dir / "pytmp"), reference=reference, batch_size=int(batch_size), vshift=int(vshift),
    )
    evaluator = SyncNetInstance(device=chosen_device)
    evaluator.loadParameters(str(initial_model))
    offset, confidence, distances = evaluator.evaluate(options, videofile=str(crop_path))
    matrix = np.asarray(distances, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2 * int(vshift) + 1 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"SyncNet returned invalid distance matrix: {matrix.shape}")
    values_tensor = torch.from_numpy(matrix.T.copy())
    mdist_tensor = torch.mean(values_tensor, dim=1)
    upstream_j_star = int(torch.argmin(mdist_tensor).item())
    upstream_sync_d = float(mdist_tensor[upstream_j_star].item())
    atomic_write_npz(output_npz, {"dists": matrix})
    record = {
        **track,
        "videofile": str(videofile),
        "crop_path": str(crop_path),
        "track_path": str(track_path),
        "distance_path": str(output_npz),
        "distance_sha256": file_sha256(output_npz),
        "distance_shape": list(matrix.shape),
        "upstream_offset": int(np.asarray(offset).reshape(-1)[0]),
        "upstream_sync_c": float(np.asarray(confidence).reshape(-1)[0]),
        "upstream_sync_d": upstream_sync_d,
        "device": chosen_device,
        "batch_size": int(batch_size),
        "vshift": int(vshift),
        "model_path": str(initial_model),
        "model_sha256": file_sha256(initial_model),
    }
    atomic_write_json(output_json, record)
    return record


def run_subprocess(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(command), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"SyncNet subprocess failed with return code {result.returncode}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videofile", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    export_distances(
        videofile=args.videofile, reference=args.reference, data_dir=args.data_dir,
        initial_model=args.initial_model, output_npz=args.output_npz, output_json=args.output_json,
        batch_size=args.batch_size, vshift=args.vshift, device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
