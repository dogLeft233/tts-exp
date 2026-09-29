from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, pcm16, sha256_array, load_self_hashed, write_json


def score_metrics(matrix: np.ndarray, rows: tuple[int, ...] | list[int] | None = None) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(value).all():
        raise ProtocolError(f"invalid SyncNet matrix shape: {value.shape}")
    chosen = value if rows is None else value[np.asarray(rows, dtype=np.int64)]
    curve = np.mean(chosen, axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {"curve": [float(x) for x in curve], "min_index": index, "offset": config.VSHIFT - index, "D": float(curve[index]), "C": float(np.median(curve) - curve[index])}


class ScoreEngine:
    def __init__(self) -> None:
        from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

        if file_sha256(config.SYNCNET_MODEL) != config.FIXED_HASHES["syncnet"]:
            raise ProtocolError("SyncNet model hash changed")
        self._scorer = SyncNetScorer(config.SYNCNET_MODEL, "cpu", config.SYNCNET_BATCH_SIZE, config.SYNCNET_THREADS)

    def score(self, media: Path, audio: Path, output_dir: Path, *, sample_id: str, video_arm: str, audio_arm: str, expected_rows: int | None = config.EMBEDDING_ROWS) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar = output_dir / "score.json"
        matrix_path = output_dir / "distance.npy"
        worker_path = output_dir / "worker.json"
        media_hash = file_sha256(media)
        if sidecar.is_file() and matrix_path.is_file() and worker_path.is_file():
            prior = load_self_hashed(sidecar)
            if prior.get("media_sha256") != media_hash or prior.get("source_audio_sha256") != file_sha256(audio):
                raise ProtocolError(f"score binding changed: {sample_id}/{video_arm}")
            return prior
        result = self._scorer.score(media, audio, output_dir / "forward", media_hash, __import__("hashlib").sha256(pcm16(audio)).hexdigest())
        matrix = np.asarray(np.load(result["matrix"], allow_pickle=False), dtype=np.float64)
        if expected_rows is not None and matrix.shape != (expected_rows, config.MATRIX_COLUMNS):
            raise ProtocolError(f"unexpected matrix shape: {matrix.shape}")
        worker = {"schema_version": 1, "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, **result}
        write_json(worker_path, worker)
        payload = {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, "media": str(media.resolve()), "media_sha256": media_hash, "source_audio": str(audio.resolve()), "source_audio_sha256": file_sha256(audio), "source_pcm_sha256": __import__("hashlib").sha256(pcm16(audio)).hexdigest(), "matrix": str(Path(result["matrix"]).resolve()), "matrix_sha256": file_sha256(Path(result["matrix"])), "matrix_shape": list(matrix.shape), "full": score_metrics(matrix), "U": {"rows": list(config.U_ROWS), **score_metrics(matrix, list(config.U_ROWS))}, "worker": str(worker_path.resolve()), "worker_sha256": file_sha256(worker_path), "fresh_forward": True}
        return write_json(sidecar, payload)


def load_matrix(score: dict[str, Any]) -> np.ndarray:
    path = Path(str(score.get("matrix", "")))
    if not path.is_file() or file_sha256(path) != str(score.get("matrix_sha256")):
        raise ProtocolError(f"matrix binding changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.shape != (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS) or not np.isfinite(value).all():
        raise ProtocolError(f"malformed matrix: {path}")
    return value
