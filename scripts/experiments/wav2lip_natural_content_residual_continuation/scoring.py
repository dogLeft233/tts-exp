from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, bytes_sha256, file_sha256, source_pcm16, verify_self_hashed_json, write_self_hashed_json


def score_metrics(matrix: np.ndarray, *, rows: list[int] | None = None) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(value).all():
        raise ProtocolError(f"invalid SyncNet matrix shape: {value.shape}")
    selected = value if rows is None else value[np.asarray(rows, dtype=np.int64)]
    curve = np.mean(selected, axis=0, dtype=np.float64)
    minimum_index = int(np.argmin(curve))
    minimum = float(curve[minimum_index])
    return {"curve": [float(item) for item in curve], "min_index": minimum_index, "offset": config.VSHIFT - minimum_index, "D": minimum, "C": float(np.median(curve) - minimum)}


class ScoreEngine:
    def __init__(self) -> None:
        from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

        if not config.SYNCNET_MODEL.is_file() or file_sha256(config.SYNCNET_MODEL) != config.SYNCNET_MODEL_SHA256:
            raise ProtocolError("SyncNet checkpoint hash changed")
        self._scorer = SyncNetScorer(config.SYNCNET_MODEL, "cpu", config.SYNCNET_BATCH_SIZE, config.SYNCNET_THREADS)

    def score(self, media: Path, source_audio: Path, output_dir: Path, *, sample_id: str, video_arm: str, audio_arm: str, reference_matrix: Path | None = None, expected_rows: int | None = config.EMBEDDING_ROWS) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = output_dir / "distance.npy"
        worker_result_path = output_dir / "worker.json"
        sidecar_path = output_dir / "score.json"
        if sidecar_path.is_file() and matrix_path.is_file() and worker_result_path.is_file():
            prior = verify_self_hashed_json(sidecar_path)
            if prior.get("media_sha256") != file_sha256(media) or prior.get("source_audio_sha256") != file_sha256(source_audio):
                raise ProtocolError(f"existing score binding changed: {sample_id}/{video_arm}/{audio_arm}")
            return prior
        if any(path.exists() for path in (matrix_path, worker_result_path, sidecar_path)):
            raise ProtocolError(f"partial score cell cannot be resumed: {sample_id}/{video_arm}/{audio_arm}")
        media_sha = file_sha256(media)
        pcm_sha = bytes_sha256(source_pcm16(source_audio))
        result = self._scorer.score(media, source_audio, output_dir, media_sha, pcm_sha)
        matrix = np.asarray(np.load(result["matrix"], allow_pickle=False), dtype=np.float64)
        if expected_rows is not None and matrix.shape != (int(expected_rows), config.MATRIX_COLUMNS):
            raise ProtocolError(f"target score matrix shape changed: {sample_id}/{video_arm}/{audio_arm}: {matrix.shape}")
        worker = {"schema_version": 1, "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, **result}
        write_self_hashed_json(worker_result_path, worker)
        sidecar = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, "media": str(media.resolve()), "media_sha256": media_sha, "source_audio": str(source_audio.resolve()), "source_audio_sha256": file_sha256(source_audio), "source_pcm_sha256": pcm_sha, "matrix": str(Path(result["matrix"]).resolve()), "matrix_sha256": file_sha256(Path(result["matrix"])), "matrix_shape": list(matrix.shape), "full": score_metrics(matrix), "U": {"rows": list(config.U_ROWS), **score_metrics(matrix, rows=list(config.U_ROWS))}, "worker": str(worker_result_path.resolve()), "worker_sha256": file_sha256(worker_result_path), "fresh_forward": True, "reference_matrix": str(reference_matrix.resolve()) if reference_matrix else None}
        return write_self_hashed_json(sidecar_path, sidecar)


def load_matrix(row: dict[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", "")))
    if not path.is_file() or str(row.get("matrix_sha256")) != file_sha256(path):
        raise ProtocolError(f"score matrix binding changed: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (config.EMBEDDING_ROWS, config.MATRIX_COLUMNS) or not np.isfinite(matrix).all():
        raise ProtocolError(f"score matrix is malformed: {path}")
    return matrix
