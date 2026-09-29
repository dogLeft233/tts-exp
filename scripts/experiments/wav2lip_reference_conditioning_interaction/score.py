from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, load_self_hashed, pcm16, write_json


def matrix_from_embeddings(
    visual: np.ndarray,
    audio: np.ndarray,
    lag_start: int = config.NATURAL_LAG_START,
    rows: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """Rebuild a real-support distance matrix for one physical lag domain.

    ``lag_start`` is the signed q displacement: -15 gives q=r+j-15 and
    -10 gives q=r+j-10. The physical offset at column j is
    ``-lag_start-j``.
    """
    visual = np.asarray(visual, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if (
        visual.ndim != 2
        or audio.ndim != 2
        or visual.shape != audio.shape
        or visual.shape[1] != 1024
    ):
        raise ProtocolError(f"embedding shape mismatch: {visual.shape}/{audio.shape}")
    selected = (
        tuple(range(visual.shape[0]))
        if rows is None
        else tuple(int(row) for row in rows)
    )
    if not selected or min(selected) < 0 or max(selected) >= visual.shape[0]:
        raise ProtocolError("embedding rows outside support")
    matrix = np.empty((len(selected), config.MATRIX_COLUMNS), dtype=np.float64)
    for output_row, row in enumerate(selected):
        indices = (
            row + np.arange(config.MATRIX_COLUMNS, dtype=np.int64) + int(lag_start)
        )
        if int(indices.min()) < 0 or int(indices.max()) >= audio.shape[0]:
            raise ProtocolError(f"lag domain has no real support for row {row}")
        difference = (
            visual[row : row + 1].astype(np.float64)
            - audio[indices].astype(np.float64)
            + 1e-6
        )
        matrix[output_row] = np.sqrt(
            np.sum(difference * difference, axis=1, dtype=np.float64)
        )
    return matrix


def metrics(
    matrix: np.ndarray,
    rows: tuple[int, ...] | list[int] | None = None,
    *,
    lag_start: int = config.NATURAL_LAG_START,
) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != config.MATRIX_COLUMNS
        or not np.isfinite(value).all()
    ):
        raise ProtocolError(f"bad matrix: {value.shape}")
    chosen = value if rows is None else value[np.asarray(rows, dtype=np.int64)]
    if chosen.shape[0] == 0:
        raise ProtocolError("empty metric rows")
    curve = chosen.mean(axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {
        "curve": [float(x) for x in curve],
        "min_index": index,
        "offset": -int(lag_start) - index,
        "lag_start": int(lag_start),
        "rows": list(range(value.shape[0]))
        if rows is None
        else [int(row) for row in rows],
        "D": float(curve[index]),
        "C": float(np.median(curve) - curve[index]),
    }


def domain_metrics(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    lag_start: int,
    rows: tuple[int, ...] = config.U_ROWS,
) -> dict[str, Any]:
    return metrics(
        matrix_from_embeddings(visual, audio, lag_start, rows), lag_start=lag_start
    )


def worker_arrays(
    score: dict[str, Any], expected_rows: int | None = config.EMBEDDING_ROWS
) -> tuple[np.ndarray, np.ndarray]:
    worker_path = Path(str(score["worker"]))
    worker = load_self_hashed(worker_path)
    visual_path, audio_path = (
        Path(str(worker["visual"])),
        Path(str(worker["audio_embedding"])),
    )
    if file_sha256(visual_path) != worker.get("visual_sha256") or file_sha256(
        audio_path
    ) != worker.get("audio_embedding_sha256"):
        raise ProtocolError("SyncNet embedding hash mismatch")
    visual, audio = (
        np.load(visual_path, allow_pickle=False),
        np.load(audio_path, allow_pickle=False),
    )
    shape_ok = (
        visual.ndim == 2
        and visual.shape[1] == 1024
        and visual.shape[0] >= config.U_ROWS[-1] + 1
    )
    if expected_rows is not None:
        shape_ok = shape_ok and visual.shape == (expected_rows, 1024)
    if not shape_ok or audio.shape != visual.shape:
        raise ProtocolError("unexpected SyncNet embedding shape")
    if not np.isfinite(visual).all() or not np.isfinite(audio).all():
        raise ProtocolError("non-finite SyncNet embedding")
    return visual, audio


class ScoreEngine:
    def __init__(self) -> None:
        from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

        if file_sha256(config.SYNCNET_MODEL) != config.FIXED_HASHES["syncnet"]:
            raise ProtocolError("SyncNet hash changed")
        self.engine = SyncNetScorer(
            config.SYNCNET_MODEL,
            "cpu",
            config.SYNCNET_BATCH_SIZE,
            config.SYNCNET_THREADS,
        )

    def score(
        self,
        media: Path,
        audio: Path,
        output: Path,
        *,
        sample_id: str,
        video_arm: str,
        audio_arm: str,
        expected_rows: int | None = config.EMBEDDING_ROWS,
    ) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        sidecar = output / "score.json"
        media_hash = file_sha256(media)
        if sidecar.is_file():
            prior = load_self_hashed(sidecar)
            if prior.get("media_sha256") != media_hash:
                raise ProtocolError(f"score input changed: {sample_id}/{video_arm}")
            return prior
        result = self.engine.score(
            media,
            audio,
            output / "forward",
            media_hash,
            hashlib.sha256(pcm16(audio)).hexdigest(),
        )
        matrix = np.asarray(
            np.load(result["matrix"], allow_pickle=False), dtype=np.float64
        )
        if expected_rows is not None and matrix.shape != (
            expected_rows,
            config.MATRIX_COLUMNS,
        ):
            raise ProtocolError(f"unexpected matrix: {matrix.shape}")
        write_json(
            output / "worker.json",
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "video_arm": video_arm,
                "audio_arm": audio_arm,
                **result,
            },
        )
        row = {
            "schema_version": 1,
            "protocol_id": "wav2lip_reference_conditioning_interaction",
            "sample_id": sample_id,
            "video_arm": video_arm,
            "audio_arm": audio_arm,
            "media": str(media.resolve()),
            "media_sha256": media_hash,
            "source_audio": str(audio.resolve()),
            "source_audio_sha256": file_sha256(audio),
            "source_pcm_sha256": hashlib.sha256(pcm16(audio)).hexdigest(),
            "matrix": str(Path(result["matrix"]).resolve()),
            "matrix_sha256": file_sha256(Path(result["matrix"])),
            "matrix_shape": list(matrix.shape),
            "full": metrics(matrix),
            "U": {"rows": list(config.U_ROWS), **metrics(matrix, list(config.U_ROWS))},
            "worker": str((output / "worker.json").resolve()),
            "worker_sha256": file_sha256(output / "worker.json"),
            "fresh_forward": True,
        }
        visual, audio_embedding = worker_arrays(row, expected_rows=expected_rows)
        row["domains"] = {
            "natural": domain_metrics(
                visual, audio_embedding, lag_start=config.NATURAL_LAG_START
            ),
            "matched_delay": domain_metrics(
                visual, audio_embedding, lag_start=config.DELAY_LAG_START
            ),
        }
        return write_json(sidecar, row)
