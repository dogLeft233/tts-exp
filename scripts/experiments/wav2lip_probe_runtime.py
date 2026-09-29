"""Small, science-agnostic runtime helpers for the three frozen probes."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
U_ROWS = tuple(range(30, 58))
VSHIFT = 15
MEL_SHAPE = (80, 308)
FRAME_COUNT = 93
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_DRAWS = 20_000


class ProtocolError(RuntimeError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON object expected: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return body


def load_self(path: Path) -> dict[str, Any]:
    value = read_json(path)
    actual = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_hash(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def source_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != 16_000
            or handle.getcomptype() != "NONE"
        ):
            raise ProtocolError(f"source is not mono PCM16/16k: {path}")
        return handle.readframes(handle.getnframes())


def pcm_array(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raise ProtocolError("PCM16 byte count is odd")
    return np.frombuffer(raw, dtype="<i2").copy()


def write_pcm16(path: Path, pcm: np.ndarray) -> None:
    values = np.asarray(pcm)
    if values.ndim != 1 or values.dtype != np.int16:
        raise ProtocolError("PCM output must be one-dimensional int16")
    if np.any(values.astype(np.int64) < -32768) or np.any(
        values.astype(np.int64) > 32767
    ):
        raise ProtocolError("PCM output is outside int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(values.astype("<i2", copy=False).tobytes())


def load_mel(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.shape != MEL_SHAPE or not np.isfinite(value).all():
        raise ProtocolError(f"invalid mel: {path} {value.shape}")
    return value


def syncnet_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    if (
        visual32.ndim != 2
        or audio32.ndim != 2
        or visual32.shape != audio32.shape
        or visual32.shape[0] < 88
        or visual32.shape[1] != 1024
    ):
        raise ProtocolError(
            f"invalid embedding shape: {visual32.shape}/{audio32.shape}"
        )
    padded = np.pad(audio32, ((VSHIFT, VSHIFT), (0, 0)), mode="constant")
    result = np.empty((88, 31), dtype=np.float32)
    for row in range(88):
        difference = visual32[row : row + 1] - padded[row : row + 31]
        result[row] = np.sqrt(
            np.sum(np.square(difference + np.float32(1e-6), dtype=np.float32), axis=1),
            dtype=np.float32,
        )
    return result


def load_worker_arrays(
    score_row: dict[str, Any], expected_rows: int | None = 88
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = score_row.get("score", score_row)
    worker_path = Path(str(score["worker"]))
    worker = load_self(worker_path)
    values: list[np.ndarray] = []
    for key in ("visual", "audio_embedding", "matrix"):
        path = Path(str(worker[key]))
        if file_sha256(path) != str(worker[f"{key}_sha256"]):
            raise ProtocolError(f"worker array hash changed: {path}")
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if key != "matrix":
            expected = (expected_rows, 1024) if expected_rows is not None else None
        else:
            expected = (expected_rows, 31) if expected_rows is not None else None
        if expected is None:
            valid_shape = (
                value.ndim == 2
                and value.shape[1] == (1024 if key != "matrix" else 31)
                and value.shape[0] >= 88
            )
        else:
            valid_shape = value.shape == expected
        if not valid_shape or not np.isfinite(value).all():
            raise ProtocolError(f"invalid worker array: {path}")
        values.append(value)
    return values[0], values[1], values[2]


def score_metrics(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != 31
        or value.shape[0] < 58
        or not np.isfinite(value).all()
    ):
        raise ProtocolError(f"invalid SyncNet matrix: {value.shape}")
    curve = np.mean(value[np.asarray(U_ROWS)], axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {
        "curve": [float(item) for item in curve],
        "min_index": index,
        "offset": 15 - index,
        "D": float(curve[index]),
        "C": float(np.median(curve) - curve[index]),
    }


def gain(candidate: dict[str, Any], natural: dict[str, Any]) -> dict[str, float]:
    k0 = int(natural["min_index"])
    return {
        "C": float(candidate["C"] - natural["C"]),
        "D": float(natural["D"] - candidate["D"]),
        "A": float(natural["curve"][k0] - candidate["curve"][k0]),
    }


def bootstrap_indices(labels: list[str]) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels))
    )


def grouped_stats(
    values: Iterable[float], groups: Iterable[str], indices: np.ndarray | None = None
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("grouped statistic requires eight groups x two records")
    means = np.asarray(
        [np.mean(grouped[label], dtype=np.float64) for label in labels],
        dtype=np.float64,
    )
    sampled = bootstrap_indices(labels) if indices is None else np.asarray(indices)
    estimates = means[sampled].mean(axis=1, dtype=np.float64)
    return {
        "mean": float(means.mean()),
        "ci99": [
            float(np.quantile(estimates, 0.005, method="linear")),
            float(np.quantile(estimates, 0.995, method="linear")),
        ],
        "group_labels": labels,
        "group_means": {
            label: float(value) for label, value in zip(labels, means, strict=True)
        },
        "group_positive_count": int(np.sum(means > 0.0)),
        "draws": int(sampled.shape[0]),
        "seed": BOOTSTRAP_SEED,
        "indices_sha256": hashlib.sha256(
            np.asarray(sampled, dtype=np.int64).tobytes()
        ).hexdigest(),
        "indices": np.asarray(sampled, dtype=np.int64).tolist(),
    }


def contrast_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    gains = [gain(row[arm], row["N"]) for row in records]
    groups = [str(row["source_group"]) for row in records]
    labels = sorted(set(groups))
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for group, value in zip(groups, gains, strict=True):
        grouped[group].append(value)
    group_means = {
        label: {
            metric: float(
                np.mean([item[metric] for item in grouped[label]], dtype=np.float64)
            )
            for metric in ("C", "D", "A")
        }
        for label in labels
    }
    indices = bootstrap_indices(labels)
    metrics = {
        metric: grouped_stats([item[metric] for item in gains], groups, indices)
        for metric in ("C", "D", "A")
    }
    joint = int(
        sum(
            all(group_means[label][metric] > 0.0 for metric in ("C", "D", "A"))
            for label in labels
        )
    )
    return {
        "records": [
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "gain": value,
            }
            for row, value in zip(records, gains, strict=True)
        ],
        "metrics": metrics,
        "joint_positive_count": joint,
        "mean_C": metrics["C"]["mean"],
    }


def gain_pass(summary: dict[str, Any]) -> bool:
    return bool(
        summary["mean_C"] > 0.05
        and summary["joint_positive_count"] >= 7
        and all(
            summary["metrics"][metric]["ci99"][0] > 0.0 for metric in ("C", "D", "A")
        )
    )


def natural_mel_from_pcm(raw: bytes) -> np.ndarray:
    import sys

    wav2lip_root = REPO / "third_party/Wav2Lip"
    if str(wav2lip_root) not in sys.path:
        sys.path.insert(0, str(wav2lip_root))
    import audio  # type: ignore

    samples = pcm_array(raw)[:61_440].astype(np.float32) / np.float32(32768.0)
    mel = np.asarray(audio.melspectrogram(samples), dtype=np.float32)
    if mel.shape != MEL_SHAPE or not np.isfinite(mel).all():
        raise ProtocolError(f"official mel has unexpected shape: {mel.shape}")
    return mel


def make_chunks(mel: np.ndarray) -> list[np.ndarray]:
    value = load_mel_from_array(mel)
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > value.shape[1]:
            chunks.append(value[:, -16:])
            break
        chunks.append(value[:, start : start + 16])
        index += 1
    if len(chunks) != FRAME_COUNT:
        raise ProtocolError(f"official chunk count changed: {len(chunks)}")
    return chunks


def load_mel_from_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != MEL_SHAPE or not np.isfinite(array).all():
        raise ProtocolError(f"invalid mel array: {array.shape}")
    return array


def matched_curve(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    result = []
    for row in U_ROWS:
        indices = row + np.arange(lag_start, lag_start + 31, dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= audio32.shape[0]:
            raise ProtocolError("matched lag domain is outside the audio embedding")
        difference = visual32[row : row + 1] - audio32[indices]
        result.append(
            np.mean(
                np.sqrt(
                    np.sum(
                        np.square(difference + np.float32(1e-6), dtype=np.float32),
                        axis=1,
                    ),
                    dtype=np.float64,
                )
            )
        )
    return np.asarray(result, dtype=np.float64)


def parent_control_gate(parent_scores: Path, parent_drivers: Path) -> dict[str, Any]:
    """Recompute the frozen pre-candidate F46 control without importing a probe."""
    manifest = load_self(parent_scores)
    driver_rows = load_self(parent_drivers).get("rows", [])
    groups = {str(row["sample_id"]): str(row["source_group"]) for row in driver_rows}
    rows = manifest.get("rows", [])
    if len(rows) != 36:
        raise ProtocolError(f"expected 36 parent control rows, got {len(rows)}")
    lookup = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row
        for row in rows
    }
    records = []
    for sample_id, group in sorted(groups.items(), key=lambda item: (item[1], item[0])):
        natural = lookup[(sample_id, "N", "N")]
        delayed = lookup[(sample_id, "N", "A_DELAY")]
        visual_n, audio_n, cached_n = load_worker_arrays(natural)
        visual_d, audio_d, cached_d = load_worker_arrays(delayed)
        if not np.array_equal(visual_n, visual_d):
            raise ProtocolError(f"delayed visual embedding differs: {sample_id}")
        if (
            np.max(np.abs(syncnet_matrix(visual_n, audio_n) - cached_n)) > 1e-4
            or np.max(np.abs(syncnet_matrix(visual_d, audio_d) - cached_d)) > 1e-4
        ):
            raise ProtocolError(f"parent matrix cannot be rebuilt: {sample_id}")
        n_curve = matched_curve(visual_n, audio_n, -15)
        d_curve = matched_curve(visual_n, audio_d, -10)
        n_index = int(np.argmin(n_curve))
        d_index = int(np.argmin(d_curve))
        old_n = score_metrics(cached_n)
        old_d = score_metrics(cached_d)
        anchor = float(
            np.mean(cached_d[np.asarray(U_ROWS), old_n["min_index"]])
            - np.mean(cached_n[np.asarray(U_ROWS), old_n["min_index"]])
        )
        records.append(
            {
                "sample_id": sample_id,
                "source_group": group,
                "old_difference": int(old_d["offset"] - old_n["offset"]),
                "matched_difference": int((10 - d_index) - (15 - n_index)),
                "anchor_damage": anchor,
            }
        )
    old_count = sum(-6 <= row["old_difference"] <= -4 for row in records)
    matched_count = sum(-6 <= row["matched_difference"] <= -4 for row in records)
    damage_values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        damage_values[row["source_group"]].append(float(row["anchor_damage"]))
    damage_labels = sorted(damage_values)
    damage_means = np.asarray(
        [np.mean(damage_values[label], dtype=np.float64) for label in damage_labels],
        dtype=np.float64,
    )
    damage_indices = np.random.Generator(np.random.PCG64(20260909)).integers(
        0, 8, size=(10_000, 8)
    )
    damage_estimates = damage_means[damage_indices].mean(axis=1)
    damage = {
        "mean": float(damage_means.mean()),
        "ci95": [
            float(np.quantile(damage_estimates, 0.025, method="linear")),
            float(np.quantile(damage_estimates, 0.975, method="linear")),
        ],
        "group_labels": damage_labels,
        "group_means": {
            label: float(value)
            for label, value in zip(damage_labels, damage_means, strict=True)
        },
        "group_positive_count": int(np.sum(damage_means > 0)),
        "draws": 10_000,
        "seed": 20260909,
        "indices_sha256": hashlib.sha256(damage_indices.tobytes()).hexdigest(),
        "indices": damage_indices.tolist(),
    }
    parity = [
        row for row in rows if str(row.get("video_arm", "")).startswith("PARITY_")
    ]
    parity_ok = len(parity) == 2 and all(
        bool(row.get("parity", {}).get("passes")) for row in parity
    )
    return {
        "status": "PASS"
        if matched_count >= 14
        and damage["ci95"][0] > 0.0
        and damage["group_positive_count"] >= 7
        and parity_ok
        else "CONTROL_FAILED",
        "old_offset_pass_count": old_count,
        "matched_offset_pass_count": matched_count,
        "anchor_damage": damage,
        "parity_ok": parity_ok,
        "records": records,
    }


@contextmanager
def gpu_lock(path: Path = Path("/tmp/tts-exp-wav2lip-gpu0.lock")) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mux_video_audio(video: Path, source_audio: Path, ffmpeg: Path) -> str:
    """Bind the untouched natural PCM to a silent FFV1 video for SyncNet input."""
    import subprocess

    temporary = video.with_name(f".{video.name}.{os.getpid()}.mux.partial")
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-i",
        str(source_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "matroska",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise ProtocolError(
            f"natural-audio mux failed: {video}: {result.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    temporary.replace(video)
    return file_sha256(video)


def _check_optional_artifact_hash(payload: dict[str, Any], path: Path) -> None:
    stored = payload.get("artifact_sha256")
    if stored is not None:
        body = dict(payload)
        body.pop("artifact_sha256", None)
        if stored != canonical_hash(body):
            raise ProtocolError(f"artifact self-hash mismatch: {path}")


def _generation_plan_map(plan: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in plan.get("rows", []):
        sample_id = str(row["sample_id"])
        for arm, mel in row["arms"].items():
            key = (sample_id, str(arm))
            if key in result:
                raise ProtocolError(f"duplicate generation plan cell: {key}")
            result[key] = {
                "output": str(row["outputs"][arm]),
                "source_audio": str(row["source_audio"]),
                "static_face": str(row["static_face"]),
                "mel": str(mel),
            }
    return result


def _validate_generation_result(
    plan: dict[str, Any],
    payload: dict[str, Any],
    result_path: Path,
    checkpoint: Path,
    protocol_id: str,
) -> None:
    _check_optional_artifact_hash(payload, result_path)
    if payload.get("protocol_id") != protocol_id or payload.get("status") != "complete":
        raise ProtocolError(
            f"generation result protocol/status mismatch: {result_path}"
        )
    runtime = payload.get("runtime", {})
    if runtime.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ProtocolError("generation checkpoint binding mismatch")
    expected = _generation_plan_map(plan)
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        key = (str(row["sample_id"]), str(row["arm"]))
        if key in actual:
            raise ProtocolError(f"duplicate generation result cell: {key}")
        actual[key] = row
    if set(actual) != set(expected):
        raise ProtocolError("generation result does not match the current plan")
    for key, expected_row in expected.items():
        row = actual[key]
        for field in ("output", "source_audio", "static_face", "mel"):
            if str(row.get(field)) != str(Path(expected_row[field]).resolve()):
                raise ProtocolError(f"generation binding mismatch: {key}/{field}")
        output = Path(str(row["output"]))
        if not output.is_file() or file_sha256(output) != str(row.get("output_sha256")):
            raise ProtocolError(f"generation output hash mismatch: {output}")


def run_gpu_plan(
    plan: Path,
    result: Path,
    protocol_id: str,
    checkpoint: Path,
    ffmpeg: Path,
    python: Path,
) -> dict[str, Any]:
    plan_payload = load_self(plan)
    if result.is_file():
        payload = read_json(result)
        _validate_generation_result(
            plan_payload, payload, result, checkpoint, protocol_id
        )
        changed = False
        if "artifact_sha256" not in payload:
            changed = True
        if changed:
            payload = write_json(result, payload)
        return payload
    import subprocess

    with gpu_lock():
        command = [
            str(python),
            "-m",
            "scripts.experiments.wav2lip_probe_gpu",
            "--plan",
            str(plan),
            "--result",
            str(result),
            "--protocol-id",
            protocol_id,
            "--checkpoint",
            str(checkpoint),
            "--ffmpeg",
            str(ffmpeg),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            str(REPO) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        log = result.with_suffix(".gpu.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=str(REPO),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result.is_file():
            raise ProtocolError(f"GPU generation failed; see {log}")
    payload = read_json(result)
    _validate_generation_result(plan_payload, payload, result, checkpoint, protocol_id)
    return write_json(result, payload)


def score_videos(
    video_rows: list[dict[str, Any]], output_dir: Path, model: Path, protocol_id: str
) -> list[dict[str, Any]]:
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = output_dir / "score_plan.json"
    result_path = output_dir / "score_result.json"
    legacy_result = output_dir.parent / "score_result.json"
    if (
        not result_path.is_file()
        and output_dir.name == "candidates"
        and legacy_result.is_file()
    ):
        result_path = legacy_result
        legacy_plan = output_dir.parent / "score_plan.json"
        if legacy_plan.is_file():
            plan = legacy_plan
    if result_path.is_file():
        payload = read_json(result_path)
        plan_payload = load_self(plan)
        _check_optional_artifact_hash(payload, result_path)
        if (
            payload.get("protocol_id") != protocol_id
            or payload.get("status") != "complete"
        ):
            raise ProtocolError(f"score result protocol/status mismatch: {result_path}")
        expected = {
            (str(row["sample_id"]), str(row["arm"])): row
            for row in plan_payload.get("rows", [])
        }
        actual = {
            (str(row["sample_id"]), str(row["video_arm"])): row
            for row in payload.get("rows", [])
        }
        if set(actual) != set(expected):
            raise ProtocolError("score result does not match the current score plan")
        changed = False
        for row in payload.get("rows", []):
            score = row["score"]
            expected_item = expected[(str(row["sample_id"]), str(row["video_arm"]))]
            if str(score.get("media")) != str(
                Path(str(expected_item["output"])).resolve()
            ) or str(score.get("source_audio")) != str(
                Path(str(expected_item["source_audio"])).resolve()
            ):
                raise ProtocolError(
                    f"score media binding mismatch: {row['sample_id']}/{row['video_arm']}"
                )
            media = Path(str(score["media"]))
            if (
                not media.is_file()
                or file_sha256(media) != str(score.get("media_sha256"))
                or str(score.get("media_sha256"))
                != str(expected_item.get("output_sha256"))
            ):
                raise ProtocolError(f"score media hash mismatch: {media}")
            if str(score.get("model_sha256")) != file_sha256(model):
                raise ProtocolError(f"score model binding mismatch: {model}")
            if "worker" not in score:
                worker_path = (
                    output_dir
                    / f"{row['sample_id']}__{row['video_arm']}__N"
                    / "worker.json"
                )
                worker = write_json(
                    worker_path,
                    {
                        "schema_version": 1,
                        "stage_id": "fresh_syncnet_forward_worker",
                        "visual": score["visual"],
                        "visual_sha256": score["visual_sha256"],
                        "audio_embedding": score["audio_embedding"],
                        "audio_embedding_sha256": score["audio_embedding_sha256"],
                        "matrix": score["matrix"],
                        "matrix_sha256": score["matrix_sha256"],
                        "visual_shape": score["visual_shape"],
                        "audio_embedding_shape": score["audio_embedding_shape"],
                        "matrix_shape": score["matrix_shape"],
                    },
                )
                score["worker"] = str(worker_path.resolve())
                score["worker_sha256"] = worker["artifact_sha256"]
                changed = True
        if changed or "artifact_sha256" not in payload:
            payload = write_json(result_path, payload)
        return payload["rows"]
    write_json(
        plan, {"schema_version": 1, "protocol_id": protocol_id, "rows": video_rows}
    )
    command = [
        "/home/wjj/.venvs/syncnet/bin/python",
        "-m",
        "scripts.experiments.wav2lip_probe_score",
        "--plan",
        str(plan),
        "--result",
        str(result_path),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--protocol-id",
        protocol_id,
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(REPO) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    log = output_dir.parent / "score.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=str(REPO),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        raise ProtocolError(f"SyncNet scoring failed; see {log}")
    payload = read_json(result_path)
    changed = False
    for row in payload.get("rows", []):
        score = row["score"]
        if "worker" not in score:
            worker_path = (
                output_dir
                / f"{row['sample_id']}__{row['video_arm']}__N"
                / "worker.json"
            )
            worker = write_json(
                worker_path,
                {
                    "schema_version": 1,
                    "stage_id": "fresh_syncnet_forward_worker",
                    "visual": score["visual"],
                    "visual_sha256": score["visual_sha256"],
                    "audio_embedding": score["audio_embedding"],
                    "audio_embedding_sha256": score["audio_embedding_sha256"],
                    "matrix": score["matrix"],
                    "matrix_sha256": score["matrix_sha256"],
                    "visual_shape": score["visual_shape"],
                    "audio_embedding_shape": score["audio_embedding_shape"],
                    "matrix_shape": score["matrix_shape"],
                },
            )
            score["worker"] = str(worker_path.resolve())
            score["worker_sha256"] = worker["artifact_sha256"]
            changed = True
    if changed:
        # Worker sidecars are part of the score artifact binding.  Recompute
        # the envelope hash after adding them so a resume can verify the
        # result instead of rejecting its own freshly-produced artifact.
        payload = write_json(result_path, payload)
    return payload["rows"]


def fresh_control_rows(
    parent_control: Path, output_dir: Path, model: Path, protocol_id: str
) -> list[dict[str, Any]]:
    parent = load_self(parent_control)
    selected = []
    references: dict[tuple[str, str], Path] = {}
    for row in parent.get("rows", []):
        arm = str(row.get("video_arm"))
        if arm == "N_REPEAT" or arm.startswith("PARITY_"):
            score = row.get("score", row)
            selected.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "arm": f"FRESH_{arm}",
                    "output": str(Path(str(score["media"])).resolve()),
                    "output_sha256": file_sha256(Path(str(score["media"]))),
                    "source_audio": str(Path(str(score["source_audio"])).resolve()),
                }
            )
            references[(str(row["sample_id"]), f"FRESH_{arm}")] = Path(
                str(row.get("parity", {}).get("actual_matrix", score["matrix"]))
            ).resolve()
    if len(selected) != 4:
        raise ProtocolError(
            f"expected 2 replay and 2 parity control cells, got {len(selected)}"
        )
    rows = score_videos(selected, output_dir, model, protocol_id)
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        reference = references[key]
        actual = np.asarray(
            np.load(Path(str(row["score"]["matrix"])), allow_pickle=False),
            dtype=np.float64,
        )
        expected = np.asarray(np.load(reference, allow_pickle=False), dtype=np.float64)
        if (
            actual.shape != expected.shape
            or float(np.max(np.abs(actual - expected))) > 1e-4
        ):
            raise ProtocolError(f"fresh control matrix mismatch: {key}")
        actual_metrics = score_metrics(actual)
        reference_metrics = score_metrics(expected)
        if (
            actual_metrics["offset"] != reference_metrics["offset"]
            or max(
                abs(a - b)
                for a, b in zip(
                    actual_metrics["curve"], reference_metrics["curve"], strict=True
                )
            )
            > 1e-6
        ):
            raise ProtocolError(f"fresh control endpoint mismatch: {key}")
        row["fresh_control_validation"] = {
            "reference_matrix": str(reference),
            "reference_matrix_sha256": file_sha256(reference),
            "matrix_max_abs": float(np.max(np.abs(actual - expected))),
            "endpoint_max_abs": max(
                abs(a - b)
                for a, b in zip(
                    actual_metrics["curve"], reference_metrics["curve"], strict=True
                )
            ),
            "offset_equal": True,
        }
    return rows
