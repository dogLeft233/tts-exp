"""Static-image B075/MFA-linear native-versus-natural-replacement experiment.

This deliberately reuses the 3-image/11-audio subset from the B025/B050
low-alpha experiment.  It never reads or generates LOCAL_SWAP audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from .static_image_bridge import config as static_config
from .static_image_bridge.analysis import bootstrap_stats, common_w, score_metrics
from .static_image_bridge.common import (
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    run_logged,
    verify_self_hashed_json,
    write_self_hashed_json,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE_STATIC_ROOT = REPO / "runs/static_image_bridge_lowalpha_20260913_v2"
SOURCE_PROTOCOL = SOURCE_STATIC_ROOT / "protocol.json"
SOURCE_SCOPE = SOURCE_STATIC_ROOT / "analysis_scope.json"
SOURCE_COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
SOURCE_AUDIO_MANIFEST = REPO / "runs/static_image_bridge_20260913/audio_manifest.json"
RUN_ROOT = REPO / "runs/static_image_b075_mfa_20260913"

IMAGE_IDS = ("3", "6", "9")
VIDEO_ARMS = ("B075", "MFA_LINEAR")
SCORE_AUDIO_ARMS = ("N", "B075", "MFA_LINEAR")
REUSED_VIDEO_ARMS = ("N", "B025", "B050")
ANALYSIS_CONDITIONS = ("B025", "B050", "B075", "MFA_LINEAR")
SEED = 42
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260913


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pcm_sha256(path: Path) -> tuple[str, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16_000:
            raise ProtocolError(f"audio is not mono 16 kHz PCM16: {path}")
        frames = handle.readframes(handle.getnframes())
        return _sha256_bytes(frames), len(frames) // 2


def _run_path(name: str) -> Path:
    return RUN_ROOT / name


def _source_records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _read_json(SOURCE_PROTOCOL)
    scope = _read_json(SOURCE_SCOPE)
    cohort = _read_json(SOURCE_COHORT)
    sample_ids = [str(value) for value in scope["sample_ids"]]
    if len(sample_ids) != 11:
        raise ProtocolError(f"B025/B050 subset must contain 11 samples, got {len(sample_ids)}")
    protocol_records = {str(row["sample_id"]): row for row in protocol["records"]}
    cohort_records = {str(row["sample_id"]): row for row in cohort["records"]}
    if set(sample_ids) - set(protocol_records) or set(sample_ids) - set(cohort_records):
        raise ProtocolError("static subset and MFA cohort do not have identical sample IDs")
    images = {str(row["id"]): row for row in protocol["images"]}
    if set(IMAGE_IDS) - set(images):
        raise ProtocolError("fixed image IDs 3/6/9 are missing from the frozen B025/B050 protocol")
    return protocol, scope, {"sample_ids": sample_ids, "records": protocol_records, "cohort": cohort_records, "images": images}


def _audio_meta(path: Path) -> dict[str, Any]:
    pcm_sha, count = _pcm_sha256(path)
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "pcm_sha256": pcm_sha,
        "sample_count": count,
        "sample_rate": 16_000,
        "channels": 1,
        "sample_width": 2,
    }


def _image_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row["path"])).resolve()
    if not path.is_file():
        raise ProtocolError(f"fixed image is missing: {path}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ProtocolError(f"fixed image cannot be decoded: {path}")
    rgb_sha = _sha256_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
    if rgb_sha != str(row["rgb_sha256"]):
        raise ProtocolError(f"fixed image RGB hash changed: {path}")
    return {
        "id": str(row["id"]),
        "path": str(path),
        "container_sha256": file_sha256(path),
        "rgb_pixel_sha256": rgb_sha,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "generation_box_xyxy": [int(value) for value in row["generation_box"]],
        "score_box": dict(row["score_box"]),
    }


def prepare() -> dict[str, Any]:
    _source_protocol, _scope, source = _source_records()
    source_audio = _read_json(SOURCE_AUDIO_MANIFEST)
    audio_rows = {str(row["sample_id"]): row for row in source_audio["rows"]}
    images = [_image_meta(source["images"][image_id]) for image_id in IMAGE_IDS]
    records: list[dict[str, Any]] = []
    for sample_id in source["sample_ids"]:
        base = source["records"][sample_id]
        cohort = source["cohort"][sample_id]
        audio_row = audio_rows.get(sample_id)
        if audio_row is None:
            raise ProtocolError(f"static B075 audio row missing: {sample_id}")
        n_path = Path(str(base["audio"]["N"]["path"])).resolve()
        b025_path = Path(str(base["audio"]["B025"]["path"])).resolve()
        b050_path = Path(str(base["audio"]["B050"]["path"])).resolve()
        b075_path = Path(str(audio_row["arms"]["B"]["path"])).resolve()
        mfa_path = Path(str(cohort["mfa_linear_audio"]["path"])).resolve()
        audio = {
            "N": _audio_meta(n_path),
            "B025": _audio_meta(b025_path),
            "B050": _audio_meta(b050_path),
            "B075": _audio_meta(b075_path),
            "MFA_LINEAR": _audio_meta(mfa_path),
        }
        if audio["N"]["pcm_sha256"] != str(base["audio"]["N"]["pcm_sha256"]):
            raise ProtocolError(f"natural audio binding changed: {sample_id}")
        if audio["B075"]["pcm_sha256"] != str(audio_row["arms"]["B"]["pcm_sha256"]):
            raise ProtocolError(f"B075 audio binding changed: {sample_id}")
        records.append({
            "sample_id": sample_id,
            "source_group": str(base["source_group"]),
            "natural_sample_count": int(audio["N"]["sample_count"]),
            "audio": audio,
            "baseline_video_paths": {
                image_id: str((SOURCE_STATIC_ROOT / "videos" / image_id / sample_id / "N.mkv").resolve())
                for image_id in IMAGE_IDS
            },
            "source_mfa_sha256": str(cohort["mfa_linear_audio"]["sha256"]),
        })
    inputs = {
        "schema_version": 1,
        "protocol_id": "static_image_b075_mfa_20260913",
        "source_protocol": str(SOURCE_PROTOCOL.resolve()),
        "source_protocol_sha256": file_sha256(SOURCE_PROTOCOL),
        "source_scope": str(SOURCE_SCOPE.resolve()),
        "source_scope_sha256": file_sha256(SOURCE_SCOPE),
        "source_cohort": str(SOURCE_COHORT.resolve()),
        "source_cohort_sha256": file_sha256(SOURCE_COHORT),
        "image_ids": list(IMAGE_IDS),
        "images": images,
        "sample_ids": source["sample_ids"],
        "records": records,
        "forbidden_condition": "LOCAL_SWAP is excluded; no LOCAL_SWAP audio is read or generated",
        "artifact_sha256": None,
    }
    inputs.pop("artifact_sha256", None)
    inputs["artifact_sha256"] = canonical_json_sha256(inputs)
    _write_json(_run_path("inputs.json"), inputs)
    protocol = {
        "schema_version": 1,
        "protocol_id": inputs["protocol_id"],
        "status": "prepared",
        "record_count": len(records),
        "image_count": len(images),
        "sampling_unit": "11 source groups x 3 fixed static portraits",
        "conditions": {
            "B075_native": "V_B075/A_B075",
            "B075_natural_replacement": "V_B075/A_N",
            "MFA_linear_native": "V_MFA_LINEAR/A_MFA_LINEAR",
            "MFA_linear_natural_replacement": "V_MFA_LINEAR/A_N",
        },
        "reused_baseline": "B025/B050 lowalpha run N.mkv and score matrices on the same 3 images and first 11 samples",
        "scoring": {
            "model": str(static_config.SYNCNET_MODEL.resolve()),
            "model_sha256": static_config.SYNCNET_MODEL_SHA256,
            "vshift": static_config.SYNCNET_VSHIFT,
            "common_support": "one W per image/sample across N, B025, B050, B075 and MFA native/replacement cells",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "source_group after averaging the 3 images",
        },
        "runtime": {
            "wav2lip_python": str(static_config.WAV2LIP_PYTHON.resolve()),
            "wav2lip_python_sha256": static_config.PYTHON_SHA256,
            "checkpoint": str(static_config.WAV2LIP_CHECKPOINT.resolve()),
            "checkpoint_sha256": static_config.WAV2LIP_CHECKPOINT_SHA256,
            "ffmpeg": str(static_config.FFMPEG.resolve()),
        },
        "forbidden_operations": ["LOCAL_SWAP", "score-based image/sample selection", "dynamic source frames"],
    }
    write_self_hashed_json(_run_path("protocol.json"), protocol)
    return inputs


def _load_inputs() -> dict[str, Any]:
    path = _run_path("inputs.json")
    if not path.is_file():
        return prepare()
    return verify_self_hashed_json(path)


def _image_by_id(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): dict(row) for row in inputs["images"]}


def _record_audio(inputs: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(row["sample_id"]): dict(row["audio"]) for row in inputs["records"]}


def _render_one(image: Mapping[str, Any], record: Mapping[str, Any], arm: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    output = _run_path(f"videos/{arm}/{image['id']}/{sample_id}.mkv")
    sidecar = output.with_suffix(".json")
    if output.is_file() and sidecar.is_file():
        row = verify_self_hashed_json(sidecar)
        if file_sha256(output) != row.get("output_sha256"):
            raise ProtocolError(f"resumed video hash changed: {output}")
        return row
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial video cell cannot be resumed: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_result = _run_path(f"videos/work/{arm}/{image['id']}/{sample_id}.worker.json")
    log = _run_path(f"logs/render_{arm}_{image['id']}_{sample_id}.log")
    audio_path = Path(str(record["audio"][arm]["path"]))
    command = [
        str(static_config.WAV2LIP_PYTHON),
        str(REPO / "scripts/experiments/static_image_bridge/render_worker.py"),
        "--image", str(image["path"]),
        "--image-rgb-sha256", str(image["rgb_pixel_sha256"]),
        "--audio", str(audio_path),
        "--box", *[str(value) for value in image["generation_box_xyxy"]],
        "--checkpoint", str(static_config.WAV2LIP_CHECKPOINT),
        "--ffmpeg", str(static_config.FFMPEG),
        "--outfile", str(output),
        "--result", str(worker_result),
        "--batch-size", str(static_config.WAV2LIP_BATCH_SIZE),
        "--seed", str(SEED),
    ]
    run_logged(command, REPO, log, env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": str(SEED)})
    worker = _read_json(worker_result)
    if worker.get("status") != "complete" or worker.get("device") != "cuda" or not output.is_file():
        raise ProtocolError(f"static Wav2Lip render failed: {image['id']}/{sample_id}/{arm}")
    row = {
        "schema_version": 1,
        "status": "complete",
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "image_id": str(image["id"]),
        "video_arm": arm,
        "input_mode": "one_fixed_png_only",
        "image_path": str(image["path"]),
        "image_rgb_sha256": str(image["rgb_pixel_sha256"]),
        "generation_box_xyxy": list(image["generation_box_xyxy"]),
        "source_frame_indices": worker.get("source_frame_indices"),
        "audio_path": str(audio_path.resolve()),
        "audio_pcm_sha256": str(record["audio"][arm]["pcm_sha256"]),
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "frame_count": worker.get("frames_rendered"),
        "fps": worker.get("fps"),
        "checkpoint_sha256": static_config.WAV2LIP_CHECKPOINT_SHA256,
        "worker_result": str(worker_result.resolve()),
        "worker_result_sha256": file_sha256(worker_result),
        "command": command,
        "log": str(log.resolve()),
    }
    if row["source_frame_indices"] != [0] * int(row["frame_count"]):
        raise ProtocolError(f"non-static source frame used: {image['id']}/{sample_id}/{arm}")
    write_self_hashed_json(sidecar, row)
    return verify_self_hashed_json(sidecar)


def render(inputs: Mapping[str, Any]) -> dict[str, Any]:
    images = _image_by_id(inputs)
    rows: list[dict[str, Any]] = []
    total = len(inputs["records"]) * len(IMAGE_IDS) * len(VIDEO_ARMS)
    done = 0
    for record in inputs["records"]:
        for image_id in IMAGE_IDS:
            for arm in VIDEO_ARMS:
                rows.append(_render_one(images[image_id], record, arm))
                done += 1
                print(f"VIDEO {done}/{total} {image_id} {record['sample_id']} {arm}", flush=True)
    payload = {"schema_version": 1, "status": "complete", "rows": rows, "count": len(rows)}
    write_self_hashed_json(_run_path("video_manifest.json"), payload)
    return verify_self_hashed_json(_run_path("video_manifest.json"))


def _score_one(video_row: Mapping[str, Any], record: Mapping[str, Any], image: Mapping[str, Any]) -> list[dict[str, Any]]:
    image_id = str(image["id"])
    sample_id = str(record["sample_id"])
    video_arm = str(video_row["video_arm"])
    output_dir = _run_path(f"scores/{video_arm}/{image_id}/{sample_id}")
    worker_files = [output_dir / f"{video_arm}__{arm}__worker.json" for arm in SCORE_AUDIO_ARMS if arm in (video_arm, "N")]
    if all(path.is_file() for path in worker_files):
        workers = [_read_json(path) for path in worker_files]
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_json = _run_path(f"score_requests/{video_arm}/{image_id}/{sample_id}.json")
        audio_map = {arm: str(record["audio"][arm]["path"]) for arm in ("N", video_arm)}
        _write_json(audio_json, audio_map)
        score_box = _run_path(f"score_support/{image_id}/score_box.json")
        if not score_box.is_file():
            _write_json(score_box, image["score_box"])
        log = _run_path(f"logs/score_{video_arm}_{image_id}_{sample_id}.log")
        command = [
            str(static_config.SYNCNET_PYTHON),
            str(REPO / "scripts/experiments/static_image_bridge/score_worker.py"),
            "--video", str(video_row["output"]),
            "--video-arm", video_arm,
            "--sample-id", sample_id,
            "--score-box", str(score_box),
            "--audio-json", str(audio_json),
            "--model", str(static_config.SYNCNET_MODEL),
            "--output-dir", str(output_dir),
            "--vshift", str(static_config.SYNCNET_VSHIFT),
            "--batch-size", str(static_config.SYNCNET_BATCH_SIZE),
        ]
        run_logged(command, REPO, log, env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": "0"})
        workers = []
        for arm in ("N", video_arm):
            path = output_dir / f"{video_arm}__{arm}__worker.json"
            if not path.is_file():
                raise ProtocolError(f"SyncNet worker output missing: {path}")
            workers.append(_read_json(path))
    rows = []
    for worker in workers:
        arm = str(worker["audio_arm"])
        matrix = Path(str(worker["matrix"])).resolve()
        if not matrix.is_file() or file_sha256(matrix) != str(worker["matrix_sha256"]):
            raise ProtocolError(f"SyncNet matrix hash changed: {matrix}")
        row = dict(worker)
        row.update({
            "schema_version": 1,
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "image_id": image_id,
            "video_arm": video_arm,
            "audio_arm": arm,
            "matrix_path": str(matrix),
            "video_path": str(video_row["output"]),
            "video_sha256": str(video_row["output_sha256"]),
            "audio_path": str(record["audio"][arm]["path"]),
            "audio_pcm_sha256": str(record["audio"][arm]["pcm_sha256"]),
            "fixed_score_box": dict(image["score_box"]),
        })
        sidecar = output_dir / f"{video_arm}__{arm}.json"
        write_self_hashed_json(sidecar, row)
        rows.append(verify_self_hashed_json(sidecar))
    return rows


def score(inputs: Mapping[str, Any]) -> dict[str, Any]:
    manifest = verify_self_hashed_json(_run_path("video_manifest.json"))
    video_rows = {(str(row["image_id"]), str(row["sample_id"]), str(row["video_arm"])): row for row in manifest["rows"]}
    images = _image_by_id(inputs)
    rows: list[dict[str, Any]] = []
    total = len(inputs["records"]) * len(IMAGE_IDS) * len(VIDEO_ARMS) * 2
    done = 0
    for record in inputs["records"]:
        for image_id in IMAGE_IDS:
            for arm in VIDEO_ARMS:
                rows_for_video = _score_one(video_rows[(image_id, str(record["sample_id"]), arm)], record, images[image_id])
                rows.extend(rows_for_video)
                done += len(rows_for_video)
                print(f"SCORE {done}/{total} {image_id} {record['sample_id']} {arm}", flush=True)
    payload = {"schema_version": 1, "status": "complete", "rows": rows, "count": len(rows)}
    write_self_hashed_json(_run_path("score_manifest.json"), payload)
    return verify_self_hashed_json(_run_path("score_manifest.json"))


def _existing_worker(image_id: str, sample_id: str, video_arm: str, audio_arm: str) -> dict[str, Any]:
    path = SOURCE_STATIC_ROOT / "matrices" / image_id / sample_id / f"{video_arm}__{audio_arm}__worker.json"
    if not path.is_file():
        raise ProtocolError(f"reused B025/B050 score worker is missing: {path}")
    row = _read_json(path)
    matrix = Path(str(row["matrix"])).resolve()
    if not matrix.is_file() or file_sha256(matrix) != str(row["matrix_sha256"]):
        raise ProtocolError(f"reused score matrix hash changed: {matrix}")
    row["matrix_path"] = str(matrix)
    return row


def _new_worker(rows: Mapping[tuple[str, str, str], Mapping[str, Any]], image_id: str, sample_id: str, video_arm: str, audio_arm: str) -> dict[str, Any]:
    try:
        return dict(rows[(image_id, sample_id, video_arm, audio_arm)])
    except KeyError as exc:
        raise ProtocolError(f"new score cell missing: {image_id}/{sample_id}/{video_arm}/{audio_arm}") from exc


def _condition_rows(
    score_rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    image_id: str,
    sample_id: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "N_N": _existing_worker(image_id, sample_id, "N", "N"),
        "B025_N": _existing_worker(image_id, sample_id, "B025", "N"),
        "B025_B025": _existing_worker(image_id, sample_id, "B025", "B025"),
        "B050_N": _existing_worker(image_id, sample_id, "B050", "N"),
        "B050_B050": _existing_worker(image_id, sample_id, "B050", "B050"),
        "B075_N": _new_worker(score_rows, image_id, sample_id, "B075", "N"),
        "B075_B075": _new_worker(score_rows, image_id, sample_id, "B075", "B075"),
        "MFA_LINEAR_N": _new_worker(score_rows, image_id, sample_id, "MFA_LINEAR", "N"),
        "MFA_LINEAR_MFA_LINEAR": _new_worker(score_rows, image_id, sample_id, "MFA_LINEAR", "MFA_LINEAR"),
    }
    return result


def _counts(values: Sequence[float]) -> dict[str, int]:
    values = [float(value) for value in values]
    return {"positive": sum(value > 0 for value in values), "nonpositive": sum(value <= 0 for value in values), "count": len(values)}


def analyze(inputs: Mapping[str, Any]) -> dict[str, Any]:
    score_manifest = verify_self_hashed_json(_run_path("score_manifest.json"))
    score_rows = {
        (str(row["image_id"]), str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row
        for row in score_manifest["rows"]
    }
    records: list[dict[str, Any]] = []
    observations: dict[str, list[dict[str, Any]]] = {condition: [] for condition in ANALYSIS_CONDITIONS}
    for record in inputs["records"]:
        sample_id = str(record["sample_id"])
        for image_id in IMAGE_IDS:
            cells = _condition_rows(score_rows, image_id, sample_id)
            W = common_w(tuple(cells.values()))
            metrics = {name: score_metrics(row, W) for name, row in cells.items()}
            baseline = metrics["N_N"]
            image_record: dict[str, Any] = {"sample_id": sample_id, "source_group": str(record["source_group"]), "image_id": image_id, "W": W.tolist(), "metrics": metrics, "conditions": {}}
            for condition in ANALYSIS_CONDITIONS:
                native = metrics[f"{condition}_{condition}"]
                replacement = metrics[f"{condition}_N"]
                values = {
                    "native_delta_C": float(native["C"] - baseline["C"]),
                    "replacement_delta_C": float(replacement["C"] - baseline["C"]),
                    "native_D_gain": float(baseline["D"] - native["D"]),
                    "replacement_D_gain": float(baseline["D"] - replacement["D"]),
                    "native_abs_offset_diff": abs(int(native["k"]) - int(baseline["k"])),
                    "replacement_abs_offset_diff": abs(int(replacement["k"]) - int(baseline["k"])),
                    "native_C": float(native["C"]),
                    "replacement_C": float(replacement["C"]),
                }
                image_record["conditions"][condition] = values
                observations[condition].append({"source_group": str(record["source_group"]), **values})
            records.append(image_record)

    summary: dict[str, Any] = {}
    for condition, rows in observations.items():
        groups = [str(row["source_group"]) for row in rows]
        summary[condition] = {
            "image_observation_count": len(rows),
            "source_group_count": len(set(groups)),
            "native_delta_C": bootstrap_stats([row["native_delta_C"] for row in rows], groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
            "replacement_delta_C": bootstrap_stats([row["replacement_delta_C"] for row in rows], groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
            "native_D_gain": bootstrap_stats([row["native_D_gain"] for row in rows], groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
            "replacement_D_gain": bootstrap_stats([row["replacement_D_gain"] for row in rows], groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED),
            "native_C_counts": _counts([row["native_delta_C"] for row in rows]),
            "replacement_C_counts": _counts([row["replacement_delta_C"] for row in rows]),
            "native_offset_within_1": sum(row["native_abs_offset_diff"] <= 1 for row in rows),
            "replacement_offset_within_1": sum(row["replacement_abs_offset_diff"] <= 1 for row in rows),
        }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": {"sample_ids": list(inputs["sample_ids"]), "image_ids": list(IMAGE_IDS), "record_count": len(records)},
        "conditions": summary,
        "records": records,
        "interpretation": {
            "native": "candidate audio drives Wav2Lip and remains the scored audio",
            "natural_replacement": "candidate audio drives Wav2Lip, but SyncNet is scored against natural audio",
            "no_local_swap": True,
        },
    }
    write_self_hashed_json(_run_path("analysis.json"), payload)
    _write_report(payload)
    return verify_self_hashed_json(_run_path("analysis.json"))


def _fmt(stats: Mapping[str, Any]) -> str:
    ci = stats["ci95"]
    return f"{float(stats['mean']):+.3f} [{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]"


def _write_report(analysis: Mapping[str, Any]) -> None:
    conditions = analysis["conditions"]
    lines = [
        "# 静态人脸 B075 / MFA-linear：native 与 natural replacement 对照",
        "",
        "## 实验口径",
        "",
        "本实验严格复用 B025/B050 静态实验的 3 张固定图片（3、6、9）和前 11 条音频样本，共 33 个 image×audio 组合。每个候选只生成一次静态图驱动视频：",
        "",
        "- native：候选音频驱动 Wav2Lip，SyncNet 也用候选音频评分；",
        "- natural replacement：同一个候选生成视频，评分音频换回自然音频 N；",
        "- 本实验没有读取、生成或评分 LOCAL_SWAP。",
        "",
        "B025/B050 的 N、B025、B050 视频和矩阵作为同口径参考；所有条件在每个 image×audio 单元上使用共同有效窗口 W，再按 3 张图平均后对 11 个 source group 做 bootstrap（10,000 次，seed=20260913）。",
        "",
        "## 结果",
        "",
        "Sync-C 的 Δ 为候选减自然基线，越大越好；D gain 为自然基线 D 减候选 D，越大越好。区间是按 11 个 source group 的 paired bootstrap。",
        "",
        "| 条件 | native ΔC | replacement ΔC | native D gain | replacement D gain | native C+ | replacement C+ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ANALYSIS_CONDITIONS:
        row = conditions[condition]
        lines.append(
            f"| {condition} | {_fmt(row['native_delta_C'])} | {_fmt(row['replacement_delta_C'])} | "
            f"{_fmt(row['native_D_gain'])} | {_fmt(row['replacement_D_gain'])} | "
            f"{row['native_C_counts']['positive']}/33 | {row['replacement_C_counts']['positive']}/33 |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "native 与 replacement 的差异只说明候选音频驱动的视频和自然音频评分之间存在兼容性差异；replacement 接近或超过 baseline 不能单独证明视频口型运动被改善。由于本实验是 11 条样本的静态图探索，不能授权训练或宣称跨图片泛化。",
        "",
        "## 产物",
        "",
        f"- 运行根目录：`{RUN_ROOT}/`",
        f"- 生成视频：`{RUN_ROOT}/videos/`",
        f"- SyncNet 矩阵：`{RUN_ROOT}/scores/`",
        f"- native/replacement 可播放封装（若已生成）：`{RUN_ROOT}/media/`",
        f"- 分析：`{RUN_ROOT}/analysis.json`",
        "",
        "## 样本与图片",
        "",
        f"- 图片：{', '.join(IMAGE_IDS)}",
        f"- 样本：{', '.join(str(value) for value in analysis['scope']['sample_ids'])}",
    ])
    _run_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mux_one(video: Path, audio: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.with_suffix(".json").is_file():
        return verify_self_hashed_json(output.with_suffix(".json"))
    command = [
        str(static_config.FFMPEG), "-y", "-v", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output),
    ]
    log = output.with_suffix(".ffmpeg.log")
    run_logged(command, REPO, log)
    raw_command = [str(static_config.FFMPEG), "-v", "error", "-i", str(output), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"]
    raw = subprocess.run(raw_command, cwd=str(REPO), capture_output=True, check=False)
    if raw.returncode != 0:
        raise ProtocolError(f"muxed audio extraction failed: {output}")
    source_raw = subprocess.run([str(static_config.FFMPEG), "-v", "error", "-i", str(audio), "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"], cwd=str(REPO), capture_output=True, check=False)
    if source_raw.returncode != 0 or raw.stdout != source_raw.stdout:
        raise ProtocolError(f"muxed audio PCM is not byte-identical to source audio: {output}")
    row = {
        "schema_version": 1,
        "status": "complete",
        "video": str(video.resolve()),
        "video_sha256": file_sha256(video),
        "audio": str(audio.resolve()),
        "audio_pcm_sha256": _pcm_sha256(audio)[0],
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "decoded_audio_pcm_sha256": _sha256_bytes(raw.stdout),
        "audio_pcm_exact": True,
        "command": command,
        "log": str(log.resolve()),
    }
    write_self_hashed_json(output.with_suffix(".json"), row)
    return verify_self_hashed_json(output.with_suffix(".json"))


def mux(inputs: Mapping[str, Any]) -> dict[str, Any]:
    manifest = verify_self_hashed_json(_run_path("video_manifest.json"))
    rows = []
    audio_by_sample = _record_audio(inputs)
    for video in manifest["rows"]:
        arm = str(video["video_arm"])
        image_id = str(video["image_id"])
        sample_id = str(video["sample_id"])
        for label, audio_arm in (("native", arm), ("natural_replacement", "N")):
            rows.append(_mux_one(Path(str(video["output"])), Path(str(audio_by_sample[sample_id][audio_arm]["path"])), _run_path(f"media/{arm}/{image_id}/{sample_id}/{label}.mkv")))
    payload = {"schema_version": 1, "status": "complete", "rows": rows, "count": len(rows), "local_swap_used": False}
    write_self_hashed_json(_run_path("media_manifest.json"), payload)
    return verify_self_hashed_json(_run_path("media_manifest.json"))


def run_all() -> dict[str, Any]:
    inputs = prepare()
    render(inputs)
    score(inputs)
    result = analyze(inputs)
    mux(inputs)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static-image B075/MFA-linear native/replacement experiment")
    parser.add_argument("command", choices=("prepare", "render", "score", "analyze", "mux", "all"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare()
    elif args.command == "render":
        render(_load_inputs())
    elif args.command == "score":
        score(_load_inputs())
    elif args.command == "analyze":
        analyze(_load_inputs())
    elif args.command == "mux":
        mux(_load_inputs())
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
