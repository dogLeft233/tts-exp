"""Run the Wav2Lip generation/evaluation four-cell experiment.

The command intentionally keeps orchestration in one small file.  Model
forward passes are delegated to the existing static-image Wav2Lip worker and
to :mod:`wav2lip_noise_cross_worker`; all persistent artifacts are bound to
their inputs so the independent checker can reject stale or cross-condition
files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.experiments.tts_native_gain_attribution.audio import (  # noqa: E402
    read_pcm16_wav,
    write_pcm16_wav,
)
from scripts.experiments.tts_native_gain_attribution.common import (  # noqa: E402
    ProtocolError,
    file_sha256,
    gpu_lease,
)
from scripts.experiments.wav2lip_noise_cross_metrics import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    VSHIFT,
    audio_manipulation_checks,
    bootstrap_group_summary,
    canonical_hash,
    common_support,
    construct_audio_conditions,
    curve_metrics,
    delayed_audio,
    effect_status,
    four_cell,
)


PROTOCOL_ID = "wav2lip_noise_cross_v1"
MANIFEST_SHA256 = "a0ac3e6b6a678fe12963746c238327a90ed0075a0e79049e215a833685805ac0"
CONDITIONS = ("RAW", "A0", "NOISE20")
MAIN_CELLS = {
    "RAW": ("RAW", "RAW"),
    "q00": ("A0", "A0"),
    "q01": ("A0", "NOISE20"),
    "q10": ("NOISE20", "A0"),
    "q11": ("NOISE20", "NOISE20"),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return body


def _read_self(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    recorded = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if recorded != canonical_hash(body):
        raise ProtocolError(f"artifact hash mismatch: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _cfg(config_path: Path) -> dict[str, Any]:
    target = config_path if config_path.is_absolute() else REPO / config_path
    if not target.is_file():
        raise ProtocolError(f"config is missing: {target}")
    value = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("config protocol_id is invalid")
    value["_path"] = str(target.resolve())
    value["_sha256"] = file_sha256(target)
    return value


def _run_dirs(root: Path) -> dict[str, Path]:
    values = {name: root / name for name in ("audio", "videos", "features", "scores", "audit", "figures")}
    for path in values.values():
        path.mkdir(parents=True, exist_ok=True)
    return values


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _image_rgb_hash(path: Path) -> str:
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ProtocolError(f"cannot decode portrait: {path}")
    if frame.shape[:2] != (224, 224):
        raise ProtocolError(f"portrait must be 224x224: {path} -> {frame.shape}")
    return _sha_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())


def _first_frame_portrait(video: Path, destination: Path) -> Path:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ProtocolError(f"cannot decode first frame: {video}")
    if frame.shape[:2] != (224, 224):
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), frame):
        raise ProtocolError(f"cannot write portrait: {destination}")
    temporary.replace(destination)
    return destination


def _probe(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"ffprobe failed for {path}: {result.stderr[-500:]}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"ffprobe output is invalid: {path}") from exc
    return value if isinstance(value, dict) else {}


def _load_records(cfg: dict[str, Any], dirs: dict[str, Path], ids: list[int]) -> list[dict[str, Any]]:
    manifest_path = _resolve(str(cfg["dataset_manifest"]))
    if file_sha256(manifest_path) != MANIFEST_SHA256:
        raise ProtocolError("dataset manifest hash changed")
    manifest = _read_json(manifest_path)
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ProtocolError("dataset manifest has no records list")
    natural_dir = _resolve(str(cfg["natural_carrier_dir"]))
    tts_dir = _resolve(str(cfg["tts_carrier_dir"]))
    portrait_dir = _resolve(str(cfg["portrait_dir"]))
    ffprobe = _resolve(str(cfg["media"]["ffprobe"]))
    result: list[dict[str, Any]] = []
    for sample_id in ids:
        if sample_id < 1 or sample_id > len(records):
            raise ProtocolError(f"sample id is outside manifest: {sample_id}")
        item = records[sample_id - 1]
        if str(item.get("dataset")) != "lrs3":
            raise ProtocolError(f"sample {sample_id} is not an LRS3 record")
        real_video = _resolve(str(item.get("video_local_path", "")))
        natural_carrier = natural_dir / f"{sample_id}.mp4"
        tts_carrier = tts_dir / f"{sample_id}.mp4"
        for label, path in (("real video", real_video), ("natural carrier", natural_carrier), ("tts carrier", tts_carrier)):
            if not path.is_file():
                raise ProtocolError(f"{label} is missing for {sample_id}: {path}")
        portrait = portrait_dir / f"{sample_id}.png"
        if not portrait.is_file():
            portrait = dirs["audit"] / "portraits" / f"{sample_id}.png"
            _first_frame_portrait(real_video, portrait)
        image_hash = _image_rgb_hash(portrait)
        source_group = Path(str(item.get("video_local_path"))).parent.name or str(item.get("speaker_key", "lrs3"))
        result.append({
            "sample_id": int(sample_id),
            "stem": str(item.get("stem", "")),
            "source_group": str(source_group),
            "speaker_key": str(item.get("speaker_key", "")),
            "dataset": str(item.get("dataset", "")),
            "real_video": str(real_video.resolve()),
            "natural_carrier": str(natural_carrier.resolve()),
            "tts_carrier": str(tts_carrier.resolve()),
            "portrait": str(portrait.resolve()),
            "portrait_rgb_sha256": image_hash,
            "hashes": {"real_video": file_sha256(real_video), "natural_carrier": file_sha256(natural_carrier), "tts_carrier": file_sha256(tts_carrier), "portrait": file_sha256(portrait)},
            "probe": {"real_video": _probe(real_video, ffprobe), "natural_carrier": _probe(natural_carrier, ffprobe), "tts_carrier": _probe(tts_carrier, ffprobe)},
        })
    return result


def _decode_audio(source: Path, destination: Path, ffmpeg: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.wav")
    command = [str(ffmpeg), "-y", "-v", "error", "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(temporary)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"audio decode failed for {source}: {result.stderr[-1000:]}")
    temporary.replace(destination)


def _audio_stage(root: Path, cfg: dict[str, Any], records: list[dict[str, Any]], resume: bool) -> dict[str, Any]:
    dirs = _run_dirs(root)
    manifest_path = dirs["audio"] / "manifest.json"
    if manifest_path.is_file() and resume:
        cached = _read_self(manifest_path)
        if all("natural_noise_power_binding" in row and all("decoded_input" in row.get("arms", {}).get(source, {}) for source in ("N", "T")) for row in cached.get("rows", [])):
            return cached
        # Older smoke artifacts from the same protocol are upgraded in place;
        # the deterministic PCM construction below reuses the decoded files.
    ffmpeg = _resolve(str(cfg["media"]["ffmpeg"]))
    rows: list[dict[str, Any]] = []
    for record in records:
        sid = int(record["sample_id"])
        raw_paths: dict[str, Path] = {}
        for source, carrier in (("N", Path(record["natural_carrier"])), ("T", Path(record["tts_carrier"]))):
            raw_path = dirs["audio"] / "decoded" / f"{sid}__{source}__RAW.wav"
            if not raw_path.is_file():
                _decode_audio(carrier, raw_path, ffmpeg)
            raw_paths[source] = raw_path
        natural = read_pcm16_wav(raw_paths["N"])
        tts = read_pcm16_wav(raw_paths["T"])
        constructed = construct_audio_conditions(natural, tts, sid, snr_db=float(cfg["audio"]["snr_db"]))
        power_path = dirs["audio"] / f"{sid}__natural_noise_power.npy"
        np.save(power_path, np.asarray(constructed["natural_noise_power_array"], dtype=np.float64), allow_pickle=False)
        power_binding = {"path": str(power_path.resolve()), "sha256": file_sha256(power_path), "shape": [int(item) for item in constructed["natural_noise_power_array"].shape], "dtype": "float64"}
        arm_rows: dict[str, Any] = {}
        for source in ("N", "T"):
            condition_rows: dict[str, Any] = {}
            item = constructed["conditions"][source]
            for condition in CONDITIONS:
                path = dirs["audio"] / f"{sid}__{source}__{condition}.wav"
                write_pcm16_wav(path, np.asarray(item[condition.lower() if condition != "NOISE20" else "noise20"], dtype=np.int16))
                condition_rows[condition] = {"path": str(path.resolve()), "sha256": file_sha256(path), "sample_count": int(read_pcm16_wav(path).size)}
            mask_path = dirs["audio"] / f"{sid}__{source}__activity.npy"
            np.save(mask_path, np.asarray(item["mask"], dtype=np.bool_), allow_pickle=False)
            condition_rows["activity_mask"] = {"path": str(mask_path.resolve()), "sha256": file_sha256(mask_path), "sample_count": int(item["mask"].size)}
            arm_rows[source] = {"decoded_input": {"path": str(raw_paths[source].resolve()), "sha256": file_sha256(raw_paths[source])}, "conditions": condition_rows, "metadata": constructed["metadata"]["arms"][source], "checks": audio_manipulation_checks(item["raw"], item["a0"], item["noise20"], item["mask"])}
        rows.append({"sample_id": sid, "source_group": record["source_group"], "g": constructed["metadata"]["g"], "arms": arm_rows, "natural_noise_power": constructed["metadata"]["natural_noise_power"], "natural_noise_power_binding": power_binding})
    return _write_json(manifest_path, {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "rows": rows})


def _video_signature(path: Path, ffprobe: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    digest = hashlib.sha256()
    count = 0
    shape: list[int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is None:
                shape = [int(item) for item in frame.shape]
            if shape != [int(item) for item in frame.shape]:
                raise ProtocolError(f"video frame shape changes: {path}")
            digest.update(np.ascontiguousarray(frame).tobytes())
            count += 1
    finally:
        capture.release()
    if count < 5 or shape is None:
        raise ProtocolError(f"video has too few decodable frames: {path}")
    probe = _probe(path, ffprobe)
    stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "video"), {})
    pts = {key: stream.get(key) for key in ("time_base", "start_time", "avg_frame_rate", "r_frame_rate")}
    return {"pixel_sha256": digest.hexdigest(), "frame_count": count, "frame_shape": shape, "pts": pts, "pts_sha256": canonical_hash(pts)}


def _render_one(record: dict[str, Any], source: str, condition: str, audio_path: Path, output: Path, receipt: Path, cfg: dict[str, Any], *, seed: int = 42, resume: bool = False) -> dict[str, Any]:
    if resume and output.is_file() and receipt.is_file():
        old = _read_json(receipt)
        if old.get("audio_sha256") == file_sha256(audio_path) and old.get("output_sha256") == file_sha256(output):
            if "pixel_sha256" not in old:
                old.update(_video_signature(output, _resolve(str(cfg["media"]["ffprobe"]))))
                receipt.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return old
    python = _resolve(str(cfg["wav2lip"]["python"]))
    checkpoint = _resolve(str(cfg["wav2lip"]["checkpoint"]))
    ffmpeg = _resolve(str(cfg["media"]["ffmpeg"]))
    worker = REPO / "scripts/experiments/static_image_bridge/render_worker.py"
    command = [str(python), str(worker), "--image", str(Path(record["portrait"])), "--image-rgb-sha256", str(record["portrait_rgb_sha256"]), "--audio", str(audio_path), "--box", *[str(value) for value in cfg["media"]["box_xyxy"]], "--checkpoint", str(checkpoint), "--ffmpeg", str(ffmpeg), "--outfile", str(output), "--result", str(receipt), "--batch-size", str(cfg["wav2lip"]["batch_size"]), "--seed", str(seed), "--device", str(cfg["wav2lip"]["device"])]
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not output.is_file() or not receipt.is_file():
        raise ProtocolError(f"Wav2Lip render failed for {record['sample_id']}/{source}/{condition}: {result.stderr[-1000:]}")
    value = _read_json(receipt)
    if value.get("status") != "complete" or value.get("audio_sha256") != file_sha256(audio_path):
        raise ProtocolError(f"render receipt binding failed: {receipt}")
    value.update(_video_signature(output, ffmpeg.with_name("ffprobe")))
    receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def _render_stage(root: Path, cfg: dict[str, Any], records: list[dict[str, Any]], audio_manifest: dict[str, Any], resume: bool, smoke: bool) -> dict[str, Any]:
    dirs = _run_dirs(root)
    render_manifest = dirs["videos"] / "manifest.json"
    if render_manifest.is_file() and resume:
        cached = _read_self(render_manifest)
        all_rows = list(cached.get("rows", [])) + list(cached.get("controls", []))
        if all("pixel_sha256" in row and "pts_sha256" in row and row.get("pixel_sha256") for row in all_rows):
            return cached
        # Upgrade receipts and row bindings without rerunning deterministic
        # inference.  A later resume still uses _render_one's full binding.
        ffprobe = _resolve(str(cfg["media"]["ffprobe"]))
        for row in all_rows:
            output = Path(str(row["video_path"]))
            receipt = Path(str(row["receipt_path"]))
            signature = _video_signature(output, ffprobe)
            row.update({"pixel_sha256": signature["pixel_sha256"], "pts_sha256": signature["pts_sha256"]})
            value = _read_json(receipt)
            value.update(signature)
            receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            row["receipt"] = value
        return _write_json(render_manifest, cached)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    record_by_id = {int(row["sample_id"]): row for row in records}
    audio_by_id = {int(row["sample_id"]): row for row in audio_manifest["rows"]}
    render_jobs: list[tuple[dict[str, Any], str, str, Path, Path]] = []
    for record in records:
        sid = int(record["sample_id"])
        for source in ("N", "T"):
            for condition in CONDITIONS:
                audio_path = Path(audio_by_id[sid]["arms"][source]["conditions"][condition]["path"])
                output = dirs["videos"] / f"{sid}__{source}__{condition}.mkv"
                receipt = dirs["videos"] / f"{sid}__{source}__{condition}.json"
                render_jobs.append((record, source, condition, audio_path, output))
    # One shared lease covers the serial renderer subprocesses.
    with gpu_lease(gpu_peak_bytes=2 << 30, disk_persistent_bytes=128 << 20, estimated_persistent=128 << 20):
        for record, source, condition, audio_path, output in render_jobs:
            receipt = output.with_suffix(".json")
            try:
                value = _render_one(record, source, condition, audio_path, output, receipt, cfg, resume=resume)
                rows.append({"sample_id": int(record["sample_id"]), "source": source, "condition": condition, "audio_path": str(audio_path), "audio_sha256": file_sha256(audio_path), "video_path": str(output.resolve()), "video_sha256": file_sha256(output), "pixel_sha256": value.get("pixel_sha256"), "pts_sha256": value.get("pts_sha256"), "receipt_path": str(receipt.resolve()), "receipt": value, "status": "complete"})
            except Exception as exc:
                errors.append({"sample_id": int(record["sample_id"]), "source": source, "condition": condition, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    # The four controls are exactly one independent A0 re-render for each
    # control ID/source pair, giving four additional logical videos.
    controls: list[dict[str, Any]] = []
    with gpu_lease(gpu_peak_bytes=2 << 30, disk_persistent_bytes=64 << 20, estimated_persistent=64 << 20):
        for sid in [int(value) for value in cfg["control_ids"] if int(value) in record_by_id]:
            for source in ("N", "T"):
                base = next((row for row in rows if int(row["sample_id"]) == sid and row["source"] == source and row["condition"] == "A0"), None)
                if base is None:
                    continue
                audio_path = Path(audio_by_id[sid]["arms"][source]["conditions"]["A0"]["path"])
                output = dirs["videos"] / f"{sid}__{source}__REPEAT_A0.mkv"
                receipt = dirs["videos"] / f"{sid}__{source}__REPEAT_A0.json"
                try:
                    value = _render_one(record_by_id[sid], source, "REPEAT_A0", audio_path, output, receipt, cfg, resume=False, seed=42)
                    controls.append({"sample_id": sid, "source": source, "condition": "REPEAT_A0", "audio_path": str(audio_path), "audio_sha256": file_sha256(audio_path), "video_path": str(output.resolve()), "video_sha256": file_sha256(output), "pixel_sha256": value.get("pixel_sha256"), "pts_sha256": value.get("pts_sha256"), "receipt_path": str(receipt.resolve()), "receipt": value, "status": "complete"})
                except Exception as exc:
                    errors.append({"sample_id": sid, "source": source, "condition": "REPEAT_A0", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    expected = len(records) * 2 * 3
    return _write_json(render_manifest, {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete" if len(rows) == expected and not errors else "partial", "smoke": bool(smoke), "expected_main_video_count": expected, "rows": rows, "controls": controls, "failures": errors})


def _score_stage(root: Path, cfg: dict[str, Any], records: list[dict[str, Any]], audio_manifest: dict[str, Any], render_manifest: dict[str, Any], resume: bool) -> dict[str, Any]:
    dirs = _run_dirs(root)
    score_manifest = dirs["scores"] / "manifest.json"
    worker_result = dirs["scores"] / "worker_result.json"
    if score_manifest.is_file() and resume:
        cached = _read_self(score_manifest)
        cached_cells = list(cached.get("cells", []))
        worker_cached = _read_self(worker_result) if worker_result.is_file() else {}
        worker_cells = [cell for row in worker_cached.get("rows", []) for cell in row.get("cells", [])]
        if all("video_input" in cell and "audio_input" in cell for cell in cached_cells) and all("video_input" in cell and "audio_input" in cell for cell in worker_cells):
            return cached
        # Re-run the isolated worker when an older manifest predates cell-level
        # input bindings.  Existing videos and decoded features are harmlessly
        # overwritten; no condition is silently accepted from the old cache.
    render_rows = {(int(row["sample_id"]), str(row["source"]), str(row["condition"])): row for row in render_manifest.get("rows", [])}
    audio_rows = {(int(row["sample_id"]), str(source), str(condition)): value["path"] for row in audio_manifest.get("rows", []) for source in ("N", "T") for condition, value in row["arms"][source]["conditions"].items() if condition in CONDITIONS}
    plan_rows: list[dict[str, Any]] = []
    for record in records:
        sid = int(record["sample_id"])
        for source in ("N", "T"):
            videos = {condition: render_rows[(sid, source, condition)]["video_path"] for condition in CONDITIONS if (sid, source, condition) in render_rows}
            audios = {condition: audio_rows[(sid, source, condition)] for condition in CONDITIONS}
            cells = [{"key": key, "video": video_condition, "audio": audio_condition} for key, (video_condition, audio_condition) in MAIN_CELLS.items()]
            plan_rows.append({"sample_id": sid, "source": source, "kind": "main", "videos": videos, "audios": audios, "cells": cells, "output_dir": str((dirs["features"] / f"{sid}__{source}").resolve())})
    for row in render_manifest.get("controls", []):
        sid, source = int(row["sample_id"]), str(row["source"])
        if row.get("status") != "complete":
            continue
        a0 = audio_rows[(sid, source, "A0")]
        plan_rows.append({"sample_id": sid, "source": source, "kind": "repeat", "videos": {"A0": row["video_path"]}, "audios": {"A0": a0}, "cells": [{"key": "REPEAT", "video": "A0", "audio": "A0"}], "output_dir": str((dirs["features"] / f"{sid}__{source}__repeat").resolve())})
        delayed_path = dirs["audio"] / f"{sid}__{source}__A0_DELAY_200MS.wav"
        if not delayed_path.is_file():
            write_pcm16_wav(delayed_path, delayed_audio(read_pcm16_wav(Path(a0)), int(cfg["audio"]["delay_ms"])))
        plan_rows.append({"sample_id": sid, "source": source, "kind": "delay", "videos": {"A0": render_rows[(sid, source, "A0")]["video_path"]}, "audios": {"DELAY": str(delayed_path.resolve())}, "cells": [{"key": "DELAY", "video": "A0", "audio": "DELAY"}], "output_dir": str((dirs["features"] / f"{sid}__{source}__delay").resolve())})
    plan_path = dirs["scores"] / "worker_plan.json"
    _write_json(plan_path, {"schema_version": 1, "protocol_id": PROTOCOL_ID, "rows": plan_rows})
    if not (resume and worker_result.is_file() and all("video_input" in cell and "audio_input" in cell for row in _read_self(worker_result).get("rows", []) for cell in row.get("cells", []))):
        python = _resolve(str(cfg["syncnet"]["python"]))
        worker = REPO / "scripts/experiments/wav2lip_noise_cross_worker.py"
        env = dict(os.environ)
        env["FFMPEG"] = str(_resolve(str(cfg["media"]["ffmpeg"])))
        command = [str(python), str(worker), "--plan", str(plan_path), "--result", str(worker_result), "--model", str(_resolve(str(cfg["syncnet"]["checkpoint"]))), "--batch-size", str(cfg["syncnet"]["batch_size"]), "--device", str(cfg["syncnet"]["device"]), "--ffmpeg", str(_resolve(str(cfg["media"]["ffmpeg"]))) ]
        with gpu_lease(gpu_peak_bytes=2 << 30, disk_persistent_bytes=256 << 20, estimated_persistent=256 << 20):
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        (dirs["scores"] / "worker.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0 and not worker_result.is_file():
            raise ProtocolError(f"SyncNet worker failed: {result.stderr[-1000:]}")
    worker_payload = _read_self(worker_result)
    cells: list[dict[str, Any]] = []
    for row in worker_payload.get("rows", []):
        for cell in row.get("cells", []):
            cells.append({"sample_id": int(row["sample_id"]), "source": str(row["source"]), "kind": str(row["kind"]), "key": str(cell["key"]), "video_condition": str(cell["video"]), "audio_condition": str(cell["audio"]), "video_input": cell["video_input"], "video_input_sha256": cell["video_input_sha256"], "audio_input": cell["audio_input"], "audio_input_sha256": cell["audio_input_sha256"], "matrix": cell["matrix"], "visual_feature": cell["visual_feature"], "audio_feature": cell["audio_feature"], "worker_row": {"sample_id": int(row["sample_id"]), "source": str(row["source"]), "kind": str(row["kind"])}, "status": "complete"})
    expected_main = len(records) * 2 * 5
    return _write_json(score_manifest, {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete" if len([item for item in cells if item["kind"] == "main"]) == expected_main else "partial", "expected_main_score_count": expected_main, "worker_result": str(worker_result.resolve()), "cells": cells, "failures": worker_payload.get("failures", [])})


def _load_matrix(cell: dict[str, Any]) -> np.ndarray:
    path = Path(str(cell["matrix"]["path"]))
    if file_sha256(path) != str(cell["matrix"]["sha256"]):
        raise ProtocolError(f"matrix hash mismatch: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise ProtocolError(f"invalid matrix: {path}")
    return value


def _analyze_stage(root: Path, cfg: dict[str, Any], audio_manifest: dict[str, Any], render_manifest: dict[str, Any], score_manifest: dict[str, Any], smoke: bool) -> dict[str, Any]:
    dirs = _run_dirs(root)
    main_cells: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for cell in score_manifest.get("cells", []):
        if cell.get("kind") == "main":
            main_cells.setdefault((int(cell["sample_id"]), str(cell["source"])), {})[str(cell["key"])] = cell
    records: list[dict[str, Any]] = []
    for (sid, source), cells in sorted(main_cells.items()):
        if not all(key in cells for key in MAIN_CELLS):
            records.append({"sample_id": sid, "source": source, "status": "INCOMPLETE_CELLS"})
            continue
        matrices = {key: _load_matrix(cells[key]) for key in MAIN_CELLS}
        support, support_status = common_support(list(matrices.values()))
        if support_status != "PASS":
            records.append({"sample_id": sid, "source": source, "status": support_status, "support_rows": support})
            continue
        metrics = {key: curve_metrics(matrix, support) for key, matrix in matrices.items()}
        decomposition = four_cell(metrics["q00"]["sync_c"], metrics["q01"]["sync_c"], metrics["q10"]["sync_c"], metrics["q11"]["sync_c"])
        row = {"sample_id": sid, "source": source, "source_group": next((str(item["source_group"]) for item in audio_manifest.get("rows", []) if int(item["sample_id"]) == sid), str(sid)), "status": "COMPLETE", "support_count": len(support), "support_rows": support, "metrics": metrics, "E": decomposition["evaluation"], "G": decomposition["generation"], "I": decomposition["interaction"], "Total": decomposition["total"], "raw_advantage_key": float(metrics["RAW"]["sync_c"]) }
        records.append(row)
    complete = [row for row in records if row.get("status") == "COMPLETE"]
    by_source = {source: [row for row in complete if row["source"] == source] for source in ("N", "T")}
    summary: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "smoke": bool(smoke), "complete_source_count": {source: len(values) for source, values in by_source.items()}, "primary": {}, "secondary": {}, "status": "COMPLETE" if complete else "INCOMPLETE"}
    for source, values in by_source.items():
        label = f"E_{source}"
        for metric, field in ((label, "E"), (f"G_{source}", "G"), (f"I_{source}", "I"), (f"Total_{source}", "Total")):
            stats = bootstrap_group_summary(values, field, primary=(metric in {"E_T", "G_T"}), seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
            stats["effect_status"] = effect_status(stats, primary=(metric in {"E_T", "G_T"}))
            (summary["primary"] if metric in {"E_T", "G_T"} else summary["secondary"])[metric] = stats
    pair_map = {(int(row["sample_id"]), str(row["source"])): row for row in complete}
    raw_values: list[dict[str, Any]] = []
    for sid in sorted({int(row["sample_id"]) for row in complete}):
        if (sid, "N") in pair_map and (sid, "T") in pair_map:
            raw_values.append({"sample_id": sid, "source_group": pair_map[(sid, "T")]["source_group"], "raw_advantage": float(pair_map[(sid, "T")]["metrics"]["RAW"]["sync_c"] - pair_map[(sid, "N")]["metrics"]["RAW"]["sync_c"]), "a0_advantage": float(pair_map[(sid, "T")]["metrics"]["q00"]["sync_c"] - pair_map[(sid, "N")]["metrics"]["q00"]["sync_c"])})
    if raw_values:
        summary["secondary"]["RAW_T_minus_N"] = bootstrap_group_summary(raw_values, "raw_advantage", seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
        summary["secondary"]["A0_T_minus_N"] = bootstrap_group_summary(raw_values, "a0_advantage", seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
    _write_json(dirs["audit"] / "per_source.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "rows": records})
    _write_json(dirs["audit"] / "summary.json", summary)
    csv_path = dirs["audit"] / "per_source.csv"
    if records:
        fields = ["sample_id", "source", "source_group", "status", "support_count", "E", "G", "I", "Total"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({field: row.get(field, "") for field in fields})
    report = ["# Wav2Lip noise-cross experiment", "", f"- protocol: `{PROTOCOL_ID}`", f"- smoke: `{bool(smoke)}`", f"- complete source arms: `{summary['complete_source_count']}`", ""]
    for key, value in summary["primary"].items():
        report.append(f"- {key}: mean={value['mean']:.6f}, CI97.5%=[{value['ci_primary'][0]:.6f}, {value['ci_primary'][1]:.6f}], status={value['effect_status']}")
    report += ["", "This is a response-to-the-specified-20-dB-coloured-noise probe. It does not establish that TTS makes the true mouth motion more accurate or explain the full native TTS gain."]
    (dirs["audit"] / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def _run_checker(root: Path) -> int:
    checker = REPO / "scripts/experiments/check_wav2lip_noise_cross.py"
    result = subprocess.run([sys.executable, str(checker), "--run-dir", str(root)], capture_output=True, text=True, check=False)
    (root / "checker.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return int(result.returncode)


def run(*, config_path: Path, run_dir: Path, smoke: bool, stage: str, resume: bool) -> int:
    cfg = _cfg(config_path)
    root = run_dir if run_dir.is_absolute() else REPO / run_dir
    root.mkdir(parents=True, exist_ok=True)
    dirs = _run_dirs(root)
    ids = [int(value) for value in (cfg["smoke_ids"] if smoke else cfg["sample_ids"])]
    if root.joinpath("protocol.json").is_file() and not resume:
        raise ProtocolError(f"run exists; use --resume: {root}")
    records = _load_records(cfg, dirs, ids)
    protocol = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "locked", "run_dir": str(root.resolve()), "smoke": bool(smoke), "sample_ids": ids, "control_ids": [int(value) for value in cfg["control_ids"] if int(value) in ids], "config_path": cfg["_path"], "config_sha256": cfg["_sha256"], "manifest_sha256": MANIFEST_SHA256, "checkpoint_sha256": file_sha256(_resolve(str(cfg["wav2lip"]["checkpoint"]))), "syncnet_checkpoint_sha256": file_sha256(_resolve(str(cfg["syncnet"]["checkpoint"]))), "expected": {"main_videos": len(records) * 2 * 3, "repeat_videos": len([value for value in cfg["control_ids"] if int(value) in ids]) * 2, "main_scores": len(records) * 2 * 5, "control_scores": len([value for value in cfg["control_ids"] if int(value) in ids]) * 4}}
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "inputs.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "records": records})
    if stage in ("audit",):
        return 0
    audio_manifest = _audio_stage(root, cfg, records, resume) if stage in ("audio", "render", "score", "analyze", "validate", "all") else _read_self(dirs["audio"] / "manifest.json")
    if stage == "audio":
        return 0
    render_manifest = _render_stage(root, cfg, records, audio_manifest, resume, smoke) if stage in ("render", "score", "analyze", "validate", "all") else _read_self(dirs["videos"] / "manifest.json")
    if stage == "render":
        return 0
    score_manifest = _score_stage(root, cfg, records, audio_manifest, render_manifest, resume) if stage in ("score", "analyze", "validate", "all") else _read_self(dirs["scores"] / "manifest.json")
    if stage == "score":
        return 0
    _analyze_stage(root, cfg, audio_manifest, render_manifest, score_manifest, smoke)
    if stage == "analyze":
        return 0
    checker_status = _run_checker(root)
    return 0 if checker_status == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("audit", "audio", "render", "score", "analyze", "validate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        return run(config_path=args.config, run_dir=args.run_dir, smoke=bool(args.smoke), stage=args.stage, resume=bool(args.resume))
    except Exception as exc:
        root = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "error.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[wav2lip_noise_cross] BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
