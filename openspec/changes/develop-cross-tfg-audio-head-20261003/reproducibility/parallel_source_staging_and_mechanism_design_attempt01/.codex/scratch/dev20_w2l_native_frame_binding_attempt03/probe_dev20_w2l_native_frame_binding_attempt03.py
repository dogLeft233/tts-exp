#!/usr/bin/env python3
"""Read-only native Wav2Lip frontend and exact inference chunk-count probe."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib
import importlib.metadata
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def require_file(path: Path, expected_sha256: str, label: str,
                 expected_bytes: int | None = None) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} is not a regular nonsymlink file: {path}")
    actual_sha = sha256_file(path)
    if ((expected_bytes is not None and info.st_size != expected_bytes)
            or actual_sha != expected_sha256):
        raise RuntimeError(f"{label} pin mismatch: {path}")
    return {"path": str(path), "bytes": info.st_size, "sha256": actual_sha}


def check_wav_header(path: Path, expected_frames: int) -> None:
    """Check the frozen files are IEEE float32 mono at 16 kHz without decoding."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError(f"not a little-endian RIFF/WAVE source: {path}")
    declared_end = min(len(raw), int.from_bytes(raw[4:8], "little") + 8)
    pos = 12
    fmt: tuple[int, int, int, int, int] | None = None
    data_size: int | None = None
    while pos + 8 <= declared_end:
        chunk_id = raw[pos:pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        start = pos + 8
        end = start + chunk_size
        if end > declared_end:
            raise RuntimeError(f"truncated WAV chunk in {path}")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise RuntimeError(f"invalid WAV fmt chunk in {path}")
            fmt = (
                int.from_bytes(raw[start:start + 2], "little"),
                int.from_bytes(raw[start + 2:start + 4], "little"),
                int.from_bytes(raw[start + 4:start + 8], "little"),
                int.from_bytes(raw[start + 12:start + 14], "little"),
                int.from_bytes(raw[start + 14:start + 16], "little"),
            )
        elif chunk_id == b"data":
            data_size = chunk_size
        pos = end + (chunk_size & 1)
    if fmt != (3, 1, 16000, 4, 32):
        raise RuntimeError(f"source WAV format is not IEEE_FLOAT_32_LE mono 16 kHz: {path}, {fmt!r}")
    if data_size != expected_frames * 4:
        raise RuntimeError(f"source WAV data length differs from the frozen sample count: {path}")


def check_plan(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pins = config["frozen_plan"]
    plan_path = Path(pins["plan_path"])
    rows_path = Path(pins["rows_path"])
    manifest_path = Path(pins["manifest_path"])
    plan_pin = require_file(plan_path, pins["plan_sha256"], "frozen Dev20 plan", pins["plan_bytes"])
    rows_pin = require_file(rows_path, pins["rows_sha256"], "frozen Dev20 rows", pins["rows_bytes"])
    manifest_pin = require_file(manifest_path, pins["manifest_sha256"], "frozen Dev20 plan manifest",
                                pins["manifest_bytes"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("run_id") != pins["run_id"]:
        raise RuntimeError("base Dev20 plan is not the pinned frozen plan")
    if sha256_bytes(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})) != pins["plan_canonical_sha256"]:
        raise RuntimeError("frozen Dev20 canonical plan seal mismatch")
    if not isinstance(plan.get("rows"), list) or len(plan["rows"]) != 180:
        raise RuntimeError("frozen Dev20 plan does not contain its 180 planned rows")
    rows_raw = rows_path.read_bytes()
    expected_rows_raw = b"".join(canonical(row) + b"\n" for row in plan["rows"])
    if rows_raw != expected_rows_raw:
        raise RuntimeError("planned_rows.jsonl does not match the frozen plan rows byte-for-byte")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (sha256_bytes(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}))
            != manifest.get("manifest_sha256")
            or manifest.get("manifest_sha256") != pins["manifest_canonical_sha256"]
            or manifest.get("plan_raw_sha256") != plan_pin["sha256"]
            or manifest.get("planned_rows_sha256") != rows_pin["sha256"]
            or manifest.get("plan_sha256") != plan.get("plan_sha256")
            or manifest.get("planned_rows") != len(plan["rows"])):
        raise RuntimeError("frozen Dev20 plan manifest does not seal the exact plan/rows")
    if manifest.get("confirm_audio_read") is not False or manifest.get("scores_read") is not False:
        raise RuntimeError("the frozen plan manifest does not prove pre-score, non-confirmation scope")

    by_sample: dict[str, dict[str, Any]] = {}
    speaker_by_sample: dict[str, str] = {}
    rows_by_sample: dict[str, list[str]] = {}
    face_by_sample: dict[str, dict[str, Any]] = {}
    method_counts: Counter[str] = Counter()
    seeds: set[int] = set()
    for row in plan["rows"]:
        if (row.get("split") != "dev" or row.get("model_id") != "wav2lip_gan_bac9a81e"
                or row.get("audio_frontend") != "wav2lip_official_16k_float32_mel"
                or row.get("generation_settings", {}).get("face_mode") != "static_image"
                or row.get("generation_settings", {}).get("fps_num") != 25
                or row.get("generation_settings", {}).get("fps_den") != 1):
            raise RuntimeError("a frozen Dev row differs from the Wav2Lip static-image 25 fps scope")
        natural = row.get("natural_pcm")
        score_audio = row.get("score_audio")
        if not isinstance(natural, dict) or score_audio != natural:
            raise RuntimeError("frozen plan row's natural/score PCM identities differ")
        sample_id = str(row["sample_id"])
        face = row.get("source_face")
        if (not isinstance(face, dict)
                or Path(str(face.get("path", ""))).suffix.lower() not in {".jpg", ".jpeg", ".png"}):
            raise RuntimeError("frozen W2L static-image face reference has an unsupported image suffix")
        if sample_id in face_by_sample and face_by_sample[sample_id] != face:
            raise RuntimeError(f"frozen Dev rows disagree on static face identity for {sample_id}")
        face_by_sample[sample_id] = face
        existing = by_sample.get(sample_id)
        if existing is not None and existing != natural:
            raise RuntimeError(f"frozen Dev rows disagree on source PCM identity for {sample_id}")
        by_sample[sample_id] = natural
        speaker = str(row["speaker_id"])
        if sample_id in speaker_by_sample and speaker_by_sample[sample_id] != speaker:
            raise RuntimeError(f"frozen Dev rows disagree on speaker identity for {sample_id}")
        speaker_by_sample[sample_id] = speaker
        rows_by_sample.setdefault(sample_id, []).append(str(row["row_id"]))
        method_counts[str(row["method_id"])] += 1
        seeds.add(int(row["seed"]))
    if (len(by_sample) != 20 or sorted(map(len, rows_by_sample.values())) != [9] * 20
            or method_counts != Counter({"identity": 60, "rms_fixed": 60, "rms_localcompress": 60})
            or seeds != {20261003, 20261004, 20261005}):
        raise RuntimeError("frozen plan must contain exactly 20 source clips, repeated across 9 rows each")
    refs = []
    for sample_id in sorted(by_sample):
        ref = dict(by_sample[sample_id])
        if (ref.get("sample_rate_hz") != 16000 or ref.get("channels") != 1
                or ref.get("subtype") != "IEEE_FLOAT_32_LE"):
            raise RuntimeError(f"frozen Dev PCM is not canonical mono float32/16 kHz: {sample_id}")
        wav_path = Path(ref["path"])
        pin = require_file(wav_path, str(ref["file_sha256"]), f"source WAV {sample_id}")
        check_wav_header(wav_path, int(ref["sample_count"]))
        refs.append({"sample_id": sample_id, "source_pcm": ref, "source_wav_pin": pin,
                     "speaker_id": speaker_by_sample[sample_id],
                     "planned_row_ids": sorted(rows_by_sample[sample_id])})
    return plan, refs, [plan_pin, rows_pin, manifest_pin]


def find_chunk_loop(inference_path: Path, expected_sha256: str) -> tuple[str, ast.Module, int]:
    raw = inference_path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError("deployed inference.py does not match the exact W2L source pin")
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(inference_path))
    step_assignment = None
    main_fn = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mel_step_size"
                                                for t in node.targets):
            step_assignment = node
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
    if step_assignment is None or not isinstance(step_assignment.value, ast.Constant) or step_assignment.value.value != 16:
        raise RuntimeError("pinned inference.py mel_step_size is not exactly 16")
    if main_fn is None:
        raise RuntimeError("pinned inference.py has no main() function")
    selected: dict[str, ast.stmt] = {}
    loop = None
    for node in main_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value == 1:
                segment = ast.get_source_segment(source, node) or ""
                if "mel_chunks.append" in segment and "start_idx + mel_step_size > len(mel[0])" in segment:
                    loop = node
            continue
        name = node.targets[0].id
        if name in {"mel_chunks", "mel_idx_multiplier", "i"}:
            selected[name] = node
    if set(selected) != {"mel_chunks", "mel_idx_multiplier", "i"} or loop is None:
        raise RuntimeError("pinned inference.py mel chunk block is not the expected official block")
    step = int(step_assignment.value.value)
    block = ast.Module(body=[step_assignment, selected["mel_chunks"], selected["mel_idx_multiplier"],
                             selected["i"], loop], type_ignores=[])
    ast.fix_missing_locations(block)
    code = compile(block, str(inference_path), "exec")
    segment = "\n".join(ast.get_source_segment(source, node) or ""
                         for node in (step_assignment, selected["mel_chunks"],
                                      selected["mel_idx_multiplier"], selected["i"], loop))
    return sha256_bytes(segment.encode("utf-8")), block, step


def verify_static_frame_mapping(inference_path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = inference_path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError("deployed inference.py does not match the exact W2L source pin")
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=str(inference_path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    datagen = functions.get("datagen")
    main_fn = functions.get("main")
    if datagen is None or main_fn is None:
        raise RuntimeError("pinned inference.py lacks its expected datagen/main functions")

    static_loop = None
    for node in ast.walk(datagen):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        if (isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate"
                and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Name)
                and node.iter.args[0].id == "mels"):
            static_loop = node
            break
    if static_loop is None:
        raise RuntimeError("pinned W2L datagen no longer emits per item from enumerate(mels)")
    selectors = [node for node in ast.walk(static_loop) if isinstance(node, ast.IfExp)
                 and isinstance(node.test, ast.Attribute) and node.test.attr == "static"
                 and isinstance(node.test.value, ast.Name) and node.test.value.id == "args"
                 and isinstance(node.body, ast.Constant) and node.body.value == 0]
    frame_batch_append = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "frame_batch"
        and node.func.attr == "append" and len(node.args) == 1
        and isinstance(node.args[0], ast.Name) and node.args[0].id == "frame_to_save"
        for node in ast.walk(static_loop))
    if not selectors or not frame_batch_append:
        raise RuntimeError("pinned W2L datagen static-image path does not map each mel item to a frame")

    writer_loop = None
    writer_calls = []
    for node in ast.walk(main_fn):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "zip":
            continue
        if [arg.id for arg in node.iter.args if isinstance(arg, ast.Name)] != ["pred", "frames", "coords"]:
            continue
        calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                 and isinstance(call.func, ast.Attribute) and call.func.attr == "write"
                 and isinstance(call.func.value, ast.Name) and call.func.value.id == "out"
                 and len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                 and call.args[0].id == "f"]
        if calls:
            writer_loop = node
            writer_calls = calls
            break
    if writer_loop is None or len(writer_calls) != 1:
        raise RuntimeError("pinned W2L inference does not write each generated frame once through VideoWriter")

    datagen_segment = ast.get_source_segment(source, static_loop) or ""
    writer_segment = ast.get_source_segment(source, writer_loop) or ""
    return {
        "static_datagen_loop_lineno": int(static_loop.lineno),
        "static_datagen_loop_end_lineno": int(static_loop.end_lineno),
        "static_datagen_loop_source_sha256": sha256_bytes(datagen_segment.encode("utf-8")),
        "writer_loop_lineno": int(writer_loop.lineno),
        "writer_loop_end_lineno": int(writer_loop.end_lineno),
        "writer_loop_source_sha256": sha256_bytes(writer_segment.encode("utf-8")),
        "one_static_frame_per_mel_chunk_verified_from_pinned_ast": True,
        "video_writer_write_per_generated_frame_verified_from_pinned_ast": True,
    }


def expected_environment() -> dict[str, str]:
    return {
        "PYTHONHASHSEED": "20261003", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def check_environment(config: dict[str, Any]) -> None:
    if sys.argv[0] == "-c" or not sys.dont_write_bytecode:
        raise RuntimeError("probe must run as a normal -B script with bytecode writes disabled")
    expected_env = expected_environment()
    bad_env = {k: {"expected": v, "actual": os.environ.get(k)}
               for k, v in expected_env.items() if os.environ.get(k) != v}
    if bad_env:
        raise RuntimeError(f"CPU-only probe environment mismatch: {bad_env}")
    if "torch" in sys.modules or "torch.cuda" in sys.modules:
        raise RuntimeError("torch was imported before the read-only mel-count probe")
    py_pin = Path(config["runtime"]["python_path"])
    if not os.path.samefile(sys.executable, py_pin):
        raise RuntimeError("probe interpreter is not the pinned Wav2Lip environment Python")


def check_runtime_sources(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    w2l_root = Path(config["runtime"]["wav2lip_root"])
    source_pins = []
    for rel, entry in sorted(config["runtime"]["source_pins"].items()):
        source_pins.append(require_file(w2l_root / rel, entry["sha256"],
                                        f"deployed W2L source {rel}", int(entry["bytes"])))
    bootstrap_path = Path(config["runtime"]["bootstrap_path"])
    bootstrap_pin = require_file(bootstrap_path, config["runtime"]["bootstrap_sha256"],
                                 "W2L compatibility bootstrap", int(config["runtime"]["bootstrap_bytes"]))
    site = sys.modules.get("sitecustomize")
    if site is None or str(Path(site.__file__).resolve(strict=True)) != str(bootstrap_path.resolve(strict=True)):
        raise RuntimeError("pinned CPU bootstrap was not imported by normal Python site initialization")
    try:
        librosa_version = importlib.metadata.version("librosa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("pinned W2L environment has no librosa distribution metadata") from exc
    if librosa_version != config["runtime"]["librosa_version"]:
        raise RuntimeError("librosa distribution version differs from the actual W2L frontend runtime pin")
    return source_pins, bootstrap_pin


def seal_receipt(result: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(result)
    sealed["receipt_sha256"] = sha256_bytes(canonical(sealed))
    return sealed


def run_static_preflight(config: dict[str, Any]) -> dict[str, Any]:
    check_environment(config)
    plan, clips, input_pins = check_plan(config)
    source_pins, bootstrap_pin = check_runtime_sources(config)
    return seal_receipt({
        "schema": "cross_tfg_dev20_w2l_native_frame_binding_preflight_v1",
        "status": "PASS_READONLY_CPU_SOURCE_AND_PCM_PREFLIGHT",
        "scope": {"source_inputs_read_only": True, "confirmation_audio_opened": False,
                  "scores_opened": False, "models_or_scorers_loaded": False,
                  "torch_imported": False, "cuda_visible_devices": ""},
        "frozen_plan": {"run_id": plan["run_id"], "plan_sha256": plan["plan_sha256"],
                        "plan_raw_sha256": input_pins[0]["sha256"],
                        "planned_rows_sha256": input_pins[1]["sha256"],
                        "manifest_raw_sha256": input_pins[2]["sha256"],
                        "sample_count": len(clips), "planned_rows": len(plan["rows"])},
        "runtime": {"python_path": str(Path(sys.executable).resolve(strict=True)),
                    "librosa_version": config["runtime"]["librosa_version"],
                    "wav2lip_sources": source_pins, "bootstrap": bootstrap_pin},
        "source_pcm_inputs": [
            {"sample_id": clip["sample_id"], "speaker_id": clip["speaker_id"],
             "sample_count": int(clip["source_pcm"]["sample_count"]),
             "source_file_sha256": clip["source_wav_pin"]["sha256"],
             "source_pcm_sha256": clip["source_pcm"]["pcm_sha256"]}
            for clip in clips
        ],
    })


def read_sealed_json(path: Path, expected_schema: str, expected_status: str) -> tuple[dict[str, Any], str]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"receipt is not a regular nonsymlink file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    expected_seal = value.get("receipt_sha256")
    if (value.get("schema") != expected_schema or value.get("status") != expected_status
            or expected_seal != sha256_bytes(canonical({k: v for k, v in value.items() if k != "receipt_sha256"}))):
        raise RuntimeError(f"receipt schema/status/self-seal check failed: {path}")
    return value, sha256_bytes(raw)


def run_probe(config: dict[str, Any], preflight_path: Path) -> dict[str, Any]:
    preflight, preflight_raw_sha = read_sealed_json(
        preflight_path, "cross_tfg_dev20_w2l_native_frame_binding_preflight_v1",
        "PASS_READONLY_CPU_SOURCE_AND_PCM_PREFLIGHT")
    check_environment(config)
    plan, clips, input_pins = check_plan(config)
    source_pins, bootstrap_pin = check_runtime_sources(config)
    if (preflight.get("frozen_plan", {}).get("plan_raw_sha256") != input_pins[0]["sha256"]
            or preflight.get("frozen_plan", {}).get("planned_rows_sha256") != input_pins[1]["sha256"]
            or preflight.get("frozen_plan", {}).get("manifest_raw_sha256") != input_pins[2]["sha256"]):
        raise RuntimeError("preflight receipt and probe input plan/rows identities differ")

    import sitecustomize
    shim_record = sitecustomize._install_wav2lip_librosa_positional_compatibility()
    if shim_record.get("librosa_version") != config["runtime"]["librosa_version"]:
        raise RuntimeError("loaded librosa version differs from the actual W2L frontend runtime pin")
    w2l_root = Path(config["runtime"]["wav2lip_root"])
    sys.path.insert(0, str(w2l_root))
    audio = importlib.import_module("audio")
    if str(Path(audio.__file__).resolve(strict=True)) != str((w2l_root / "audio.py").resolve(strict=True)):
        raise RuntimeError("imported audio frontend does not originate from the pinned deployed W2L source")
    if "torch" in sys.modules or "torch.cuda" in sys.modules:
        raise RuntimeError("native audio frontend import unexpectedly imported torch")
    hparams = importlib.import_module("hparams")
    if str(Path(hparams.__file__).resolve(strict=True)) != str((w2l_root / "hparams.py").resolve(strict=True)):
        raise RuntimeError("imported W2L hparams do not originate from the pinned deployed source")
    import numpy as np

    loop_sha, loop_module, mel_step_size = find_chunk_loop(
        w2l_root / "inference.py", config["runtime"]["source_pins"]["inference.py"]["sha256"])
    mapping_evidence = verify_static_frame_mapping(
        w2l_root / "inference.py", config["runtime"]["source_pins"]["inference.py"]["sha256"])
    clip_results = []
    for clip in clips:
        ref = clip["source_pcm"]
        waveform = audio.load_wav(ref["path"], 16000)
        waveform = np.asarray(waveform)
        if waveform.dtype != np.dtype("float32") or waveform.ndim != 1 or waveform.size != int(ref["sample_count"]):
            raise RuntimeError(f"native W2L load_wav result differs from frozen float32/length for {clip['sample_id']}")
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"native W2L load_wav returned nonfinite PCM for {clip['sample_id']}")
        pcm_sha = sha256_bytes(np.ascontiguousarray(waveform, dtype=np.dtype("<f4")).tobytes())
        if pcm_sha != ref["pcm_sha256"]:
            raise RuntimeError(f"native W2L load_wav PCM bytes differ from frozen PCM pin for {clip['sample_id']}")
        mel = np.asarray(audio.melspectrogram(waveform))
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < mel_step_size or not np.isfinite(mel).all():
            raise RuntimeError(f"native W2L melspectrogram shape/content is invalid for {clip['sample_id']}")
        fps_num = int(plan["rows"][0]["generation_settings"]["fps_num"])
        fps_den = int(plan["rows"][0]["generation_settings"]["fps_den"])
        scope = {"fps": fps_num / fps_den, "mel": mel}
        exec(compile(loop_module, str(w2l_root / "inference.py"), "exec"), scope, scope)
        chunks = scope.get("mel_chunks")
        if not isinstance(chunks, list) or not chunks or any(tuple(chunk.shape) != (80, mel_step_size) for chunk in chunks):
            raise RuntimeError(f"official W2L chunk loop emitted an unexpected chunk sequence for {clip['sample_id']}")
        mel_ptr = int(mel.__array_interface__["data"][0])
        frame_stride = int(mel.strides[1])
        if frame_stride <= 0:
            raise RuntimeError("native W2L mel frame stride must be positive")
        chunk_starts = []
        for chunk in chunks:
            chunk_ptr = int(chunk.__array_interface__["data"][0])
            byte_offset = chunk_ptr - mel_ptr
            if (byte_offset < 0 or byte_offset % frame_stride != 0
                    or not np.shares_memory(chunk, mel)):
                raise RuntimeError(f"W2L chunk is not a source mel view for {clip['sample_id']}")
            start = byte_offset // frame_stride
            if not np.array_equal(chunk, mel[:, start:start + mel_step_size]):
                raise RuntimeError(f"W2L chunk values do not equal the recorded source mel window for {clip['sample_id']}")
            chunk_starts.append(int(start))
        final_tail_start = int(mel.shape[1] - mel_step_size)
        if chunk_starts[-1] != final_tail_start:
            raise RuntimeError(f"W2L final chunk does not use its exact tail window for {clip['sample_id']}")
        clip_results.append({
            "sample_id": clip["sample_id"],
            "speaker_id": clip["speaker_id"],
            "source_path": ref["path"],
            "source_file_sha256": ref["file_sha256"],
            "source_pcm_float32_sha256": pcm_sha,
            "sample_rate_hz": int(ref["sample_rate_hz"]),
            "source_channels": int(ref["channels"]),
            "sample_count": int(ref["sample_count"]),
            "mel_shape": [int(mel.shape[0]), int(mel.shape[1])],
            "mel_dtype": str(mel.dtype),
            "fps_num": fps_num,
            "fps_den": fps_den,
            "mel_step_size": int(mel_step_size),
            "expected_inference_frame_count": int(len(chunks)),
            "frame_count_observed_from_model_or_video": False,
            "chunk_start_mel_frame_indices": chunk_starts,
            "final_tail_start_mel_frame_index": final_tail_start,
            "last_two_chunk_start_indices": chunk_starts[-2:],
            "final_chunk_reuses_previous_start": len(chunk_starts) > 1 and chunk_starts[-1] == chunk_starts[-2],
            "bound_row_count": len(clip["planned_row_ids"]),
            "bound_row_ids": clip["planned_row_ids"],
        })
    if "torch" in sys.modules or "torch.cuda" in sys.modules:
        raise RuntimeError("native CPU frontend probe imported torch/model code")
    return seal_receipt({
        "schema": "cross_tfg_dev20_w2l_native_frame_binding_v1",
        "status": "PASS_CPU_NATIVE_FRONTEND_AND_EXPECTED_FRAME_COUNT_BINDING",
        "scope": {
            "source_inputs_read_only": True,
            "one_new_immutable_receipt_written": True,
            "dev_only": True,
            "confirmation_audio_opened": False,
            "scores_opened": False,
            "model_or_scorer_imported_or_invoked": False,
            "cuda_visible_devices": "",
            "torch_imported": False,
            "candidate_selection": False,
        },
        "frozen_plan": {
            "run_id": plan["run_id"],
            "plan_sha256": plan["plan_sha256"],
            "plan_raw_sha256": input_pins[0]["sha256"],
            "planned_rows_sha256": input_pins[1]["sha256"],
            "plan_manifest_raw_sha256": input_pins[2]["sha256"],
            "sample_count": len(clip_results),
            "planned_rows": len(plan["rows"]),
        },
        "preflight_receipt": {"path": str(preflight_path), "raw_sha256": preflight_raw_sha,
                              "receipt_sha256": preflight["receipt_sha256"]},
        "runtime": {
            "python_path": str(Path(sys.executable).resolve(strict=True)),
            "wav2lip_root": str(w2l_root),
            "wav2lip_sources": source_pins,
            "bootstrap": bootstrap_pin,
            "librosa_compatibility": shim_record,
            "exact_chunk_loop_source_sha256": loop_sha,
            "static_frame_mapping_evidence": mapping_evidence,
            "inference_model_or_face_modules_imported": False,
        },
        "frame_count_semantics": {
            "count_source": "native audio.melspectrogram(wav2lip.audio.load_wav(frozen_natural_wav, 16000)) then AST-executed exact pinned inference.py mel chunk block",
            "static_image_output_mapping": "one yielded mel chunk maps to one datagen frame and one VideoWriter frame; both loops were checked from the exact pinned inference.py AST",
            "expected_count_is_source_derived_not_observed": True,
            "future_binding_key": "sample_id + natural_pcm.pcm_sha256 + sample_count + fps_num/fps_den",
            "future_duration_contract": "bind per-sample expected_inference_frame_count to the frozen same-length candidate generator PCM; transform contract preserves sample_count",
            "not_duration_rounded": True,
        },
        "clips": clip_results,
        "source_and_model_filesystem_mutation_by_probe": False,
    })


def run_status(result_path: Path) -> dict[str, Any]:
    receipt, raw_sha = read_sealed_json(result_path, "cross_tfg_dev20_w2l_native_frame_binding_v1",
                                        "PASS_CPU_NATIVE_FRONTEND_AND_EXPECTED_FRAME_COUNT_BINDING")
    return {"schema": "cross_tfg_dev20_w2l_native_frame_binding_status_v1",
            "status": "PASS_READONLY_STATUS", "receipt_path": str(result_path),
            "receipt_bytes": result_path.stat().st_size, "receipt_raw_sha256": raw_sha,
            "receipt_sha256": receipt["receipt_sha256"], "clips": len(receipt.get("clips", [])),
            "torch_imported": receipt.get("scope", {}).get("torch_imported"),
            "cuda_visible_devices": receipt.get("scope", {}).get("cuda_visible_devices")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "probe", "status"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "status":
        if args.result is None:
            parser.error("--result is required in status mode")
        print(json.dumps(run_status(args.result), sort_keys=True, separators=(",", ":")))
        return 0
    if args.output is None:
        parser.error("--output is required in preflight/probe mode")
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite frame-count receipt: {output}")
    if args.mode == "preflight":
        result = run_static_preflight(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        if args.preflight is None:
            parser.error("--preflight is required in probe mode")
        result = run_probe(json.loads(args.config.read_text(encoding="utf-8")), args.preflight)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "receipt_path": str(output), "receipt_bytes": len(raw),
                      "receipt_raw_sha256": sha256_bytes(raw),
                      "receipt_sha256": result["receipt_sha256"],
                      "clips": result.get("frozen_plan", {}).get("sample_count", len(result.get("clips", []))),
                      "torch_imported": False, "cuda_visible_devices": ""},
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
