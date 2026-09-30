"""Female AISHELL external replication of the frozen TTS source-axis protocol.

This is a separate 40-utterance/20-speaker primary-face producer. It reuses the
male CAL axes and evaluation helpers but writes an independently sealed 640-row
plan, so the original 80/40 dual-face evaluation contract is never patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, scoring, wav2lip

EXPECTED_SPEC_SHA256 = "fea242a042902ab0171fb649d93cd3b8a79615c7525287d47c848bb0eb030cef"
EXPECTED_INPUT_SHA256 = "199d6a2ebcdf54b6acd19f39e4604fd88cbb10c0e9d8ee4fcbaa173e5f9ba9c5"
EXPECTED_PATH_MAP_SHA256 = "e670abe8cd7ba19c372823ef7e8515df1167cc8fb3fdb8b8f93dcdbd3259d405"
EXPECTED_EMBEDDED_MANIFEST_FINGERPRINT = "349446ccd48ce3e5854cb7440dbd00f8b81a7d5eda224fb356c0d78af444c2b8"
EXPECTED_NATURAL_PAIR_IDENTITY_SHA256 = "f92ea432944426926ad6dfbe67ebee032a2995c7921340f9ba5ef873a60ac141"
EXPECTED_PAIRED_AUDIO_IDENTITY_SHA256 = "baf8064209cb4e3e50fbc243ff79d5b99161601ce94af2cafcbb5cae2efbb667"
EXPECTED_SOURCE_SELECTION_SHA256 = "4474a7523723e477205c1eaffded7461540f1d9477f15fde6249ef44f1867e56"
EXPECTED_FROZEN_TTS_SHA256 = "74da45be84d15eb7b7aab52f347c07a291c6d54a282302e48a09b8b037a686e1"
EXPECTED_CALIBRATION_SUMMARY_SHA256 = "34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a"
EXPECTED_CALIBRATION_AXES_SHA256 = "e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244"
EXPECTED_SHARD_SHAS = (
    "7a12a78d6fda004d72285a067693a0466ed60f7e7d492a80257989483bb8a0d3",
    "e8cf8e02fd016b57f5d1f1594b6ce92f12c2189d1e5bfe2fea3403df49b6a0bd",
)
FACE_ID = "primary"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _immutable_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    expected_sha = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        if _sha(path) != expected_sha:
            raise FileExistsError(f"immutable adapter metadata already exists with different bytes: {path}")
        return expected_sha
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return expected_sha


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flags() -> dict[str, bool]:
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _set_flags() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _instrument(events: list[dict[str, Any]], active: dict[str, str | None]):
    originals = []
    for module, name in (
        (wav2lip, "cache_face_features"),
        (wav2lip, "encode_latents"),
        (wav2lip, "decode_latents"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (scoring, "lip_embeddings_from_frames"),
        (evaluate, "_cross_distance_matrix"),
    ):
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapped(*args, __module=module.__name__, __name=name, __original=original, **kwargs):
            _set_flags()
            events.append({
                "operation": __name, "module": __module,
                "sample_id": active.get("sample_id"), "condition": active.get("condition"),
                "flags_at_call": _flags(), "time_monotonic": time.monotonic(),
            })
            return __original(*args, **kwargs)

        setattr(module, name, wrapped)
    return originals


def _prepare_rows(manifest_path: Path, input_root: Path, adapter_root: Path, shard_paths: tuple[Path, Path]):
    if _sha(manifest_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("female portable manifest raw SHA differs from its frozen input seal")
    path_map_path = manifest_path.with_name("portable_path_map.json")
    if _sha(path_map_path) != EXPECTED_PATH_MAP_SHA256:
        raise ValueError("female portable path-map SHA differs from its frozen input seal")
    shard_shas = tuple(_sha(path) for path in shard_paths)
    if shard_shas != EXPECTED_SHARD_SHAS:
        raise ValueError(f"female input shard SHA mismatch: {shard_shas}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != EXPECTED_EMBEDDED_MANIFEST_FINGERPRINT:
        raise ValueError("female manifest embedded fingerprint differs from its frozen input seal")
    if manifest.get("input_identity_sha256") != EXPECTED_NATURAL_PAIR_IDENTITY_SHA256:
        raise ValueError("female manifest ordered natural-pair identity differs from its frozen input seal")
    if manifest.get("source_selection_manifest_file_sha256") != EXPECTED_SOURCE_SELECTION_SHA256:
        raise ValueError("female source-selection manifest reference SHA differs from its frozen input seal")
    if manifest.get("tts_frozen_manifest_file_sha256") != EXPECTED_FROZEN_TTS_SHA256:
        raise ValueError("female frozen-TTS manifest reference SHA differs from its frozen input seal")
    if manifest.get("tts_metadata_model") != "qwen3-tts-vc-2026-01-22":
        raise ValueError("female portable manifest uses an unexpected Qwen model")
    records = manifest.get("records", [])
    if len(records) != 40 or any(row.get("split") != "evaluation" for row in records):
        raise ValueError(f"female replication requires exactly 40 evaluation rows, found {len(records)}")
    if len({row.get("id") for row in records}) != 40:
        raise ValueError("female input manifest contains duplicate utterance IDs")
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_speaker.setdefault(str(row["speaker"]), []).append(row)
        if row.get("speaker_metadata_sex") != "F":
            raise ValueError(f"speaker metadata is not F for {row['id']}")
    if len(by_speaker) != 20 or any(len(rows) != 2 for rows in by_speaker.values()):
        raise ValueError("female input manifest must contain two utterances for each of 20 speakers")

    natural_identity = [
        {"id": row["id"], "speaker": row["speaker"], "text": row["text"],
         "split": row["split"], "ref_sha256": row["natural_audio_sha256"]}
        for row in records
    ]
    if _canonical_hash(natural_identity) != EXPECTED_NATURAL_PAIR_IDENTITY_SHA256:
        raise ValueError("female ordered natural identity recomputation failed")
    pair_identity = [
        {"id": row["id"], "speaker": row["speaker"], "split": row["split"],
         "natural_sha256": row["natural_audio_sha256"],
         "tts_sha256": row["canonical_tts_16k_audio_sha256"]}
        for row in records
    ]
    paired_hash = _canonical_hash(pair_identity)
    if paired_hash != EXPECTED_PAIRED_AUDIO_IDENTITY_SHA256:
        raise ValueError("female ordered natural/TTS audio-pair identity recomputation failed")

    input_rows: list[dict[str, Any]] = []
    tts_rows: dict[str, dict[str, Any]] = {}
    adapter_map = []
    for row in records:
        sample_id, speaker = str(row["id"]), str(row["speaker"])
        natural_source = (input_root / row["natural_audio_path"]).resolve(strict=True)
        tts_source = (input_root / row["canonical_tts_16k_audio_path"]).resolve(strict=True)
        if _sha(natural_source) != row["natural_audio_sha256"]:
            raise ValueError(f"female natural WAV SHA mismatch for {sample_id}")
        if _sha(tts_source) != row["canonical_tts_16k_audio_sha256"]:
            raise ValueError(f"female canonical TTS WAV SHA mismatch for {sample_id}")
        natural_rate, natural_pcm = wavfile.read(str(natural_source))
        tts_rate, tts_pcm = wavfile.read(str(tts_source))
        if natural_rate != 16000 or natural_pcm.ndim != 1 or natural_pcm.dtype != np.int16:
            raise ValueError(f"female natural WAV must be mono 16k PCM16 for {sample_id}")
        if tts_rate != 16000 or tts_pcm.ndim != 1 or tts_pcm.dtype != np.int16:
            raise ValueError(f"female canonical TTS WAV must be mono 16k PCM16 for {sample_id}")
        if not (3.0 <= natural_pcm.size / natural_rate <= 10.0):
            raise ValueError(f"female natural duration is outside the frozen [3,10]s range for {sample_id}")
        natural_utterance = sample_id.split("_", 1)[1] + ".wav"
        natural_link = adapter_root / "audio" / speaker / natural_utterance
        tts_link = adapter_root / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
        natural_link.parent.mkdir(parents=True, exist_ok=True)
        tts_link.parent.mkdir(parents=True, exist_ok=True)
        for link, source in ((natural_link, natural_source), (tts_link, tts_source)):
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve(strict=True) != source:
                    raise FileExistsError(f"female input adapter path already has a different target: {link}")
            else:
                link.symlink_to(source)
        adapter_map.append({
            "sample_id": sample_id, "speaker_id": speaker,
            "natural_adapter_path": str(natural_link), "natural_source_path": str(natural_source),
            "natural_sha256": row["natural_audio_sha256"],
            "tts_adapter_path": str(tts_link), "tts_source_path": str(tts_source),
            "tts_sha256": row["canonical_tts_16k_audio_sha256"],
        })
        input_rows.append({
            "id": sample_id, "speaker": speaker, "split": "evaluation", "text": row["text"],
            "ref_sha256": row["natural_audio_sha256"],
        })
        tts_rows[sample_id] = {"qwen_canonical_16k": {"sha256": row["canonical_tts_16k_audio_sha256"]}}

    planned_rows = evaluate._plan_rows(input_rows, tts_rows, (FACE_ID,), adapter_root)
    if len(planned_rows) != 640:
        raise AssertionError(f"female sealed plan must contain 640 rows, found {len(planned_rows)}")
    plan_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in planned_rows:
        plan_by_sample.setdefault(row["sample_id"], []).append(row)
    if any(len(rows) != 16 for rows in plan_by_sample.values()):
        raise AssertionError("female row grid is not 16 conditions per sample")
    return manifest, records, input_rows, tts_rows, planned_rows, adapter_map, paired_hash, shard_shas


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("female_replication.py", "__init__.py", "evaluate.py", "ops.py", "scoring.py", "wav2lip.py")
    return {name: _sha(package / name) for name in names}


def _git_revision(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unavailable"


def _write_seal(args: argparse.Namespace, rows: list[dict[str, Any]], manifest: dict[str, Any],
                adapter_map: list[dict[str, Any]], adapter_manifest_sha: str,
                paired_hash: str, shard_shas: tuple[str, str],
                calibration: dict[str, Any], d: np.ndarray, u: np.ndarray) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    serialized_rows = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    row_bytes = serialized_rows.encode("utf-8")
    row_sha = hashlib.sha256(row_bytes).hexdigest()
    row_path = run_dir / "evaluation_rows.jsonl"
    row_path.write_bytes(row_bytes)
    code_hashes = _code_hashes()
    axes_path = Path(args.calibration_axes).resolve(strict=True)
    summary_path = Path(args.calibration_summary).resolve(strict=True)
    spec_path = Path(args.spec).resolve(strict=True)
    protocol = {
        "schema_version": 1,
        "classification": "frozen external female-speaker source-axis replication; row-level artifacts only; no cohort effect summary",
        "experiment_id": "tts-sex-replication-20261001",
        "declared_git_commit": "fcc4438e9eafa17c78dc49e205c418461bd0aa54",
        "spec": {"path": str(spec_path), "sha256": _sha(spec_path)},
        "cohort": {
            "selection": "first two lexicographic eligible utterances per F-labeled AISHELL speaker",
            "speaker_metadata_scope": "official publisher metadata label; not inferred from voice",
            "evaluation_rows": 40, "evaluation_speakers": 20, "two_clips_per_speaker": True,
            "ordered_pair_identity_sha256": EXPECTED_NATURAL_PAIR_IDENTITY_SHA256,
            "ordered_natural_tts_audio_pair_identity_sha256": paired_hash,
        },
        "female_input_manifest": {
            "path": str(Path(args.input_manifest).resolve()), "raw_sha256": _sha(args.input_manifest),
            "embedded_fingerprint": manifest["manifest_sha256"],
            "source_selection_manifest_raw_sha256": EXPECTED_SOURCE_SELECTION_SHA256,
            "frozen_tts_manifest_raw_sha256": EXPECTED_FROZEN_TTS_SHA256,
            "model": manifest["tts_metadata_model"],
            "path_map_path": str(Path(args.input_manifest).resolve().with_name("portable_path_map.json")),
            "path_map_raw_sha256": _sha(Path(args.input_manifest).resolve().with_name("portable_path_map.json")),
            "transport_shards": [
                {"path": str(path.resolve()), "sha256": sha}
                for path, sha in zip((Path(args.shard_one), Path(args.shard_two)), shard_shas, strict=True)
            ],
            "adapter_map_sha256": _canonical_hash(adapter_map),
            "adapter_map": adapter_map,
            "adapter_manifest_path": str(Path(args.input_adapter_root).resolve() / "female_adapter_manifest.json"),
            "adapter_manifest_sha256": adapter_manifest_sha,
        },
        "external_male_calibration": {
            "summary_path": str(summary_path), "summary_sha256": _sha(summary_path),
            "axes_path": str(axes_path), "axes_npz_sha256": _sha(axes_path),
            "d_source_sha256": ops.array_sha256(d), "u_level_sha256": ops.array_sha256(u),
            "rms_target": ops.TARGET_RMS,
            "anchor": calibration["global_anchor"],
            "direction_reused_without_female_refit": True,
        },
        "input_adapter_root": str(Path(args.input_adapter_root).resolve()),
        "face_image": {"id": FACE_ID, "path": str(Path(args.primary_face).resolve()), "sha256": _sha(args.primary_face)},
        "row_count": len(rows), "row_manifest_path": str(row_path), "row_manifest_sha256": row_sha,
        "score_grid": {
            "fourcell_rows_per_sample": 8,
            "source_axis_rows_per_sample": 8,
            "conditions": list(evaluate.SOURCE_AXIS_CONDITIONS),
            "level_cells": list(evaluate.LEVEL_CELLS),
            "face_ids": [FACE_ID], "main_support": "guard20",
            "support_views": ["full", "valid", "guard20", "guard15"],
            "anchor": {"k": 3, "column": 18, "official_syncnet_offset": -3},
        },
        "method": {
            "seed": 20260930, "rms_target": ops.TARGET_RMS,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "deterministic_algorithms": True, "reassert_after_face_detection": True,
            "mel_frontend": "official Wav2Lip mel chunks; librosa 0.10 compatibility shim; defaults unchanged",
            "syncnet_audio_units": "RAW PCM16 exact integer amplitude; FIXED float32 promoted to float64 and multiplied by 32768, no round/cast/clipping",
            "syncnet_visual_frontend": "official FFmpeg JPEG extraction and OpenCV BGR decode",
            "level_cells": {"q00": ["RAW", "RAW"], "q10": ["FIXED", "RAW"], "q01": ["RAW", "FIXED"], "q11": ["FIXED", "FIXED"]},
            "source_axis_audio": "unchanged RAW natural PCM and its full A row domain for all 8 axis conditions",
            "support_policy": "common recipient support within each N/T four-cell family and across source-axis conditions; empty guard20 is unavailable, row retained",
            "lag_domain": {"vshift": 15, "columns": 31, "k_by_column": "column - 15", "official_offset": "-k", "padding": "zero embedding outside real A domain"},
        },
        "production_code_sha256": code_hashes,
        "runtime_dependencies": {
            "wav2lip_root": str(Path(args.wav2lip_root).resolve()),
            "wav2lip_revision": _git_revision(Path(args.wav2lip_root).resolve()),
            "wav2lip_checkpoint": str(Path(args.wav2lip_checkpoint).resolve()),
            "wav2lip_checkpoint_sha256": _sha(args.wav2lip_checkpoint),
            "syncnet_root": str(Path(args.syncnet_root).resolve()),
            "syncnet_revision": _git_revision(Path(args.syncnet_root).resolve()),
            "syncnet_checkpoint": str(Path(args.syncnet_checkpoint).resolve()),
            "syncnet_checkpoint_sha256": _sha(args.syncnet_checkpoint),
            "ffmpeg": str(Path(args.ffmpeg).resolve()), "ffmpeg_sha256": _sha(args.ffmpeg),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "numpy": str(np.__version__), "opencv": str(cv2.__version__),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "unavailable",
        },
        "started_at_utc": _utc_now(),
    }
    _atomic_json(run_dir / "evaluation_protocol.json", protocol)
    _atomic_json(run_dir / "code_seal.json", {
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": row_sha, "production_code_sha256": code_hashes,
    })
    _atomic_json(run_dir / "evaluation_status.json", {"status": "sealed_pre_effects", "planned_rows": len(rows), "completed_clips": 0})
    return protocol


def _run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty female run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.input_manifest).resolve(strict=True)
    spec_path = Path(args.spec).resolve(strict=True)
    if _sha(spec_path) != EXPECTED_SPEC_SHA256:
        raise ValueError("female replication spec SHA differs from its frozen seal")
    if _sha(Path(args.calibration_summary).resolve(strict=True)) != EXPECTED_CALIBRATION_SUMMARY_SHA256:
        raise ValueError("male calibration summary SHA differs from its frozen seal")
    if _sha(Path(args.calibration_axes).resolve(strict=True)) != EXPECTED_CALIBRATION_AXES_SHA256:
        raise ValueError("male calibration axes SHA differs from their frozen seal")
    calibration = json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))
    if calibration.get("status") != "calibration_complete":
        raise ValueError("male external CAL summary is not complete")
    anchor = calibration.get("global_anchor", {})
    if (int(anchor.get("selected_k", 999)) != 3 or int(anchor.get("anchor_column", -1)) != 18
            or int(anchor.get("official_syncnet_offset", 999)) != -3):
        raise ValueError("female replication requires externally frozen male anchor k=3/col18/offset=-3")
    with np.load(args.calibration_axes, allow_pickle=False) as archive:
        d = np.asarray(archive["d_source"], dtype=np.float64)
        u = np.asarray(archive["u_level"], dtype=np.float64)
    if ops.array_sha256(d) != calibration["axes"]["d_sha256"] or ops.array_sha256(u) != calibration["axes"]["u_sha256"]:
        raise ValueError("external male axes do not match the sealed calibration summary")
    if not np.isclose(ops.TARGET_RMS, 0.0376838172675746, rtol=0.0, atol=0.0):
        raise ValueError(f"fixed RMS target differs from the frozen value: {ops.TARGET_RMS}")

    input_root = Path(args.input_root).resolve(strict=True)
    adapter_root = Path(args.input_adapter_root).resolve()
    manifest, records, eval_rows, tts_rows, rows, adapter_map, paired_hash, shard_shas = _prepare_rows(
        manifest_path, input_root, adapter_root, (Path(args.shard_one).resolve(strict=True), Path(args.shard_two).resolve(strict=True)),
    )
    adapter_map_path = adapter_root / "female_adapter_manifest.json"
    adapter_map_file_sha = _immutable_json(
        adapter_map_path, {"manifest_sha256": EXPECTED_INPUT_SHA256, "adapter_map": adapter_map},
    )
    protocol = _write_seal(
        args, rows, manifest, adapter_map, adapter_map_file_sha,
        paired_hash, shard_shas, calibration, d, u,
    )
    _atomic_json(run_dir / "attempts.json", {"attempts": [{"attempt": 1, "status": "running", "started_at_utc": protocol["started_at_utc"], "code_sha256": protocol["production_code_sha256"]}]})

    torch.manual_seed(20260930)
    np.random.seed(20260930)
    _set_flags()
    frontend = wav2lip.install_librosa_compatibility_shim()
    device = "cuda"
    model, model_meta = wav2lip.load_model(Path(args.wav2lip_root).resolve(), args.wav2lip_checkpoint, device=device)
    syncnet, syncnet_meta = scoring.load_syncnet(Path(args.syncnet_root).resolve(), args.syncnet_checkpoint, device=device)
    face_path = Path(args.primary_face).resolve(strict=True)
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"cannot read primary face image: {face_path}")
    flags_before_detection = _flags()
    face_meta = wav2lip.detect_static_face(frame, Path(args.wav2lip_root).resolve(), device=device)
    flags_after_detection = _flags()
    _set_flags()
    flags_after_override = _flags()
    face_meta.update({"path": str(face_path), "sha256": _sha(face_path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    _atomic_json(run_dir / "faces_primary.json", {
        "face_id": FACE_ID, "face": face_meta,
        "flags_before_detection": flags_before_detection,
        "flags_after_detection": flags_after_detection,
        "flags_after_explicit_override": flags_after_override,
        "post_detection_false_verified": flags_after_override == {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True},
        "wav2lip_checkpoint_sha256": _sha(args.wav2lip_checkpoint),
        "production_code_sha256": protocol["production_code_sha256"],
        "frontend_compatibility": frontend,
    })
    _atomic_json(run_dir / "faces" / "primary_runtime.json", {
        "face_id": FACE_ID, "device": torch.cuda.get_device_name(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda), "numpy": str(np.__version__), "opencv": str(cv2.__version__),
        "wav2lip_model": model_meta, "syncnet_model": syncnet_meta,
        "frontend_compatibility": frontend,
        "wav2lip_batch_size": args.wav2lip_batch_size, "syncnet_batch_size": args.syncnet_batch_size,
    })

    eval_by_id = {row["id"]: row for row in eval_rows}
    record_by_id = {row["id"]: row for row in records}
    plan_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        plan_by_sample.setdefault(row["sample_id"], []).append(row)
    face_cache_batch_sizes = range(1, args.wav2lip_batch_size + 1)
    events: list[dict[str, Any]] = []
    active: dict[str, str | None] = {"sample_id": None, "condition": None}
    originals = _instrument(events, active)
    results_path = run_dir / "evaluation_results.jsonl"
    attempts_path = run_dir / "evaluation_attempts.jsonl"
    forward_events_path = run_dir / "forward_events.jsonl"
    completed = 0
    failed = 0
    failure_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        _set_flags()
        active["condition"] = "shared_face_cache"
        face_cache = wav2lip.cache_face_features(
            model, frame, face_meta["generation_box_xyxy"], face_cache_batch_sizes, device=device,
        )
        first_id = eval_rows[0]["id"]
        first_record = record_by_id[first_id]
        first_natural = adapter_root / "audio" / first_record["speaker"] / (first_id.split("_", 1)[1] + ".wav")
        _, first_waveform, _ = evaluate._read_pcm16(first_natural)
        _, first_chunks, _, _ = wav2lip.mel_chunks(first_waveform, Path(args.wav2lip_root).resolve())
        active.update({"sample_id": first_id, "condition": "direct_decode_parity"})
        _set_flags()
        parity = wav2lip.parity_check_batches(
            model, first_chunks, frame, face_meta["generation_box_xyxy"], face_cache,
            batch_size=args.wav2lip_batch_size, device=device,
        )
        parity.update({"sample_id": first_id, "face_id": FACE_ID, "source_level": "natural_raw", "scope": "head and tail batches only; not all frames"})
        if not parity["uint8_prediction_exact"]:
            raise AssertionError(f"female direct decoder/cache parity failed: {parity}")
        _atomic_json(run_dir / "faces" / "primary_direct_decode_parity.json", parity)
        for event in events:
            _append_jsonl(forward_events_path, event)
        for row in eval_rows:
            sample_id = row["id"]
            speaker = row["speaker"]
            sample_started = time.perf_counter()
            active.update({"sample_id": sample_id, "condition": "context_encode"})
            first_event = len(events)
            planned = plan_by_sample[sample_id]
            row_results: list[dict[str, Any]] = []
            try:
                record = record_by_id[sample_id]
                natural_path = Path(planned[0]["natural_input_path"])
                tts_path = Path(planned[0]["tts_input_path"])
                context_npz, context_json = run_dir / "samples" / f"{sample_id}_context.npz", run_dir / "samples" / f"{sample_id}_context.json"
                context_row = {**row, "tts_hash": record["canonical_tts_16k_audio_sha256"]}
                arrays, context_meta = evaluate._make_context(
                    context_row, natural_path, tts_path, model=model, syncnet=syncnet,
                    wav2lip_root=Path(args.wav2lip_root).resolve(), d=d, u=u,
                    wav2lip_batch_size=args.wav2lip_batch_size,
                    syncnet_batch_size=args.syncnet_batch_size, device=device,
                )
                evaluate._write_context(run_dir, sample_id, arrays, context_meta)
                lips: dict[str, np.ndarray] = {}
                visual_meta: dict[str, dict[str, Any]] = {}
                for visual_id, latent_key in evaluate._visual_specifications():
                    active["condition"] = visual_id
                    lips[visual_id], visual_meta[visual_id] = evaluate._load_or_make_visual(
                        run_dir, sample_id, FACE_ID, visual_id, arrays[latent_key], frame, face_meta,
                        model, face_cache, syncnet,
                        argparse.Namespace(ffmpeg=args.ffmpeg, wav2lip_batch_size=args.wav2lip_batch_size,
                                           syncnet_batch_size=args.syncnet_batch_size), device,
                    )
                fourcell_support: dict[str, int] = {}
                for source in ("natural", "tts"):
                    fourcell_support[source] = min(
                        len(lips[f"{source}_raw"]), len(lips[f"{source}_fixed"]),
                        len(arrays[evaluate._audio_key(source, "raw")]),
                        len(arrays[evaluate._audio_key(source, "fixed")]),
                    )
                axis_support = min(
                    *(len(lips["natural_fixed" if condition == "baseline" else f"axis_{condition}"]) for condition in evaluate.SOURCE_AXIS_CONDITIONS),
                    len(arrays[evaluate._audio_key("natural", "raw")]),
                )
                if min(fourcell_support.values()) < 1 or axis_support < 1:
                    raise ValueError(f"no nonempty frozen common support: fourcell={fourcell_support}, axis={axis_support}")
                pair_geometry: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

                def geometry(visual_id: str, audio_key: str):
                    key = (visual_id, audio_key)
                    if key not in pair_geometry:
                        lip, audio = lips[visual_id], arrays[audio_key]
                        cross_raw, zero_raw = evaluate._cross_distance_matrix(lip, audio, device)
                        cross_unit, zero_unit = evaluate._cross_distance_matrix(evaluate._row_unit(lip), evaluate._row_unit(audio), device)
                        pair_geometry[key] = cross_raw, zero_raw, cross_unit, zero_unit
                    return pair_geometry[key]

                for row_plan in planned:
                    support_n = fourcell_support[row_plan["source"]] if row_plan["family"] == "fourcell" else axis_support
                    visual_id = row_plan["visual_id"]
                    audio_key = row_plan["audio_key"]
                    active["condition"] = f"{row_plan['family']}:{row_plan['source']}:{row_plan['condition']}"
                    cross_raw, zero_raw, cross_unit, zero_unit = geometry(visual_id, audio_key)
                    audio_n = len(arrays[audio_key])
                    matrix_raw = evaluate._lag_matrix_from_cross(cross_raw, zero_raw, support_n, support_n, vshift=15)
                    matrix_unit = evaluate._lag_matrix_from_cross(cross_unit, zero_unit, support_n, support_n, vshift=15)
                    support = evaluate._support_indices(support_n)
                    score_path = run_dir / row_plan["score_npz"]
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        score_path,
                        distance_matrix_raw_float32=matrix_raw,
                        distance_matrix_unit_float32=matrix_unit,
                        cross_distance_raw_float32=cross_raw,
                        cross_distance_unit_float32=cross_unit,
                        cross_distance_to_zero_raw_float32=zero_raw,
                        cross_distance_to_zero_unit_float32=zero_unit,
                        common_support_indices_int32=np.arange(support_n, dtype=np.int32),
                        full_support_indices_int32=support["full"], valid_support_indices_int32=support["valid"],
                        guard20_indices_int32=support["guard20"], guard15_indices_int32=support["guard15"],
                        lag_columns_int32=np.arange(31, dtype=np.int32), lag_k_int32=np.arange(-15, 16, dtype=np.int32),
                    )
                    row_results.append({
                        **row_plan, "status": "scored", "recorded_at_utc": _utc_now(),
                        "attempt_id": "female_replication_attempt01",
                        "context_npz": str(context_npz.relative_to(run_dir)), "context_npz_sha256": _sha(context_npz),
                        "context_json": str(context_json.relative_to(run_dir)), "context_json_sha256": _sha(context_json),
                        "visual_npz": str((run_dir / row_plan["visual_npz"]).relative_to(run_dir)),
                        "visual_npz_sha256": visual_meta[visual_id]["feature_npz_sha256"],
                        "visual_metadata": visual_meta[visual_id],
                        "score_npz": str(score_path.relative_to(run_dir)), "score_npz_sha256": _sha(score_path),
                        "audio_embedding_key": audio_key, "audio_embedding_sha256": ops.array_sha256(arrays[audio_key]),
                        "lip_embedding_sha256": ops.array_sha256(lips[visual_id]),
                        "common_support_n": int(support_n), "audio_row_domain_n": int(audio_n),
                        "scoring_audio_domain_n": int(support_n), "visual_query_domain_n": int(len(lips[visual_id])),
                        "support_policy": "shared row indices within each female recipient/family; fixed endpoint guards retained",
                        "anchor": {"source": "male CAL q10", "column": 18, "k": 3, "official_offset_frames": -3},
                        "raw_geometry": evaluate._metric_supports(matrix_raw, 18),
                        "row_unit_geometry": evaluate._metric_supports(matrix_unit, 18),
                        "input_pcm_identity": {
                            "natural_raw_exact": True, "tts_raw_exact": True,
                            "natural_raw_units_sha256": ops.array_sha256(arrays["natural_audio_units_raw_float64"]),
                            "tts_raw_units_sha256": ops.array_sha256(arrays["tts_audio_units_raw_float64"]),
                        },
                        "axis_diagnostics": context_meta["axis_diagnostics"].get(row_plan["condition"]) if row_plan["family"] == "source_axis" else None,
                    })
                for result in row_results:
                    _append_jsonl(results_path, result)
                for event in events[first_event:]:
                    _append_jsonl(forward_events_path, event)
                completed += 1
                _append_jsonl(attempts_path, {
                    "attempt": 1, "sample_id": sample_id, "speaker_id": speaker, "status": "complete",
                    "score_rows_written": len(row_results), "elapsed_seconds": time.perf_counter() - sample_started,
                    "recorded_at_utc": _utc_now(),
                })
                _atomic_json(run_dir / "evaluation_status.json", {
                    "status": "running", "planned_clips": 40, "completed_clips": completed,
                    "failed_clips": failed, "planned_rows": 640, "completed_rows": completed * 16,
                    "last_sample_id": sample_id,
                })
                print(json.dumps({"event": "clip_complete", "sample_id": sample_id, "completed_clips": completed,
                                  "completed_rows": completed * 16, "elapsed_seconds": time.perf_counter() - sample_started}), flush=True)
            except Exception as exc:
                failed += 1
                error = {
                    "sample_id": sample_id, "speaker_id": speaker, "error_type": type(exc).__name__,
                    "error": str(exc)[:2000], "traceback_tail": traceback.format_exc()[-4000:],
                    "elapsed_seconds": time.perf_counter() - sample_started,
                }
                failure_rows.append(error)
                for row_plan in planned:
                    _append_jsonl(results_path, {**row_plan, "status": "failed", **error, "recorded_at_utc": _utc_now()})
                for event in events[first_event:]:
                    _append_jsonl(forward_events_path, event)
                _append_jsonl(attempts_path, {"attempt": 1, "status": "failed", **error, "recorded_at_utc": _utc_now()})
                _atomic_json(run_dir / "evaluation_status.json", {
                    "status": "running", "planned_clips": 40, "completed_clips": completed,
                    "failed_clips": failed, "planned_rows": 640,
                    "completed_rows": completed * 16, "failed_rows": failed * 16,
                    "last_sample_id": sample_id,
                })
                print(json.dumps({"event": "clip_failed", **error}, ensure_ascii=False), flush=True)
    finally:
        for module, name, original in originals:
            setattr(module, name, original)

    all_results = _read_jsonl(results_path)
    final = {
        "status": "complete" if failed == 0 and completed == 40 else "complete_with_failures",
        "classification": "sealed external female replication; row-level results only",
        "run_dir": str(run_dir), "face_id": FACE_ID,
        "planned_clips": 40, "completed_clips": completed, "failed_clips": failed,
        "planned_rows": 640,
        "completed_rows": sum(row.get("status") == "scored" for row in all_results),
        "failed_rows": sum(row.get("status") == "failed" for row in all_results),
        "elapsed_seconds": time.perf_counter() - started, "finished_at_utc": _utc_now(),
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": _sha(run_dir / "evaluation_rows.jsonl"),
        "production_code_sha256": protocol["production_code_sha256"],
        "face_flags": {"before_detection": flags_before_detection, "after_detection": flags_after_detection,
                       "after_explicit_override": flags_after_override},
        "actual_forward_flag_values": sorted({json.dumps(row["flags_at_call"], sort_keys=True) for row in events}),
        "failures": failure_rows,
    }
    _atomic_json(run_dir / "evaluation_summary_primary.json", final)
    _atomic_json(run_dir / "evaluation_status.json", final)
    _atomic_json(run_dir / "attempts.json", {"attempts": [{"attempt": 1, "status": final["status"],
                  "completed_clips": completed, "failed_clips": failed, "finished_at_utc": final["finished_at_utc"]}]})
    _atomic_json(run_dir / "all_files_sha256.json", {
        str(path.relative_to(run_dir)): _sha(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    })
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="extracted female archive tree containing audio/ and manifest/")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--shard-one", required=True)
    parser.add_argument("--shard-two", required=True)
    parser.add_argument("--input-adapter-root", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-axes", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    result = _run(build_parser().parse_args())
    print(json.dumps({"status": result["status"], "completed_rows": result["completed_rows"],
                      "failed_rows": result["failed_rows"], "run_dir": result["run_dir"]}, separators=(",", ":")))
