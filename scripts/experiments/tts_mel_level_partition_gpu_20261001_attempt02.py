"""GPU consumer for the frozen mel-interior/boundary partition arrays.

The producer in ``tts_mel_level_partition_20261001.py`` is the only mel
frontend. This consumer validates and reads its sealed chunks, renders four
fresh Wav2Lip videos per evaluation utterance, and scores them against one
shared RAW-natural SyncNet audio embedding cache.

Run ``preflight`` first. It writes its immutable plan and all operational
logs outside the initially empty artifact directory. ``run`` refuses an
output directory containing any files and refuses a plan whose source or
input seals have changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SPEC_SHA256 = "abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8"
CPU_EVAL_PLAN_SHA256 = "ab6d713246ed6a9f11ecfcea08be2aae791a23e240ee4ab5791e31745be17f66"
CPU_EVAL_ROWS_SHA256 = "208c4bff640c6d7e0db21a0f58695e8946690ea8a0b6f4030fe9ef467e88fca9"
CPU_EVAL_MANIFEST_SHA256 = "4f4b9bfb1b51cb0e75585126ba3bde1c87d40bb444ea0581dc9512e0568a3f9"
CPU_EVAL_LEDGER_SHA256 = "d9d52ced138ce13df0602b1a14ecf7d9646df41db80af1d86e80e1530a970172"
CPU_CAL_PLAN_SHA256 = "8dabc6068f7c1e2b6ae52907e990a54ac875e15b0097b13a1c77205814b4b403"
CPU_CAL_MANIFEST_SHA256 = "afc6e6c72a729d62944916644cacdf89e405a73f1e0f678026b090df9b032c4d"
CPU_CAL_QA_SHA256 = "47f2bbf7ceff904395389d2e16b43af77af36e9ff4765b7ab7f46d765a4d9318"
CPU_CAL_LEDGER_SHA256 = "02b8d55dc713a7977c6f0137a2df8f9711dbbf3321b3ba23f443e9ce28d96a8c"
INPUT_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
FACE_REGISTRY_SHA256 = "8ada80c9c7a3628edef7428e9f218b475839a8a4f03079dcd9794e45ad0c4bbb"
FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
FIXED_TARGET_RMS = 0.0376838172675746

CONDITIONS = ("q00", "q10", "q01", "q11")
SUPPORTS = ("full", "valid", "guard20", "guard15")
GEOMETRIES = ("raw", "unit")
VSHIFT = 15
ANCHOR_COLUMN = 18  # k=+3; official offset=-3
ZERO_COLUMN = 15    # k=0
EMBEDDING_DIM = 1024
EPS = 1e-6
IDENTITY_FIELDS = ("id", "speaker", "text", "split", "ref_sha256")


class RunnerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _payload_sha(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read JSON metadata: {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"metadata must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RunnerError(f"{label} file is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RunnerError(f"{label} SHA-256 mismatch at {path}: {actual} != {expected}")
    return actual


def _verify_ledger(root: Path, expected_raw_sha: str) -> dict[str, str]:
    ledger_path = root / "all_files_sha256.json"
    _require_sha(ledger_path, expected_raw_sha, "CPU all-files ledger")
    ledger = _read_json(ledger_path)
    if not ledger or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 for key, value in ledger.items()):
        raise RunnerError(f"malformed CPU all-files ledger: {ledger_path}")
    for relative, digest in ledger.items():
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents:
            raise RunnerError(f"CPU ledger path escapes its run root: {relative}")
        _require_sha(candidate, digest, f"CPU ledger entry {relative}")
    found = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger_path
    }
    if found != set(ledger):
        raise RunnerError(f"CPU ledger membership differs: missing={sorted(set(ledger)-found)[:3]}, extra={sorted(found-set(ledger))[:3]}")
    return ledger


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise RunnerError(f"cannot load sealed mel arrays {path}: {type(exc).__name__}: {exc}") from exc


def _expected_supports(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32),
        "guard15": np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32),
    }


def _verify_mel_artifact(root: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("split") != "evaluation" or artifact.get("source_condition") != "natural" or artifact.get("status") != "prepared":
        raise RunnerError(f"unexpected EVAL mel artifact status/condition for {artifact.get('id')}")
    npz_path = Path(str(artifact.get("mel_path", "")))
    if not npz_path.is_absolute():
        npz_path = root / npz_path
    _require_sha(npz_path, str(artifact.get("mel_file_sha256", "")), f"mel npz {artifact.get('id')}")
    arrays = _load_npz(npz_path)
    required = {"M0", "M1", "U", "S", "mel_starts", *(f"q{suffix}" for suffix in ("00", "10", "01", "11")), *(f"chunks_q{suffix}" for suffix in ("00", "10", "01", "11"))}
    if not required.issubset(arrays):
        raise RunnerError(f"mel npz lacks required arrays for {artifact.get('id')}: {sorted(required-set(arrays))}")
    array_meta = artifact.get("arrays")
    if not isinstance(array_meta, Mapping):
        raise RunnerError(f"array hash inventory is missing for {artifact.get('id')}")
    for name in required:
        value = arrays[name]
        expected = array_meta.get(name)
        if not isinstance(expected, Mapping):
            raise RunnerError(f"array metadata missing for {artifact.get('id')}:{name}")
        if list(value.shape) != expected.get("shape") or str(value.dtype) != expected.get("dtype"):
            raise RunnerError(f"array shape/dtype differs for {artifact.get('id')}:{name}")
        if _array_sha256(value) != expected.get("array_sha256"):
            raise RunnerError(f"array digest differs for {artifact.get('id')}:{name}")
    m0, m1 = arrays["M0"], arrays["M1"]
    u, s = arrays["U"], arrays["S"]
    if m0.dtype != np.float32 or m1.dtype != np.float32 or m0.shape != m1.shape or m0.ndim != 2 or m0.shape[0] != 80:
        raise RunnerError(f"mel endpoint contract differs for {artifact.get('id')}: {m0.shape}/{m0.dtype}, {m1.shape}/{m1.dtype}")
    expected_u = (m0 > np.float32(-4.0)) & (m0 < np.float32(4.0)) & (m1 > np.float32(-4.0)) & (m1 < np.float32(4.0))
    if u.dtype != np.bool_ or s.dtype != np.bool_ or not np.array_equal(u, expected_u) or not np.array_equal(s, ~expected_u):
        raise RunnerError(f"frozen U/S masks do not follow the sealed endpoint rule for {artifact.get('id')}")
    expected_cells = {
        "q00": m0,
        "q10": np.where(u, m1, m0).astype(np.float32, copy=False),
        "q01": np.where(u, m0, m1).astype(np.float32, copy=False),
        "q11": m1,
    }
    starts = arrays["mel_starts"]
    if starts.dtype != np.int32 or starts.ndim != 1 or starts.size == 0:
        raise RunnerError(f"mel_starts contract differs for {artifact.get('id')}")
    for condition, expected in expected_cells.items():
        actual = arrays[condition]
        if actual.dtype != np.float32 or not np.array_equal(actual, expected):
            raise RunnerError(f"four-cell endpoint/copy identity failed for {artifact.get('id')}:{condition}")
        expected_chunks = np.stack([actual[:, int(start):int(start)+16] for start in starts], axis=0).astype(np.float32, copy=False)
        chunks = arrays[f"chunks_{condition}"]
        if chunks.dtype != np.float32 or chunks.shape != expected_chunks.shape or not np.array_equal(chunks, expected_chunks):
            raise RunnerError(f"frozen chunk identity failed for {artifact.get('id')}:{condition}")
    if int(artifact.get("chunk_count", -1)) != len(starts):
        raise RunnerError(f"chunk count differs from sealed mel artifact for {artifact.get('id')}")
    fixed_path = Path(str(artifact.get("fixed_pcm_path", "")))
    if not fixed_path.is_file():
        raise RunnerError(f"sealed RMS-fixed float32 waveform is missing for {artifact.get('id')}: {fixed_path}")
    _require_sha(fixed_path, str(artifact.get("fixed_pcm_file_sha256", "")), f"RMS-fixed waveform {artifact.get('id')}")
    try:
        fixed_waveform = np.load(fixed_path, allow_pickle=False)
    except Exception as exc:
        raise RunnerError(f"cannot decode sealed RMS-fixed waveform for {artifact.get('id')}: {type(exc).__name__}") from exc
    if fixed_waveform.dtype != np.float32 or fixed_waveform.ndim != 1 or len(fixed_waveform) != int(artifact.get("sample_count", -1)) or not np.isfinite(fixed_waveform).all():
        raise RunnerError(f"sealed RMS-fixed waveform violates float32/clock contract for {artifact.get('id')}")
    fixed_pcm_array_sha = _array_sha256(fixed_waveform)
    if fixed_pcm_array_sha != artifact.get("fixed_pcm_array_sha256"):
        raise RunnerError(f"sealed RMS-fixed waveform array SHA differs for {artifact.get('id')}")
    gain_meta = artifact.get("gain")
    if not isinstance(gain_meta, Mapping) or gain_meta.get("target_rms") != FIXED_TARGET_RMS:
        raise RunnerError(f"sealed RMS-fixed waveform target differs for {artifact.get('id')}")
    source_path = Path(str(artifact.get("source_wav_path", "")))
    if not source_path.is_file():
        raise RunnerError(f"RAW natural PCM is missing for {artifact.get('id')}: {source_path}")
    _require_sha(source_path, str(artifact.get("source_wav_sha256", "")), f"RAW natural PCM {artifact.get('id')}")
    from scipy.io import wavfile

    rate, pcm = wavfile.read(str(source_path))
    if rate != 16_000 or pcm.ndim != 1 or pcm.dtype != np.int16 or len(pcm) != int(artifact.get("sample_count", -1)):
        raise RunnerError(f"RAW natural PCM decode clock/format differs for {artifact.get('id')}: {rate}/{pcm.shape}/{pcm.dtype}")
    pcm_sha = _array_sha256(np.asarray(pcm, dtype="<i2"))
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    waveform_sha = _array_sha256(waveform)
    if waveform_sha != artifact.get("source_waveform_array_sha256"):
        raise RunnerError(f"RAW natural float32 input differs from sealed CPU frontend input for {artifact.get('id')}")
    return {
        "id": str(artifact["id"]),
        "speaker": str(artifact["speaker"]),
        "split": "evaluation",
        "text": artifact.get("text"),
        "ref_sha256": artifact.get("ref_sha256"),
        "source_condition": "natural",
        "source_wav_path": str(source_path.resolve()),
        "source_wav_file_sha256": str(artifact["source_wav_sha256"]),
        "sample_count": int(artifact["sample_count"]),
        "sample_rate_hz": int(artifact["sample_rate_hz"]),
        "source_pcm_array_sha256": pcm_sha,
        "source_waveform_array_sha256": waveform_sha,
        "fixed_pcm_path": str(fixed_path.resolve()),
        "fixed_pcm_file_sha256": str(artifact["fixed_pcm_file_sha256"]),
        "fixed_pcm_array_sha256": fixed_pcm_array_sha,
        "fixed_pcm_dtype": str(fixed_waveform.dtype),
        "fixed_pcm_sample_count": int(fixed_waveform.size),
        "fixed_pcm_sample_rate_hz": 16_000,
        "fixed_pcm_target_rms": FIXED_TARGET_RMS,
        "U_count": int(np.count_nonzero(u)),
        "S_count": int(np.count_nonzero(s)),
        "mel_path": str(npz_path.resolve()),
        "mel_file_sha256": str(artifact["mel_file_sha256"]),
        "chunk_count": int(artifact["chunk_count"]),
        "mel_arrays": {
            name: {
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
                "array_sha256": _array_sha256(arrays[name]),
            }
            for name in ("M0", "M1", "U", "S", "mel_starts", *(f"q{suffix}" for suffix in ("00", "10", "01", "11")), *(f"chunks_q{suffix}" for suffix in ("00", "10", "01", "11")))
        },
    }


def _verify_calibration_root(root: Path) -> dict[str, Any]:
    plan_path = root / "plan.json"
    manifest_path = root / "run_manifest.json"
    qa_path = root / "calibration_qa.json"
    for path, digest, label in (
        (plan_path, CPU_CAL_PLAN_SHA256, "CAL plan"),
        (manifest_path, CPU_CAL_MANIFEST_SHA256, "CAL run manifest"),
        (qa_path, CPU_CAL_QA_SHA256, "CAL QA"),
    ):
        _require_sha(path, digest, label)
    ledger = _verify_ledger(root, CPU_CAL_LEDGER_SHA256)
    qa = _read_json(qa_path)
    manifest = _read_json(manifest_path)
    if qa.get("status") != "PASS" or qa.get("model_forward_performed") is not False or qa.get("effects_read") is not False:
        raise RunnerError("sealed CAL QA is not the expected CPU-only PASS")
    if manifest.get("input_identity_sha256") != INPUT_IDENTITY_SHA256:
        raise RunnerError("CAL run has a different canonical input identity")
    return {
        "root": str(root.resolve()),
        "plan_sha256": CPU_CAL_PLAN_SHA256,
        "run_manifest_sha256": CPU_CAL_MANIFEST_SHA256,
        "calibration_qa_sha256": CPU_CAL_QA_SHA256,
        "all_files_ledger_sha256": CPU_CAL_LEDGER_SHA256,
        "ledger_entries_verified": len(ledger),
    }


def _verify_evaluation_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan_path = root / "plan.json"
    rows_path = root / "rows.jsonl"
    manifest_path = root / "run_manifest.json"
    for path, digest, label in (
        (plan_path, CPU_EVAL_PLAN_SHA256, "EVAL plan"),
        (rows_path, CPU_EVAL_ROWS_SHA256, "EVAL rows"),
        (manifest_path, CPU_EVAL_MANIFEST_SHA256, "EVAL run manifest"),
    ):
        _require_sha(path, digest, label)
    ledger = _verify_ledger(root, CPU_EVAL_LEDGER_SHA256)
    plan, manifest = _read_json(plan_path), _read_json(manifest_path)
    if plan.get("spec", {}).get("sha256") != SPEC_SHA256:
        raise RunnerError("EVAL CPU plan is bound to a different frozen spec")
    if plan.get("input_identity_sha256") != INPUT_IDENTITY_SHA256 or manifest.get("input_identity_sha256") != INPUT_IDENTITY_SHA256:
        raise RunnerError("EVAL CPU plan/run identity differs from the frozen 100-row cohort")
    if plan.get("split") != "evaluation" or manifest.get("split") != "evaluation":
        raise RunnerError("EVAL CPU run split is not evaluation")
    if plan.get("model_forward_performed") is not False or plan.get("effects_read") is not False or manifest.get("model_forward_performed") is not False or manifest.get("effects_read") is not False:
        raise RunnerError("EVAL CPU mel run is not a pre-effect, no-forward source")
    payload_digest = _payload_sha(manifest)
    if manifest.get("payload_sha256") != payload_digest:
        raise RunnerError(f"EVAL run manifest payload fingerprint mismatch: {payload_digest}")
    raw_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"malformed EVAL row plan line {line_number}") from exc
            if not isinstance(value, dict):
                raise RunnerError(f"EVAL row plan line {line_number} is not an object")
            raw_rows.append(value)
    if len(raw_rows) != 80 or len(manifest.get("artifacts", [])) != 80:
        raise RunnerError("frozen EVAL CPU run must have exactly 80 natural rows")
    artifacts = {str(row.get("id")): row for row in manifest["artifacts"]}
    rows = []
    for ordered_index, row in enumerate(raw_rows):
        if int(row.get("ordered_index", -1)) != ordered_index:
            raise RunnerError(f"EVAL row ordering differs at index {ordered_index}")
        if row.get("source_condition") != "natural" or row.get("split") != "evaluation":
            raise RunnerError("frozen EVAL plan contains an unexpected source or split")
        artifact = artifacts.get(str(row.get("id")))
        if artifact is None or artifact.get("speaker") != row.get("speaker") or artifact.get("source_wav_sha256") != row.get("expected_wav_sha256"):
            raise RunnerError(f"EVAL row/artifact identity differs for {row.get('id')}")
        sample = _verify_mel_artifact(root, artifact)
        sample["text"] = row.get("text")
        sample["ref_sha256"] = row.get("ref_sha256")
        sample["ordered_index"] = ordered_index
        rows.append(sample)
    if len({row["id"] for row in rows}) != 80 or len({row["speaker"] for row in rows}) != 40:
        raise RunnerError("frozen EVAL cohort is not 80 unique utterances from 40 speakers")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["speaker"]] = counts.get(row["speaker"], 0) + 1
    if set(counts.values()) != {2}:
        raise RunnerError("frozen EVAL cohort does not have two clips per speaker")
    return plan, rows, {"root": str(root.resolve()), "plan_sha256": CPU_EVAL_PLAN_SHA256, "rows_sha256": CPU_EVAL_ROWS_SHA256, "run_manifest_sha256": CPU_EVAL_MANIFEST_SHA256, "all_files_ledger_sha256": CPU_EVAL_LEDGER_SHA256, "ledger_entries_verified": len(ledger)}


def _runtime_metadata() -> dict[str, Any]:
    import cv2
    import python_speech_features
    import python_speech_features.base as psf_base
    import python_speech_features.sigproc as psf_sigproc
    import torch

    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "opencv": cv2.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "forward_flags_before_override": _torch_flags(torch),
    }
    for package in ("python_speech_features", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-as-distribution"
    versions["python_speech_features_sources"] = {
        "package": {"path": str(Path(python_speech_features.__file__).resolve()), "sha256": _sha256_file(python_speech_features.__file__)},
        "base": {"path": str(Path(psf_base.__file__).resolve()), "sha256": _sha256_file(psf_base.__file__)},
        "sigproc": {"path": str(Path(psf_sigproc.__file__).resolve()), "sha256": _sha256_file(psf_sigproc.__file__)},
    }
    return versions


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _path_entry(root: Path, name: str) -> dict[str, str]:
    path = root / name
    if not path.is_file():
        raise RunnerError(f"required model source file is missing: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _actual_wav2lip_import_metadata(wav2lip_root: Path, portable_root: Path) -> dict[str, Any]:
    helper = _wav2lip_import(wav2lip_root, portable_root)
    audio, wav2lip_class = helper.import_wav2lip(wav2lip_root)
    import importlib

    models_package = importlib.import_module("models")
    model_module = importlib.import_module(wav2lip_class.__module__)
    hparams_module = importlib.import_module("hparams")
    module_map = {
        "audio": Path(audio.__file__).resolve(),
        "hparams": Path(hparams_module.__file__).resolve(),
        "models_package": Path(models_package.__file__).resolve(),
        "wav2lip_model": Path(model_module.__file__).resolve(),
    }
    root = wav2lip_root.resolve()
    if any(root not in item.parents for item in module_map.values()):
        raise RunnerError(f"Wav2Lip resolved imports outside the bound checkout: {module_map}")
    return {
        key: {"path": str(path), "sha256": _sha256_file(path)}
        for key, path in module_map.items()
    }


def _asset_registry(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = _read_json(path)
    if assets.get("schema_version") != 1:
        raise RunnerError("execution asset registry schema_version must be 1")
    if assets.get("model_forward_performed") is not False or assets.get("effects_read") is not False:
        raise RunnerError("execution asset registry must be a no-forward, no-effects snapshot")
    if assets.get("created_by_source_sha256") != _sha256_file(__file__):
        raise RunnerError("execution asset registry was generated by a different runner source SHA")
    face_registry = assets.get("face_registry")
    if not isinstance(face_registry, Mapping):
        raise RunnerError("execution asset registry has no face_registry object")
    source_path = Path(str(face_registry.get("source_path", "")))
    source_sha = str(face_registry.get("source_raw_sha256", ""))
    if source_sha != FACE_REGISTRY_SHA256:
        raise RunnerError("primary face metadata is not bound to norm08 face registry SHA")
    _require_sha(source_path, FACE_REGISTRY_SHA256, "norm08 primary-face registry")
    face = assets.get("primary_face")
    if not isinstance(face, Mapping):
        raise RunnerError("execution asset registry has no primary_face object")
    face_path = Path(str(face.get("path", "")))
    _require_sha(face_path, FACE_SHA256, "frozen primary face image")
    if face.get("sha256") != FACE_SHA256:
        raise RunnerError("primary face image SHA differs from the frozen image")
    expected_face = {
        "width": 512,
        "height": 512,
        "generation_box_xyxy": [168, 88, 381, 385],
        "score_box": {"box": [59, 16, 489, 446], "side": 430, "center": [274.505615234375, 231.59503936767578]},
        "s3fd_threshold": 0.9,
        "zero_padding": True,
    }
    for key, expected in expected_face.items():
        if face.get(key) != expected:
            raise RunnerError(f"primary face field differs from the norm08 frozen metadata: {key}")
    wav2lip = assets.get("wav2lip")
    syncnet = assets.get("syncnet")
    ffmpeg = assets.get("ffmpeg")
    for label, item in (("wav2lip", wav2lip), ("syncnet", syncnet), ("ffmpeg", ffmpeg)):
        if not isinstance(item, Mapping):
            raise RunnerError(f"execution asset registry has no {label} object")
    wroot, sroot = Path(str(wav2lip.get("root", ""))), Path(str(syncnet.get("root", "")))
    checkpoint = Path(str(wav2lip.get("checkpoint_path", "")))
    sync_checkpoint = Path(str(syncnet.get("checkpoint_path", "")))
    ffmpeg_path = Path(str(ffmpeg.get("path", "")))
    for file_path, item, label in ((checkpoint, wav2lip, "Wav2Lip checkpoint"), (sync_checkpoint, syncnet, "SyncNet checkpoint"), (ffmpeg_path, ffmpeg, "ffmpeg")):
        _require_sha(file_path, str(item.get("sha256", "")), label)
    source_files = {
        "wav2lip": {name: _path_entry(wroot, name) for name in ("audio.py", "hparams.py", "models/__init__.py", "models/wav2lip.py")},
        "syncnet": {name: _path_entry(sroot, name) for name in ("SyncNetInstance.py", "SyncNetModel.py")},
    }
    declared = assets.get("source_files")
    if not isinstance(declared, Mapping):
        raise RunnerError("execution asset registry must bind Wav2Lip and SyncNet source file hashes")
    for model_name in ("wav2lip", "syncnet"):
        for name, actual in source_files[model_name].items():
            expected = declared.get(model_name, {}).get(name) if isinstance(declared.get(model_name), Mapping) else None
            if not isinstance(expected, Mapping) or expected.get("sha256") != actual["sha256"]:
                raise RunnerError(f"model source SHA is not bound in execution asset registry: {model_name}/{name}")
    ffmpeg_version = _safe_executable_version(ffmpeg_path)
    if ffmpeg.get("version") is not None and ffmpeg.get("version") != ffmpeg_version:
        raise RunnerError("ffmpeg version differs from the execution asset registry")
    face_path = face_path.resolve()
    return assets, {
        "registry_path": str(path.resolve()),
        "registry_raw_sha256": _sha256_file(path),
        "face_registry_path": str(source_path.resolve()),
        "face_registry_raw_sha256": FACE_REGISTRY_SHA256,
        "primary_face": {**dict(face), "path": str(face_path)},
        "wav2lip": {**dict(wav2lip), "root": str(wroot.resolve()), "checkpoint_path": str(checkpoint.resolve())},
        "syncnet": {**dict(syncnet), "root": str(sroot.resolve()), "checkpoint_path": str(sync_checkpoint.resolve())},
        "ffmpeg": {**dict(ffmpeg), "path": str(ffmpeg_path.resolve())},
        "source_files": source_files,
        "actual_wav2lip_imports": assets.get("actual_wav2lip_imports"),
        "ffmpeg_version": ffmpeg_version,
    }


def _safe_executable_version(executable: Path) -> str:
    try:
        result = subprocess.run([str(executable), "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError(f"cannot query executable version: {executable}: {type(exc).__name__}") from exc
    return result.stdout.decode("utf-8", errors="replace").splitlines()[0][:500]


def build_assets(args: argparse.Namespace) -> None:
    """Create a nonsecret, hash-bound runtime registry without model forwards."""

    face_registry = args.face_registry.resolve(strict=True)
    face_image = args.face_image.resolve(strict=True)
    wav2lip_root = args.wav2lip_root.resolve(strict=True)
    syncnet_root = args.syncnet_root.resolve(strict=True)
    wav2lip_checkpoint = args.wav2lip_checkpoint.resolve(strict=True)
    syncnet_checkpoint = args.syncnet_checkpoint.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    output = args.out.resolve()
    if output.exists():
        raise RunnerError(f"asset registry output already exists; use a new preflight namespace: {output}")
    _require_sha(face_registry, FACE_REGISTRY_SHA256, "norm08 primary-face registry")
    _require_sha(face_image, FACE_SHA256, "frozen primary face image")
    import cv2

    frame = cv2.imread(str(face_image), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (512, 512, 3):
        raise RunnerError("frozen primary face image does not decode to the norm08 512x512 color frame")
    source_files = {
        "wav2lip": {name: _path_entry(wav2lip_root, name) for name in ("audio.py", "hparams.py", "models/__init__.py", "models/wav2lip.py")},
        "syncnet": {name: _path_entry(syncnet_root, name) for name in ("SyncNetInstance.py", "SyncNetModel.py")},
    }
    actual_imports = _actual_wav2lip_import_metadata(wav2lip_root, args.portable_code_root.resolve(strict=True))
    expected_imports = {
        "audio": source_files["wav2lip"]["audio.py"]["sha256"],
        "hparams": source_files["wav2lip"]["hparams.py"]["sha256"],
        "models_package": source_files["wav2lip"]["models/__init__.py"]["sha256"],
        "wav2lip_model": source_files["wav2lip"]["models/wav2lip.py"]["sha256"],
    }
    if {key: item["sha256"] for key, item in actual_imports.items()} != expected_imports:
        raise RunnerError("actual imported Wav2Lip modules do not match the declared checkout layout")
    registry = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "created_by_source_sha256": _sha256_file(__file__),
        "face_registry": {"source_path": str(face_registry), "source_raw_sha256": FACE_REGISTRY_SHA256},
        "primary_face": {
            "path": str(face_image),
            "sha256": FACE_SHA256,
            "width": 512,
            "height": 512,
            "decoded_bgr_pixel_sha256": hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
            "generation_box_xyxy": [168, 88, 381, 385],
            "score_box": {"box": [59, 16, 489, 446], "side": 430, "center": [274.505615234375, 231.59503936767578]},
            "s3fd_threshold": 0.9,
            "zero_padding": True,
        },
        "wav2lip": {
            "root": str(wav2lip_root), "checkpoint_path": str(wav2lip_checkpoint), "sha256": _sha256_file(wav2lip_checkpoint),
        },
        "syncnet": {
            "root": str(syncnet_root), "checkpoint_path": str(syncnet_checkpoint), "sha256": _sha256_file(syncnet_checkpoint),
        },
        "ffmpeg": {"path": str(ffmpeg), "sha256": _sha256_file(ffmpeg), "version": _safe_executable_version(ffmpeg)},
        "source_files": source_files,
        "actual_wav2lip_imports": actual_imports,
        "model_forward_performed": False,
        "effects_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, registry)
    print(json.dumps({"status": "PASS", "registry_path": str(output), "registry_raw_sha256": _sha256_file(output),
                      "model_forward_performed": False, "effects_read": False}, ensure_ascii=False, sort_keys=True), flush=True)


def preflight(args: argparse.Namespace) -> None:
    eval_root = args.evaluation_run.resolve(strict=True)
    cal_root = args.calibration_run.resolve(strict=True)
    outdir = args.outdir.resolve()
    control = args.control_dir.resolve()
    if outdir == control or outdir in control.parents or control in outdir.parents:
        raise RunnerError("control directory and artifact output directory must be separate trees")
    if outdir.exists() and any(outdir.iterdir()):
        raise RunnerError(f"fresh artifact output directory must be absent or empty: {outdir}")
    if control.exists() and any(control.iterdir()):
        raise RunnerError(f"fresh control directory must be absent or empty: {control}")
    _require_sha(args.spec.resolve(strict=True), SPEC_SHA256, "frozen experiment spec")
    cal_binding = _verify_calibration_root(cal_root)
    eval_cpu_plan, rows, eval_binding = _verify_evaluation_root(eval_root)
    assets, normalized_assets = _asset_registry(args.execution_assets.resolve(strict=True))
    portable_root = args.portable_code_root.resolve(strict=True)
    helper_path = portable_root / "portable_source_axis_20260930" / "wav2lip.py"
    ops_path = portable_root / "portable_source_axis_20260930" / "ops.py"
    if not helper_path.is_file() or not ops_path.is_file():
        raise RunnerError(f"portable Wav2Lip helper package is missing below {portable_root}")
    if int(args.wav2lip_batch_size) != 64:
        raise RunnerError("frozen mel runner requires Wav2Lip batch size 64")
    if int(args.syncnet_batch_size) != 20:
        raise RunnerError("frozen SyncNet batch size is 20")
    actual_wav2lip_imports = _actual_wav2lip_import_metadata(Path(normalized_assets["wav2lip"]["root"]), portable_root)
    expected_wav2lip_imports = {
        "audio": normalized_assets["source_files"]["wav2lip"]["audio.py"]["sha256"],
        "hparams": normalized_assets["source_files"]["wav2lip"]["hparams.py"]["sha256"],
        "models_package": normalized_assets["source_files"]["wav2lip"]["models/__init__.py"]["sha256"],
        "wav2lip_model": normalized_assets["source_files"]["wav2lip"]["models/wav2lip.py"]["sha256"],
    }
    if {key: item["sha256"] for key, item in actual_wav2lip_imports.items()} != expected_wav2lip_imports:
        raise RunnerError("Python resolved Wav2Lip module files differ from the registry source bindings")
    if assets.get("actual_wav2lip_imports") != actual_wav2lip_imports:
        raise RunnerError("runtime Wav2Lip import paths/hashes differ from the asset registry")
    runtime = _runtime_metadata()
    if not runtime["cuda_available"]:
        raise RunnerError("preflight host has no CUDA runtime; GPU runner assets cannot be staged")
    ffmpeg_path = Path(normalized_assets["ffmpeg"]["path"])
    ffmpeg_version = normalized_assets["ffmpeg_version"]
    import cv2

    face_pixels = cv2.imread(normalized_assets["primary_face"]["path"], cv2.IMREAD_COLOR)
    if face_pixels is None or face_pixels.shape != (512, 512, 3):
        raise RunnerError("frozen primary face image does not decode to the norm08 512x512 color frame")
    face_pixel_sha = hashlib.sha256(np.ascontiguousarray(face_pixels).tobytes()).hexdigest()
    if normalized_assets["primary_face"].get("decoded_bgr_pixel_sha256") != face_pixel_sha:
        raise RunnerError("decoded primary-face pixel hash differs from the execution asset registry")
    plan = {
        "schema_version": 1,
        "experiment_id": "confirm-mel-level-partition-20261001",
        "consumer_id": "tts-mel-level-partition-gpu-20261001",
        "status": "frozen_pre_forward",
        "created_at_utc": _utc_now(),
        "spec": {"sha256": SPEC_SHA256, "path": str(args.spec.resolve(strict=True))},
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(__file__)},
        "cpu_calibration": cal_binding,
        "cpu_evaluation": eval_binding,
        "eval_cpu_plan_input_manifest_sha256": eval_cpu_plan.get("input_manifest", {}).get("raw_sha256"),
        "eval_cpu_tts_pair_manifest_sha256": eval_cpu_plan.get("tts_pair_manifest", {}).get("raw_sha256"),
        "input_identity_sha256": INPUT_IDENTITY_SHA256,
        "partition_definition": {
            "U": "(-4 < M0 < 4) and (-4 < M1 < 4)",
            "S": "logical complement of U",
            "q00": "exact saved M0 at U and S",
            "q10": "where(U, M1, M0)",
            "q01": "where(U, M0, M1)",
            "q11": "exact saved M1 at U and S",
            "source_arrays": "read from sealed CPU mel_partition.npz; no frontend recomputation",
        },
        "execution_assets": normalized_assets,
        "execution_assets_raw_sha256": normalized_assets["registry_raw_sha256"],
        "primary_face_decoded_pixel_sha256": face_pixel_sha,
        "portable_code_root": str(portable_root),
        "portable_helper_sha256": _sha256_file(helper_path),
        "portable_ops_sha256": _sha256_file(ops_path),
        "actual_wav2lip_imports": actual_wav2lip_imports,
        "runtime": runtime,
        "ffmpeg_version": ffmpeg_version,
        "generation": {
            "conditions": list(CONDITIONS),
            "video_container": "MP4",
            "video_codec": "libx264",
            "video_pixel_format": "yuv420p",
            "fps": 25,
            "generation_box_xyxy": normalized_assets["primary_face"]["generation_box_xyxy"],
            "batch_size": 64,
            "audio_encoder_latents_saved_for_all_four_cells": True,
            "face_skip_policy": "one 64-image frozen-face cache; slice cached skip tensors to actual tail-batch size; model.eval() makes each duplicate image independent",
            "forward_flag_policy": {
                "torch_use_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cudnn_allow_tf32": False,
                "matmul_allow_tf32": False,
                "torch_numpy_seed": 20260930,
                "cublas_workspace_config": ":4096:8",
            },
        },
        "syncnet": {
            "visual_decode": "ffmpeg image2pipe MJPEG decoded by OpenCV without temporary frame files",
            "audio_source": "same sealed RAW-natural PCM for all four cells; one forward_aud feature cache per clip",
            "batch_size": 20,
            "lag_columns": 31,
            "k_by_column": "column - 15",
            "official_offset_by_k": "-k",
            "fixed_anchor_column": ANCHOR_COLUMN,
            "fixed_anchor_k": 3,
            "audio_feature_padding": "pairwise distance to zero embedding outside real A rows",
            "support_policy": "shared query prefix n=min(A rows, every one of four V row counts); full and valid all n rows; guard20 [20,n-20); guard15 [15,n-15)",
            "visual_window_count": "decoded video frame count minus 5, matching the official helper path",
            "metric_definitions": {
                "C": "median of the 31-column lag curve minus minimum",
                "C_anchor": "median minus fixed column18 (k=+3; official offset=-3)",
                "B": "median of the 31-column lag curve",
                "D": "minimum of the 31-column lag curve",
                "search_gain": "D0-D; D0 is column15 (k=0)",
            },
        },
        "model_forward_performed": False,
        "effects_read": False,
        "planned_samples": rows,
        "planned_sample_count": len(rows),
        "planned_generation_count": len(rows) * len(CONDITIONS),
        "planned_score_families": len(rows) * len(CONDITIONS) * len(GEOMETRIES),
        "output_dir": str(outdir),
        "control_dir": str(control),
        "cli": {"wav2lip_batch_size": 64, "syncnet_batch_size": 20},
    }
    plan["payload_sha256"] = _payload_sha(plan)
    outdir.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    _atomic_json(control / "execution_plan.json", plan)
    (control / "execution_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")
    shutil.copyfile(Path(__file__).resolve(), control / "producer_snapshot.py")
    report = {
        "status": "PASS",
        "execution_plan_path": str((control / "execution_plan.json").resolve()),
        "execution_plan_raw_sha256": _sha256_file(control / "execution_plan.json"),
        "execution_plan_payload_sha256": plan["payload_sha256"],
        "code_sha256": plan["code"]["sha256"],
        "evaluation_samples": len(rows),
        "planned_cells": plan["planned_generation_count"],
        "output_directory_created_empty": True,
        "output_directory_contents": [],
        "assets_registry_sha256": normalized_assets["registry_raw_sha256"],
        "portable_helper_sha256": plan["portable_helper_sha256"],
        "portable_ops_sha256": plan["portable_ops_sha256"],
        "calibration_and_evaluation_cpu_seals_verified": True,
        "model_forward_performed": False,
        "effects_read": False,
    }
    _atomic_json(control / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


def _torch_flags(torch: Any) -> dict[str, Any]:
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _configure_inference_flags(torch: Any) -> dict[str, Any]:
    """Apply and return the fixed deterministic inference policy."""
    torch.manual_seed(20260930)
    np.random.seed(20260930)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    actual = _torch_flags(torch)
    expected = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    }
    if actual != expected:
        raise RunnerError(f"inference flags differ from the frozen deterministic policy: {actual}")
    return actual


def _record_event(events_path: Path, torch: Any, *, forward: str, **metadata: Any) -> dict[str, Any]:
    flags = _torch_flags(torch)
    expected = {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_allow_tf32": False, "matmul_allow_tf32": False}
    if flags != expected:
        raise RunnerError(f"forward {forward} attempted with flags outside the frozen deterministic policy: {flags}")
    event = {"event_id": time.time_ns(), "utc": _utc_now(), "forward": forward, "torch_flags_before": flags, **metadata}
    _jsonl(events_path, event)
    return event


def _wav2lip_import(root: Path, portable_root: Path):
    root_text = str(root.resolve())
    portable_text = str(portable_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if portable_text not in sys.path:
        sys.path.insert(0, portable_text)
    import importlib

    helper = importlib.import_module("portable_source_axis_20260930.wav2lip")
    return helper


def _read_raw_pcm(path: Path, expected_samples: int, expected_sha: str, expected_pcm_sha: str) -> np.ndarray:
    from scipy.io import wavfile

    if _sha256_file(path) != expected_sha:
        raise RunnerError(f"RAW A input SHA changed after preflight: {path}")
    rate, pcm = wavfile.read(str(path))
    if rate != 16_000 or pcm.ndim != 1 or pcm.dtype != np.int16 or len(pcm) != expected_samples:
        raise RunnerError(f"RAW A PCM violates the sealed clock/format: {path} rate={rate} shape={pcm.shape} dtype={pcm.dtype}")
    result = np.asarray(pcm, dtype=np.int16)
    actual_pcm_sha = _array_sha256(result.astype("<i2", copy=False))
    if actual_pcm_sha != expected_pcm_sha:
        raise RunnerError(f"RAW A decoded PCM array SHA changed after preflight: {path}")
    return result


def _video_chunk_latents(model: Any, chunks: np.ndarray, *, torch: Any, device: str, batch_size: int, events_path: Path, sample_id: str, condition: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    value = np.asarray(chunks)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[1:] != (80, 16) or not len(value) or not np.isfinite(value).all():
        raise RunnerError(f"invalid sealed mel chunks for {sample_id}/{condition}: {value.shape}/{value.dtype}")
    pieces: list[np.ndarray] = []
    event_rows: list[dict[str, Any]] = []
    model.audio_encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(value), batch_size):
            stop = min(start + batch_size, len(value))
            current = np.ascontiguousarray(value[start:stop])
            batch = torch.from_numpy(current[:, None, :, :]).to(device)
            event = _record_event(events_path, torch, forward="wav2lip_audio_encoder", sample_id=sample_id, condition=condition,
                                  batch_start=start, batch_stop=stop, input_shape=list(batch.shape), input_dtype=str(batch.dtype), device=device)
            output = model.audio_encoder(batch)
            if output.ndim != 4 or tuple(output.shape[1:]) != (512, 1, 1):
                raise RunnerError(f"unexpected Wav2Lip latent shape for {sample_id}/{condition}: {tuple(output.shape)}")
            pieces.append(output[:, :, 0, 0].detach().cpu().numpy().astype(np.float32, copy=True))
            event_rows.append(event)
    result = np.concatenate(pieces, axis=0)
    if not np.isfinite(result).all():
        raise RunnerError(f"nonfinite Wav2Lip latents for {sample_id}/{condition}")
    return result, event_rows


def _cache_face_skips(model: Any, helper: Any, frame: np.ndarray, generation_box: Sequence[int], *, torch: Any, device: str, events_path: Path, sample_id: str) -> list[Any]:
    count = 64
    tensor = helper.image_tensor_from_frame(frame, generation_box, count, device)
    x = tensor
    skips = []
    with torch.inference_mode():
        for block_index, block in enumerate(model.face_encoder_blocks):
            _record_event(events_path, torch, forward="wav2lip_face_encoder_block", sample_id=sample_id, cache_batch_size=count,
                          block_index=block_index, input_shape=list(x.shape), device=device)
            x = block(x)
            skips.append(x.detach())
    return skips


def _decode_to_video(model: Any, helper: Any, latents: np.ndarray, face_skips: Sequence[Any], frame: np.ndarray,
                     generation_box: Sequence[int], output_path: Path, ffmpeg: Path, *, torch: Any,
                     device: str, batch_size: int, events_path: Path, sample_id: str, condition: str) -> dict[str, Any]:
    import cv2

    encoder = helper._PipeEncoder(output_path, ffmpeg, int(frame.shape[1]), int(frame.shape[0]), codec="libx264")
    x1, y1, x2, y2 = [int(v) for v in generation_box]
    try:
        for start in range(0, len(latents), batch_size):
            stop = min(start + batch_size, len(latents))
            current = np.ascontiguousarray(latents[start:stop], dtype=np.float32)
            batch_n = len(current)
            event = _record_event(events_path, torch, forward="wav2lip_decoder", sample_id=sample_id, condition=condition,
                                  batch_start=start, batch_stop=stop, input_shape=[batch_n, 512], device=device)
            prediction = helper.decode_latents(model, current, [item[:batch_n] for item in face_skips], device=device)
            pixels = (prediction.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            for face_patch in pixels:
                full = frame.copy()
                resized = cv2.resize(face_patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                full[y1:y2, x1:x2] = resized
                encoder.write(full)
    except BaseException:
        if encoder.process.poll() is None:
            encoder.process.kill()
        encoder.process.wait()
        raise
    result = encoder.close()
    result["decoder_event_last"] = event if len(latents) else None
    result["batch_size"] = int(batch_size)
    return result


def _crop_video(helper: Any, video_path: Path, score_box: Mapping[str, Any], output_path: Path, ffmpeg: Path) -> dict[str, Any]:
    result = helper.crop_score_video(video_path, dict(score_box), output_path, ffmpeg)
    return result


class _SyncNet:
    def __init__(self, syncnet_root: Path, checkpoint: Path, *, device: str, batch_size: int, events_path: Path, ffmpeg: Path):
        import torch

        root_text = str(syncnet_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from SyncNetInstance import SyncNetInstance

        self.torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.events_path = events_path
        self.ffmpeg = ffmpeg
        self.model = SyncNetInstance(device=device)
        self.model.loadParameters(str(checkpoint))
        self.model.eval()
        self.network = self.model.__S__
        self.network.eval()
        self.checkpoint_sha256 = _sha256_file(checkpoint)
        self.loaded_state_sha256 = _state_dict_sha256(self.model)
        self.source_sha256 = {
            name: _sha256_file(syncnet_root / name)
            for name in ("SyncNetInstance.py", "SyncNetModel.py")
        }

    def audio(self, pcm: np.ndarray, *, sample_id: str) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
        try:
            import python_speech_features
        except ImportError as exc:
            raise RunnerError("python_speech_features is required by the frozen SyncNet audio frontend") from exc
        if pcm.dtype != np.int16 or pcm.ndim != 1 or len(pcm) < 640 * 6:
            raise RunnerError(f"RAW A PCM too short or invalid for SyncNet: {sample_id}")
        mfcc = np.asarray(python_speech_features.mfcc(pcm, 16_000), dtype=np.float64)
        if mfcc.ndim != 2 or mfcc.shape[1] != 13 or not np.isfinite(mfcc).all():
            raise RunnerError(f"invalid SyncNet MFCC shape for {sample_id}: {mfcc.shape}")
        count_clock = int(len(pcm) // 640) - 5
        count_mfcc = int((mfcc.shape[0] - 20) // 4 + 1)
        count = min(count_clock, count_mfcc)
        if count < 1:
            raise RunnerError(f"RAW A has no SyncNet windows for {sample_id}")
        windows = np.stack([mfcc[index * 4:index * 4 + 20].T for index in range(count)], axis=0).astype(np.float32)
        pieces = []
        events = []
        with self.torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                stop = min(start + self.batch_size, len(windows))
                batch = self.torch.from_numpy(windows[start:stop])[:, None, :, :].to(self.device)
                event = _record_event(self.events_path, self.torch, forward="syncnet_forward_aud", sample_id=sample_id,
                                      batch_start=start, batch_stop=stop, input_shape=list(batch.shape), device=self.device)
                pieces.append(self.network.forward_aud(batch).detach().cpu().numpy().astype(np.float32))
                events.append(event)
        features = np.concatenate(pieces, axis=0)
        if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM or not np.isfinite(features).all():
            raise RunnerError(f"invalid SyncNet A embedding shape for {sample_id}: {features.shape}")
        meta = {"sample_count": int(len(pcm)), "mfcc_shape": list(mfcc.shape), "feature_count": int(len(features)),
                "window_length_mfcc": 20, "window_stride_mfcc": 4, "sample_rate_hz": 16_000,
                "mfcc_source": "python_speech_features.mfcc default parameters", "source_pcm_array_sha256": _array_sha256(pcm.astype("<i2", copy=False))}
        return features, meta, events

    def _mjpeg_frames(self, video: Path) -> Iterable[np.ndarray]:
        import cv2

        command = [str(self.ffmpeg), "-y", "-loglevel", "error", "-i", str(video), "-threads", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RunnerError("ffmpeg SyncNet decoder did not expose pipes")
        buffer = bytearray()
        try:
            while True:
                chunk = process.stdout.read(1 << 20)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        del buffer[:-1]
                        break
                    if start:
                        del buffer[:start]
                    end = buffer.find(b"\xff\xd9", 2)
                    if end < 0:
                        break
                    jpeg = np.frombuffer(bytes(buffer[:end + 2]), dtype=np.uint8)
                    del buffer[:end + 2]
                    frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                    if frame is None:
                        process.kill()
                        raise RunnerError(f"ffmpeg emitted an invalid JPEG frame for {video}")
                    yield frame
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            rc = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        if rc != 0:
            raise RunnerError(f"ffmpeg SyncNet decode failed ({rc}) for {video}: {stderr[-800:]}")
        if buffer:
            raise RunnerError(f"ffmpeg returned an incomplete JPEG stream for {video}")

    def visual(self, video: Path, *, sample_id: str, condition: str) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
        import cv2

        outputs: list[np.ndarray] = []
        buffered: list[np.ndarray] = []
        frame_count = 0
        events: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for frame in self._mjpeg_frames(video):
                frame_count += 1
                buffered.append(frame)
                if len(buffered) >= self.batch_size + 5:
                    windows = np.stack([buffered[index:index+5] for index in range(self.batch_size)], axis=0)
                    tensor = self.torch.from_numpy(np.transpose(windows, (0, 4, 1, 2, 3)).astype(np.float32, copy=False)).to(self.device)
                    event = _record_event(self.events_path, self.torch, forward="syncnet_forward_lip", sample_id=sample_id, condition=condition,
                                          input_shape=list(tensor.shape), device=self.device, frame_start=frame_count-len(buffered))
                    outputs.append(self.network.forward_lip(tensor).detach().cpu().numpy().astype(np.float32))
                    events.append(event)
                    buffered = buffered[self.batch_size:]
            if len(buffered) >= 6:
                windows = np.stack([buffered[index:index+5] for index in range(len(buffered)-5)], axis=0)
                tensor = self.torch.from_numpy(np.transpose(windows, (0, 4, 1, 2, 3)).astype(np.float32, copy=False)).to(self.device)
                event = _record_event(self.events_path, self.torch, forward="syncnet_forward_lip", sample_id=sample_id, condition=condition,
                                      input_shape=list(tensor.shape), device=self.device, frame_start=frame_count-len(buffered))
                outputs.append(self.network.forward_lip(tensor).detach().cpu().numpy().astype(np.float32))
                events.append(event)
        if not outputs:
            raise RunnerError(f"SyncNet visual extraction yielded no windows: {video}")
        features = np.concatenate(outputs, axis=0)
        expected_windows = frame_count - 5
        if expected_windows < 1 or len(features) != expected_windows:
            raise RunnerError(f"SyncNet official visual window count mismatch for {video}: {len(features)} != {expected_windows}")
        if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM or not np.isfinite(features).all():
            raise RunnerError(f"invalid SyncNet V embedding shape for {video}: {features.shape}")
        metadata = {"frame_count": frame_count, "feature_count": int(len(features)), "window_length_frames": 5, "window_count_rule": "frame_count - 5",
                    "visual_decode": "ffmpeg image2pipe MJPEG + OpenCV", "temporary_frame_files_written": False}
        return features, metadata, events


def _row_unit(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (x / np.maximum(norms, np.float32(1e-12))).astype(np.float32, copy=False)


def _cross_distance(video: np.ndarray, audio: np.ndarray, torch: Any, *, chunk_rows: int = 8) -> np.ndarray:
    import torch.nn.functional as F

    v = np.asarray(video, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1] or not len(v) or not len(a):
        raise RunnerError(f"invalid full V×A embedding geometry {v.shape}/{a.shape}")
    result = np.empty((len(v), len(a)), dtype=np.float32)
    audio_tensor = torch.from_numpy(np.ascontiguousarray(a))
    with torch.inference_mode():
        for start in range(0, len(v), chunk_rows):
            stop = min(start + chunk_rows, len(v))
            batch = torch.from_numpy(np.ascontiguousarray(v[start:stop]))
            left = batch[:, None, :].expand(-1, len(a), -1).reshape(-1, v.shape[1])
            right = audio_tensor[None, :, :].expand(stop-start, -1, -1).reshape(-1, a.shape[1])
            distances = torch.nn.functional.pairwise_distance(left, right, p=2, eps=EPS)
            result[start:stop] = distances.reshape(stop-start, len(a)).cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RunnerError("nonfinite V×A distance matrix")
    return result


def _zero_distance(video: np.ndarray, torch: Any) -> np.ndarray:
    import torch.nn.functional as F

    value = torch.from_numpy(np.ascontiguousarray(np.asarray(video, dtype=np.float32)))
    with torch.inference_mode():
        result = F.pairwise_distance(value, torch.zeros_like(value), p=2, eps=EPS).cpu().numpy().astype(np.float32)
    return result


def _lag_distance(cross: np.ndarray, zero: np.ndarray, query_count: int, audio_count: int) -> np.ndarray:
    matrix = np.empty((query_count, 31), dtype=np.float32)
    for query in range(query_count):
        for column, lag in enumerate(range(-VSHIFT, VSHIFT + 1)):
            audio_index = query + lag
            matrix[query, column] = cross[query, audio_index] if 0 <= audio_index < audio_count else zero[query]
    return matrix


def _metric_payload(matrix: np.ndarray, supports: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in SUPPORTS:
        indices = supports[name]
        if not len(indices):
            result[name] = {"status": "too_short", "support_n": 0, "indices": []}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        min_index = int(np.argmin(curve))
        dmin = float(curve[min_index])
        d0 = float(curve[ZERO_COLUMN])
        median = float(np.median(curve))
        result[name] = {
            "status": "ok", "support_n": int(len(indices)), "indices": indices.tolist(), "curve": curve.tolist(),
            "C": median - dmin,
            "C_anchor": median - float(curve[ANCHOR_COLUMN]),
            "B": median,
            "D": dmin,
            "D0": d0,
            "search_gain": d0 - dmin,
            "D_anchor": float(curve[ANCHOR_COLUMN]),
            "best_lag": int(min_index - VSHIFT),
        }
    return result


def _save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value), allow_pickle=False)
    array = np.asarray(value)
    return {"path": str(path.resolve()), "file_sha256": _sha256_file(path), "array_sha256": _array_sha256(array), "shape": list(array.shape), "dtype": str(array.dtype)}


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return {"path": str(path.resolve()), "file_sha256": _sha256_file(path), "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "array_sha256": _array_sha256(np.asarray(value))} for key, value in values.items()}}


def _produce(args: argparse.Namespace) -> None:
    plan_path = args.plan.resolve(strict=True)
    plan = _read_json(plan_path)
    if plan.get("payload_sha256") != _payload_sha(plan):
        raise RunnerError("execution plan embedded payload fingerprint does not match")
    if plan.get("code", {}).get("sha256") != _sha256_file(__file__):
        raise RunnerError("runner source SHA differs from the pre-effect execution plan")
    outdir = Path(plan["output_dir"]).resolve()
    control = Path(plan["control_dir"]).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise RunnerError(f"artifact output directory is not empty: {outdir}")
    if control != plan_path.parent.resolve():
        raise RunnerError("plan must remain inside its frozen control directory")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RunnerError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set by the launcher before CUDA initialization")
    if (control / "execution_events.jsonl").exists() or (control / "execution_rows.jsonl").exists() is False:
        raise RunnerError("control directory is not a fresh preflight attempt")
    if _sha256_file(plan_path) != _sha256_file(control / "execution_plan.json"):
        raise RunnerError("execution plan changed after preflight")
    assets = plan["execution_assets"]
    runtime_now = _runtime_metadata()
    if runtime_now != plan.get("runtime"):
        raise RunnerError(f"runtime differs from preflight: {runtime_now}")
    if _sha256_file(assets["registry_path"]) != plan["execution_assets_raw_sha256"]:
        raise RunnerError("execution asset registry changed after preflight")
    _require_sha(Path(plan["spec"]["path"]), SPEC_SHA256, "frozen experiment spec")
    _verify_calibration_root(Path(plan["cpu_calibration"]["root"]))
    _, current_rows, _ = _verify_evaluation_root(Path(plan["cpu_evaluation"]["root"]))
    if [(row["id"], row["source_wav_file_sha256"], row["mel_file_sha256"]) for row in current_rows] != [
        (row["id"], row["source_wav_file_sha256"], row["mel_file_sha256"]) for row in plan["planned_samples"]
    ]:
        raise RunnerError("sealed EVAL source rows changed after preflight")
    helper_source = Path(plan["portable_code_root"]) / "portable_source_axis_20260930" / "wav2lip.py"
    ops_source = Path(plan["portable_code_root"]) / "portable_source_axis_20260930" / "ops.py"
    if _sha256_file(helper_source) != plan["portable_helper_sha256"] or _sha256_file(ops_source) != plan["portable_ops_sha256"]:
        raise RunnerError("portable model helper source changed after preflight")
    helper = _wav2lip_import(Path(assets["wav2lip"]["root"]), Path(plan["portable_code_root"]))
    actual_wav2lip_imports = _actual_wav2lip_import_metadata(Path(assets["wav2lip"]["root"]), Path(plan["portable_code_root"]))
    if actual_wav2lip_imports != plan["actual_wav2lip_imports"]:
        raise RunnerError("actual Wav2Lip import paths/hashes changed after preflight")
    import cv2
    import torch

    forward_flags = _configure_inference_flags(torch)
    face_path = Path(assets["primary_face"]["path"])
    frame = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (512, 512, 3):
        raise RunnerError(f"frozen primary face image failed decode/size check: {face_path}")
    image_digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    if image_digest != plan["primary_face_decoded_pixel_sha256"]:
        raise RunnerError("decoded frozen face pixels changed after preflight")
    model, model_info = helper.load_model(assets["wav2lip"]["root"], assets["wav2lip"]["checkpoint_path"], device="cuda")
    if model_info.get("checkpoint_sha256") != assets["wav2lip"].get("sha256"):
        raise RunnerError("loaded Wav2Lip checkpoint SHA differs from execution registry")
    sync = _SyncNet(Path(assets["syncnet"]["root"]), Path(assets["syncnet"]["checkpoint_path"]), device="cuda", batch_size=20,
                    events_path=control / "execution_events.jsonl", ffmpeg=Path(assets["ffmpeg"]["path"]))
    events_path = control / "execution_events.jsonl"
    output_rows_path = control / "execution_results.jsonl"
    if not events_path.exists():
        events_path.touch()
    if output_rows_path.exists():
        raise RunnerError("execution results already exist; use a new attempt namespace")
    runtime_record = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "runtime": runtime_now,
        "torch_flags_after_deterministic_override_before_forward": forward_flags,
        "deterministic_override": {"applied": True, "seed": 20260930, "cublas_workspace_config": ":4096:8"},
        "wav2lip": {**model_info, "source_files": plan["execution_assets"]["source_files"]["wav2lip"]},
        "syncnet": {
            "root": assets["syncnet"]["root"],
            "checkpoint_path": assets["syncnet"]["checkpoint_path"],
            "checkpoint_sha256": sync.checkpoint_sha256,
            "loaded_state_sha256": sync.loaded_state_sha256,
            "source_files": sync.source_sha256,
            "batch_size": 20,
            "device": "cuda",
        },
        "ffmpeg": {"path": assets["ffmpeg"]["path"], "sha256": assets["ffmpeg"]["sha256"], "version": plan["ffmpeg_version"]},
        "face_registry_raw_sha256": assets["face_registry_raw_sha256"],
        "primary_face_file_sha256": assets["primary_face"]["sha256"],
        "primary_face_decoded_pixel_sha256": image_digest,
        "effects_read": False,
        "model_forward_performed_at_creation": False,
    }
    _atomic_json(control / "execution_runtime.json", runtime_record)
    _record_event(events_path, torch, forward="wav2lip_face_encoder_cache", cache_shape=[64, 6, 96, 96], device="cuda",
                  flags_policy="cudnn_benchmark explicitly set false", primary_face_decoded_pixel_sha256=image_digest)
    shared_face_skips = _cache_face_skips(model, helper, frame, assets["primary_face"]["generation_box_xyxy"],
                                          torch=torch, device="cuda", events_path=events_path, sample_id="__shared_primary_face__")
    # One audio model call per clip; all four conditions share the exact cache.
    rows_completed = 0
    failures = 0
    for sample in plan["planned_samples"]:
        sample_id = sample["id"]
        row_result: dict[str, Any] = {"id": sample_id, "speaker": sample["speaker"], "split": "evaluation", "status": "running", "cells": {}, "model_forward_performed": True, "effects_read": False}
        try:
            pcm = _read_raw_pcm(Path(sample["source_wav_path"]), sample["sample_count"], sample["source_wav_file_sha256"], sample["source_pcm_array_sha256"])
            a_features, a_meta, a_events = sync.audio(pcm, sample_id=sample_id)
            audio_meta = _save_npy(outdir / "features" / sample_id / "A_raw.npy", a_features)
            audio_meta["forward_events"] = a_events
            audio_meta["feature_meta"] = a_meta
            row_result["shared_A"] = audio_meta
            _require_sha(Path(sample["mel_path"]), sample["mel_file_sha256"], f"mel NPZ {sample_id}")
            arrays = _load_npz(Path(sample["mel_path"]))
            cell_records: dict[str, dict[str, Any]] = {}
            v_features: dict[str, np.ndarray] = {}
            for condition in CONDITIONS:
                cell_result: dict[str, Any] = {"condition": condition, "status": "running"}
                row_result["cells"][condition] = cell_result
                try:
                    chunks = arrays[f"chunks_{condition}"]
                    if _array_sha256(chunks) != sample["mel_arrays"][f"chunks_{condition}"]["array_sha256"]:
                        raise RunnerError(f"sealed chunks changed after preflight: {sample_id}/{condition}")
                    latents, encoder_events = _video_chunk_latents(model, chunks, torch=torch, device="cuda", batch_size=64,
                                                                   events_path=events_path, sample_id=sample_id, condition=condition)
                    z_record = _save_npy(outdir / "latents" / sample_id / f"{condition}.npy", latents)
                    z_record["input_chunk_array_sha256"] = _array_sha256(chunks)
                    z_record["input_chunk_file_sha256"] = sample["mel_file_sha256"]
                    z_record["encoder_forward_events"] = encoder_events
                    video_path = outdir / "videos" / sample_id / f"{condition}.mp4"
                    video_meta = _decode_to_video(model, helper, latents, shared_face_skips, frame,
                                                  assets["primary_face"]["generation_box_xyxy"], video_path,
                                                  Path(assets["ffmpeg"]["path"]), torch=torch, device="cuda", batch_size=64,
                                                  events_path=events_path, sample_id=sample_id, condition=condition)
                    crop_path = outdir / "scored_video" / sample_id / f"{condition}.mkv"
                    crop_meta = _crop_video(helper, video_path, assets["primary_face"]["score_box"], crop_path, Path(assets["ffmpeg"]["path"]))
                    v, v_meta, v_events = sync.visual(crop_path, sample_id=sample_id, condition=condition)
                    v_record = _save_npy(outdir / "features" / sample_id / f"V_{condition}.npy", v)
                    v_record["feature_meta"] = v_meta
                    v_record["forward_events"] = v_events
                    v_features[condition] = v
                    cell_result.update({"status": "generated_scored", "latents": z_record, "video": video_meta, "scored_video": crop_meta, "V": v_record})
                    cell_records[condition] = cell_result
                except Exception as exc:
                    cell_result.update({"status": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc)[:2000]})
                    failures += 1
            if len(v_features) == len(CONDITIONS):
                row_result["scores"] = _score_clip(outdir, sample_id, v_features, a_features, torch)
                row_result["status"] = "complete" if all(row_result["cells"][condition]["status"] == "generated_scored" for condition in CONDITIONS) else "complete_with_cell_failures"
            else:
                row_result["status"] = "incomplete_cells"
            rows_completed += 1
        except Exception as exc:
            row_result["status"] = "failed"
            row_result["failure_type"] = type(exc).__name__
            row_result["failure_message"] = str(exc)[:2000]
            failures += 1
        _jsonl(output_rows_path, row_result)
        print(json.dumps({"id": sample_id, "status": row_result["status"], "completed_rows": rows_completed, "planned_rows": 80}, ensure_ascii=False, sort_keys=True), flush=True)
    _finalize(plan, outdir, control, rows_completed, failures)


def _score_clip(outdir: Path, sample_id: str, v_features: Mapping[str, np.ndarray], a_features: np.ndarray, torch: Any) -> dict[str, Any]:
    common_n = min(len(a_features), *(len(v_features[name]) for name in CONDITIONS))
    if common_n < 1:
        raise RunnerError(f"no common V/A support for {sample_id}")
    supports = _expected_supports(common_n)
    result: dict[str, Any] = {"common_support_n": int(common_n), "A_count": int(len(a_features)), "V_counts": {name: int(len(v_features[name])) for name in CONDITIONS}, "geometries": {}}
    for geometry in GEOMETRIES:
        a = a_features if geometry == "raw" else _row_unit(a_features)
        result["geometries"][geometry] = {}
        for condition in CONDITIONS:
            v_full = v_features[condition]
            v = v_full if geometry == "raw" else _row_unit(v_full)
            cross = _cross_distance(v, a, torch)
            zero = _zero_distance(v, torch)
            lag = _lag_distance(cross, zero, common_n, len(a))
            dist_path = outdir / "distances" / sample_id / f"{condition}_{geometry}.npz"
            dist_meta = _save_npz(dist_path, {
                "cross_v_to_a_float32": cross,
                "zero_boundary_float32": zero,
                "lag_distance_float32": lag,
                **{f"support_{name}_int32": indices for name, indices in supports.items()},
                "lag_columns_int32": np.arange(31, dtype=np.int32),
                "lag_k_int32": np.arange(-15, 16, dtype=np.int32),
            })
            result["geometries"][geometry][condition] = {"distance_artifact": dist_meta, "metrics": _metric_payload(lag, supports)}
    return result


def _finalize(plan: Mapping[str, Any], outdir: Path, control: Path, rows_completed: int, failures: int) -> None:
    results_path = control / "execution_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 80:
        raise RunnerError(f"execution results must retain all 80 intended rows, got {len(rows)}")
    ledger: dict[str, str] = {}
    for path in sorted(item for item in outdir.rglob("*") if item.is_file()):
        ledger[path.relative_to(outdir).as_posix()] = _sha256_file(path)
    manifest = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "consumer_id": plan["consumer_id"],
        "status": "complete" if all(row.get("status") == "complete" for row in rows) else "complete_with_failures",
        "finished_at_utc": _utc_now(),
        "execution_plan_path": str((control / "execution_plan.json").resolve()),
        "execution_plan_raw_sha256": _sha256_file(control / "execution_plan.json"),
        "execution_plan_payload_sha256": plan["payload_sha256"],
        "runner_code_sha256": plan["code"]["sha256"],
        "execution_assets_raw_sha256": plan["execution_assets_raw_sha256"],
        "execution_runtime_path": str((control / "execution_runtime.json").resolve()),
        "execution_runtime_raw_sha256": _sha256_file(control / "execution_runtime.json"),
        "cpu_calibration": plan["cpu_calibration"],
        "cpu_evaluation": plan["cpu_evaluation"],
        "input_identity_sha256": plan["input_identity_sha256"],
        "planned_rows": 80,
        "completed_rows": rows_completed,
        "planned_cells": 320,
        "failed_cells_or_rows": failures,
        "rows_path": str(results_path.resolve()),
        "rows_raw_sha256": _sha256_file(results_path),
        "execution_events_path": str((control / "execution_events.jsonl").resolve()),
        "execution_events_raw_sha256": _sha256_file(control / "execution_events.jsonl"),
        "execution_event_count": sum(1 for line in (control / "execution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()),
        "execution_runtime_sha256": _sha256_file(control / "execution_runtime.json"),
        "producer_snapshot_sha256": _sha256_file(control / "producer_snapshot.py"),
        "preflight_report_sha256": _sha256_file(control / "preflight_report.json"),
        "artifact_root": str(outdir.resolve()),
        "artifact_count": len(ledger),
        "artifact_ledger": ledger,
        "model_forward_performed": True,
        "mel_frontend_recomputed": False,
        "effects_read": False,
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    _atomic_json(control / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "rows": rows_completed, "cells": 320, "failures": failures,
                      "run_manifest": str((control / "run_manifest.json").resolve()), "run_manifest_sha256": _sha256_file(control / "run_manifest.json"),
                      "artifact_count": len(ledger)}, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="verify sealed CPU arrays/assets and freeze a no-forward plan")
    pre.add_argument("--calibration-run", required=True, type=Path)
    pre.add_argument("--evaluation-run", required=True, type=Path)
    pre.add_argument("--execution-assets", required=True, type=Path)
    pre.add_argument("--portable-code-root", required=True, type=Path, help="parent directory of portable_source_axis_20260930 package")
    pre.add_argument("--spec", required=True, type=Path)
    pre.add_argument("--outdir", required=True, type=Path)
    pre.add_argument("--control-dir", required=True, type=Path)
    pre.add_argument("--wav2lip-batch-size", type=int, default=64)
    pre.add_argument("--syncnet-batch-size", type=int, default=20)
    pre.set_defaults(func=preflight)
    assets = sub.add_parser("build-assets", help="hash the frozen model/face runtime into a nonsecret registry; no model forward")
    assets.add_argument("--face-registry", required=True, type=Path)
    assets.add_argument("--face-image", required=True, type=Path)
    assets.add_argument("--wav2lip-root", required=True, type=Path)
    assets.add_argument("--wav2lip-checkpoint", required=True, type=Path)
    assets.add_argument("--syncnet-root", required=True, type=Path)
    assets.add_argument("--syncnet-checkpoint", required=True, type=Path)
    assets.add_argument("--ffmpeg", required=True, type=Path)
    assets.add_argument("--portable-code-root", required=True, type=Path)
    assets.add_argument("--out", required=True, type=Path)
    assets.set_defaults(func=build_assets)
    run = sub.add_parser("run", help="produce the frozen 80x4 videos and shared-A SyncNet scores")
    run.add_argument("--plan", required=True, type=Path)
    run.set_defaults(func=_produce)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
