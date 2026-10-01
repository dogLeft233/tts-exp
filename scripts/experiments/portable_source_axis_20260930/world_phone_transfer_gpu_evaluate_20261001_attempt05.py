"""Seal and score the frozen WORLD phone-transfer evaluation grid.

This runner consumes the completed CPU-only physical waveform producer and
creates Wav2Lip videos, official SyncNet embeddings, unique full cross-distance
matrices, and every frozen four-cell alias. It does not calculate cohort
effects. Run ``preflight`` before ``run``; each run uses a fresh output path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, scoring, wav2lip

SPEC_SHA256 = "f085e2915349523891a478e3fc4dd1dada8493d68d3d73608bcc042d4a4e62ca"
EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
ALIGNMENT_OVERLAY_SHA256 = "4822cf2dbaa1733c93447ef92446cc28cb5381f6c3c3d8d783cca5c477251b7e"
ALIASES_SHA256 = "d14260e47869003756a0efd6e84719011c0a2550b3af07ead5f62b902fb22f3f"
PHYSICAL_MANIFEST_SHA256 = "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3"
PHYSICAL_MANIFEST_FINGERPRINT_SHA256 = "eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa7ed9de8"
PHYSICAL_PLAN_SHA256 = "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e"
CAL_QA_FREEZE_SHA256 = "0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d"
FACE_REGISTRY_SHA256 = "8ada80c9c7a3628edef7428e9f218b475839a8a4f03079dcd9794e45ad0c4bbb"
INPUT_MANIFEST_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
TTS_PAIR_MANIFEST_SHA256 = "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2"
TTS_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
CALIBRATION_SUMMARY_SHA256 = "34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a"
CALIBRATION_AXES_SHA256 = "e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244"
PSF_SOURCE_TREE_SHA256 = "a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17"
PSF_PARENT_PATH = "/root/tts-research-system-20261001/deps/python_speech_features_only_attempt05"
WORLD_ARMS = ("RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
FAMILIES = ("raw", "within_world")
CELLS = ("q00", "q10", "q01", "q11")
SUPPORT_NAMES = ("full", "valid", "guard20", "guard15")
VSHIFT = 15
ANCHOR = {"k": 3, "column": 18, "official_offset": -3}
FIXED_RMS = 0.0376838172675746
EXPECTED_FORWARD_FLAGS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
}


SHA256_CONSTANTS = {
    name: value for name, value in globals().copy().items()
    if name.endswith("_SHA256") and isinstance(value, str)
}
INVALID_SHA256_CONSTANTS = {
    name: value for name, value in SHA256_CONSTANTS.items()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
}
if INVALID_SHA256_CONSTANTS:
    raise ValueError(f"invalid SHA-256 constants: {INVALID_SHA256_CONSTANTS}")


class RunnerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected JSON object: {path}")
    return value


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RunnerError(f"missing {label}: {path}")
    actual = _sha(path)
    if actual != expected:
        raise RunnerError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def _python_source_tree(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = []
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*.py")
        if candidate.is_file() and "__pycache__" not in candidate.parts
    ):
        relative = path.relative_to(root).as_posix()
        file_sha = _sha(path)
        size = path.stat().st_size
        files.append({"relative_path": relative, "sha256": file_sha, "size_bytes": size})
        digest.update(f"{relative}\0{file_sha}\0{size}\n".encode("utf-8"))
    return {"path": str(root), "sha256": digest.hexdigest(), "file_count": len(files), "files": files}


def _execution_assets(args: argparse.Namespace) -> dict[str, Any]:
    import python_speech_features

    wav2lip_root = Path(args.wav2lip_root).resolve(strict=True)
    syncnet_root = Path(args.syncnet_root).resolve(strict=True)
    wav2lip_checkpoint = Path(args.wav2lip_checkpoint).resolve(strict=True)
    syncnet_checkpoint = Path(args.syncnet_checkpoint).resolve(strict=True)
    ffmpeg_path = Path(args.ffmpeg).resolve(strict=True)
    face_path = Path(args.primary_face).resolve(strict=True)
    for path, label in (
        (wav2lip_checkpoint, "Wav2Lip checkpoint"),
        (syncnet_checkpoint, "SyncNet checkpoint"),
        (ffmpeg_path, "ffmpeg executable"),
        (face_path, "primary face image"),
    ):
        if not path.is_file():
            raise RunnerError(f"missing {label}: {path}")
    if not os.access(ffmpeg_path, os.X_OK):
        raise RunnerError(f"ffmpeg is not executable: {ffmpeg_path}")
    face_image = cv2.imread(str(face_path))
    if face_image is None or tuple(face_image.shape[:2]) != (512, 512):
        raise RunnerError(f"primary face image must decode as frozen 512x512 pixels: {face_path}")
    psf_package = Path(python_speech_features.__file__).resolve().parent
    psf_tree = _python_source_tree(psf_package)
    if psf_package.parent != Path(PSF_PARENT_PATH).resolve():
        raise RunnerError(f"python_speech_features is not imported from the validated private PSF parent: {psf_package.parent}")
    if psf_tree["sha256"] != PSF_SOURCE_TREE_SHA256 or psf_tree["file_count"] != 3:
        raise RunnerError(f"isolated python_speech_features source tree differs from the frozen runtime: {psf_tree['sha256']}/{psf_tree['file_count']}")
    wav2lip_tree = _python_source_tree(wav2lip_root)
    syncnet_tree = _python_source_tree(syncnet_root)
    if not wav2lip_tree["file_count"] or not syncnet_tree["file_count"]:
        raise RunnerError("Wav2Lip or SyncNet checkout has no Python source files")
    try:
        ffmpeg_first_line = subprocess.run(
            [str(ffmpeg_path), "-version"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise RunnerError(f"ffmpeg version probe failed: {exc}") from exc
    try:
        psf_distribution_version = importlib.metadata.version("python_speech_features")
        psf_distribution_metadata_available = True
    except importlib.metadata.PackageNotFoundError:
        # This deployment deliberately uses the three-file, hash-pinned pure
        # source package without copying dist-info into the model environment.
        psf_distribution_version = "unknown; source tree SHA-256 is pinned below"
        psf_distribution_metadata_available = False
    return {
        "python": {"executable": str(Path(sys.executable).resolve()), "version": platform.python_version(), "sha256": _sha(Path(sys.executable).resolve())},
        "runtime": {
            "torch": str(torch.__version__), "cuda_build": str(torch.version.cuda),
            "numpy": str(np.__version__), "scipy": importlib.metadata.version("scipy"),
            "opencv": str(cv2.__version__), "librosa": importlib.metadata.version("librosa"),
            "python_speech_features": psf_distribution_version,
        },
        "python_speech_features_source": {
            **psf_tree, "module_path": str(Path(python_speech_features.__file__).resolve()),
            "isolated_parent": str(psf_package.parent), "expected_source_tree_sha256": PSF_SOURCE_TREE_SHA256,
            "distribution_metadata_available": psf_distribution_metadata_available,
        },
        "wav2lip": {
            "root": str(wav2lip_root), "checkpoint": str(wav2lip_checkpoint),
            "checkpoint_sha256": _sha(wav2lip_checkpoint), "source_tree": wav2lip_tree,
        },
        "syncnet": {
            "root": str(syncnet_root), "checkpoint": str(syncnet_checkpoint),
            "checkpoint_sha256": _sha(syncnet_checkpoint), "source_tree": syncnet_tree,
        },
        "ffmpeg": {"path": str(ffmpeg_path), "sha256": _sha(ffmpeg_path), "version_first_line": ffmpeg_first_line},
        "face": {"path": str(face_path), "sha256": _sha(face_path), "width": int(face_image.shape[1]), "height": int(face_image.shape[0])},
        "preflight_gpu_policy": "preflight does not query CUDA availability or create a CUDA context",
    }


def _embedded_fingerprint(value: Mapping[str, Any], label: str) -> str:
    observed = value.get("manifest_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise RunnerError(f"{label} lacks a SHA-256 embedded fingerprint")
    computed = hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "manifest_sha256"})).hexdigest()
    if observed != computed:
        raise RunnerError(f"{label} embedded fingerprint mismatch: {observed} != {computed}")
    return observed


def _load_calibration(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary_path = Path(args.calibration_summary).resolve(strict=True)
    axes_path = Path(args.calibration_axes).resolve(strict=True)
    _require_sha(summary_path, CALIBRATION_SUMMARY_SHA256, "frozen CAL summary")
    _require_sha(axes_path, CALIBRATION_AXES_SHA256, "frozen CAL axes")
    summary = _json(summary_path)
    if summary.get("status") != "calibration_complete":
        raise RunnerError("frozen CAL is not complete")
    anchor = summary.get("global_anchor", {})
    observed_anchor = {
        "k": int(anchor.get("selected_k", 9999)),
        "column": int(anchor.get("anchor_column", -1)),
        "official_offset": int(anchor.get("official_syncnet_offset", 9999)),
    }
    if observed_anchor != ANCHOR:
        raise RunnerError(f"frozen CAL anchor mismatch: {observed_anchor}")
    if summary.get("axes", {}).get("npz_sha256") != CALIBRATION_AXES_SHA256:
        raise RunnerError("frozen CAL summary does not bind the expected axis archive")
    with np.load(axes_path, allow_pickle=False) as archive:
        d = np.asarray(archive["d_source"], dtype=np.float64)
        u = np.asarray(archive["u_level"], dtype=np.float64)
    if d.shape != (ops.LATENT_DIM,) or u.shape != (ops.LATENT_DIM,):
        raise RunnerError("frozen CAL d/u axes are not 512-vectors")
    if ops.array_sha256(d) != summary.get("axes", {}).get("d_sha256") or ops.array_sha256(u) != summary.get("axes", {}).get("u_sha256"):
        raise RunnerError("frozen CAL axis arrays do not match the summary")
    d_perp = d - (float(d @ u) / float(u @ u)) * u
    if not np.isfinite(d_perp).all() or float(d_perp @ d_perp) <= 1e-12:
        raise RunnerError("frozen CAL d component outside u is degenerate")
    metadata = {
        "summary": {"path": str(summary_path), "sha256": _sha(summary_path)},
        "axes": {"path": str(axes_path), "sha256": _sha(axes_path), "d_sha256": ops.array_sha256(d), "u_sha256": ops.array_sha256(u), "d_perp_sha256": ops.array_sha256(d_perp)},
        "anchor": observed_anchor,
    }
    return d, u, {**metadata, "d_perp": d_perp}


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
    observed = _flags()
    if observed != EXPECTED_FORWARD_FLAGS:
        raise RunnerError(f"deterministic forward flags did not hold: {observed}")


def _instrument(
    events: list[dict[str, Any]], active: dict[str, str | None],
    model: Any, syncnet: Any, forward_guard_counts: dict[str, int],
):
    originals = []
    targets = (
        (wav2lip, "cache_face_features"),
        (wav2lip, "encode_latents"),
        (wav2lip, "decode_latents"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (scoring, "lip_embeddings_from_frames"),
        (evaluate, "_cross_distance_matrix"),
    )
    for module, name in targets:
        original = getattr(module, name)
        originals.append((module, name, original, "set"))

        def wrapped(*args, __module=module, __name=name, __original=original, **kwargs):
            _set_flags()
            observed_flags = _flags()
            if observed_flags != EXPECTED_FORWARD_FLAGS:
                raise RunnerError(f"forward flag assertion failed before {__name}: {observed_flags}")
            event = {
                "operation": __name,
                "sample_id": active.get("sample_id"),
                "direction": active.get("direction"),
                "stream_id": active.get("stream_id"),
                "visual_id": active.get("visual_id"),
                "audio_id": active.get("audio_id"),
                "flags_at_call": observed_flags,
                "time_monotonic": time.monotonic(),
            }
            events.append(event)
            if active.get("event_log"):
                _append_jsonl(Path(str(active["event_log"])), event)
            return __original(*args, **kwargs)

        setattr(module, name, wrapped)
    forward_targets = [
        (model.audio_encoder, "forward", "wav2lip.audio_encoder.forward"),
        *[(block, "forward", f"wav2lip.face_encoder_blocks[{index}].forward") for index, block in enumerate(model.face_encoder_blocks)],
        *[(block, "forward", f"wav2lip.face_decoder_blocks[{index}].forward") for index, block in enumerate(model.face_decoder_blocks)],
        (model.output_block, "forward", "wav2lip.output_block.forward"),
        (syncnet.__S__, "forward_aud", "syncnet.forward_aud"),
        (syncnet.__S__, "forward_lip", "syncnet.forward_lip"),
    ]
    for owner, name, label in forward_targets:
        original = getattr(owner, name)
        had_instance_override = name in getattr(owner, "__dict__", {})

        def guarded_forward(*args, __original=original, __label=label, **kwargs):
            _set_flags()
            observed_flags = _flags()
            if observed_flags != EXPECTED_FORWARD_FLAGS:
                raise RunnerError(f"forward flag assertion failed before {__label}: {observed_flags}")
            forward_guard_counts[__label] = forward_guard_counts.get(__label, 0) + 1
            return __original(*args, **kwargs)

        setattr(owner, name, guarded_forward)
        originals.append((owner, name, original, "set" if had_instance_override else "delete"))
    return originals


def _restore_instrument(originals: list[tuple[Any, str, Any, str]]) -> None:
    for owner, name, original, restore_mode in originals:
        if restore_mode == "delete":
            delattr(owner, name)
        else:
            setattr(owner, name, original)


def _read_audio(path: Path, expected_sha: str, expected_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if _sha(path) != expected_sha:
        raise RunnerError(f"audio file SHA mismatch: {path}")
    sample_rate, data = wavfile.read(str(path))
    if sample_rate != ops.SAMPLE_RATE or data.ndim != 1 or data.size == 0:
        raise RunnerError(f"expected mono 16-kHz waveform: {path}: {sample_rate}/{data.shape}/{data.dtype}")
    if data.dtype == np.int16:
        pcm16 = data.copy()
        waveform = (pcm16.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
        units = pcm16.astype(np.float64)
        if not np.array_equal(waveform.astype(np.float64) * np.float64(32768.0), units):
            raise RunnerError(f"PCM16 float-to-MFCC unit identity failed: {path}")
        subtype = "PCM_16"
    elif data.dtype == np.float32:
        waveform = data.copy()
        if not np.isfinite(waveform).all():
            raise RunnerError(f"non-finite float PCM: {path}")
        units, _ = ops.pcm_to_syncnet_integer_units(waveform)
        pcm16 = None
        subtype = "FLOAT32"
    else:
        raise RunnerError(f"expected PCM16 RAW or FLOAT32 generated audio, got {data.dtype}: {path}")
    if expected_samples is not None and int(data.size) != int(expected_samples):
        raise RunnerError(f"waveform length differs from producer manifest: {path}: {data.size} != {expected_samples}")
    meta = {
        "path": str(path.resolve()), "file_sha256": expected_sha,
        "sample_rate_hz": int(sample_rate), "channels": 1,
        "samples": int(data.size), "duration_s": float(data.size / sample_rate),
        "file_dtype": str(data.dtype), "subtype": subtype,
        "float32_pcm_sha256": ops.array_sha256(waveform),
        "syncnet_units_dtype": str(units.dtype),
        "syncnet_units_sha256": ops.array_sha256(units),
        "peak_abs_float32": float(np.max(np.abs(waveform))),
        "peak_over_one": bool(np.max(np.abs(waveform)) > 1.0),
        "quantization": "none; PCM16 is promoted exactly; FLOAT32 is multiplied by 32768 in float64",
        "clipping": "none",
    }
    return waveform, units, {**meta, "pcm16": pcm16}


def _make_alias_rows(sample_id: str, speaker: str, ordered_index: int, direction: str, recipient: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for arm in WORLD_ARMS:
            cells = {
                "q00": ("V_RT", "A_RAW" if family == "raw" else "A_RT"),
                "q10": (f"V_{arm}", "A_RAW" if family == "raw" else "A_RT"),
                "q01": ("V_RT", f"A_{arm}"),
                "q11": (f"V_{arm}", f"A_{arm}"),
            }
            for cell in CELLS:
                visual_id, audio_id = cells[cell]
                matrix_id = f"{visual_id}__{audio_id}"
                rows.append({
                    "ordered_eval_index": ordered_index,
                    "row_id": f"{sample_id}__{direction}__{family}__{arm}__{cell}",
                    "sample_id": sample_id, "speaker_id": speaker, "split": "evaluation",
                    "family": family, "direction": direction,
                    "donor_condition": "tts" if direction == "tts_to_natural" else "natural",
                    "recipient_condition": recipient, "condition": arm, "cell": cell,
                    "face_id": "primary", "visual_id": visual_id, "audio_id": audio_id,
                    "matrix_id": matrix_id, "status": "planned",
                })
    bridge = {
        "q00": ("V_FIXED", "A_RAW"), "q10": ("V_RT", "A_RAW"),
        "q01": ("V_FIXED", "A_RT"), "q11": ("V_RT", "A_RT"),
    }
    for cell in CELLS:
        visual_id, audio_id = bridge[cell]
        matrix_id = f"{visual_id}__{audio_id}"
        rows.append({
            "ordered_eval_index": ordered_index,
            "row_id": f"{sample_id}__{direction}__bridge__{cell}",
            "sample_id": sample_id, "speaker_id": speaker, "split": "evaluation",
            "family": "bridge", "direction": direction,
            "donor_condition": "tts" if direction == "tts_to_natural" else "natural",
            "recipient_condition": recipient, "condition": "RT_bridge", "cell": cell,
            "face_id": "primary", "visual_id": visual_id, "audio_id": audio_id,
            "matrix_id": matrix_id, "status": "planned",
        })
    if len(rows) != 76 or len({row["matrix_id"] for row in rows}) != 36:
        raise AssertionError("frozen WORLD alias cardinality changed")
    return rows


def _validate_aliases() -> dict[str, Any]:
    examples = _make_alias_rows("S0000_TEST", "S0000", 0, "tts_to_natural", "natural")
    keys = sorted({row["matrix_id"] for row in examples})
    videos = sorted({row["visual_id"] for row in examples})
    audios = sorted({row["audio_id"] for row in examples})
    if (len(keys), len(videos), len(audios), len(examples)) != (36, 10, 10, 76):
        raise AssertionError("WORLD alias coverage does not match the frozen annex")
    return {"unique_matrix_count": len(keys), "unique_video_count": len(videos), "unique_audio_count": len(audios), "label_count": len(examples)}


def _load_frozen_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_manifest_path = Path(args.input_manifest).resolve(strict=True)
    tts_manifest_path = Path(args.tts_pair_manifest).resolve(strict=True)
    input_root = Path(args.input_root).resolve(strict=True)
    physical_manifest_path = Path(args.physical_manifest).resolve(strict=True)
    physical_plan_path = Path(args.physical_plan).resolve(strict=True)
    freeze_path = Path(args.cal_qa_freeze).resolve(strict=True)
    face_registry_path = Path(args.face_registry).resolve(strict=True)
    spec_path = Path(args.spec).resolve(strict=True)
    bindings_path = Path(args.execution_bindings).resolve(strict=True)
    overlay_path = Path(args.alignment_overlay).resolve(strict=True)
    aliases_path = Path(args.aliases).resolve(strict=True)
    calibration_summary_path = Path(args.calibration_summary).resolve(strict=True)
    calibration_axes_path = Path(args.calibration_axes).resolve(strict=True)

    fixed_paths = (
        (spec_path, SPEC_SHA256, "WORLD physical specification"),
        (bindings_path, EXECUTION_BINDINGS_SHA256, "WORLD execution bindings"),
        (overlay_path, ALIGNMENT_OVERLAY_SHA256, "physical alignment overlay"),
        (aliases_path, ALIASES_SHA256, "WORLD alias annex"),
        (physical_manifest_path, PHYSICAL_MANIFEST_SHA256, "physical EVAL manifest"),
        (physical_plan_path, PHYSICAL_PLAN_SHA256, "physical EVAL plan"),
        (freeze_path, CAL_QA_FREEZE_SHA256, "independent CAL QA freeze"),
        (face_registry_path, FACE_REGISTRY_SHA256, "frozen face registry"),
        (input_manifest_path, INPUT_MANIFEST_SHA256, "natural/TTS input manifest"),
        (tts_manifest_path, TTS_PAIR_MANIFEST_SHA256, "frozen Qwen pair manifest"),
        (calibration_summary_path, CALIBRATION_SUMMARY_SHA256, "frozen CAL summary"),
        (calibration_axes_path, CALIBRATION_AXES_SHA256, "frozen CAL axes"),
    )
    file_seals = {label: {"path": str(path), "sha256": _require_sha(path, expected, label)} for path, expected, label in fixed_paths}
    input_manifest = _json(input_manifest_path)
    tts_manifest = _json(tts_manifest_path)
    physical_manifest = _json(physical_manifest_path)
    physical_plan = _json(physical_plan_path)
    freeze = _json(freeze_path)
    face_registry = _json(face_registry_path)
    d_axis, u_axis, calibration = _load_calibration(args)
    execution_assets = _execution_assets(args)

    _embedded_fingerprint(input_manifest, "natural/TTS input manifest")
    if tts_manifest.get("input_identity_sha256") != TTS_IDENTITY_SHA256:
        raise RunnerError("frozen Qwen identity fingerprint mismatch")
    if tts_manifest.get("source_manifest", {}).get("raw_sha256") != file_seals["natural/TTS input manifest"]["sha256"]:
        raise RunnerError("Qwen pair manifest source SHA does not match selected input manifest")
    _embedded_fingerprint(tts_manifest, "Qwen pair manifest")
    if _embedded_fingerprint(physical_manifest, "physical EVAL manifest") != PHYSICAL_MANIFEST_FINGERPRINT_SHA256:
        raise RunnerError("physical manifest embedded fingerprint differs from the staged run")
    if physical_manifest.get("status") != "EVALUATION_WAVEFORMS_READY" or physical_manifest.get("split") != "evaluation":
        raise RunnerError("physical EVAL waveform manifest is not ready")
    _embedded_fingerprint(physical_plan, "physical EVAL plan")
    freeze_status = str(freeze.get("status", "")).upper()
    if freeze_status != "FROZEN":
        raise RunnerError(f"independent CAL QA freeze is not frozen: {freeze.get('status')}")
    if face_registry.get("face_id") != "primary" or face_registry.get("face", {}).get("sha256") != "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6":
        raise RunnerError("frozen primary face registry identity mismatch")
    if execution_assets["face"]["sha256"] != face_registry.get("face", {}).get("sha256"):
        raise RunnerError("primary face image bytes differ from the frozen registry")
    if face_registry.get("face", {}).get("generation_box_xyxy") != [168, 88, 381, 385]:
        raise RunnerError("primary face generation box differs from sealed registry")
    if face_registry.get("face", {}).get("score_box", {}).get("box") != [59, 16, 489, 446]:
        raise RunnerError("primary SyncNet score box differs from sealed registry")

    all_natural_rows = input_manifest.get("records", [])
    all_tts_rows = tts_manifest.get("records", [])
    natural_rows = [row for row in all_natural_rows if row.get("split") == "evaluation"]
    tts_rows = {row.get("id"): row for row in all_tts_rows if row.get("split") == "evaluation"}
    physical_rows = {row.get("id"): row for row in physical_manifest.get("records", []) if row.get("split") == "evaluation"}
    if len(natural_rows) != 80 or len(tts_rows) != 80 or len(physical_rows) != 80:
        raise RunnerError(f"cohort count mismatch: input={len(natural_rows)} tts={len(tts_rows)} physical={len(physical_rows)}")
    if len({row.get("speaker") for row in natural_rows}) != 40:
        raise RunnerError("natural input cohort is not 40 speakers")

    identity_fields = tuple(tts_manifest.get("input_identity_fields", ("id", "speaker", "text", "split", "ref_sha256")))
    all_tts_by_id = {row.get("id"): row for row in all_tts_rows}
    if len(all_natural_rows) != 100 or len(all_tts_rows) != 100 or len(all_tts_by_id) != 100:
        raise RunnerError(f"frozen source/TTS identity cohort mismatch: input={len(all_natural_rows)} tts={len(all_tts_rows)} unique_tts={len(all_tts_by_id)}")
    if set(row.get("id") for row in all_natural_rows) != set(all_tts_by_id):
        raise RunnerError("frozen source and TTS identity ID sets differ")
    ordered_identities = []
    for source_row in all_natural_rows:
        identity = {field: source_row.get(field) for field in identity_fields}
        if identity != all_tts_by_id[source_row["id"]].get("input_identity"):
            raise RunnerError(f"source and TTS frozen identity mismatch: {source_row['id']}")
        ordered_identities.append(identity)
    identity_digest = hashlib.sha256(_canonical(ordered_identities)).hexdigest()
    if identity_digest != TTS_IDENTITY_SHA256 or identity_digest != tts_manifest.get("input_identity_sha256"):
        raise RunnerError(f"ordered full source identity digest mismatch: {identity_digest}")

    natural_by_id = {row["id"]: row for row in natural_rows}
    tts_by_id = tts_rows
    plan_contexts = []
    all_rows = []
    for ordered_index, row in enumerate(natural_rows):
        sample_id = row["id"]
        qrow = tts_by_id.get(sample_id)
        physical = physical_rows.get(sample_id)
        if qrow is None or physical is None or qrow.get("speaker") != row.get("speaker"):
            raise RunnerError(f"missing/mismatched TTS or physical row: {sample_id}")
        ident = {field: row.get(field) for field in identity_fields}
        if ident != qrow.get("input_identity"):
            raise RunnerError(f"natural and Qwen pair identity mismatch: {sample_id}")
        source_sha = row.get("ref_sha256")
        if not source_sha or physical.get("natural", {}).get("source_audio_sha256") != source_sha:
            raise RunnerError(f"physical producer natural source differs from frozen input: {sample_id}")
        tts_sha = qrow.get("qwen_canonical_16k", {}).get("sha256")
        if not tts_sha or physical.get("tts", {}).get("source_audio_sha256") != tts_sha:
            raise RunnerError(f"physical producer TTS source differs from frozen pair manifest: {sample_id}")
        utterance = sample_id.split("_", 1)[1]
        natural_path = input_root / "audio" / row["speaker"] / f"{utterance}.wav"
        tts_path = input_root / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
        if _sha(natural_path) != source_sha or _sha(tts_path) != tts_sha:
            raise RunnerError(f"selected natural/TTS PCM file hash mismatch: {sample_id}")
        if not physical.get("physical_alignment_pair_qualified"):
            raise RunnerError(f"physical MFA overlay failed qualification: {sample_id}")
        directions = physical.get("directions", {})
        expected_directions = {
            "tts_to_natural": ("natural", "tts"),
            "natural_to_tts": ("tts", "natural"),
        }
        context_plans = []
        for direction, (recipient, donor) in expected_directions.items():
            transfer = directions.get(direction)
            if not isinstance(transfer, dict) or transfer.get("recipient_condition") != recipient or transfer.get("donor_condition") != donor:
                raise RunnerError(f"donor/recipient direction binding mismatch: {sample_id}/{direction}")
            mapping = transfer.get("mapping", {})
            if not mapping.get("pair_qualified"):
                raise RunnerError(f"unqualified phone map: {sample_id}/{direction}")
            arms = transfer.get("arms", {})
            if set(arms) != set(WORLD_ARMS):
                raise RunnerError(f"WORLD arm set mismatch: {sample_id}/{direction}: {sorted(arms)}")
            fixed_info = physical.get(recipient, {}).get("fixed_audio")
            if not isinstance(fixed_info, dict) or fixed_info.get("sample_rate_hz") != ops.SAMPLE_RATE:
                raise RunnerError(f"missing fixed recipient waveform: {sample_id}/{recipient}")
            fixed_path = Path(fixed_info["path"]).resolve(strict=True)
            fixed_meta = {"path": str(fixed_path), "sha256": fixed_info["file_sha256"], "samples": int(fixed_info["samples"])}
            if _sha(fixed_path) != fixed_meta["sha256"]:
                raise RunnerError(f"fixed recipient waveform hash mismatch: {sample_id}/{recipient}")
            if fixed_meta["samples"] != int(physical[recipient]["raw_decoded_samples"]):
                raise RunnerError(f"fixed recipient waveform length mismatch: {sample_id}/{recipient}")
            fixed_wave_check, _, fixed_audio_check = _read_audio(fixed_path, fixed_meta["sha256"], fixed_meta["samples"])
            if fixed_audio_check["subtype"] != "FLOAT32":
                raise RunnerError(f"fixed recipient audio must remain FLOAT32: {sample_id}/{recipient}")
            computed_fixed_rms = float(ops.rms(fixed_wave_check))
            if abs(computed_fixed_rms - FIXED_RMS) > 1e-6:
                raise RunnerError(f"fixed recipient computed RMS differs from frozen target: {sample_id}/{recipient}: {computed_fixed_rms}")
            declared_rms = fixed_info.get("realized_rms", fixed_info.get("fixed_rms", physical[recipient].get("fixed_rms")))
            if declared_rms is not None and abs(float(declared_rms) - computed_fixed_rms) > 1e-6:
                raise RunnerError(f"fixed recipient declared/computed RMS mismatch: {sample_id}/{recipient}")
            fixed_meta["computed_global_rms"] = computed_fixed_rms
            arm_plan = {}
            for arm in WORLD_ARMS:
                arm_meta = arms[arm]
                if arm_meta.get("status") != "ok":
                    raise RunnerError(f"WORLD arm not successful: {sample_id}/{direction}/{arm}")
                arm_path = Path(arm_meta["path"]).resolve(strict=True)
                if _sha(arm_path) != arm_meta.get("file_sha256"):
                    raise RunnerError(f"WORLD arm file SHA mismatch: {sample_id}/{direction}/{arm}")
                if int(arm_meta.get("samples", -1)) != fixed_meta["samples"]:
                    raise RunnerError(f"WORLD arm does not preserve recipient length: {sample_id}/{direction}/{arm}")
                arm_plan[arm] = {"path": str(arm_path), "sha256": arm_meta["file_sha256"], "samples": int(arm_meta["samples"]), "float32_pcm_sha256": arm_meta["float32_pcm_sha256"]}
            raw_path, raw_sha = (natural_path, source_sha) if recipient == "natural" else (tts_path, tts_sha)
            context = {
                "ordered_eval_index": ordered_index, "sample_id": sample_id,
                "speaker_id": row["speaker"], "direction": direction,
                "recipient_condition": recipient, "donor_condition": donor,
                "raw_audio": {"path": str(raw_path.resolve()), "sha256": raw_sha},
                "fixed_audio": fixed_meta, "arms": arm_plan,
                "mapping_sha256": transfer["phone_map"].get("file_sha256"),
                "mapping_fingerprint": transfer.get("mapping", {}).get("manifest_sha256"),
                "qualification": {
                    "original_mfa_pair_qualified": physical.get("mfa_pair_qualified"),
                    "physical_alignment_pair_qualified": physical.get("physical_alignment_pair_qualified"),
                    "overlay_used": physical.get("physical_overlay_used"),
                    "physical_overlay_record_sha256": physical.get("physical_overlay_record_sha256"),
                },
            }
            context_plans.append(context)
            all_rows.extend(_make_alias_rows(sample_id, row["speaker"], ordered_index, direction, recipient))
        plan_contexts.extend(context_plans)

    if len(plan_contexts) != 160 or len(all_rows) != 12_160:
        raise RunnerError(f"WORLD row count mismatch: contexts={len(plan_contexts)}, labels={len(all_rows)}")
    unique_matrix_count = sum(len({row["matrix_id"] for row in all_rows if row["sample_id"] == c["sample_id"] and row["direction"] == c["direction"]}) for c in plan_contexts)
    if unique_matrix_count != 5_760:
        raise RunnerError(f"unique WORLD matrix count mismatch: {unique_matrix_count}")

    return natural_rows, {
        "source_manifest": {"path": str(input_manifest_path), "sha256": file_seals["natural/TTS input manifest"]["sha256"], "embedded_fingerprint": input_manifest.get("manifest_sha256")},
        "tts_pair_manifest": {"path": str(tts_manifest_path), "sha256": file_seals["frozen Qwen pair manifest"]["sha256"], "embedded_fingerprint": tts_manifest.get("manifest_sha256"), "input_identity_sha256": identity_digest},
        "physical_manifest": {"path": str(physical_manifest_path), "sha256": file_seals["physical EVAL manifest"]["sha256"], "embedded_fingerprint": physical_manifest.get("manifest_sha256")},
        "physical_plan": {"path": str(physical_plan_path), "sha256": file_seals["physical EVAL plan"]["sha256"], "embedded_fingerprint": physical_plan.get("manifest_sha256")},
        "cal_qa_freeze": {"path": str(freeze_path), "sha256": file_seals["independent CAL QA freeze"]["sha256"], "embedded_fingerprint": freeze.get("manifest_sha256", freeze.get("fingerprint"))},
        "calibration": {key: value for key, value in calibration.items() if key != "d_perp"},
        "face_registry": {"path": str(face_registry_path), "sha256": file_seals["frozen face registry"]["sha256"], "face_id": face_registry["face_id"], "face": face_registry["face"]},
        "spec": file_seals["WORLD physical specification"],
        "execution_bindings": file_seals["WORLD execution bindings"],
        "alignment_overlay": file_seals["physical alignment overlay"],
        "aliases_annex": file_seals["WORLD alias annex"],
        "execution_assets": execution_assets,
        "input_root": str(input_root),
        "context_plans": plan_contexts,
        "rows": all_rows,
        "physical_manifest_record_count": len(physical_manifest["records"]),
        "calibration_anchor": dict(ANCHOR),
        "fixed_rms": FIXED_RMS,
        "cohort": {"pairs": 80, "speakers": 40, "recipient_contexts": 160, "labels": 12_160, "unique_matrices": 5_760, "unique_videos": 1_600},
    }


def _write_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.control_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"preflight control directory must be empty: {output}")
    rows, source = _load_frozen_inputs(args)
    alias_counts = _validate_aliases()
    if alias_counts != {"unique_matrix_count": 36, "unique_video_count": 10, "unique_audio_count": 10, "label_count": 76}:
        raise AssertionError(alias_counts)
    code_files = {
        name: str((Path(__file__).resolve().parent / name).resolve())
        for name in (Path(__file__).name, "__init__.py", "evaluate.py", "ops.py", "scoring.py", "wav2lip.py")
    }
    code_hashes = {name: _sha(path) for name, path in code_files.items()}
    protocol = {
        "schema_version": 1,
        "classification": "sealed WORLD physical GPU evaluation; raw row artifacts and explicit four-cell aliases; no effect summary",
        "created_at_utc": _now(),
        "input_seals": {key: value for key, value in source.items() if key not in {"context_plans", "rows"}},
        "cohort": source["cohort"],
        "directions": {
            "tts_to_natural": {"donor": "tts", "recipient": "natural", "primary_EX_minus_RT_sign": "positive"},
            "natural_to_tts": {"donor": "natural", "recipient": "tts", "primary_EX_minus_RT_sign": "negative"},
        },
        "arms": list(WORLD_ARMS), "families": list(FAMILIES), "cells": list(CELLS),
        "alias_cardinality_per_recipient": alias_counts,
        "alias_specification": {
            "raw": {"q00": ["V_RT", "A_RAW"], "q10": ["V_X", "A_RAW"], "q01": ["V_RT", "A_X"], "q11": ["V_X", "A_X"]},
            "within_world": {"q00": ["V_RT", "A_RT"], "q10": ["V_X", "A_RT"], "q01": ["V_RT", "A_X"], "q11": ["V_X", "A_X"]},
            "bridge": {"q00": ["V_FIXED", "A_RAW"], "q10": ["V_RT", "A_RAW"], "q01": ["V_FIXED", "A_RT"], "q11": ["V_RT", "A_RT"]},
        },
        "audio_contract": "RAW PCM16 units are exact int16 promoted to float64; generated/fixed float32 PCM becomes float64*32768, no quantization, clipping, or rounding",
        "visual_contract": "fresh Wav2Lip mel/z/video per nine WORLD arms plus unsynthesized FIXED recipient, official FFmpeg crop/JPEG/OpenCV SyncNet lip path; crops/JPEGs temporary",
        "common_support": "one per recipient context: minimum actual count among all 10 visual lip windows and all 10 audio embedding domains; shared by both families; each score NPZ retains full V-by-full-A raw/unit cross matrices and full-V zero distances, while 31-lag curves query only the common visual prefix against the complete A domain",
        "supports": {"full": "all shared query rows", "valid": "all actual common V/A query rows; lag padding may occur at endpoints", "guard20": "indices 20..n-21", "guard15": "indices 15..n-16"},
        "anchor": {**ANCHOR, "selected_from": "frozen 20-pair CAL summary", "no evaluation re-estimation": True},
        "model_flags": {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True, "enforced_at_each_instrumented_call": True},
        "seeds": {"torch": 20261001, "numpy": 20261001},
        "planned_runtime_parameters": {"wav2lip_batch_size": args.wav2lip_batch_size, "syncnet_batch_size": args.syncnet_batch_size, "device": "cuda"},
        "storage": {"score_npz": "one compressed NPZ per unique V/A pair, containing raw/unit VxA cross, zero, and 31-lag matrices", "context_npz": "full waveform units, mel/chunks, z, MFCC, and A/V embeddings per recipient", "score_crop_and_jpeg": "temporary and deleted after lip embedding extraction", "videos": "retained MKV output and SHA"},
        "code_files": {name: {"path": path, "sha256": code_hashes[name]} for name, path in code_files.items()},
        "planned_row_count": len(source["rows"]),
        "planned_row_manifest_sha256": hashlib.sha256(b"".join(_canonical(row) + b"\n" for row in source["rows"])).hexdigest(),
        "planned_context_count": len(source["context_plans"]),
        "planned_unique_matrix_count": source["cohort"]["unique_matrices"],
        "planned_unique_video_count": source["cohort"]["unique_videos"],
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical(protocol)).hexdigest()
    plan = {
        "schema_version": 1, "status": "preflight_pass", "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "contexts": source["context_plans"], "rows": source["rows"],
        "created_at_utc": _now(),
    }
    plan["plan_sha256"] = hashlib.sha256(_canonical(plan)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "execution_plan.json", plan)
    _write_json(output / "preflight_status.json", {
        "status": "preflight_pass", "planned_contexts": 160, "planned_rows": 12_160,
        "unique_matrices": 5_760, "unique_videos": 1_600,
        "plan_sha256": plan["plan_sha256"], "protocol_sha256": protocol["protocol_sha256"],
        "input_manifest_sha256": source["source_manifest"]["sha256"],
        "physical_manifest_sha256": source["physical_manifest"]["sha256"],
        "preflight_finished_at_utc": _now(),
    })
    return {"status": "preflight_pass", "contexts": 160, "rows": 12_160, "unique_matrices": 5_760, "unique_videos": 1_600, "plan_sha256": plan["plan_sha256"], "protocol_sha256": protocol["protocol_sha256"]}


def _write_wave_and_features(
    run_dir: Path, temporary_root: Path, sample_id: str, direction: str, visual_id: str,
    waveform: np.ndarray, frame: np.ndarray, face_meta: dict[str, Any], model: Any,
    face_cache: dict[int, list[Any]], syncnet: Any, ffmpeg: Path, device: str,
    wav2lip_batch_size: int, syncnet_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    video_path = run_dir / "videos" / sample_id / direction / f"{visual_id}.mkv"
    video_meta = wav2lip.render_video(
        model, np.asarray(face_meta["z"], dtype=np.float32), frame,
        face_meta["face"]["generation_box_xyxy"], face_cache, video_path, ffmpeg,
        batch_size=wav2lip_batch_size, device=device,
    )
    generated_frame_count = int(len(face_meta["z"]))
    encoded_frame_count = int(video_meta.get("frame_count", -1))
    if encoded_frame_count != generated_frame_count:
        raise RunnerError(
            f"encoded video frame count differs from latent count: "
            f"{sample_id}/{direction}/{visual_id}: {encoded_frame_count} != {generated_frame_count}"
        )
    decoded_video_frame_count = 0
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RunnerError(f"OpenCV could not decode generated video: {video_path}")
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded_video_frame_count += 1
    finally:
        capture.release()
    if decoded_video_frame_count != generated_frame_count:
        raise RunnerError(
            f"decoded video frame count differs from latent count: "
            f"{sample_id}/{direction}/{visual_id}: {decoded_video_frame_count} != {generated_frame_count}"
        )
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{sample_id}_{direction}_{visual_id}_", dir=temporary_root))
    crop_path = temp_dir / "score_crop.mkv"
    jpeg_dir = temp_dir / "official_jpeg"
    try:
        crop_meta = wav2lip.crop_score_video(video_path, face_meta["face"]["score_box"], crop_path, ffmpeg)
        cropped_frame_count = int(crop_meta.get("frame_count", -1))
        if cropped_frame_count != decoded_video_frame_count:
            raise RunnerError(
                f"score crop frame count differs from decoded source video: "
                f"{sample_id}/{direction}/{visual_id}: {cropped_frame_count} != {decoded_video_frame_count}"
            )
        crop_sha = _sha(crop_path)
        jpeg_frames, jpeg_meta = scoring.extract_official_jpegs(crop_path, jpeg_dir, ffmpeg)
        jpeg_frame_count = int(jpeg_meta.get("frame_count", -1))
        if jpeg_frame_count != cropped_frame_count:
            raise RunnerError(
                f"official JPEG frame count differs from crop count: "
                f"{sample_id}/{direction}/{visual_id}: {jpeg_frame_count} != {cropped_frame_count}"
            )
        lip = scoring.lip_embeddings_from_frames(syncnet, jpeg_frames, device=device, batch_size=syncnet_batch_size)
        expected_lip_count = cropped_frame_count - 5
        if len(lip) != expected_lip_count:
            raise RunnerError(f"official SyncNet lip windows disagree with decoded JPEG count: {sample_id}/{visual_id}")
        meta = {
            "video_path": str(video_path.relative_to(run_dir)), "video_sha256": _sha(video_path),
            "video_size_bytes": video_path.stat().st_size, "video_meta": video_meta,
            "score_crop_ephemeral_sha256": crop_sha,
            "score_crop_method": crop_meta.get("crop_method"),
            "latent_frame_count": generated_frame_count,
            "encoded_video_frame_count": encoded_frame_count,
            "decoded_video_frame_count": decoded_video_frame_count,
            "score_crop_frame_count": cropped_frame_count,
            "official_jpeg": {key: value for key, value in jpeg_meta.items() if key != "jpeg_directory"} | {"jpeg_directory_retention": "ephemeral"},
            "frame_count": jpeg_frame_count,
            "official_jpeg_frame_count": jpeg_frame_count, "lip_window_count": len(lip),
            "lip_embedding_shape": list(lip.shape), "lip_embedding_dtype": str(lip.dtype),
            "lip_embedding_sha256": ops.array_sha256(lip),
            "face_registry_sha256": face_meta["registry_sha256"],
            "face_geometry": face_meta["face"],
        }
        visual_path = run_dir / "visuals" / sample_id / direction / f"{visual_id}.npz"
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(visual_path, latent_z_float32=np.asarray(face_meta["z"], dtype=np.float32), lip_embeddings_float32=lip)
        meta["visual_npz"] = str(visual_path.relative_to(run_dir))
        meta["visual_npz_sha256"] = _sha(visual_path)
        _write_json(visual_path.with_suffix(".json"), meta)
        return np.asarray(lip, dtype=np.float32), meta
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _compute_stream_audio(
    stream_id: str, waveform: np.ndarray, units: np.ndarray, syncnet: Any,
    device: str, syncnet_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mfcc, audio_features, audio_meta = evaluate._audio_forward(syncnet, units, device=device, batch_size=syncnet_batch_size)
    window_count = len(audio_features)
    expected = len(units) // 640 - 5
    if expected < 1:
        raise RunnerError(f"audio input is shorter than official SyncNet support for {stream_id}: {len(units)} samples")
    if window_count != expected:
        raise RunnerError(f"audio window count mismatch for {stream_id}: {window_count} != {expected}")
    return mfcc, audio_features.astype(np.float32, copy=False), {
        "stream_id": stream_id, "waveform_shape": list(waveform.shape),
        "waveform_dtype": str(waveform.dtype), "waveform_sha256": ops.array_sha256(waveform),
        "units_shape": list(units.shape), "units_dtype": str(units.dtype),
        "units_sha256": ops.array_sha256(units), "mfcc_shape": list(mfcc.shape),
        "mfcc_dtype": str(mfcc.dtype), "mfcc_sha256": ops.array_sha256(mfcc),
        "audio_embedding_shape": list(audio_features.shape),
        "expected_syncnet_window_count": int(expected),
        "actual_syncnet_window_count": int(window_count),
        "audio_embedding_sha256": ops.array_sha256(audio_features), "syncnet_audio": audio_meta,
    }


def _score_pair(
    run_dir: Path, context_key: str, matrix_id: str, visual_id: str, audio_id: str,
    lip: np.ndarray, audio: np.ndarray, common_count: int, support: dict[str, np.ndarray], device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v = np.asarray(lip, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if len(v) < common_count or len(a) < common_count or common_count < 1:
        raise RunnerError(f"empty or misaligned common support for {context_key}/{matrix_id}")
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1]:
        raise RunnerError(f"invalid full visual/audio embedding domains for {context_key}/{matrix_id}")
    cross, zero = evaluate._cross_distance_matrix(v, a, device)
    if cross.shape != (len(v), len(a)) or zero.shape != (len(v),):
        raise RunnerError(f"raw cross-distance computation truncated a full domain for {context_key}/{matrix_id}")
    lag = evaluate._lag_matrix_from_cross(cross, zero, common_count, len(a), vshift=VSHIFT)
    v_unit, a_unit = evaluate._row_unit(v), evaluate._row_unit(a)
    cross_unit, zero_unit = evaluate._cross_distance_matrix(v_unit, a_unit, device)
    if cross_unit.shape != (len(v), len(a)) or zero_unit.shape != (len(v),):
        raise RunnerError(f"unit cross-distance computation truncated a full domain for {context_key}/{matrix_id}")
    lag_unit = evaluate._lag_matrix_from_cross(cross_unit, zero_unit, common_count, len(a_unit), vshift=VSHIFT)
    if lag.shape != (common_count, 2 * VSHIFT + 1) or lag_unit.shape != (common_count, 2 * VSHIFT + 1):
        raise RunnerError(f"lag gather did not preserve common visual query prefix for {context_key}/{matrix_id}")
    path = run_dir / "scores" / context_key / f"{matrix_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cross_distance_raw_float32=cross,
        cross_distance_to_zero_raw_float32=zero,
        lag_distance_raw_float32=lag,
        cross_distance_unit_float32=cross_unit,
        cross_distance_to_zero_unit_float32=zero_unit,
        lag_distance_unit_float32=lag_unit,
        common_support_full_int32=support["full"],
        common_support_valid_int32=support["valid"],
        common_support_guard20_int32=support["guard20"],
        common_support_guard15_int32=support["guard15"],
    )
    metadata = {
        "matrix_id": matrix_id, "visual_id": visual_id, "audio_id": audio_id,
        "score_npz": str(path.relative_to(run_dir)), "score_npz_sha256": _sha(path),
        "visual_rows": int(len(lip)), "audio_rows": int(len(audio)),
        "common_window_count": int(common_count),
        "common_query_prefix_count": int(common_count),
        "full_visual_domain_count": int(len(v)),
        "full_audio_domain_count": int(len(a)),
        "full_visual_domain_sha256": ops.array_sha256(v),
        "full_audio_domain_sha256": ops.array_sha256(a),
        "matrix_shapes": {"cross_raw": list(cross.shape), "lag_raw": list(lag.shape), "cross_unit": list(cross_unit.shape), "lag_unit": list(lag_unit.shape)},
        "array_sha256": {
            "cross_distance_raw_float32": ops.array_sha256(cross),
            "cross_distance_to_zero_raw_float32": ops.array_sha256(zero),
            "lag_distance_raw_float32": ops.array_sha256(lag),
            "cross_distance_unit_float32": ops.array_sha256(cross_unit),
            "cross_distance_to_zero_unit_float32": ops.array_sha256(zero_unit),
            "lag_distance_unit_float32": ops.array_sha256(lag_unit),
        },
        "raw_support_metrics": evaluate._metric_supports(lag, ANCHOR["column"]),
        "unit_support_metrics": evaluate._metric_supports(lag_unit, ANCHOR["column"]),
    }
    return metadata, {"raw": lag, "unit": lag_unit}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is required for WORLD neural evaluation")
    control_dir = Path(args.control_dir).resolve(strict=True)
    plan_path = control_dir / "execution_plan.json"
    plan = _json(plan_path)
    plan_sha = plan.get("plan_sha256")
    without_sha = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if hashlib.sha256(_canonical(without_sha)).hexdigest() != plan_sha:
        raise RunnerError("preflight plan self-hash mismatch")
    _require_sha(plan_path, args.plan_sha256, "execution plan")
    if plan.get("status") != "preflight_pass":
        raise RunnerError("WORLD GPU run requires preflight_pass")
    protocol = plan.get("protocol", {})
    if plan.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise RunnerError("execution plan protocol binding mismatch")
    protocol_without_sha = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if hashlib.sha256(_canonical(protocol_without_sha)).hexdigest() != protocol.get("protocol_sha256"):
        raise RunnerError("preflight protocol self-hash mismatch")
    for name, record in protocol.get("code_files", {}).items():
        code_path = Path(record["path"]).resolve(strict=True)
        if _sha(code_path) != record.get("sha256"):
            raise RunnerError(f"preflight code file changed before run: {name}")
    # Recheck sealed manifests, source hashes, arm files, and the full planned
    # row/context grids before creating run output or loading either model.
    _, current_source = _load_frozen_inputs(args)
    current_seals = {key: value for key, value in current_source.items() if key not in {"context_plans", "rows"}}
    if _canonical(current_seals) != _canonical(protocol.get("input_seals")):
        raise RunnerError("frozen input seal inventory changed between preflight and run")
    if _canonical(current_source["context_plans"]) != _canonical(plan.get("contexts")):
        raise RunnerError("source contexts changed between preflight and run")
    if _canonical(current_source["rows"]) != _canonical(plan.get("rows")):
        raise RunnerError("row labels changed between preflight and run")
    runtime_parameters = {"wav2lip_batch_size": args.wav2lip_batch_size, "syncnet_batch_size": args.syncnet_batch_size, "device": "cuda"}
    if runtime_parameters != protocol.get("planned_runtime_parameters"):
        raise RunnerError("runtime batch/device parameters differ from the preflight seal")
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty WORLD output: {run_dir}")
    temp_root = Path(args.jpeg_temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("contexts", "visuals", "videos", "scores"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    protocol = dict(plan["protocol"])
    protocol["run_started_at_utc"] = _now()
    protocol["run_dir"] = str(run_dir)
    protocol["jpeg_temp_root"] = str(temp_root)
    protocol["runtime_parameters"] = runtime_parameters
    protocol["runtime_parameters_sha256"] = hashlib.sha256(_canonical(protocol["runtime_parameters"])).hexdigest()
    rows = plan["rows"]
    rows_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    rows_sha = hashlib.sha256(rows_bytes).hexdigest()
    if rows_sha != protocol["planned_row_manifest_sha256"]:
        raise RunnerError("preflight ordered-row content changed")
    row_path = run_dir / "evaluation_rows.jsonl"
    row_path.write_bytes(rows_bytes)
    protocol["row_manifest_sha256"] = rows_sha
    protocol["execution_plan_sha256"] = plan_sha
    protocol["protocol_sha256"] = hashlib.sha256(_canonical({key: value for key, value in protocol.items() if key != "protocol_sha256"})).hexdigest()
    _write_json(run_dir / "evaluation_protocol.json", protocol)
    _write_json(run_dir / "code_seal.json", {
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": rows_sha,
        "execution_plan_sha256": plan_sha,
        "code_files": protocol["code_files"],
    })
    _write_json(run_dir / "run_status.json", {"status": "running", "planned_contexts": 160, "planned_rows": 12_160, "completed_contexts": 0, "failed_contexts": 0, "plan_sha256": plan_sha})
    _append_jsonl(run_dir / "attempts.jsonl", {"event": "run_started", "status": "running", "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"), "started_at_utc": _now()})

    torch.manual_seed(20261001)
    np.random.seed(20261001)
    _set_flags()
    frontend = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(Path(args.wav2lip_root).resolve(), args.wav2lip_checkpoint, device="cuda")
    syncnet, syncnet_meta = scoring.load_syncnet(Path(args.syncnet_root).resolve(), args.syncnet_checkpoint, device="cuda")
    frame = cv2.imread(str(Path(args.primary_face).resolve()))
    if frame is None or tuple(frame.shape[:2]) != (512, 512):
        raise RunnerError("primary face image could not be decoded at its frozen 512x512 dimensions")
    face_registry = _json(Path(args.face_registry).resolve(strict=True))
    face = face_registry["face"]
    face_meta = {"registry_sha256": FACE_REGISTRY_SHA256, "face": face}
    if _sha(Path(args.primary_face).resolve()) != face.get("sha256"):
        raise RunnerError("primary face input hash differs from face registry")
    d_axis, u_axis, calibration = _load_calibration(args)
    d_perp = np.asarray(calibration["d_perp"], dtype=np.float64)
    events: list[dict[str, Any]] = []
    forward_guard_counts: dict[str, int] = {}
    active: dict[str, str | None] = {"sample_id": None, "direction": None, "stream_id": None, "visual_id": None, "audio_id": None, "event_log": str(run_dir / "forward_events.jsonl")}
    originals = _instrument(events, active, model, syncnet, forward_guard_counts)
    completed = 0
    failed: list[dict[str, Any]] = []
    try:
        for context_index, context in enumerate(plan["contexts"], 1):
            sample_id = context["sample_id"]
            direction = context["direction"]
            context_key = f"{sample_id}__{direction}"
            active.update({"sample_id": sample_id, "direction": direction, "stream_id": None, "visual_id": None, "audio_id": None})
            started = time.perf_counter()
            try:
                raw_wave, raw_units, raw_meta = _read_audio(Path(context["raw_audio"]["path"]), context["raw_audio"]["sha256"])
                if raw_meta["subtype"] != "PCM_16":
                    raise RunnerError(f"RAW recipient scoring audio must be PCM16: {sample_id}/{direction}")
                fixed_wave, _, fixed_meta = _read_audio(Path(context["fixed_audio"]["path"]), context["fixed_audio"]["sha256"], context["fixed_audio"]["samples"])
                if fixed_meta["subtype"] != "FLOAT32":
                    raise RunnerError(f"FIXED recipient generation audio must remain float32: {sample_id}/{direction}")

                streams: dict[str, dict[str, Any]] = {}
                active["stream_id"] = "A_RAW"
                raw_mfcc, raw_audio, raw_audio_meta = _compute_stream_audio("A_RAW", raw_wave, raw_units, syncnet, "cuda", args.syncnet_batch_size)
                streams["A_RAW"] = {"waveform": raw_wave, "units": raw_units, "mfcc": raw_mfcc, "audio": raw_audio, "metadata": {**raw_meta, "audio": raw_audio_meta, "mfcc_sha256": ops.array_sha256(raw_mfcc)}}

                # Every WORLD arm is both a video-generation stream and its own A_X.
                for arm in WORLD_ARMS:
                    arm_info = context["arms"][arm]
                    waveform, units, meta = _read_audio(Path(arm_info["path"]), arm_info["sha256"], arm_info["samples"])
                    if meta["subtype"] != "FLOAT32" or meta["float32_pcm_sha256"] != arm_info["float32_pcm_sha256"]:
                        raise RunnerError(f"WORLD arm decoded float waveform identity mismatch: {sample_id}/{direction}/{arm}")
                    active["stream_id"] = arm
                    mel, chunks, starts, front = wav2lip.mel_chunks(waveform, args.wav2lip_root)
                    z = wav2lip.encode_latents(model, chunks, device="cuda", batch_size=args.wav2lip_batch_size)
                    mfcc, audio_features, audio_meta = _compute_stream_audio(f"A_{arm}", waveform, units, syncnet, "cuda", args.syncnet_batch_size)
                    streams[f"A_{arm}"] = {
                        "waveform": waveform, "units": units, "mfcc": mfcc, "audio": audio_features,
                        "mel": mel, "chunks": chunks, "mel_starts": np.asarray(starts, dtype=np.int32), "z": z,
                        "metadata": {**meta, "audio": audio_meta, "mel_frontend": front, "mel_sha256": ops.array_sha256(mel), "chunk_sha256": ops.array_sha256(chunks), "z_sha256": ops.array_sha256(z), "mfcc_sha256": ops.array_sha256(mfcc)},
                    }

                active["stream_id"] = "V_FIXED"
                fixed_mel, fixed_chunks, fixed_starts, fixed_front = wav2lip.mel_chunks(fixed_wave, args.wav2lip_root)
                fixed_z = wav2lip.encode_latents(model, fixed_chunks, device="cuda", batch_size=args.wav2lip_batch_size)
                fixed_stream = {"waveform": fixed_wave, "mel": fixed_mel, "chunks": fixed_chunks, "mel_starts": np.asarray(fixed_starts, dtype=np.int32), "z": fixed_z, "metadata": {**fixed_meta, "mel_frontend": fixed_front, "mel_sha256": ops.array_sha256(fixed_mel), "chunk_sha256": ops.array_sha256(fixed_chunks), "z_sha256": ops.array_sha256(fixed_z)}}
                fixed_speech_mask, fixed_speech_mask_meta = ops.fixed_speech_mask(fixed_wave, fixed_stream["mel_starts"])
                if fixed_speech_mask.shape != (len(fixed_z),):
                    raise RunnerError(f"fixed-recipient speech mask does not match latent frames: {sample_id}/{direction}")
                for arm in WORLD_ARMS:
                    arm_stream = streams[f"A_{arm}"]
                    if len(arm_stream["z"]) != len(fixed_z) or not np.array_equal(arm_stream["mel_starts"], fixed_stream["mel_starts"]):
                        raise RunnerError(f"fixed-recipient mask cannot be transferred to WORLD frames: {sample_id}/{direction}/{arm}")

                # Four-cell labels share one real lip/audio window domain per recipient.
                generation_streams = {f"V_{arm}": streams[f"A_{arm}"] for arm in WORLD_ARMS}
                generation_streams["V_FIXED"] = fixed_stream
                counts = set()
                for item in generation_streams.values():
                    length = len(item["z"])
                    if length <= 0:
                        raise RunnerError(f"Wav2Lip produced no frames: {sample_id}/{direction}")
                    counts.add(min(args.wav2lip_batch_size, length))
                    tail = length % args.wav2lip_batch_size
                    if tail:
                        counts.add(tail)
                active["stream_id"] = "face_feature_cache"
                face_cache = wav2lip.cache_face_features(model, frame, face["generation_box_xyxy"], sorted(counts), device="cuda")

                lip_by_visual: dict[str, np.ndarray] = {}
                visual_meta: dict[str, dict[str, Any]] = {}
                for visual_id, item in generation_streams.items():
                    active.update({"stream_id": visual_id, "visual_id": visual_id})
                    entry_face = {"face": face, "z": item["z"], "registry_sha256": FACE_REGISTRY_SHA256}
                    lip, vmeta = _write_wave_and_features(
                        run_dir, temp_root, sample_id, direction, visual_id, item["waveform"],
                        frame, entry_face, model, face_cache, syncnet, Path(args.ffmpeg), "cuda",
                        args.wav2lip_batch_size, args.syncnet_batch_size,
                    )
                    lip_by_visual[visual_id] = lip
                    visual_meta[visual_id] = vmeta

                audio_by_id = {"A_RAW": raw_audio}
                audio_by_id.update({f"A_{arm}": streams[f"A_{arm}"]["audio"] for arm in WORLD_ARMS})
                common_count = min(
                    *(len(value) for value in lip_by_visual.values()),
                    *(len(value) for value in audio_by_id.values()),
                )
                if common_count < 1:
                    raise RunnerError(f"no common official SyncNet window support: {sample_id}/{direction}")
                support = evaluate._support_indices(common_count)

                # Save reconstructible raw audio, mel/MFCC, z, A/V and common supports.
                context_arrays: dict[str, np.ndarray] = {
                    "raw_recipient_waveform_float32": raw_wave,
                    "raw_syncnet_units_float64": raw_units,
                    "raw_mfcc_float64": raw_mfcc,
                    "raw_audio_embeddings_float32": raw_audio,
                    "fixed_recipient_waveform_float32": fixed_wave,
                    "fixed_recipient_mel_float32": fixed_mel,
                    "fixed_recipient_chunks_float32": fixed_chunks,
                    "fixed_recipient_mel_starts_int32": np.asarray(fixed_starts, dtype=np.int32),
                    "fixed_recipient_z_float32": fixed_z,
                    "fixed_recipient_speech_mask_bool": fixed_speech_mask.astype(bool, copy=False),
                    "fixed_recipient_speech_frame_rms_float64": np.asarray(fixed_speech_mask_meta["frame_rms"], dtype=np.float64),
                    "calibration_d_source_float64": np.asarray(d_axis, dtype=np.float64),
                    "calibration_u_level_float64": np.asarray(u_axis, dtype=np.float64),
                    "calibration_d_outside_u_float64": d_perp,
                    "common_support_full_int32": support["full"],
                    "common_support_valid_int32": support["valid"],
                    "common_support_guard20_int32": support["guard20"],
                    "common_support_guard15_int32": support["guard15"],
                }
                for arm in WORLD_ARMS:
                    stream = streams[f"A_{arm}"]
                    for key, value in (("waveform", stream["waveform"]), ("units", stream["units"]), ("mfcc", stream["mfcc"]), ("audio_embeddings", stream["audio"]), ("mel", stream["mel"]), ("chunks", stream["chunks"]), ("mel_starts", stream["mel_starts"]), ("z", stream["z"])):
                        context_arrays[f"arm_{arm}_{key}"] = value
                    context_arrays[f"arm_{arm}_lip_embeddings"] = lip_by_visual[f"V_{arm}"]
                context_arrays["fixed_recipient_lip_embeddings"] = lip_by_visual["V_FIXED"]
                context_npz = run_dir / "contexts" / f"{context_key}.npz"
                context_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(context_npz, **context_arrays)
                context_meta = {
                    **context,
                    "raw_audio_meta": {key: value for key, value in raw_meta.items() if key != "pcm16"},
                    "fixed_audio_meta": {key: value for key, value in fixed_meta.items() if key != "pcm16"},
                    "streams": {key: {k: v for k, v in value["metadata"].items() if k != "pcm16"} for key, value in streams.items()},
                    "fixed_frontend": fixed_stream["metadata"],
                    "fixed_recipient_speech_mask": fixed_speech_mask_meta,
                    "latent_axis_descriptive_binding": {
                        "calibration": {key: value for key, value in calibration.items() if key != "d_perp"},
                        "interpretation": "saved fixed-recipient speech mask and z trajectories for descriptive projection onto frozen CAL d/u axes; no new axis fit and no mediation label",
                        "mask_sha256": ops.array_sha256(fixed_speech_mask.astype(np.uint8)),
                        "d_source_sha256": ops.array_sha256(d_axis),
                        "u_level_sha256": ops.array_sha256(u_axis),
                        "d_outside_u_sha256": ops.array_sha256(d_perp),
                    },
                    "visuals": visual_meta,
                    "common_window_count": int(common_count),
                    "support_indices": {key: value.tolist() for key, value in support.items()},
                    "context_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": ops.array_sha256(value)} for key, value in context_arrays.items()},
                    "context_npz": {"path": str(context_npz.relative_to(run_dir)), "sha256": _sha(context_npz), "size_bytes": context_npz.stat().st_size},
                    "actual_flags_at_forward": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events if event.get("sample_id") == sample_id and event.get("direction") == direction}),
                }
                _write_json(context_npz.with_suffix(".json"), context_meta)

                # Matrix cache is keyed only by actual visual/audio artifact identities.
                pair_rows = [row for row in rows if row["sample_id"] == sample_id and row["direction"] == direction]
                pairs = sorted({(row["matrix_id"], row["visual_id"], row["audio_id"]) for row in pair_rows})
                if len(pairs) != 36:
                    raise RunnerError(f"alias unique pair count changed for {context_key}: {len(pairs)}")
                matrices: dict[str, dict[str, Any]] = {}
                for matrix_id, visual_id, audio_id in pairs:
                    active.update({"stream_id": matrix_id, "visual_id": visual_id, "audio_id": audio_id})
                    audio = audio_by_id[audio_id]
                    matrix_meta, _matrix_values = _score_pair(
                        run_dir, context_key, matrix_id, visual_id, audio_id,
                        lip_by_visual[visual_id], audio, common_count, support, "cuda",
                    )
                    matrices[matrix_id] = matrix_meta
                for row in pair_rows:
                    matrix_meta = matrices[row["matrix_id"]]
                    result = {
                        **row,
                        "status": "scored",
                        "score_npz": matrix_meta["score_npz"],
                        "score_npz_sha256": matrix_meta["score_npz_sha256"],
                        "matrix_id": row["matrix_id"],
                        "common_window_count": common_count,
                        "audio_domain_count": len(audio_by_id[row["audio_id"]]),
                        "visual_domain_count": len(lip_by_visual[row["visual_id"]]),
                        "fixed_anchor_k": ANCHOR["k"],
                        "fixed_anchor_column": ANCHOR["column"],
                        "fixed_anchor_official_offset": ANCHOR["official_offset"],
                        "raw_support_metrics": matrix_meta["raw_support_metrics"],
                        "unit_support_metrics": matrix_meta["unit_support_metrics"],
                        "context_npz": str(context_npz.relative_to(run_dir)),
                        "context_npz_sha256": context_meta["context_npz"]["sha256"],
                        "visual_npz": visual_meta[row["visual_id"]]["visual_npz"],
                        "visual_npz_sha256": visual_meta[row["visual_id"]]["visual_npz_sha256"],
                        "video_path": visual_meta[row["visual_id"]]["video_path"],
                        "video_sha256": visual_meta[row["visual_id"]]["video_sha256"],
                        "audio_array_key": "raw_audio_embeddings_float32" if row["audio_id"] == "A_RAW" else f"arm_{row['audio_id'][2:]}_audio_embeddings",
                        "common_support_full": support["full"].tolist(),
                        "common_support_guard20": support["guard20"].tolist(),
                        "common_support_guard15": support["guard15"].tolist(),
                    }
                    _append_jsonl(run_dir / "evaluation_results.jsonl", result)
                completed += 1
                _append_jsonl(run_dir / "attempts.jsonl", {
                    "sample_id": sample_id, "direction": direction, "status": "complete",
                    "completed_contexts": completed, "planned_contexts": 160,
                    "context_row_count": len(pair_rows), "unique_matrix_count": len(matrices),
                    "common_window_count": common_count, "elapsed_seconds": time.perf_counter() - started,
                })
                _write_json(run_dir / "run_status.json", {
                    "status": "running", "planned_contexts": 160, "planned_rows": 12_160,
                    "completed_contexts": completed, "failed_contexts": len(failed),
                    "last_context": context_key, "plan_sha256": plan_sha,
                    "forward_guard_counts": dict(forward_guard_counts),
                    "forward_flags_required_and_asserted": EXPECTED_FORWARD_FLAGS,
                })
            except Exception as exc:
                fail = {
                    "sample_id": sample_id, "direction": direction,
                    "error_type": type(exc).__name__, "error": str(exc)[:2000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                failed.append(fail)
                _append_jsonl(run_dir / "attempts.jsonl", {"status": "failed", "completed_contexts": completed, **fail})
                for row in rows:
                    if row["sample_id"] == sample_id and row["direction"] == direction:
                        _append_jsonl(run_dir / "evaluation_results.jsonl", {**row, "status": "failed", "error_type": fail["error_type"], "error": fail["error"]})
                _write_json(run_dir / "run_status.json", {
                    "status": "running", "planned_contexts": 160, "planned_rows": 12_160,
                    "completed_contexts": completed, "failed_contexts": len(failed),
                    "last_context": context_key, "plan_sha256": plan_sha,
                    "forward_guard_counts": dict(forward_guard_counts),
                    "forward_flags_required_and_asserted": EXPECTED_FORWARD_FLAGS,
                })
    finally:
        _restore_instrument(originals)

    rows_results = [json.loads(line) for line in (run_dir / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    final = {
        "status": "complete_with_failures" if failed else "complete",
        "classification": "sealed WORLD GPU raw artifacts and cell aliases; no effects summarized",
        "planned_contexts": 160, "completed_contexts": completed, "failed_contexts": len(failed),
        "planned_rows": 12_160, "scored_rows": sum(row.get("status") == "scored" for row in rows_results),
        "failed_rows": sum(row.get("status") == "failed" for row in rows_results),
        "unique_matrix_artifacts": sum(1 for _ in (run_dir / "scores").rglob("*.npz")),
        "unique_video_artifacts": sum(1 for _ in (run_dir / "videos").rglob("*.mkv")),
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": _sha(row_path), "execution_plan_sha256": plan_sha,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(0), "numpy": np.__version__, "opencv": cv2.__version__, "frontend_compatibility": frontend, "wav2lip_model": model_meta, "syncnet_model": syncnet_meta},
        "flags_at_forward": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events}),
        "forward_event_count": len(events), "forward_event_log_sha256": _sha(run_dir / "forward_events.jsonl") if (run_dir / "forward_events.jsonl").is_file() else None,
        "forward_guard_counts": dict(forward_guard_counts),
        "forward_flags_required_and_asserted": EXPECTED_FORWARD_FLAGS,
        "failed_context_details": failed, "finished_at_utc": _now(),
    }
    _write_json(run_dir / "run_status.json", final)
    _write_json(run_dir / "all_files_sha256.json", {
        str(path.relative_to(run_dir)).replace("\\", "/"): _sha(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    })
    return final


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--tts-pair-manifest", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-axes", required=True)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--physical-plan", required=True)
    parser.add_argument("--cal-qa-freeze", required=True)
    parser.add_argument("--face-registry", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--execution-bindings", required=True)
    parser.add_argument("--alignment-overlay", required=True)
    parser.add_argument("--aliases", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_shared_arguments(preflight)
    preflight.add_argument("--control-dir", required=True)
    run = subparsers.add_parser("run")
    _add_shared_arguments(run)
    run.add_argument("--control-dir", required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--jpeg-temp-root", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = _write_preflight(args) if args.command == "preflight" else _run(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if result.get("status") in {"preflight_pass", "complete"} else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
