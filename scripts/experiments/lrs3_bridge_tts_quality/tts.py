"""Stage 01: produce the two provider inputs for the bridge experiment.

The cloud side is deliberately an audit of already registered canonical files.
This module never instantiates the DashScope provider.  The local provider is
loaded once, then called serially with a per-record RNG sandbox so that a
failed or resumed record cannot alter another record's seed.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16, write_pcm16
from .common import (
    ProtocolError,
    assert_not_sealed,
    copy_verified,
    file_sha256,
    record_seed,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .gpu import gpu_lease, release_torch_memory


def _transcript_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _as_mono_float(audio: Any) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(audio)
    original_dtype = str(array.dtype)
    original_shape = list(array.shape)
    if array.ndim == 0 or array.size == 0:
        raise ProtocolError("provider returned an empty waveform")
    if array.ndim > 1:
        if array.ndim != 2:
            raise ProtocolError(f"provider waveform has unsupported shape: {array.shape}")
        # Qwen backends return [channels, samples]; soundfile-style backends
        # commonly return [samples, channels].  The small dimension is the
        # only acceptable channel axis for this experiment.
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.mean(axis=0)
            channel_axis = 0
        elif array.shape[1] <= 8 and array.shape[0] > array.shape[1]:
            array = array.mean(axis=1)
            channel_axis = 1
        else:
            raise ProtocolError(f"cannot determine mono channel axis: {array.shape}")
    else:
        channel_axis = None
    if np.issubdtype(array.dtype, np.integer):
        if array.dtype.itemsize != 2:
            raise ProtocolError(f"provider integer waveform is not int16: {array.dtype}")
        values = array.astype(np.float64) / 32768.0
    else:
        values = array.astype(np.float64, copy=False)
    values = values.reshape(-1)
    if not np.isfinite(values).all():
        raise ProtocolError("provider waveform contains non-finite samples")
    if not np.any(values != 0.0):
        raise ProtocolError("provider waveform is silent")
    peak = float(np.max(np.abs(values)))
    if peak > 1.0 + 1e-7:
        raise ProtocolError(f"provider waveform exceeds [-1,1]: peak={peak}")
    return values, {"original_dtype": original_dtype, "original_shape": original_shape, "channel_axis": channel_axis, "peak": peak}


def canonicalize_provider_audio(
    audio: Any,
    provider_sample_rate: int,
    natural_samples: int,
    natural_duration_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert provider output to the registered 16 kHz PCM16 contract."""

    if int(provider_sample_rate) <= 0:
        raise ProtocolError("provider sample rate is invalid")
    values, source_qc = _as_mono_float(audio)
    duration_s = float(values.size) / int(provider_sample_rate)
    ratio = duration_s / float(natural_duration_s)
    if duration_s > config.MAX_AUDIO_SECONDS or not config.MIN_DURATION_RATIO <= ratio <= config.MAX_DURATION_RATIO:
        raise ProtocolError(
            f"provider duration gate failed: duration={duration_s:.6f}s ratio={ratio:.6f}"
        )
    if int(provider_sample_rate) == config.SAMPLE_RATE:
        resampled = values
        resample_meta = {"applied": False, "up": 1, "down": 1, "window": None, "padtype": None}
    else:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(config.SAMPLE_RATE, int(provider_sample_rate))
        up = config.SAMPLE_RATE // divisor
        down = int(provider_sample_rate) // divisor
        resampled = np.asarray(
            resample_poly(
                values,
                up,
                down,
                window=("kaiser", 5.0),
                padtype="constant",
            ),
            dtype=np.float64,
        )
        resample_meta = {
            "applied": True,
            "up": up,
            "down": down,
            "window": ["kaiser", 5.0],
            "padtype": "constant",
        }
    if not np.isfinite(resampled).all() or resampled.size == 0:
        raise ProtocolError("resampling returned an empty or non-finite waveform")
    peak_after_resample = float(np.max(np.abs(resampled)))
    if peak_after_resample > 1.0 + 1e-7:
        raise ProtocolError("resampling produced out-of-range samples; clipping is forbidden")
    # There is intentionally no normalization, stretching, denoising, or
    # silence trimming here.  quantize_pcm16 performs the registered rint
    # conversion and rejects values outside the int16 range.
    from .audio import quantize_pcm16, waveform_qc

    pcm = quantize_pcm16(resampled, scale=32768.0)
    return pcm, {
        "provider_sample_rate_hz": int(provider_sample_rate),
        "provider_samples": int(values.size),
        "provider_duration_s": duration_s,
        "duration_ratio_to_natural": ratio,
        "canonical_sample_rate_hz": config.SAMPLE_RATE,
        "canonical_samples": int(pcm.size),
        "canonical_duration_s": float(pcm.size / config.SAMPLE_RATE),
        "source_qc": source_qc,
        "canonical_qc": waveform_qc(pcm.astype(np.float64) / 32768.0),
        "resample": resample_meta,
        "normalization": False,
        "time_stretch": False,
        "denoise": False,
        "quantization": "rint(x*32768).astype(<i2)",
    }


@contextlib.contextmanager
def isolated_seed(seed: int):
    """Set provider RNGs for one call and restore every caller state."""

    import torch

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed) & 0xFFFFFFFF)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _provider_config() -> dict[str, Any]:
    return {
        "provider": config.LOCAL_PROVIDER,
        "language": config.LANGUAGE,
        "faster_qwen3": {
            "model_id": config.LOCAL_MODEL,
            "clone_mode": config.LOCAL_CLONE_MODE,
            "x_vector_only": False,
            "append_silence": True,
            "strict_backend": True,
        },
    }


def _load_existing_local_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = verify_self_hashed_json(path) if path.name.endswith(".json") and payload_is_self_hashed(path) else None
    if payload is None:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload.get("results", []))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        raise ProtocolError("local manifest has no row list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            sample_id = str(row.get("sample_id", ""))
            local = row.get("LOCAL")
            if sample_id and isinstance(local, Mapping):
                result[sample_id] = local
            elif sample_id:
                result[sample_id] = row
    return result


def payload_is_self_hashed(path: Path) -> bool:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(payload, dict) and isinstance(payload.get("artifact_sha256"), str)
    except (OSError, ValueError):
        return False


def _local_row_is_compatible(row: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[bool, str]:
    expected_reference = str(record["natural_audio"]["sha256"])
    expected_text_hash = _transcript_hash(str(record["transcript"]))
    expected_model = config.LOCAL_MODEL
    if str(row.get("provider", row.get("backend", ""))) != config.LOCAL_PROVIDER:
        return False, "provider differs"
    if str(row.get("model", row.get("model_id", ""))) != expected_model:
        return False, "model differs"
    if str(row.get("language", "")) != config.LANGUAGE:
        return False, "language differs"
    if str(row.get("reference_audio_sha256", row.get("natural_audio_sha256", ""))) != expected_reference:
        return False, "reference audio differs"
    if str(row.get("transcript_sha256", row.get("tts_transcript_sha256", ""))) != expected_text_hash:
        return False, "transcript differs"
    if str(row.get("actual_backend", "")) != config.LOCAL_PROVIDER:
        return False, "actual backend is missing or differs"
    if str(row.get("provider_implementation_sha256", "")) != file_sha256(config.REPO / "scripts/tts/faster_qwen3.py"):
        return False, "provider implementation binding differs"
    decoding_kwargs = row.get("decoding_kwargs")
    if not isinstance(decoding_kwargs, Mapping) or decoding_kwargs.get("language") != config.LANGUAGE or decoding_kwargs.get("clone_mode") != config.LOCAL_CLONE_MODE or decoding_kwargs.get("strict_backend") is not True:
        return False, "decoding/backend policy differs"
    if not str(row.get("resolved_model_class", "")):
        return False, "resolved model class is missing"
    path_value = row.get("canonical_audio", row.get("canonical_16k_audio", row.get("output")))
    digest = row.get("canonical_audio_sha256", row.get("canonical_16k_audio_sha256", row.get("output_sha256")))
    if not path_value or not digest:
        return False, "canonical binding is missing"
    path = Path(str(path_value))
    if not path.is_file() or file_sha256(path) != str(digest):
        return False, "canonical file is missing or changed"
    raw_path = row.get("raw_waveform")
    raw_digest = row.get("raw_waveform_sha256")
    if not raw_path or not raw_digest:
        return False, "raw provider waveform binding is missing or changed"
    raw_path_obj = Path(str(raw_path))
    assert_not_sealed(path)
    assert_not_sealed(raw_path_obj)
    if not raw_path_obj.is_file() or file_sha256(raw_path_obj) != str(raw_digest):
        return False, "raw provider waveform binding is missing or changed"
    try:
        values, meta = read_pcm16(path)
    except Exception as exc:  # noqa: BLE001 - compatibility audit returns a reason
        return False, f"canonical format invalid: {exc}"
    if values.size <= 0 or meta["file_sha256"] != str(digest):
        return False, "canonical hash mismatch"
    return True, "compatible"


def _cloud_row(record: Mapping[str, Any]) -> dict[str, Any]:
    cloud = record["cloud_tts"]
    canonical = cloud["canonical_audio"]
    provider_audio = cloud["provider_audio"]
    canonical_path = Path(str(canonical["path"]))
    provider_path = Path(str(provider_audio["path"]))
    if file_sha256(canonical_path) != str(canonical["sha256"]):
        raise ProtocolError(f"cloud canonical changed: {record['sample_id']}")
    if file_sha256(provider_path) != str(provider_audio["sha256"]):
        raise ProtocolError(f"cloud provider raw audio changed: {record['sample_id']}")
    values, pcm_meta = read_pcm16(canonical_path)
    if values.size != int(cloud["canonical_samples"]):
        raise ProtocolError(f"cloud canonical sample count changed: {record['sample_id']}")
    if pcm_meta["file_sha256"] != str(canonical["sha256"]):
        raise ProtocolError(f"cloud canonical file hash changed: {record['sample_id']}")
    return {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "provider": config.CLOUD_PROVIDER,
        "model": config.CLOUD_MODEL,
        "language": config.LANGUAGE,
        "reference_audio": record["natural_audio"]["path"],
        "reference_audio_sha256": record["natural_audio"]["sha256"],
        "transcript": str(record["transcript"]),
        "transcript_sha256": _transcript_hash(str(record["transcript"])),
        "provider_audio": provider_audio,
        # Keep the provider-row contract identical to LOCAL: downstream
        # stages consume the audio path as a string and the digest separately.
        "canonical_audio": str(canonical_path.resolve()),
        "canonical_audio_sha256": str(canonical["sha256"]),
        "decoded_pcm_sha256": pcm_meta["decoded_pcm_sha256"],
        "canonical_samples": int(values.size),
        "voice_id": cloud.get("voice_id", ""),
        "new_cloud_call": False,
        "historical_metadata_record_sha256": cloud["metadata_record_sha256"],
        "status": "ok",
    }


def _copy_existing_local(row: Mapping[str, Any], record: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    source_value = row.get("canonical_audio", row.get("canonical_16k_audio", row.get("output")))
    if not source_value:
        raise ProtocolError("existing local row has no canonical audio path")
    source = Path(str(source_value)).resolve()
    digest = str(row.get("canonical_audio_sha256", row.get("canonical_16k_audio_sha256", row.get("output_sha256"))))
    copy_verified(source, destination, digest)
    values, meta = read_pcm16(destination)
    return {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "provider": config.LOCAL_PROVIDER,
        "model": config.LOCAL_MODEL,
        "language": config.LANGUAGE,
        "clone_mode": config.LOCAL_CLONE_MODE,
        "provider_class": row.get("provider_class", ""),
        "resolved_model_class": row.get("resolved_model_class", ""),
        "actual_backend": row.get("actual_backend", config.LOCAL_PROVIDER),
        "provider_implementation_sha256": row.get("provider_implementation_sha256", ""),
        "backend_meta": row.get("backend_meta", {}),
        "reference_audio": record["natural_audio"]["path"],
        "reference_audio_sha256": record["natural_audio"]["sha256"],
        "transcript": str(record["transcript"]),
        "transcript_sha256": _transcript_hash(str(record["transcript"])),
        "canonical_audio": str(destination.resolve()),
        "canonical_audio_sha256": meta["file_sha256"],
        "decoded_pcm_sha256": meta["decoded_pcm_sha256"],
        "canonical_samples": int(values.size),
        "raw_waveform": str(Path(str(row["raw_waveform"])).resolve()),
        "raw_waveform_sha256": str(row["raw_waveform_sha256"]),
        "raw_waveform_dtype": row.get("raw_waveform_dtype"),
        "raw_waveform_samples": row.get("raw_waveform_samples"),
        "raw_waveform_qc": row.get("raw_waveform_qc", {}),
        "seed": row.get("seed"),
        "decoding_kwargs": row.get("decoding_kwargs", {}),
        "generation": "reused_compatible_existing_local_output",
        "status": "ok",
    }


def _generate_local_one(provider: Any, record: Mapping[str, Any], destination: Path, raw_destination: Path) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    seed = record_seed(sample_id)
    text = str(record["transcript"])
    reference = Path(str(record["natural_audio"]["path"]))
    with isolated_seed(seed):
        result = provider.generate_voice_clone(
            text=text,
            ref_audio_path=reference,
            ref_text=text,
            language=config.LANGUAGE,
        )
    provider_sr = int(result.sample_rate)
    raw_values, raw_meta = _as_mono_float(result.audio)
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_destination, raw_values, allow_pickle=False)
    raw_file_hash = file_sha256(raw_destination)
    canonical, canonical_meta = canonicalize_provider_audio(
        result.audio,
        provider_sr,
        int(record["natural_audio"]["sample_count"]),
        int(record["natural_audio"]["sample_count"]) / config.SAMPLE_RATE,
    )
    write_pcm16(destination, canonical)
    _, pcm_meta = read_pcm16(destination)
    backend_meta = dict(getattr(result, "backend_meta", {}) or {})
    if backend_meta.get("backend", config.LOCAL_PROVIDER) != config.LOCAL_PROVIDER:
        raise ProtocolError(f"local provider returned an unexpected backend: {backend_meta.get('backend')}")
    provider_subconfig = dict(getattr(provider, "sub", {}) or {})
    decoding_kwargs = {
        "language": config.LANGUAGE,
        "clone_mode": config.LOCAL_CLONE_MODE,
        "x_vector_only": bool(provider_subconfig.get("x_vector_only", False)),
        "append_silence": bool(provider_subconfig.get("append_silence", True)),
        "strict_backend": bool(getattr(provider, "strict_backend", False)),
    }
    if decoding_kwargs["strict_backend"] is not True:
        raise ProtocolError("local TTS provider is not running with strict backend binding")
    if provider_subconfig.get("max_new_tokens") is not None:
        decoding_kwargs["max_new_tokens"] = int(provider_subconfig["max_new_tokens"])
    resolved_model = getattr(getattr(provider, "_model", None), "__class__", type(None))
    return {
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "provider": config.LOCAL_PROVIDER,
        "model": config.LOCAL_MODEL,
        "language": config.LANGUAGE,
        "clone_mode": config.LOCAL_CLONE_MODE,
        "provider_class": f"{provider.__class__.__module__}.{provider.__class__.__name__}",
        "resolved_model_class": f"{resolved_model.__module__}.{resolved_model.__name__}",
        "actual_backend": backend_meta.get("backend", config.LOCAL_PROVIDER),
        "provider_implementation_sha256": file_sha256(config.REPO / "scripts/tts/faster_qwen3.py"),
        "backend_meta": backend_meta,
        "reference_audio": str(reference.resolve()),
        "reference_audio_sha256": str(record["natural_audio"]["sha256"]),
        "transcript": text,
        "transcript_sha256": _transcript_hash(text),
        "seed": seed,
        "decoding_kwargs": decoding_kwargs,
        "raw_waveform": str(raw_destination.resolve()),
        "raw_waveform_sha256": raw_file_hash,
        "raw_waveform_dtype": str(raw_values.dtype),
        "raw_waveform_samples": int(raw_values.size),
        "raw_waveform_qc": raw_meta,
        "provider_sample_rate_hz": provider_sr,
        "canonical_audio": str(destination.resolve()),
        "canonical_audio_sha256": pcm_meta["file_sha256"],
        "decoded_pcm_sha256": pcm_meta["decoded_pcm_sha256"],
        "canonical_samples": int(canonical.size),
        "canonicalization": canonical_meta,
        "new_cloud_call": False,
        "generation": "single_registered_local_synthesis",
        "status": "ok",
    }


def _read_current_manifest(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    payload = verify_self_hashed_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ProtocolError("existing TTS manifest has no rows")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sample_id = str(row.get("sample_id", ""))
        local = row.get("LOCAL")
        if sample_id and isinstance(local, Mapping):
            result[sample_id] = local
    return result


def _current_row_usable(row: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    ok, _ = _local_row_is_compatible(row, record)
    return ok and row.get("generation") in {"single_registered_local_synthesis", "reused_compatible_existing_local_output"}


def run_tts_stage(
    run_root: Path,
    cohort: Mapping[str, Any],
    setup: Mapping[str, Any],
    *,
    local_manifest: Path | None = None,
) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    paths.tts.mkdir(parents=True, exist_ok=True)
    if cohort.get("status") != "complete" or len(cohort.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete before TTS stage")
    setup_sha256 = file_sha256(paths.protocol / "setup.json")
    prior = _read_current_manifest(paths.tts / "tts_manifest.json")
    external = _load_existing_local_rows(local_manifest) if local_manifest else {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cloud_refs: list[dict[str, Any]] = []
    provider = None
    generated_count = 0
    started = time.monotonic()
    gpu_held = False
    try:
        from scripts.tts.factory import get_tts_provider

        # Provider creation is kept inside the lease.  Importing the factory
        # itself is cheap, while model construction can allocate the card.
        with gpu_lease("local_tts"):
            gpu_held = True
            provider = get_tts_provider(_provider_config(), run_id=run_root.name, repo_root=config.REPO, env=dict(os.environ))
            for index, record in enumerate(cohort["records"], 1):
                sample_id = str(record["sample_id"])
                try:
                    destination = paths.tts / "local" / f"{sample_id}.wav"
                    raw_destination = paths.tts / "local_raw" / f"{sample_id}.npy"
                    current = prior.get(sample_id)
                    if current is not None and _current_row_usable(current, record):
                        row = dict(current)
                        row["resumed"] = True
                    elif sample_id in external:
                        compatible, reason = _local_row_is_compatible(external[sample_id], record)
                        if not compatible:
                            raise ProtocolError(f"external local output rejected: {reason}")
                        row = _copy_existing_local(external[sample_id], record, destination)
                        row["resumed"] = False
                    else:
                        if current is not None:
                            raise ProtocolError("existing local row is incomplete or has changed; refusing regeneration")
                        row = _generate_local_one(provider, record, destination, raw_destination)
                        generated_count += 1
                    if str(row.get("reference_audio_sha256")) != str(record["natural_audio"]["sha256"]):
                        raise ProtocolError("local reference identity changed")
                    rows.append(row)
                    print(f"TTS {index}/{config.EXPECTED_RECORD_COUNT} LOCAL {sample_id}", flush=True)
                except Exception as exc:  # noqa: BLE001 - preserve each failed synthesis
                    failures.append({"sample_id": sample_id, "provider": "LOCAL", "error_type": type(exc).__name__, "error": str(exc), "seed": record_seed(sample_id)})
    except Exception as exc:  # noqa: BLE001 - preserve stage failure in manifest
        failures.append({"sample_id": None, "provider": "LOCAL", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if gpu_held:
            release_torch_memory()
    # Cloud validation happens outside the GPU lease and never constructs a
    # cloud provider or makes a network request.
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        try:
            cloud = _cloud_row(record)
            cloud_refs.append(cloud)
        except Exception as exc:  # noqa: BLE001 - preserve cloud audit failure per record
            failures.append({"sample_id": sample_id, "provider": "CLOUD", "error_type": type(exc).__name__, "error": str(exc), "new_cloud_call": False})
    by_id = {str(row["sample_id"]): row for row in rows}
    cloud_by_id = {str(row["sample_id"]): row for row in cloud_refs}
    paired_rows = [
        {"sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]), "LOCAL": by_id.get(str(record["sample_id"])), "CLOUD": cloud_by_id.get(str(record["sample_id"]))}
        for record in cohort["records"]
    ]
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT and len(cloud_refs) == config.EXPECTED_RECORD_COUNT
    payload = {
        "schema_version": 1,
        "stage_id": "01_tts",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": setup_sha256,
        "status": "complete" if complete else "blocked",
        "record_count": len(paired_rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "local_generated_success_count": generated_count,
        "local_success_budget": config.EXPECTED_RECORD_COUNT,
        "cloud_new_call_count": 0,
        "cloud_new_call_budget": 0,
        "providers": {"LOCAL": {"provider": config.LOCAL_PROVIDER, "model": config.LOCAL_MODEL}, "CLOUD": {"provider": config.CLOUD_PROVIDER, "model": config.CLOUD_MODEL}},
        "rows": paired_rows,
        "failures": failures,
        "elapsed_s": time.monotonic() - started,
        "selection_policy": "no quality, duration, alignment, or score based selection",
    }
    write_self_hashed_json(paths.tts / "tts_manifest.json", payload)
    write_self_hashed_json(paths.tts / "cloud_refs.json", {"schema_version": 1, "stage_id": "01_tts", "protocol_id": config.PROTOCOL_ID, "status": "complete" if len(cloud_refs) == config.EXPECTED_RECORD_COUNT else "blocked", "new_cloud_call_count": 0, "rows": cloud_refs, "failures": [row for row in failures if row.get("provider") == "CLOUD"]})
    write_self_hashed_json(paths.tts / "failures.json", {"schema_version": 1, "stage_id": "01_tts", "protocol_id": config.PROTOCOL_ID, "status": payload["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"TTS stage incomplete: local={len(rows)}/{config.EXPECTED_RECORD_COUNT}, cloud={len(cloud_refs)}/{config.EXPECTED_RECORD_COUNT}")
    return payload


__all__ = [
    "canonicalize_provider_audio",
    "isolated_seed",
    "run_tts_stage",
]
