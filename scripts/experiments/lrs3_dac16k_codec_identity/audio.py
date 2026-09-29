from __future__ import annotations

import importlib
import math
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config
from .common import file_sha256, verify_self_hashed_json, write_self_hashed_json
from .protocol import ProtocolError


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        payload = handle.readframes(sample_count)
    if channels != 1 or sample_width != 2 or sample_rate != config.SAMPLE_RATE:
        raise ProtocolError(f"audio format violates PCM16/16k mono contract: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or values.size < 1024:
        raise ProtocolError(f"audio sample count is invalid: {path}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "pcm_sha256": file_sha256(path),
    }


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return values.astype(np.float32) / 32768.0


def float_to_pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError("waveform contains non-finite values")
    if values.size == 0 or float(np.abs(values).max()) >= 1.0:
        raise ValueError("waveform is clipped or empty")
    return np.rint(values.astype(np.float64) * 32768.0).clip(-32768, 32767).astype("<i2")


def write_pcm16(path: Path, values: np.ndarray) -> str:
    pcm = float_to_pcm16(values) if np.asarray(values).dtype != np.int16 else np.asarray(values, dtype="<i2").reshape(-1)
    if pcm.size == 0:
        raise ValueError("cannot write empty PCM")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)
    return file_sha256(path)


def load_dac_model(source_root: Path, checkpoint: Path, device: str) -> Any:
    if not (source_root / "dac" / "__init__.py").is_file():
        raise FileNotFoundError(f"DAC source package is missing: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    dac_utils = importlib.import_module("dac.utils")
    model = dac_utils.load_model(
        tag=config.DAC_TAG,
        load_path=str(checkpoint.parent.parent),
        model_type=config.DAC_MODEL_TYPE,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if int(getattr(model, "sample_rate", -1)) != config.DAC_SAMPLE_RATE:
        raise ProtocolError("DAC model sample-rate binding differs")
    if int(getattr(model, "hop_length", -1)) != config.DAC_HOP_LENGTH:
        raise ProtocolError("DAC model hop-length binding differs")
    if bool(model.training) or any(parameter.requires_grad for parameter in model.parameters()):
        raise ProtocolError("DAC model must be frozen and in eval mode")
    if int(getattr(model, "n_codebooks", 0)) <= 0:
        raise ProtocolError("DAC model has no quantizers")
    return model


def _model_forward(model: Any, waveform: np.ndarray, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model_input_count = int(math.ceil(waveform.size / config.DAC_HOP_LENGTH) + 1) * config.DAC_HOP_LENGTH
    padded_waveform = np.pad(waveform, (0, model_input_count - waveform.size))
    input_tensor = torch.from_numpy(pcm16_to_float(padded_waveform)).to(device=device, dtype=torch.float32).view(1, 1, -1)
    expected_padded = model_input_count
    with torch.inference_mode():
        output = model(input_tensor, sample_rate=config.DAC_SAMPLE_RATE, n_quantizers=None)
    if not isinstance(output, Mapping) or "audio" not in output:
        raise ProtocolError("DAC forward did not return an audio mapping")
    decoded = output["audio"].detach().float().cpu().numpy().reshape(-1)
    raw_count = int(decoded.size)
    if raw_count < waveform.size:
        raise ProtocolError(f"DAC output is shorter than input: {raw_count} < {waveform.size}")
    decoded = decoded[: waveform.size].copy()
    if not np.isfinite(decoded).all() or float(np.abs(decoded).max()) >= 1.0:
        raise ProtocolError("DAC decoded waveform failed finite/clipping QC")
    codes = output.get("codes")
    latents = output.get("latents")
    z = output.get("z")
    provenance = {
        "input_sample_count": int(waveform.size),
        "model_internal_padded_sample_count": expected_padded,
        "raw_decoder_sample_count": raw_count,
        "decoded_crop": {"action": "right_crop_only", "sample_count": int(waveform.size)},
        "code_shape": list(codes.shape) if isinstance(codes, torch.Tensor) else None,
        "latent_shape": list(latents.shape) if isinstance(latents, torch.Tensor) else None,
        "quantized_latent_shape": list(z.shape) if isinstance(z, torch.Tensor) else None,
        "quantizer_count": int(codes.shape[1]) if isinstance(codes, torch.Tensor) and codes.ndim >= 2 else None,
        "code_frame_count": int(codes.shape[-1]) if isinstance(codes, torch.Tensor) else None,
        "n_quantizers": None,
    }
    return {"waveform": decoded, "provenance": provenance}, {"raw_peak": float(np.abs(decoded).max()), "raw_rms": float(np.sqrt(np.mean(decoded ** 2))), "raw_finite": True}


def reconstruct_record(model: Any, record: Mapping[str, Any], output_dir: Path, device: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    natural_path = Path(str(record["natural_audio"]))
    natural, natural_meta = read_pcm16(natural_path)
    if natural_meta["pcm_sha256"] != record["natural_audio_sha256"] or natural.size != int(record["natural_sample_count"]):
        raise ProtocolError(f"natural audio binding changed: {sample_id}")
    output_path = output_dir / f"{sample_id}__dac.wav"
    sidecar_path = output_path.with_suffix(".json")
    if output_path.is_file() and sidecar_path.is_file():
        sidecar = verify_self_hashed_json(sidecar_path)
        if sidecar.get("protocol_id") != config.PROTOCOL_ID or sidecar.get("model_binding_sha256") != binding["protocol_sha256"] or sidecar.get("source_audio_sha256") != record["natural_audio_sha256"] or sidecar.get("output_sha256") != file_sha256(output_path):
            raise ProtocolError(f"existing DAC sidecar is incompatible: {sample_id}")
        return sidecar
    if output_path.exists() != sidecar_path.exists():
        raise ProtocolError(f"partial DAC output cannot be resumed: {sample_id}")
    result, raw_qc = _model_forward(model, natural, device)
    output_sha256 = write_pcm16(output_path, result["waveform"])
    _, output_meta = read_pcm16(output_path)
    if output_meta["sample_count"] != natural.size or output_meta["pcm_sha256"] != output_sha256:
        raise ProtocolError(f"DAC output format validation failed: {sample_id}")
    sidecar = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage_id": "01_audio",
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "source_audio": str(natural_path),
        "source_audio_sha256": record["natural_audio_sha256"],
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "model_binding_sha256": binding["protocol_sha256"],
        "config": {"sample_rate": config.DAC_SAMPLE_RATE, "model_type": config.DAC_MODEL_TYPE, "bitrate": config.DAC_BITRATE, "n_quantizers": None, "model_input_padding": "right_pad_to_one_hop_beyond_next_multiple_before_forward", "postprocess": "crop_only_then_one_pcm16_canonicalization"},
        "input": natural_meta,
        "inference": result["provenance"],
        "raw_output_qc": raw_qc,
        "output_qc": {**output_meta, "finite": True, "clipped_sample_count": int(np.sum(np.abs(result["waveform"]) >= 1.0))},
    }
    write_self_hashed_json(sidecar_path, sidecar)
    return sidecar


def run_stage01(cohort: Mapping[str, Any], protocol: Mapping[str, Any], output_stage: Path, device: str) -> dict[str, Any]:
    checkpoint = Path(str(protocol["checkpoint"]["path"]))
    if file_sha256(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise ProtocolError("DAC checkpoint changed after Stage 00")
    source_root = Path(str(protocol["source"]["path"]))
    if not source_root.is_dir() or source_root.resolve() != config.DAC_SOURCE_ROOT.resolve():
        raise ProtocolError("DAC source root changed after Stage 00")
    model = load_dac_model(source_root, checkpoint, device)
    protocol_binding = {"protocol_sha256": file_sha256(config.STAGES["00_protocol"] / "protocol.json")}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_dir = output_stage / "audio"
    for index, record in enumerate(cohort["records"], 1):
        try:
            row = reconstruct_record(model, record, output_dir, device, protocol_binding)
            rows.append({**record, "dac": row})
            print(f"DAC {index}/{len(cohort['records'])} {record['sample_id']}", flush=True)
        except (ImportError, KeyError, OSError, ProtocolError, RuntimeError, TypeError, ValueError, FloatingPointError) as exc:
            failures.append({"sample_id": str(record["sample_id"]), "error_type": type(exc).__name__, "error": str(exc)})
            print(f"FAIL {index}/{len(cohort['records'])} {record['sample_id']}: {exc}", flush=True)
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT
    manifest = {
        "schema_version": 1,
        "stage_id": "01_audio",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "rows": rows,
        "failures": failures,
        "protocol_sha256": protocol_binding["protocol_sha256"],
    }
    write_self_hashed_json(output_stage / "audio_manifest.json", manifest)
    write_self_hashed_json(output_stage / "failures.json", {"schema_version": 1, "stage_id": "01_audio", "status": "complete" if complete else "blocked", "failures": failures})
    if not complete:
        raise ProtocolError(f"DAC reconstruction incomplete: {len(rows)}/{config.EXPECTED_RECORD_COUNT}")
    return manifest
