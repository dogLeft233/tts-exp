"""I/O, frozen SSL extraction, and optional MFA subprocess helpers.

Heavy dependencies are imported inside the functions that need them.  This
keeps the audit and the pure unit tests CPU-only and makes it impossible for a
help/audit invocation to load a GPU model accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .lrs3_phone_rules_metrics import (
    SILENCE_LABELS,
    derive_frame_times,
    is_speech_label,
    normalize_phone,
)


class WorkerError(RuntimeError):
    """Raised for deterministic worker/input failures."""


_INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:?[\s\r\n]*"
    r"xmin\s*=\s*([0-9.eE+\-]+)\s*"
    r"xmax\s*=\s*([0-9.eE+\-]+)\s*"
    r"text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return sha256_file(target)


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise WorkerError(f"non-finite JSON constant {value}: {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_pcm16(path: str | Path, *, sample_rate: int = 16_000) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            payload = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise WorkerError(f"cannot read WAV: {source}") from exc
    if channels != 1 or width != 2 or rate != sample_rate:
        raise WorkerError(f"audio is not mono {sample_rate} Hz PCM16: {source}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != count or values.size == 0:
        raise WorkerError(f"invalid PCM16 sample count: {source}")
    return values, {
        "path": str(source.resolve()),
        "sample_rate": int(rate),
        "channels": int(channels),
        "sample_width": int(width),
        "sample_count": int(values.size),
        "duration_s": float(values.size / rate),
        "container_sha256": sha256_file(source),
        "pcm_sha256": sha256_bytes(values.astype("<i2", copy=False).tobytes()),
    }


def write_pcm16(path: str | Path, values: np.ndarray, *, sample_rate: int = 16_000) -> dict[str, Any]:
    target = Path(path)
    pcm = np.asarray(values)
    if pcm.ndim != 1 or pcm.dtype != np.int16 or pcm.size == 0:
        raise WorkerError("output must be a non-empty one-dimensional int16 array")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype("<i2", copy=False).tobytes())
    temporary.replace(target)
    return {
        "path": str(target.resolve()),
        "sample_rate": int(sample_rate),
        "channels": 1,
        "sample_width": 2,
        "sample_count": int(pcm.size),
        "container_sha256": sha256_file(target),
        "pcm_sha256": sha256_bytes(pcm.astype("<i2", copy=False).tobytes()),
    }


def parse_textgrid(path: str | Path) -> list[dict[str, Any]]:
    """Parse the phones tier while retaining silence and all IPA symbols."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerError(f"cannot read TextGrid: {source}") from exc
    marker = 'name = "phones"'
    if marker not in raw:
        raise WorkerError(f"phones tier missing: {source}")
    start = raw.index(marker)
    next_item = raw.find("item [", start + len(marker))
    section = raw[start:] if next_item < 0 else raw[start:next_item]
    tokens: list[dict[str, Any]] = []
    for match in _INTERVAL_RE.finditer(section):
        begin = float(match.group(1))
        end = float(match.group(2))
        label = match.group(3).strip()
        if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
            raise WorkerError(f"invalid TextGrid interval in {source}")
        normalized = normalize_phone(label)
        tokens.append({
            "token": label,
            "label": normalized,
            "start_s": begin,
            "end_s": end,
            "duration_s": end - begin,
            "silence": not is_speech_label(normalized, SILENCE_LABELS),
            "speech": is_speech_label(normalized, SILENCE_LABELS),
        })
    if not tokens:
        raise WorkerError(f"phones tier has no intervals: {source}")
    previous_end = -math.inf
    for index, token in enumerate(tokens):
        begin = float(token["start_s"])
        if begin < previous_end - 1e-7:
            raise WorkerError(f"overlapping TextGrid intervals at {source}:{index}")
        previous_end = float(token["end_s"])
    if not any(bool(token["speech"]) for token in tokens):
        raise WorkerError(f"TextGrid contains no speech token: {source}")
    return tokens


def token_signature(tokens: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "label": normalize_phone(token.get("label", token.get("token", ""))),
            "start_s": float(token["start_s"]),
            "end_s": float(token["end_s"]),
            "speech": bool(token.get("speech", not token.get("silence", False))),
        }
        for token in tokens
    ]
    return canonical_json_sha256(canonical)


def frontend_config_from_model(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", model)
    kernels = getattr(config, "conv_kernel", None)
    strides = getattr(config, "conv_stride", None)
    if kernels is None or strides is None:
        raise WorkerError("model config lacks conv_kernel/conv_stride")
    dilation = getattr(config, "conv_dilation", None)
    padding = getattr(config, "conv_padding", None)
    if padding is None:
        padding = [0] * len(kernels)
    return {
        "conv_kernel": [int(x) for x in kernels],
        "conv_stride": [int(x) for x in strides],
        "conv_dilation": [int(x) for x in (dilation or [1] * len(kernels))],
        "conv_padding": [int(x) for x in padding],
    }


def processor_fingerprint(processor: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"class": type(processor).__name__}
    for key in ("do_normalize", "return_attention_mask", "sampling_rate", "feature_size"):
        if hasattr(processor, key):
            value = getattr(processor, key)
            if isinstance(value, (str, int, float, bool, type(None))):
                data[key] = value
    to_dict = getattr(processor, "to_dict", None)
    if callable(to_dict):
        try:
            data["config"] = to_dict()
        except (AttributeError, TypeError, ValueError, RuntimeError):  # pragma: no cover - third-party processor oddity
            data["config"] = "unavailable"
    return data


def load_ssl_bundle(
    model_name: str,
    *,
    processor_name: str | None = None,
    revision: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load exactly one local frozen model and its processor."""

    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - dependency-gated environment
        raise WorkerError("torch and transformers are required for SSL extraction") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError(f"requested device is unavailable: {device}")
    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if revision is not None:
        kwargs["revision"] = revision
    processor_id = processor_name or model_name
    try:
        processor = AutoProcessor.from_pretrained(processor_id, **kwargs)
    except (OSError, ValueError, RuntimeError, KeyError):
        try:
            processor = AutoFeatureExtractor.from_pretrained(processor_id, **kwargs)
        except (OSError, ValueError, RuntimeError, KeyError) as second_error:
            raise WorkerError(f"cannot load local processor {processor_id}") from second_error
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, **kwargs)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frontend = frontend_config_from_model(model)
    return {
        "model_name": model_name,
        "processor_name": processor_id,
        "revision": revision,
        "device": device,
        "model": model,
        "processor": processor,
        "frontend": frontend,
        "processor_fingerprint": processor_fingerprint(processor),
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
    }


def extract_features(
    model: Any,
    processor: Any,
    audio: np.ndarray,
    sample_rate: int,
    layers: Sequence[int],
    *,
    device: str = "cpu",
    frontend: Mapping[str, Any] | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """Run one complete audio through a frozen processor/model pair."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise WorkerError("torch is required for SSL extraction") from exc
    if sample_rate <= 0:
        raise WorkerError("sample_rate must be positive")
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
        raise WorkerError("audio must be a finite mono waveform")
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=False)
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value
    with torch.no_grad():
        try:
            outputs = model(**model_inputs, output_hidden_states=True)
        except TypeError:
            outputs = model(**model_inputs)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise WorkerError("model did not return hidden_states")
    requested = [int(layer) for layer in layers]
    if any(layer < 0 or layer >= len(hidden_states) for layer in requested):
        raise WorkerError(f"requested layer outside hidden_states: {requested}, count={len(hidden_states)}")
    selected: dict[int, np.ndarray] = {}
    frame_count = None
    for layer in requested:
        tensor = hidden_states[layer]
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise WorkerError(f"unexpected hidden state shape at layer {layer}: {tuple(tensor.shape)}")
        value = tensor[0].detach().cpu().numpy().astype(np.float32)
        if frame_count is None:
            frame_count = value.shape[0]
        if value.shape[0] != frame_count:
            raise WorkerError("hidden-state frame counts disagree")
        if not np.all(np.isfinite(value)):
            raise WorkerError(f"non-finite hidden state at layer {layer}")
        selected[layer] = value
    if frame_count is None:
        raise WorkerError("no layers requested")
    frontend_data = dict(frontend or frontend_config_from_model(model))
    frame_times = derive_frame_times(
        frame_count,
        sample_rate,
        frontend_data["conv_kernel"],
        frontend_data["conv_stride"],
        frontend_data.get("conv_dilation"),
        frontend_data.get("conv_padding"),
    )
    return selected, frame_times, {
        "frame_count": int(frame_count),
        "layers": requested,
        "sample_rate": int(sample_rate),
        "frame_time_convention": "frontend_receptive_field_centers",
        "frontend": frontend_data,
        "processor": processor_fingerprint(processor),
    }


def build_mfa_command(
    corpus_dir: str | Path,
    dictionary: str | Path,
    acoustic_model: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "mfa",
    clean: bool = True,
    overwrite: bool = True,
) -> list[str]:
    command = [
        executable,
        "align",
        str(Path(corpus_dir)),
        str(Path(dictionary)),
        str(Path(acoustic_model)),
        str(Path(output_dir)),
    ]
    if clean:
        command.append("--clean")
    if overwrite:
        command.append("--overwrite")
    return command


def realign_audio(
    records: Sequence[Mapping[str, Any]],
    *,
    audio_key: str,
    output_dir: str | Path,
    mfa_config: Mapping[str, Any],
    repo_root: str | Path,
    execute: bool = True,
) -> dict[str, Any]:
    """Materialize a side-specific MFA corpus and optionally run MFA.

    The caller supplies a frozen record list and a text field.  This function
    never infers a different side by position and never reuses another arm's
    alignment.
    """

    root = Path(output_dir)
    corpus = root / "corpus"
    aligned = root / "aligned"
    corpus.mkdir(parents=True, exist_ok=True)
    for record in records:
        sample_id = str(record["sample_id"])
        audio_path = Path(record[audio_key])
        text = str(record.get("mfa_transcript", record.get("transcript", ""))).strip()
        if not audio_path.is_file() or not text:
            raise WorkerError(f"MFA input is incomplete for {sample_id}")
        destination_audio = corpus / f"{sample_id}.wav"
        destination_text = corpus / f"{sample_id}.lab"
        destination_audio.write_bytes(audio_path.read_bytes())
        destination_text.write_text(text + "\n", encoding="utf-8")
    command = build_mfa_command(
        corpus,
        mfa_config["dictionary"],
        mfa_config["acoustic_model"],
        aligned,
        executable=str(mfa_config.get("executable", "mfa")),
        clean=True,
        overwrite=True,
    )
    log_path = root / "mfa.log"
    if execute:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=str(Path(repo_root)),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise WorkerError(f"MFA failed with exit {completed.returncode}: {log_path}")
    return {
        "corpus_dir": str(corpus),
        "aligned_dir": str(aligned),
        "command": command,
        "log_path": str(log_path),
        "executed": bool(execute),
        "sample_count": len(records),
    }


__all__ = [
    "WorkerError",
    "build_mfa_command",
    "canonical_json_sha256",
    "extract_features",
    "frontend_config_from_model",
    "load_ssl_bundle",
    "parse_textgrid",
    "processor_fingerprint",
    "read_json",
    "read_pcm16",
    "realign_audio",
    "sha256_file",
    "token_signature",
    "write_json_atomic",
    "write_pcm16",
]
