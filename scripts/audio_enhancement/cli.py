"""Command-line waveform conversion for identity, RMS, profile, and learned head."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import read_canonical_wav, write_float32_wav
from .baselines import waveform_rms
from .pipeline import transform


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"method": "identity"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration JSON must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_CHECKPOINT_SCHEMA = "cross_tfg_audio_head_checkpoint_v1"
_HEAD_CONFIG_KEYS = (
    "n_fft",
    "hop_length",
    "hidden_channels",
    "max_residual_db",
    "level_normalize",
    "frequency_center_mask",
    "frame_energy_normalize",
)


def _load_checkpoint_payload(checkpoint: Path, device: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("learned-head CLI mode requires PyTorch") from exc
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a tensor state-dict mapping or a fit checkpoint object")
    return payload


def _fit_checkpoint_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("schema") != _CHECKPOINT_SCHEMA:
        return None
    architecture = payload.get("architecture")
    transform_config = payload.get("transform_config")
    if not isinstance(architecture, dict) or not isinstance(transform_config, dict):
        raise ValueError("fit checkpoint is missing its sealed architecture/transform configuration")
    head_config = {key: architecture[key] for key in _HEAD_CONFIG_KEYS if key in architecture}
    if set(head_config) != set(_HEAD_CONFIG_KEYS):
        raise ValueError("fit checkpoint architecture is incomplete for inference")
    if transform_config.get("method") != "spectral_head":
        raise ValueError("fit checkpoint does not describe a spectral-head deployment transform")
    if int(transform_config.get("sample_rate_hz", -1)) != 16_000:
        raise ValueError("fit checkpoint transform is not bound to 16 kHz input")
    if transform_config.get("head_architecture") != architecture:
        raise ValueError("fit checkpoint transform and head architecture bindings disagree")
    for key in ("plan_raw_sha256", "split_manifest_sha256"):
        digest = payload.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ValueError(f"fit checkpoint is missing a valid {key}")
    if int(payload.get("completed_steps", 0)) <= 0:
        raise ValueError("fit checkpoint has no completed optimizer steps")
    result = dict(transform_config)
    result["head"] = head_config
    result["checkpoint_training_metadata"] = {
        "plan_path": payload.get("plan_path"),
        "plan_raw_sha256": payload.get("plan_raw_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "fit_model_id": payload.get("fit_model_id"),
        "seed": payload.get("seed"),
        "completed_steps": payload.get("completed_steps"),
    }
    return result


def _resolve_checkpoint_config(
    requested: dict[str, Any], embedded: dict[str, Any] | None, *, config_was_explicit: bool
) -> dict[str, Any]:
    if embedded is None:
        if not config_was_explicit:
            raise ValueError("a bare state-dict checkpoint requires an explicit --config with head architecture")
        return requested
    if not config_was_explicit:
        return embedded
    if requested.get("method") != "spectral_head":
        raise ValueError("a fit checkpoint can only be applied with method=spectral_head")
    embedded_head = embedded["head"]
    requested_head = requested.get("head", {})
    if not isinstance(requested_head, dict):
        raise ValueError("config.head must be an object")
    for key, value in requested_head.items():
        if key not in embedded_head or value != embedded_head[key]:
            raise ValueError(f"explicit head config conflicts with fit checkpoint metadata: {key}")
    for key, value in embedded.items():
        if key in {"head", "checkpoint_training_metadata"}:
            continue
        if key in requested and requested[key] != value:
            raise ValueError(f"explicit transform config conflicts with fit checkpoint metadata: {key}")
    return embedded


def _load_head(config: dict[str, Any], checkpoint_payload: dict[str, Any], device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("learned-head CLI mode requires PyTorch") from exc
    from .heads import SpectralResidualHead

    head_config = config.get("head", {})
    if not isinstance(head_config, dict):
        raise ValueError("config.head must be an object")
    head = SpectralResidualHead(
        n_fft=int(head_config.get("n_fft", 512)),
        hop_length=int(head_config.get("hop_length", 128)),
        hidden_channels=int(head_config.get("hidden_channels", 24)),
        max_residual_db=float(head_config.get("max_residual_db", 3.0)),
        level_normalize=bool(head_config.get("level_normalize", True)),
        frequency_center_mask=bool(head_config.get("frequency_center_mask", False)),
        frame_energy_normalize=bool(head_config.get("frame_energy_normalize", False)),
    ).to(device)
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain a tensor state-dict mapping")
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    return head


def run(
    input_path: Path,
    output_path: Path,
    *,
    config_path: Path | None = None,
    natural_reference_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
    generated_video: Path | None = None,
    mux_original_wav: Path | None = None,
    mux_output: Path | None = None,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    if natural_reference_path is None:
        natural_reference_path = output_path.with_name(f"{output_path.stem}.natural_16k_f32.wav")
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    natural_resolved = natural_reference_path.resolve()
    mux_original_resolved = (mux_original_wav or input_path).resolve()
    if input_resolved == output_resolved:
        raise ValueError("enhanced output must not overwrite its source audio")
    if natural_resolved in {input_resolved, output_resolved}:
        raise ValueError("natural reference, source, and enhanced output need distinct paths")
    if mux_output is not None and mux_output.resolve() in {
        input_resolved,
        output_resolved,
        natural_resolved,
        mux_original_resolved,
    }:
        raise ValueError("mux output must be distinct from audio source/output paths")
    if mux_original_wav is not None and mux_original_resolved in {output_resolved, natural_resolved}:
        raise ValueError("mux original WAV must remain distinct from generated audio outputs")
    if not overwrite:
        planned_outputs = [output_path, natural_reference_path]
        if mux_output is not None:
            planned_outputs.append(mux_output)
        existing = [str(path) for path in planned_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to replace existing outputs: {existing}")
    config = _load_config(config_path)
    checkpoint_payload = None
    if checkpoint_path is not None:
        checkpoint_payload = _load_checkpoint_payload(checkpoint_path, device)
        config = _resolve_checkpoint_config(
            config,
            _fit_checkpoint_config(checkpoint_payload),
            config_was_explicit=config_path is not None,
        )
    method = str(config.get("method", "identity"))
    source, read_info = read_canonical_wav(input_path)
    natural_write_info = write_float32_wav(
        natural_reference_path,
        source,
        sample_rate=read_info.sample_rate,
        peak_ceiling=None,
        overwrite=overwrite,
    )
    head = None
    if method == "spectral_head":
        if checkpoint_payload is None:
            raise ValueError("spectral_head mode requires --checkpoint")
        head = _load_head(config, checkpoint_payload, device)
    enhanced = transform(source, config, head=head)
    write_info = write_float32_wav(
        output_path,
        enhanced,
        sample_rate=read_info.sample_rate,
        peak_ceiling=config.get("peak_ceiling"),
        overwrite=overwrite,
    )
    result: dict[str, Any] = {
        "status": "PASS",
        "method": method,
        "config_sha256": _sha256_file(config_path) if config_path is not None else None,
        "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path is not None else None,
        "cli_source_sha256": _sha256_file(Path(__file__).resolve()),
        "application_config": config,
        "embedded_checkpoint_training_metadata": config.get("checkpoint_training_metadata"),
        "source_pcm": {
            "path": read_info.path,
            "file_sha256": read_info.file_sha256,
            "sample_rate_hz": read_info.source_sample_rate,
            "channels": read_info.source_channels,
            "sample_count": read_info.source_frames,
            "source_encoding": read_info.source_encoding,
        },
        "natural_pcm": {
            "path": natural_write_info.path,
            "file_sha256": natural_write_info.file_sha256,
            "pcm_sha256": natural_write_info.pcm_float32_sha256,
            "sample_rate_hz": read_info.sample_rate,
            "channels": 1,
            "sample_count": read_info.frames,
        },
        "enhanced_pcm": {
            "path": write_info.path,
            "file_sha256": write_info.file_sha256,
            "pcm_sha256": write_info.pcm_float32_sha256,
            "sample_rate_hz": write_info.sample_rate,
            "channels": write_info.channels,
            "sample_count": write_info.frames,
            "subtype": write_info.subtype,
            "peak_gain": write_info.applied_peak_gain,
            "peak_after_limit": write_info.peak_after_limit,
        },
        "same_sample_count": read_info.frames == write_info.frames,
        "clock_contract": {
            "input_sample_rate_hz": int(read_info.sample_rate),
            "output_sample_rate_hz": int(write_info.sample_rate),
            "input_sample_count": int(read_info.frames),
            "output_sample_count": int(write_info.frames),
            "same_sample_count": read_info.frames == write_info.frames,
        },
        "output_finite": bool(np.isfinite(enhanced).all()),
        "rms_diagnostics": {
            "natural_input_rms": waveform_rms(source),
            "enhanced_output_rms": waveform_rms(enhanced),
            "configured_post_rms_match": config.get("post_rms_match", "none"),
            "fixed_positive_control_target_rms": config.get("target_rms", 0.0376838172675746),
            "final_writer_peak_gain": write_info.applied_peak_gain,
        },
    }
    if generated_video is not None or mux_output is not None:
        if generated_video is None or mux_output is None:
            raise ValueError("--generated-video and --mux-output must be supplied together")
        from .overdub import mux_original_pcm

        mux_info = mux_original_pcm(
            generated_video,
            mux_original_resolved,
            mux_output,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            overwrite=overwrite,
        )
        result["mux"] = {
            "source_audio_path": str(mux_original_resolved),
            "source_audio_file_sha256": _sha256_file(mux_original_resolved),
            "canonical_16khz_reference_path": str(natural_reference_path.resolve()),
            "canonical_16khz_reference_file_sha256": natural_write_info.file_sha256,
            "output_path": mux_info.output_path,
            "output_file_sha256": mux_info.output_file_sha256,
            "video_codec": mux_info.video_codec,
            "video_stream_sha256": mux_info.video_stream_sha256,
            "video_frame_count": mux_info.video_frame_count,
            "video_timestamps_sha256": mux_info.video_timestamps_sha256,
            "audio_codec": mux_info.audio_codec,
            "audio_sample_rate_hz": mux_info.audio_sample_rate,
            "audio_channels": mux_info.audio_channels,
            "audio_sample_count": mux_info.decoded_audio_sample_count,
            "audio_float32_sha256": mux_info.decoded_audio_float32_sha256,
            "audio_matches_original_pcm": mux_info.audio_matches_original_decoded_pcm,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--natural-reference-wav", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generated-video", type=Path)
    parser.add_argument(
        "--mux-original-wav",
        type=Path,
        help="PCM WAV to preserve in the final container (defaults to input_wav, retaining its rate/channels)",
    )
    parser.add_argument("--mux-output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.input_wav,
            args.output_wav,
            config_path=args.config,
            natural_reference_path=args.natural_reference_wav,
            checkpoint_path=args.checkpoint,
            device=args.device,
            generated_video=args.generated_video,
            mux_original_wav=args.mux_original_wav,
            mux_output=args.mux_output,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
