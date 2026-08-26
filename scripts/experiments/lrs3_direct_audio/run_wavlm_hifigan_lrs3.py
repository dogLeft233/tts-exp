#!/usr/bin/env python3
"""Run LRS3 WavLM encode/decode, Wav2Lip, SyncNet, and strict replacement."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
DEFAULT_KNN_VC_SOURCE = Path(torch.hub.get_dir()) / f"bshall_knn-vc_{KNN_VC_REVISION}"
SAMPLE_RATE = 16_000
FRAME_STRIDE_SAMPLES = 320
FEATURE_DIM = 1024
MIN_TRACK = 50
SEED = 20260826


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_asset(value: str | Path, source_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_records(manifest_path: Path, source_root: Path, start: int, count: int) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "lrs3" or not isinstance(manifest.get("records"), list):
        raise ValueError("manifest is not an LRS3 record manifest")
    records = manifest["records"]
    selected = records[start : start + count]
    if len(selected) != count:
        raise ValueError(f"manifest has {len(selected)} records in requested range, expected {count}")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        seen.add(sample_id)
        video = resolve_asset(row["video_local_path"], source_root)
        natural_audio = resolve_asset(row["natural_audio_path"], source_root)
        output.append(
            {
                "sample_id": sample_id,
                "source_group": str(row.get("source_group", "")),
                "clip_id": str(row.get("clip_id", "")),
                "transcript": str(row.get("transcript", "")),
                "video": str(video),
                "video_sha256": file_sha256(video),
                "natural_audio": str(natural_audio),
                "natural_audio_sha256": file_sha256(natural_audio),
                "manifest_audio_sha256": str(row.get("natural_audio_sha256", "")),
            }
        )
    return output


def load_model(device: str, source: str | Path) -> Any:
    source_path = Path(source)
    if not source_path.is_dir():
        raise FileNotFoundError(f"local kNN-VC source is missing: {source_path}")
    model = torch.hub.load(
        str(source_path),
        "knn_vc",
        source="local",
        prematched=True,
        pretrained=True,
        device=device,
    )
    if int(getattr(model, "sr", -1)) != SAMPLE_RATE or int(getattr(model, "hop_length", -1)) != FRAME_STRIDE_SAMPLES:
        raise ValueError("unexpected WavLM/HiFi-GAN sample-rate or frame-stride interface")
    for module_name in ("wavlm", "hifigan"):
        module = getattr(model, module_name, None)
        if module is None or getattr(module, "training", True):
            raise ValueError(f"{module_name} must be present and in eval mode")
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model


def model_checkpoint_metadata() -> dict[str, Any]:
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    result: dict[str, Any] = {}
    for filename in ("WavLM-Large.pt", "prematch_g_02500000.pt"):
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        result[filename] = {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size}
    return result


def read_natural_audio(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"natural audio has sample rate {sample_rate}, expected {SAMPLE_RATE}: {path}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"natural audio is not mono: {path}")
    if values.size < 1024 or not np.isfinite(values).all():
        raise ValueError(f"natural audio is empty or non-finite: {path}")
    if float(np.abs(values).max()) > 1.0:
        raise ValueError(f"natural audio exceeds PCM range: {path}")
    return values


def exact_length(values: np.ndarray, target_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    raw_count = int(values.size)
    if raw_count > target_count:
        result = values[:target_count].copy()
        action = "right_crop"
    elif raw_count < target_count:
        result = np.pad(values, (0, target_count - raw_count)).astype(np.float32)
        action = "right_zero_pad"
    else:
        result = values.copy()
        action = "none"
    return result, {
        "raw_decoder_sample_count": raw_count,
        "target_natural_sample_count": int(target_count),
        "action": action,
        "adjustment_sample_count": abs(raw_count - target_count),
    }


def audio_qc(values: np.ndarray, natural_count: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "sample_count": int(values.size),
        "natural_sample_count": int(natural_count),
        "exact_natural_sample_count": bool(values.size == natural_count),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.abs(values).max()) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "clipped_sample_count": int((np.abs(values) >= 1.0).sum()),
    }


def feature_distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    frames = min(int(left.shape[0]), int(right.shape[0]))
    if frames <= 0 or left.shape[1] != right.shape[1]:
        raise ValueError("feature distance requires compatible non-empty sequences")
    left = left[:frames].float()
    right = right[:frames].float().to(left.device)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(left, right, dim=-1)
    return {
        "cosine_distance": float(cosine.mean().cpu()),
        "mse": float(torch.mean((left - right) ** 2).cpu()),
        "compared_frames": frames,
    }


def resynthesize(model: Any, natural: np.ndarray, output_path: Path) -> dict[str, Any]:
    waveform = torch.from_numpy(natural).unsqueeze(0)
    with torch.inference_mode():
        encoded = torch.as_tensor(
            model.get_features(waveform.to(next(model.wavlm.parameters()).device), vad_trigger_level=0),
            dtype=torch.float32,
        )
        if encoded.ndim != 2 or encoded.shape[1] != FEATURE_DIM or encoded.shape[0] == 0:
            raise ValueError(f"unexpected WavLM feature shape: {tuple(encoded.shape)}")
        decoded = model.vocode(encoded.unsqueeze(0).to(encoded.device)).squeeze(0).detach().cpu().numpy()
        decoded, length_adjustment = exact_length(decoded, natural.size)
        reencoded = torch.as_tensor(
            model.get_features(torch.from_numpy(decoded).unsqueeze(0).to(encoded.device), vad_trigger_level=0),
            dtype=torch.float32,
        )
    if not np.isfinite(decoded).all() or float(np.abs(decoded).max()) > 1.0:
        raise ValueError(f"decoded waveform is non-finite or outside PCM range: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, decoded, SAMPLE_RATE, subtype="PCM_16")
    return {
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "length_adjustment": length_adjustment,
        "audio": audio_qc(decoded, natural.size),
        "natural_feature_shape": list(encoded.shape),
        "reencoded_feature_shape": list(reencoded.shape),
        "reencoded_to_natural": feature_distance(reencoded, encoded),
    }


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(command), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, command))}; see {log_path}")


def render_wav2lip(
    repo_root: Path,
    wav2lip_python: Path,
    checkpoint: Path,
    face: Path,
    audio: Path,
    output: Path,
    work_dir: Path,
    log_path: Path,
) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "temp").mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(wav2lip_python),
        str(repo_root / "third_party/Wav2Lip/inference.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(face),
        "--audio",
        str(audio),
        "--outfile",
        str(output),
        "--face_det_batch_size",
        "16",
        "--wav2lip_batch_size",
        "16",
        "--nosmooth",
    ]
    try:
        run_logged(command, work_dir, log_path)
        render_mode = "detected"
    except RuntimeError:
        if "Face not detected!" not in log_path.read_text(encoding="utf-8"):
            raise
        fallback_log_path = log_path.with_name(f"{log_path.stem}.static.log")
        run_logged([*command, "--static", "True"], work_dir, fallback_log_path)
        render_mode = "static_first_frame_fallback"
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Wav2Lip did not produce a non-empty video: {output}")
    return render_mode


def ffprobe_streams(path: Path, ffprobe: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned no stream list: {path}")
    return streams


def decode_pcm16(path: Path, ffmpeg: str) -> bytes:
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def strict_mux(video: Path, audio: Path, output: Path, ffmpeg: str, ffprobe: str, log_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "matroska",
            str(output),
        ],
        output.parent,
        log_path,
    )
    streams = ffprobe_streams(output, ffprobe)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError(f"strict mux does not have one video and one audio stream: {output}")
    audio_stream = audios[0]
    if audio_stream.get("codec_name") != "pcm_s16le" or int(audio_stream.get("sample_rate", 0)) != SAMPLE_RATE or int(audio_stream.get("channels", 0)) != 1:
        raise ValueError(f"strict mux audio stream violates PCM contract: {output}")
    expected_pcm = decode_pcm16(audio, ffmpeg)
    actual_pcm = decode_pcm16(output, ffmpeg)
    expected_hash = hashlib.sha256(expected_pcm).hexdigest()
    actual_hash = hashlib.sha256(actual_pcm).hexdigest()
    if expected_pcm != actual_pcm:
        raise ValueError(f"strict mux changed decoded PCM: {output}")
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "video_stream": {
            key: videos[0].get(key)
            for key in ("codec_name", "width", "height", "r_frame_rate", "time_base")
        },
        "audio_stream": {
            key: audio_stream.get(key)
            for key in ("codec_name", "sample_rate", "channels", "channel_layout")
        },
        "decoded_pcm_sha256": actual_hash,
        "expected_audio_decoded_pcm_sha256": expected_hash,
        "pcm_exact_match": True,
        "video_source_sha256": file_sha256(video),
        "audio_source_sha256": file_sha256(audio),
    }


def parse_syncnet(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ValueError(f"could not parse SyncNet output: {log_path}")
    result: dict[str, float | int] = {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(distance.group(1)),
        "av_offset": int(offset.group(1)) if offset else 0,
    }
    if not all(math.isfinite(float(result[key])) for key in ("sync_c", "sync_d", "av_offset")):
        raise ValueError(f"SyncNet output is non-finite: {log_path}")
    return result


def score_syncnet(
    syncnet_root: Path,
    syncnet_python: Path,
    syncnet_model: Path,
    media: Path,
    output_dir: Path,
    reference: str,
    log_dir: Path,
) -> dict[str, Any]:
    data_dir = output_dir / "syncnet" / reference
    data_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / f"{reference}.pipeline.log"
    score_log = log_dir / f"{reference}.score.log"
    run_logged(
        [
            str(syncnet_python),
            "run_pipeline.py",
            "--videofile",
            str(media),
            "--reference",
            reference,
            "--data_dir",
            str(data_dir),
            "--min_track",
            str(MIN_TRACK),
            "--overwrite",
        ],
        syncnet_root,
        pipeline_log,
    )
    run_logged(
        [
            str(syncnet_python),
            "run_syncnet.py",
            "--videofile",
            str(media),
            "--reference",
            reference,
            "--data_dir",
            str(data_dir),
            "--initial_model",
            str(syncnet_model),
        ],
        syncnet_root,
        score_log,
    )
    return {
        "reference": reference,
        "media": str(media),
        "media_sha256": file_sha256(media),
        "log": str(score_log),
        **parse_syncnet(score_log),
    }


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], seed: int, draws: int = 10_000) -> dict[str, float]:
    if len(values) != len(groups) or not values:
        raise ValueError("bootstrap inputs are empty or have different lengths")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        by_group[str(group)].append(float(value))
    labels = sorted(by_group)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        means[index] = np.concatenate([np.asarray(by_group[label]) for label in sampled]).mean()
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95_low": float(np.quantile(means, 0.025)),
        "bootstrap_95_high": float(np.quantile(means, 0.975)),
        "n_records": len(values),
        "n_source_groups": len(labels),
    }


def summarize_scores(records: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sample = {(str(row["sample_id"]), str(row["cell"])): row for row in scores}
    contrasts: dict[str, list[float]] = defaultdict(list)
    contrast_groups: dict[str, list[str]] = defaultdict(list)
    wins: dict[str, dict[str, int]] = defaultdict(lambda: {"c": 0, "d": 0, "joint": 0})
    for record in records:
        sample_id = str(record["sample_id"])
        source_group = str(record["source_group"])
        baseline = by_sample[(sample_id, "natural_video_natural_audio")]
        direct = by_sample[(sample_id, "direct_video_direct_audio")]
        replacement = by_sample[(sample_id, "direct_video_natural_audio")]
        for name, row in (("driver_direct_minus_natural", direct), ("replacement_direct_video_minus_natural_video", replacement)):
            c = float(row["sync_c"]) - float(baseline["sync_c"])
            d = float(baseline["sync_d"]) - float(row["sync_d"])
            contrasts[f"{name}_sync_c"].append(c)
            contrasts[f"{name}_sync_d"].append(d)
            contrast_groups[f"{name}_sync_c"].append(source_group)
            contrast_groups[f"{name}_sync_d"].append(source_group)
            if c > 0:
                wins[name]["c"] += 1
            if d > 0:
                wins[name]["d"] += 1
            if c > 0 and d > 0:
                wins[name]["joint"] += 1
    result: dict[str, Any] = {}
    for name, values in contrasts.items():
        result[name] = cluster_bootstrap(values, contrast_groups[name], SEED + len(result))
    result["wins"] = {name: {**counts, "n": len(records)} for name, counts in wins.items()}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    source_root = args.source_root.resolve()
    records = load_records(manifest_path, source_root, args.start, args.count)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav2lip_checkpoint = args.repo_root.resolve() / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
    syncnet_model = args.repo_root.resolve() / "third_party/syncnet_python/data/syncnet_v2.model"
    wav2lip_root = args.repo_root.resolve() / "third_party/Wav2Lip"
    syncnet_root = args.repo_root.resolve() / "third_party/syncnet_python"
    for path in (wav2lip_checkpoint, syncnet_model, wav2lip_root / "inference.py", syncnet_root / "run_pipeline.py", syncnet_root / "run_syncnet.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.wav2lip_python.is_file() or not args.syncnet_python.is_file():
        raise FileNotFoundError("Wav2Lip or SyncNet Python environment is missing")
    model_source = args.model_source
    model_metadata = model_checkpoint_metadata()
    model = load_model(args.device, model_source)
    reconstructions: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        natural_path = Path(record["natural_audio"])
        natural = read_natural_audio(natural_path)
        output_path = output_dir / "audio" / f"{record['sample_id']}__wavlm_hifigan.wav"
        recon = resynthesize(model, natural, output_path)
        reconstructions.append({**record, "natural_sample_count": int(natural.size), "natural_duration_s": float(natural.size / SAMPLE_RATE), "resynthesis": recon})
        print(f"AUDIO {index}/{len(records)} {record['sample_id']} {recon['reencoded_to_natural']['cosine_distance']:.4f}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    render_records: dict[str, dict[str, Path]] = {}
    for index, record in enumerate(reconstructions, 1):
        sample_id = str(record["sample_id"])
        face = Path(record["video"])
        natural = Path(record["natural_audio"])
        direct = Path(record["resynthesis"]["output_path"])
        natural_video = output_dir / "videos" / "natural_driver" / f"{sample_id}.mp4"
        direct_video = output_dir / "videos" / "direct_driver" / f"{sample_id}.mp4"
        render_wav2lip(args.repo_root.resolve(), args.wav2lip_python, wav2lip_checkpoint, face, natural, natural_video, output_dir / "wav2lip_work" / "natural" / sample_id, output_dir / "logs" / "wav2lip" / f"{sample_id}.natural.log")
        render_wav2lip(args.repo_root.resolve(), args.wav2lip_python, wav2lip_checkpoint, face, direct, direct_video, output_dir / "wav2lip_work" / "direct" / sample_id, output_dir / "logs" / "wav2lip" / f"{sample_id}.direct.log")
        render_records[sample_id] = {"natural_video": natural_video, "direct_video": direct_video, "natural_audio": natural, "direct_audio": direct}
        print(f"VIDEO {index}/{len(reconstructions)} {sample_id}", flush=True)

    cells = (
        ("natural_video_natural_audio", "natural_video", "natural_audio"),
        ("direct_video_direct_audio", "direct_video", "direct_audio"),
        ("direct_video_natural_audio", "direct_video", "natural_audio"),
    )
    scores: list[dict[str, Any]] = []
    muxes: list[dict[str, Any]] = []
    for index, record in enumerate(reconstructions, 1):
        sample_id = str(record["sample_id"])
        paths = render_records[sample_id]
        for cell, video_key, audio_key in cells:
            mux_path = output_dir / "mux" / cell / f"{sample_id}.mkv"
            mux = strict_mux(paths[video_key], paths[audio_key], mux_path, args.ffmpeg, args.ffprobe, output_dir / "logs" / "mux" / f"{sample_id}.{cell}.log")
            muxes.append({"sample_id": sample_id, "cell": cell, **mux})
            score = score_syncnet(args.syncnet_root.resolve(), args.syncnet_python, syncnet_model, mux_path, output_dir, f"{cell}__{sample_id}", output_dir / "logs" / "syncnet")
            scores.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "cell": cell, **score})
            print(f"SCORE {index}/{len(reconstructions)} {cell} {sample_id} C={score['sync_c']:.3f} D={score['sync_d']:.3f}", flush=True)
    expected_scores = len(reconstructions) * len(cells)
    if len(scores) != expected_scores or len(muxes) != expected_scores:
        raise RuntimeError(f"incomplete result: scores={len(scores)} muxes={len(muxes)} expected={expected_scores}")
    summary = {
        "schema_version": 1,
        "experiment_type": "lrs3_wavlm_hifigan_direct_resynthesis_wav2lip_syncnet",
        "status": "complete",
        "dataset": "lrs3",
        "selection": {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "source_root": str(source_root),
            "start": args.start,
            "count": len(records),
            "selection_rule": "manifest_order_prefix",
            "sample_ids": [str(record["sample_id"]) for record in records],
            "sample_ids_sha256": hashlib.sha256("\n".join(str(record["sample_id"]) for record in records).encode()).hexdigest(),
            "source_groups": len({str(record["source_group"]) for record in records}),
        },
        "audio_protocol": {
            "encoder": "WavLM-Large layer 6",
            "decoder": "prematched HiFi-GAN from pinned bshall/knn-vc",
            "sample_rate": SAMPLE_RATE,
            "frame_stride_samples": FRAME_STRIDE_SAMPLES,
            "feature_dim": FEATURE_DIM,
            "input": "natural LRS3 waveform",
            "output": "exact natural sample count, PCM_16 WAV",
            "model_source": str(model_source),
            "model_revision": KNN_VC_REVISION,
            "checkpoints": model_metadata,
        },
        "video_protocol": {
            "tfg": "frozen Wav2Lip",
            "wav2lip_checkpoint": str(wav2lip_checkpoint),
            "wav2lip_checkpoint_sha256": file_sha256(wav2lip_checkpoint),
            "face_input": "same original LRS3 video for both natural and direct driver arms",
        },
        "syncnet_protocol": {
            "model": str(syncnet_model),
            "model_sha256": file_sha256(syncnet_model),
            "min_track": MIN_TRACK,
            "scorer": "official file-level SyncNet V2",
        },
        "replacement_protocol": {
            "format": "Matroska with copied video stream and PCM s16le audio",
            "video_stream_copy": True,
            "decoded_pcm_must_equal_source": True,
            "primary_contrast": "direct_video_natural_audio vs natural_video_natural_audio",
        },
        "cells": [cell for cell, _, _ in cells],
        "reconstructions": reconstructions,
        "muxes": muxes,
        "scores": scores,
        "statistics": summarize_scores(reconstructions, scores),
        "runtime": {
            "argv": sys.argv,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-source", type=Path, default=DEFAULT_KNN_VC_SOURCE)
    parser.add_argument("--wav2lip-python", type=Path, default=Path.home() / ".venvs/wav2lip/bin/python")
    parser.add_argument("--syncnet-python", type=Path, default=Path.home() / ".venvs/syncnet/bin/python")
    parser.add_argument("--syncnet-root", type=Path, default=REPO_ROOT / "third_party/syncnet_python")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if args.start < 0 or args.count <= 0:
        parser.error("--start must be non-negative and --count must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "scores": len(summary["scores"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
