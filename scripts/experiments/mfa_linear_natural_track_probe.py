"""Paired, exploratory N/M/T video comparison on two clean MFA-linear records.

M and T videos are retimed against N by the locally ported NAS SyncDrift
algorithm. T first gets a deterministic uniform duration match; this is
recorded separately from the SyncDrift correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.experiments.mfa_linear_video_retiming.generation import (
    canonicalize_tail,
    read_video_frames,
)
from scripts.experiments.mfa_linear_video_retiming.official import score_official_cell
from scripts.experiments.static_image_bridge.render_worker import encode_ffv1_stream
from scripts.purify_syncdrift import get_video_info, render_purified_video

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/aishell1_mfa_linear_n25_resample_poly_20260814"
REGISTRY = REPO / "runs/phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3/00_protocol/registry.json"
IDS = ("1", "201")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
WAV2LIP_MODEL = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
SYNCNET = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(argv: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = str(FFMPEG.parent) + os.pathsep + env.get("PATH", "")
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(argv, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {argv}; see {log}")


def uniform_match(raw_video: Path, target_video: Path, target_frames: int) -> dict:
    frames, metadata = read_video_frames(raw_video)
    source_frames = len(frames)
    # Endpoints are pinned; half-up nearest keeps every output frame authentic.
    positions = np.linspace(0.0, source_frames - 1.0, target_frames, dtype=np.float64)
    indexes = np.floor(positions + 0.5).astype(np.int64)
    target_video.parent.mkdir(parents=True, exist_ok=True)
    encode_ffv1_stream(frames[indexes], width=metadata["width"], height=metadata["height"], output=target_video, ffmpeg=FFMPEG)
    decoded, _ = read_video_frames(target_video)
    if not np.array_equal(decoded, frames[indexes]):
        raise RuntimeError("uniform retime changed pixel values")
    return {"source_frames": source_frames, "target_frames": target_frames,
            "index_sha256": hashlib.sha256(indexes.tobytes()).hexdigest(),
            "repeated_frame_ratio": float(np.mean(np.diff(indexes) == 0)),
            "skipped_frame_ratio": float(np.mean(np.diff(indexes) > 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=REPO / "runs/mfa_linear_natural_track_probe_20260924")
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    summary_path = SOURCE / "mfa3_clean_mfa_linear_gate_fix2_20260814/summary.json"
    manifest_path = SOURCE / "mfa3_clean_gate_20260814/manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    tts_meta_path = REPO / "runs/aishell1_mfa_linear_n25_20260813/tts_strict/tts_meta.json"
    tts_meta = json.loads(tts_meta_path.read_text())
    portrait = json.loads(REGISTRY.read_text())["portraits"]["3"]
    if sha(Path(portrait["path"])) != portrait["container_sha256"]:
        raise RuntimeError("portrait hash mismatch")
    if sha(WAV2LIP_MODEL) != "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8":
        raise RuntimeError("Wav2Lip model hash mismatch")
    if sha(SYNCNET_MODEL) != "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442":
        raise RuntimeError("SyncNet model hash mismatch")

    config = {"repo_root": str(REPO), "paths": {
        "ffmpeg": str(FFMPEG), "ffprobe": str(FFPROBE), "syncnet_python": str(SYNCNET_PYTHON),
        "syncnet_root": str(SYNCNET), "syncnet_model": str(SYNCNET_MODEL)},
        "official": {"min_track": 100, "batch_size": 20,
                     "pipeline_timeout_seconds": 900, "syncnet_timeout_seconds": 900}}
    rows = []
    for sample_id in IDS:
        source = summary["results"][sample_id]
        paired_key = source["paired_key"]
        if source["sample_id"] != sample_id or source["speaker_id"] != manifest["samples"][sample_id]:
            raise RuntimeError("sample identity mismatch")
        if not paired_key.endswith(f"__BAC009{source['speaker_id']}W0122"):
            raise RuntimeError("paired key mismatch")
        audios = {"N": Path(manifest["natural_source"][sample_id]),
                  "M": Path(source["audio_path"]), "T": Path(manifest["tts_source"][sample_id])}
        tts_record = tts_meta["results"][sample_id]
        if (tts_record["paired_key"] != paired_key or
                tts_record["transcript"] != manifest["transcripts"][sample_id] or
                Path(tts_record["canonical_16k_audio"]).resolve() != audios["T"].resolve() or
                tts_record["canonical_audio_sha256"] != sha(audios["T"])):
            raise RuntimeError("TTS text, pairing, or hash mismatch")
        if sha(audios["M"]) != source["audio_sha256"]:
            raise RuntimeError("MFA-linear audio hash mismatch")
        sample_rates = {}
        durations = {}
        for role, audio in audios.items():
            sample_rate, pcm = wavfile.read(audio)
            if sample_rate != 16000 or pcm.ndim != 1:
                raise RuntimeError(f"invalid {role} audio: {audio}")
            sample_rates[role] = len(pcm)
            durations[role] = len(pcm) / sample_rate
        if sample_rates["M"] != sample_rates["N"] or abs(durations["T"] / durations["N"] - 1) > 0.2:
            raise RuntimeError("length selection contract failed")
        target_frames = math.ceil(sample_rates["N"] / 640)
        base = root / sample_id
        source_meta = {"sample_id": sample_id, "paired_key": paired_key,
                       "speaker_id": source["speaker_id"], "transcript": manifest["transcripts"][sample_id],
                       "natural_samples": sample_rates["N"], "target_frames": target_frames,
                       "audio": {role: {"path": str(path), "sha256": sha(path), "duration_s": durations[role]}
                                 for role, path in audios.items()}}
        save(base / "sources.json", source_meta)

        videos = {}
        for role, audio in audios.items():
            raw = base / "render" / f"{role}_raw.mkv"
            receipt = raw.with_suffix(".json")
            if not raw.exists():
                run([str(WAV2LIP_PYTHON), str(REPO / "scripts/experiments/static_image_bridge/render_worker.py"),
                     "--image", portrait["path"], "--image-rgb-sha256", portrait["rgb_pixel_sha256"],
                     "--audio", str(audio), "--box", *map(str, portrait["generation_box_xyxy"]),
                     "--checkpoint", str(WAV2LIP_MODEL), "--ffmpeg", str(FFMPEG),
                     "--outfile", str(raw), "--result", str(receipt), "--batch-size", "4",
                     "--seed", "20260924", "--device", "cuda"], base / "logs" / f"render_{role}.log")
            render_receipt = json.loads(receipt.read_text())
            if render_receipt["audio_sha256"] != sha(audio) or render_receipt["output_sha256"] != sha(raw):
                raise RuntimeError(f"render receipt mismatch: {role} {sample_id}")
            matched = base / "render" / f"{role}_matched.mkv"
            match_receipt = base / "render" / f"{role}_match.json"
            if not matched.exists():
                if role == "T":
                    info = uniform_match(raw, matched, target_frames)
                else:
                    info = canonicalize_tail(raw, matched, target_frame_count=target_frames,
                                             expected_raw_count=render_receipt["decoded_frame_count"], ffmpeg=FFMPEG)
                save(match_receipt, {"role": role, "video_sha256": sha(matched), "details": info})
            if json.loads(match_receipt.read_text())["video_sha256"] != sha(matched):
                raise RuntimeError("matched video changed")
            mp4 = base / "input" / f"{role}.mp4"
            if not mp4.exists():
                mp4.parent.mkdir(parents=True, exist_ok=True)
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(matched), "-i", str(audios["N"]),
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "16",
                     "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "25", "-frames:v", str(target_frames),
                     "-c:a", "aac", "-b:a", "192k", str(mp4)], base / "logs" / f"mux_{role}.log")
            videos[role] = mp4

        scored = {}
        for role in ("N", "M", "T"):
            for variant in (("raw", "identity") if role == "N" else ("raw", "aligned")):
                if variant == "aligned":
                    output = base / "aligned" / f"{role}.mp4"
                    result_file = base / "aligned" / role / "results.json"
                    if not result_file.exists():
                        run([sys.executable, str(REPO / "scripts/purify_syncdrift.py"),
                             "--input_video", str(videos[role]), "--output_video", str(output),
                             "--save_dir", str(base / "aligned" / role), "--mode", "window",
                             "--device", "cuda", "--syncnet_root", str(SYNCNET),
                             "--syncnet_model", str(SYNCNET_MODEL)], base / "logs" / f"align_{role}.log")
                    video = output
                elif variant == "identity":
                    video = base / "identity" / "N.mp4"
                    if not video.exists():
                        info = get_video_info(videos["N"])
                        render_purified_video(videos["N"], video, info,
                                              np.arange(info.frame_count, dtype=np.int64), base / "identity")
                else:
                    video = videos[role]
                cell = {"sample_id": sample_id, "paired_key": paired_key,
                        "speaker_id": source["speaker_id"], "portrait_id": "3", "video_arm": f"{role}_{variant}",
                        "audio_role": "N", "video": str(video), "audio": str(audios["N"])}
                score = score_official_cell(config, cell, run_dir=base, resume=True)
                scored[f"{role}_{variant}"] = {"sync_c": score["official_sync_c"],
                                               "sync_d": score["official_sync_d"],
                                               "offset": score["official_offset"],
                                               "video": str(video), "video_sha256": sha(video),
                                               "score_receipt": str(base / "06_official/cells" / f"p3_s{sample_id}_v{role}_{variant}_aN/result.json"),
                                               "track_frames": score["crop"]["frame_count"]}
                print(f"{sample_id} {role}_{variant} C={score['official_sync_c']:.3f}", flush=True)
        baseline = scored["N_raw"]["sync_c"]
        for role in ("M", "T"):
            scored[f"{role}_aligned"]["delta_vs_natural"] = scored[f"{role}_aligned"]["sync_c"] - baseline
        row = {**source_meta, "scores": scored}
        save(base / "result.json", row)
        rows.append(row)
    save(root / "summary.json", {"schema_version": 1, "status": "COMPLETE", "portrait_id": "3",
                                 "selection": "clean MFA IDs with |T/N duration ratio - 1| <= 0.20; IDs 1, 201",
                                 "alignment": "uniform T duration match followed by NAS window SyncDrift; M uses NAS window SyncDrift",
                                 "scorer": "official local SyncNet pipeline, min_track=100, all cells remuxed with original PCM16 N",
                                 "samples": rows})


if __name__ == "__main__":
    main()
