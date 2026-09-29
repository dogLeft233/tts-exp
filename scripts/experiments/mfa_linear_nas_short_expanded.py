"""Ten-speaker replication of the NAS short preset using existing AISHELL n100 renders.

All five official cells per record use the original natural PCM16 audio. The
TTS video is uniformly matched to the natural frame count before purification.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from scipy.io import wavfile

from scripts.experiments.mfa_linear_natural_track_probe import (
    FFMPEG, FFPROBE, REPO, SYNCNET, SYNCNET_MODEL, SYNCNET_PYTHON,
    save, run, sha, uniform_match,
)
from scripts.experiments.mfa_linear_video_retiming.generation import canonicalize_tail
from scripts.experiments.mfa_linear_video_retiming.official import score_official_cell
from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError


SOURCE = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
ROOT = REPO / "runs/mfa_linear_nas_short_expanded_20260924"
IDS = ("a1_002", "a1_010", "a1_014", "a1_019", "a1_028",
       "a1_031", "a1_040", "a1_044", "a1_049", "a1_057")


def read_pcm(path: Path) -> int:
    rate, pcm = wavfile.read(path)
    if rate != 16000 or pcm.ndim != 1 or pcm.dtype not in ("int16", "float32"):
        raise RuntimeError(f"expected mono 16k audio: {path}")
    return len(pcm)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cohort_file = SOURCE / "00_pairs/cohort.json"
    tts_file = SOURCE / "01_tts_retry/tts_meta.json"
    mfa_file = SOURCE / "04_mfa_linear/summary.json"
    cohort = json.loads(cohort_file.read_text())
    tts = json.loads(tts_file.read_text())
    mfa = json.loads(mfa_file.read_text())
    cohort_rows = {r["sample_id"]: r for r in cohort["records"]}
    source_render = SOURCE / "05_wav2lip_syncnet_n100"
    source_manifest_file = source_render / "manifest.json"
    source_manifest = json.loads(source_manifest_file.read_text())
    preset = REPO.parent / "nas-cch-code/voice_def/cch_419/run_1s_no_blend_purify_w05_a100.sh"
    protocol = {
        "source_cohort": str(cohort_file), "source_cohort_sha256": sha(cohort_file),
        "source_tts_meta": str(tts_file), "source_tts_meta_sha256": sha(tts_file),
        "source_mfa_summary": str(mfa_file), "source_mfa_summary_sha256": sha(mfa_file),
        "source_wav2lip_manifest": str(source_manifest_file),
        "source_wav2lip_manifest_sha256": sha(source_manifest_file),
        "nas_preset": str(preset), "nas_preset_sha256": sha(preset),
        "selection": "first eligible record per speaker in cohort order; duration >=4.8s; abs(T/N-1)<=0.2; all three preexisting videos; first 10 speakers",
        "sample_ids": list(IDS), "video_arms": ["N", "M", "T"],
        "nas_window_sec": 0.5, "nas_stride_sec": 0.1,
        "nas_search_min_ms": -1000, "nas_search_max_ms": 1000,
        "nas_batch_size": 20, "nas_min_track": 40,
        "official_min_track": 100, "official_batch_size": 20,
        "audio_role": "original natural PCM16 for all scores",
        "video_source": "existing n100 Wav2Lip renders, N/M tail-normalized, T uniform duration matched",
    }
    protocol_file = ROOT / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise RuntimeError("frozen protocol mismatch")
    save(protocol_file, protocol)

    # Reproduce the predeclared selection without post-score choices.
    eligible = []
    seen = set()
    for row in cohort["records"]:
        sid, speaker = row["sample_id"], row["speaker_id"]
        if speaker in seen or sid not in tts["results"] or sid not in mfa["results"]:
            continue
        ratio = float(tts["results"][sid]["duration_ratio"])
        if row["natural_duration_s"] < 4.8 or abs(ratio - 1.0) > 0.2:
            continue
        videos = [source_render / "wav2lip" / arm / speaker / f"{sid}.mp4"
                  for arm in ("natural_raw", "mfa_linear", "raw_tts")]
        if not all(path.is_file() for path in videos):
            continue
        eligible.append(sid)
        seen.add(speaker)
        if len(eligible) == len(IDS):
            break
    if tuple(eligible) != IDS:
        raise RuntimeError(f"selection drift: {eligible}")
    if sha(SYNCNET_MODEL) != source_manifest["syncnet_model_sha256"]:
        raise RuntimeError("SyncNet model changed")

    config = {"repo_root": str(REPO), "paths": {
        "ffmpeg": str(FFMPEG), "ffprobe": str(FFPROBE),
        "syncnet_python": str(SYNCNET_PYTHON), "syncnet_root": str(SYNCNET),
        "syncnet_model": str(SYNCNET_MODEL)},
        "official": {"min_track": 100, "batch_size": 20,
                     "pipeline_timeout_seconds": 900, "syncnet_timeout_seconds": 900}}
    rows = []
    for sid in IDS:
        row = cohort_rows[sid]
        speaker = row["speaker_id"]
        tts_row, mfa_row = tts["results"][sid], mfa["results"][sid]
        natural = Path(row["audio_path"])
        raw_tts = Path(tts_row["canonical_16k_audio"])
        mfa_audio = Path(mfa_row["audio_path"])
        if (row["paired_key"] != tts_row["paired_key"] or
                row["paired_key"] != mfa_row["paired_key"] or
                row["transcript"] != tts_row["transcript"] or
                sha(natural) != row["audio_sha256"] or
                sha(raw_tts) != tts_row["canonical_audio_sha256"] or
                sha(mfa_audio) != mfa_row["audio_sha256"]):
            raise RuntimeError(f"pair/text/hash mismatch: {sid}")
        n_samples, m_samples, t_samples = (read_pcm(p) for p in (natural, mfa_audio, raw_tts))
        if n_samples != m_samples or abs(t_samples / n_samples - 1.0) > 0.2:
            raise RuntimeError(f"duration contract failed: {sid}")
        target_frames = math.ceil(n_samples / 640)
        sample_root = ROOT / sid
        normalized = {}
        for arm, source_arm in (("N", "natural_raw"), ("M", "mfa_linear"), ("T", "raw_tts")):
            source_video = source_render / "wav2lip" / source_arm / speaker / f"{sid}.mp4"
            intermediate = sample_root / "normalized" / f"{arm}.mkv"
            receipt_file = sample_root / "normalized" / f"{arm}.json"
            if not receipt_file.exists():
                if arm == "T":
                    detail = uniform_match(source_video, intermediate, target_frames)
                else:
                    detail = canonicalize_tail(source_video, intermediate,
                                               target_frame_count=target_frames, ffmpeg=FFMPEG)
                save(receipt_file, {"source_video": str(source_video),
                                    "source_video_sha256": sha(source_video),
                                    "video_sha256": sha(intermediate), "detail": detail})
            receipt = json.loads(receipt_file.read_text())
            if receipt["source_video_sha256"] != sha(source_video) or receipt["video_sha256"] != sha(intermediate):
                raise RuntimeError(f"normalized video changed: {sid} {arm}")
            video = sample_root / "input" / f"{arm}.mp4"
            if not video.exists():
                video.parent.mkdir(parents=True, exist_ok=True)
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(intermediate), "-i", str(natural),
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "16",
                     "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "25",
                     "-frames:v", str(target_frames), "-c:a", "aac", "-b:a", "192k", str(video)],
                    sample_root / "logs" / f"mux_{arm}.log")
            normalized[arm] = video

        scores = {}
        purify = {}
        for arm in ("N", "M", "T"):
            video = normalized[arm]
            if arm != "N":
                aligned = sample_root / "nas_w05" / f"{arm}.mp4"
                out = sample_root / "nas_w05" / arm
                if not (out / "results.json").exists():
                    out.mkdir(parents=True, exist_ok=True)
                    argv = [sys.executable, str(REPO / "scripts/purify_syncdrift.py"),
                            "--input_video", str(video), "--output_video", str(aligned),
                            "--save_dir", str(out), "--mode", "window", "--window_sec", "0.5",
                            "--stride_sec", "0.1", "--search_min_ms", "-1000",
                            "--search_max_ms", "1000", "--batch_size", "20",
                            "--min_track", "40", "--device", "cuda",
                            "--syncnet_root", str(SYNCNET), "--syncnet_model", str(SYNCNET_MODEL)]
                    run(argv, sample_root / "logs" / f"purify_{arm}.log")
                purify[arm] = json.loads((out / "results.json").read_text())
                if not aligned.is_file():
                    raise RuntimeError(f"NAS result missing: {sid} {arm}")
                normalized[f"{arm}_nas"] = aligned
            for variant in (("raw",) if arm == "N" else ("raw", "nas")):
                label = f"{arm}_{variant}"
                eval_video = normalized[arm] if variant == "raw" else normalized[label]
                cell = {"sample_id": sid, "paired_key": row["paired_key"],
                        "speaker_id": speaker, "portrait_id": "n100",
                        "video_arm": label, "audio_role": "N", "video": str(eval_video),
                        "audio": str(natural)}
                score_dir = sample_root / "06_official/cells" / f"pn100_s{sid}_v{label}_aN"
                try:
                    result = score_official_cell(config, cell, run_dir=sample_root, resume=True)
                    if not result["mux"]["audio_pcm_exact"] or not result["mux_pixel_pts_audit"]["pixels_identical"]:
                        raise RuntimeError(f"media verification failed: {sid} {label}")
                    score_status = "PASS"
                    receipt_path = score_dir / "result.json"
                except ProtocolError as exc:
                    if str(exc) != "OFFICIAL_BOUNDARY_PEAK":
                        raise
                    log_path = score_dir / "syncnet.log"
                    log = log_path.read_text()
                    values = [re.findall(r"Confidence:\s*([-+]?\d+(?:\.\d+)?)", log),
                              re.findall(r"Min dist:\s*([-+]?\d+(?:\.\d+)?)", log),
                              re.findall(r"AV offset:\s*(-?\d+)", log)]
                    if any(len(matches) != 1 for matches in values) or abs(int(values[2][0])) != 15:
                        raise RuntimeError(f"unreadable boundary score: {sid} {label}") from exc
                    result = {"official_sync_c": float(values[0][0]),
                              "official_sync_d": float(values[1][0]),
                              "official_offset": int(values[2][0])}
                    score_status = "BOUNDARY_PEAK"
                    receipt_path = log_path
                scores[label] = {"sync_c": result["official_sync_c"],
                                 "sync_d": result["official_sync_d"],
                                 "offset": result["official_offset"],
                                 "status": score_status,
                                 "video": str(eval_video), "video_sha256": sha(eval_video),
                                 "score_receipt": str(receipt_path)}
                print(sid, label, f"C={scores[label]['sync_c']:.3f}", score_status, flush=True)
        item = {"sample_id": sid, "speaker_id": speaker, "paired_key": row["paired_key"],
                "natural_duration_s": n_samples / 16000,
                "tts_natural_duration_ratio": t_samples / n_samples,
                "source_audio": {"N": str(natural), "M": str(mfa_audio), "T": str(raw_tts)},
                "scores": scores,
                "purify": {arm: {key: purify[arm][key] for key in (
                    "repeated_frame_ratio", "skipped_frame_ratio",
                    "mean_abs_correction_ms", "max_abs_correction_ms", "valid_window_ratio")}
                           for arm in ("M", "T")}}
        save(sample_root / "result.json", item)
        rows.append(item)
        save(ROOT / "summary.json", {"protocol": protocol, "status": "COMPLETE" if len(rows) == len(IDS) else "IN_PROGRESS", "rows": rows})


if __name__ == "__main__":
    main()
