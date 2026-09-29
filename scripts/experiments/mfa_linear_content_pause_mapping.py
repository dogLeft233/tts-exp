#!/usr/bin/env python3
"""Paired MFA-linear pause/mapping intervention on the frozen n=10 cohort.

Stages: audio, render, score, analyze.  The existing M video is the reference.
All generated videos are scored with the same natural AAC-decoded audio clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.knn_vc_retrieval import frame_owners, matched_span_map, mfa_linear_target
from scripts.pilot_generate_mfa_linear import exact_natural_length, extend_tokens_for_feature_tail
from scripts.wavlm_knn_vc_adapter import KNN_VC_REVISION, WavLMKNNVCAdapter
from scripts.experiments.mfa_linear_video_retiming.generation import canonicalize_tail
from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
from scripts.experiments.mfa_linear_vocoder_wav2lip_split import score_matrix

COHORT = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
PRIOR = REPO / "runs/mfa_linear_vocoder_wav2lip_split_20260925"
OUT = REPO / "runs/mfa_linear_content_pause_prosody_mapping_20260925"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PY = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
ARMS = ("P", "C", "B", "S")  # true pause, speech-place control, boundary, silence feature


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(argv: list[str], log: Path, cwd: Path = REPO) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        result = subprocess.run(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed {result.returncode}: {argv[:3]}; log={log}")


def missing_pauses() -> dict[str, list[tuple[float, float]]]:
    prior = read(REPO / "runs/mfa_linear_phoneme_video_audit_20260925/analysis.json")
    pauses: dict[str, list[tuple[float, float]]] = {}
    for p in prior["silences_ge_100ms"]:
        if not p["matched_to_tts_silence"]:
            pauses.setdefault(p["sample_id"], []).append((float(p["start_s"]), float(p["end_s"])))
    return pauses


def pause_zero(audio: np.ndarray, intervals: list[tuple[float, float]], sr: int = 16000) -> np.ndarray:
    """Only change flagged natural-pause spans; 20 ms fades avoid sharp clicks."""
    result = audio.copy()
    for start, end in intervals:
        a, b = max(0, round(start * sr)), min(len(result), round(end * sr))
        if b <= a:
            raise ValueError("empty pause")
        fade = min(round(.020 * sr), (b - a) // 3)
        gain = np.zeros(b - a, dtype=np.float32)
        if fade:
            gain[:fade] = np.linspace(1., 0., fade, dtype=np.float32)
            gain[-fade:] = np.linspace(0., 1., fade, dtype=np.float32)
        result[a:b] *= gain
    return result


def speech_control_intervals(natural: np.ndarray, intervals: list[tuple[float, float]], sid: str) -> list[tuple[float, float]]:
    """Deterministically place equal-duration zero patches in natural speech."""
    rng = random.Random(int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16))
    chosen = []
    duration = len(natural) / 16000
    for lo, hi in intervals:
        width = hi - lo
        candidates = [i / 100 for i in range(0, max(0, int((duration - width) * 100)), 5)]
        rng.shuffle(candidates)
        for start in candidates:
            end = start + width
            if any(start < b + .25 and end > a - .25 for a, b in [*intervals, *chosen]):
                continue
            segment = natural[round(start * 16000):round(end * 16000)]
            if len(segment) and float(np.sqrt(np.mean(segment * segment))) >= .008:
                chosen.append((start, end))
                break
        else:
            raise ValueError(f"no matched speech control for {sid} interval {lo}-{hi}")
    return chosen


def control() -> None:
    """Append the predeclared equal-duration speech-location control to the manifest."""
    path = OUT / "manifest.json"
    payload = read(path)
    for row in payload["rows"]:
        sid = row["id"]
        n, nsr = sf.read(row["natural"], dtype="float32")
        m, msr = sf.read(row["mfa"], dtype="float32")
        if nsr != 16000 or msr != 16000 or len(n) != len(m):
            raise ValueError(f"clock mismatch {sid}")
        intervals = speech_control_intervals(n, row["missing_pauses"], sid)
        values = pause_zero(m, intervals)
        target = OUT / "audio" / f"{sid}_C.wav"
        sf.write(target, values, 16000, subtype="FLOAT")
        row["speech_control_intervals"] = intervals
        row["arms"]["C"] = {"audio": str(target), "sha256": sha(target), "changed": bool(intervals),
                            "max_abs_from_M": float(np.max(np.abs(values - m))),
                            "rms": float(np.sqrt(np.mean(values ** 2)))}
    payload["protocol"] += "; C=equal-duration deterministic high-energy natural-speech location control with >=250ms gap from true pause"
    write(path, payload)
    print("control", sum(len(r["speech_control_intervals"]) for r in payload["rows"]), "events", flush=True)


def feature_variants(tts: torch.Tensor, natural_tokens: list[dict], tts_tokens: list[dict],
                     baseline: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """B changes only interpolation crossing matched phone spans; S only unmatched silences."""
    owners = frame_owners(len(baseline), natural_tokens)
    tts_owners = frame_owners(len(tts), tts_tokens)
    mapping, _ = matched_span_map(natural_tokens, tts_tokens)
    b = baseline.clone()
    s = baseline.clone()
    changed_b = changed_s = 0
    silence_indices = [i for i, owner in enumerate(tts_owners) if owner.is_silence]
    if not silence_indices:
        raise ValueError("TTS has no silence feature frames")
    for i, owner in enumerate(owners):
        target = mapping.get(owner.span_index)
        if target is None:
            if owner.is_silence:
                # Interpolate across the longest available TTS silence span.
                spans = [j for j, tok in enumerate(tts_tokens) if tok["is_silence"]]
                best = max(spans, key=lambda j: float(tts_tokens[j]["end_s"]) - float(tts_tokens[j]["start_s"]))
                allowed = [j for j, o in enumerate(tts_owners) if o.span_index == best]
                if allowed:
                    j = allowed[min(len(allowed) - 1, int(owner.relative_position * len(allowed)))]
                    s[i] = tts[j]
                    changed_s += 1
            continue
        tok = tts_tokens[target]
        allowed = [j for j, o in enumerate(tts_owners) if o.span_index == target]
        if allowed:
            p = (float(tok["start_s"]) + owner.relative_position *
                 (float(tok["end_s"]) - float(tok["start_s"]))) * 50 - .5
            q = min(max(p, allowed[0]), allowed[-1])
            if abs(q - p) > 1e-6:
                left = int(q)
                right = min(allowed[-1], left + 1)
                b[i] = tts[left] + (q - left) * (tts[right] - tts[left])
                changed_b += 1
    return {"B": b, "S": s}, {"B": changed_b, "S": changed_s}


def audio() -> None:
    rows = read(PRIOR / "manifest.json")["rows"]
    tokens = read(COHORT / "03_tokens_paired/tokens.json")["records"]
    tts_meta = read(COHORT / "01_tts_retry/tts_meta.json")["results"]
    pauses = missing_pauses()
    adapter = WavLMKNNVCAdapter.load_pretrained(
        device="cuda", source=REPO / "third_party/knn-vc", revision=KNN_VC_REVISION)
    result = []
    for row in rows:
        sid = row["id"]
        n, nsr = sf.read(row["natural"], dtype="float32")
        m, msr = sf.read(row["mfa"], dtype="float32")
        tts_path = Path(tts_meta[sid]["canonical_16k_audio"])
        t, tsr = sf.read(tts_path, dtype="float32")
        if (nsr, msr, tsr) != (16000, 16000, 16000) or len(n) != len(m):
            raise ValueError(f"clock mismatch {sid}")
        if sha(tts_path) != tokens[sid]["tts"]["audio_sha256"]:
            raise ValueError(f"TTS token/audio mismatch {sid}")
        nf = adapter.extract(torch.from_numpy(n).unsqueeze(0))
        tf = adapter.extract(torch.from_numpy(t).unsqueeze(0))
        nt, _ = extend_tokens_for_feature_tail(tokens[sid]["natural"]["tokens"], len(nf))
        tt, _ = extend_tokens_for_feature_tail(tokens[sid]["tts"]["tokens"], len(tf))
        baseline, meta = mfa_linear_target(len(nf), tf, nt, tt)
        baseline_wave, _ = exact_natural_length(adapter.vocode(baseline).numpy(), len(n))
        baseline_max_abs = float(np.max(np.abs(baseline_wave - m)))
        mapping_eligible = baseline_max_abs <= 5e-4
        variants, counts = feature_variants(tf, nt, tt, baseline)
        intervals = pauses.get(sid, [])
        audios = {"P": pause_zero(m, intervals)}
        for arm in ("B", "S"):
            if counts[arm] and mapping_eligible:
                audios[arm], _ = exact_natural_length(adapter.vocode(variants[arm]).numpy(), len(n))
            else:
                audios[arm] = m.copy()
        record = {"id": sid, "speaker": row["speaker"], "face": row["face"],
                  "natural": row["natural"], "mfa": row["mfa"],
                  "samples": len(n), "target_frames": row["target_frames"],
                  "tts": str(tts_path), "baseline_max_abs": baseline_max_abs,
                  "mapping_eligible": mapping_eligible,
                  "missing_pauses": intervals, "mapping_meta": meta, "changed_feature_frames": counts,
                  "arms": {}}
        for arm, values in audios.items():
            changed = bool(np.max(np.abs(values - m)) > 1e-7)
            path = OUT / "audio" / f"{sid}_{arm}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, values, 16000, subtype="FLOAT")
            record["arms"][arm] = {"audio": str(path), "sha256": sha(path), "changed": changed,
                                    "max_abs_from_M": float(np.max(np.abs(values - m))),
                                    "rms": float(np.sqrt(np.mean(values ** 2)))}
        result.append(record)
        print("audio", sid, "repro", round(baseline_max_abs, 7), "mapping_eligible", mapping_eligible, "frames", counts,
              "pauses", len(intervals), flush=True)
        write(OUT / "manifest.json", {"protocol": "frozen same 10 samples, face, WavLM-L6/HiFiGAN; P=missing-pause waveform zero with 20ms fades, B=within-phone interpolation clamp, S=unmatched silence uses longest TTS silence frames", "model": adapter.metadata(), "rows": result})


def render() -> None:
    for row in read(OUT / "manifest.json")["rows"]:
        sid = row["id"]
        for arm in ARMS:
            if not row["arms"][arm]["changed"]:
                continue
            root = OUT / "video" / sid / arm
            raw, normalized, video = root / "raw.mp4", root / "normalized.mkv", root / "video.mp4"
            if not raw.is_file():
                work = OUT / "wav2lip_work" / sid / arm
                (work / "temp").mkdir(parents=True, exist_ok=True)
                root.mkdir(parents=True, exist_ok=True)
                run([str(WAV2LIP_PY), str(WAV2LIP / "inference.py"),
                     "--checkpoint_path", str(WAV2LIP / "checkpoints/wav2lip_gan.pth"),
                     "--face", row["face"], "--audio", row["arms"][arm]["audio"],
                     "--outfile", str(raw), "--face_det_batch_size", "4",
                     "--wav2lip_batch_size", "4", "--nosmooth"],
                    OUT / "logs" / f"{sid}_{arm}_wav2lip.log", work)
            if not normalized.is_file():
                detail = canonicalize_tail(raw, normalized, target_frame_count=row["target_frames"], ffmpeg=FFMPEG)
                write(root / "normalized.json", detail)
            if not video.is_file():
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(normalized),
                     "-i", row["natural"], "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                     "-r", "25", "-frames:v", str(row["target_frames"]),
                     "-c:a", "aac", "-b:a", "192k", str(video)],
                    OUT / "logs" / f"{sid}_{arm}_mux.log")
            print("render", sid, arm, flush=True)


def crop(row: dict, arm: str) -> Path:
    sid = row["id"]
    root = OUT / "pipeline" / sid / arm
    ref = f"cp_{sid}_{arm}"
    path = root / "pycrop" / ref / "00000.avi"
    if not path.is_file():
        run([str(SYNCNET_PY), "run_pipeline.py", "--videofile",
             str(OUT / "video" / sid / arm / "video.mp4"), "--reference", ref,
             "--data_dir", str(root), "--min_track", "100", "--overwrite"],
            OUT / "logs" / f"{sid}_{arm}_pipeline.log", SYNCNET)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def score() -> None:
    rows = read(OUT / "manifest.json")["rows"]
    reference = {r["id"]: r["cells"] for r in read(PRIOR / "scores.json")["rows"]}
    engine = SyncNetEngine(batch_size=32, device="cuda")
    results = []
    try:
        for row in rows:
            sid = row["id"]
            audio_path = PRIOR / "clock" / sid / "N/audio.wav"
            audio_features, _ = engine.extract_audio(audio_path)
            base = reference[sid]["VM_AN"]
            baseline_crop = PRIOR / "fresh_pipeline" / sid / "M" / "pycrop" / f"split_{sid}_M" / "00000.avi"
            _, baseline_meta = engine.extract_visual(baseline_crop)
            cells = {"M": base}
            for arm in ARMS:
                if row["arms"][arm]["changed"]:
                    visual, meta = engine.extract_visual(crop(row, arm))
                    if meta["frame_count"] != baseline_meta["frame_count"]:
                        raise ValueError(f"frame count mismatch {sid}/{arm}: {meta}")
                    cells[arm] = score_matrix(engine.distance_matrix(visual, audio_features))
                else:
                    cells[arm] = base
            results.append({"id": sid, "speaker": row["speaker"], "cells": cells})
            write(OUT / "scores" / f"{sid}.json", results[-1])
            print("score", sid, {k: v["sync_c"] for k,v in cells.items()}, flush=True)
    finally:
        engine.close()
    write(OUT / "scores.json", {"protocol": "existing official natural PCM, same face and official crop, SyncNet V2 embeddings/score; unchanged arms inherit frozen M cell", "rows": results})


def analyze() -> None:
    rows = read(OUT / "manifest.json")["rows"]
    scores = {r["id"]: r["cells"] for r in read(OUT / "scores.json")["rows"]}
    summaries = {}
    for arm in ("M", *ARMS):
        values = np.array([scores[r["id"]][arm]["sync_c"] for r in rows])
        deltas = np.array([scores[r["id"]][arm]["sync_c"] - scores[r["id"]]["M"]["sync_c"] for r in rows])
        affected = [r for r in rows if r["arms"].get(arm, {}).get("changed")]
        summaries[arm] = {"sync_c_mean": float(values.mean()), "delta_vs_M_mean": float(deltas.mean()),
                          "improved": int(np.sum(deltas > 0)), "changed_samples": len(affected),
                          "affected_delta_mean": float(np.mean([scores[r["id"]][arm]["sync_c"] - scores[r["id"]]["M"]["sync_c"] for r in affected])) if affected else 0.0,
                          "sync_d_mean": float(np.mean([scores[r["id"]][arm]["sync_d"] for r in rows])),
                          "fixed_lag_distance_mean": float(np.mean([scores[r["id"]][arm]["fixed_lag_distance"] for r in rows])),
                          "boundary_offsets": int(sum(scores[r["id"]][arm]["offset_at_boundary"] for r in rows))}
    output = {"cohort_size": len(rows), "missing_pause_events": sum(len(r["missing_pauses"]) for r in rows),
              "summaries": summaries,
              "rows": [{"id": r["id"], "speaker": r["speaker"], "missing_pauses": r["missing_pauses"],
                        "baseline_max_abs": r["baseline_max_abs"], "changed_feature_frames": r["changed_feature_frames"],
                        "sync_c": {a: scores[r["id"]][a]["sync_c"] for a in ("M", *ARMS)}} for r in rows]}
    write(OUT / "analysis.json", output)
    lines = ["# MFA-linear 内容、停顿、韵律与映射诊断", "", "## 固定自然音轨的 SyncNet V2", "",
             "10 条同脸、同 Wav2Lip、同自然音轨和官方预处理链。M 是冻结基线。", "",
             "| Arm | Mean Sync-C | Δ vs M | 改善条数 | 实际改变条数 | 受影响条目平均 Δ | Mean Sync-D | 固定 0-lag 距离 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("M", *ARMS):
        s = summaries[arm]
        lines.append(f"| {arm} | {s['sync_c_mean']:.3f} | {s['delta_vs_M_mean']:+.3f} | {s['improved']}/10 | {s['changed_samples']} | {s['affected_delta_mean']:+.3f} | {s['sync_d_mean']:.3f} | {s['fixed_lag_distance_mean']:.3f} |")
    lines += ["", "P: 仅对已知缺失停顿的 M 波形做 20ms 淡入淡出静音；C: 等时长自然语音位置静音对照；B: 匹配音素内约束插值；S: 缺失停顿帧用 TTS 静音特征代替全局 TTS 回退。",
              "", "## 逐条 Sync-C", "", "| 样本 | 缺失停顿 | M | P | C | B | S |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in output["rows"]:
        c = r["sync_c"]
        lines.append(f"| {r['id']} | {len(r['missing_pauses'])} | {c['M']:.3f} | {c['P']:.3f} | {c['C']:.3f} | {c['B']:.3f} | {c['S']:.3f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def acoustic() -> None:
    """Check that each audio intervention changes the actual Wav2Lip mel input."""
    if str(WAV2LIP) not in sys.path:
        sys.path.insert(0, str(WAV2LIP))
    import audio as wav2lip_audio

    rows = read(OUT / "manifest.json")["rows"]
    output = []
    for row in rows:
        mel = {}
        for arm, path in (("N", row["natural"]), ("M", row["mfa"]),
                          *((arm, row["arms"][arm]["audio"]) for arm in ARMS)):
            mel[arm] = wav2lip_audio.melspectrogram(wav2lip_audio.load_wav(path, 16000))
        if len({x.shape for x in mel.values()}) != 1:
            raise ValueError(f"mel grid mismatch {row['id']}")
        count = row["target_frames"]
        times = (np.arange(count) + .5) / 25
        pause = np.zeros(count, dtype=bool)
        for a, b in row["missing_pauses"]:
            pause |= (times >= a) & (times < b)
        chunks = {}
        for arm, x in mel.items():
            chunks[arm] = np.stack([x[:, min(int(frame * 80 / 25), x.shape[1] - 16):][:, :16]
                                    for frame in range(count)])
        errors = {arm: np.mean(np.abs(chunks[arm] - chunks["N"]), axis=(1, 2))
                  for arm in ("M", *ARMS)}
        result = {"id": row["id"], "pause_windows": int(pause.sum()),
                  "all_window_mae": {arm: float(e.mean()) for arm, e in errors.items()},
                  "pause_window_mae": {arm: float(e[pause].mean()) if pause.any() else None
                                       for arm, e in errors.items()}}
        output.append(result)
    write(OUT / "acoustic.json", {"protocol": "Wav2Lip exact melspectrogram and 16-frame chunks; natural-mel MAE at all video frames and preidentified missing-pause frame centres", "rows": output})
    print("acoustic", len(output), "samples", flush=True)


def finalize() -> None:
    """Combine the independently scored arms into the reproducible run report."""
    pause = read(OUT / "analysis.json")
    pro = read(OUT / "prosody" / "analysis.json")
    verification = read(OUT / "prosody" / "verification.json")["rows"]
    content = read(OUT / "content_prosody_audit.json")["rows"]
    asr = read(OUT / "content_asr_audit_small.json")["rows"]
    pa = read(OUT / "acoustic.json")["rows"]
    prosody_acoustic = read(OUT / "prosody" / "acoustic.json")["rows"]
    lines = ["# MFA-linear 内容、停顿、韵律与映射消融", "",
             "同批 n=10、同自然音轨、同官方 SyncNet 时钟。完整协议与限制见 `basic-memory/docs/experiments/33-mfa-linear-content-pause-prosody-mapping.md`。",
             "", "## 停顿与映射", "",
             "| Arm | Sync-C | Δ vs M | 受影响条目平均 Δ | 改善/变更 | Sync-D |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ("M", *ARMS):
        s = pause["summaries"][arm]
        affected = "—" if arm == "M" else f"{s['affected_delta_mean']:+.3f}"
        improved = "—" if arm == "M" else f"{s['improved']}/{s['changed_samples']}"
        lines.append(f"| {arm} | {s['sync_c_mean']:.3f} | {s['delta_vs_M_mean']:+.3f} | {affected} | {improved} | {s['sync_d_mean']:.3f} |")
    lines += ["", "P=缺失停顿静音；C=等时长发声位置对照；B=音素内帧约束；S=静音特征回退。", "",
              "## 韵律", "", "| Arm | Sync-C | 配对差 | 改善数 |", "|---|---:|---:|---:|"]
    for arm in ("W", "F", "E"):
        s = pro["summaries"][arm]
        delta = 0.0 if abs(s["mean_delta"]) < .0005 else s["mean_delta"]
        lines.append(f"| {arm} | {s['mean_sync_c']:.3f} | {arm}−{s['reference']} = {delta:+.3f} | {s['improved']}/10 |")
    f0w = float(np.mean([r["median_abs_f0_semitones"]["W"] for r in verification]))
    f0f = float(np.mean([r["median_abs_f0_semitones"]["F"] for r in verification]))
    em = float(np.mean([r["median_abs_energy_db"]["M"] for r in verification]))
    ee = float(np.mean([r["median_abs_energy_db"]["E"] for r in verification]))
    lines += ["", f"F0 误差 W→F = {f0w:.3f}→{f0f:.3f} 半音；能量误差 M→E = {em:.3f}→{ee:.3f} dB。",
              "", "## 内容与 Wav2Lip 输入", "",
              f"输入文本一致 {sum(x['text_exact_equal'] for x in content)}/10；强制对齐音素标签匹配 {sum(x['matched_forced_phone_labels'] for x in content)}/{sum(x['natural_phones'] for x in content)}。",
              "Whisper-small 平均 CER（N/TTS/M）：" + "/".join(f"{np.mean([r['arms'][a]['cer'] for r in asr]):.3f}" for a in ("N", "TTS", "M")) + "。",
              "缺失停顿窗口 Wav2Lip mel MAE（M/P/C/B/S）：" + "/".join(f"{np.mean([r['pause_window_mae'][a] for r in pa if r['pause_windows']]):.3f}" for a in ("M", *ARMS)) + "。",
              "全帧 Wav2Lip mel MAE（M/W/F/E）：" + "/".join(f"{np.mean([r['mel_mae_to_N'][a] for r in prosody_acoustic]):.3f}" for a in ("M", "W", "F", "E")) + "。",
              "", "## 解释", "",
              "缺失停顿位置修复稳定改善自然音轨 Sync-C，等时长发声位置对照下降；静音特征回退同方向。",
              "F0 和能量干预确实命中声学目标，但平均 Sync-C 未提升。WORLD 原样重合成自身下降较多，F 必须与 W 比。",
              "a1_057 的旧 M 无法用当前冻结映射精确重现，B/S 在该条未运行；本批仅作机制诊断。",
              "", "## 逐样本", "", "| ID | M | P | C | B | S | W | F | E |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pro_by_id = {r["id"]: r["cells"] for r in pro["rows"]}
    for row in pause["rows"]:
        sid = row["id"]
        c = row["sync_c"]
        p = pro_by_id[sid]
        lines.append(f"| {sid} | " + " | ".join(f"{c[a]:.3f}" for a in ("M", *ARMS)) +
                     " | " + " | ".join(f"{p[a]['sync_c']:.3f}" for a in ("W", "F", "E")) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("finalized", OUT / "report.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audio", "control", "render", "score", "acoustic", "analyze", "finalize"))
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == "__main__":
    main()
