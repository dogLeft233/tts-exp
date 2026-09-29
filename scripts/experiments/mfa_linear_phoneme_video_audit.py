#!/usr/bin/env python3
"""Audit MFA-linear speech timing and generated mouth motion against natural speech.

Uses the existing 10-speaker short-video probe and a fresh MFA alignment of the
MFA-linear WAVs. Local SyncNet distances are diagnostic 200 ms windows, not
phoneme-level Sync-C scores. Run stages: mouth, syncnet, analyze.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
COHORT = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
PRIOR = REPO / "runs/mfa_linear_nas_short_expanded_20260924"
OUT = REPO / "runs/mfa_linear_phoneme_video_audit_20260925"
FPS = 25
TONE = re.compile("[\u02e5-\u02e9]")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ids() -> list[str]:
    return load_json(PRIOR / "summary.json")["protocol"]["sample_ids"]


def media(id: str, arm: str) -> Path:
    return PRIOR / id / "input" / f"{arm}.mp4"


def crop(id: str, arm: str) -> tuple[Path, Path]:
    name = f"pn100_s{id}_v{arm}_raw_aN"
    ref = f"mfa_linear_video_retiming_{name}"
    root = PRIOR / id / "06_official/cells" / name / "pipeline"
    return root / "pycrop" / ref / "00000.avi", root / "pyavi" / ref / "audio.wav"


def mouth_stage() -> None:
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(REPO / "checkpoints/mediapipe/face_landmarker.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
        for id in ids():
            for arm in ("N", "M"):
                dest = OUT / "mouth" / f"{id}_{arm}.npz"
                if dest.is_file():
                    print("cached", id, arm, flush=True)
                    continue
                capture = cv2.VideoCapture(str(media(id, arm)))
                openings, widths, centers = [], [], []
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = model.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        )
                        if not result.face_landmarks:
                            openings.append(float("nan"))
                            widths.append(float("nan"))
                            centers.append((float("nan"), float("nan")))
                            continue
                        face = result.face_landmarks[0]
                        eye = np.hypot(face[33].x - face[263].x, face[33].y - face[263].y)
                        openings.append(float(np.hypot(face[13].x - face[14].x, face[13].y - face[14].y) / eye))
                        widths.append(float(np.hypot(face[61].x - face[291].x, face[61].y - face[291].y) / eye))
                        centers.append(((face[13].x + face[14].x) / 2, (face[13].y + face[14].y) / 2))
                finally:
                    capture.release()
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, opening=np.asarray(openings), width=np.asarray(widths), center=np.asarray(centers))
                print("mouth", id, arm, len(openings), "missing", int(np.isnan(openings).sum()), flush=True)


def syncnet_stage() -> None:
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    engine = SyncNetEngine(batch_size=32, device="cuda")
    try:
        for id in ids():
            dest = OUT / "syncnet" / f"{id}.npz"
            if dest.is_file():
                print("cached syncnet", id, flush=True)
                continue
            n_video, n_audio = crop(id, "N")
            m_video, m_audio = crop(id, "M")
            if not all(p.is_file() for p in (n_video, n_audio, m_video, m_audio)):
                raise FileNotFoundError(f"missing existing official SyncNet crop for {id}")
            an, _ = engine.extract_audio(n_audio)
            am, _ = engine.extract_audio(m_audio)
            if an.shape != am.shape or not np.array_equal(an, am):
                raise ValueError(f"{id}: N and M official audio tracks differ")
            vn, _ = engine.extract_visual(n_video)
            vm, _ = engine.extract_visual(m_video)
            dn = engine.distance_matrix(vn, an)
            dm = engine.distance_matrix(vm, an)
            count = min(len(dn), len(dm))
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, distance_N=dn[:count].astype(np.float32), distance_M=dm[:count].astype(np.float32))
            print("syncnet", id, count, "zero lag delta", round(float(np.mean(dm[:count, 15] - dn[:count, 15])), 3), flush=True)
    finally:
        engine.close()


def phones(id: str, record: dict) -> tuple[list[dict], list[dict]]:
    n = [p for p in record["natural"]["tokens"] if not p["is_silence"]]
    speaker = record["speaker_id"]
    path = OUT / "mfa_textgrids" / speaker / f"{speaker}_{id}_mfa_linear.json"
    raw = load_json(path)["tiers"]["phones"]["entries"]
    m = [
        {"token": TONE.sub("", label), "start_s": float(a), "end_s": float(b)}
        for a, b, label in raw if label not in ("", "sil", "sp", "spn")
    ]
    return n, m


def analyze_stage() -> None:
    import soundfile as sf

    tokens = load_json(COHORT / "03_tokens_paired/tokens.json")["records"]
    scores = {r["sample_id"]: r for r in load_json(PRIOR / "summary.json")["rows"]}
    mfa_outputs = load_json(COHORT / "04_mfa_linear/summary.json")["results"]
    all_rows, samples, silences = [], [], []
    for id in ids():
        record = tokens[id]
        nphones, mphones = phones(id, record)
        matcher = SequenceMatcher(
            a=[p["token"] for p in nphones], b=[p["token"] for p in mphones], autojunk=False
        )
        pairs = [(block.a + k, block.b + k) for block in matcher.get_matching_blocks() for k in range(block.size)]
        mouths = {arm: np.load(OUT / "mouth" / f"{id}_{arm}.npz") for arm in ("N", "M")}
        d = np.load(OUT / "syncnet" / f"{id}.npz")
        length = min(len(mouths["N"]["opening"]), len(mouths["M"]["opening"]))
        opening_n, opening_m = [mouths[a]["opening"][:length] for a in ("N", "M")]
        scale = float(np.nanpercentile(opening_n, 90) - np.nanpercentile(opening_n, 10))
        if scale <= 0:
            scale = 1.0
        dn, dm = d["distance_N"][:, 15], d["distance_M"][:, 15]
        # The embedding window covers frames [i, i+4]; midpoint is frame i+2.
        local_t = (np.arange(min(len(dn), len(dm))) + 2.5) / FPS
        lip_correlation = float(np.corrcoef(opening_n, opening_m)[0, 1])
        lag_scores = []
        for lag in range(-5, 6):
            a = opening_n[lag:] if lag >= 0 else opening_n[:lag]
            b = opening_m[:-lag] if lag > 0 else opening_m[-lag:] if lag < 0 else opening_m
            lag_scores.append((lag, float(np.corrcoef(a, b)[0, 1])))
        best_lag, best_correlation = max(lag_scores, key=lambda item: item[1])

        natural_audio, natural_sr = sf.read(record["natural"]["audio"], dtype="float32")
        mfa_audio, mfa_sr = sf.read(mfa_outputs[id]["audio_path"], dtype="float32")
        if natural_sr != 16_000 or mfa_sr != 16_000 or len(natural_audio) != len(mfa_audio):
            raise ValueError(f"{id}: natural/M audio sample-clock mismatch")
        full_n, full_t = record["natural"]["tokens"], record["tts"]["tokens"]
        original_matcher = SequenceMatcher(
            a=[p["token"] for p in full_n], b=[p["token"] for p in full_t], autojunk=False
        )
        matched_n = {block.a + k for block in original_matcher.get_matching_blocks() for k in range(block.size)}
        for index, phone in enumerate(full_n):
            a, b = float(phone["start_s"]), float(phone["end_s"])
            if not phone["is_silence"] or b - a < .1:
                continue
            n_slice = natural_audio[int(a*natural_sr):int(b*natural_sr)]
            m_slice = mfa_audio[int(a*mfa_sr):int(b*mfa_sr)]
            frame_mask = ((np.arange(length) + .5) / FPS >= a) & ((np.arange(length) + .5) / FPS < b)
            local_mask = (local_t >= a) & (local_t < b)
            silences.append({
                "sample_id": id, "start_s": a, "end_s": b,
                "duration_s": round(b-a, 3), "matched_to_tts_silence": index in matched_n,
                "natural_rms": float(np.sqrt(np.mean(n_slice*n_slice))),
                "mfa_rms": float(np.sqrt(np.mean(m_slice*m_slice))),
                "mouth_open_delta": float(np.mean(opening_m[frame_mask] - opening_n[frame_mask])) if frame_mask.any() else None,
                "syncnet_distance_delta": float(np.mean(dm[:len(local_t)][local_mask] - dn[:len(local_t)][local_mask])) if local_mask.any() else None,
            })
        for ni, mi in pairs:
            np_, mp_ = nphones[ni], mphones[mi]
            a, b = float(np_["start_s"]), float(np_["end_s"])
            idx = np.arange(length)
            frame_mask = ((idx + .5) / FPS >= a) & ((idx + .5) / FPS < b)
            local_mask = (local_t >= a) & (local_t < b)
            open_diff = opening_m[frame_mask] - opening_n[frame_mask]
            valid = np.isfinite(open_diff)
            local_delta = dm[:len(local_t)][local_mask] - dn[:len(local_t)][local_mask]
            row = {
                "sample_id": id, "phone": np_["token"], "natural_phone_index": ni,
                "natural_start_s": a, "natural_end_s": b, "natural_duration_s": b-a,
                "mfa_start_s": mp_["start_s"], "mfa_end_s": mp_["end_s"],
                "mfa_center_error_ms": round(500 * (mp_["start_s"] + mp_["end_s"] - a - b), 1),
                "mfa_start_error_ms": round(1000 * (mp_["start_s"] - a), 1),
                "mfa_end_error_ms": round(1000 * (mp_["end_s"] - b), 1),
                "mouth_frame_count": int(valid.sum()),
                "mouth_abs_gap_norm": round(float(np.mean(np.abs(open_diff[valid])) / scale), 3) if valid.any() else None,
                "mouth_signed_gap_norm": round(float(np.mean(open_diff[valid]) / scale), 3) if valid.any() else None,
                "syncnet_window_count": int(local_mask.sum()),
                "syncnet_distance_delta": round(float(np.mean(local_delta)), 3) if len(local_delta) else None,
            }
            all_rows.append(row)
        sample_rows = [r for r in all_rows if r["sample_id"] == id]
        sample = {
            "sample_id": id, "speaker_id": record["speaker_id"],
            "natural_phone_count": len(nphones), "mfa_phone_count": len(mphones),
            "matched_phone_count": len(pairs),
            "mfa_center_abs_error_median_ms": round(float(np.median([abs(r["mfa_center_error_ms"]) for r in sample_rows])), 1),
            "mfa_center_abs_error_p90_ms": round(float(np.percentile([abs(r["mfa_center_error_ms"]) for r in sample_rows], 90)), 1),
            "mouth_mean_abs_gap_norm": round(float(np.nanmean(np.abs(opening_m - opening_n)) / scale), 3),
            "lip_open_corr_zero_lag": round(lip_correlation, 3),
            "lip_open_best_lag_frames": best_lag,
            "lip_open_best_lag_corr": round(best_correlation, 3),
            "unmatched_pause_count_ge_100ms": sum(
                not p["matched_to_tts_silence"] for p in silences if p["sample_id"] == id
            ),
            "mouth_missing_N": int(np.isnan(opening_n).sum()), "mouth_missing_M": int(np.isnan(opening_m).sum()),
            "syncnet_zero_lag_N": round(float(np.mean(dn)), 3),
            "syncnet_zero_lag_M": round(float(np.mean(dm)), 3),
            "official_sync_c_N": scores[id]["scores"]["N_raw"]["sync_c"],
            "official_sync_c_M": scores[id]["scores"]["M_raw"]["sync_c"],
        }
        samples.append(sample)
        print("analyze", id, sample["matched_phone_count"], "/", len(nphones), "median error", sample["mfa_center_abs_error_median_ms"], flush=True)
    payload = {
        "protocol": "fresh M MFA alignment, natural phone grid, same natural audio, N/M raw Wav2Lip input videos; 25 fps",
        "syncnet_note": "fixed zero lag 5-frame (200 ms) embedding distances; per-phone aggregates include neighbor phonemes and are diagnostic only",
        "mouth_note": "MediaPipe lip aperture distance(13,14)/distance(33,263), gap normalized by N p90-p10 per sample",
        "samples": samples, "phones": all_rows, "silences_ge_100ms": silences,
    }
    write_json(OUT / "analysis.json", payload)


def visualize_stage() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = load_json(OUT / "analysis.json")
    token_records = load_json(COHORT / "03_tokens_paired/tokens.json")["records"]
    for id in ids():
        n = np.load(OUT / "mouth" / f"{id}_N.npz")["opening"]
        m = np.load(OUT / "mouth" / f"{id}_M.npz")["opening"]
        distance = np.load(OUT / "syncnet" / f"{id}.npz")
        count = min(len(distance["distance_N"]), len(distance["distance_M"]))
        local_delta = distance["distance_M"][:count, 15] - distance["distance_N"][:count, 15]
        times = (np.arange(count) + 2.5) / FPS
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot((np.arange(len(n)) + .5) / FPS, n, label="natural video", linewidth=1.4)
        ax1.plot((np.arange(len(m)) + .5) / FPS, m, label="MFA-linear video", linewidth=1.3)
        ax1.set_ylabel("normalized lip opening")
        ax1.legend(loc="upper right")
        ax2.plot(times, local_delta, color="purple", linewidth=1)
        ax2.axhline(0, color="gray", linewidth=.8)
        ax2.set_ylabel("local SyncNet M-N distance")
        ax2.set_xlabel("natural audio time (s)")
        tokens = [p for p in token_records[id]["natural"]["tokens"] if not p["is_silence"]]
        for i, p in enumerate(tokens):
            t = float(p["start_s"])
            ax1.axvline(t, color="gray", alpha=.2, linewidth=.6)
            if i % 3 == 0:
                ax1.text(t, 1.01, p["token"], transform=ax1.get_xaxis_transform(), fontsize=7, rotation=60)
        ax1.set_xlim(0, min(len(n), len(m)) / FPS)
        fig.suptitle(f"{id}: generated lip opening versus natural; local SyncNet uses 200 ms windows")
        fig.tight_layout()
        path = OUT / "figures" / f"{id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("figure", id, path, flush=True)


def examples_stage() -> None:
    examples = [
        ("a1_044", "p", .97, 1.12),
        ("a1_019", "a", 1.76, 2.05),
        ("a1_028", "ŋ", 3.28, 3.40),
        ("a1_010", "n", 4.95, 5.11),
        ("a1_049", "pause", 3.11, 3.43),
        ("a1_010", "pause", 1.24, 1.43),
    ]
    for id, label, start, end in examples:
        times = [max(0, start - .04), start + .04, (start + end) / 2, end - .04, end + .04]
        rows = []
        for arm in ("N", "M"):
            centers = np.load(OUT / "mouth" / f"{id}_{arm}.npz")["center"]
            capture = cv2.VideoCapture(str(media(id, arm)))
            panels = []
            try:
                for t in times:
                    frame_index = min(len(centers)-1, max(0, int(t * FPS)))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"failed to read {id} {arm} frame {frame_index}")
                    h, w = frame.shape[:2]
                    cx = int(centers[frame_index, 0] * w)
                    cy = int(centers[frame_index, 1] * h)
                    x1, x2 = max(0, cx-90), min(w, cx+90)
                    y1, y2 = max(0, cy-65), min(h, cy+65)
                    panel = cv2.resize(frame[y1:y2, x1:x2], (360, 260))
                    cv2.putText(panel, f"{arm} {t:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    panels.append(panel)
            finally:
                capture.release()
            rows.append(np.concatenate(panels, axis=1))
        dest = OUT / "examples" / f"{id}_{label}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), np.concatenate(rows, axis=0))
        print("example", dest, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("mouth", "syncnet", "analyze", "visualize", "examples"))
    stage = parser.parse_args().stage
    {"mouth": mouth_stage, "syncnet": syncnet_stage, "analyze": analyze_stage, "visualize": visualize_stage, "examples": examples_stage}[stage]()


if __name__ == "__main__":
    main()
