#!/usr/bin/env python3
"""Read-only content metadata and WORLD F0/energy audit for the frozen n=10 set."""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
PRIOR = REPO / "runs/mfa_linear_vocoder_wav2lip_split_20260925"
OUT = REPO / "runs/mfa_linear_content_pause_prosody_mapping_20260925"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y, sr = sf.read(path, dtype="float64")
    if sr != 16000 or y.ndim != 1:
        raise ValueError(path)
    f0, times = pyworld.dio(y, sr, f0_floor=70., f0_ceil=450., frame_period=20.)
    f0 = pyworld.stonemask(y, f0, times, sr)
    rms = np.empty(len(times))
    half = round(.020 * sr)
    for i, t in enumerate(times):
        c = round(t * sr)
        seg = y[max(0, c - half):min(len(y), c + half)]
        rms[i] = float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.
    return f0, rms


def at(values: np.ndarray, time: float) -> float:
    return float(values[min(len(values) - 1, max(0, round(time * 50)))])


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> None:
    rows = read(PRIOR / "manifest.json")["rows"]
    tokens = read(COHORT / "03_tokens_paired/tokens.json")["records"]
    meta = read(COHORT / "01_tts_retry/tts_meta.json")["results"]
    results = []
    for row in rows:
        sid = row["id"]
        rec = tokens[sid]
        tn = rec["natural"]["tokens"]
        tt = rec["tts"]["tokens"]
        non_n = [x for x in tn if not x["is_silence"]]
        non_t = [x for x in tt if not x["is_silence"]]
        matcher = SequenceMatcher(a=[x["token"] for x in non_n], b=[x["token"] for x in non_t], autojunk=False)
        pairs = [(non_n[b.a + k], non_t[b.b + k]) for b in matcher.get_matching_blocks() for k in range(b.size)]
        f0n, en = track(Path(row["natural"]))
        f0t, et = track(Path(meta[sid]["canonical_16k_audio"]))
        f0m, em = track(Path(row["mfa"]))
        f0r, er = track(Path(row["reconstruction"]))
        f0_abs_nt, f0_abs_nm, f0_abs_nr, db_nt, db_nm, db_nr, duration_ratios = [], [], [], [], [], [], []
        f0n_samples, f0t_samples, f0m_samples = [], [], []
        en_samples, et_samples, em_samples = [], [], []
        for n, t in pairs:
            dn = float(n["end_s"]) - float(n["start_s"])
            dt = float(t["end_s"]) - float(t["start_s"])
            duration_ratios.append(dt / dn)
            for frac in (.2, .4, .6, .8):
                time_n = float(n["start_s"]) + frac * dn
                time_t = float(t["start_s"]) + frac * dt
                pn, pt, pm, pr = at(f0n, time_n), at(f0t, time_t), at(f0m, time_n), at(f0r, time_n)
                rn, rt, rm, rr = at(en, time_n), at(et, time_t), at(em, time_n), at(er, time_n)
                if pn > 0 and pt > 0:
                    f0_abs_nt.append(abs(12 * np.log2(pt / pn)))
                    f0n_samples.append(pn)
                    f0t_samples.append(pt)
                if pn > 0 and pm > 0:
                    f0_abs_nm.append(abs(12 * np.log2(pm / pn)))
                    f0m_samples.append(pm)
                if pn > 0 and pr > 0:
                    f0_abs_nr.append(abs(12 * np.log2(pr / pn)))
                if rn > 1e-4 and rt > 1e-4:
                    db_nt.append(20 * np.log10(rt / rn))
                    en_samples.append(rn)
                    et_samples.append(rt)
                if rn > 1e-4 and rm > 1e-4:
                    db_nm.append(20 * np.log10(rm / rn))
                    em_samples.append(rm)
                if rn > 1e-4 and rr > 1e-4:
                    db_nr.append(20 * np.log10(rr / rn))
        results.append({
            "id": sid, "text_exact_equal": rec["transcript"] == meta[sid]["transcript"],
            "text": rec["transcript"], "natural_phones": len(non_n), "tts_phones": len(non_t),
            "matched_forced_phone_labels": len(pairs),
            "unknown_speech_pairs": len(rec["unknown_speech_pairs"]),
            "median_tts_to_natural_phone_duration_ratio": median_or_none(duration_ratios),
            "median_abs_f0_semitones_N_TTS": median_or_none(f0_abs_nt),
            "median_abs_f0_semitones_N_M": median_or_none(f0_abs_nm),
            "median_abs_f0_semitones_N_R": median_or_none(f0_abs_nr),
            "median_db_TTS_minus_N": median_or_none(db_nt),
            "median_db_M_minus_N": median_or_none(db_nm),
            "median_db_R_minus_N": median_or_none(db_nr),
            "paired_voiced_points_N_TTS": len(f0_abs_nt), "paired_voiced_points_N_M": len(f0_abs_nm),
            "paired_voiced_points_N_R": len(f0_abs_nr),
            "paired_energy_points_N_TTS": len(db_nt), "paired_energy_points_N_M": len(db_nm),
        })
        print("audit", sid, "phones", len(pairs), "F0", results[-1]["median_abs_f0_semitones_N_TTS"], results[-1]["median_abs_f0_semitones_N_M"], flush=True)
    output = {"protocol": "Exact text and independent natural/TTS MFA label comparison (both forced from supplied text); WORLD DIO+StoneMask 20ms F0, 40ms RMS, matched phones sampled at 20/40/60/80% relative position. F0 and dB are descriptive; forced phone labels are not independent ASR.",
              "rows": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "content_prosody_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
