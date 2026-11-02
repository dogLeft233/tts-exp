"""Independent CPU audit: raw official logs/distance arrays and paired statistics.

Does not import either the execution runner or the analysis implementation.
"""
import argparse
import hashlib
import json
import pickle
import re
import wave
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(run, source):
    errors, track_checks, media_checks, audio_frontend_checks = [], [], [], []
    video = {(r["sample_id"], r["role"]): r for r in load(source / "video_manifest.json")["records"]}
    audio = {(r["sample_id"], r["role"]): r for r in load(source / "audio_manifest.json")["records"]}
    receipts = {}
    for key, v in sorted(video.items()):
        a = audio[key]
        path = run / "scores" / f"{key[0]}_{key[1]}.json"
        if not path.exists():
            errors.append(f"Missing receipt {key}")
            continue
        r = load(path)
        receipts[key] = r
        vh, ah = digest(v["path"]), digest(a["path"])
        with wave.open(a["path"], "rb") as handle:
            fmt = [handle.getnchannels(), handle.getsampwidth(), handle.getframerate()]
            pcm = hashlib.sha256(handle.readframes(handle.getnframes())).hexdigest()
        ok = vh == v["sha256"] and ah == a["sha256"] and pcm == a["pcm_sha256"] and fmt == [1, 2, 16000]
        media_checks.append({"sample_id": key[0], "role": key[1], "PASS": ok,
                             "video_sha256": vh, "audio_sha256": ah, "pcm_sha256": pcm})
        if not ok:
            errors.append(f"Source media hash/format mismatch {key}")
        pretrack = run / "data" / "pyavi" / r["reference"] / "audio.wav"
        if pretrack.exists():
            def raw_pcm(path):
                with wave.open(str(path), "rb") as h:
                    if (h.getnchannels(), h.getsampwidth(), h.getframerate()) != (1, 2, 16000):
                        raise ValueError("Unexpected pretrack audio format")
                    return np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float64)
            canonical, official = raw_pcm(a["path"]), raw_pcm(pretrack)
            cc = correlate(official, canonical, method="fft")
            lags = correlation_lags(len(official), len(canonical))
            valid = np.abs(lags) <= 1600
            delay = int(lags[valid][np.argmax(cc[valid])])
            aa = canonical[max(0, -delay):min(len(canonical), len(official)-delay)]
            bb = official[max(0, delay):min(len(official), len(canonical)+delay)]
            correlation = float(np.corrcoef(aa, bb)[0, 1]) if aa.std() > 0 and bb.std() > 0 else None
            audio_frontend_checks.append({"sample_id": key[0], "role": key[1],
                "canonical_samples": len(canonical), "official_pretrack_samples": len(official),
                "extra_samples": len(official)-len(canonical),
                "best_pcm_correlation_delay_samples_within_100ms": delay,
                "best_pcm_correlation_delay_ms": delay / 16,
                "aligned_correlation": correlation,
                "aligned_pcm_identical": bool(np.array_equal(aa, bb)),
                "aligned_rms_ratio": float(np.linalg.norm(bb) / np.linalg.norm(aa)) if np.linalg.norm(aa) else None,
                "interpretation": "Descriptive official preprocessing change; not a lip-sync error or isolated causal effect"})
        logpath = Path(r["syncnet_log"])
        if not logpath.is_file():
            errors.append(f"No syncnet log {key}")
            continue
        raw = logpath.read_text(encoding="utf-8", errors="replace")
        cs = [float(x) for x in re.findall(r"\bConfidence:\s*([-+\d.eE]+)", raw)]
        ds = [float(x) for x in re.findall(r"\bMin dist:\s*([-+\d.eE]+)", raw)]
        offsets = [int(x) for x in re.findall(r"\bAV offset:\s*(-?\d+)", raw)]
        if not cs or not (len(cs) == len(ds) == len(offsets)):
            errors.append(f"Missing/misaligned printed tracks {key}")
            continue
        if [r["sync_c"], r["sync_d"], r["av_offset"]] != [cs[0], ds[0], offsets[0]]:
            errors.append(f"First-track receipt mismatch {key}")
        if len(r["tracks"]) != len(cs):
            errors.append(f"Captured track count mismatch {key}")
        ap = Path(r["activesd_path"])
        tp = Path(r["tracks_pickle_path"])
        arrays = pickle.loads(ap.read_bytes())
        tracks = pickle.loads(tp.read_bytes())
        if len(arrays) != len(cs) or len(tracks) != len(cs):
            errors.append(f"Pickle track count mismatch {key}")
            continue
        for i, dist in enumerate(arrays):
            x = np.asarray(dist)
            if x.ndim != 2 or x.shape[1] != 31 or not np.isfinite(x).all():
                errors.append(f"Invalid distance matrix {key}:{i}")
                continue
            curve = x.astype(np.float64).mean(axis=0)
            idx = int(np.argmin(curve))
            c = float(np.median(curve) - curve[idx])
            d = float(curve[idx])
            off = 15 - idx
            frames = np.asarray(tracks[i]["track"]["frame"])
            check = {"sample_id": key[0], "role": key[1], "track_index": i,
                     "matrix_shape": list(x.shape), "frame_count": len(frames),
                     "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
                     "C_unrounded_cpu": c, "D_unrounded_cpu": d, "offset_cpu": off,
                     "C_printed": cs[i], "D_printed": ds[i], "offset_printed": offsets[i],
                     "max_print_error": max(abs(c - cs[i]), abs(d - ds[i]))}
            # Official reports 3 decimals; float64/torch float32 reduction may
            # differ by a few ulps, so allow 0.00051 rather than demand equality.
            check["PASS"] = check["max_print_error"] <= 0.00051 and off == offsets[i] and 0 < len(x) <= len(frames) - 5
            track_checks.append(check)
            if not check["PASS"]:
                errors.append(f"Raw distance/printed/support mismatch {key}:{i}")
    result = load(run / "analysis.json")
    old = {r["sample_id"]: r for r in load(source / "analysis.json")["records"]}
    person_values, max_person_error = {}, 0.0
    for person in result["records"]:
        sid = person["sample_id"]
        computed = {}
        for metric, field in [("C", "sync_c"), ("D", "sync_d"), ("offset", "av_offset")]:
            vals = [receipts[sid, r][field] for r in ["N", "R", "T1", "T2", "T3"]]
            n, rr, t = vals[0], vals[1], sum(vals[2:]) / 3
            computed[metric] = {"T-N": t-n, "T-R": t-rr, "R-N": rr-n}
            max_person_error = max(max_person_error, *[abs(computed[metric][k] - person["official_contrasts"][metric][k]) for k in computed[metric]])
        changes = {}
        for policy, roles in old[sid]["policies"].items():
            n, rr = roles["N"]["C"], roles["R"]["C"]
            t = sum(roles[r]["C"] for r in ["T1", "T2", "T3"]) / 3
            previous = {"T-N": t-n, "T-R": t-rr, "R-N": rr-n}
            changes[policy] = {k: computed["C"][k] - previous[k] for k in previous}
        qualified = [r for r in ["T1", "T2", "T3"] if old[sid]["draw_reliability"][r]["qualified"]]
        qcontrasts = {}
        for metric, field in [("C", "sync_c"), ("D", "sync_d"), ("offset", "av_offset")]:
            t = sum(receipts[sid, r][field] for r in qualified) / len(qualified)
            n, rr = receipts[sid, "N"][field], receipts[sid, "R"][field]
            qcontrasts[metric] = {"T-N": t-n, "T-R": t-rr, "R-N": rr-n}
        person_values[sid] = {"metrics": computed, "changes": changes, "qualified": qcontrasts}
    if max_person_error > 1e-12:
        errors.append("Per-speaker contrast error")
    bootstrap_checks, max_stat_error = 0, 0.0
    proto = load(run / "analysis_protocol.json")["bootstrap"]
    for name, block in result["cohorts"].items():
        people = [p for p in result["records"] if name == "all40" or p["cohort"] == name]
        ids = [p["sample_id"] for p in people]
        if not ids:
            continue
        # Independent weighted-frequency calculation, rather than indexing
        # the contrasts and calling mean as the analysis does.
        draws = np.random.Generator(np.random.PCG64(proto["seed"])).integers(0, len(ids), (proto["replicates"], len(ids)))
        weights = np.asarray([np.bincount(d, minlength=len(ids)) for d in draws], dtype=np.float64) / len(ids)
        pairs = []
        for metric in ["C", "D", "offset"]:
            for k in ["T-N", "T-R", "R-N"]:
                pairs.append(([person_values[s]["metrics"][metric][k] for s in ids], block["metrics"][metric]["contrasts"][k]))
                pairs.append(([person_values[s]["qualified"][metric][k] for s in ids], block["qualified_supplement"][metric][k]))
        for policy in ["common", "native_full", "equal_count_spread"]:
            for k in ["T-N", "T-R", "R-N"]:
                pairs.append(([person_values[s]["changes"][policy][k] for s in ids], block["protocol_changes_C"][policy][k]))
        for role in ["T1", "T2", "T3"]:
            for k, control in [("T-N", "N"), ("T-R", "R")]:
                pairs.append(([receipts[s, role]["sync_c"] - receipts[s, control]["sync_c"] for s in ids], block["per_seed_C"][role][k]))
        for values, published in pairs:
            x = np.asarray(values)
            boot = weights @ x
            actual = [float(sum(values) / len(values)), *np.percentile(boot, [2.5, 97.5, 1.25, 98.75])]
            reported = [published["mean"], *published["ci95"], *published["ci97_5"]]
            err = float(np.max(np.abs(np.asarray(actual) - reported)))
            max_stat_error = max(max_stat_error, err)
            bootstrap_checks += 1
    if max_stat_error > 1e-12:
        errors.append("Independent bootstrap mismatch")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "audit_source_sha256": digest(__file__), "analysis_sha256": digest(run / "analysis.json"),
            "source_cells_checked": len(media_checks), "official_tracks_checked": len(track_checks),
            "bootstrap_summaries_checked": bootstrap_checks, "max_person_contrast_error": max_person_error,
            "max_bootstrap_error": max_stat_error,
            "max_rounded_score_error": max((x["max_print_error"] for x in track_checks), default=None),
            "media_checks": media_checks, "track_checks": track_checks,
            "audio_frontend_checks": audio_frontend_checks,
            "limitations": ["CPU recomputation of saved distances, not independent model forwards",
                            "Source media rehashed independently; runner's full remux validation is separate",
                            "No human perceptual or lip-accuracy audit"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.run, args.source)
    with (args.run / "independent_audit.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["media_checks", "track_checks", "audio_frontend_checks"]}))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
