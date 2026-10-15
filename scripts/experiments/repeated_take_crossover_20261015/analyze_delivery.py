"""Summarize the frozen repeated-take crossover from saved SyncNet distances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUN = REPO / "runs" / "repeated_take_crossover_20261015_attempt01"
ROLES = ("N1", "N2", "T1", "T2")
LAGS = tuple(range(-15, 16))
PROTOCOL_SHA256 = "f86fa38175cd8adecadc2c638b6f90f5862824d57d21c545a9fd24e9b1f3369d"
MANIFEST_SHA256 = "ae2711bea33339de911e15cee0832488af0b411ffc224d81721c88a892bc12ae"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def three(value: float) -> str:
    return f"{value:.3f}"


def svg_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def heatmap_svg(matrix: dict[str, dict[str, float]]) -> str:
    width, height, left, top, cell = 720, 610, 170, 130, 110
    values = [matrix[v][a] for v in ROLES for a in ROLES]
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.0

    def color(value: float) -> str:
        t = (value - low) / (high - low)
        stops = ((43, 108, 176), (247, 247, 247), (197, 48, 48))
        if t < 0.5:
            a, b, q = stops[0], stops[1], 2 * t
        else:
            a, b, q = stops[1], stops[2], 2 * (t - 0.5)
        rgb = tuple(round(x + q * (y - x)) for x, y in zip(a, b))
        return "#" + "".join(f"{x:02x}" for x in rgb)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:24px;font-weight:700}.sub{font-size:14px;fill:#4b5563}.lab{font-size:15px}.val{font-size:17px;font-weight:700}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="42" text-anchor="middle" class="title">Mean Sync-C by driver and evaluator take</text>',
        '<text x="360" y="68" text-anchor="middle" class="sub">Speaker-equal means; higher C indicates a sharper lag minimum</text>',
        '<text x="390" y="102" text-anchor="middle" class="lab">Evaluation audio take</text>',
        '<text x="30" y="350" text-anchor="middle" class="lab" transform="rotate(-90 30 350)">Driving video take</text>',
    ]
    for j, role in enumerate(ROLES):
        out.append(f'<text x="{left + j * cell + cell/2}" y="{top - 12}" text-anchor="middle" class="lab">{role}</text>')
        out.append(f'<text x="{left - 18}" y="{top + j * cell + cell/2 + 5}" text-anchor="end" class="lab">{role}</text>')
    for i, video_role in enumerate(ROLES):
        for j, audio_role in enumerate(ROLES):
            x, y = left + j * cell, top + i * cell
            value = matrix[video_role][audio_role]
            fill = color(value)
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="3"/>')
            out.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 6}" text-anchor="middle" class="val">{three(value)}</text>')
    out.extend([
        '<text x="390" y="585" text-anchor="middle" class="sub">Rows are video drivers; columns are source PCM used for SyncNet.</text>',
        '</svg>',
    ])
    return "\n".join(out) + "\n"


def paired_loss_svg(speakers: list[str], contrasts: list[dict], summary: dict) -> str:
    width, height = 1120, 700
    panel_x = (70, 585)
    top, row_h, panel_w = 160, 34, 465
    colors = ("#1f77b4", "#d95f02", "#2ca02c", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#17becf")
    by_speaker = {row["speaker"]: row for row in contrasts}
    keys = (("within_N_loss_C", "cross_N_loss_C"),
            ("within_T_loss_C", "cross_T_loss_C"))
    all_values = [float(row[key]) for row in contrasts for pair in keys for key in pair]
    low = min(0.0, min(all_values)) - 0.15
    high = max(0.0, max(all_values)) + 0.15
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:23px;font-weight:700}.sub{font-size:13px;fill:#4b5563}.panel{font-size:18px;font-weight:700}.tick,.lab{font-size:12px}.sum{font-size:12px;font-weight:600}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="38" text-anchor="middle" class="title">Matched-to-mismatched Sync-C losses</text>',
        '<text x="560" y="62" text-anchor="middle" class="sub">Each line pairs within-class alternate-take loss with cross-class loss for one source</text>',
        '<circle cx="408" cy="88" r="5" fill="#34495e"/>',
        '<text x="420" y="92" class="lab">Within-class alternate-take loss</text>',
        '<rect x="650" y="83" width="10" height="10" fill="#34495e"/>',
        '<text x="668" y="92" class="lab">Cross-class mean loss</text>',
    ]
    panels = (("Natural-driven videos", *keys[0]), ("TTS-driven videos", *keys[1]))
    for p, (title, within_key, cross_key) in enumerate(panels):
        px = panel_x[p]
        left, right = px + 82, px + panel_w - 15
        plot_w = right - left

        def xmap(value: float) -> float:
            return left + (value - low) / (high - low) * plot_w

        out.append(f'<text x="{px + panel_w/2}" y="126" text-anchor="middle" class="panel">{title}</text>')
        zero = xmap(0.0)
        out.append(f'<line x1="{zero:.2f}" y1="{top - 10}" x2="{zero:.2f}" y2="{top + row_h*len(speakers) + 14}" stroke="#9ca3af" stroke-dasharray="5,4"/>')
        for tick in np.linspace(low, high, 5):
            x = xmap(float(tick))
            out.append(f'<line x1="{x:.2f}" y1="{top - 4}" x2="{x:.2f}" y2="{top + row_h*len(speakers) + 3}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{x:.2f}" y="{top + row_h*len(speakers) + 23}" text-anchor="middle" class="tick">{three(float(tick))}</text>')
        for i, speaker in enumerate(speakers):
            row = by_speaker[speaker]
            y = top + i * row_h
            x1, x2 = xmap(row[within_key]), xmap(row[cross_key])
            color = colors[i % len(colors)]
            out.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" class="lab">{html.escape(speaker)}</text>')
            out.append(f'<line x1="{x1:.2f}" y1="{y}" x2="{x2:.2f}" y2="{y}" stroke="{color}" stroke-width="2" opacity="0.75"/>')
            out.append(f'<circle cx="{x1:.2f}" cy="{y}" r="4" fill="{color}"/>')
            out.append(f'<rect x="{x2 - 3.5:.2f}" y="{y - 3.5:.2f}" width="7" height="7" fill="{color}"/>')
        for j, (key, label, color) in enumerate(((within_key, "Within-class", "#1f77b4"),
                                                  (cross_key, "Cross-class", "#d95f02"))):
            y = top + row_h * len(speakers) + 70 + 62 * j
            stat = summary[key]
            mean = stat["mean"]
            lo, hi = stat["ci95_descriptive"]
            xlo, xhi, xm = xmap(lo), xmap(hi), xmap(mean)
            out.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" class="sum">{label}</text>')
            out.append(f'<line x1="{xlo:.2f}" y1="{y}" x2="{xhi:.2f}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
            out.append(f'<line x1="{xlo:.2f}" y1="{y - 5}" x2="{xlo:.2f}" y2="{y + 5}" stroke="{color}"/>')
            out.append(f'<line x1="{xhi:.2f}" y1="{y - 5}" x2="{xhi:.2f}" y2="{y + 5}" stroke="{color}"/>')
            out.append(f'<path d="M {xm:.2f} {y - 5:.2f} L {xm + 5:.2f} {y:.2f} L {xm:.2f} {y + 5:.2f} L {xm - 5:.2f} {y:.2f} Z" fill="{color}"/>')
            out.append(f'<text x="{left}" y="{y + 20}" class="sum">Mean {three(mean)} · 95% descriptive CI [{three(lo)}, {three(hi)}]</text>')
    out.extend([
        '<text x="560" y="684" text-anchor="middle" class="sub">Positive loss means lower C with mismatched evaluation audio; within-minus-cross remains descriptive.</text>',
        '</svg>',
    ])
    return "\n".join(out) + "\n"


def summarize(run: Path, config_path: Path) -> tuple[dict, dict]:
    protocol_path = run / "protocol_v3_local_qwen_fp32.json"
    protocol = read_json(protocol_path)
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("protocol v3 SHA-256 mismatch")
    manifest_path = run / protocol["manifest_path"]
    if sha256(manifest_path) != MANIFEST_SHA256 or protocol["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("source manifest v3 SHA-256 mismatch")
    source_manifest = read_json(manifest_path)
    config_path = config_path.resolve(strict=True)
    config_sha = sha256(config_path)
    config = read_json(config_path)
    audio_path, video_path, score_path = (run / name for name in
                                           ("audio_manifest.json", "video_manifest.json", "matrix_scores.json"))
    audio, video, matrix = map(read_json, (audio_path, video_path, score_path))
    if matrix.get("status") != "complete":
        raise ValueError("matrix score manifest is not complete")
    if matrix.get("config_sha256") != config_sha or video.get("config_sha256") != config_sha:
        raise ValueError("score/video manifest does not bind the requested full config")
    if config_sha not in audio.get("input_config_sha256s", []):
        raise ValueError("audio manifest does not bind the requested full config")
    speakers = [str(s) for s in protocol["speakers"]]
    if len(speakers) != 8 or set(speakers) != {str(x["speaker"]) for x in matrix["speaker_results"]}:
        raise ValueError("score speaker roster differs from frozen protocol")
    audio_by = {(str(r["speaker"]), str(r["role"])): r for r in audio["records"]}
    video_by = {(str(r["speaker"]), str(r["video_role"])): r for r in video["records"]}
    score_by = {str(r["speaker"]): r for r in matrix["speaker_results"]}
    if len(audio_by) != 32 or len(video_by) != 32:
        raise ValueError("audio/video manifests must each contain 32 source records")

    cells: list[dict] = []
    duration_rows: list[dict] = []
    distance_receipts: list[dict] = []
    c_by_speaker: dict[str, dict[tuple[str, str], float]] = {}
    d0_by_speaker: dict[str, dict[tuple[str, str], float]] = {}
    for sid in speakers:
        scored = score_by[sid]
        support = scored["support"]
        start = int(support["query_start_window_inclusive"])
        end = int(support["query_end_window_exclusive"])
        count = int(support["query_window_count"])
        if end - start != count or int(support["lag_radius_frames"]) != 15:
            raise ValueError(f"bad common support for {sid}")
        c_by_speaker[sid], d0_by_speaker[sid] = {}, {}
        for role in ROLES:
            ar = audio_by[(sid, role)]
            vr = video_by[(sid, role)]
            probe = scored["features"]["visual"][role]["probe"]
            fractions = support["common_support_fraction_of_each"][role]
            duration_rows.append({
                "speaker": sid, "take": role,
                "audio_duration_s": float(ar["duration_s"]), "audio_samples": int(ar["samples"]),
                "audio_rms": float(ar["rms_after_pcm16"]),
                "video_duration_s": float(probe["duration"]),
                "video_frames": int(probe["nb_frames"]),
                "common_query_start": start, "common_query_end_exclusive": end,
                "common_query_count": count,
                "query_anchor_start_s": float(support["query_anchor_start_s"]),
                "query_anchor_end_s_exclusive": float(support["query_anchor_end_s_exclusive"]),
                "audio_feature_windows": int(support["feature_window_counts"]["audio"][role]),
                "video_feature_windows": int(support["feature_window_counts"]["video"][role]),
                "audio_support_fraction": float(fractions["audio"]),
                "video_support_fraction": float(fractions["video"]),
            })
        for video_role in ROLES:
            for audio_role in ROLES:
                key = f"V_{video_role}_A_{audio_role}"
                saved = scored["cells"][key]
                distance_path = run / "distances" / sid / f"V_{video_role}__A_{audio_role}.npy"
                distance_sha = sha256(distance_path)
                if distance_sha != saved["distance_sha256"]:
                    raise ValueError(f"distance SHA-256 mismatch: {sid}/{key}")
                values = np.load(distance_path, allow_pickle=False)
                if values.shape != tuple(saved["distance_shape"]) or values.shape != (count, 31):
                    raise ValueError(f"distance shape mismatch: {sid}/{key}")
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite distance values: {sid}/{key}")
                curve = np.mean(values, axis=0, dtype=np.float64)
                if not np.allclose(curve, saved["full_mean_distance_curve"], rtol=0, atol=1e-8):
                    raise ValueError(f"saved lag curve differs from distance array: {sid}/{key}")
                b = float(np.median(curve))
                j = int(np.argmin(curve))
                dmin, d0 = float(curve[j]), float(curve[15])
                c = b - dmin
                offset = 15 - j
                if (abs(c - float(saved["C"])) > 1e-7 or
                        abs(dmin - float(saved["Dmin"])) > 1e-7 or
                        abs(d0 - float(saved["D0"])) > 1e-7 or
                        offset != int(saved["offset_audio_minus_video_frames"])):
                    raise ValueError(f"saved endpoints differ from distance array: {sid}/{key}")
                c_by_speaker[sid][(video_role, audio_role)] = c
                d0_by_speaker[sid][(video_role, audio_role)] = d0
                cells.append({
                    "speaker": sid, "video_role": video_role, "audio_role": audio_role,
                    "C": c, "B_median": b, "Dmin": dmin, "D0": d0,
                    "offset_audio_minus_video_frames": offset,
                    "offset_at_lag_boundary": j in (0, 30),
                    "argmin_index": j, "common_query_count": count,
                    "distance_sha256": distance_sha,
                })
                distance_receipts.append({
                    "speaker": sid, "cell": key, "path": str(distance_path.resolve()),
                    "sha256": distance_sha, "shape": list(values.shape),
                })

    means = {}
    for metric, source in (("C", c_by_speaker), ("D0", d0_by_speaker)):
        means[metric] = {
            v: {a: float(np.mean([source[sid][(v, a)] for sid in speakers], dtype=np.float64))
                for a in ROLES}
            for v in ROLES
        }
    contrasts = []
    for sid in speakers:
        c = c_by_speaker[sid]
        matched_n = (c[("N1", "N1")] + c[("N2", "N2")]) / 2
        within_n = (c[("N1", "N2")] + c[("N2", "N1")]) / 2
        cross_n = float(np.mean([c[(v, a)] for v in ("N1", "N2") for a in ("T1", "T2")]))
        matched_t = (c[("T1", "T1")] + c[("T2", "T2")]) / 2
        within_t = (c[("T1", "T2")] + c[("T2", "T1")]) / 2
        cross_t = float(np.mean([c[(v, a)] for v in ("T1", "T2") for a in ("N1", "N2")]))
        contrasts.append({
            "speaker": sid,
            "matched_N_C": matched_n, "within_N_C": within_n, "cross_N_C": cross_n,
            "within_N_loss_C": matched_n - within_n, "cross_N_loss_C": matched_n - cross_n,
            "within_minus_cross_N_C_descriptive": within_n - cross_n,
            "matched_T_C": matched_t, "within_T_C": within_t, "cross_T_C": cross_t,
            "within_T_loss_C": matched_t - within_t, "cross_T_loss_C": matched_t - cross_t,
            "within_minus_cross_T_C_descriptive": within_t - cross_t,
            "native_self_gap_T_minus_N_C": matched_t - matched_n,
        })
    metric_names = [key for key in contrasts[0] if key != "speaker"]
    stats = protocol["statistics"]
    draws, seed = int(stats["bootstrap_draws"]), int(stats["seed"])
    if draws != 2000 or seed != 20261015:
        raise ValueError("bootstrap settings differ from the frozen protocol")
    rng = np.random.Generator(np.random.PCG64(seed))
    sample_ix = rng.integers(0, len(speakers), size=(draws, len(speakers)))
    contrast_summary = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in contrasts], dtype=np.float64)
        boot = values[sample_ix].mean(axis=1)
        contrast_summary[name] = {
            "mean": float(values.mean()),
            "ci95_descriptive": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "positive_sources": int(np.sum(values > 0)),
            "source_count": len(speakers),
        }

    offset_summary = []
    for vr in ROLES:
        for ar in ROLES:
            group = [r for r in cells if r["video_role"] == vr and r["audio_role"] == ar]
            neg = sum(r["offset_audio_minus_video_frames"] == 15 for r in group)
            pos = sum(r["offset_audio_minus_video_frames"] == -15 for r in group)
            boundary = sum(r["offset_at_lag_boundary"] for r in group)
            offset_summary.append({
                "video_role": vr, "audio_role": ar, "speaker_count": len(group),
                "boundary_count": boundary, "negative_boundary_offset_count": neg,
                "positive_boundary_offset_count": pos,
                "boundary_fraction": boundary / len(group),
                "mean_offset_frames": float(np.mean([r["offset_audio_minus_video_frames"] for r in group])),
            })
    data = {
        "speakers": speakers, "means": means, "cells": cells, "duration_rows": duration_rows,
        "contrasts": contrasts, "contrast_summary": contrast_summary,
        "offset_summary": offset_summary, "distance_receipts": distance_receipts,
        "matrix_sha256": sha256(score_path), "audio_sha256": sha256(audio_path),
        "video_sha256": sha256(video_path), "config_sha256": config_sha,
        "protocol_sha256": PROTOCOL_SHA256, "source_manifest_sha256": MANIFEST_SHA256,
        "distance_array_count": len(distance_receipts),
    }
    sources = {
        "protocol": {"path": str(protocol_path.resolve()), "sha256": PROTOCOL_SHA256},
        "source_manifest": {"path": str(manifest_path.resolve()), "sha256": MANIFEST_SHA256},
        "full_matrix_config": {"path": str(config_path), "sha256": config_sha},
        "audio_manifest": {"path": str(audio_path.resolve()), "sha256": data["audio_sha256"]},
        "video_manifest": {"path": str(video_path.resolve()), "sha256": data["video_sha256"]},
        "matrix_scores": {"path": str(score_path.resolve()), "sha256": data["matrix_sha256"]},
        "provider_tts": config.get("provider_tts", {}),
        "source_media": [
            {
                "speaker": sid, "role": role,
                "source_audio_path": audio_by[(sid, role)]["source_path"],
                "source_audio_sha256": audio_by[(sid, role)]["source_sha256"],
                "canonical_audio_path": audio_by[(sid, role)]["canonical_path"],
                "canonical_audio_sha256": audio_by[(sid, role)]["canonical_sha256"],
                "video_path": video_by[(sid, role)]["video_path"],
                "video_sha256": video_by[(sid, role)]["video_sha256"],
            }
            for sid in speakers for role in ROLES
        ],
    }
    remote_root = "/home/wjj/tts-audio/tts-exp"
    sources["fixed_face"] = protocol["source_face"]
    sources["wav2lip_inference"] = {
        "path": f"{remote_root}/third_party/Wav2Lip/inference.py",
        "sha256": video["wav2lip_inference_sha256"],
    }
    sources["wav2lip_checkpoint"] = {
        "path": f"{remote_root}/third_party/Wav2Lip/checkpoints/wav2lip_gan.pth",
        "sha256": video["wav2lip_checkpoint_sha256"],
    }
    sources["syncnet_model"] = {
        "path": f"{remote_root}/third_party/syncnet_python/data/syncnet_v2.model",
        "sha256": matrix["syncnet_model_sha256"],
    }
    sources["matrix_runner"] = {
        "path": f"{remote_root}/scripts/experiments/repeated_take_crossover_20261015/run_matrix.py",
        "sha256": "9a35c45306f48271493d357796e2e23e63e7cbc6dbb543847b27e203352fa219",
    }
    data["sources"] = sources
    data["protocol"] = protocol
    data["matrix"] = matrix
    return data, protocol


def write_delivery(run: Path, out: Path, data: dict, protocol: dict) -> dict:
    if out.exists() or (out.resolve() == run.resolve()) or run.resolve() not in out.resolve().parents:
        raise ValueError(f"output must be a new subdirectory of the frozen run: {out}")
    out.mkdir(parents=True, exist_ok=False)
    speakers = data["speakers"]
    roles = list(ROLES)
    for metric in ("C", "D0"):
        rows = [{"video_role": v, **{f"A_{a}": data["means"][metric][v][a] for a in roles}}
                for v in roles]
        write_csv(out / f"mean_{metric}.csv", ["video_role", *[f"A_{a}" for a in roles]], rows)
    write_csv(out / "source_contrasts.csv", list(data["contrasts"][0]), data["contrasts"])
    contrast_rows = []
    for metric, item in data["contrast_summary"].items():
        lo, hi = item["ci95_descriptive"]
        contrast_rows.append({
            "metric": metric, "mean": item["mean"], "ci95_low": lo, "ci95_high": hi,
            "positive_sources": item["positive_sources"], "source_count": item["source_count"],
            "interpretation": "descriptive",
        })
    write_csv(out / "contrast_summary.csv", list(contrast_rows[0]), contrast_rows)
    write_csv(out / "duration_support.csv", list(data["duration_rows"][0]), data["duration_rows"])
    write_csv(out / "offset_by_cell.csv", list(data["cells"][0]), data["cells"])
    write_csv(out / "offset_boundary_summary.csv", list(data["offset_summary"][0]), data["offset_summary"])
    result = {
        "schema": "tts-exp/repeated-take-crossover-analysis/v1",
        "protocol_sha256": data["protocol_sha256"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "full_matrix_config_sha256": data["config_sha256"],
        "matrix_scores_sha256": data["matrix_sha256"],
        "speakers": speakers, "roles": roles, "means": data["means"],
        "source_contrasts": data["contrasts"], "bootstrap_summary": data["contrast_summary"],
        "statistics": protocol["statistics"], "common_lag_support_frames": list(LAGS),
        "distance_array_count": data["distance_array_count"],
    }
    write_json(out / "analysis_results.json", result)
    svg_write(out / "sync_c_heatmap.svg", heatmap_svg(data["means"]["C"]))
    svg_write(out / "paired_losses.svg", paired_loss_svg(speakers, data["contrasts"], data["contrast_summary"]))
    artifact_rows = []
    for path in sorted(out.iterdir(), key=lambda p: p.name):
        if path.is_file():
            artifact_rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "tts-exp/repeated-take-crossover-delivery/v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": {
            "primary_clock": data["matrix"].get("primary_clock"),
            "metric_definitions": data["matrix"].get("metric_definitions"),
            "contrasts": protocol.get("contrasts"),
            "interpretation": "within-minus-cross is descriptive; matched T-minus-N combines driver and evaluator changes",
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "analyzer_path": str(Path(__file__).resolve()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
        "sources": data["sources"],
        "verification": {
            "speaker_count": len(speakers),
            "distance_arrays_hashed_and_recomputed": data["distance_array_count"],
            "common_lag_support_frames": list(LAGS),
            "distance_receipts": data["distance_receipts"],
        },
        "artifacts": artifact_rows,
    }
    write_json(out / "delivery_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="frozen full 8-speaker matrix config")
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, required=True, help="new delivery directory under the run")
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    if run != RUN.resolve():
        raise ValueError(f"run directory differs from frozen attempt: {RUN}")
    data, protocol = summarize(run, args.config)
    manifest = write_delivery(run, args.output_dir.resolve(), data, protocol)
    print(json.dumps({
        "status": manifest["status"], "output_dir": str(args.output_dir.resolve()),
        "distance_arrays_recomputed": manifest["verification"]["distance_arrays_hashed_and_recomputed"],
        "artifact_count": len(manifest["artifacts"]),
        "delivery_manifest": str(args.output_dir.resolve() / "delivery_manifest.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
