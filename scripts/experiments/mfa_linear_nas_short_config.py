"""Replay the NAS 0.5s/0.1s/±1000ms purification preset on frozen N/M/T videos."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.experiments.mfa_linear_natural_track_probe import (
    FFMPEG, FFPROBE, IDS, REPO, SYNCNET, SYNCNET_MODEL, SYNCNET_PYTHON, save, sha,
)
from scripts.experiments.mfa_linear_video_retiming.official import score_official_cell


def main() -> None:
    parent = REPO / "runs/mfa_linear_natural_track_probe_20260924"
    root = parent / "nas_w05_s01_a1000"
    root.mkdir(parents=True, exist_ok=True)
    source_preset = (REPO.parent / "nas-cch-code/voice_def/cch_419/run_1s_no_blend_purify_w05_a100.sh")
    protocol = {"preset_source": str(source_preset), "preset_source_sha256": sha(source_preset),
                "window_sec": 0.5, "stride_sec": 0.1, "search_ms": [-1000, 1000],
                "batch_size": 20, "offset_min_track": 40, "eval_min_track": 100,
                "input_ids": list(IDS), "video_arms": ["M", "T"],
                "evaluation": "official local SyncNet V2, original natural PCM16 audio"}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("NAS preset protocol changed")
    else:
        save(protocol_path, protocol)
    source = json.loads((parent / "summary.json").read_text())
    by_id = {record["sample_id"]: record for record in source["samples"]}
    config = {"repo_root": str(REPO), "paths": {
        "ffmpeg": str(FFMPEG), "ffprobe": str(FFPROBE), "syncnet_python": str(SYNCNET_PYTHON),
        "syncnet_root": str(SYNCNET), "syncnet_model": str(SYNCNET_MODEL)},
        "official": {"min_track": 100, "batch_size": 20,
                     "pipeline_timeout_seconds": 900, "syncnet_timeout_seconds": 900}}
    rows = []
    for sample_id in IDS:
        record = by_id[sample_id]
        natural_audio = Path(record["audio"]["N"]["path"])
        natural_c = float(record["scores"]["N_raw"]["sync_c"])
        for arm in ("M", "T"):
            input_video = parent / sample_id / "input" / f"{arm}.mp4"
            output_video = root / sample_id / f"{arm}_nas_w05.mp4"
            save_dir = root / sample_id / f"{arm}_purify"
            if not (save_dir / "results.json").exists():
                log = root / sample_id / f"{arm}_purify.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                argv = [sys.executable, str(REPO / "scripts/purify_syncdrift.py"),
                        "--input_video", str(input_video), "--output_video", str(output_video),
                        "--save_dir", str(save_dir), "--mode", "window",
                        "--window_sec", "0.5", "--stride_sec", "0.1",
                        "--search_min_ms", "-1000", "--search_max_ms", "1000",
                        "--batch_size", "20", "--min_track", "40", "--device", "cuda",
                        "--syncnet_root", str(SYNCNET), "--syncnet_model", str(SYNCNET_MODEL)]
                with log.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"NAS short preset failed: {sample_id} {arm}; see {log}")
            cell = {"sample_id": sample_id, "paired_key": record["paired_key"],
                    "speaker_id": record["speaker_id"], "portrait_id": "3", "video_arm": f"{arm}_nas_w05",
                    "audio_role": "N", "video": str(output_video), "audio": str(natural_audio)}
            receipt = score_official_cell(config, cell, run_dir=root / sample_id, resume=True)
            purify = json.loads((save_dir / "results.json").read_text())
            score_path = root / sample_id / "06_official/cells" / f"p3_s{sample_id}_v{arm}_nas_w05_aN/result.json"
            row = {"sample_id": sample_id, "arm": arm, "natural_c": natural_c,
                   "raw_c": float(record["scores"][f"{arm}_raw"]["sync_c"]),
                   "nas_default_w5_c": float(record["scores"][f"{arm}_aligned"]["sync_c"]),
                   "local_w2_c": float(next(r["window2_c"] for r in
                      json.loads((parent / "short_window_diagnostic/summary.json").read_text())["rows"]
                      if r["sample_id"] == sample_id and r["arm"] == arm)),
                   "nas_short_w05_c": float(receipt["official_sync_c"]),
                   "nas_short_w05_d": float(receipt["official_sync_d"]),
                   "nas_short_offset": int(receipt["official_offset"]),
                   "audio_pcm_exact": bool(receipt["mux"]["audio_pcm_exact"]),
                   "video": str(output_video), "video_sha256": sha(output_video),
                   "mean_abs_correction_ms": purify["mean_abs_correction_ms"],
                   "max_abs_correction_ms": purify["max_abs_correction_ms"],
                   "valid_window_ratio": purify["valid_window_ratio"],
                   "repeated_frame_ratio": purify["repeated_frame_ratio"],
                   "skipped_frame_ratio": purify["skipped_frame_ratio"],
                   "score_receipt": str(score_path)}
            rows.append(row)
            save(root / "summary.json", {"protocol": protocol, "rows": rows,
                                         "status": "COMPLETE" if len(rows) == len(IDS) * 2 else "IN_PROGRESS"})
            print(sample_id, arm, f"C={row['nas_short_w05_c']:.3f}",
                  f"N={natural_c:.3f}", f"repeat={row['repeated_frame_ratio']:.3f}",
                  f"skip={row['skipped_frame_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
