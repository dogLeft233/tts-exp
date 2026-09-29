"""One frozen short-window sensitivity check of the natural-track probe."""
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
    root = parent / "short_window_diagnostic"
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"window_sec": 2.0, "stride_sec": 0.5, "search_ms": [-240, 240],
                "input_ids": list(IDS), "video_arms": ["M", "T"],
                "evaluation": "official local SyncNet V2, min_track=100, original N PCM16"}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("diagnostic protocol changed")
    else:
        save(protocol_path, protocol)
    parent_summary = json.loads((parent / "summary.json").read_text())
    by_id = {row["sample_id"]: row for row in parent_summary["samples"]}
    config = {"repo_root": str(REPO), "paths": {
        "ffmpeg": str(FFMPEG), "ffprobe": str(FFPROBE), "syncnet_python": str(SYNCNET_PYTHON),
        "syncnet_root": str(SYNCNET), "syncnet_model": str(SYNCNET_MODEL)},
        "official": {"min_track": 100, "batch_size": 20,
                     "pipeline_timeout_seconds": 900, "syncnet_timeout_seconds": 900}}
    rows = []
    for sample_id in IDS:
        source = by_id[sample_id]
        audio = Path(source["audio"]["N"]["path"])
        baseline = float(source["scores"]["N_raw"]["sync_c"])
        for arm in ("M", "T"):
            video = parent / sample_id / "input" / f"{arm}.mp4"
            output = root / sample_id / f"{arm}_w2.mp4"
            save_dir = root / sample_id / f"{arm}_purify"
            if not (save_dir / "results.json").exists():
                log = root / sample_id / f"{arm}_purify.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                argv = [sys.executable, str(REPO / "scripts/purify_syncdrift.py"),
                        "--input_video", str(video), "--output_video", str(output),
                        "--save_dir", str(save_dir), "--mode", "window",
                        "--window_sec", "2.0", "--stride_sec", "0.5",
                        "--search_min_ms", "-240", "--search_max_ms", "240",
                        "--device", "cuda", "--syncnet_root", str(SYNCNET),
                        "--syncnet_model", str(SYNCNET_MODEL)]
                with log.open("w", encoding="utf-8") as handle:
                    proc = subprocess.run(argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise RuntimeError(f"purify {sample_id} {arm} failed; see {log}")
            cell = {"sample_id": sample_id, "paired_key": source["paired_key"],
                    "speaker_id": source["speaker_id"], "portrait_id": "3", "video_arm": f"{arm}_w2",
                    "audio_role": "N", "video": str(output), "audio": str(audio)}
            receipt = score_official_cell(config, cell, run_dir=root / sample_id, resume=True)
            purify = json.loads((save_dir / "results.json").read_text())
            row = {"sample_id": sample_id, "arm": arm, "baseline_c": baseline,
                   "raw_c": float(source["scores"][f"{arm}_raw"]["sync_c"]),
                   "window5_c": float(source["scores"][f"{arm}_aligned"]["sync_c"]),
                   "window2_c": float(receipt["official_sync_c"]),
                   "window2_d": float(receipt["official_sync_d"]),
                   "window2_offset": int(receipt["official_offset"]),
                   "window2_pcm_exact": receipt["mux"]["audio_pcm_exact"],
                   "window2_video": str(output), "window2_video_sha256": sha(output),
                   "mean_abs_correction_ms": purify["mean_abs_correction_ms"],
                   "max_abs_correction_ms": purify["max_abs_correction_ms"],
                   "valid_window_ratio": purify["valid_window_ratio"],
                   "result_receipt": str(root / sample_id / "06_official/cells" / f"p3_s{sample_id}_v{arm}_w2_aN/result.json")}
            print(sample_id, arm, f"N={baseline:.3f}", f"raw={row['raw_c']:.3f}",
                  f"w5={row['window5_c']:.3f}", f"w2={row['window2_c']:.3f}", flush=True)
            rows.append(row)
            save(root / "summary.json", {"protocol": protocol, "rows": rows,
                                         "status": "COMPLETE" if len(rows) == len(IDS) * 2 else "IN_PROGRESS"})


if __name__ == "__main__":
    main()
