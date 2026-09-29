from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    args = parser.parse_args()
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

    plan = rt.read_json(args.plan)
    scorer = SyncNetScorer(args.model, device="cpu", batch_size=20, threads=2)
    rows = []
    for item in plan["rows"]:
        media = Path(str(item["output"]))
        source_audio = Path(str(item["source_audio"]))
        pcm = rt.source_pcm16(source_audio)
        score_dir = args.output_dir / f"{item['sample_id']}__{item['arm']}__N"
        score = scorer.score(media, source_audio, score_dir, str(item["output_sha256"]), rt.bytes_sha256(pcm))
        matrix = np.asarray(np.load(Path(str(score["matrix"])), allow_pickle=False), dtype=np.float64)
        score["full"] = rt.score_metrics(matrix)
        score.update({"protocol_id": args.protocol_id, "sample_id": item["sample_id"], "video_arm": item["arm"], "audio_arm": "N"})
        rows.append({"sample_id": item["sample_id"], "video_arm": item["arm"], "audio_arm": "N", "score": score})
    args.result.parent.mkdir(parents=True, exist_ok=True)
    rt.write_json(args.result, {"schema_version": 1, "protocol_id": args.protocol_id, "status": "complete", "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
