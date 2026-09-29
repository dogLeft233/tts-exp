#!/usr/bin/env python3
"""Score an existing phoneme-TFG run with the repository's official SyncNet path.

This adapter deliberately does not regenerate media.  It runs the same two
official commands used by ``scripts/04_eval.py``:

    run_pipeline.py -> run_syncnet.py

Each media cell gets a unique reference key and its own provenance record.  The
worker is sequential because face detection and SyncNet share the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


CONFIDENCE_RE = re.compile(r"Confidence:\s+([0-9.]+)")
MIN_DIST_RE = re.compile(r"Min dist:\s+([0-9.]+)")
OFFSET_RE = re.compile(r"AV offset:\s+(-?[0-9]+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"\n[returncode] {completed.returncode}\n")
        handle.write(f"[elapsed_seconds] {time.monotonic() - started:.3f}\n")
    return int(completed.returncode)


def parse_score(log_path: Path) -> dict[str, float | int] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = CONFIDENCE_RE.search(text)
    min_dist = MIN_DIST_RE.search(text)
    offset = OFFSET_RE.search(text)
    if confidence is None or min_dist is None:
        return None
    return {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(min_dist.group(1)),
        "av_offset": int(offset.group(1)) if offset else None,
    }


def discover_receipts(run_root: Path) -> list[Path]:
    receipts = sorted((run_root / "03_video" / "wav2lip").glob("*/*.receipt.json"))
    if not receipts:
        raise FileNotFoundError(f"no Wav2Lip receipts under {run_root / '03_video'}")
    return receipts


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    syncnet_root = args.syncnet_root.resolve()
    # Preserve the venv entrypoint path.  ``Path.resolve()`` follows the
    # symlink to system Python and drops the venv's site-packages (notably cv2).
    syncnet_python = args.syncnet_python
    model = args.model.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    receipts = discover_receipts(run_root)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()

    write_json(
        output_root / "protocol.json",
        {
            "schema_version": 1,
            "evaluation": "official_syncnet_v2",
            "run_root": str(run_root),
            "syncnet_root": str(syncnet_root),
            "syncnet_python": str(syncnet_python),
            "syncnet_model": str(model),
            "syncnet_model_sha256": file_sha256(model),
            "min_track": int(args.min_track),
            "worker_policy": "sequential_single_gpu",
            "receipt_count": len(receipts),
        },
    )

    for index, receipt_path in enumerate(receipts, 1):
        receipt = read_json(receipt_path)
        video = Path(str(receipt["output"])).resolve()
        sample_id = str(receipt["sample_id"])
        arm = str(receipt["arm"])
        cell_key = str(receipt["cell_key"])
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", cell_key)
        reference = f"official_{safe_key}"
        data_dir = output_root / "data"
        pipeline_log = output_root / "logs" / f"{safe_key}.pipeline.log"
        sync_log = output_root / "logs" / f"{safe_key}.syncnet.log"
        score_path = output_root / "scores" / f"{safe_key}.json"

        base: dict[str, Any] = {
            "cell_key": cell_key,
            "sample_id": sample_id,
            "source_group": receipt.get("source_group"),
            "arm": arm,
            "tfg": receipt.get("tfg"),
            "video": str(video),
            "video_sha256": file_sha256(video) if video.is_file() else None,
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "reference": reference,
            "min_track": int(args.min_track),
            "scorer": "official_syncnet_v2",
        }

        if not video.is_file():
            failure = {**base, "stage": "input", "error": "video_missing"}
            failures.append(failure)
            print(f"FAIL {index}/{len(receipts)} {cell_key}: video missing", flush=True)
            continue

        pipeline_rc = run_logged(
            [
                str(syncnet_python),
                str(syncnet_root / "run_pipeline.py"),
                "--videofile",
                str(video),
                "--reference",
                reference,
                "--data_dir",
                str(data_dir),
                "--min_track",
                str(args.min_track),
                "--overwrite",
            ],
            cwd=syncnet_root,
            log_path=pipeline_log,
        )
        if pipeline_rc != 0:
            failure = {**base, "stage": "run_pipeline", "returncode": pipeline_rc}
            failures.append(failure)
            print(f"FAIL {index}/{len(receipts)} {cell_key}: pipeline rc={pipeline_rc}", flush=True)
            continue

        sync_rc = run_logged(
            [
                str(syncnet_python),
                str(syncnet_root / "run_syncnet.py"),
                "--videofile",
                str(video),
                "--reference",
                reference,
                "--data_dir",
                str(data_dir),
                "--initial_model",
                str(model),
            ],
            cwd=syncnet_root,
            log_path=sync_log,
        )
        parsed = parse_score(sync_log) if sync_rc == 0 else None
        if sync_rc != 0 or parsed is None:
            failure = {
                **base,
                "stage": "run_syncnet",
                "returncode": sync_rc,
                "parsed": parsed is not None,
            }
            failures.append(failure)
            print(f"FAIL {index}/{len(receipts)} {cell_key}: sync rc={sync_rc}", flush=True)
            continue

        record = {
            **base,
            **parsed,
            "pipeline_log": str(pipeline_log.resolve()),
            "syncnet_log": str(sync_log.resolve()),
        }
        write_json(score_path, record)
        records.append(record)
        print(
            f"OK {index}/{len(receipts)} {cell_key} "
            f"C={record['sync_c']:.3f} D={record['sync_d']:.3f} "
            f"offset={record['av_offset']}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "evaluation": "official_syncnet_v2",
        "status": "complete" if not failures and len(records) == len(receipts) else "incomplete",
        "expected": len(receipts),
        "completed": len(records),
        "failed": len(failures),
        "min_track": int(args.min_track),
        "records": records,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: <run-root>/07_official_syncnet/full",
    )
    parser.add_argument(
        "--syncnet-root",
        type=Path,
        default=repo / "third_party/syncnet_python",
    )
    parser.add_argument(
        "--syncnet-python",
        type=Path,
        default=Path.home() / ".venvs/syncnet/bin/python",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--min-track", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_root = args.run_root.resolve()
    if args.output_root is None:
        args.output_root = args.run_root / "07_official_syncnet" / "full"
    if args.model is None:
        args.model = args.syncnet_root / "data/syncnet_v2.model"
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "expected": result["expected"],
                "completed": result["completed"],
                "failed": result["failed"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
