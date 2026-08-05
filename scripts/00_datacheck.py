#!/usr/bin/env python3
"""00_datacheck.py - Read-only data quality check for wav & png assets (issue #2).

Checks every wav: sample rate, channels, sample width, duration.
Checks every png: width, height.
Warns (<4s wav) but never blocks the pipeline (per plan Q10/Q21).
Output: runs/<run_id>/00_datacheck/00_datacheck.json
"""

import argparse
import json
import struct
import wave
from pathlib import Path


def read_png_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) from a PNG's IHDR chunk, or None on failure.

    PNG spec: first 8 bytes = signature, then chunks.
    IHDR is the first chunk and is always exactly 13 bytes.
    """
    try:
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    except Exception:
        return None


def check_audio(audio_dir: Path, sample_ids: list[int]) -> list[dict]:
    results = []
    for i in sample_ids:
        p = audio_dir / f"{i}.wav"
        entry: dict = {"sample_id": i, "file": str(p), "exists": p.exists()}
        if not p.exists():
            entry["error"] = "missing"
            results.append(entry)
            continue
        try:
            with wave.open(str(p), "rb") as wf:
                entry["channels"] = wf.getnchannels()
                entry["sample_rate"] = wf.getframerate()
                entry["sample_width_bits"] = wf.getsampwidth() * 8
                entry["frames"] = wf.getnframes()
                dur = wf.getnframes() / wf.getframerate()
                entry["duration_s"] = round(dur, 2)
                if dur < 4.0:
                    entry["warning"] = f"duration {dur:.2f}s < 4s; SyncNet may be unstable"
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)
    return results


def check_images(
    image_dir: Path,
    sample_ids: list[int],
    image_path: Path | None = None,
) -> list[dict]:
    results = []
    for i in sample_ids:
        p = image_path if image_path is not None else image_dir / f"{i}.png"
        entry: dict = {"sample_id": i, "file": str(p), "exists": p.exists()}
        if not p.exists():
            entry["error"] = "missing"
            results.append(entry)
            continue
        size = read_png_size(p)
        if size is None:
            entry["error"] = "invalid_png"
        else:
            entry["width"], entry["height"] = size
        results.append(entry)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Data quality check for tts-exp")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "runs" / args.run_id / "00_datacheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    from utils import detect_sample_ids, load_config, resolve_repo_path
    cfg = load_config(repo, args.config or None)
    sample_ids = detect_sample_ids(repo, args.smoke, cfg=cfg)
    audio_dir = resolve_repo_path(
        repo, cfg.get("paths", {}).get("audio_dir", "data/data/audio")
    )
    image_dir = resolve_repo_path(
        repo, cfg.get("paths", {}).get("image_dir", "data/data/image")
    )
    fixed_image = cfg.get("paths", {}).get("fixed_image")
    image_path = resolve_repo_path(repo, fixed_image) if fixed_image else None

    audio_results = check_audio(audio_dir, sample_ids)
    image_results = check_images(image_dir, sample_ids, image_path=image_path)

    warnings = [e for e in audio_results if "warning" in e]
    errors = [e for e in audio_results + image_results if "error" in e]

    report = {
        "run_id": args.run_id,
        "smoke": args.smoke,
        "audio": audio_results,
        "images": image_results,
        "warnings": [w["warning"] for w in warnings],
        "errors": errors,
        "summary": {
            "total": len(sample_ids),
            "audio_ok": sum(1 for e in audio_results if "error" not in e),
            "images_ok": sum(1 for e in image_results if "error" not in e),
            "warnings": len(warnings),
            "errors": len(errors),
        },
    }

    (out_dir / "00_datacheck.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[datacheck] {report['summary']}")
    for w in warnings:
        print(f"[datacheck] WARNING: sample {w['sample_id']} - {w['warning']}")
    for error in errors:
        print(f"[datacheck] ERROR: sample {error['sample_id']} - {error['error']}")

    # Warnings remain non-blocking; missing or invalid assets are blocking.
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
