"""Apply one frozen v1 shared frequency profile to canonical 16 kHz audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .audio_io import read_canonical_wav, write_float32_wav
from .shared_frequency_profile_v1 import PROFILES_BY_ID, profile_dict, transform_shared_profile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    input_wav: Path,
    output_wav: Path,
    *,
    profile_id: str,
    receipt_path: Path | None = None,
    target_rms: float = 0.0376838172675746,
    peak_ceiling: float = 0.98,
) -> dict[str, Any]:
    source_path = input_wav.resolve(strict=True)
    output_path = output_wav.resolve()
    receipt_path = (receipt_path or output_path.with_suffix(output_path.suffix + ".json")).resolve()
    if output_path == source_path or receipt_path in {source_path, output_path}:
        raise ValueError("source, enhanced waveform, and receipt require distinct paths")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite a profile output or receipt")
    if profile_id not in PROFILES_BY_ID:
        raise ValueError(f"unknown profile_id: {profile_id}")

    samples, read_info = read_canonical_wav(source_path)
    enhanced, diagnostics = transform_shared_profile(
        samples,
        profile_id,
        sample_rate_hz=read_info.sample_rate,
        target_rms=target_rms,
        peak_ceiling=peak_ceiling,
    )
    write_info = write_float32_wav(
        output_path,
        enhanced,
        sample_rate=read_info.sample_rate,
        peak_ceiling=None,
        overwrite=False,
    )
    if read_info.frames != write_info.frames or read_info.sample_rate != write_info.sample_rate:
        raise RuntimeError("profile output failed the fixed-rate/fixed-sample-count contract")

    result = {
        "schema": "cross_tfg_shared_frequency_profile_cli_result_v1",
        "status": "PASS",
        "source_pcm": {
            "path": str(source_path),
            "file_sha256": read_info.file_sha256,
            "pcm_float32_sha256": read_info.pcm_float32_sha256,
            "source_sample_rate_hz": read_info.source_sample_rate,
            "sample_rate_hz": read_info.sample_rate,
            "channels": read_info.source_channels,
            "sample_count": read_info.frames,
            "source_encoding": read_info.source_encoding,
        },
        "enhanced_pcm": {
            "path": str(output_path),
            "file_sha256": write_info.file_sha256,
            "pcm_float32_sha256": write_info.pcm_float32_sha256,
            "sample_rate_hz": write_info.sample_rate,
            "channels": write_info.channels,
            "sample_count": write_info.frames,
            "subtype": write_info.subtype,
        },
        "same_sample_count": read_info.frames == write_info.frames,
        "profile": profile_dict(profile_id),
        "diagnostics": diagnostics,
        "cli_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    result["receipt_path"] = str(receipt_path)
    result["receipt_sha256"] = _sha256_file(receipt_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument("--profile-id", required=True, choices=tuple(PROFILES_BY_ID))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--target-rms", type=float, default=0.0376838172675746)
    parser.add_argument("--peak-ceiling", type=float, default=0.98)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.input_wav,
            args.output_wav,
            profile_id=args.profile_id,
            receipt_path=args.receipt,
            target_rms=args.target_rms,
            peak_ceiling=args.peak_ceiling,
        )
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
