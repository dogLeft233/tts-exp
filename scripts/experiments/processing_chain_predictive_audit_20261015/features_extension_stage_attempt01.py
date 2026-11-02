"""Create a normal-schema feature extract for a frozen secondary asset manifest.

This entrypoint delegates decoding, PCM hashing, DSP and row serialization to the
qualified ``features.py`` implementation. It does not assign source layers or
join scores; those bindings must already be present in the supplied manifest.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FEATURE_SOURCE = Path(__file__).with_name("features.py")
FEATURE_SOURCE_SHA256 = "c7100359a218cd566c678a22c92ac8df0051ffeddc6075a776ae966be0902f0e"


class ExtensionStageError(RuntimeError):
    pass


def _sha(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _load_features():
    source_sha, _ = _sha(FEATURE_SOURCE.resolve(strict=True))
    if source_sha != FEATURE_SOURCE_SHA256:
        raise ExtensionStageError("qualified scientific features.py pin mismatch")
    spec = importlib.util.spec_from_file_location("processing_chain_features_extension", FEATURE_SOURCE)
    if spec is None or spec.loader is None:
        raise ExtensionStageError("cannot load the qualified feature implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract the frozen seven-family schema for one bound extension cohort.")
    parser.add_argument("--assets-json", required=True, type=Path)
    parser.add_argument("--assets-sha256", required=True)
    parser.add_argument("--extension-id", required=True)
    parser.add_argument("--source-layer", required=True)
    parser.add_argument("--expected-assets", required=True, type=int)
    parser.add_argument("--expected-pairs", required=True, type=int)
    parser.add_argument("--expected-source-groups", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    assets_path = args.assets_json.resolve(strict=True)
    assets_sha, assets_bytes = _sha(assets_path)
    if assets_sha != args.assets_sha256.lower():
        raise ExtensionStageError("assets manifest body differs from the supplied exact pin")
    payload = json.loads(assets_path.read_text(encoding="utf-8"))
    if payload.get("extension_id") != args.extension_id:
        raise ExtensionStageError("extension_id differs from the frozen invocation")
    assets = payload.get("assets")
    if not isinstance(assets, list) or len(assets) != args.expected_assets:
        raise ExtensionStageError("asset count differs from the frozen invocation")
    if payload.get("counts", {}).get("asset_count") != args.expected_assets:
        raise ExtensionStageError("manifest asset_count differs from the frozen invocation")
    if payload.get("counts", {}).get("pair_count") != args.expected_pairs:
        raise ExtensionStageError("manifest pair_count differs from the frozen invocation")
    if payload.get("counts", {}).get("source_group_count") != args.expected_source_groups:
        raise ExtensionStageError("manifest source_group_count differs from the frozen invocation")
    if len({str(a.get("audio_uid")) for a in assets}) != len(assets):
        raise ExtensionStageError("audio_uid values are not unique")
    if {str(a.get("source_layer")) for a in assets} != {args.source_layer}:
        raise ExtensionStageError("asset source_layer differs from the bound extension layer")
    pair_conditions: dict[str, list[str]] = collections.defaultdict(list)
    for asset in assets:
        pair_conditions[str(asset.get("pair_uid"))].append(str(asset.get("condition")))
        source = Path(str(asset.get("source_path", ""))).expanduser()
        if not source.is_absolute():
            raise ExtensionStageError(f"source_path is not absolute: {asset.get('audio_uid')}")
    if len(pair_conditions) != args.expected_pairs or any(sorted(v) != ["natural", "tts"] for v in pair_conditions.values()):
        raise ExtensionStageError("the extension manifest does not contain exactly one natural/TTS row per pair")
    if len({str(a.get("source_group")) for a in assets}) != args.expected_source_groups:
        raise ExtensionStageError("distinct source-group count differs from the frozen invocation")

    output_arg = args.output_dir.expanduser()
    if output_arg.exists() or output_arg.is_symlink():
        raise ExtensionStageError("extension output directory must be absent")
    output = output_arg.resolve(strict=False)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ExtensionStageError("extension output parent must already exist and must not be a symlink")

    module, feature_sha = _load_features()
    ffmpeg = module._tool_version(args.ffmpeg)
    ffprobe = module._tool_version(args.ffprobe)
    binaries = {"ffmpeg": ffmpeg, "ffprobe": ffprobe}
    output.mkdir()
    cache_dir = output / "cache"
    cache_dir.mkdir()
    rows: list[dict[str, Any]] = []
    row_receipts: list[dict[str, Any]] = []
    for asset in sorted(assets, key=lambda a: (str(a["pair_uid"]), str(a["condition"]))):
        source = Path(str(asset["source_path"])).expanduser()
        row, receipt = module._asset_row(asset, source, ffprobe["path"], ffmpeg["path"], binaries, feature_sha, cache_dir)
        rows.append(row)
        row_receipts.append(receipt)
    features_path = output / "features.csv"
    module._write_csv(features_path, rows)
    csv_sha, csv_bytes = _sha(features_path)
    receipt = {
        "schema": "processing_chain_feature_extension_receipt_v1",
        "status": "COMPLETE",
        "extension_id": args.extension_id,
        "source_layer": args.source_layer,
        "scope": "one already-bound extension; canonical seven-family DSP and PCM decoder reused unchanged",
        "started_utc": started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "process": {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "cwd": os.getcwd(), "returncode": 0},
        "asset_manifest": {"path": str(assets_path), "sha256": assets_sha, "bytes": assets_bytes},
        "source": {"path": str(FEATURE_SOURCE.resolve()), "sha256": feature_sha},
        "runtime": {"python": sys.version, "platform": platform.platform(), "ffmpeg": ffmpeg, "ffprobe": ffprobe,
                    "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}},
        "counts": {"assets": len(rows), "pairs": len(pair_conditions), "source_groups": len({str(a["source_group"]) for a in assets}),
                   "cache_reused": sum(bool(r["cache_reused"]) for r in row_receipts), "decoded_fresh": sum(not bool(r["cache_reused"]) for r in row_receipts)},
        "row_receipts": row_receipts,
        "outputs": {"features_csv": {"path": str(features_path), "sha256": csv_sha, "bytes": csv_bytes}},
    }
    receipt_path = output / "feature_extraction_receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(_json_bytes(receipt))
    print(json.dumps({"status": "COMPLETE", "extension_id": args.extension_id, "rows": len(rows), "features_sha256": csv_sha,
                      "receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtensionStageError as exc:
        print(f"FEATURE_EXTENSION_STAGE_ERROR:{exc}", file=sys.stderr)
        raise SystemExit(2)
