"""Run the frozen 148-row primary feature stage while reusing the 24-row smoke cache.

The scientific extractor remains the separately pinned `features.py` source.
This wrapper only validates the full binding, carries forward verified cache
entries for the smoke rows, and calls the unchanged extractor for the other
124 assets. It performs no model, TTS, ASR, video, or SyncNet work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_SHA256 = "ef9cfc52e8a99001686b9af605cf62502f6e70b1c9a9ae1c2001471d12da1622"
INVENTORY_SHA256 = "623049c59c902ecc294f895017ab1f91ea55b188a741e41d7ddfbce1fe2a42d5"
SMOKE_PROTOCOL_SHA256 = "939aefd8861ebf440f968e6cf2a9d05244eb92e1a5c8646a1c920a531e728c6b"
EXTRACTOR_SHA256 = "c7100359a218cd566c678a22c92ac8df0051ffeddc6075a776ae966be0902f0e"
ROOT = Path(__file__).resolve().parents[3]


class FullStageError(RuntimeError):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_extractor(path: Path):
    source_sha, _ = _sha_file(path)
    if source_sha != EXTRACTOR_SHA256:
        raise FullStageError(f"scientific extractor SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location("processing_chain_features_pinned", path)
    if spec is None or spec.loader is None:
        raise FullStageError("cannot import pinned scientific extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source_sha


def _cache_binding(module, asset: dict[str, Any], binaries: dict[str, Any], file_sha: str) -> dict[str, Any]:
    return {
        "source_file_sha256": file_sha,
        "asset_context": {
            "audio_uid": asset.get("audio_uid"),
            "source_layer": asset.get("source_layer"),
            "generator_lineage_verified": module._lineage_generator_verified(asset),
            "source_media_metadata": asset.get("source_media_metadata", {}),
        },
        "ffmpeg": binaries["ffmpeg"],
        "ffprobe": binaries["ffprobe"],
        "decode_args": module.DECODE_ARGS,
        "extractor_sha256": EXTRACTOR_SHA256,
        "schema_version": module.SCHEMA_VERSION,
    }


def _row_from_cached(module, asset: dict[str, Any], path: Path, file_sha: str, file_bytes: int, cached: dict[str, Any], cache_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    values = cached["feature_values"]
    missing = cached["missing_reasons"]
    row = {name: asset.get(name) for name in module.METADATA_COLUMNS if name not in {"pcm_sha256", "missing_reasons_json"}}
    row.update({
        "file_sha256": file_sha,
        "pcm_sha256": cached["pcm_sha256"],
        "source_path": str(path),
        "missing_reasons_json": json.dumps(missing, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        **values,
    })
    receipt = {
        "audio_uid": asset.get("audio_uid"),
        "pair_uid": asset.get("pair_uid"),
        "condition": asset.get("condition"),
        "source_path": str(path),
        "file_sha256": file_sha,
        "file_bytes": file_bytes,
        "pcm_sha256": cached["pcm_sha256"],
        "decoded_sample_count": values.get("timeline_audio_sample_count"),
        "cache_key_sha256": cache_path.stem,
        "cache_path": str(cache_path),
        "cache_reused": True,
        "cache_origin": "qualified 24-wave feature smoke cache; body copied byte-for-byte",
        "ffprobe_stream": cached["probe"].get("stream", {}),
        "ffprobe_format": cached["probe"].get("format", {}),
        "missing_reasons": missing,
    }
    return row, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen full primary feature extraction; reuse only the validated smoke caches.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/protocol_full_stage_v1.json")
    parser.add_argument("--assets-json", type=Path, default=ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/inventory_A/asset_inventory_attempt02.json")
    parser.add_argument("--smoke-receipt", type=Path, default=ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/feature_smoke_attempt01/feature_extraction_receipt.json")
    parser.add_argument("--smoke-protocol", type=Path, default=ROOT / "runs/processing_chain_predictive_audit_20261015_attempt01/protocol_v1.json")
    parser.add_argument("--extractor", type=Path, default=Path(__file__).with_name("features.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started_utc = _utc()
    started = time.perf_counter()
    protocol_path = args.protocol.resolve(strict=True)
    assets_path = args.assets_json.resolve(strict=True)
    smoke_receipt_path = args.smoke_receipt.resolve(strict=True)
    smoke_protocol_path = args.smoke_protocol.resolve(strict=True)
    extractor_path = args.extractor.resolve(strict=True)
    output_arg = args.output_dir.expanduser()
    if output_arg.is_symlink() or output_arg.exists():
        raise FullStageError(f"full feature output must be fresh and not a symlink: {output_arg}")
    output = output_arg.resolve(strict=False)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FullStageError("full feature output parent must already exist and must not be a symlink")

    protocol_sha, protocol_bytes = _sha_file(protocol_path)
    inventory_sha, inventory_bytes = _sha_file(assets_path)
    if protocol_sha != PROTOCOL_SHA256 or inventory_sha != INVENTORY_SHA256:
        raise FullStageError("full-stage protocol or primary inventory differs from the frozen input pins")
    smoke_protocol_sha, smoke_protocol_bytes = _sha_file(smoke_protocol_path)
    if smoke_protocol_sha != SMOKE_PROTOCOL_SHA256:
        raise FullStageError("smoke protocol differs from the qualified cache producer protocol")
    protocol = _load_json(protocol_path)
    smoke_protocol = _load_json(smoke_protocol_path)
    inventory = _load_json(assets_path)
    smoke = _load_json(smoke_receipt_path)
    module, extractor_sha = _load_extractor(extractor_path)
    if smoke.get("source", {}).get("sha256") != EXTRACTOR_SHA256:
        raise FullStageError("smoke receipt does not name the pinned scientific extractor")
    if smoke.get("protocol", {}).get("sha256") != SMOKE_PROTOCOL_SHA256 or smoke.get("asset_inventory", {}).get("sha256") != INVENTORY_SHA256:
        raise FullStageError("smoke receipt source pins differ from the qualified cache input")
    if protocol.get("status") != "FROZEN" or not protocol.get("execution", {}).get("full_feature_extraction_authorized"):
        raise FullStageError("full-stage extraction is not authorized in the frozen protocol")
    if smoke_protocol.get("cohort_binding", {}).get("primary") != protocol.get("cohort_binding", {}).get("primary"):
        raise FullStageError("full-stage primary scientific binding differs from the smoke protocol")

    primary = protocol["cohort_binding"]["primary"]
    assets = inventory.get("assets")
    if not isinstance(assets, list) or len(assets) != 148:
        raise FullStageError("primary inventory must contain exactly 148 assets")
    pair_ids = {row.get("pair_uid") for row in assets}
    group_ids = {row.get("source_group") for row in assets}
    if len(pair_ids) != 74 or pair_ids != set(primary.get("pair_uids", [])) or len(group_ids) != 13:
        raise FullStageError("primary full-cohort pair or source-group roster differs")
    if (len(pair_ids), len(assets), len(group_ids)) != (int(primary["pair_count"]), int(primary["audio_count"]), int(primary["source_group_count"])):
        raise FullStageError("primary full-cohort counts differ from protocol")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in assets:
        by_pair.setdefault(str(row.get("pair_uid")), []).append(row)
        if row.get("source_layer") != "generator" or row.get("condition") not in {"natural", "tts"}:
            raise FullStageError("primary assets must remain the frozen generator-layer natural/tts arms")
        if not Path(str(row.get("source_path", ""))).is_absolute():
            raise FullStageError(f"asset source path is not absolute: {row.get('audio_uid')}")
        digest = str(row.get("file_sha256", ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise FullStageError(f"invalid file SHA for {row.get('audio_uid')}")
    if any(sorted(row.get("condition") for row in rows) != ["natural", "tts"] for rows in by_pair.values()):
        raise FullStageError("every full-stage pair must have exactly one natural and one TTS asset")

    ffmpeg_info = module._tool_version(str(args.ffmpeg))
    ffprobe_info = module._tool_version(str(args.ffprobe))
    binaries = {"ffmpeg": ffmpeg_info, "ffprobe": ffprobe_info}
    smoke_rows = smoke.get("row_receipts", [])
    if len(smoke_rows) != 24 or smoke.get("cohort", {}).get("audio_count") != 24:
        raise FullStageError("qualified smoke receipt must have exactly 24 audio rows")
    asset_by_uid = {str(row["audio_uid"]): row for row in assets}
    smoke_uid_set = {str(row.get("audio_uid")) for row in smoke_rows}
    if len(smoke_uid_set) != 24 or not smoke_uid_set.issubset(asset_by_uid):
        raise FullStageError("smoke rows are not 24 unique members of the exact full inventory")
    smoke_cache: dict[str, tuple[Path, dict[str, Any], str, int]] = {}
    cache_dir_source = None
    for row in smoke_rows:
        uid = str(row["audio_uid"])
        asset = asset_by_uid[uid]
        source_path = Path(str(asset["source_path"])).expanduser()
        source_sha, source_bytes = _sha_file(source_path)
        if source_sha != row.get("file_sha256") or source_sha != asset.get("file_sha256"):
            raise FullStageError(f"smoke cache source body changed: {uid}")
        cache_path = Path(str(row["cache_path"])).resolve(strict=True)
        cache_key = str(row.get("cache_key_sha256", ""))
        if cache_path.name != f"{cache_key}.json" or cache_path.parent.resolve() != Path(str(smoke["outputs"]["features_csv"]["path"])).resolve().parent / "cache":
            raise FullStageError(f"smoke cache path does not match the qualified output tree: {uid}")
        cache_raw_sha, cache_bytes = _sha_file(cache_path)
        cached = _load_json(cache_path)
        binding = _cache_binding(module, asset, binaries, source_sha)
        if cached.get("schema") != "processing_chain_wave_feature_cache_v1" or cached.get("cache_binding") != binding:
            raise FullStageError(f"smoke cache binding is invalid: {uid}")
        if module._sha256_bytes(module._json_bytes(binding)) != cache_key:
            raise FullStageError(f"smoke cache key is invalid: {uid}")
        if row.get("cache_key_sha256") != cache_key or row.get("pcm_sha256") != cached.get("pcm_sha256"):
            raise FullStageError(f"smoke receipt/cache row differs: {uid}")
        cache_dir_source = cache_path.parent
        smoke_cache[uid] = (cache_path, cached, cache_raw_sha, cache_bytes)
    if cache_dir_source is None:
        raise FullStageError("no qualified smoke cache directory was found")

    output.mkdir()
    cache_dir = output / "cache"
    cache_dir.mkdir()
    for uid, (source_cache, _cached, source_cache_sha, source_cache_bytes) in smoke_cache.items():
        destination = cache_dir / source_cache.name
        shutil.copyfile(source_cache, destination)
        copied_sha, copied_bytes = _sha_file(destination)
        if (copied_sha, copied_bytes) != (source_cache_sha, source_cache_bytes):
            raise FullStageError(f"smoke cache copy was not byte-identical: {uid}")

    ordered = sorted(assets, key=lambda row: (row["pair_uid"], row["condition"]))
    feature_rows: list[dict[str, Any]] = []
    row_receipts: list[dict[str, Any]] = []
    for asset in ordered:
        uid = str(asset["audio_uid"])
        source = Path(str(asset["source_path"])).expanduser()
        if uid in smoke_cache:
            source_sha, source_bytes = _sha_file(source)
            cache_name = smoke_cache[uid][0].name
            copied_cache_path = cache_dir / cache_name
            cached = _load_json(copied_cache_path)
            row, receipt = _row_from_cached(module, asset, source, source_sha, source_bytes, cached, copied_cache_path)
        else:
            row, receipt = module._asset_row(asset, source, ffprobe_info["path"], ffmpeg_info["path"], binaries, extractor_sha, cache_dir)
        feature_rows.append(row)
        row_receipts.append(receipt)
    if len(feature_rows) != 148 or len({row["audio_uid"] for row in feature_rows}) != 148:
        raise FullStageError("full-stage output did not produce exactly 148 unique audio rows")
    if sum(bool(row.get("cache_reused")) for row in row_receipts) != 24:
        raise FullStageError("full-stage cache reuse count differs from the exact 24 smoke rows")

    features_path = output / "features.csv"
    module._write_csv(features_path, feature_rows)
    csv_sha, csv_bytes = _sha_file(features_path)
    cache_provenance = []
    for uid, (source_cache, _cached, source_cache_sha, source_cache_bytes) in sorted(smoke_cache.items()):
        cache_provenance.append({"audio_uid": uid, "source_path": str(source_cache), "source_sha256": source_cache_sha, "source_bytes": source_cache_bytes, "copied_path": str(cache_dir / source_cache.name)})
    receipt = {
        "schema": "processing_chain_feature_full_receipt_v1",
        "status": "COMPLETE",
        "scope": "148 primary generator-layer WAV assets; 24 qualified smoke caches reused byte-for-byte; 124 remaining assets passed through unchanged scientific extractor",
        "started_utc": started_utc,
        "finished_utc": _utc(),
        "elapsed_seconds": time.perf_counter() - started,
        "process": {"pid": os.getpid(), "ppid": os.getppid(), "startticks": module._process_startticks(os.getpid()), "argv": sys.argv, "cwd": str(Path.cwd()), "returncode": 0},
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha, "bytes": protocol_bytes},
        "asset_inventory": {"path": str(assets_path), "sha256": inventory_sha, "bytes": inventory_bytes},
        "smoke_protocol": {"path": str(smoke_protocol_path), "sha256": smoke_protocol_sha, "bytes": smoke_protocol_bytes},
        "smoke_receipt": {"path": str(smoke_receipt_path), "sha256": _sha_file(smoke_receipt_path)[0], "bytes": _sha_file(smoke_receipt_path)[1]},
        "source": {"path": str(extractor_path), "sha256": extractor_sha, "bytes": _sha_file(extractor_path)[1]},
        "wrapper": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__).resolve())[0], "bytes": _sha_file(Path(__file__).resolve())[1]},
        "runtime": {"python": sys.version, "ffmpeg": ffmpeg_info, "ffprobe": ffprobe_info, "cpu_threads_requested": 2, "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}, "decode": {"sample_rate_hz": module.SAMPLE_RATE_HZ, "channels": 1, "dtype": "float32 little-endian", "dither_method": "none", "args": module.DECODE_ARGS}},
        "cohort": {"cohort_id": primary["cohort_id"], "pair_count": 74, "source_group_count": 13, "audio_count": 148},
        "cache_reuse": {"smoke_audio_count": 24, "remaining_audio_count": 124, "smoke_cache_body_copy_sha256_checked": True, "cache_dir": str(cache_dir), "smoke_cache_rows": cache_provenance},
        "row_receipts": row_receipts,
        "outputs": {"features_csv": {"path": str(features_path), "sha256": csv_sha, "bytes": csv_bytes}},
        "scientific_extractor_sha256": EXTRACTOR_SHA256,
    }
    receipt_path = output / "feature_extraction_receipt.json"
    receipt_path.write_bytes(_json_bytes(receipt))
    print(json.dumps({"status": "COMPLETE", "rows": len(feature_rows), "reused_smoke_cache": 24, "new_feature_rows": 124, "output": str(output), "features_sha256": csv_sha, "receipt": str(receipt_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FullStageError as exc:
        print(f"FEATURE_FULL_STAGE_ERROR:{exc}", file=sys.stderr)
        raise SystemExit(2)
