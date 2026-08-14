#!/usr/bin/env python3
"""Build n25 TTS metadata using n15's in-memory scipy resample_poly contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resample_to_16k(source: Path, destination: Path) -> dict[str, Any]:
    values, source_rate = sf.read(source, dtype="float32", always_2d=False)
    if values.ndim != 1 or int(source_rate) != 24000:
        raise ValueError(f"expected mono 24k source: {source} rate={source_rate} ndim={values.ndim}")
    output = resample_poly(np.asarray(values, dtype=np.float32), 2, 3).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite resampled output: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, 16000, subtype="FLOAT")
    return {"source_sample_rate": int(source_rate), "source_samples": int(values.size), "canonical_sample_rate": 16000, "canonical_samples": int(output.size), "canonical_sha256": sha256_file(destination)}


def build(cohort_path: Path, strict_tts_meta_path: Path, outdir: Path) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_tts_meta_path.read_text(encoding="utf-8"))
    records = list(cohort.get("records", []))
    strict_by_id = {str(row["sample_id"]): row for row in strict.get("results", {}).values()}
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(records) != 25 or len(strict_by_id) != 25:
        raise ValueError("expected complete n25 cohort and strict TTS metadata")
    outdir = outdir.resolve(); audio_dir = outdir / "tts_resample_poly_16k"; audio_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for row in records:
        sid = str(row["sample_id"]); strict_row = strict_by_id.get(sid)
        if strict_row is None or strict_row["paired_key"] != row["paired_key"] or strict_row["speaker_id"] != row["speaker_id"]:
            raise ValueError(f"identity mismatch at {sid}")
        source = Path(str(row["tts_source_path"]))
        if not source.is_file() or sha256_file(source) != row["tts_source_sha256"]:
            raise ValueError(f"strict source hash mismatch at {sid}")
        output = audio_dir / f"{sid}.wav"
        qc = resample_to_16k(source, output)
        results[sid] = {"sample_id": int(sid), "paired_key": row["paired_key"], "speaker_id": row["speaker_id"], "split": row["split"], "condition": "tts", "transcript": row["transcript"], "provider": "frozen_strict_manifest_source", "source_audio": str(source), "source_audio_sha256": row["tts_source_sha256"], "canonical_16k_audio": str(output), "canonical_audio_sha256": qc["canonical_sha256"], "resample_contract": "scipy.signal.resample_poly(up=2, down=3, float32, in_memory_equivalent)", "resample_qc": qc, "status": "ok"}
    summary = {"schema_version": 1, "manifest_type": "aishell1_n25_tts_resample_poly_16k", "cohort_manifest": str(cohort_path.resolve()), "cohort_manifest_sha256": sha256_file(cohort_path.resolve()), "strict_tts_meta": str(strict_tts_meta_path.resolve()), "strict_tts_meta_sha256": sha256_file(strict_tts_meta_path.resolve()), "provider": "frozen_strict_manifest_source", "samples_total": 25, "samples_ok": len(results), "failures": [], "complete": len(results) == 25, "results": results}
    outdir.mkdir(parents=True, exist_ok=True); (outdir / "tts_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--cohort", type=Path, required=True); parser.add_argument("--strict-tts-meta", type=Path, required=True); parser.add_argument("--outdir", type=Path, required=True); args = parser.parse_args()
    result = build(args.cohort.resolve(), args.strict_tts_meta.resolve(), args.outdir.resolve())
    print(json.dumps({"samples_ok": result["samples_ok"], "complete": result["complete"]}))
