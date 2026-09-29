"""Freeze the 8-speaker strict AISHELL cohort and generate its MFA-linear arm."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .assets import read_pcm16_mono, validate_audio_pair
from .common import ProtocolError, canonical_json_sha256, file_sha256, verify_json, write_json
from .config import load_config

SPEAKERS = ("S0765", "S0770", "S0901", "S0906", "S0912", "S0913", "S0914", "S0915")
SAMPLE_IDS = tuple(str(base + offset) for base in (1, 51, 101, 151, 201, 251, 301, 351) for offset in range(8))


def _token_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = source.get("tokens")
    if not source.get("strict_gate_pass") or not isinstance(rows, list) or not rows:
        raise ProtocolError(f"STRICT_TOKENS_MISSING:{source.get('sample_id')}:{source.get('condition')}")
    result = []
    for token in rows:
        start, end = float(token["start_s"]), float(token["end_s"])
        if end < start:
            raise ProtocolError("INVALID_MFA_TOKEN_TIME")
        result.append({**token, "duration_s": end - start})
    return result


def build_cohort(config: Mapping[str, Any], source_path: Path, canonical_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if int(source.get("counts", {}).get("strict_pairs", -1)) != 392:
        raise ProtocolError("STRICT_SOURCE_PAIR_COUNT_CHANGED")
    paired: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in source["records"]:
        key = str(row["paired_key"])
        side = str(row["condition"])
        if side not in {"natural", "tts"} or side in paired[key]:
            raise ProtocolError(f"STRICT_PAIR_DUPLICATE:{key}:{side}")
        paired[key][side] = row
    selected: list[dict[str, Any]] = []
    tokens: dict[str, Any] = {}
    tts: dict[str, Any] = {}
    for speaker_index, speaker in enumerate(SPEAKERS):
        keys = sorted(key for key, sides in paired.items() if sides.get("natural", {}).get("speaker_id") == speaker)
        if len(keys) < 8:
            raise ProtocolError(f"STRICT_SPEAKER_TOO_SMALL:{speaker}")
        for ordinal, key in enumerate(keys[:8]):
            sides = paired[key]
            if set(sides) != {"natural", "tts"}:
                raise ProtocolError(f"STRICT_PAIR_INCOMPLETE:{key}")
            natural, synthetic = sides["natural"], sides["tts"]
            sid = SAMPLE_IDS[speaker_index * 8 + ordinal]
            if str(natural["sample_id"]) != sid or str(synthetic["sample_id"]) != sid:
                raise ProtocolError(f"STRICT_SAMPLE_ID_MISMATCH:{key}:{sid}")
            for field in ("paired_key", "speaker_id", "transcript", "split"):
                if natural.get(field) != synthetic.get(field):
                    raise ProtocolError(f"STRICT_PAIR_{field.upper()}_MISMATCH:{sid}")
            if synthetic.get("tts_provider") != "faster_qwen3" or natural.get("speaker_id") != speaker:
                raise ProtocolError(f"STRICT_SOURCE_CONDITION_MISMATCH:{sid}")
            local_paths = {}
            for side, row in sides.items():
                wav = Path(config["paths"]["cohort_audio_root"]) / side / f"{int(sid):04d}.wav"
                if not wav.is_file() or file_sha256(wav) != row["source_sha256"]:
                    raise ProtocolError(f"STRICT_AUDIO_SHA_MISMATCH:{sid}:{side}")
                info = sf.info(wav)
                rate, count = int(info.samplerate), int(info.frames)
                local_paths[side] = str(wav.resolve())
                if rate != (16000 if side == "natural" else 24000) or info.channels != 1 or count < rate:
                    raise ProtocolError(f"STRICT_AUDIO_INVALID:{sid}:{side}")
            canonical = canonical_dir / f"{sid}.wav"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            values, rate = sf.read(local_paths["tts"], dtype="float32", always_2d=False)
            if rate != 24000 or values.ndim != 1:
                raise ProtocolError(f"TTS_RESAMPLE_INPUT_INVALID:{sid}")
            converted = resample_poly(np.asarray(values, dtype=np.float32), 2, 3).astype(np.float32)
            if not np.isfinite(converted).all():
                raise ProtocolError(f"TTS_RESAMPLE_NONFINITE:{sid}")
            if canonical.exists():
                existing, existing_rate = sf.read(canonical, dtype="float32", always_2d=False)
                if existing_rate != 16000 or not np.array_equal(existing, converted):
                    raise ProtocolError(f"TTS_CANONICAL_SOURCE_MISMATCH:{sid}")
            else:
                sf.write(canonical, converted, 16000, subtype="FLOAT")
            canonical_info = sf.info(canonical)
            if canonical_info.samplerate != 16000 or canonical_info.channels != 1 or canonical_info.subtype != "FLOAT":
                raise ProtocolError(f"TTS_CANONICAL_INVALID:{sid}")
            selected.append({"sample_id": sid, "paired_key": key, "speaker_id": speaker,
                             "split": natural["split"], "transcript": natural["transcript"],
                             "audio_path": local_paths["natural"], "natural_source_sha256": natural["source_sha256"],
                             "tts_audio_path": local_paths["tts"], "tts_source_sha256": synthetic["source_sha256"],
                             "prior_seen": int(sid) in set(range(1, 6)) | set(range(101, 106)) | set(range(201, 206)) | set(range(251, 256)) | set(range(301, 306))})
            tokens[sid] = {"sample_id": sid, "paired_key": key, "speaker_id": speaker,
                           "split": natural["split"], "transcript": natural["transcript"],
                           "natural": {"audio_sha256": natural["source_sha256"], "tokens": _token_rows(natural)},
                           "tts": {"audio_sha256": synthetic["source_sha256"], "tokens": _token_rows(synthetic)}}
            tts[sid] = {"sample_id": sid, "paired_key": key, "speaker_id": speaker,
                        "split": natural["split"], "transcript": natural["transcript"],
                        "source_audio": local_paths["tts"], "canonical_16k_audio": str(canonical.resolve()),
                        "canonical_audio_sha256": file_sha256(canonical),
                        "resample_contract": "scipy.signal.resample_poly(float32,2,3); sf.write FLOAT 16k",
                        "source_audio_sha256": synthetic["source_sha256"]}
    manifest = {"schema_version": 2, "manifest_type": "mfa_linear_video_retiming_v2_cohort",
                "source_manifest": str(source_path.resolve()), "source_manifest_sha256": file_sha256(source_path),
                "speaker_ids": list(SPEAKERS), "sample_ids": list(SAMPLE_IDS), "records": selected}
    token_payload = {"schema_version": 2, "samples_ok": 64, "failures": [], "records": tokens}
    tts_payload = {"schema_version": 2, "results": tts}
    return manifest, token_payload, tts_payload


def _verify_completed(config: Mapping[str, Any], cohort: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    if int(summary.get("samples_total", -1)) != 64 or int(summary.get("samples_ok", -1)) != 64 or summary.get("failures"):
        raise ProtocolError("INPUT_INCOMPLETE:MFA64")
    results = summary.get("results", {})
    if set(results) != set(SAMPLE_IDS):
        raise ProtocolError("INPUT_INCOMPLETE:MFA_RESULT_IDS")
    if not isinstance(summary.get("model"), Mapping) or not summary["model"]:
        raise ProtocolError("MFA_MODEL_PROVENANCE_MISSING")
    expected_dir = Path(str(summary["cohort_manifest"])).resolve().parent / "pending_generation"
    expected_sources = ("scripts/pilot_generate_mfa_linear.py", "scripts/wavlm_knn_vc_adapter.py",
                        "scripts/knn_vc_retrieval.py")
    bindings = summary.get("generator_bindings", {})
    for source in expected_sources:
        path = (Path(config["repo_root"]) / source).resolve()
        if bindings.get(source) != file_sha256(path):
            raise ProtocolError(f"MFA_GENERATOR_SOURCE_CHANGED:{source}")
    for row in cohort["records"]:
        sid = row["sample_id"]
        m = results[sid]
        if any(str(m.get(field)) != str(row[field]) for field in ("sample_id", "paired_key", "speaker_id", "split", "transcript")):
            raise ProtocolError(f"MFA_IDENTITY_MISMATCH:{sid}")
        if Path(str(m.get("audio_path", ""))).resolve().parent != expected_dir.resolve():
            raise ProtocolError(f"MFA_OUTPUT_FROM_OTHER_RUN:{sid}")
        if not m.get("exact_natural_length") or int(m["natural_samples"]) != int(m["output_samples"]):
            raise ProtocolError(f"MFA_LENGTH_MISMATCH:{sid}")
        validate_audio_pair(row["audio_path"], m["audio_path"], natural_sha256=row["natural_source_sha256"],
                            mfa_sha256=m["audio_sha256"])


def prepare(config: Mapping[str, Any], source_path: Path, output_dir: Path, *, generate_audio: bool = True) -> dict[str, Any]:
    if output_dir.name != "00_cohort":
        raise ProtocolError("COHORT_OUTPUT_DIR_MUST_BE_00_COHORT")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, tokens, tts = build_cohort(config, source_path, output_dir / "tts_canonical_16k")
    files = {"cohort_v2.json": manifest, "tokens_v2.json": tokens, "tts_meta_v2.json": tts}
    for name, payload in files.items():
        path = output_dir / name
        if path.exists():
            existing = verify_json(path, self_hash=True)
            existing.pop("artifact_sha256")
            if canonical_json_sha256(existing) != canonical_json_sha256(payload):
                raise ProtocolError(f"COHORT_PREP_FINGERPRINT_MISMATCH:{name}")
        else:
            write_json(path, payload, self_hash=True)
    summary_path = output_dir / "mfa_summary_v2.json"
    old = verify_json(summary_path, self_hash=True) if summary_path.exists() else None
    results = dict(old.get("results", {})) if old else {}
    model = old.get("model") if old else None
    if old and (old.get("cohort_manifest_sha256") != file_sha256(output_dir / "cohort_v2.json") or old.get("tokens_sha256") != file_sha256(output_dir / "tokens_v2.json")):
        raise ProtocolError("MFA_PREP_INPUT_FINGERPRINT_MISMATCH")
    # Every completed row is revalidated before reuse. A changed source is fatal.
    by_id = {row["sample_id"]: row for row in manifest["records"]}
    for sid, result in list(results.items()):
        row = by_id[sid]
        validate_audio_pair(row["audio_path"], result["audio_path"], natural_sha256=row["natural_source_sha256"],
                            mfa_sha256=result["audio_sha256"])
    missing = [row for row in manifest["records"] if row["sample_id"] not in results]
    ledger_path = output_dir / "gpu_budget.json"
    ledger = verify_json(ledger_path, self_hash=True) if ledger_path.exists() else {"schema_version": 2, "active_gpu_seconds": 0.0, "attempts": []}
    if missing and generate_audio:
        from .run import _resource_preflight
        resources = _resource_preflight(config, output_dir.parent, require_gpu=True)
        if resources["status"] != "READY":
            raise ProtocolError(f"RESOURCE_WAIT:{resources.get('reason')}")
        if float(ledger["active_gpu_seconds"]) >= float(config["budget"]["total_gpu_seconds"]):
            raise ProtocolError("BUDGET_LIMITED:COHORT_PREP")
        subset = {**manifest, "records": missing, "manifest_type": "mfa_linear_video_retiming_v2_subset"}
        subset_path = output_dir / "pending_manifest.json"
        write_json(subset_path, subset)
        batch_dir = output_dir / "pending_generation"
        command = [str(Path(config["paths"]["mfa_python"]).absolute()), str(Path(config["repo_root"]) / "scripts/pilot_generate_mfa_linear.py"),
                   "--manifest", str(subset_path), "--tts-meta", str(output_dir / "tts_meta_v2.json"),
                   "--tokens", str(output_dir / "tokens_v2.json"), "--outdir", str(batch_dir), "--device", "cuda"]
        start = time.monotonic()
        ledger["attempts"].append({"sample_ids": [row["sample_id"] for row in missing], "command": command, "status": "INFLIGHT",
                                   "input_sha256": file_sha256(subset_path)})
        write_json(ledger_path, ledger, self_hash=True)
        log = output_dir / "mfa_generate.log"
        try:
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.run(command, cwd=config["repo_root"], stdout=handle, stderr=subprocess.STDOUT,
                                         timeout=max(1, int(float(config["budget"]["total_gpu_seconds"]) - float(ledger["active_gpu_seconds"]))), check=False)
        finally:
            ledger["active_gpu_seconds"] = float(ledger["active_gpu_seconds"]) + time.monotonic() - start
            ledger["attempts"][-1]["status"] = "RETURNED"
            write_json(ledger_path, ledger, self_hash=True)
        generated = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
        model = generated.get("model")
        for sid, item in generated.get("results", {}).items():
            if sid not in by_id:
                raise ProtocolError(f"MFA_UNEXPECTED_RESULT:{sid}")
            results[sid] = {**item, "transcript": by_id[sid]["transcript"]}
        if process.returncode and not generated.get("failures"):
            raise ProtocolError(f"MFA_GENERATOR_FAILED:{process.returncode}")
    failures = [{"sample_id": row["sample_id"], "reason": "MISSING_MFA_OUTPUT"} for row in manifest["records"] if row["sample_id"] not in results]
    summary = {"schema_version": 2, "arm": "mfa_linear", "status": "COMPLETE" if not failures else "INPUT_INCOMPLETE",
               "cohort_manifest": str((output_dir / "cohort_v2.json").resolve()),
               "cohort_manifest_sha256": file_sha256(output_dir / "cohort_v2.json"),
               "tokens_path": str((output_dir / "tokens_v2.json").resolve()), "tokens_sha256": file_sha256(output_dir / "tokens_v2.json"),
               "samples_total": 64, "samples_ok": len(results), "results": results, "failures": failures,
               "model": model,
               "generator_bindings": {source: file_sha256(Path(config["repo_root"]) / source) for source in
                                      ("scripts/pilot_generate_mfa_linear.py", "scripts/wavlm_knn_vc_adapter.py",
                                       "scripts/knn_vc_retrieval.py")},
               "gpu_budget_path": str(ledger_path.resolve()), "gpu_budget_sha256": file_sha256(ledger_path) if ledger_path.exists() else None}
    write_json(summary_path, summary, self_hash=True)
    if not failures:
        _verify_completed(config, manifest, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", required=True)
    args = parser.parse_args(argv)
    summary = prepare(load_config(args.config), args.source_manifest.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": summary["status"], "samples_ok": summary["samples_ok"], "samples_total": 64}))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
