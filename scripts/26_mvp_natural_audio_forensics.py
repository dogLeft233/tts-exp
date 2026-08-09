#!/usr/bin/env python3
"""Paired natural/MVP audio forensics for AISHELL samples 51--100.

This driver is intentionally explicit: scores must be supplied as a ledger and
no unrelated historical run is discovered automatically.  The analysis is
observational; it describes paired differences and associations, not causes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
DEFAULT_NATURAL = REPO / "results" / "rhythm_style_500" / "aishell1_test_400" / "natural"
DEFAULT_TRANSCRIPTS = REPO / "results" / "rhythm_style_500" / "aishell1_test_400" / "transcripts"
DEFAULT_NATURAL_TEXTGRID = REPO / "results" / "rhythm_style_500" / "aishell1_test_400" / "mfa" / "natural"
DEFAULT_MVP = (REPO / "results" / "rhythm_style_500" / "aishell1_test_400"
               / "test_controls_best" / "mvp" / "conditions" / "mvp")
DEFAULT_OUT = (REPO / "results" / "rhythm_style_500" / "aishell1_test_400"
               / "forensics" / "mvp_vs_natural_51_100")
DEFAULT_IDS = tuple(range(51, 101))
DEFAULT_MFA_BIN = os.environ.get("MFA_BIN", "mfa")
EXPECTED_SUMMARY = {
    "natural_sync_c": 5.04526,
    "mvp_sync_c": 4.16242,
    "natural_sync_d": 7.90412,
    "mvp_sync_d": 7.93172,
    "natural_av_offset": 0.700,
    "mvp_av_offset": 0.080,
}


def parse_ids(spec: str) -> tuple[int, ...]:
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            values.update(range(int(left), int(right) + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("sample ID specification is empty")
    return tuple(sorted(values))


def _sample_id(path: Path) -> int | None:
    match = re.fullmatch(r"0*(\d+)", path.stem)
    return int(match.group(1)) if match else None


def collect_audio_files(directory: Path, sample_ids: Iterable[int]) -> dict[int, Path]:
    expected = set(sample_ids)
    if not directory.is_dir():
        raise FileNotFoundError(f"audio directory does not exist: {directory}")
    found: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in sorted(directory.glob("*.wav")):
        sid = _sample_id(path)
        if sid is None or sid not in expected:
            continue
        if sid in found:
            duplicates.append(sid)
        found[sid] = path
    missing = sorted(expected - set(found))
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            details.append(f"duplicates={sorted(set(duplicates))}")
        raise ValueError(f"invalid audio cohort in {directory}: " + ", ".join(details))
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_metadata(path: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    finite = bool(np.isfinite(data).all())
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    clipped = int(np.sum(np.abs(data) >= 0.999969))
    return {
        "sample_rate_native": int(sr),
        "channels_native": int(data.shape[1]) if data.ndim == 2 else 1,
        "sample_count_native": int(len(data)),
        "duration_native": float(len(data) / sr) if sr else None,
        "subtype": str(info.subtype),
        "sha256": sha256_file(path),
        "peak_native": peak,
        "finite": finite,
        "clipped_samples": clipped,
        "clipped_fraction": float(clipped / data.size) if data.size else 0.0,
    }


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

def load_mvp_sidecars(
    files: dict[int, Path],
    natural_files: dict[int, Path] | None = None,
) -> dict[str, Any]:
    """Validate per-sample MVP metadata against the supplied audio cohort."""
    records: dict[str, Any] = {}
    for sid, wav_path in sorted(files.items()):
        sidecar = wav_path.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"MVP sidecar does not exist: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(payload.get("sample_id", -1)) != sid:
            raise ValueError(f"MVP sidecar sample mismatch: {sidecar}")
        if payload.get("output_path") and Path(str(payload["output_path"])).name != wav_path.name:
            raise ValueError(f"MVP sidecar output mismatch: {sidecar}")
        metadata = wav_metadata(wav_path)
        checks = {
            "sample_rate": metadata["sample_rate_native"],
            "sample_count": metadata["sample_count_native"],
            "finite": metadata["finite"],
            "clipped_sample_count": metadata["clipped_samples"],
        }
        for key, actual in checks.items():
            if key in payload and payload[key] != actual:
                raise ValueError(f"MVP sidecar {key} mismatch for sample {sid}: {payload[key]} != {actual}")
        if payload.get("natural_sample_count") is not None and payload.get("exact_natural_length"):
            if int(payload["natural_sample_count"]) != metadata["sample_count_native"]:
                raise ValueError(f"MVP exact-length metadata mismatch for sample {sid}")
        natural_meta = None
        if natural_files is not None and payload.get("natural_sample_count") is not None:
            natural_meta = wav_metadata(natural_files[sid])
            if int(payload["natural_sample_count"]) != natural_meta["sample_count_native"]:
                raise ValueError(
                    f"MVP sidecar natural provenance mismatch for sample {sid}: "
                    f"sidecar={payload['natural_sample_count']} supplied={natural_meta['sample_count_native']}"
                )
            declared_hash = payload.get("natural_sha256")
            if declared_hash and declared_hash != natural_meta["sha256"]:
                raise ValueError(f"MVP sidecar natural hash mismatch for sample {sid}")
            declared_path = payload.get("natural_path")
            if declared_path and Path(str(declared_path)).name != natural_files[sid].name:
                raise ValueError(f"MVP sidecar natural path mismatch for sample {sid}")
        records[str(sid)] = {
            "sidecar_path": str(sidecar),
            "sample_id": sid,
            "condition": payload.get("condition"),
            "speaker_id": payload.get("speaker_id"),
            "alpha": payload.get("alpha"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "natural_path": payload.get("natural_path"),
            "tts_path": payload.get("tts_path"),
            "natural_sha256": payload.get("natural_sha256"),
            "tts_sha256": payload.get("tts_sha256"),
            "exact_natural_length": payload.get("exact_natural_length"),
            "target_type": payload.get("target_type"),
            "weak_target_warning": payload.get("weak_target_warning"),
            "natural_supplied_sha256": natural_meta["sha256"] if natural_meta else None,
        }
    return records


def _normalized_transcript(text: str) -> str:
    return re.sub(r"[^一-鿿]", "", text)


def collect_transcript_files(directory: Path, sample_ids: Iterable[int]) -> dict[int, Path]:
    expected = set(sample_ids)
    if not directory.is_dir():
        raise FileNotFoundError(f"transcript directory does not exist: {directory}")
    found: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in sorted(directory.glob("*.txt")):
        sid = _sample_id(path)
        if sid is None or sid not in expected:
            continue
        if sid in found:
            duplicates.append(sid)
        found[sid] = path
    missing = sorted(expected - set(found))
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            details.append(f"duplicates={sorted(set(duplicates))}")
        raise ValueError(f"invalid transcript cohort in {directory}: " + ", ".join(details))
    return found


def validate_transcript_cohort(
    transcript_files: dict[int, Path],
    natural_files: dict[int, Path],
) -> dict[str, Any]:
    records = {}
    for sid, transcript_path in sorted(transcript_files.items()):
        text = transcript_path.read_text(encoding="utf-8").strip()
        normalized = _normalized_transcript(text)
        if not normalized:
            raise ValueError(f"empty Mandarin transcript for sample {sid}: {transcript_path}")
        records[str(sid)] = {
            "sample_id": sid,
            "path": str(transcript_path),
            "sha256": sha256_file(transcript_path),
            "text": text,
            "normalized_text": normalized,
            "natural_audio": str(natural_files[sid]),
        }
    return records


def _parse_named_textgrid_tier(text: str, tier_name: str) -> list[tuple[float, float, str]]:
    tier_pattern = re.compile(
        rf'name\s*=\s*"{re.escape(tier_name)}"(?P<body>.*?)(?=\n\s*item \[|\Z)',
        re.S,
    )
    match = tier_pattern.search(text)
    if not match:
        return []
    interval_pattern = re.compile(
        r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([0-9.eE+-]+)\s*'
        r'xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*"([^"]*)"',
        re.S,
    )
    return [(float(a), float(b), label.strip()) for a, b, label in interval_pattern.findall(match.group("body"))]


def parse_textgrid_tiers(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"words": _parse_named_textgrid_tier(text, "words"), "phones": _parse_named_textgrid_tier(text, "phones")}


def _strip_phone_tone(symbol: str) -> str:
    return re.sub(r"[˥˦˧˨˩¹²³⁴⁵0-5]+$", "", symbol.strip())


def validate_textgrid(
    path: Path,
    audio_path: Path,
    transcript: str,
) -> dict[str, Any]:
    tiers = parse_textgrid_tiers(path)
    if not tiers["words"] or not tiers["phones"]:
        raise ValueError(f"TextGrid lacks named words/phones tiers: {path}")
    audio_duration = float(wav_metadata(audio_path)["duration_native"])
    all_intervals = tiers["words"] + tiers["phones"]
    for tier_name, intervals in tiers.items():
        previous = -1e-9
        for start, end, _ in intervals:
            if not (0 <= start < end <= audio_duration + 0.05):
                raise ValueError(f"TextGrid interval outside audio bounds in {path}: {start}, {end}")
            if start < previous - 1e-6:
                raise ValueError(f"TextGrid intervals are not ordered in {tier_name}: {path}")
            previous = end
    word_text = _normalized_transcript("".join(label for _, _, label in tiers["words"] if label))
    expected_text = _normalized_transcript(transcript)
    if word_text and word_text != expected_text:
        raise ValueError(f"TextGrid words/transcript mismatch for {path}")
    phones = [x for x in tiers["phones"] if x[2].strip().lower() not in {"", "sil", "sp", "pau", "spn", "<eps>"}]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "audio_sha256": wav_metadata(audio_path)["sha256"],
        "audio_duration_s": audio_duration,
        "word_count": sum(bool(label.strip()) for _, _, label in tiers["words"]),
        "phone_count": len(phones),
        "phone_coverage": float(sum(b - a for a, b, _ in phones) / audio_duration) if audio_duration else 0.0,
        "tiers": tiers,
        "alignment_source": "mfa_existing" if "mfa" in str(path) else "mfa_generated",
        "strict_gate_pass": True,
    }


def _score_row(raw: dict[str, Any], default_condition: str | None = None) -> dict[str, Any]:
    def pick(*names: str, required: bool = True) -> Any:
        for name in names:
            if name in raw and raw[name] not in (None, ""):
                return raw[name]
        if required:
            raise ValueError(f"score row missing one of {names}: {raw}")
        return None

    def pick(*names: str, required: bool = True) -> Any:
        for name in names:
            if name in raw and raw[name] not in (None, ""):
                return raw[name]
        if required:
            raise ValueError(f"score row missing one of {names}: {raw}")
        return None

    sid = int(pick("sample_id", "sample", "id"))
    condition = str(pick("condition", "arm", required=False) or default_condition or "")
    aliases = {"nat": "natural", "natural_raw": "natural", "mvp_raw": "mvp"}
    condition = aliases.get(condition, condition)
    if condition not in {"natural", "mvp"}:
        raise ValueError(f"unsupported score condition {condition!r} for sample {sid}")
    return {
        "sample_id": sid,
        "condition": condition,
        "sync_c": float(pick("sync_c", "sync-c", "confidence")),
        "sync_d": float(pick("sync_d", "sync-d", "min_distance")),
        "av_offset": float(pick("av_offset", "offset", required=False))
        if pick("av_offset", "offset", required=False) is not None else None,
        "run_id": str(pick("run_id", "eval_run", required=False) or ""),
        "video_path": str(pick("video_path", "video", required=False) or ""),
    }


def load_score_ledger(path: Path, sample_ids: Iterable[int]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"score ledger does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        payload = load_json_or_jsonl(path)
        if isinstance(payload, dict):
            raw_rows = payload.get("scores", payload.get("rows", payload))
            if isinstance(raw_rows, dict):
                raw_rows = [dict(value, sample_id=key) for key, value in raw_rows.items()]
        else:
            raw_rows = payload
    if not isinstance(raw_rows, list):
        raise ValueError("score ledger must contain a list of rows")
    rows = [_score_row(dict(row)) for row in raw_rows]
    expected = set(sample_ids)
    keys = [(row["sample_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("score ledger contains duplicate sample/condition rows")
    actual = set(keys)
    required = {(sid, condition) for sid in expected for condition in ("natural", "mvp")}
    if actual != required:
        raise ValueError(f"score ledger cohort mismatch: missing={sorted(required - actual)}, extra={sorted(actual - required)}")
    return sorted(rows, key=lambda row: (row["sample_id"], row["condition"]))


def score_pairs(rows: list[dict[str, Any]], sample_ids: Iterable[int]) -> list[dict[str, Any]]:
    by_key = {(row["sample_id"], row["condition"]): row for row in rows}
    result = []
    for sid in sample_ids:
        natural = by_key[(sid, "natural")]
        mvp = by_key[(sid, "mvp")]
        result.append({
            "sample_id": sid,
            "natural_sync_c": natural["sync_c"], "mvp_sync_c": mvp["sync_c"],
            "delta_sync_c": mvp["sync_c"] - natural["sync_c"],
            "sync_c_decline": natural["sync_c"] - mvp["sync_c"],
            "natural_sync_d": natural["sync_d"], "mvp_sync_d": mvp["sync_d"],
            "delta_sync_d": mvp["sync_d"] - natural["sync_d"],
            "natural_av_offset": natural["av_offset"], "mvp_av_offset": mvp["av_offset"],
            "delta_av_offset": (mvp["av_offset"] - natural["av_offset"])
            if natural["av_offset"] is not None and mvp["av_offset"] is not None else None,
            "natural_run_id": natural.get("run_id", ""), "mvp_run_id": mvp.get("run_id", ""),
            "natural_video_path": natural.get("video_path", ""), "mvp_video_path": mvp.get("video_path", ""),
        })
    return result


def reconcile_scores(
    pairs: list[dict[str, Any]],
    tolerance: float = 1e-4,
    expected: dict[str, float] | None = None,
) -> dict[str, float]:
    summary = {
        "natural_sync_c": float(np.mean([row["natural_sync_c"] for row in pairs])),
        "mvp_sync_c": float(np.mean([row["mvp_sync_c"] for row in pairs])),
        "delta_sync_c": float(np.mean([row["delta_sync_c"] for row in pairs])),
        "natural_sync_d": float(np.mean([row["natural_sync_d"] for row in pairs])),
        "mvp_sync_d": float(np.mean([row["mvp_sync_d"] for row in pairs])),
        "delta_sync_d": float(np.mean([row["delta_sync_d"] for row in pairs])),
    }
    offsets = [(row["natural_av_offset"], row["mvp_av_offset"]) for row in pairs
               if row["natural_av_offset"] is not None and row["mvp_av_offset"] is not None]
    if offsets:
        summary["natural_av_offset"] = float(np.mean([x[0] for x in offsets]))
        summary["mvp_av_offset"] = float(np.mean([x[1] for x in offsets]))
        summary["delta_av_offset"] = summary["mvp_av_offset"] - summary["natural_av_offset"]
    if expected is not None:
        mismatches = {key: (summary.get(key), value) for key, value in expected.items()
                      if key in summary and abs(summary[key] - value) > tolerance}
        if mismatches:
            raise ValueError(f"score ledger does not reconcile with supplied paired evaluation: {mismatches}")
    return summary


def _import_feature_extractor():
    path = REPO / "scripts" / "08_audio_features_v2.py"
    spec = importlib.util.spec_from_file_location("audio_features_v2_forensics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load feature extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_features_v2


def extract_audio_features(files: dict[int, Path], condition: str) -> list[dict[str, Any]]:
    extractor = _import_feature_extractor()
    rows = []
    for sid, path in sorted(files.items()):
        metadata = wav_metadata(path)
        features = extractor(str(path), target_sr=16000)
        if features is None:
            raise ValueError(f"feature extraction failed: {path}")
        row = {"sample_id": sid, "condition": condition, "filepath": str(path), **metadata, **features}
        rows.append(row)
    return rows


def paired_feature_rows(natural: list[dict[str, Any]], mvp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nat = {row["sample_id"]: row for row in natural}
    gen = {row["sample_id"]: row for row in mvp}
    keys = sorted(set(nat) & set(gen))
    skip = {"sample_id", "condition", "filepath", "sha256", "subtype", "finite"}
    features = sorted(set(nat[keys[0]]) & set(gen[keys[0]]) - skip) if keys else []
    output = []
    for sid in keys:
        row: dict[str, Any] = {"sample_id": sid}
        for feature in features:
            a, b = nat[sid].get(feature), gen[sid].get(feature)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                row[f"natural_{feature}"] = a
                row[f"mvp_{feature}"] = b
                row[f"delta_{feature}"] = b - a if a is not None and b is not None else None
            else:
                row[f"natural_{feature}"] = a
                row[f"mvp_{feature}"] = b
        output.append(row)
    return output


def probe_mfa(
    mfa_bin: str | Path,
    dictionary: str | Path,
    acoustic_model: str | Path,
) -> dict[str, Any]:
    """Probe MFA and explicitly report whether requested Mandarin assets exist."""
    executable = shutil.which(str(mfa_bin)) or (str(mfa_bin) if Path(str(mfa_bin)).exists() else None)
    result: dict[str, Any] = {
        "mfa_bin": str(mfa_bin),
        "resolved_mfa_bin": executable,
        "dictionary": str(dictionary),
        "acoustic_model": str(acoustic_model),
        "status": "unavailable",
        "errors": [],
    }
    if executable is None:
        result["errors"].append("MFA executable not found")
        return result

    def capture(*args: str) -> tuple[int, str, str]:
        proc = subprocess.run([executable, *args], capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr

    code, stdout, stderr = capture("version")
    result["version_returncode"] = code
    result["version"] = (stdout or stderr).strip()
    if code != 0:
        result["errors"].append(stderr.strip() or "mfa version failed")

    for kind, requested_path in (("dictionary", dictionary), ("acoustic", acoustic_model)):
        code, stdout, stderr = capture("model", "list", kind)
        result[f"{kind}_inventory_returncode"] = code
        result[f"{kind}_inventory"] = (stdout or stderr).strip()
        requested_name = Path(str(requested_path)).name
        exists = Path(str(requested_path)).is_file() or requested_name in (stdout + "\n" + stderr)
        result[f"{kind}_available"] = bool(exists)
        if code != 0:
            result["errors"].append(f"mfa model list {kind} failed")
    result["status"] = "available" if (
        result.get("dictionary_available") and result.get("acoustic_available") and not result["errors"]
    ) else "blocked"
    return result


def stage_mfa_corpus(
    audio_files: dict[int, Path],
    transcript_files: dict[int, Path],
    root: Path,
    condition: str,
) -> dict[str, Any]:
    """Stage only the requested cohort for an isolated MFA invocation."""
    condition_root = root / condition
    audio_dir = condition_root / "audio"
    transcript_dir = condition_root / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in sorted(audio_files):
        audio_target = audio_dir / f"{sid:04d}.wav"
        transcript_target = transcript_dir / f"{sid:04d}.lab"
        shutil.copy2(audio_files[sid], audio_target)
        shutil.copy2(transcript_files[sid], transcript_target)
        records.append({
            "sample_id": sid,
            "condition": condition,
            "audio_source": str(audio_files[sid]),
            "audio_staged": str(audio_target),
            "audio_sha256": sha256_file(audio_files[sid]),
            "transcript_source": str(transcript_files[sid]),
            "transcript_staged": str(transcript_target),
            "transcript_sha256": sha256_file(transcript_files[sid]),
        })
    manifest = {"condition": condition, "root": str(condition_root), "records": records}
    (condition_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def run_mfa_batch(
    staged_root: Path,
    output_dir: Path,
    mfa_bin: str | Path,
    dictionary: str | Path,
    acoustic_model: str | Path,
    num_jobs: int = 4,
) -> dict[str, Any]:
    """Run strict MFA only when the requested assets are available."""
    probe = probe_mfa(mfa_bin, dictionary, acoustic_model)
    result: dict[str, Any] = {"probe": probe, "status": probe["status"], "output_dir": str(output_dir)}
    if probe["status"] != "available":
        result["reason"] = "requested MFA dictionary/acoustic model unavailable"
        return result
    audio_dir = staged_root / "audio"
    transcript_dir = staged_root / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(probe["resolved_mfa_bin"]), "align", str(audio_dir), str(transcript_dir),
        str(dictionary), str(acoustic_model), str(output_dir), "--single_speaker",
        "--clean", "--num_jobs", str(num_jobs),
    ]
    result["command"] = command
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout[-4000:]
    result["stderr"] = proc.stderr[-4000:]
    result["status"] = "available" if proc.returncode == 0 else "failed"
    if proc.returncode != 0:
        result["reason"] = "MFA command failed"
    return result


def _pause_kind(label: str) -> str | None:
    value = label.strip().lower()
    if value in {"", "sil", "sp", "pau", "<eps>", "h#"}:
        return "pause"
    if value == "spn":
        return "noise"
    return None


def extract_phone_pause_rows(
    alignment: dict[str, Any],
    sample_id: int,
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tiers = alignment["tiers"]
    intervals = tiers["phones"]
    duration = float(alignment["audio_duration_s"])
    speech = [(a, b, label) for a, b, label in intervals if _pause_kind(label) is None and b > a]
    speech_start = min((a for a, _, _ in speech), default=0.0)
    speech_end = max((b for _, b, _ in speech), default=duration)
    phone_rows = []
    pause_source = []
    for a, b, label in intervals:
        kind = _pause_kind(label)
        if kind == "pause":
            pause_source.append((a, b))
        elif kind is None and b > a:
            idx = len(phone_rows)
            phone_rows.append({
                "sample_id": sample_id, "condition": condition, "phone_index": idx,
                "symbol": label.strip(), "base_symbol": _strip_phone_tone(label),
                "start_s": a, "end_s": b, "duration_s": b - a,
                "center_s": (a + b) / 2, "start_norm": a / duration if duration else None,
                "end_norm": b / duration if duration else None,
                "duration_norm": (b - a) / duration if duration else None,
                "audio_sha256": alignment["audio_sha256"], "textgrid_sha256": alignment["sha256"],
                "alignment_status": "strict",
            })
    merged: list[list[float]] = []
    for a, b in sorted(pause_source):
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    pause_rows = []
    for idx, (a, b) in enumerate(merged):
        if b <= speech_start + 1e-6:
            pause_type = "leading"
            left_index, right_index = None, 0 if phone_rows else None
        elif a >= speech_end - 1e-6:
            pause_type = "trailing"
            left_index, right_index = len(phone_rows) - 1 if phone_rows else None, None
        else:
            pause_type = "internal"
            left_index = max((i for i, row in enumerate(phone_rows) if row["end_s"] <= a + 1e-6), default=None)
            right_index = next((i for i, row in enumerate(phone_rows) if row["start_s"] >= b - 1e-6), None)
        pause_rows.append({
            "sample_id": sample_id, "condition": condition, "pause_index": idx,
            "pause_type": pause_type, "start_s": a, "end_s": b, "duration_s": b - a,
            "start_norm": a / duration if duration else None, "end_norm": b / duration if duration else None,
            "left_phone_index": left_index, "right_phone_index": right_index,
            "audio_sha256": alignment["audio_sha256"], "textgrid_sha256": alignment["sha256"],
            "alignment_status": "strict",
        })
    durations = [row["duration_s"] for row in phone_rows]
    internal = [row["duration_s"] for row in pause_rows if row["pause_type"] == "internal"]
    mean_duration = float(np.mean(durations)) if durations else None
    std_duration = float(np.std(durations)) if durations else None
    summary = {
        "alignment_available": True, "alignment_phone_count": len(phone_rows),
        "alignment_total_duration": duration, "speech_span_start": speech_start,
        "speech_span_end": speech_end, "speech_span_duration": max(0.0, speech_end - speech_start),
        "phone_duration_mean": mean_duration, "phone_duration_std": std_duration,
        "phone_duration_cv": (std_duration / mean_duration) if mean_duration else None,
        "internal_pause_count": len(internal), "internal_pause_total": float(sum(internal)),
        "internal_pause_median": float(np.median(internal)) if internal else 0.0,
        "internal_pause_max": float(max(internal, default=0.0)),
        "pause_fraction": float(sum(internal) / duration) if duration else None,
        "leading_silence": float(sum(row["duration_s"] for row in pause_rows if row["pause_type"] == "leading")),
        "trailing_silence": float(sum(row["duration_s"] for row in pause_rows if row["pause_type"] == "trailing")),
        "noise_interval_count": sum(_pause_kind(label) == "noise" for _, _, label in intervals),
    }
    return phone_rows, pause_rows, summary


def extract_local_prosody(audio_path: Path, phone_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract fixed-grid F0 and energy summaries for validated phone intervals."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    frame_length = 1024
    hop_length = 256
    frame_times = librosa.frames_to_time(
        np.arange(1 + max(0, (len(y) - frame_length) // hop_length)),
        sr=sr, hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    try:
        f0, voiced, _ = librosa.pyin(
            y, fmin=50, fmax=600, sr=sr, frame_length=2048, hop_length=hop_length,
        )
        f0 = np.asarray(f0, dtype=float)
        voiced = np.asarray(voiced, dtype=bool)
    except Exception:
        f0 = np.full(len(frame_times), np.nan, dtype=float)
        voiced = np.zeros(len(frame_times), dtype=bool)
    n = min(len(frame_times), len(rms), len(f0), len(voiced))
    frame_times, rms, f0, voiced = frame_times[:n], rms[:n], f0[:n], voiced[:n]
    rows = []
    for phone in phone_rows:
        mask = (frame_times >= phone["start_s"]) & (frame_times < phone["end_s"])
        f0_values = f0[mask & voiced & np.isfinite(f0)]
        energy = rms[mask]
        f0_slope = None
        if len(f0_values) >= 2:
            times = frame_times[mask & voiced & np.isfinite(f0)]
            f0_slope = float(np.polyfit(times - times[0], f0_values, 1)[0]) if len(np.unique(times)) > 1 else None
        energy_db = 20.0 * np.log10(np.maximum(energy, 1e-8)) if len(energy) else np.array([])
        status = "ok" if len(energy) and len(f0_values) else ("unvoiced" if len(energy) else "no_frames")
        rows.append({
            **phone,
            "f0_mean": float(np.mean(f0_values)) if len(f0_values) else None,
            "f0_median": float(np.median(f0_values)) if len(f0_values) else None,
            "f0_std": float(np.std(f0_values)) if len(f0_values) else None,
            "f0_range": float(np.ptp(f0_values)) if len(f0_values) else None,
            "f0_start": float(f0_values[0]) if len(f0_values) else None,
            "f0_end": float(f0_values[-1]) if len(f0_values) else None,
            "f0_slope_hz_s": f0_slope,
            "voiced_frames": int(len(f0_values)),
            "frame_count": int(len(energy)),
            "voiced_fraction": float(np.mean(voiced[mask])) if len(energy) else None,
            "rms_mean": float(np.mean(energy)) if len(energy) else None,
            "rms_median": float(np.median(energy)) if len(energy) else None,
            "rms_std": float(np.std(energy)) if len(energy) else None,
            "rms_max": float(np.max(energy)) if len(energy) else None,
            "energy_db_mean": float(np.mean(energy_db)) if len(energy_db) else None,
            "energy_db_std": float(np.std(energy_db)) if len(energy_db) else None,
            "prosody_status": status,
            "prosody_method": "librosa.pyin+rms",
            "prosody_frame_hop_s": hop_length / sr,
        })
    return rows


def pair_phone_rows(natural: list[dict[str, Any]], mvp: list[dict[str, Any]], sample_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nat_labels = [row["base_symbol"] for row in natural]
    mvp_labels = [row["base_symbol"] for row in mvp]
    exact = len(natural) == len(mvp) and nat_labels == mvp_labels
    audit = {
        "sample_id": sample_id,
        "phone_pairing_status": "exact" if exact else "mismatch",
        "natural_phone_count": len(natural),
        "mvp_phone_count": len(mvp),
        "label_mismatch_count": sum(a != b for a, b in zip(nat_labels, mvp_labels)) + abs(len(natural) - len(mvp)),
    }
    if not exact:
        return [], audit
    rows = []
    for i, (a, b) in enumerate(zip(natural, mvp)):
        rows.append({
            "sample_id": sample_id, "phone_index": i, "symbol": a["symbol"],
            "natural_symbol": a["symbol"], "mvp_symbol": b["symbol"],
            "natural_duration_s": a["duration_s"], "mvp_duration_s": b["duration_s"],
            "delta_duration_s": b["duration_s"] - a["duration_s"],
            "abs_delta_duration_s": abs(b["duration_s"] - a["duration_s"]),
            "natural_start_s": a["start_s"], "mvp_start_s": b["start_s"],
            "delta_start_s": b["start_s"] - a["start_s"],
            "natural_end_s": a["end_s"], "mvp_end_s": b["end_s"],
            "delta_end_s": b["end_s"] - a["end_s"],
            "abs_boundary_error_s": max(abs(b["start_s"] - a["start_s"]), abs(b["end_s"] - a["end_s"])),
        })
    return rows, audit


def pair_pause_rows(natural: list[dict[str, Any]], mvp: list[dict[str, Any]], sample_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compatible = len(natural) == len(mvp) and all(a["pause_type"] == b["pause_type"] for a, b in zip(natural, mvp))
    audit = {
        "sample_id": sample_id,
        "pause_pairing_status": "ordinal_type_match" if compatible else "structure_mismatch",
        "natural_pause_count": len(natural), "mvp_pause_count": len(mvp),
    }
    if not compatible:
        return [], audit
    rows = []
    for i, (a, b) in enumerate(zip(natural, mvp)):
        rows.append({
            "sample_id": sample_id, "pause_index": i, "pause_type": a["pause_type"],
            "natural_duration_s": a["duration_s"], "mvp_duration_s": b["duration_s"],
            "delta_duration_s": b["duration_s"] - a["duration_s"],
            "natural_start_s": a["start_s"], "mvp_start_s": b["start_s"],
            "delta_start_s": b["start_s"] - a["start_s"],
            "natural_end_s": a["end_s"], "mvp_end_s": b["end_s"],
            "delta_end_s": b["end_s"] - a["end_s"],
        })
    return rows, audit


def pair_prosody_rows(natural: list[dict[str, Any]], mvp: list[dict[str, Any]], sample_id: int) -> list[dict[str, Any]]:
    if len(natural) != len(mvp) or [r["base_symbol"] for r in natural] != [r["base_symbol"] for r in mvp]:
        return []
    rows = []
    fields = ["f0_mean", "f0_std", "f0_range", "f0_slope_hz_s", "voiced_fraction", "rms_mean", "rms_std", "rms_max", "energy_db_mean", "energy_db_std"]
    for i, (a, b) in enumerate(zip(natural, mvp)):
        row = {"sample_id": sample_id, "phone_index": i, "symbol": a["symbol"]}
        for field in fields:
            av, bv = a.get(field), b.get(field)
            row[f"natural_{field}"] = av
            row[f"mvp_{field}"] = bv
            row[f"delta_{field}"] = bv - av if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
        rows.append(row)
    return rows


def aggregate_alignment_features(
    sample_id: int,
    natural_summary: dict[str, Any],
    mvp_summary: dict[str, Any],
    phone_pairs: list[dict[str, Any]],
    pause_pairs: list[dict[str, Any]],
    prosody_pairs: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"sample_id": sample_id, **audit}
    numeric = sorted(set(natural_summary) & set(mvp_summary))
    for key in numeric:
        av, bv = natural_summary.get(key), mvp_summary.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            row[f"natural_{key}"] = av
            row[f"mvp_{key}"] = bv
            row[f"delta_{key}"] = bv - av
    if phone_pairs:
        boundary = [r["abs_boundary_error_s"] for r in phone_pairs]
        row["paired_phone_count"] = len(phone_pairs)
        row["boundary_error_mean_s"] = float(np.mean(boundary))
        row["boundary_error_median_s"] = float(np.median(boundary))
        row["boundary_error_max_s"] = float(np.max(boundary))
        row["duration_abs_error_mean_s"] = float(np.mean([r["abs_delta_duration_s"] for r in phone_pairs]))
    else:
        row["paired_phone_count"] = 0
    if pause_pairs:
        row["paired_pause_count"] = len(pause_pairs)
        row["pause_duration_abs_error_mean_s"] = float(np.mean([abs(r["delta_duration_s"]) for r in pause_pairs]))
    else:
        row["paired_pause_count"] = 0
    if prosody_pairs:
        for field in ("f0_mean", "f0_std", "f0_range", "f0_slope_hz_s", "voiced_fraction", "rms_mean", "rms_std", "rms_max", "energy_db_mean", "energy_db_std"):
            values = [r[f"delta_{field}"] for r in prosody_pairs if isinstance(r.get(f"delta_{field}"), (int, float))]
            if values:
                row[f"delta_{field}_mean"] = float(np.mean(values))
                row[f"delta_{field}_median"] = float(np.median(values))
    row["alignment_feature_status"] = "available" if phone_pairs or pause_pairs else "unavailable"
    return row
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"xmin\s*=\s*([0-9.eE+-]+).*?xmax\s*=\s*([0-9.eE+-]+).*?text\s*=\s*\"([^\"]*)\"", re.S)
    return [(float(a), float(b), label.strip()) for a, b, label in pattern.findall(text)]


def read_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([0-9.eE+-]+)\s*"
        r"xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
        re.S,
    )
    return [(float(a), float(b), label.strip()) for a, b, label in pattern.findall(text)]


def textgrid_features(path: Path) -> dict[str, Any]:
    intervals = read_textgrid_intervals(path)
    if not intervals:
        raise ValueError(f"no intervals found in TextGrid: {path}")
    sil_labels = {"", "sil", "sp", "spn", "<eps>", "h#"}
    pauses = [(a, b) for a, b, label in intervals if label.lower() in sil_labels and b > a]
    phones = [(a, b, label) for a, b, label in intervals if label.lower() not in sil_labels and b > a]
    pauses.sort()
    merged: list[list[float]] = []
    for a, b in pauses:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    internal = [p for p in merged if p[0] > (intervals[0][0] if intervals else 0) and p[1] < (intervals[-1][1] if intervals else 0)]
    durations = np.array([b - a for a, b, _ in phones], dtype=float)
    total = intervals[-1][1] - intervals[0][0]
    return {
        "alignment_available": True,
        "alignment_phone_count": len(phones),
        "alignment_total_duration": total,
        "phone_duration_mean": float(np.mean(durations)) if len(durations) else None,
        "phone_duration_std": float(np.std(durations)) if len(durations) else None,
        "phone_duration_cv": float(np.std(durations) / np.mean(durations)) if len(durations) and np.mean(durations) else None,
        "internal_pause_count": len(internal),
        "internal_pause_total": float(sum(b - a for a, b in internal)),
        "internal_pause_median": float(np.median([b - a for a, b in internal])) if internal else 0.0,
        "internal_pause_max": float(max((b - a for a, b in internal), default=0.0)),
        "pause_fraction": float(sum(b - a for a, b in internal) / total) if total > 0 else None,
        "leading_silence": float(max(0.0, pauses[0][1] - pauses[0][0])) if pauses and pauses[0][0] <= intervals[0][0] + 1e-6 else 0.0,
        "trailing_silence": float(max(0.0, pauses[-1][1] - pauses[-1][0])) if pauses and pauses[-1][1] >= intervals[-1][1] - 1e-6 else 0.0,
    }


def paired_permutation(values: np.ndarray, n: int = 10000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    null = np.empty(n)
    for i in range(n):
        null[i] = float(np.mean(values * rng.choice([-1.0, 1.0], size=len(values))))
    return float(np.mean(np.abs(null) >= abs(observed))), observed


def bootstrap_mean(values: np.ndarray, n: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.mean(values))


def feature_correlations(pairs: list[dict[str, Any]], feature_rows: list[dict[str, Any]], family: str = "global_acoustics") -> list[dict[str, Any]]:
    score = {row["sample_id"]: row for row in pairs}
    output = []
    if not pairs or not feature_rows:
        return output
    for key in sorted(feature_rows[0]):
        if not key.startswith("delta_"):
            continue
        feature = key[6:]
        x, y_c, y_d = [], [], []
        for row in feature_rows:
            value = row.get(key)
            sid = row["sample_id"]
            if isinstance(value, (int, float)) and math.isfinite(value) and sid in score:
                x.append(value); y_c.append(score[sid]["delta_sync_c"]); y_d.append(score[sid]["delta_sync_d"])
        for target in ("delta_sync_c", "sync_c_decline", "delta_sync_d"):
            if target == "sync_c_decline":
                x_target = []
                y_target = []
                for row in feature_rows:
                    sid = row["sample_id"]
                    value = row.get(key)
                    decline = score[sid].get("sync_c_decline")
                    if isinstance(value, (int, float)) and math.isfinite(value) and isinstance(decline, (int, float)) and sid in score:
                        x_target.append(value)
                        y_target.append(decline)
                x_use, y_use = x_target, y_target
            else:
                x_use, y_use = x, (y_c if target == "delta_sync_c" else y_d)
            if len(x_use) < 3 or len(set(x_use)) < 2 or len(set(y_use)) < 2:
                rho, p = None, None
                status = "insufficient_or_constant"
            else:
                rho, p = spearmanr(x_use, y_use)
                rho, p = float(rho), float(p)
                status = "ok"
            output.append({"family": family, "feature": feature, "target": target, "n": len(x_use), "spearman_rho": rho, "p_value": p, "status": status})
    valid = [row for row in output if row["p_value"] is not None]
    for row in output:
        row["q_value"] = None
    for target in ("delta_sync_c", "sync_c_decline", "delta_sync_d"):
        subset = [row for row in valid if row["target"] == target]
        order = sorted(subset, key=lambda row: row["p_value"])
        running = 1.0
        for rank, row in reversed(list(enumerate(order, 1))):
            running = min(running, row["p_value"] * len(order) / rank)
            row["q_value"] = min(1.0, running)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def build_report(summary: dict[str, Any], correlations: list[dict[str, Any]], alignment_available: bool) -> str:
    top = sorted((row for row in correlations if row.get("q_value") is not None), key=lambda row: row["q_value"])[:10]
    lines = ["# MVP vs natural audio forensics (samples 51–100)", "", "## Status", ""]
    statuses = summary.get("statuses", {})
    for key, value in statuses.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Paired score result", ""]
    if summary.get("scores"):
        lines.append(f"- Paired samples: {summary['n_pairs']}")
        lines.append(f"- Sync-C: natural {summary['scores']['natural_sync_c']:.5f}; MVP {summary['scores']['mvp_sync_c']:.5f}; MVP-natural {summary['scores']['delta_sync_c']:.5f}; decline {summary['scores']['sync_c_decline']:.5f}")
        lines.append(f"- Sync-D: natural {summary['scores']['natural_sync_d']:.5f}; MVP {summary['scores']['mvp_sync_d']:.5f}; MVP-natural {summary['scores']['delta_sync_d']:.5f}")
        if "natural_av_offset" in summary["scores"]:
            lines.append(f"- AV offset: natural {summary['scores']['natural_av_offset']:.3f}; MVP {summary['scores']['mvp_av_offset']:.3f}; MVP-natural {summary['scores']['delta_av_offset']:.3f}")
    else:
        lines.append("- SyncNet score ledger was unavailable; score deltas and score correlations were not computed.")
    lines += ["", "## Interpretation", "", "The analysis is observational. A feature difference or correlation with SyncNet does not establish causality and cannot by itself separate waveform effects from Ditto, preprocessing, tracking, or SyncNet interactions.", ""]
    lines.append("MVP alignment-derived timing and pause features were available." if alignment_available else "MVP alignment-derived timing and pause features were unavailable; no uniform or source-TTS fallback was used.")
    lines += ["", "## Highest FDR-ranked associations", ""]
    if top:
        for row in top:
            lines.append(f"- `{row['feature']}` vs `{row['target']}`: rho={row['spearman_rho']:.3f}, p={row['p_value']:.4g}, q={row['q_value']:.4g}, n={row['n']}")
    else:
        lines.append("- No non-constant feature association was estimable.")
    lines += ["", "## Evidence boundaries", "", "The historical paired score pattern rules out a simple claim that MVP failed because of a larger global AV offset: its mean offset improved while Sync-C declined. Timing, pause structure, local prosody, phonetic clarity, and coupled acoustic transformations remain hypotheses until independent MVP MFA and a complete per-sample score ledger are available.", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sample_ids = parse_ids(args.ids)
    natural_files = collect_audio_files(Path(args.natural_dir), sample_ids)
    mvp_files = collect_audio_files(Path(args.mvp_dir), sample_ids)
    transcript_files = collect_transcript_files(Path(args.transcript_dir), sample_ids)
    transcript_records = validate_transcript_cohort(transcript_files, natural_files)
    mvp_sidecars = load_mvp_sidecars(mvp_files, natural_files)

    score_rows = load_score_ledger(Path(args.scores), sample_ids) if args.scores else []
    pairs = score_pairs(score_rows, sample_ids) if score_rows else []
    scores = reconcile_scores(pairs, tolerance=args.score_tolerance) if pairs else {}

    natural_features = extract_audio_features(natural_files, "natural")
    mvp_features = extract_audio_features(mvp_files, "mvp")
    audio_pairs = paired_feature_rows(natural_features, mvp_features)
    correlations = feature_correlations(pairs, audio_pairs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mfa_root = Path(args.mfa_output_dir) if args.mfa_output_dir else out / "mfa"
    staging_root = out / "mfa_staging"
    staging = {}
    if not args.skip_mfa:
        staging["natural"] = stage_mfa_corpus(natural_files, transcript_files, staging_root, "natural")
        staging["mvp"] = stage_mfa_corpus(mvp_files, transcript_files, staging_root, "mvp")
    (out / "mfa_staging_manifest.json").write_text(
        json.dumps(staging, indent=2, sort_keys=True), encoding="utf-8"
    )
    probe = probe_mfa(args.mfa_bin, args.mfa_dictionary, args.mfa_model)
    mfa_result = {"probe": probe, "natural": None, "mvp": None}
    if not args.skip_mfa:
        mfa_result["mvp"] = run_mfa_batch(
            staging_root / "mvp", mfa_root / "mvp", args.mfa_bin,
            args.mfa_dictionary, args.mfa_model, args.mfa_num_jobs,
        )
    (out / "mfa_probe.json").write_text(json.dumps(mfa_result, indent=2, sort_keys=True), encoding="utf-8")

    natural_tg_dir = Path(args.natural_textgrid_dir)
    mvp_tg_dir = Path(args.mvp_textgrid_dir) if args.mvp_textgrid_dir else mfa_root / "mvp"
    phone_rows: list[dict[str, Any]] = []
    pause_rows: list[dict[str, Any]] = []
    prosody_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    phone_delta_rows: list[dict[str, Any]] = []
    pause_delta_rows: list[dict[str, Any]] = []
    prosody_delta_rows: list[dict[str, Any]] = []
    alignment_audit: list[dict[str, Any]] = []
    natural_alignments: dict[int, dict[str, Any]] = {}
    mvp_alignments: dict[int, dict[str, Any]] = {}

    for sid in sample_ids:
        natural_path = natural_tg_dir / f"{sid:04d}.TextGrid"
        if not natural_path.is_file():
            alignment_audit.append({"sample_id": sid, "alignment_status": "natural_missing"})
            continue
        natural_alignment = validate_textgrid(natural_path, natural_files[sid], transcript_records[str(sid)]["text"])
        natural_alignments[sid] = natural_alignment
        nat_phone, nat_pause, nat_summary = extract_phone_pause_rows(natural_alignment, sid, "natural")
        nat_prosody = extract_local_prosody(natural_files[sid], nat_phone)
        phone_rows.extend(nat_phone)
        pause_rows.extend(nat_pause)
        prosody_rows.extend(nat_prosody)

        mvp_path = mvp_tg_dir / f"{sid:04d}.TextGrid"
        if not mvp_path.is_file():
            alignment_audit.append({"sample_id": sid, "alignment_status": "mvp_unavailable", "natural_phone_count": len(nat_phone)})
            continue
        mvp_alignment = validate_textgrid(mvp_path, mvp_files[sid], transcript_records[str(sid)]["text"])
        mvp_alignments[sid] = mvp_alignment
        mvp_phone, mvp_pause, mvp_summary = extract_phone_pause_rows(mvp_alignment, sid, "mvp")
        mvp_prosody = extract_local_prosody(mvp_files[sid], mvp_phone)
        phone_rows.extend(mvp_phone)
        pause_rows.extend(mvp_pause)
        prosody_rows.extend(mvp_prosody)

        paired_phones, phone_audit = pair_phone_rows(nat_phone, mvp_phone, sid)
        paired_pauses, pause_audit = pair_pause_rows(nat_pause, mvp_pause, sid)
        paired_prosody = pair_prosody_rows(nat_prosody, mvp_prosody, sid)
        audit = {**phone_audit, **pause_audit}
        alignment_audit.append({**audit, "alignment_status": "paired" if paired_phones else "sequence_or_count_mismatch"})
        phone_delta_rows.extend(paired_phones)
        pause_delta_rows.extend(paired_pauses)
        prosody_delta_rows.extend(paired_prosody)
        alignment_rows.append(aggregate_alignment_features(
            sid, nat_summary, mvp_summary, paired_phones, paired_pauses, paired_prosody, audit,
        ))

    alignment_available = len(mvp_alignments) == len(sample_ids)
    correlations.extend(feature_correlations(pairs, alignment_rows, "timing_pause_prosody"))
    status = {
        "corpus": "complete",
        "natural_alignment": "complete" if len(natural_alignments) == len(sample_ids) else "partial",
        "mvp_alignment": "complete" if alignment_available else ("blocked" if mfa_result["mvp"] and mfa_result["mvp"].get("status") == "blocked" else "unavailable"),
        "phone_pause_prosody": "complete" if alignment_available else "unavailable",
        "score_ledger": "complete" if pairs else "unavailable",
        "correlations": "complete" if pairs and alignment_rows else "unavailable",
    }
    status["overall"] = "complete" if all(value == "complete" for key, value in status.items() if key != "overall") else "partial"
    summary = {
        "n_pairs": len(pairs), "sample_ids": list(sample_ids), "scores": scores,
        "alignment_available": alignment_available, "statuses": status,
        "transcripts": transcript_records, "mvp_sidecars": mvp_sidecars,
        "audio_files": {"natural": {str(k): str(v) for k, v in natural_files.items()}, "mvp": {str(k): str(v) for k, v in mvp_files.items()}},
        "alignment_audit": alignment_audit, "mfa": mfa_result,
    }
    write_csv(out / "paired_scores.csv", pairs)
    write_csv(out / "paired_audio_features.csv", audio_pairs)
    write_csv(out / "phone_intervals.csv", phone_rows)
    write_csv(out / "pause_intervals.csv", pause_rows)
    write_csv(out / "prosody_intervals.csv", prosody_rows)
    write_csv(out / "paired_phone_deltas.csv", phone_delta_rows)
    write_csv(out / "paired_pause_deltas.csv", pause_delta_rows)
    write_csv(out / "paired_prosody_deltas.csv", prosody_delta_rows)
    write_csv(out / "paired_alignment_features.csv", alignment_rows)
    write_csv(out / "alignment_audit.csv", alignment_audit)
    write_csv(out / "correlations.csv", correlations)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "config_and_provenance.json").write_text(json.dumps({
        "argv": sys.argv, "repo": str(REPO), "sample_ids": list(sample_ids),
        "natural_dir": str(args.natural_dir), "mvp_dir": str(args.mvp_dir),
        "transcript_dir": str(args.transcript_dir), "natural_textgrid_dir": str(args.natural_textgrid_dir),
        "mvp_textgrid_dir": str(args.mvp_textgrid_dir) if args.mvp_textgrid_dir else None,
        "scores": str(args.scores) if args.scores else None,
        "mfa_bin": str(args.mfa_bin), "mfa_dictionary": str(args.mfa_dictionary), "mfa_model": str(args.mfa_model),
        "no_source_tts_or_uniform_fallback": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    (out / "report.md").write_text(build_report(summary, correlations, alignment_available), encoding="utf-8")
    if args.require_complete and status["overall"] != "complete":
        raise ValueError(f"analysis is incomplete: {status}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-dir", default=DEFAULT_NATURAL, type=Path)
    parser.add_argument("--mvp-dir", default=DEFAULT_MVP, type=Path)
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPTS, type=Path)
    parser.add_argument("--scores", type=Path, help="explicit CSV/JSON/JSONL score ledger")
    parser.add_argument("--natural-textgrid-dir", default=DEFAULT_NATURAL_TEXTGRID, type=Path)
    parser.add_argument("--mvp-textgrid-dir", type=Path)
    parser.add_argument("--mfa-bin", default=DEFAULT_MFA_BIN)
    parser.add_argument("--mfa-dictionary", default="mandarin_mfa")
    parser.add_argument("--mfa-model", default="mandarin_mfa")
    parser.add_argument("--mfa-num-jobs", default=4, type=int)
    parser.add_argument("--mfa-output-dir", type=Path)
    parser.add_argument("--skip-mfa", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--ids", default="51-100")
    parser.add_argument("--score-tolerance", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="use fewer permutations/bootstrap samples")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        # Kept as a public CLI switch for smoke runs; statistical routines remain deterministic.
        pass
    try:
        summary = run(args)
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["scores"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
