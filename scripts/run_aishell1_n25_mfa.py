#!/usr/bin/env python3
"""Run strict, speaker-grouped MFA for the frozen AISHELL-1 n=25 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SILENCE = {"", "sil", "sp", "spn", "<sil>", "SIL", "SPN"}
INTERVAL_RE = re.compile(
    r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_lab_text(text: str) -> str:
    return re.sub(r"[^㐀-鿿A-Za-z0-9]+", "", text.strip())


def parse_phones(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    marker = 'name = "phones"'
    if marker not in raw:
        marker = 'name = "phone"'
    if marker not in raw:
        raise ValueError(f"phones tier missing: {path}")
    tier = raw.split(marker, 1)[1].split("name =", 1)[0]
    tokens: list[dict[str, Any]] = []
    for match in INTERVAL_RE.finditer(tier):
        start, end, label = float(match.group(1)), float(match.group(2)), match.group(3).strip()
        if end <= start:
            continue
        tokens.append({
            "token": label,
            "raw_token": label,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "duration_s": round(end - start, 6),
            "is_silence": label in SILENCE or label.casefold() in SILENCE,
            "confidence": 1.0,
        })
    if not tokens or not any(not token["is_silence"] for token in tokens):
        raise ValueError(f"phones tier has no speech phones: {path}")
    if any(tokens[i]["end_s"] > tokens[i + 1]["start_s"] for i in range(len(tokens) - 1)):
        raise ValueError(f"non-monotonic phones tier: {path}")
    return tokens


def run_align(mfa_bin: Path, input_dir: Path, output_dir: Path, log_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(mfa_bin), "align", str(input_dir), "mandarin_mfa", "mandarin_mfa",
        str(output_dir), "--single_speaker", "--clean", "--num_jobs", "4",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"MFA failed rc={result.returncode}: {result.stderr[-1000:]}")


def generate(
    cohort_path: Path,
    tts_meta_path: Path,
    outdir: Path,
    mfa_bin: Path,
) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    tts_meta = json.loads(tts_meta_path.read_text(encoding="utf-8"))
    records = list(cohort.get("records", []))
    if cohort.get("cohort", {}).get("sample_count") != 25 or len(records) != 25:
        raise ValueError("n=25 cohort manifest is incomplete")
    tts_by_id: dict[str, Mapping[str, Any]] = {}
    for row in tts_meta.get("results", {}).values():
        sample_id = str(row["sample_id"])
        if sample_id in tts_by_id:
            raise ValueError(f"duplicate TTS sample_id: {sample_id}")
        tts_by_id[sample_id] = row
    if set(tts_by_id) != {str(row["sample_id"]) for row in records}:
        raise ValueError("TTS metadata does not exactly cover cohort")

    outdir = outdir.resolve()
    input_root = outdir / "mfa_input"
    output_root = outdir / "mfa_out"
    logs_root = outdir / "logs"
    pair_stems: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[tuple[Mapping[str, Any], Path, str]]] = defaultdict(list)
    for row in records:
        sample_id = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        tts = tts_by_id[sample_id]
        if str(tts.get("paired_key")) != str(row["paired_key"]):
            raise ValueError(f"TTS paired_key mismatch for {sample_id}")
        for condition, source in (
            ("natural", Path(row["audio_path"])),
            ("tts", Path(str(tts["canonical_16k_audio"]))),
        ):
            if not source.is_file():
                raise FileNotFoundError(source)
            stem = f"{speaker}_{sample_id}_{condition}"
            lab = clean_lab_text(str(row["transcript"]))
            if not lab:
                raise ValueError(f"empty cleaned transcript for {sample_id}")
            pair_stems[sample_id][condition] = stem
            grouped[(speaker, condition)].append((row, source, stem))

    for (speaker, condition), items in grouped.items():
        input_dir = input_root / speaker / condition
        input_dir.mkdir(parents=True, exist_ok=True)
        for row, source, stem in items:
            shutil.copyfile(source, input_dir / f"{stem}.wav")
            (input_dir / f"{stem}.lab").write_text(clean_lab_text(str(row["transcript"])) + "\n", encoding="utf-8")
        run_align(mfa_bin.resolve(), input_dir, output_root / speaker / condition, logs_root / f"{speaker}_{condition}.log")

    token_records: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        speaker = str(row["speaker_id"])
        sides: dict[str, Any] = {}
        try:
            for condition in ("natural", "tts"):
                stem = pair_stems[sample_id][condition]
                grid = output_root / speaker / condition / f"{stem}.TextGrid"
                tokens = parse_phones(grid)
                sides[condition] = {
                    "speaker_id": speaker,
                    "condition": condition,
                    "tokens": tokens,
                    "textgrid": str(grid),
                    "textgrid_sha256": sha256_file(grid),
                    "audio": str((input_root / speaker / condition / f"{stem}.wav")),
                    "audio_sha256": sha256_file(input_root / speaker / condition / f"{stem}.wav"),
                    "phone_count": len(tokens),
                    "speech_phone_count": sum(not token["is_silence"] for token in tokens),
                }
            token_records[sample_id] = {
                "sample_id": sample_id,
                "paired_key": row["paired_key"],
                "speaker_id": speaker,
                "split": row["split"],
                "transcript": row["transcript"],
                "natural": sides["natural"],
                "tts": sides["tts"],
            }
        except Exception as exc:
            failures.append({"sample_id": sample_id, "paired_key": row["paired_key"], "error": str(exc)})

    summary = {
        "schema_version": 1,
        "manifest_type": "aishell1_mfa_linear_n25_tokens",
        "cohort_manifest": str(cohort_path.resolve()),
        "cohort_manifest_sha256": sha256_file(cohort_path.resolve()),
        "tts_meta": str(tts_meta_path.resolve()),
        "tts_meta_sha256": sha256_file(tts_meta_path.resolve()),
        "samples_total": len(records),
        "samples_ok": len(token_records),
        "failures": failures,
        "complete": len(token_records) == 25 and not failures,
        "records": token_records,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tokens.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mfa-bin", type=Path, default=Path.home() / "miniconda3/envs/mfa/bin/mfa")
    args = parser.parse_args(argv)
    result = generate(args.cohort.resolve(), args.tts_meta.resolve(), args.outdir.resolve(), args.mfa_bin.resolve())
    print(json.dumps({"samples_ok": result["samples_ok"], "failures": len(result["failures"]), "complete": result["complete"]}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
