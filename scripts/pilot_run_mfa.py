#!/usr/bin/env python3
"""Run MFA (mandarin_mfa) alignment on the pilot utterances, both sides.

For each sample builds .wav + .lab for the natural side (manifest audio) and
the tts side (canonical 16k), runs `mfa align` once per dataset, parses the
phones tier of every TextGrid into token spans, and writes tokens.json.

Run with the conda mfa env available at --mfa-bin.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

TEXTGRID_INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:\s*"
    r"xmin\s*=\s*([\d.eE+-]+)\s*"
    r"xmax\s*=\s*([\d.eE+-]+)\s*"
    r'text\s*=\s*"([^"]*)"',
    re.DOTALL,
)
KEEP_CHARS_RE = re.compile(r"[^一-鿿A-Za-z0-9%]+")
SILENCE_LABELS = {"", "sil", "sp", "spn", "<sil>"}


def clean_lab_text(text: str) -> str:
    cleaned = KEEP_CHARS_RE.sub("", text.strip())
    return cleaned


def parse_phones(textgrid: Path) -> list[dict[str, Any]]:
    raw = textgrid.read_text(encoding="utf-8")
    tier = None
    for name in ("phones", "phone"):
        marker = f'name = "{name}"'
        if marker in raw:
            tier = raw.split(marker, 1)[1]
            break
    if tier is None:
        raise ValueError(f"no phones tier in {textgrid}")
    tier = tier.split("name =", 1)[0]
    tokens: list[dict[str, Any]] = []
    for match in TEXTGRID_INTERVAL_RE.finditer(tier):
        start, end, phone = float(match.group(1)), float(match.group(2)), match.group(3).strip()
        if end <= start:
            continue
        tokens.append({
            "token": phone,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "duration_s": round(end - start, 6),
            "is_silence": phone.casefold() in SILENCE_LABELS,
        })
    return tokens


def build_inputs(manifest: Mapping[str, Any], tts_meta: Mapping[str, Any], input_dir: Path) -> dict[str, dict[str, str]]:
    input_dir.mkdir(parents=True, exist_ok=True)
    tts_by_id = {str(row["sample_id"]): row for row in tts_meta.get("results", {}).values()}
    pairs: dict[str, dict[str, str]] = {}
    for row in manifest["records"]:
        sample_id = str(row["sample_id"])
        natural = Path(row["audio_path"])
        tts = Path(tts_by_id[sample_id]["canonical_16k_audio"])
        for condition, wav in (("natural", natural), ("tts", tts)):
            stem = f"{sample_id}__{condition}"
            cleaned = clean_lab_text(row["transcript"])
            if not cleaned:
                raise ValueError(f"empty lab text for sample {sample_id}")
            shutil.copyfile(wav, input_dir / f"{stem}.wav")
            (input_dir / f"{stem}.lab").write_text(cleaned + "\n", encoding="utf-8")
            pairs.setdefault(sample_id, {})[condition] = stem
    return pairs


def run_mfa(mfa_bin: Path, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(mfa_bin), "align", str(input_dir),
        "mandarin_mfa", "mandarin_mfa", str(output_dir),
        "--single_speaker", "--clean", "--num_jobs", "4",
    ]
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    print(result.stdout[-1500:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-2500:], flush=True)
        raise RuntimeError(f"mfa align failed rc={result.returncode}")


def build_tokens(pairs: dict[str, dict[str, str]], align_out: Path, outdir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for sample_id, stems in pairs.items():
        side_tokens: dict[str, Any] = {}
        ok = True
        for condition, stem in stems.items():
            grid = align_out / f"{stem}.TextGrid"
            if not grid.is_file():
                failures.append({"sample_id": sample_id, "condition": condition, "error": f"missing {grid}"})
                ok = False
                continue
            try:
                tokens = parse_phones(grid)
            except ValueError as exc:
                failures.append({"sample_id": sample_id, "condition": condition, "error": str(exc)})
                ok = False
                continue
            if not tokens:
                failures.append({"sample_id": sample_id, "condition": condition, "error": "empty phones tier"})
                ok = False
                continue
            side_tokens[condition] = {
                "tokens": tokens,
                "textgrid": str(grid),
                "phone_count": len(tokens),
                "speech_phone_count": sum(1 for t in tokens if not t["is_silence"]),
                "starts_at_zero": abs(tokens[0]["start_s"]) <= 1e-6,
            }
        if ok:
            records[sample_id] = side_tokens
    summary = {
        "schema_version": 1,
        "samples_total": len(pairs),
        "samples_ok": len(records),
        "failures": failures,
        "records": records,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tokens.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({"samples_ok": len(records), "failures": len(failures)}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mfa-bin", type=Path, default=Path.home() / "miniconda3/envs/mfa/bin/mfa")
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tts_meta = json.loads(args.tts_meta.read_text(encoding="utf-8"))
    input_dir = outdir / "mfa_input"
    align_out = outdir / "mfa_out"
    pairs = build_inputs(manifest, tts_meta, input_dir)
    run_mfa(args.mfa_bin.resolve(), input_dir, align_out)
    build_tokens(pairs, align_out, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
