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
CJK_WORD_RE = re.compile(r"[一-鿿]+")
SILENCE_LABELS = {"", "sil", "sp", "spn", "<sil>"}
DIGITS = "零一二三四五六七八九"
UNIT_AT = {2: "十", 3: "百", 4: "千"}


def cn_number(n: int) -> str:
    """Read an integer in Mandarin (supports 0..99999999, larger via digit-by-digit)."""
    if n == 0:
        return "零"
    if n >= 100_000_000:
        return "".join(DIGITS[int(c)] for c in str(n))
    wan, rest = divmod(n, 10_000)
    parts: list[str] = []
    if wan:
        parts.append(cn_under_10000(wan, leading=True))
        parts.append("万")
    if rest:
        if wan and rest < 1_000:
            parts.append("零")
        parts.append(cn_under_10000(rest, leading=wan == 0))
    return "".join(parts)


def cn_under_10000(n: int, leading: bool) -> str:
    if n == 0:
        return "零"
    s = str(n)
    length = len(s)
    parts: list[str] = []
    zero_pending = False
    for i, ch in enumerate(s):
        pos = length - i
        d = int(ch)
        if d == 0:
            if parts:
                zero_pending = True
            continue
        if zero_pending and parts:
            parts.append("零")
        zero_pending = False
        if d == 1 and pos == 2 and i == 0 and leading:
            parts.append("十")
            continue
        parts.append(DIGITS[d])
        unit = UNIT_AT.get(pos)
        if unit:
            parts.append(unit)
    return "".join(parts)


def clean_lab_text(text: str) -> str:
    """Normalize a transcript for MFA mandarin_mfa dictionary lookup.

    - CJK runs split into individual characters (character-segmented);
    - punctuation dropped (MFA also treats it as word break markers);
    - arabic numerals expanded to their Mandarin reading;
    - "<digits>%" expands to "百分之<reading>";
    - bare Latin tokens kept as-is (they will be OOV -> spn).
    """
    tokens = []
    for token in re.findall(r"[一-鿿]+|[0-9]+%|[0-9]+|[A-Za-z]+|[%]", text):
        if CJK_WORD_RE.fullmatch(token):
            tokens.extend(token)
        elif re.fullmatch(r"[0-9]+%", token):
            tokens.append("百分之" + cn_number(int(token[:-1])))
        elif re.fullmatch(r"[0-9]+", token):
            tokens.append(cn_number(int(token)))
        elif token == "%":
            tokens.append("百分之")
        else:
            tokens.append(token)
    return " ".join(tokens)


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
