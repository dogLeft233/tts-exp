#!/usr/bin/env python3
"""Run a non-overwriting MFA probe for one frozen Mandarin compound token.

This diagnostic preserves the original TextGrids and qualification manifest.
It changes only the LAB tokenization for one already-selected EVAL pair, using
an exact compound entry already present in the frozen MFA dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


SAMPLE_ID = "S0064_BAC009S0064W0121"
SPEAKER = "S0064"
EXPECTED_AUDIO_SHA256 = {
    "natural": "6950b7da2ef6e437c429ccc288817e86893951792d442215c56e359659ca7da5",
    "tts": "950d4cd832ac904462f0636d9112a5d49c6bafc5cc989d6b5cef64586a052e11",
}
OLD_LAB = "确 实 侵 蚀 了 不 少 房 企 的 净 利 润"
NEW_LAB = "确 实 侵蚀 了 不 少 房 企 的 净 利 润"
UNKNOWN = {"spn", "unk", "unknown", "oov", "<unk>", "<unknown>"}
SILENCE = {"", "sil", "sp", "spn", "<sil>"}
INTERVAL_RE = re.compile(
    r"intervals\s+\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*"
    r"xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def fingerprint(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def load_grid(path: Path, audio_duration: float) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    xmax_match = re.search(r"^\s*xmax\s*=\s*([\d.]+)", raw, re.MULTILINE)
    grid_xmax = float(xmax_match.group(1)) if xmax_match else None
    tiers: list[dict[str, Any]] = []
    for block in re.split(r"(?=\s*item\s+\[\d+\]:)", raw):
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        intervals = [
            {"start_s": float(m.group(1)), "end_s": float(m.group(2)), "label": m.group(3)}
            for m in INTERVAL_RE.finditer(block)
        ]
        tiers.append({"name": name_match.group(1), "intervals": intervals})

    phone_tier = next(
        (tier for tier in tiers if tier["name"].casefold() in {"phones", "phone"}), None
    )
    word_tier = next(
        (tier for tier in tiers if tier["name"].casefold() in {"words", "word"}), None
    )
    if phone_tier is None:
        raise ValueError(f"no phone interval tier in {path}")

    phones = phone_tier["intervals"]
    errors: list[str] = []
    if not phones or not any(item["label"].strip().casefold() not in SILENCE for item in phones):
        errors.append("missing_speech_phone")
    if any(item["label"].strip().casefold() in UNKNOWN for item in phones):
        errors.append("unknown_or_oov_phone")
    for item in phones:
        if item["end_s"] - item["start_s"] <= 1e-6:
            errors.append("nonpositive_phone_interval")
        if item["start_s"] < -0.04 or item["end_s"] > audio_duration + 0.04:
            errors.append("phone_interval_outside_audio")
    for left, right in zip(phones, phones[1:]):
        if left["end_s"] - right["start_s"] > 0.001:
            errors.append("phone_overlap_over_tolerance")
    if grid_xmax is None or abs(grid_xmax - audio_duration) > 0.04:
        errors.append("textgrid_clock_mismatch")

    compound = None
    if word_tier is not None:
        compound = next((item for item in word_tier["intervals"] if item["label"] == "侵蚀"), None)
    return {
        "textgrid_xmax_s": grid_xmax,
        "audio_duration_s": audio_duration,
        "phone_count": len(phones),
        "speech_phone_count": sum(item["label"].strip().casefold() not in SILENCE for item in phones),
        "unknown_oov_phone_count": sum(item["label"].strip().casefold() in UNKNOWN for item in phones),
        "phone_labels": [item["label"] for item in phones],
        "word_tier_has_compound_token": compound is not None,
        "compound_word_interval": compound,
        "errors": sorted(set(errors)),
        "qualified_by_frozen_structure_rules": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-corpus", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mfa-bin", type=Path, required=True)
    parser.add_argument("--mfa-root-dir", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--acoustic-model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cal-qualification", type=Path, required=True)
    parser.add_argument("--old-eligibility", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    corpus_source = args.original_corpus.resolve(strict=True)
    outdir = args.outdir.resolve(strict=False)
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite probe directory: {outdir}")
    old_eligibility = json.loads(args.old_eligibility.resolve(strict=True).read_text(encoding="utf-8"))
    old_row = next((row for row in old_eligibility.get("records", []) if row.get("id") == SAMPLE_ID), None)
    if old_row is None:
        raise ValueError(f"sample {SAMPLE_ID} is missing from the frozen eligibility manifest")
    if old_row.get("pair_qualified") is not False:
        raise ValueError("the original frozen row is not marked unqualified; refusing this targeted probe")
    outdir.mkdir(parents=True)
    corpus = outdir / "corpus" / SPEAKER
    grids = outdir / "textgrids" / SPEAKER
    logs = outdir / "logs"
    corpus.mkdir(parents=True)
    grids.mkdir(parents=True)
    logs.mkdir(parents=True)

    conditions: dict[str, dict[str, Any]] = {}
    for condition in ("natural", "tts"):
        source = corpus_source / f"{SAMPLE_ID}__{condition}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = sha256_file(source)
        if source_sha != EXPECTED_AUDIO_SHA256[condition]:
            raise ValueError(f"frozen {condition} input hash mismatch: {source_sha}")
        audio_path = corpus / source.name
        shutil.copy2(source, audio_path)
        lab_path = audio_path.with_suffix(".lab")
        lab_path.write_text(NEW_LAB + "\n", encoding="utf-8")
        conditions[condition] = {
            "source_audio_path": str(source),
            "source_audio_sha256": source_sha,
            "staged_audio_path": str(audio_path),
            "staged_audio_sha256": sha256_file(audio_path),
            "lab_path": str(lab_path),
            "lab_sha256": sha256_file(lab_path),
            "original_lab_text": OLD_LAB,
            "probe_lab_text": NEW_LAB,
            "original_textgrid_sha256": old_row[condition].get("textgrid_sha256"),
            "expected_textgrid_path": str(grids / f"{SAMPLE_ID}__{condition}.TextGrid"),
        }

    dictionary_lines = args.dictionary.read_text(encoding="utf-8").splitlines()
    dictionary_rows = [line for line in dictionary_lines if line.startswith("侵蚀\t")]
    if not dictionary_rows:
        raise ValueError("frozen dictionary has no exact 侵蚀 compound entry")
    character_rows = [line for line in dictionary_lines if line.startswith("蚀\t")]
    command = [
        str(args.mfa_bin.resolve(strict=True)),
        "align",
        str(corpus),
        str(args.dictionary.resolve(strict=True)),
        str(args.acoustic_model.resolve(strict=True)),
        str(grids),
        "--single_speaker",
        "--clean",
        "--num_jobs",
        "2",
    ]
    version_result = subprocess.run(
        [str(args.mfa_bin.resolve(strict=True)), "version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    version_line = (version_result.stdout + " " + version_result.stderr).strip()
    if version_result.returncode != 0 or "3.4.1" not in version_line:
        raise ValueError(f"expected frozen MFA 3.4.1, got {version_line!r}")
    source_hashes = {
        "protocol_sha256": sha256_file(args.protocol.resolve(strict=True)),
        "cal_qualification_sha256": sha256_file(args.cal_qualification.resolve(strict=True)),
        "old_eligibility_sha256": sha256_file(args.old_eligibility.resolve(strict=True)),
        "dictionary_sha256": sha256_file(args.dictionary.resolve(strict=True)),
        "mfa_binary_sha256": sha256_file(args.mfa_bin.resolve(strict=True)),
    }
    manifest: dict[str, Any] = {
        "schema": "mfa-single-pair-compound-oov-repair-probe-v1",
        "status": "PREPARED",
        "sample_id": SAMPLE_ID,
        "speaker": SPEAKER,
        "split": "evaluation",
        "purpose": "diagnostic alternate tokenization before physical forwards; original qualification remains unchanged",
        "original_eligibility_status": "unobserved_unknown_or_oov_phone",
        "publisher_transcript": "确实侵蚀了不少房企的净利润",
        "transcript_changed": False,
        "tokenization_change": {
            "original_lab_tokens": OLD_LAB.split(),
            "probe_lab_tokens": NEW_LAB.split(),
            "joined_text": "侵蚀",
            "dictionary_compound_entry_exists": True,
            "single_character_entry_exists": bool(character_rows),
            "compound_dictionary_rows": dictionary_rows,
            "single_character_dictionary_rows": character_rows,
        },
        "conditions": conditions,
        "mfa": {
            "version_required": "3.4.1",
            "version_command_output": version_line,
            "binary_path": str(args.mfa_bin.resolve(strict=True)),
            "mfa_root_dir": str(args.mfa_root_dir.resolve(strict=False)),
            "dictionary_path": str(args.dictionary.resolve(strict=True)),
            "dictionary_name": "mandarin_china_mfa",
            "acoustic_model_path": str(args.acoustic_model.resolve(strict=True)),
            "acoustic_model_name": "mandarin_mfa",
            "protocol_sha256": source_hashes["protocol_sha256"],
            "cal_qualification_sha256": source_hashes["cal_qualification_sha256"],
            "old_eligibility_sha256": source_hashes["old_eligibility_sha256"],
            "dictionary_sha256": source_hashes["dictionary_sha256"],
            "mfa_binary_sha256": source_hashes["mfa_binary_sha256"],
            "environment": {"CUDA_VISIBLE_DEVICES": "", "num_jobs": 2},
        },
        "command": command,
    }
    manifest["manifest_sha256"] = fingerprint(manifest)
    manifest_path = outdir / "probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    env = dict(os.environ)
    conda_prefix = args.mfa_bin.resolve(strict=True).parent.parent
    env.update({
        "MFA_ROOT_DIR": str(args.mfa_root_dir.resolve(strict=False)),
        "TMPDIR": str((outdir / "tmp").resolve(strict=False)),
        "CUDA_VISIBLE_DEVICES": "",
        "CONDA_PREFIX": str(conda_prefix),
        "PATH": str(args.mfa_bin.resolve(strict=True).parent) + os.pathsep + os.environ.get("PATH", ""),
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout,
        env=env,
        check=False,
    )
    log_path = logs / "mfa_align.json"
    log_payload = {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alignment_rows: dict[str, Any] = {}
    for condition in ("natural", "tts"):
        grid = Path(conditions[condition]["expected_textgrid_path"])
        if grid.is_file():
            duration_match = re.search(r"^\s*xmax\s*=\s*([\d.]+)", grid.read_text(encoding="utf-8"), re.MULTILINE)
            duration = float(duration_match.group(1)) if duration_match else float("nan")
            quality = load_grid(grid, duration)
            alignment_rows[condition] = {"textgrid_path": str(grid), "textgrid_sha256": sha256_file(grid), **quality}
        else:
            alignment_rows[condition] = {"textgrid_path": str(grid), "textgrid_sha256": None, "errors": ["missing_textgrid"], "qualified_by_frozen_structure_rules": False}

    manifest["status"] = "PROBE_ALIGNED" if result.returncode == 0 else "PROBE_ALIGNMENT_FAILED"
    manifest["alignment_returncode"] = result.returncode
    manifest["alignment_log_path"] = str(log_path)
    manifest["alignment_log_sha256"] = sha256_file(log_path)
    manifest["alignment_rows"] = alignment_rows
    manifest["both_conditions_structurally_qualified"] = bool(
        result.returncode == 0
        and all(row.get("qualified_by_frozen_structure_rules") for row in alignment_rows.values())
    )
    manifest["manifest_sha256"] = fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "both_conditions_structurally_qualified": manifest["both_conditions_structurally_qualified"],
        "alignment_rows": {
            condition: {
                "textgrid_sha256": row.get("textgrid_sha256"),
                "phone_count": row.get("phone_count"),
                "unknown_oov_phone_count": row.get("unknown_oov_phone_count"),
                "errors": row.get("errors"),
                "compound_word_interval": row.get("compound_word_interval"),
            }
            for condition, row in alignment_rows.items()
        },
    }, ensure_ascii=True, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
