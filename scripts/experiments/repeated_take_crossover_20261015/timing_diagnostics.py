"""Audio-only MFA timing diagnostics for the repeated-take crossover.

This helper preserves MFA's original TextGrids and measures phone-timing
differences after subtracting each recording's first recognized-phone onset.
It never warps audio/video and never reads SyncNet outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "repeated_take_crossover_20261015_attempt01"
RUN_DIR = REPO_ROOT / "runs" / RUN_ID
DEFAULT_MANIFEST = (
    RUN_DIR
    / "assets_discovery"
    / "aishell_repeated_realizations"
    / "manifest_v3.json"
)
DEFAULT_ASSET_ROOT = DEFAULT_MANIFEST.parent
OUT_DIR = RUN_DIR / "timing_diagnostics"
CANONICAL_OUT_DIR = RUN_DIR / "timing_diagnostics_canonical32"
DEFAULT_AUDIO_MANIFEST = RUN_DIR / "audio_manifest.json"
# Pins from .codex/scratch/native_axis_fit_mfa_20261014_attempt02_retry02.
MFA = Path("/home/wjj/miniconda3/envs/mfa/bin/mfa")
MFA_PYTHON = Path("/home/wjj/miniconda3/envs/mfa/bin/python")
DICTIONARY = Path(
    "/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict"
)
ACOUSTIC_MODEL = Path(
    "/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip"
)
EXPECTED_VERSION = "2.2.17"
EXPECTED_SHA256 = {
    "mfa": "a9564d7e02a373b4a33030159d7c7ff21420e4d5891f7a2d76279af0aeefedf2",
    "python": "f300e18232f8c20fbd059deb5279973b384a66673724c8a89af622f84ef48af3",
    "dictionary": "0881d4e338e26509d022992cb5b897ead24a176448f534481238cb3a7fe949f0",
    "acoustic_model": "c4c91c499ada0de35346481d4af69c7328075ddb0ce16a8cfb19fc74d0401199",
}
ACTUAL_SILENCE_LABELS = {"", "sil", "sp", "noise"}
UNKNOWN_LABELS = {"spn", "<unk>", "unk"}


class DiagnosticError(RuntimeError):
    """An input, execution, or output does not match this diagnostic contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def _safe_component(value: str, label: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise DiagnosticError(f"unsafe {label}: {value!r}")
    return value


def _resolve_source(row: Mapping[str, Any], asset_root: Path) -> Path:
    relative = row.get("staged_path_relative")
    source = row.get("path")
    if isinstance(relative, str) and relative:
        candidate = (asset_root / relative).resolve()
        if not candidate.is_relative_to(asset_root.resolve()):
            raise DiagnosticError(f"staged path escapes asset root: {relative}")
        return candidate
    if isinstance(source, str) and source:
        return Path(source).resolve()
    raise DiagnosticError("each audio row needs staged_path_relative or path")


def _natural_rows(manifest_path: Path, asset_root: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    manifest_speakers = manifest.get("speakers", [])
    if not isinstance(manifest_speakers, list) or not manifest_speakers:
        raise DiagnosticError("manifest must contain a non-empty speakers array")
    records: list[dict[str, Any]] = []
    seen_speakers: set[str] = set()
    for row in manifest_speakers:
        if not isinstance(row, dict):
            raise DiagnosticError("manifest speaker entries must be objects")
        speaker = str(row.get("speaker", ""))
        if not speaker or speaker in seen_speakers:
            raise DiagnosticError(f"manifest has missing or duplicate speaker: {speaker!r}")
        seen_speakers.add(speaker)
        target = row.get("target_transcript")
        normalized = row.get("normalized_target_transcript")
        if not isinstance(target, str) or not target or target != normalized:
            raise DiagnosticError(f"{speaker} transcript normalization is not exact")
        pair = row.get("target_pair")
        if not isinstance(pair, list):
            raise DiagnosticError(f"{speaker} has no target_pair")
        roles = {item.get("role"): item for item in pair if isinstance(item, dict)}
        if set(roles) != {"N1", "N2"}:
            raise DiagnosticError(f"{speaker} must have exactly N1 and N2")
        for role in ("N1", "N2"):
            item = roles[role]
            if item.get("transcript") != target:
                raise DiagnosticError(f"{speaker}/{role} transcript differs from target")
            records.append(
                {
                    "speaker": speaker,
                    "role": role,
                    "utterance_id": str(item.get("utterance_id", "")),
                    "transcript": target,
                    "source_path": str(_resolve_source(item, asset_root)),
                    "expected_sha256": str(item.get("raw_sha256", "")),
                    "expected_bytes": item.get("source_member_bytes"),
                    "source_kind": "AISHELL-1 distinct utterance ID; repetition metadata unspecified",
                }
            )
    return records


def _extra_rows(path: Path | None, natural: Sequence[Mapping[str, Any]], asset_root: Path) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = read_json(path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise DiagnosticError("extra-input JSON must contain an items array")
    target_by_speaker = {
        row["speaker"]: row["transcript"] for row in natural if row["role"] == "N1"
    }
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise DiagnosticError("extra-input items must be objects")
        speaker = str(item.get("speaker", ""))
        role = str(item.get("role", ""))
        if speaker not in target_by_speaker or role not in {"T1", "T2"}:
            raise DiagnosticError(f"extra input has unsupported speaker/role: {speaker}/{role}")
        if item.get("transcript") != target_by_speaker[speaker]:
            raise DiagnosticError(f"{speaker}/{role} transcript differs from natural target")
        records.append(
            {
                "speaker": speaker,
                "role": role,
                "utterance_id": str(item.get("utterance_id", role)),
                "transcript": item["transcript"],
                "source_path": str(_resolve_source(item, asset_root)),
                "expected_sha256": str(item.get("raw_sha256", item.get("sha256", ""))),
                "expected_bytes": item.get("bytes"),
                "source_kind": str(item.get("source_kind", "TTS output; provider provenance must be retained")),
            }
        )
    return records


def _target_transcripts(manifest_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    manifest = read_json(manifest_path)
    speakers = manifest.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise DiagnosticError("transcript manifest must contain a non-empty speakers array")
    targets: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    for row in speakers:
        if not isinstance(row, dict):
            raise DiagnosticError("transcript manifest speaker entries must be objects")
        speaker = str(row.get("speaker", ""))
        target = row.get("target_transcript")
        if not speaker or speaker in targets or not isinstance(target, str) or not target:
            raise DiagnosticError(f"invalid speaker or target transcript: {speaker!r}")
        if row.get("normalized_target_transcript") != target:
            raise DiagnosticError(f"{speaker} transcript normalization is not exact")
        pair = row.get("target_pair")
        role_rows = {item.get("role"): item for item in pair if isinstance(item, dict)} if isinstance(pair, list) else {}
        if set(role_rows) != {"N1", "N2"} or any(item.get("transcript") != target for item in role_rows.values()):
            raise DiagnosticError(f"{speaker} transcript manifest must pin matching N1/N2 target rows")
        targets[speaker] = target
        rows[speaker] = row
    return targets, rows


def build_canonical_inputs(args: argparse.Namespace) -> None:
    targets, manifest_rows = _target_transcripts(args.manifest.resolve(strict=True))
    if len(targets) != 8:
        raise DiagnosticError(f"expected the frozen eight-speaker transcript set, found {len(targets)}")
    audio_path = args.audio_manifest.resolve(strict=True)
    audio = read_json(audio_path)
    if audio.get("sample_rate_hz") != 16000 or audio.get("trim_pad_warp") is not False:
        raise DiagnosticError("audio manifest must pin unwarped 16 kHz canonical audio")
    target_rms = audio.get("target_rms")
    if not isinstance(target_rms, (int, float)) or target_rms <= 0:
        raise DiagnosticError("audio manifest lacks a valid target RMS")
    records = audio.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("audio manifest must contain records")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DiagnosticError("audio manifest records must be objects")
        key = (str(record.get("speaker", "")), str(record.get("role", "")))
        if key in by_key:
            raise DiagnosticError(f"audio manifest has duplicate canonical row: {key}")
        by_key[key] = record
    expected = {(speaker, role) for speaker in targets for role in ("N1", "N2", "T1", "T2")}
    if set(by_key) != expected:
        missing, extra = sorted(expected - set(by_key)), sorted(set(by_key) - expected)
        raise DiagnosticError(f"audio manifest rows are not the frozen 8x4 cohort; missing={missing}, extra={extra}")

    items: list[dict[str, Any]] = []
    for speaker in targets:
        transcript = targets[speaker]
        for role in ("N1", "N2", "T1", "T2"):
            record = by_key[(speaker, role)]
            if (record.get("sample_rate_hz") != 16000 or record.get("channels") != 1 or
                    record.get("sample_width_bytes") != 2 or record.get("trim_pad_warp") is not False or
                    record.get("target_rms") != target_rms):
                raise DiagnosticError(f"canonical format/clock pin mismatch for {speaker}/{role}")
            path_text = record.get("canonical_path")
            expected_sha = record.get("canonical_sha256")
            if not isinstance(path_text, str) or len(str(expected_sha)) != 64:
                raise DiagnosticError(f"canonical path/hash missing for {speaker}/{role}")
            canonical_path = Path(path_text).resolve(strict=True)
            actual_sha = sha256_file(canonical_path)
            if actual_sha != expected_sha:
                raise DiagnosticError(f"canonical WAV hash differs from audio manifest: {speaker}/{role}")
            natural_row = manifest_rows[speaker]
            target_pair = {row["role"]: row for row in natural_row["target_pair"]}
            utterance_id = str(target_pair[role]["utterance_id"]) if role in target_pair else role
            items.append({
                "speaker": speaker,
                "role": role,
                "utterance_id": utterance_id,
                "transcript": transcript,
                "path": str(canonical_path),
                "canonical_sha256": actual_sha,
                "canonical_bytes": canonical_path.stat().st_size,
                "source_kind": str(record.get("source_kind", "natural canonical" if role.startswith("N") else "TTS canonical")),
                "raw_source_path": record.get("source_path"),
                "raw_source_sha256": record.get("source_sha256"),
                "canonical_pcm_sha256": record.get("canonical_pcm_sha256"),
            })

    output = args.output.resolve()
    if output.exists():
        raise DiagnosticError(f"refusing to overwrite canonical input JSON: {output}")
    write_json(output, {
        "schema_version": 1,
        "run_id": RUN_ID,
        "input_contract": "audio_manifest canonical mono PCM16 16 kHz at fixed RMS; no trim/pad/warp",
        "audio_manifest_path": str(audio_path),
        "audio_manifest_sha256": sha256_file(audio_path),
        "transcript_manifest_path": str(args.manifest.resolve()),
        "transcript_manifest_sha256": sha256_file(args.manifest.resolve()),
        "target_rms": target_rms,
        "sample_rate_hz": 16000,
        "items": items,
    })
    print(f"Wrote {len(items)} canonical audio/transcript rows to {output}")


def _canonical_all_role_rows(path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path.resolve(strict=True))
    targets, _ = _target_transcripts(manifest_path)
    if len(targets) != 8:
        raise DiagnosticError(f"expected the frozen eight-speaker transcript set, found {len(targets)}")
    if payload.get("transcript_manifest_sha256") != sha256_file(manifest_path):
        raise DiagnosticError("canonical input JSON was built from a different transcript manifest")
    audio_manifest_path = payload.get("audio_manifest_path")
    audio_manifest_sha = payload.get("audio_manifest_sha256")
    if (not isinstance(audio_manifest_path, str) or not isinstance(audio_manifest_sha, str) or
            not re.fullmatch(r"[0-9a-f]{64}", audio_manifest_sha)):
        raise DiagnosticError("canonical input JSON lacks its audio-manifest path/hash pin")
    pinned_audio_manifest = Path(audio_manifest_path).resolve(strict=True)
    if sha256_file(pinned_audio_manifest) != audio_manifest_sha:
        raise DiagnosticError("audio manifest changed after canonical input JSON was built")
    if payload.get("sample_rate_hz") != 16000:
        raise DiagnosticError("canonical input JSON lacks its audio-manifest clock/hash pin")
    items = payload.get("items")
    if not isinstance(items, list):
        raise DiagnosticError("canonical input JSON must contain an items array")
    expected = {(speaker, role) for speaker in targets for role in ("N1", "N2", "T1", "T2")}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise DiagnosticError("canonical input items must be objects")
        speaker, role = str(item.get("speaker", "")), str(item.get("role", ""))
        key = (speaker, role)
        if key not in expected or key in seen:
            raise DiagnosticError(f"invalid or duplicate canonical speaker/role: {key}")
        if item.get("transcript") != targets[speaker]:
            raise DiagnosticError(f"{speaker}/{role} transcript differs from the frozen target")
        path_text = item.get("path")
        sha = item.get("canonical_sha256")
        if not isinstance(path_text, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise DiagnosticError(f"canonical path/hash missing for {speaker}/{role}")
        rows.append({
            "speaker": speaker,
            "role": role,
            "utterance_id": str(item.get("utterance_id", role)),
            "transcript": item["transcript"],
            "source_path": str(Path(path_text).resolve()),
            "expected_sha256": sha,
            "expected_bytes": item.get("canonical_bytes"),
            "source_kind": str(item.get("source_kind", "canonical audio manifest source")),
            "raw_source_path": item.get("raw_source_path"),
            "raw_source_sha256": item.get("raw_source_sha256"),
            "canonical_pcm_sha256": item.get("canonical_pcm_sha256"),
        })
        seen.add(key)
    if seen != expected:
        raise DiagnosticError(f"canonical input JSON does not contain exact 8x4 cohort; missing={sorted(expected-seen)}")
    return rows, payload


def _index_path(out_dir: Path) -> Path:
    return out_dir / "input_manifest.json"


def prepare(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    asset_root = args.asset_root.resolve()
    canonical_payload: dict[str, Any] | None = None
    if args.all_inputs is not None:
        if args.extra_inputs is not None:
            raise DiagnosticError("--all-inputs cannot be combined with --extra-inputs")
        if args.out.resolve() == OUT_DIR.resolve():
            raise DiagnosticError("canonical all-role inputs require the separate timing_diagnostics_canonical32 output directory")
        rows, canonical_payload = _canonical_all_role_rows(args.all_inputs, manifest_path)
        expected_count = 4 * len(_target_transcripts(manifest_path)[0])
        if len(rows) != expected_count:
            raise DiagnosticError(f"canonical all-role input must contain {expected_count} rows")
    else:
        natural = _natural_rows(manifest_path, asset_root)
        rows = natural + _extra_rows(args.extra_inputs, natural, asset_root)
    if len({(row["speaker"], row["role"]) for row in rows}) != len(rows):
        raise DiagnosticError("duplicate speaker/role in prepared input rows")

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = _index_path(args.out)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if index_path.exists():
        prior = read_json(index_path)
        for item in prior.get("items", []):
            existing[(item["speaker"], item["role"])] = item
        if args.all_inputs is not None:
            requested_keys = {(row["speaker"], row["role"]) for row in rows}
            if not set(existing) <= requested_keys:
                raise DiagnosticError("canonical output directory already contains rows outside the final all-role input set")

    staged: list[dict[str, Any]] = list(existing.values())
    for row in rows:
        speaker = _safe_component(row["speaker"], "speaker")
        role = _safe_component(row["role"], "role")
        utterance_id = _safe_component(row["utterance_id"], "utterance_id")
        source = Path(row["source_path"]).resolve(strict=True)
        if not source.is_file():
            raise DiagnosticError(f"source is not a file: {source}")
        actual_sha = sha256_file(source)
        if len(row["expected_sha256"]) != 64 or actual_sha != row["expected_sha256"]:
            raise DiagnosticError(f"source SHA256 does not match manifest: {source}")
        actual_bytes = source.stat().st_size
        if row["expected_bytes"] is not None and actual_bytes != int(row["expected_bytes"]):
            raise DiagnosticError(f"source byte count does not match manifest: {source}")

        stem = f"{utterance_id}__{role}"
        record = {
            **row,
            "input_stem": stem,
            "source_sha256": actual_sha,
            "source_bytes": actual_bytes,
        }
        key = (speaker, role)
        prior = existing.get(key)
        if prior is not None and any(
            prior.get(field) != record.get(field)
            for field in ("utterance_id", "transcript", "source_sha256", "source_bytes")
        ):
            raise DiagnosticError(f"prepared input is already pinned differently: {speaker}/{role}")
        if prior is None:
            staged.append(record)

        cohorts = ("natural", "all") if role in {"N1", "N2"} else ("all",)
        for cohort in cohorts:
            directory = args.out / "input" / cohort / speaker
            directory.mkdir(parents=True, exist_ok=True)
            wav_path = directory / f"{stem}.wav"
            lab_path = directory / f"{stem}.lab"
            if wav_path.exists():
                if sha256_file(wav_path) != actual_sha:
                    raise DiagnosticError(f"staged audio differs from source pin: {wav_path}")
            else:
                shutil.copyfile(source, wav_path)
            lab_text = str(row["transcript"]) + "\n"
            if lab_path.exists():
                if lab_path.read_text(encoding="utf-8") != lab_text:
                    raise DiagnosticError(f"staged transcript differs from pinned transcript: {lab_path}")
            else:
                lab_path.write_text(lab_text, encoding="utf-8")

    staged.sort(key=lambda row: (row["speaker"], row["role"]))
    write_json(
        index_path,
        {
            "run_id": RUN_ID,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "asset_root": str(asset_root),
            "input_mode": "canonical_audio_manifest_all_roles" if canonical_payload is not None else "natural_raw_plus_optional_tts",
            "all_inputs_path": str(args.all_inputs.resolve()) if args.all_inputs is not None else None,
            "all_inputs_sha256": sha256_file(args.all_inputs.resolve()) if args.all_inputs is not None else None,
            "audio_manifest_path": canonical_payload.get("audio_manifest_path") if canonical_payload is not None else None,
            "audio_manifest_sha256": canonical_payload.get("audio_manifest_sha256") if canonical_payload is not None else None,
            "mfa_version_pin": EXPECTED_VERSION,
            "dictionary_sha256_pin": EXPECTED_SHA256["dictionary"],
            "acoustic_model_sha256_pin": EXPECTED_SHA256["acoustic_model"],
            "items": staged,
            "speakers": list(dict.fromkeys(
                row["speaker"] for row in staged if row["role"] in {"N1", "N2"}
            )),
        },
    )
    print(f"Prepared {len(staged)} pinned audio/transcript rows in {args.out}")


def _check_mfa_pins() -> dict[str, Any]:
    paths = {
        "mfa": MFA,
        "python": MFA_PYTHON,
        "dictionary": DICTIONARY,
        "acoustic_model": ACOUSTIC_MODEL,
    }
    actual: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise DiagnosticError(f"pinned {name} file is missing: {path}")
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise DiagnosticError(f"pinned {name} SHA256 differs: {path}")
        actual[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    version = subprocess.run(
        [
            str(MFA_PYTHON),
            "-B",
            "-c",
            "import importlib.metadata as metadata; print(metadata.version('montreal-forced-aligner'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = version.stdout.strip()
    if observed != EXPECTED_VERSION:
        raise DiagnosticError(f"MFA version mismatch: expected {EXPECTED_VERSION}, got {observed!r}")
    return {
        "mfa_version": observed,
        "mfa_version_stderr": version.stderr.strip(),
        "bindings": actual,
    }


def _mfa_command(input_dir: Path, output_dir: Path, temp_dir: Path) -> list[str]:
    return [
        str(MFA),
        "align",
        "--clean",
        "--overwrite",
        str(input_dir),
        "mandarin_china_mfa",
        "mandarin_mfa",
        str(output_dir),
        "--single_speaker",
        "--num_jobs",
        "1",
        "--temporary_directory",
        str(temp_dir),
    ]


def _selected_speakers(index: Mapping[str, Any], requested: Sequence[str] | None) -> list[str]:
    available = index.get("speakers")
    if not isinstance(available, list) or not all(isinstance(speaker, str) for speaker in available):
        available = list(dict.fromkeys(
            row["speaker"] for row in index.get("items", []) if row.get("role") in {"N1", "N2"}
        ))
    if not available:
        raise DiagnosticError("input manifest has no natural-pair speakers")
    selected = list(requested) if requested else list(available)
    if len(set(selected)) != len(selected):
        raise DiagnosticError("speaker selection contains duplicates")
    missing = [speaker for speaker in selected if speaker not in available]
    if missing:
        raise DiagnosticError(f"requested speakers are absent from input manifest: {missing}")
    return selected


def align(args: argparse.Namespace) -> None:
    index = read_json(_index_path(args.out))
    speakers = _selected_speakers(index, args.speakers)
    selected = [row for row in index.get("items", []) if row["role"] in ({"N1", "N2"} if args.cohort == "natural" else {"N1", "N2", "T1", "T2"})]
    if not selected:
        raise DiagnosticError("no selected MFA inputs")
    selected = [row for row in selected if row["speaker"] in speakers]
    if args.cohort == "all":
        for speaker in speakers:
            roles = {row["role"] for row in selected if row["speaker"] == speaker}
            if not roles >= {"N1", "N2", "T1", "T2"}:
                raise DiagnosticError(
                    f"all cohort requires N1/N2/T1/T2 for {speaker}; found {sorted(roles)}"
                )
    pins = _check_mfa_pins()
    output_tag = _safe_component(args.output_tag or args.cohort, "output tag")
    output_root = args.out / "textgrids" / output_tag
    if output_root.exists() and any(output_root.iterdir()):
        raise DiagnosticError(f"output already contains files; choose another --output-tag: {output_root}")
    temp_root = args.out / "temporary" / output_tag
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    speaker_results: list[dict[str, Any]] = []

    for speaker in speakers:
        speaker_rows = [row for row in selected if row["speaker"] == speaker]
        expected_roles = {row["role"] for row in speaker_rows}
        required = {"N1", "N2"}
        if not expected_roles >= required:
            raise DiagnosticError(f"{speaker} is missing N1/N2 inputs")
        input_dir = args.out / "input" / args.cohort / speaker
        if not input_dir.is_dir():
            raise DiagnosticError(f"MFA input directory is missing: {input_dir}")
        output_dir = output_root / speaker
        temp_dir = temp_root / speaker
        if output_dir.exists() and any(output_dir.iterdir()):
            raise DiagnosticError(f"speaker output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        command = _mfa_command(input_dir, output_dir, temp_dir)
        log_path = args.out / "logs" / output_tag / f"{speaker}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        environment = dict(os.environ)
        environment["PATH"] = str(MFA.parent) + os.pathsep + environment.get("PATH", "")
        with log_path.open("wb") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outputs = []
        for row in speaker_rows:
            grid = output_dir / f"{row['input_stem']}.TextGrid"
            if grid.is_file():
                outputs.append(
                    {"path": str(grid), "bytes": grid.stat().st_size, "sha256": sha256_file(grid)}
                )
        result_row = {
            "speaker": speaker,
            "command": command,
            "cwd": str(REPO_ROOT),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "temporary_directory": str(temp_dir),
            "log": str(log_path),
            "log_sha256": sha256_file(log_path),
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "outputs": outputs,
        }
        speaker_results.append(result_row)
        write_json(
            args.out / f"mfa_run_{output_tag}.json",
            {
                "status": "RUNNING" if result.returncode == 0 else "FAILED",
                "run_id": RUN_ID,
                "cohort": args.cohort,
                "speakers": speakers,
                "output_tag": output_tag,
                "cwd": str(REPO_ROOT),
                "mfa_pins": pins,
                "input_manifest_sha256": sha256_file(_index_path(args.out)),
                "speaker_results": speaker_results,
            },
        )
        if result.returncode != 0:
            raise DiagnosticError(f"MFA failed for {speaker} with code {result.returncode}; see {log_path}")
        if len(outputs) != len(speaker_rows):
            raise DiagnosticError(f"MFA did not produce all TextGrids for {speaker}")

    write_json(
        args.out / f"mfa_run_{output_tag}.json",
        {
            "status": "PASS",
            "run_id": RUN_ID,
            "cohort": args.cohort,
            "speakers": speakers,
            "output_tag": output_tag,
            "cwd": str(REPO_ROOT),
            "mfa_pins": pins,
            "input_manifest_sha256": sha256_file(_index_path(args.out)),
            "speaker_results": speaker_results,
        },
    )
    print(f"MFA completed; original TextGrids are in {output_root}")


def parse_textgrid(path: Path) -> dict[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    item_starts = list(re.finditer(r"(?m)^\s*item\s*\[\d+\]:", text))
    tiers: dict[str, list[dict[str, Any]]] = {}
    for position, match in enumerate(item_starts):
        end = item_starts[position + 1].start() if position + 1 < len(item_starts) else len(text)
        block = text[match.end() : end]
        cls = re.search(r'(?m)^\s*class\s*=\s*"([^"]+)"', block)
        name = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', block)
        if not cls or cls.group(1) != "IntervalTier" or not name:
            continue
        intervals: list[dict[str, Any]] = []
        starts = list(re.finditer(r"(?m)^\s*intervals\s*\[\d+\]:", block))
        for i, interval_match in enumerate(starts):
            interval_end = starts[i + 1].start() if i + 1 < len(starts) else len(block)
            interval_block = block[interval_match.end() : interval_end]
            xmin = re.search(r"(?m)^\s*xmin\s*=\s*([^\r\n]+)", interval_block)
            xmax = re.search(r"(?m)^\s*xmax\s*=\s*([^\r\n]+)", interval_block)
            label = re.search(r'(?m)^\s*text\s*=\s*"(.*)"\s*$', interval_block)
            if not (xmin and xmax and label):
                raise DiagnosticError(f"malformed interval in {path}")
            start = float(xmin.group(1).strip())
            end_time = float(xmax.group(1).strip())
            if end_time < start:
                raise DiagnosticError(f"negative interval duration in {path}")
            intervals.append(
                {
                    "index": i + 1,
                    "start_s": start,
                    "end_s": end_time,
                    "duration_s": end_time - start,
                    "label": label.group(1).replace('""', '"'),
                }
            )
        tiers[name.group(1)] = intervals
    if "phones" not in tiers:
        raise DiagnosticError(f"TextGrid has no phones tier: {path}")
    return tiers


def _phone_intervals(
    tiers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], float | None, float, list[str], int, float]:
    raw = list(tiers["phones"])
    full_labels = [str(item["label"]) for item in raw]
    recognized = [
        dict(item)
        for item in raw
        if str(item["label"]).casefold() not in ACTUAL_SILENCE_LABELS | UNKNOWN_LABELS
    ]
    non_silence = [
        item for item in raw
        if str(item["label"]).casefold() not in ACTUAL_SILENCE_LABELS
    ]
    unknown = [item for item in raw if str(item["label"]).casefold() in UNKNOWN_LABELS]
    onset = min((float(item["start_s"]) for item in recognized), default=None)
    coverage = len(recognized) / len(non_silence) if non_silence else 0.0
    unknown_duration = sum(float(item["duration_s"]) for item in unknown)
    return recognized, onset, coverage, full_labels, len(unknown), unknown_duration


def _lcs_pairs(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    n, m = len(left), len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if left[i]["label"] == right[j]["label"]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if left[i]["label"] == right[j]["label"] and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean_signed_s": None, "median_abs_s": None, "mae_s": None, "rmse_s": None}
    return {
        "n": len(values),
        "mean_signed_s": statistics.mean(values),
        "median_abs_s": statistics.median(abs(value) for value in values),
        "mae_s": statistics.mean(abs(value) for value in values),
        "rmse_s": math.sqrt(statistics.mean(value * value for value in values)),
    }


def _phone_support_coverage(
    phones: Sequence[Mapping[str, Any]], start_s: float, end_s: float
) -> dict[str, Any]:
    clipped: list[tuple[float, float]] = []
    for phone in phones:
        left = max(start_s, float(phone["start_s"]))
        right = min(end_s, float(phone["end_s"]))
        if right > left:
            clipped.append((left, right))
    clipped.sort()
    union_seconds = 0.0
    if clipped:
        union_start, union_end = clipped[0]
        for left, right in clipped[1:]:
            if left <= union_end:
                union_end = max(union_end, right)
            else:
                union_seconds += union_end - union_start
                union_start, union_end = left, right
        union_seconds += union_end - union_start
    duration = end_s - start_s
    return {
        "query_start_s_inclusive": start_s,
        "query_end_s_exclusive": end_s,
        "query_duration_s": duration,
        "recognized_phone_intervals_intersecting": len(clipped),
        "recognized_phone_union_duration_s": union_seconds,
        "recognized_phone_fraction_of_query_interval": union_seconds / duration if duration > 0 else None,
        "interpretation": "MFA recognized-phone intervals overlapping the absolute scoring-clock interval; forced alignment is an estimate, not speech ground truth",
    }


def summarize(args: argparse.Namespace) -> None:
    index = read_json(_index_path(args.out))
    speakers = _selected_speakers(index, args.speakers)
    selected = [row for row in index.get("items", []) if row["role"] in ({"N1", "N2"} if args.cohort == "natural" else {"N1", "N2", "T1", "T2"})]
    selected = [row for row in selected if row["speaker"] in speakers]
    output_root = args.out / "textgrids" / args.output_tag
    comparisons: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    support_by_speaker: dict[str, dict[str, Any]] = {}
    support_coverage_by_speaker: dict[str, dict[str, Any]] = {}
    if args.matrix_support is not None:
        matrix = read_json(args.matrix_support.resolve(strict=True))
        matrix_rows = matrix.get("speaker_results")
        if not isinstance(matrix_rows, list):
            raise DiagnosticError("matrix support JSON must contain speaker_results")
        for matrix_row in matrix_rows:
            if not isinstance(matrix_row, dict):
                raise DiagnosticError("matrix speaker support records must be objects")
            speaker = str(matrix_row.get("speaker", ""))
            support = matrix_row.get("support")
            if not speaker or not isinstance(support, dict) or speaker in support_by_speaker:
                raise DiagnosticError("matrix support has missing/duplicate speaker or support object")
            start_s = support.get("query_anchor_start_s")
            end_s = support.get("query_anchor_end_s_exclusive")
            if not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)) or end_s <= start_s:
                raise DiagnosticError(f"matrix support has invalid absolute query interval: {speaker}")
            support_by_speaker[speaker] = support
        if any(speaker not in support_by_speaker for speaker in speakers):
            missing = [speaker for speaker in speakers if speaker not in support_by_speaker]
            raise DiagnosticError(f"matrix support is missing selected speakers: {missing}")

    for speaker in speakers:
        rows = [row for row in selected if row["speaker"] == speaker]
        parsed: dict[str, dict[str, Any]] = {}
        for row in rows:
            grid = output_root / speaker / f"{row['input_stem']}.TextGrid"
            if not grid.is_file():
                raise DiagnosticError(f"original TextGrid is missing: {grid}")
            tiers = parse_textgrid(grid)
            phones, onset, coverage, full_labels, unknown_count, unknown_duration = _phone_intervals(tiers)
            if onset is None:
                raise DiagnosticError(f"no recognized phones in {grid}")
            parsed[row["role"]] = {
                "row": row,
                "grid": grid,
                "grid_sha256": sha256_file(grid),
                "tiers": tiers,
                "phones": phones,
                "onset": onset,
                "coverage": coverage,
                "full_phone_labels": full_labels,
                "unknown_phone_count": unknown_count,
                "unknown_phone_duration_s": unknown_duration,
            }
        if args.matrix_support is not None:
            support = support_by_speaker[speaker]
            start_s = float(support["query_anchor_start_s"])
            end_s = float(support["query_anchor_end_s_exclusive"])
            support_coverage_by_speaker[speaker] = {
                "matrix_support": {
                    key: support.get(key)
                    for key in (
                        "query_start_window_inclusive", "query_end_window_exclusive",
                        "query_window_count", "lag_radius_frames", "query_anchor_start_s",
                        "query_anchor_end_s_exclusive", "visual_window_convention",
                        "audio_window_convention", "source_timeline",
                    )
                },
                "recognized_phone_coverage_by_role": {
                    role: _phone_support_coverage(parsed[role]["phones"], start_s, end_s)
                    for role in sorted(parsed)
                },
            }
        roles = sorted(parsed, key=lambda role: (role not in {"N1", "N2"}, role))
        role_pairs = [(roles[i], roles[j]) for i in range(len(roles)) for j in range(i + 1, len(roles))]
        for role_a, role_b in role_pairs:
            a, b = parsed[role_a], parsed[role_b]
            left, right = a["phones"], b["phones"]
            matches = _lcs_pairs(left, right)
            rows_for_pair: list[dict[str, Any]] = []
            for match_index, (i, j) in enumerate(matches, 1):
                p, q = left[i], right[j]
                a_start = p["start_s"] - a["onset"]
                a_end = p["end_s"] - a["onset"]
                b_start = q["start_s"] - b["onset"]
                b_end = q["end_s"] - b["onset"]
                row = {
                    "speaker": speaker,
                    "transcript": a["row"]["transcript"],
                    "role_a": role_a,
                    "role_b": role_b,
                    "match_index": match_index,
                    "phone_label": p["label"],
                    "phone_index_a": p["index"],
                    "phone_index_b": q["index"],
                    "a_start_relative_s": a_start,
                    "b_start_relative_s": b_start,
                    "start_delta_after_onset_s": b_start - a_start,
                    "a_end_relative_s": a_end,
                    "b_end_relative_s": b_end,
                    "end_delta_after_onset_s": b_end - a_end,
                    "a_center_relative_s": (a_start + a_end) / 2,
                    "b_center_relative_s": (b_start + b_end) / 2,
                    "center_delta_after_onset_s": ((b_start + b_end) - (a_start + a_end)) / 2,
                    "duration_a_s": p["duration_s"],
                    "duration_b_s": q["duration_s"],
                    "duration_delta_s": q["duration_s"] - p["duration_s"],
                }
                detailed.append(row)
                rows_for_pair.append(row)
            labels_a = [phone["label"] for phone in left]
            labels_b = [phone["label"] for phone in right]
            max_length = max(len(left), len(right))
            comparisons.append(
                {
                    "speaker": speaker,
                    "transcript": a["row"]["transcript"],
                    "role_a": role_a,
                    "role_b": role_b,
                    "textgrid_a": str(a["grid"]),
                    "textgrid_a_sha256": a["grid_sha256"],
                    "textgrid_b": str(b["grid"]),
                    "textgrid_b_sha256": b["grid_sha256"],
                    "phone_sequence_a": labels_a,
                    "phone_sequence_b": labels_b,
                    "full_phone_labels_a": a["full_phone_labels"],
                    "full_phone_labels_b": b["full_phone_labels"],
                    "phone_sequence_exact_match": labels_a == labels_b,
                    "phone_lcs_matched": len(matches),
                    "phone_pair_coverage": len(matches) / max_length if max_length else 0.0,
                    "phone_coverage_a": a["coverage"],
                    "phone_coverage_b": b["coverage"],
                    "unknown_phone_count_a": a["unknown_phone_count"],
                    "unknown_phone_count_b": b["unknown_phone_count"],
                    "unknown_phone_duration_a_s": a["unknown_phone_duration_s"],
                    "unknown_phone_duration_b_s": b["unknown_phone_duration_s"],
                    "first_recognized_phone_onset_a_s": a["onset"],
                    "first_recognized_phone_onset_b_s": b["onset"],
                    "global_onset_delta_b_minus_a_s": b["onset"] - a["onset"],
                    "relative_start_error": _stats(row["start_delta_after_onset_s"] for row in rows_for_pair),
                    "relative_end_error": _stats(row["end_delta_after_onset_s"] for row in rows_for_pair),
                    "relative_center_error": _stats(row["center_delta_after_onset_s"] for row in rows_for_pair),
                    "duration_difference": _stats(row["duration_delta_s"] for row in rows_for_pair),
                }
            )

    write_json(args.out / f"timing_summary_{args.output_tag}.json", {
        "run_id": RUN_ID,
        "cohort": args.cohort,
        "output_tag": args.output_tag,
        "speakers": speakers,
        "input_manifest_sha256": sha256_file(_index_path(args.out)),
        "matrix_support_path": str(args.matrix_support.resolve()) if args.matrix_support is not None else None,
        "matrix_support_sha256": sha256_file(args.matrix_support.resolve()) if args.matrix_support is not None else None,
        "interpretation": "Audio-only forced-alignment diagnostics; MFA boundaries are estimates, not human boundary ground truth.",
        "pairs": comparisons,
        "scoring_interval_phone_coverage": support_coverage_by_speaker,
    })
    detailed_path = args.out / f"phone_timing_comparisons_{args.output_tag}.jsonl"
    with detailed_path.open("w", encoding="utf-8") as stream:
        for row in detailed:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {len(comparisons)} pairwise comparisons and {len(detailed)} matched-phone rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare", help="verify and copy pinned audio and transcripts")
    stage.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    stage.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    stage.add_argument("--extra-inputs", type=Path, help="optional JSON containing T1/T2 audio rows")
    stage.add_argument("--all-inputs", type=Path, help="all-role JSON built from the canonical audio manifest")
    stage.set_defaults(func=prepare)
    canonical = sub.add_parser("build-canonical-inputs", help="pin the final canonical 8x4 audio cohort")
    canonical.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    canonical.add_argument("--audio-manifest", type=Path, default=DEFAULT_AUDIO_MANIFEST)
    canonical.add_argument("--output", type=Path, default=CANONICAL_OUT_DIR / "all_role_inputs.json")
    canonical.set_defaults(func=build_canonical_inputs)
    for name, function in (("align", align), ("summarize", summarize)):
        item = sub.add_parser(name)
        item.add_argument("--cohort", choices=("natural", "all"), default="natural")
        item.add_argument("--speakers", nargs="+", help="optional subset from the staged manifest speakers")
        item.add_argument("--output-tag", default=None)
        if name == "summarize":
            item.add_argument("--matrix-support", type=Path, help="matrix.json containing per-speaker absolute query intervals")
        item.set_defaults(func=function)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out is None:
        args.out = CANONICAL_OUT_DIR if args.command == "prepare" and args.all_inputs is not None else OUT_DIR
    args.out = args.out.resolve()
    if hasattr(args, "output_tag") and args.output_tag is None:
        args.output_tag = args.cohort
    try:
        args.func(args)
    except (DiagnosticError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
