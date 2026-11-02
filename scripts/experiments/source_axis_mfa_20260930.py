#!/usr/bin/env python3
"""Prepare and run frozen 3.4.1 Mandarin MFA for source-axis paired audio.

The calibration protocol is written before alignment. Evaluation alignment is
blocked until the CAL-only support/qualification manifest is frozen. TextGrid
quality failures affect only phone-endpoint availability, not paired acoustic
records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs/recovered_20260930/local/qwen_source_axis/hf_selected"
DEFAULT_INPUT = DEFAULT_ROOT / "input_manifest.json"
DEFAULT_PAIR = DEFAULT_ROOT / "tts_outputs/tts_pair_manifest.json"
DEFAULT_TTS_META = DEFAULT_ROOT / "tts_outputs/tts_meta.json"
DEFAULT_OUT = DEFAULT_ROOT / "mfa_20260930"
VISUAL_PROTOCOL_PATH = REPO_ROOT / "scripts/experiments/tts_mechanism_visual_20260930.py"
SILENCE = {"", "sil", "sp", "spn", "<sil>"}
UNKNOWN = {"spn", "unk", "unknown", "oov", "<unk>", "<unknown>"}

# Frozen before looking at any evaluation TextGrid or endpoint. Tolerance is one
# 25-fps video frame; class support is based only on CAL alignments.
PROTOCOL = {
    "schema_version": 1,
    "protocol_id": "source-axis-mfa-3.4.1-cal-gated-20260930-v2",
    "mfa_version_required": "3.4.1",
    "dictionary_name": "mandarin_china_mfa",
    "acoustic_model_name": "mandarin_mfa",
    "lab_normalization": "scripts/run_aishell1_n25_mfa.py::clean_lab_text; Chinese character segmentation, punctuation removal, Mandarin number expansion",
    "mfa_command_flags": ["align", "<corpus_dir>", "<dictionary_path>", "<acoustic_model_path>", "<output_dir>", "--single_speaker", "--clean", "--num_jobs", "2"],
    "cpu_only_environment": {"CUDA_VISIBLE_DEVICES": "", "num_jobs": 2},
    "textgrid_clock": {"boundary_tolerance_s": 0.04, "overlap_tolerance_s": 0.001, "interval_min_duration_s": 0.000001},
    "sample_qualification": {
        "both_conditions_required": True,
        "at_least_one_speech_phone_each_side": True,
        "reject_unknown_or_oov_phone_each_side": True,
        "reject_nonpositive_phone_interval": True,
        "reject_phone_overlap_over_tolerance": True,
        "reject_phone_interval_outside_audio_over_tolerance": True,
        "first_phone_start_within_audio_tolerance": True,
        "last_phone_end_within_audio_tolerance": True,
    },
    "visual_phone_endpoint": {
        "classes": ["bilabial", "open_vowel"],
        "source": "scripts/experiments/tts_mechanism_visual_20260930.py::normalize_phone/phone_class",
        "count_gate": "deferred to visual CAL baseline, after frozen CAL latency and common valid MediaPipe frames; MFA alignment qualification records counts only",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in payload.items() if k != "manifest_sha256"})).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw: str, repo_root: Path, override_root: Path | None = None, expected_hash: str | None = None) -> Path:
    # Manifests may contain Windows drive-letter paths while this adapter runs
    # on Linux; normalize separators before using the basename fallback.
    given = Path(str(raw).replace("\\", "/")).expanduser()
    candidates: list[Path] = []
    if given.is_absolute():
        candidates.append(given)
    else:
        candidates.append((repo_root / given).resolve())
        candidates.append((Path.cwd() / given).resolve())
    if override_root:
        root = override_root.resolve()
        candidates.extend([root / given.name, *(p for p in root.rglob(given.name) if p.is_file())])
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and (expected_hash is None or sha256_file(path) == expected_hash):
            return path.resolve()
    raise FileNotFoundError(f"missing input or hash mismatch: {given.name}")


def load_textgrid_tools() -> tuple[Any, Path]:
    module_path = REPO_ROOT / "scripts/run_aishell1_n25_mfa.py"
    spec = importlib.util.spec_from_file_location("mfa_normalizer_source", module_path)
    if not spec or not spec.loader:
        raise RuntimeError("could not load pinned project MFA normalizer/parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_path


def load_visual_phone_tools() -> tuple[Any, Path]:
    spec = importlib.util.spec_from_file_location("source_axis_visual_phone_rules", VISUAL_PROTOCOL_PATH)
    if not spec or not spec.loader:
        raise RuntimeError("could not load frozen visual phone-class rules")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, VISUAL_PROTOCOL_PATH


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source, pairs, tts_meta = map(read_json, (args.input_manifest.resolve(), args.pair_manifest.resolve(), args.tts_meta.resolve()))
    if source.get("manifest_sha256") and source["manifest_sha256"] != fingerprint(source):
        raise ValueError("input manifest fingerprint mismatch")
    if pairs.get("manifest_sha256") and pairs["manifest_sha256"] != fingerprint(pairs):
        raise ValueError("pair manifest fingerprint mismatch")
    if not tts_meta.get("complete") or tts_meta.get("samples_ok") != 100 or tts_meta.get("samples_failed") != 0:
        raise ValueError("TTS metadata must report a complete 100/100 run")
    source_rows = source.get("records", [])
    pair_by_id = {str(r["id"]): r for r in pairs.get("records", [])}
    tts_by_id = tts_meta.get("results", {})
    if len(source_rows) != 100 or set(pair_by_id) != {str(r["id"]) for r in source_rows} or set(tts_by_id) != set(pair_by_id):
        raise ValueError("frozen source, pair, and TTS inputs must cover exactly 100 records")
    pair_sha = pairs.get("source_manifest", {}).get("raw_sha256")
    if pair_sha and pair_sha != sha256_file(args.input_manifest.resolve()):
        raise ValueError("pair manifest source hash differs from selected input manifest")
    frozen_fingerprint = pairs.get("run_frozen_input_manifest", {}).get("embedded_fingerprint")
    if frozen_fingerprint != tts_meta.get("input_manifest_sha256"):
        raise ValueError("TTS result is not bound to the frozen input identity")
    for row in source_rows:
        sample_id = str(row["id"])
        pair = pair_by_id[sample_id]
        result = tts_by_id[sample_id]
        identity = pair.get("input_identity", {})
        expected = {"id": row["id"], "speaker": row["speaker"], "text": row["text"], "split": row["split"], "ref_sha256": row["ref_sha256"]}
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError(f"input identity binding mismatch for {sample_id}")
        if result.get("status") != "ok" or result.get("canonical_audio_sha256") != pair.get("qwen_canonical_16k", {}).get("sha256"):
            raise ValueError(f"canonical TTS binding mismatch for {sample_id}")
    return source, pairs, tts_meta, source_rows


def make_protocol_manifest(outdir: Path, args: argparse.Namespace, source: dict[str, Any], pairs: dict[str, Any], tts_meta: dict[str, Any], normalizer_path: Path) -> dict[str, Any]:
    protocol = dict(PROTOCOL)
    visual_rules, visual_path = load_visual_phone_tools()
    protocol["visual_phone_endpoint_rules"] = {
        "source_path": visual_path.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": sha256_file(visual_path),
        "class_function": "phone_class",
        "normalizer_function": "normalize_phone",
        "bilabial_bases": sorted(visual_rules.BILABIAL_BASES),
        "open_vowel_bases": sorted(visual_rules.OPEN_VOWEL_BASES),
        "stripped_phone_marks": list(visual_rules.STRIPPED_PHONE_MARKS),
        "required_calibration_clip_coverage": visual_rules.MIN_CALIBRATION_CLIP_COVERAGE,
        "count_freeze_procedure": "The visual protocol counts phone tokens that overlap at least one common valid frame after its CAL-frozen latency, then selects the largest per-clip minimum count reached by at least 80% of CAL clips. This MFA adapter does not freeze that count without video tracks.",
    }
    protocol_hash = hashlib.sha256(canonical_json(protocol)).hexdigest()
    protocol_path = outdir / "alignment_protocol_frozen.json"
    payload = {
        "schema_version": 1,
        "manifest_type": "source_axis_mfa_protocol_freeze",
        "status": "FROZEN_BEFORE_ALIGNMENT",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "inputs": {
            "source_manifest_sha256": sha256_file(args.input_manifest.resolve()),
            "source_manifest_fingerprint": source.get("manifest_sha256"),
            "pair_manifest_sha256": sha256_file(args.pair_manifest.resolve()),
            "pair_manifest_fingerprint": pairs.get("manifest_sha256"),
            "tts_meta_sha256": sha256_file(args.tts_meta.resolve()),
            "tts_meta_input_fingerprint": tts_meta.get("input_manifest_sha256"),
        },
        "normalizer_parser": {"path": normalizer_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(normalizer_path)},
        "visual_phone_rules": {"path": visual_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(visual_path)},
        "adapter_source": {"path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(Path(__file__).resolve())},
    }
    payload["manifest_sha256"] = fingerprint(payload)
    if protocol_path.exists():
        previous = read_json(protocol_path)
        if previous.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError("output directory has a different frozen MFA protocol")
    else:
        write_json(protocol_path, payload)
    return payload


def prepare(args: argparse.Namespace) -> int:
    source, pairs, tts_meta, rows = load_inputs(args)
    normalizer, normalizer_path = load_textgrid_tools()
    outdir = args.outdir.resolve()
    protocol_manifest = make_protocol_manifest(outdir, args, source, pairs, tts_meta, normalizer_path)
    pair_by_id = {str(x["id"]): x for x in pairs["records"]}
    meta_by_id = tts_meta["results"]
    producer_path = outdir / "mfa_producer_manifest.json"
    if producer_path.exists():
        old = read_json(producer_path)
        if old.get("protocol_manifest_sha256") != protocol_manifest["manifest_sha256"]:
            raise ValueError("existing producer output is bound to another protocol")
    else:
        old = {}
    records: dict[str, Any] = old.get("records", {})
    for row in rows:
        sample_id = str(row["id"])
        speaker = str(row["speaker"])
        split = str(row["split"])
        pair, tts = pair_by_id[sample_id], meta_by_id[sample_id]
        files: dict[str, Any] = {}
        for condition, source_raw, expected_hash in (
            ("natural", str(row["ref_path"]), str(row["ref_sha256"])),
            ("tts", str(tts["canonical_16k_audio"]), str(tts["canonical_audio_sha256"])),
        ):
            override = args.natural_root if condition == "natural" else args.canonical_root
            origin = resolve_path(source_raw, args.repo_root.resolve(), override, expected_hash)
            staged_dir = outdir / "corpus" / split / speaker
            staged_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{sample_id}__{condition}"
            wav_out = staged_dir / f"{stem}.wav"
            lab_out = staged_dir / f"{stem}.lab"
            if wav_out.exists() and sha256_file(wav_out) != expected_hash:
                raise ValueError(f"staged audio hash differs from frozen source: {stem}")
            if not wav_out.exists():
                shutil.copyfile(origin, wav_out)
            label = normalizer.clean_lab_text(str(row["text"]))
            if not label:
                raise ValueError(f"empty normalized transcript: {sample_id}")
            expected_lab = label + "\n"
            if lab_out.exists() and lab_out.read_text(encoding="utf-8") != expected_lab:
                raise ValueError(f"normalized transcript changed: {stem}")
            if not lab_out.exists():
                lab_out.write_text(expected_lab, encoding="utf-8")
            files[condition] = {
                "source_path": str(origin),
                "source_audio_sha256": expected_hash,
                "staged_audio_path": str(wav_out.resolve()),
                "staged_audio_sha256": sha256_file(wav_out),
                "transcript": str(row["text"]),
                "transcript_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                "normalized_lab_text": label,
                "lab_path": str(lab_out.resolve()),
                "lab_sha256": sha256_file(lab_out),
                "expected_textgrid_path": str((outdir / "textgrids" / split / speaker / f"{stem}.TextGrid").resolve()),
            }
        records[sample_id] = {
            "id": sample_id, "speaker": speaker, "split": split,
            "source_identity_sha256": pair["input_identity_sha256"],
            "natural_raw_audio_sha256": files["natural"]["source_audio_sha256"],
            "canonical_tts_16k_audio_sha256": files["tts"]["source_audio_sha256"],
            "natural_transcript_sha256": files["natural"]["transcript_sha256"],
            "tts_transcript_sha256": files["tts"]["transcript_sha256"],
            "conditions": files,
        }
    payload = {
        "schema_version": 1,
        "manifest_type": "source_axis_mfa_producer_manifest",
        "status": "INPUTS_PREPARED",
        "mfa_version_required": PROTOCOL["mfa_version_required"],
        "dictionary_name": PROTOCOL["dictionary_name"],
        "acoustic_model_name": PROTOCOL["acoustic_model_name"],
        "protocol_manifest_path": str((outdir / "alignment_protocol_frozen.json").resolve()),
        "protocol_manifest_sha256": protocol_manifest["manifest_sha256"],
        "source_manifest_path": str(args.input_manifest.resolve()),
        "source_manifest_file_sha256": sha256_file(args.input_manifest.resolve()),
        "source_manifest_fingerprint": source.get("manifest_sha256"),
        "pair_manifest_file_sha256": sha256_file(args.pair_manifest.resolve()),
        "pair_manifest_fingerprint": pairs.get("manifest_sha256"),
        "tts_meta_file_sha256": sha256_file(args.tts_meta.resolve()),
        "tts_meta_input_fingerprint": tts_meta.get("input_manifest_sha256"),
        "normalizer_parser_path": normalizer_path.relative_to(REPO_ROOT).as_posix(),
        "normalizer_parser_sha256": sha256_file(normalizer_path),
        "adapter_source_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        "records_expected": 100,
        "conditions_expected": 200,
        "records": records,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    write_json(producer_path, payload)
    print(json.dumps({"records": len(records), "conditions": sum(len(v["conditions"]) for v in records.values()), "protocol_sha256": protocol_manifest["protocol_sha256"], "producer_manifest_sha256": payload["manifest_sha256"], "out": str(producer_path)}, separators=(",", ":")))
    return 0


def mfa_version(mfa_bin: Path) -> str:
    result = subprocess.run([str(mfa_bin), "version"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def run_alignment(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    producer_path = outdir / "mfa_producer_manifest.json"
    producer = read_json(producer_path)
    protocol_manifest = read_json(Path(producer["protocol_manifest_path"]))
    if protocol_manifest.get("manifest_sha256") != producer.get("protocol_manifest_sha256"):
        raise ValueError("alignment protocol manifest hash mismatch")
    if sha256_file(Path(__file__).resolve()) != protocol_manifest["adapter_source"]["sha256"]:
        raise ValueError("MFA adapter source changed after protocol freeze")
    split = args.split
    if split == "evaluation":
        gate = outdir / "calibration_qualification_frozen.json"
        if not gate.is_file():
            raise ValueError("evaluation alignment is blocked until CAL qualification/support is frozen")
        gate_payload = read_json(gate)
        if gate_payload.get("status") != "FROZEN" or gate_payload.get("protocol_manifest_sha256") != protocol_manifest.get("manifest_sha256"):
            raise ValueError("CAL gate is missing, not frozen, or bound to a different protocol")
    mfa_bin = args.mfa_bin.resolve()
    dictionary = args.dictionary.resolve()
    acoustic = args.acoustic.resolve()
    if not mfa_bin.is_file() or not dictionary.exists() or not acoustic.exists():
        raise FileNotFoundError("MFA executable, dictionary, or acoustic model missing")
    version_line = mfa_version(mfa_bin)
    if PROTOCOL["mfa_version_required"] not in version_line:
        raise ValueError(f"expected MFA 3.4.1, got {version_line}")
    model_info = {
        "mfa_version": version_line,
        "mfa_binary_path": str(mfa_bin), "mfa_binary_sha256": sha256_file(mfa_bin),
        "dictionary_name": PROTOCOL["dictionary_name"], "dictionary_path": str(dictionary), "dictionary_sha256": hash_tree(dictionary),
        "acoustic_model_name": PROTOCOL["acoustic_model_name"], "acoustic_model_path": str(acoustic), "acoustic_model_sha256": hash_tree(acoustic),
        "mfa_root_dir": str(args.mfa_root_dir.resolve()),
        "num_jobs": 2,
        "single_speaker": True,
        "clean": True,
        "environment": {"CUDA_VISIBLE_DEVICES": "", "TMPDIR": str((outdir / "tmp").resolve())},
    }
    model_info["configuration_sha256"] = hashlib.sha256(canonical_json(model_info)).hexdigest()
    old_config = producer.get("mfa_configuration")
    if old_config and old_config.get("configuration_sha256") != model_info["configuration_sha256"]:
        raise ValueError("MFA environment/model/config differs from prior alignment run")
    producer["mfa_configuration"] = model_info
    producer["alignment_command_template"] = [str(mfa_bin), "align", "<speaker_corpus>", str(dictionary), str(acoustic), "<textgrid_output>", "--single_speaker", "--clean", "--num_jobs", "2"]
    tmpdir = outdir / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    selected = [r for r in producer["records"].values() if r["split"] == split]
    speakers = sorted({r["speaker"] for r in selected})
    command_records = producer.setdefault("alignments", {})
    env = dict(os.environ)
    env.update({"MFA_ROOT_DIR": model_info["mfa_root_dir"], "TMPDIR": str(tmpdir), "CUDA_VISIBLE_DEVICES": ""})
    for index, speaker in enumerate(speakers, 1):
        input_dir = outdir / "corpus" / split / speaker
        output_dir = outdir / "textgrids" / split / speaker
        log_path = outdir / "logs" / split / f"{speaker}.log"
        output_dir.mkdir(parents=True, exist_ok=True); log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(mfa_bin), "align", str(input_dir), str(dictionary), str(acoustic), str(output_dir), "--single_speaker", "--clean", "--num_jobs", "2"]
        existing_rows = [r for r in selected if r["speaker"] == speaker]
        grids_ok = all(Path(r["conditions"][condition]["expected_textgrid_path"]).is_file() for r in existing_rows for condition in ("natural", "tts"))
        if grids_ok and not args.force:
            rc = 0
            cached = True
            stdout = "reused existing TextGrids"
            stderr = ""
        else:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout, env=env)
            rc, cached, stdout, stderr = result.returncode, False, result.stdout, result.stderr
        log_path.write_text(json.dumps({"command": command, "returncode": rc, "cached": cached, "stdout": stdout, "stderr": stderr}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files = []
        for rec in existing_rows:
            for condition in ("natural", "tts"):
                grid = Path(rec["conditions"][condition]["expected_textgrid_path"])
                files.append({
                    "id": rec["id"], "speaker": speaker, "split": split, "condition": condition,
                    "input_audio_sha256": rec["conditions"][condition]["source_audio_sha256"],
                    "staged_audio_sha256": rec["conditions"][condition]["staged_audio_sha256"],
                    "pair_natural_raw_audio_sha256": rec["natural_raw_audio_sha256"],
                    "pair_canonical_tts_16k_audio_sha256": rec["canonical_tts_16k_audio_sha256"],
                    "transcript_sha256": rec["conditions"][condition]["transcript_sha256"],
                    "textgrid_path": str(grid),
                    "textgrid_sha256": sha256_file(grid) if grid.is_file() else None,
                    "alignment_protocol_sha256": producer["protocol_manifest_sha256"],
                    "mfa_configuration_sha256": model_info["configuration_sha256"],
                    "mfa_version": model_info["mfa_version"],
                    "dictionary_name": model_info["dictionary_name"],
                    "dictionary_sha256": model_info["dictionary_sha256"],
                    "acoustic_model_name": model_info["acoustic_model_name"],
                    "acoustic_model_sha256": model_info["acoustic_model_sha256"],
                    "status": "produced" if grid.is_file() and rc == 0 else "missing_or_failed",
                })
        command_records[f"{split}/{speaker}"] = {
            "command": command,
            "command_sha256": hashlib.sha256(canonical_json(command)).hexdigest(),
            "returncode": rc, "cached": cached, "log_path": str(log_path.resolve()), "log_sha256": sha256_file(log_path),
            "textgrids": files,
        }
        producer["status"] = "CAL_ALIGNMENT_COMPLETE" if split == "calibration" else "EVALUATION_ALIGNMENT_COMPLETE"
        producer["last_alignment_split"] = split
        producer["manifest_sha256"] = fingerprint(producer)
        write_json(producer_path, producer)
        print(f"ALIGNED {split} speaker {index}/{len(speakers)} {speaker} rc={rc}", flush=True)
        if rc != 0:
            raise RuntimeError(f"MFA alignment failed for {split}/{speaker}; see safe local log {log_path}")
    return 0


def parse_grid(grid: Path, parser_module: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokens = parser_module.parse_phones(grid)
    raw = grid.read_text(encoding="utf-8")
    xmax_match = re.search(r"xmax\s*=\s*([\d.]+)", raw)
    xmax = float(xmax_match.group(1)) if xmax_match else None
    return tokens, {"textgrid_xmax_s": xmax}


def classify_phone(phone: str, visual_rules: Any) -> str | None:
    return visual_rules.phone_class(phone)


def check_condition_grid(grid: Path, duration_s: float, parser_module: Any, visual_rules: Any) -> dict[str, Any]:
    rules = PROTOCOL["sample_qualification"]
    timing = PROTOCOL["textgrid_clock"]
    errors: list[str] = []
    try:
        tokens, grid_meta = parse_grid(grid, parser_module)
    except Exception as exc:
        return {"qualified": False, "errors": [f"parse_error:{type(exc).__name__}"], "tokens": [], "textgrid_sha256": sha256_file(grid) if grid.is_file() else None}
    unknowns = [x for x in tokens if str(x["token"]).casefold() in UNKNOWN]
    speech = [x for x in tokens if not x["is_silence"] or str(x["token"]).casefold() in UNKNOWN]
    if not speech:
        errors.append("no_speech_phone")
    if unknowns and rules["reject_unknown_or_oov_phone_each_side"]:
        errors.append("unknown_or_oov_phone")
    for i, token in enumerate(tokens):
        if float(token["duration_s"]) < timing["interval_min_duration_s"]:
            errors.append("nonpositive_phone_duration"); break
        if float(token["start_s"]) < -timing["boundary_tolerance_s"] or float(token["end_s"]) > duration_s + timing["boundary_tolerance_s"]:
            errors.append("phone_outside_audio_clock"); break
        if i and float(tokens[i - 1]["end_s"]) - float(token["start_s"]) > timing["overlap_tolerance_s"]:
            errors.append("phone_overlap"); break
    if tokens:
        if float(tokens[0]["start_s"]) > timing["boundary_tolerance_s"]:
            errors.append("audio_start_not_covered")
        if float(tokens[-1]["end_s"]) < duration_s - timing["boundary_tolerance_s"]:
            errors.append("audio_end_not_covered")
    textgrid_xmax = grid_meta.get("textgrid_xmax_s")
    if textgrid_xmax is None:
        errors.append("textgrid_xmax_missing")
    elif abs(float(textgrid_xmax) - duration_s) > timing["boundary_tolerance_s"]:
        errors.append("textgrid_clock_mismatch")
    classes = Counter(
        category for x in speech
        if (category := classify_phone(str(x["token"]), visual_rules)) is not None
    )
    return {
        "qualified": not errors,
        "errors": sorted(set(errors)),
        "phone_count_including_silence": len(tokens),
        "speech_phone_count": len(speech),
        "unknown_oov_phone_count": len(unknowns),
        "open_vowel_phone_count": classes["open_vowel"],
        "bilabial_phone_count": classes["bilabial"],
        "phone_labels": [str(x["token"]) for x in tokens],
        "tokens": tokens,
        "grid_timing": grid_meta,
        "textgrid_sha256": sha256_file(grid),
    }


def qualify_cal(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    protocol_manifest = read_json(outdir / "alignment_protocol_frozen.json")
    producer_path = outdir / "mfa_producer_manifest.json"
    producer = read_json(producer_path)
    if producer.get("protocol_manifest_sha256") != protocol_manifest.get("manifest_sha256"):
        raise ValueError("MFA producer manifest has a different protocol binding")
    if sha256_file(Path(__file__).resolve()) != protocol_manifest["adapter_source"]["sha256"]:
        raise ValueError("MFA adapter source changed after protocol freeze")
    parser_module, normalizer_path = load_textgrid_tools()
    visual_rules, visual_path = load_visual_phone_tools()
    if sha256_file(normalizer_path) != producer["normalizer_parser_sha256"]:
        raise ValueError("normalizer/parser source changed since input preparation")
    if sha256_file(visual_path) != protocol_manifest["visual_phone_rules"]["sha256"]:
        raise ValueError("visual phone-class source changed since input preparation")
    cal = [r for r in producer["records"].values() if r["split"] == "calibration"]
    if len(cal) != 20:
        raise ValueError("calibration qualification requires exactly 20 frozen pairs")
    per_condition: dict[str, Any] = {}
    eligible_pairs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rec in sorted(cal, key=lambda x: x["id"]):
        sides: dict[str, Any] = {}
        for condition in ("natural", "tts"):
            meta = rec["conditions"][condition]
            try:
                import soundfile as sf
                pcm, sample_rate = sf.read(meta["staged_audio_path"], dtype="float32", always_2d=True)
                decoded_frames = int(pcm.shape[0])
                if decoded_frames <= 0 or pcm.shape[1] != 1 or int(sample_rate) != 16_000:
                    raise ValueError("staged audio must fully decode to nonempty mono 16 kHz PCM")
                duration_s = decoded_frames / int(sample_rate)
                grid = Path(meta["expected_textgrid_path"])
                if not grid.is_file():
                    raise FileNotFoundError(grid.name)
                parsed = check_condition_grid(grid, duration_s, parser_module, visual_rules)
                parsed.update({
                    "input_audio_sha256": meta["source_audio_sha256"],
                    "staged_audio_sha256": meta["staged_audio_sha256"],
                    "textgrid_path": str(grid.resolve()),
                    "sample_rate_hz": int(sample_rate),
                    "audio_decoded_frames": decoded_frames,
                    "audio_duration_s": round(duration_s, 9),
                })
                sides[condition] = parsed
            except Exception as exc:
                sides[condition] = {"qualified": False, "errors": [f"input_or_grid_error:{type(exc).__name__}"], "input_audio_sha256": meta["source_audio_sha256"], "textgrid_sha256": None}
        pair_ok = bool(sides["natural"].get("qualified") and sides["tts"].get("qualified"))
        item = {"id": rec["id"], "speaker": rec["speaker"], "pair_qualified": pair_ok, "natural": sides["natural"], "tts": sides["tts"]}
        per_condition[rec["id"]] = item
        if pair_ok:
            eligible_pairs.append(item)
        else:
            failures.append({"id": rec["id"], "speaker": rec["speaker"], "natural_errors": sides["natural"].get("errors", []), "tts_errors": sides["tts"].get("errors", [])})
    phone_coverage: dict[str, Any] = {}
    for klass in PROTOCOL["visual_phone_endpoint"]["classes"]:
        supporting = []
        for pair in eligible_pairs:
            if pair["natural"].get(f"{klass}_phone_count", 0) > 0 and pair["tts"].get(f"{klass}_phone_count", 0) > 0:
                supporting.append(pair)
        phone_coverage[klass] = {
            "cal_pair_ids": [x["id"] for x in supporting],
            "cal_speakers": sorted({x["speaker"] for x in supporting}),
            "cal_pair_count": len(supporting),
            "cal_speaker_count": len({x["speaker"] for x in supporting}),
            "count_gate_frozen_here": False,
        }
    phone_inventory = sorted({
        str(phone["token"])
        for item in per_condition.values()
        for condition in ("natural", "tts")
        for phone in item[condition].get("tokens", [])
    })
    phone_inventory_sha256 = hashlib.sha256(canonical_json(phone_inventory)).hexdigest()
    freeze = {
        "schema_version": 1,
        "manifest_type": "source_axis_mfa_calibration_qualification_freeze",
        "status": "FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_manifest_sha256": protocol_manifest["manifest_sha256"],
        "producer_manifest_sha256_before_qualification": producer.get("manifest_sha256"),
        "mfa_configuration": producer.get("mfa_configuration"),
        "selection_split": "calibration",
        "evaluation_textgrids_read": False,
        "qualification_rules": PROTOCOL["sample_qualification"],
        "visual_phone_count_gate": "not_frozen_by_MFA; see frozen visual CAL baseline procedure and source hash in alignment_protocol_frozen.json",
        "sample_counts": {"total": 20, "pair_qualified": len(eligible_pairs), "pair_failed": len(failures)},
        "phone_class_coverage_descriptive_only": phone_coverage,
        "calibration_phone_inventory": phone_inventory,
        "calibration_phone_inventory_sha256": phone_inventory_sha256,
        "calibration_pairs": per_condition,
        "failures": failures,
    }
    freeze["manifest_sha256"] = fingerprint(freeze)
    write_json(outdir / "calibration_qualification_frozen.json", freeze)
    producer["calibration_qualification_manifest"] = str((outdir / "calibration_qualification_frozen.json").resolve())
    producer["calibration_qualification_manifest_sha256"] = freeze["manifest_sha256"]
    producer["status"] = "CAL_QUALIFICATION_FROZEN"
    producer["manifest_sha256"] = fingerprint(producer)
    write_json(producer_path, producer)
    print(json.dumps({"status": "FROZEN", "cal_pairs_qualified": len(eligible_pairs), "cal_pairs_failed": len(failures), "phone_class_coverage_descriptive_only": {k: {x: v[x] for x in ("cal_pair_count", "cal_speaker_count")} for k, v in phone_coverage.items()}, "manifest_sha256": freeze["manifest_sha256"]}, separators=(",", ":")))
    return 0


def score_evaluation(args: argparse.Namespace) -> int:
    outdir = args.outdir.resolve()
    gate_path = outdir / "calibration_qualification_frozen.json"
    gate = read_json(gate_path)
    if gate.get("status") != "FROZEN":
        raise ValueError("calibration gate is not frozen")
    producer = read_json(outdir / "mfa_producer_manifest.json")
    if gate.get("protocol_manifest_sha256") != producer.get("protocol_manifest_sha256"):
        raise ValueError("calibration gate is bound to a different MFA protocol")
    parser_module, normalizer_path = load_textgrid_tools()
    visual_rules, visual_path = load_visual_phone_tools()
    if sha256_file(normalizer_path) != producer["normalizer_parser_sha256"]:
        raise ValueError("normalizer/parser source changed")
    protocol_manifest = read_json(Path(producer["protocol_manifest_path"]))
    if sha256_file(Path(__file__).resolve()) != protocol_manifest["adapter_source"]["sha256"]:
        raise ValueError("MFA adapter source changed after protocol freeze")
    if sha256_file(visual_path) != protocol_manifest["visual_phone_rules"]["sha256"]:
        raise ValueError("visual phone-class source changed")
    eval_rows = [r for r in producer["records"].values() if r["split"] == "evaluation"]
    results = []
    for rec in sorted(eval_rows, key=lambda x: x["id"]):
        sides = {}
        for condition in ("natural", "tts"):
            meta = rec["conditions"][condition]
            try:
                import soundfile as sf
                pcm, sample_rate = sf.read(meta["staged_audio_path"], dtype="float32", always_2d=True)
                decoded_frames = int(pcm.shape[0])
                if decoded_frames <= 0 or pcm.shape[1] != 1 or int(sample_rate) != 16_000:
                    raise ValueError("staged audio must fully decode to nonempty mono 16 kHz PCM")
                grid = Path(meta["expected_textgrid_path"])
                parsed = check_condition_grid(grid, decoded_frames / int(sample_rate), parser_module, visual_rules) if grid.is_file() else {"qualified": False, "errors": ["missing_textgrid"]}
                parsed.update({"input_audio_sha256": meta["source_audio_sha256"], "textgrid_sha256": sha256_file(grid) if grid.is_file() else None})
            except Exception as exc:
                parsed = {"qualified": False, "errors": [f"input_or_grid_error:{type(exc).__name__}"], "input_audio_sha256": meta["source_audio_sha256"], "textgrid_sha256": None}
            sides[condition] = parsed
        pair_ok = bool(sides["natural"].get("qualified") and sides["tts"].get("qualified"))
        class_observed = {
            klass: bool(pair_ok and sides["natural"].get(f"{klass}_phone_count", 0) and sides["tts"].get(f"{klass}_phone_count", 0))
            for klass in PROTOCOL["visual_phone_endpoint"]["classes"]
        }
        results.append({"id": rec["id"], "speaker": rec["speaker"], "split": "evaluation", "pair_qualified": pair_ok, "class_has_phone_tokens_both_sides_before_visual_frame_gate": class_observed, "natural": sides["natural"], "tts": sides["tts"]})
    payload = {
        "schema_version": 1,
        "manifest_type": "source_axis_mfa_evaluation_eligibility",
        "status": "MECHANICAL_APPLICATION_OF_FROZEN_CAL_RULES",
        "protocol_manifest_sha256": gate["protocol_manifest_sha256"],
        "calibration_qualification_sha256": gate["manifest_sha256"],
        "visual_phone_count_gate": "not inferable from TextGrid alone; apply existing visual CAL baseline gate after frozen latency/common-frame calculation",
        "records_total": len(results),
        "records_pair_qualified": sum(r["pair_qualified"] for r in results),
        "records_pair_unobserved": sum(not r["pair_qualified"] for r in results),
        "records": results,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    write_json(outdir / "evaluation_endpoint_eligibility.json", payload)
    print(json.dumps({"records_total": len(results), "qualified": payload["records_pair_qualified"], "unobserved": payload["records_pair_unobserved"], "manifest_sha256": payload["manifest_sha256"]}, separators=(",", ":")))
    return 0


def common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR)
    p.add_argument("--tts-meta", type=Path, default=DEFAULT_TTS_META)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--natural-root", type=Path)
    p.add_argument("--canonical-root", type=Path)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", parents=[common_parser()], help="copy/hash 100 N and canonical TTS inputs, freeze lab/config")
    prep.set_defaults(func=prepare)
    align = sub.add_parser("align", parents=[common_parser()], help="align one split; evaluation requires the frozen CAL gate")
    align.add_argument("--split", choices=["calibration", "evaluation"], required=True)
    align.add_argument("--mfa-bin", type=Path, required=True)
    align.add_argument("--mfa-root-dir", type=Path, required=True)
    align.add_argument("--dictionary", type=Path, required=True)
    align.add_argument("--acoustic", type=Path, required=True)
    align.add_argument("--timeout", type=int, default=3600)
    align.add_argument("--force", action="store_true")
    align.set_defaults(func=run_alignment)
    qualify = sub.add_parser("qualify-cal", parents=[common_parser()], help="parse CAL TextGrids and freeze support/quality before eval")
    qualify.set_defaults(func=qualify_cal)
    score = sub.add_parser("score-evaluation", parents=[common_parser()], help="apply frozen CAL rules to evaluation TextGrids")
    score.set_defaults(func=score_evaluation)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
