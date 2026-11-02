#!/usr/bin/env python3
"""Create a sealed, CPU-only MFA alignment set for the female replication.

The script consumes only frozen source/TTS/audio manifests and writes a new
attempt directory. It applies the existing male CAL MFA structure rules to
female N/T TextGrids without reading visual or SyncNet outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_MFA_VERSION = "3.4.1"
EXPECTED_NORMALIZER_SHA256 = "a5ae2fa9c5f57e25a090dc0432738fd1d6b29655fa1139ff45a0098d3b4a79b3"
EXPECTED_VISUAL_RULES_SHA256 = "5a43a0796104010e69285ddb3c6ea98ea0d008b4cbc8f78a831e678884984d0f"
EXPECTED_PARENT_CAL_QUAL_SHA256 = "dbebb40ce39f1a8f4a71493c0ae19871e97948d186908d2f1545e0015a301650"
UNKNOWN = {"spn", "unk", "unknown", "oov", "<unk>", "<unknown>"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in payload.items() if k != "manifest_sha256"})).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def check_manifest_hash(payload: dict[str, Any], path: Path) -> None:
    embedded = payload.get("manifest_sha256")
    if not embedded or embedded != fingerprint(payload):
        raise ValueError(f"embedded fingerprint mismatch: {path.name}")


def read_pcm(path: Path, expected_sha256: str, *, sf: Any) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"missing or hash-mismatched audio: {path.name}")
    pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if pcm.ndim != 2 or pcm.shape[1] != 1 or int(sample_rate) != 16000 or pcm.shape[0] <= 0:
        raise ValueError(f"audio must fully decode to nonempty mono 16 kHz PCM: {path.name}")
    if not bool(__import__("numpy").isfinite(pcm).all()):
        raise ValueError(f"audio contains nonfinite PCM: {path.name}")
    return {"decoded_frames": int(pcm.shape[0]), "sample_rate_hz": int(sample_rate), "duration_s": float(pcm.shape[0] / sample_rate)}


def prepare_and_align(args: argparse.Namespace) -> int:
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty attempt: {out}")
    out.mkdir(parents=True, exist_ok=True)
    portable_path = args.portable_manifest.resolve(strict=True)
    path_map_path = args.path_map.resolve(strict=True)
    selection_path = args.source_selection.resolve(strict=True)
    tts_frozen_path = args.tts_frozen.resolve(strict=True)
    parent_protocol_path = args.parent_protocol.resolve(strict=True)
    parent_cal_path = args.parent_cal_qualification.resolve(strict=True)
    parent_producer_path = args.parent_producer.resolve(strict=True)
    normalizer_path = args.normalizer_source.resolve(strict=True)
    visual_snapshot_path = args.visual_snapshot.resolve(strict=True)
    mfa_bin = args.mfa_bin.resolve(strict=True)
    dictionary = args.dictionary.resolve(strict=True)
    acoustic = args.acoustic.resolve(strict=True)
    audio_root = args.audio_root.resolve(strict=True)

    portable = read_json(portable_path)
    path_map = read_json(path_map_path)
    selection = read_json(selection_path)
    frozen_tts = read_json(tts_frozen_path)
    parent_protocol = read_json(parent_protocol_path)
    parent_cal = read_json(parent_cal_path)
    parent_producer = read_json(parent_producer_path)
    for payload, path in ((portable, portable_path), (path_map, path_map_path), (selection, selection_path), (frozen_tts, tts_frozen_path), (parent_protocol, parent_protocol_path), (parent_cal, parent_cal_path)):
        check_manifest_hash(payload, path)
    if sha256_file(normalizer_path) != EXPECTED_NORMALIZER_SHA256:
        raise ValueError("pinned character normalizer/parser source hash changed")
    if sha256_file(visual_snapshot_path) != EXPECTED_VISUAL_RULES_SHA256:
        raise ValueError("frozen CAL visual phone-rule source hash changed")
    if sha256_file(parent_cal_path) != EXPECTED_PARENT_CAL_QUAL_SHA256:
        raise ValueError("parent male CAL qualification raw file hash changed")
    if parent_cal.get("status") != "FROZEN" or parent_cal.get("sample_counts", {}).get("pair_qualified") != 20:
        raise ValueError("male CAL qualification is not the frozen 20/20 source gate")
    protocol_data = parent_protocol.get("protocol", {})
    rules = protocol_data.get("sample_qualification", {})
    timing = protocol_data.get("textgrid_clock", {})
    if not rules or not timing or protocol_data.get("mfa_version_required") != EXPECTED_MFA_VERSION:
        raise ValueError("parent frozen MFA protocol has incomplete rules/version")
    # The portable path-map producer binds this field to the portable manifest's
    # embedded fingerprint, while its own source-manifest fields bind raw bytes.
    # Keep both values in the resulting producer manifest so the relationship is
    # explicit and independently checkable.
    if path_map.get("portable_input_manifest_sha256") != portable.get("manifest_sha256"):
        raise ValueError("female path map is not bound to the portable manifest fingerprint")
    if path_map.get("source_selection_manifest_file_sha256") != sha256_file(selection_path):
        raise ValueError("female path map is not bound to source-selection bytes")
    if portable.get("source_selection_manifest_file_sha256") != sha256_file(selection_path):
        raise ValueError("female portable manifest does not bind source-selection bytes")
    if portable.get("tts_frozen_manifest_file_sha256") != sha256_file(tts_frozen_path):
        raise ValueError("female portable manifest does not bind frozen TTS input bytes")
    if portable.get("source_selection_manifest_fingerprint") != selection.get("manifest_sha256"):
        raise ValueError("female portable manifest source-selection fingerprint differs")
    if portable.get("tts_metadata_model") != "qwen3-tts-vc-2026-01-22":
        raise ValueError("female portable manifest provider model differs")
    if portable.get("split") != "evaluation" or portable.get("status") != "FROZEN_BEFORE_MODEL_FORWARD":
        raise ValueError("female input manifest is not a sealed evaluation cohort")
    if frozen_tts.get("source_manifest_sha256") != sha256_file(selection_path) or frozen_tts.get("source_manifest_fingerprint") != selection.get("manifest_sha256"):
        raise ValueError("female frozen TTS manifest is not bound to the selected source manifest")
    if portable.get("evaluation_outcomes_read") or portable.get("evaluation_waveforms_read") or portable.get("model_effects_read"):
        raise ValueError("female input manifest indicates outcome/effect access")

    rows = portable.get("records", [])
    selection_rows = {str(r["id"]): r for r in selection.get("records", [])}
    frozen_rows = {str(r["id"]): r for r in frozen_tts.get("records", [])}
    path_rows = {str(r["id"]): r for r in path_map.get("records", [])}
    if len(rows) != 40 or len(selection_rows) != 40 or len(frozen_rows) != 40 or len(path_rows) != 40:
        raise ValueError("female cohort must contain exactly 40 frozen rows in every source manifest")
    if set(selection_rows) != set(frozen_rows) or set(selection_rows) != set(path_rows) or set(selection_rows) != {str(r["id"]) for r in rows}:
        raise ValueError("female manifests do not describe the same exact 40 sample IDs")
    speakers = {str(row["speaker"]) for row in rows}
    counts = Counter(str(row["speaker"]) for row in rows)
    if len(speakers) != 20 or set(counts.values()) != {2}:
        raise ValueError("female cohort must have two utterances for each of 20 speakers")

    identities = []
    for row in rows:
        sample_id = str(row["id"])
        selected = selection_rows[sample_id]
        frozen = frozen_rows[sample_id]
        mapped = path_rows[sample_id]
        if row.get("split") != "evaluation" or row.get("speaker_metadata_sex") != "F":
            raise ValueError(f"female evaluation/speaker-sex binding failed: {sample_id}")
        if selected.get("speaker") != row.get("speaker") or selected.get("text") != row.get("text") or selected.get("split") != row.get("split"):
            raise ValueError(f"source selection identity mismatch: {sample_id}")
        if any(frozen.get(k) != row.get(k) for k in ("speaker", "text", "split")):
            raise ValueError(f"frozen TTS input identity mismatch: {sample_id}")
        if selected.get("ref_sha256") != row.get("natural_audio_sha256") or frozen.get("ref_sha256") != row.get("natural_audio_sha256"):
            raise ValueError(f"natural source PCM identity mismatch: {sample_id}")
        if mapped.get("natural_sha256") != row.get("natural_audio_sha256") or mapped.get("canonical_tts_16k_sha256") != row.get("canonical_tts_16k_audio_sha256"):
            raise ValueError(f"portable path-map hash mismatch: {sample_id}")
        identities.append({"id": sample_id, "speaker": row["speaker"], "text": row["text"], "split": row["split"], "ref_sha256": row["natural_audio_sha256"]})
    identity_digest = hashlib.sha256(canonical_json(identities)).hexdigest()
    if identity_digest != portable.get("input_identity_sha256"):
        raise ValueError("female ordered pair identity digest differs from portable input manifest")

    normalizer = load_module(normalizer_path, "female_mfa_pinned_normalizer")
    visual_snapshot_sha = sha256_file(visual_snapshot_path)
    mfa_version = subprocess.run([str(mfa_bin), "version"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout.strip().splitlines()[0]
    if EXPECTED_MFA_VERSION not in mfa_version:
        raise ValueError(f"expected MFA {EXPECTED_MFA_VERSION}, got {mfa_version}")
    parent_config = parent_producer.get("mfa_configuration", {})
    model_info = {
        "mfa_version": mfa_version,
        "mfa_binary_path": str(mfa_bin),
        "mfa_binary_sha256": sha256_file(mfa_bin),
        "dictionary_name": "mandarin_china_mfa",
        "dictionary_path": str(dictionary),
        "dictionary_sha256": hash_tree(dictionary),
        "acoustic_model_name": "mandarin_mfa",
        "acoustic_model_path": str(acoustic),
        "acoustic_model_sha256": hash_tree(acoustic),
        "mfa_root_dir": str(args.mfa_root.resolve()),
        "num_jobs": 2,
        "single_speaker": True,
        "clean": True,
        "environment": {"CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(args.mfa_tmp.resolve())},
    }
    model_info["configuration_sha256"] = hashlib.sha256(canonical_json(model_info)).hexdigest()
    if model_info["mfa_binary_sha256"] != parent_config.get("mfa_binary_sha256") or model_info["dictionary_sha256"] != parent_config.get("dictionary_sha256") or model_info["acoustic_model_sha256"] != parent_config.get("acoustic_model_sha256"):
        raise ValueError("female MFA binary/model/dictionary differs from frozen male configuration")
    if model_info["configuration_sha256"] != parent_config.get("configuration_sha256"):
        raise ValueError("female MFA runtime configuration differs from frozen male CAL/EVAL configuration")

    code_path = Path(__file__).resolve()
    run_protocol = {
        "schema_version": 1,
        "manifest_type": "female_replication_mfa_protocol",
        "status": "FROZEN_BEFORE_ALIGNMENT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"split": "evaluation", "speaker_sex": "F", "sample_count": 40, "speaker_count": 20, "utterances_per_speaker": 2},
        "conditions": ["natural", "tts"],
        "alignment_total_textgrids": 80,
        "selection_identity_sha256": identity_digest,
        "source_manifests": {
            "portable_input_manifest_path": str(portable_path), "portable_input_manifest_raw_sha256": sha256_file(portable_path), "portable_input_manifest_fingerprint": portable.get("manifest_sha256"),
            "portable_path_map_raw_sha256": sha256_file(path_map_path), "portable_path_map_fingerprint": path_map.get("manifest_sha256"),
            "source_selection_manifest_raw_sha256": sha256_file(selection_path), "source_selection_manifest_fingerprint": selection.get("manifest_sha256"),
            "tts_frozen_manifest_raw_sha256": sha256_file(tts_frozen_path), "tts_frozen_manifest_fingerprint": frozen_tts.get("manifest_sha256"),
            "aishell1_source_dataset": "official OpenSLR SLR33 data_aishell.tgz",
            "official_archive_md5": "2f494334227864a8a8fec932999db9d8",
            "official_archive_sha256": "a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe",
            "publisher_transcript_revision": "bbe295d530192a4cd41644b711c9aecd087df653",
        },
        "parent_frozen_male_mfa": {
            "protocol_raw_sha256": sha256_file(parent_protocol_path), "protocol_manifest_sha256": parent_protocol.get("manifest_sha256"),
            "calibration_qualification_raw_sha256": sha256_file(parent_cal_path), "calibration_qualification_manifest_sha256": parent_cal.get("manifest_sha256"),
            "calibration_pairs_qualified": parent_cal.get("sample_counts", {}).get("pair_qualified"),
            "female_qualification_rules": rules, "female_textgrid_clock_rules": timing,
        },
        "mfa_configuration": model_info,
        "source_code": {"adapter_path": str(code_path), "adapter_sha256": sha256_file(code_path), "normalizer_path": str(normalizer_path), "normalizer_sha256": sha256_file(normalizer_path), "frozen_visual_phone_rules_path": str(visual_snapshot_path), "frozen_visual_phone_rules_sha256": visual_snapshot_sha},
        "outcome_blindness": {"female_evaluation_outcomes_read": False, "female_evaluation_waveforms_used_only_for_MFA_alignment_and_clock_validation": True, "visual_or_syncnet_effects_read": False},
    }
    run_protocol["manifest_sha256"] = fingerprint(run_protocol)
    protocol_out = out / "female_mfa_protocol_frozen.json"
    if protocol_out.exists():
        raise FileExistsError(protocol_out)
    write_json(protocol_out, run_protocol)

    file_records: dict[str, Any] = {}
    for row in sorted(rows, key=lambda r: str(r["id"])):
        sample_id = str(row["id"])
        speaker = str(row["speaker"])
        normalized = normalizer.clean_lab_text(str(row["text"]))
        if not normalized:
            raise ValueError(f"empty normalized LAB transcript: {sample_id}")
        conditions: dict[str, Any] = {}
        for cond, rel_path, file_sha in (
            ("natural", row["natural_audio_path"], row["natural_audio_sha256"]),
            ("tts", row["canonical_tts_16k_audio_path"], row["canonical_tts_16k_audio_sha256"]),
        ):
            source_audio = resolve_under(audio_root, str(rel_path))
            audio_info = read_pcm(source_audio, str(file_sha), sf=__import__("soundfile"))
            if cond == "natural" and int(row["natural_samples"]) != audio_info["decoded_frames"]:
                raise ValueError(f"natural EOF decoded sample count mismatch: {sample_id}")
            if cond == "tts" and int(row["canonical_tts_samples"]) != audio_info["decoded_frames"]:
                raise ValueError(f"canonical TTS EOF decoded sample count mismatch: {sample_id}")
            stem = f"{sample_id}__{cond}"
            staged_dir = out / "corpus" / "evaluation" / speaker
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_wav = staged_dir / f"{stem}.wav"
            lab = staged_dir / f"{stem}.lab"
            if staged_wav.exists() or lab.exists():
                raise FileExistsError(f"unexpected existing staged file: {stem}")
            shutil.copyfile(source_audio, staged_wav)
            lab.write_text(normalized + "\n", encoding="utf-8")
            if sha256_file(staged_wav) != file_sha:
                raise ValueError(f"staged WAV hash changed: {stem}")
            conditions[cond] = {
                "input_path": str(source_audio), "input_audio_sha256": str(file_sha), "staged_audio_path": str(staged_wav),
                "staged_audio_sha256": sha256_file(staged_wav), "sample_rate_hz": audio_info["sample_rate_hz"],
                "decoded_frames": audio_info["decoded_frames"], "audio_duration_s": audio_info["duration_s"],
                "transcript": str(row["text"]), "transcript_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                "normalized_lab_text": normalized, "lab_path": str(lab), "lab_sha256": sha256_file(lab),
                "textgrid_path": str((out / "textgrids" / "evaluation" / speaker / f"{stem}.TextGrid").resolve()),
            }
        file_records[sample_id] = {"id": sample_id, "speaker": speaker, "split": "evaluation", "sex": "F", "text": str(row["text"]), "source_selection_ref_sha256": selection_rows[sample_id]["ref_sha256"], "conditions": conditions}
    producer = {
        "schema_version": 1, "manifest_type": "female_replication_mfa_producer", "status": "INPUTS_PREPARED",
        "protocol_path": str(protocol_out.resolve()), "protocol_manifest_sha256": run_protocol["manifest_sha256"],
        "mfa_configuration": model_info, "records_expected": 40, "conditions_expected": 80,
        "records": file_records, "alignment_groups": {},
    }
    producer["manifest_sha256"] = fingerprint(producer)
    producer_path = out / "female_mfa_producer_manifest.json"
    write_json(producer_path, producer)
    print(json.dumps({"event": "female_mfa_prepared", "samples": len(file_records), "speakers": len(speakers), "conditions": 2, "input_identity_sha256": identity_digest, "protocol_sha256": run_protocol["manifest_sha256"], "producer_sha256": producer["manifest_sha256"], "outdir": str(out)}, separators=(",", ":")), flush=True)

    env = dict(os.environ)
    env["MFA_ROOT_DIR"] = str(args.mfa_root.resolve())
    env["TMPDIR"] = str(args.mfa_tmp.resolve())
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PATH"] = str(mfa_bin.parent) + os.pathsep + env.get("PATH", "")
    args.mfa_tmp.mkdir(parents=True, exist_ok=True)
    speakers_sorted = sorted(speakers)
    for index, speaker in enumerate(speakers_sorted, 1):
        input_dir = out / "corpus" / "evaluation" / speaker
        output_dir = out / "textgrids" / "evaluation" / speaker
        log_path = out / "logs" / "evaluation" / f"{speaker}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [str(mfa_bin), "align", str(input_dir), str(dictionary), str(acoustic), str(output_dir), "--single_speaker", "--clean", "--num_jobs", "2"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout, env=env)
        log_payload = {"command": command, "command_sha256": hashlib.sha256(canonical_json(command)).hexdigest(), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        write_json(log_path, log_payload)
        grids = []
        for rec in file_records.values():
            if rec["speaker"] != speaker:
                continue
            for cond in ("natural", "tts"):
                grid = Path(rec["conditions"][cond]["textgrid_path"])
                grids.append({"sample_id": rec["id"], "condition": cond, "textgrid_path": str(grid), "textgrid_sha256": sha256_file(grid) if grid.is_file() else None, "status": "produced" if result.returncode == 0 and grid.is_file() else "missing_or_failed"})
        producer["alignment_groups"][speaker] = {"command": command, "command_sha256": log_payload["command_sha256"], "returncode": result.returncode, "log_path": str(log_path.resolve()), "log_sha256": sha256_file(log_path), "textgrids": grids}
        producer["status"] = "ALIGNMENT_IN_PROGRESS"
        producer["manifest_sha256"] = fingerprint(producer)
        write_json(producer_path, producer)
        print(f"ALIGNED female speaker {index}/{len(speakers_sorted)} {speaker} rc={result.returncode}", flush=True)
    producer["status"] = "ALIGNMENT_COMPLETE"
    producer["manifest_sha256"] = fingerprint(producer)
    write_json(producer_path, producer)

    parser = normalizer
    eligible = []
    failures = []
    detailed: dict[str, Any] = {}
    for rec in sorted(file_records.values(), key=lambda r: r["id"]):
        side_results = {}
        for cond in ("natural", "tts"):
            meta = rec["conditions"][cond]
            grid = Path(meta["textgrid_path"])
            errors: list[str] = []
            tokens: list[dict[str, Any]] = []
            grid_xmax = None
            try:
                if not grid.is_file():
                    raise FileNotFoundError(grid.name)
                tokens = parser.parse_phones(grid)
                raw_grid = grid.read_text(encoding="utf-8")
                m = re.search(r"xmax\s*=\s*([\d.]+)", raw_grid)
                grid_xmax = float(m.group(1)) if m else None
                if grid_xmax is None:
                    errors.append("textgrid_xmax_missing")
                elif abs(grid_xmax - float(meta["audio_duration_s"])) > float(timing["boundary_tolerance_s"]):
                    errors.append("textgrid_clock_mismatch")
                unknown = [t for t in tokens if str(t["token"]).casefold() in UNKNOWN]
                speech = [t for t in tokens if not t["is_silence"] or str(t["token"]).casefold() in UNKNOWN]
                if not speech and rules.get("at_least_one_speech_phone_each_side"):
                    errors.append("no_speech_phone")
                if unknown and rules.get("reject_unknown_or_oov_phone_each_side"):
                    errors.append("unknown_or_oov_phone")
                tolerance = float(timing["boundary_tolerance_s"])
                overlap_tolerance = float(timing["overlap_tolerance_s"])
                min_duration = float(timing["interval_min_duration_s"])
                for i, token in enumerate(tokens):
                    if float(token["duration_s"]) < min_duration:
                        errors.append("nonpositive_phone_duration"); break
                    if float(token["start_s"]) < -tolerance or float(token["end_s"]) > float(meta["audio_duration_s"]) + tolerance:
                        errors.append("phone_outside_audio_clock"); break
                    if i and float(tokens[i - 1]["end_s"]) - float(token["start_s"]) > overlap_tolerance:
                        errors.append("phone_overlap"); break
                if tokens:
                    if tokens[0]["start_s"] > tolerance:
                        errors.append("audio_start_not_covered")
                    if tokens[-1]["end_s"] < float(meta["audio_duration_s"]) - tolerance:
                        errors.append("audio_end_not_covered")
                elif rules.get("at_least_one_speech_phone_each_side"):
                    errors.append("no_phone_intervals")
                if grid.is_file():
                    errors = sorted(set(errors))
            except Exception as exc:
                errors.append(f"grid_or_input_error:{type(exc).__name__}")
            side_results[cond] = {
                "qualified": not errors, "errors": sorted(set(errors)), "phone_count_including_silence": len(tokens),
                "speech_phone_count": sum(not t["is_silence"] or str(t["token"]).casefold() in UNKNOWN for t in tokens),
                "unknown_oov_phone_count": sum(str(t["token"]).casefold() in UNKNOWN for t in tokens),
                "phone_labels": [str(t["token"]) for t in tokens], "phone_intervals": tokens,
                "input_audio_sha256": meta["input_audio_sha256"], "staged_audio_sha256": meta["staged_audio_sha256"],
                "textgrid_path": str(grid), "textgrid_sha256": sha256_file(grid) if grid.is_file() else None,
                "audio_decoded_frames": meta["decoded_frames"], "sample_rate_hz": meta["sample_rate_hz"],
                "audio_duration_s": meta["audio_duration_s"], "textgrid_xmax_s": grid_xmax,
                "mfa_configuration_sha256": model_info["configuration_sha256"],
            }
        pair_ok = bool(side_results["natural"]["qualified"] and side_results["tts"]["qualified"])
        result = {"id": rec["id"], "speaker": rec["speaker"], "split": "evaluation", "pair_qualified": pair_ok, "natural": side_results["natural"], "tts": side_results["tts"]}
        detailed[rec["id"]] = result
        if pair_ok:
            eligible.append(rec["id"])
        else:
            failures.append({"id": rec["id"], "speaker": rec["speaker"], "natural_errors": side_results["natural"]["errors"], "tts_errors": side_results["tts"]["errors"]})
    eligibility = {
        "schema_version": 1, "manifest_type": "female_replication_mfa_endpoint_eligibility",
        "status": "MECHANICAL_APPLICATION_OF_FROZEN_MALE_CAL_RULES", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": run_protocol["manifest_sha256"], "producer_manifest_sha256_before_eligibility": producer["manifest_sha256"],
        "parent_male_cal_qualification_sha256": parent_cal.get("manifest_sha256"),
        "evaluation_outcomes_read": False, "evaluation_model_effects_read": False,
        "rules": rules, "textgrid_clock": timing, "sample_counts": {"total": len(file_records), "pair_qualified": len(eligible), "pair_unobserved": len(file_records) - len(eligible)},
        "records": detailed, "failures": failures,
    }
    eligibility["manifest_sha256"] = fingerprint(eligibility)
    write_json(out / "female_endpoint_eligibility.json", eligibility)
    producer["endpoint_eligibility_path"] = str((out / "female_endpoint_eligibility.json").resolve())
    producer["endpoint_eligibility_sha256"] = eligibility["manifest_sha256"]
    producer["status"] = "FEMALE_ENDPOINT_QUALIFICATION_COMPLETE"
    producer["manifest_sha256"] = fingerprint(producer)
    write_json(producer_path, producer)
    print(json.dumps({"event": "female_mfa_qualification_complete", "total": len(file_records), "qualified": len(eligible), "unobserved": len(file_records) - len(eligible), "producer_sha256": producer["manifest_sha256"], "eligibility_sha256": eligibility["manifest_sha256"], "outdir": str(out)}, separators=(",", ":")), flush=True)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--portable-manifest", type=Path, required=True)
    p.add_argument("--path-map", type=Path, required=True)
    p.add_argument("--source-selection", type=Path, required=True)
    p.add_argument("--tts-frozen", type=Path, required=True)
    p.add_argument("--parent-protocol", type=Path, required=True)
    p.add_argument("--parent-cal-qualification", type=Path, required=True)
    p.add_argument("--parent-producer", type=Path, required=True)
    p.add_argument("--normalizer-source", type=Path, required=True)
    p.add_argument("--visual-snapshot", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--mfa-bin", type=Path, required=True)
    p.add_argument("--mfa-root", type=Path, required=True)
    p.add_argument("--mfa-tmp", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--acoustic", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=3600)
    return prepare_and_align(p.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
