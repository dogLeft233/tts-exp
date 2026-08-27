from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    AlignmentError,
    parse_textgrid,
    transcript_sha256,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.mfa3_alignment import (
    MFA_ACOUSTIC_MODEL,
    MFA_DEFAULT_ROOT,
    MFA_DICTIONARY,
    MFA_VERSION,
    NORMALIZATION_POLICY_ID,
    normalize_mfa3_transcript,
    normalization_events,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
STAGE00_ID = "00_protocol_lock_retry1"
CANONICAL_STAGE00_ID = "00_protocol_lock_retry2"
STAGE01_ID = "01_mfa3_screen_retry1"
EXPLORATORY_ROOT = REPO / f"runs/{PROTOCOL_ID}"
CANONICAL_STAGE00_PATH = EXPLORATORY_ROOT / CANONICAL_STAGE00_ID / "manifest.json"
CANDIDATE_STAGE_ID = "02_candidate_audio_retry4"
CANONICAL_CANDIDATE_STAGE_ID = "02_candidate_audio_retry4"
CANONICAL_CANDIDATE_PATH = EXPLORATORY_ROOT / CANONICAL_CANDIDATE_STAGE_ID / "candidate_manifest.json"
CANONICAL_FACE_PREFLIGHT_PATH = EXPLORATORY_ROOT / "03_face_preflight_retry1" / "face_preflight_manifest.json"
CANONICAL_SYNCNET_PREFLIGHT_PATH = EXPLORATORY_ROOT / "03_syncnet_preflight_retry1" / "syncnet_preflight_manifest.json"
CANONICAL_SYNCNET_PREFLIGHT_SHA256 = "269d79ec48c5664374388f05bfd768081e75ed7c797c0e0ea132dafe2e8b70c4"
CANONICAL_FACE_READY_RENDER_PARENT = EXPLORATORY_ROOT / "03_strict_replacement_face_ready_retry1"
EXPECTED_SOURCE_RECORDS = 500
MIN_CLEAN_RECORDS = 24
SAMPLE_RATE = 16_000

SOURCE_MANIFEST = REPO / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
TTS_META = REPO / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"
PARENT_MANIFEST = REPO / "runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1/manifest.json"
PARENT_TEST_LOCK = REPO / "runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1/test_lock.json"
TEACHER_MANIFEST = REPO / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teacher_manifest.json"
ORIGINAL_STAGE00 = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_20260825/00_protocol_lock_mfa3_retry4/manifest.json"
MFA_EXECUTABLE = Path("/home/wjj/miniconda3/envs/mfa3/bin/mfa")
MFA_ROOT_DIR = MFA_DEFAULT_ROOT
MFA_DICTIONARY_PATH = MFA_ROOT_DIR / "pretrained_models/dictionary/english_mfa.dict"
MFA_ACOUSTIC_MODEL_PATH = MFA_ROOT_DIR / "pretrained_models/acoustic/english_mfa.zip"
WAVLM_CHECKPOINT = Path("/home/wjj/.cache/torch/hub/checkpoints/WavLM-Large.pt")
VOCODER_CHECKPOINT = Path("/home/wjj/.cache/torch/hub/checkpoints/prematch_g_02500000.pt")
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
MFA_ROOT_CONFIG = MFA_ROOT_DIR / "global_config.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def asset(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def dictionary_words(path: Path = MFA_DICTIONARY_PATH) -> set[str]:
    words: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            words.add(line.split()[0].upper())
    if not words:
        raise ValueError(f"MFA dictionary is empty: {path}")
    return words


def dictionary_missing_tokens(text: str, words: set[str]) -> list[str]:
    normalized = normalize_mfa3_transcript(text)
    return sorted({token for token in normalized.split() if token not in words})


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _validate_source_and_tts(source: Mapping[str, Any], tts: Mapping[str, Any], source_path: Path) -> None:
    if source.get("sample_count") != EXPECTED_SOURCE_RECORDS or len(source.get("records", [])) != EXPECTED_SOURCE_RECORDS:
        raise ValueError("canonical LRS3 source must be n=500")
    if not tts.get("complete") or tts.get("sample_count") != EXPECTED_SOURCE_RECORDS:
        raise ValueError("canonical Qwen TTS metadata must be complete n=500")
    if tts.get("source_manifest_sha256") != file_sha256(source_path):
        raise ValueError("TTS metadata is not bound to the source manifest")


def build_exploratory_pool(
    *,
    source_path: Path = SOURCE_MANIFEST,
    tts_path: Path = TTS_META,
    parent_path: Path = PARENT_MANIFEST,
    teacher_path: Path = TEACHER_MANIFEST,
    original_stage00_path: Path = ORIGINAL_STAGE00,
    dictionary_path: Path = MFA_DICTIONARY_PATH,
) -> dict[str, Any]:
    source = load_json(source_path)
    tts = load_json(tts_path)
    parent = load_json(parent_path)
    teacher = load_json(teacher_path)
    original = load_json(original_stage00_path)
    _validate_source_and_tts(source, tts, source_path)
    if parent.get("status") != "complete" or parent.get("decision") != "GO":
        raise ValueError("parent split protocol is not complete GO")
    if original.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_20260825":
        raise ValueError("unexpected original MFA3 Stage00 protocol")

    split = parent.get("protocol", {}).get("split", {})
    fit_groups = [str(value) for value in split.get("fit_groups", [])]
    sealed_groups = {
        str(value)
        for key in ("internal_dev_groups", "validation_groups", "test_groups")
        for value in split.get(key, [])
    }
    if not fit_groups or set(fit_groups) & sealed_groups:
        raise ValueError("invalid parent fit/sealed group split")

    source_records = source.get("records", [])
    source_by_id = {str(row["sample_id"]): row for row in source_records}
    tts_by_id = {str(key): value for key, value in tts.get("results", {}).items()}
    teacher_ids = {str(row["sample_id"]) for row in teacher.get("records", [])}
    original_ids = {str(row["sample_id"]) for row in original.get("cohort", {}).get("records", [])}
    excluded_ids = teacher_ids | original_ids
    words = dictionary_words(dictionary_path)

    eligible: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    rejection_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    fit_count = 0
    for source_row in source_records:
        sample_id = str(source_row.get("sample_id", ""))
        group = str(source_row.get("source_group", ""))
        if group not in fit_groups:
            continue
        fit_count += 1
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty source sample ID: {sample_id}")
        seen.add(sample_id)
        if sample_id in teacher_ids:
            reason = "teacher_cache_exclusion"
        elif sample_id in original_ids:
            reason = "original_mfa3_cohort_exclusion"
        else:
            tts_row = tts_by_id.get(sample_id)
            if not isinstance(tts_row, Mapping):
                reason = "missing_tts_join"
            elif tts_row.get("source_group") != group:
                reason = "source_group_join_mismatch"
            elif tts_row.get("reference_audio_sha256") != source_row.get("natural_audio_sha256"):
                reason = "natural_reference_hash_mismatch"
            elif tts_row.get("canonical_audio_sha256") is None:
                reason = "missing_tts_audio_hash"
            else:
                try:
                    normalized = normalize_mfa3_transcript(str(source_row.get("transcript", "")))
                    normalized_tts = normalize_mfa3_transcript(str(tts_row.get("tts_transcript", "")))
                except Exception as exc:
                    reason = f"normalization_error:{exc}"
                else:
                    if normalized != normalized_tts:
                        reason = "natural_tts_transcript_mismatch"
                    else:
                        missing = sorted({token for token in normalized.split() if token not in words})
                        if missing:
                            reason = "dictionary_prefilter:" + ",".join(missing)
                        else:
                            natural_path = _repo_path(str(source_row["natural_audio_path"]))
                            tts_audio = Path(str(tts_row["canonical_16k_audio"])).resolve()
                            if not natural_path.is_file() or not tts_audio.is_file():
                                reason = "audio_path_missing"
                            else:
                                eligible.append({
                                    "sample_id": sample_id,
                                    "source_group": group,
                                    "natural_audio": str(natural_path),
                                    "natural_audio_sha256": str(source_row["natural_audio_sha256"]),
                                    "tts_audio": str(tts_audio),
                                    "tts_audio_sha256": str(tts_row["canonical_audio_sha256"]),
                                    "transcript": str(source_row["transcript"]),
                                    "tts_transcript": str(tts_row["tts_transcript"]),
                                    "transcript_sha256": transcript_sha256(str(source_row["transcript"])),
                                    "tts_transcript_sha256": transcript_sha256(str(tts_row["tts_transcript"])),
                                    "normalized_transcript": normalized,
                                    "normalized_tts_transcript": normalized_tts,
                                    "normalization_events": normalization_events(str(source_row["transcript"])),
                                    "natural_audio_samples": int(source_row["natural_audio_samples"]),
                                    "tts_audio_samples": int(tts_row["canonical_samples"]),
                                })
                                continue
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        rejection_rows.append({"sample_id": sample_id, "source_group": group, "reason": reason})

    if fit_count != 271:
        raise ValueError(f"unexpected fit pool size: {fit_count}")
    if len(eligible) < MIN_CLEAN_RECORDS:
        raise ValueError(f"dictionary-prefiltered exploratory pool is too small: {len(eligible)}")
    if {row["sample_id"] for row in eligible} & excluded_ids:
        raise ValueError("exploratory pool intersects excluded records")

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection": {
            "source_manifest_order": True,
            "fit_groups_only": True,
            "sealed_groups_excluded": True,
            "teacher_cache_excluded": True,
            "original_mfa3_cohort_excluded": True,
            "score_based_selection": False,
            "alignment_result_based_selection": False,
            "dictionary_prefilter": "all normalized tokens must exist in pinned english_mfa dictionary",
            "normalization_policy_id": NORMALIZATION_POLICY_ID,
        },
        "counts": {
            "source_records": EXPECTED_SOURCE_RECORDS,
            "fit_pool_records": fit_count,
            "eligible_records": len(eligible),
            "eligible_source_groups": len({row["source_group"] for row in eligible}),
            "teacher_excluded": len(teacher_ids),
            "original_cohort_excluded": len(original_ids),
            "rejection_counts": rejection_counts,
        },
        "records": eligible,
        "rejections": rejection_rows,
        "dictionary": asset(dictionary_path),
    }


def build_stage00_artifacts(
    pool: Mapping[str, Any],
    *,
    stage_id: str = STAGE00_ID,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = list(pool["records"])
    parents = {
        "source_manifest": asset(SOURCE_MANIFEST),
        "tts_meta": asset(TTS_META),
        "parent_manifest": asset(PARENT_MANIFEST),
        "parent_test_lock": asset(PARENT_TEST_LOCK),
        "teacher_manifest": asset(TEACHER_MANIFEST),
        "original_mfa3_stage00": asset(ORIGINAL_STAGE00),
    }
    mfa_contract = {
        "version": MFA_VERSION,
        "root_dir": str(MFA_ROOT_DIR.resolve()),
        "executable": asset(MFA_EXECUTABLE),
        "root_config": asset(MFA_ROOT_CONFIG),
        "dictionary": {"name": MFA_DICTIONARY, **asset(MFA_DICTIONARY_PATH)},
        "acoustic_model": {"name": MFA_ACOUSTIC_MODEL, **asset(MFA_ACOUSTIC_MODEL_PATH)},
    }
    candidate_assets = {
        "wavlm": asset(WAVLM_CHECKPOINT),
        "vocoder": asset(VOCODER_CHECKPOINT),
    }
    candidate_contract = {
        "repository": "bshall/knn-vc",
        "revision": KNN_VC_REVISION,
        "sample_rate_hz": SAMPLE_RATE,
        "frame_stride_samples": 320,
        "feature_layer": 6,
        "feature_dim": 1024,
        "prematched_vocoder": True,
        "loudness_normalization": False,
        "exact_length_policy": "right_crop_or_right_zero_pad_only",
        "speech_fallback_frames_required": 0,
    }
    test_lock = {
        "status": "sealed_unvisited",
        "scope": "internal_dev_validation_and_test",
        "fit_media_opened": False,
        "internal_dev_media_opened": False,
        "validation_media_opened": False,
        "test_media_opened": False,
        "derived_features_created": False,
        "scores_created": False,
        "test_groups": json.loads(PARENT_TEST_LOCK.read_text(encoding="utf-8")).get("test_groups", []),
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_exploratory_pool_protocol_lock",
        "protocol_id": PROTOCOL_ID,
        "stage_id": stage_id,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": STAGE01_ID if stage_id == STAGE00_ID else CANDIDATE_STAGE_ID,
        "parents": parents,
        "mfa_contract": mfa_contract,
        "assets": candidate_assets,
        "candidate_contract": candidate_contract,
        "pool": dict(pool),
        "cohort": {
            "record_count": len(records),
            "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in records]),
            "source_group_count": len({row["source_group"] for row in records}),
            "records": records,
        },
        "test_lock": test_lock,
        "media_access": {
            "fit_media_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "mfa_run": False,
            "features_created": False,
            "scores_created": False,
        },
    }
    summary = {
        "stage_id": stage_id,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "fit_pool_records": pool["counts"]["fit_pool_records"],
        "eligible_records": len(records),
        "eligible_source_groups": len({row["source_group"] for row in records}),
        "sealed_splits_unvisited": True,
        "mfa_run": False,
    }
    decision = {
        "stage_id": stage_id,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": STAGE01_ID if stage_id == STAGE00_ID else CANDIDATE_STAGE_ID,
        "reason": "deterministic disjoint fit-only pool frozen before MFA execution",
    }
    return manifest, summary, decision, test_lock


def write_stage00(
    output_dir: Path,
    pool: Mapping[str, Any],
    *,
    stage_id: str = STAGE00_ID,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty Stage00 output: {output_dir}")
    manifest, summary, decision, test_lock = build_stage00_artifacts(pool, stage_id=stage_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", decision)
    write_json(output_dir / "test_lock.json", test_lock)
    (output_dir / "manifest.sha256").write_text(file_sha256(output_dir / "manifest.json") + "\n", encoding="utf-8")
    return summary


def build_mfa3_command(input_dir: Path, output_dir: Path, *, mfa_executable: Path = MFA_EXECUTABLE) -> list[str]:
    return [
        str(mfa_executable),
        "align",
        "--clean",
        "--overwrite",
        str(input_dir),
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        str(output_dir),
    ]


def prepare_corpus(records: Sequence[Mapping[str, Any]], output_dir: Path, *, audio_key: str, hash_key: str) -> list[dict[str, str]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"MFA corpus directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        audio_path = Path(str(record[audio_key])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if file_sha256(audio_path) != str(record[hash_key]):
            raise AlignmentError(f"audio hash mismatch for {sample_id}")
        with wave.open(str(audio_path), "rb") as handle:
            if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
                raise AlignmentError(f"audio format mismatch for {sample_id}")
        destination = output_dir / f"{sample_id}.wav"
        shutil.copy2(audio_path, destination)
        transcript_key = "normalized_tts_transcript" if audio_key == "tts_audio" else "normalized_transcript"
        transcript = str(record[transcript_key])
        (output_dir / f"{sample_id}.lab").write_text(transcript + "\n", encoding="utf-8")
        entries.append({"sample_id": sample_id, "audio_path": str(destination), "transcript_sha256": transcript_sha256(transcript)})
    return entries


def run_mfa3_alignment(
    input_dir: Path,
    output_dir: Path,
    *,
    expected_sample_ids: Sequence[str],
    mfa_executable: Path = MFA_EXECUTABLE,
    mfa_root_dir: Path = MFA_ROOT_DIR,
    runner=subprocess.run,
) -> dict[str, Any]:
    expected = [str(value) for value in expected_sample_ids]
    if not expected or len(expected) != len(set(expected)):
        raise AlignmentError("exploratory MFA execution requires unique ordered sample IDs")
    executable_path = mfa_executable.resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(executable_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(mfa_root_dir.resolve())
    environment["PATH"] = os.pathsep.join([str(executable_path.parent), environment.get("PATH", "")])
    command = build_mfa3_command(input_dir, output_dir, mfa_executable=executable_path)
    result = runner(command, check=False, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"MFA3 alignment failed with exit {result.returncode}: {result.stderr[-2000:]}")
    missing = [sample_id for sample_id in expected if not (output_dir / f"{sample_id}.TextGrid").is_file()]
    return {
        "command": command,
        "returncode": int(result.returncode),
        "expected_count": len(expected),
        "textgrid_count": len(expected) - len(missing),
        "missing_sample_ids": missing,
        "mfa_root_dir": str(mfa_root_dir.resolve()),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def parse_alignment_pair(record: Mapping[str, Any], natural_grid: Path, tts_grid: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    natural_tokens = None
    tts_tokens = None
    for side, path in (("natural", natural_grid), ("tts", tts_grid)):
        if not path.is_file():
            failures.append({"sample_id": record["sample_id"], "side": side, "error": "missing TextGrid"})
            continue
        try:
            tokens = parse_textgrid(path)
        except Exception as exc:
            failures.append({
                "sample_id": record["sample_id"],
                "side": side,
                "error": str(exc),
                "textgrid_sha256": file_sha256(path),
            })
            continue
        if side == "natural":
            natural_tokens = tokens
        else:
            tts_tokens = tokens
    if failures or natural_tokens is None or tts_tokens is None:
        return None, failures
    return {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "natural_textgrid": str(natural_grid),
        "natural_textgrid_sha256": file_sha256(natural_grid),
        "tts_textgrid": str(tts_grid),
        "tts_textgrid_sha256": file_sha256(tts_grid),
        "natural_tokens": natural_tokens,
        "tts_tokens": tts_tokens,
        "normalized_transcript_sha256": str(record["transcript_sha256"]),
        "normalized_tts_transcript_sha256": str(record["tts_transcript_sha256"]),
        "normalization_events": record["normalization_events"],
    }, []


def validate_stage00(
    stage00_path: Path,
    stage00: Mapping[str, Any],
    *,
    expected_stage_id: str = STAGE00_ID,
    expected_path: Path | None = None,
) -> None:
    if expected_path is not None and stage00_path.resolve() != expected_path.resolve():
        raise ValueError("exploratory Stage00 path is not the canonical candidate input")
    if stage00.get("protocol_id") != PROTOCOL_ID or stage00.get("stage_id") != expected_stage_id:
        raise ValueError("unexpected exploratory Stage00 identity")
    if stage00.get("status") != "complete" or stage00.get("engineering_decision") != "GO":
        raise ValueError("exploratory Stage00 is not complete GO")
    companion = stage00_path.parent / "manifest.sha256"
    if companion.read_text(encoding="utf-8").strip() != file_sha256(stage00_path):
        raise ValueError("exploratory Stage00 manifest companion hash mismatch")
    contract = stage00.get("mfa_contract", {})
    if contract.get("version") != MFA_VERSION or contract.get("dictionary", {}).get("name") != MFA_DICTIONARY:
        raise ValueError("exploratory Stage00 MFA contract mismatch")
    for key, path in (
        ("executable", MFA_EXECUTABLE),
        ("root_config", MFA_ROOT_CONFIG),
        ("dictionary", MFA_DICTIONARY_PATH),
        ("acoustic_model", MFA_ACOUSTIC_MODEL_PATH),
    ):
        binding = contract[key]
        if Path(str(binding["path"])).resolve() != path.resolve() or file_sha256(path) != binding["sha256"]:
            raise ValueError(f"exploratory MFA {key} binding mismatch")
