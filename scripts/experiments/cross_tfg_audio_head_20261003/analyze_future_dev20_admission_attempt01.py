#!/usr/bin/env python3
"""Apply the frozen, prospective Dev20 admission policy to pinned artifacts.

This analyzer is read-only with respect to input artifacts. It creates one new
output directory, keeps every planned row in the denominator, and refuses to
compute score effects unless the freeze, complete 540-row result, ASR, visual,
and independent-consumer gates all pass. It does not call ASR or render models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import unicodedata
import io
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POLICY_REL = Path("openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.json")
POLICY_RAW_SHA256 = "f215ecff1dc9908eebfc3195a96d76b041bea47ca0fa7df5d64b6481031398ed"
PROTOCOL_REL = Path(".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json")
PROTOCOL_RAW_SHA256 = "73ac37eee2c8c728a7123fa85834dc53104a4b809cbc017593bd86939816eb17"
CONSUMER_REL = Path("scripts/audio_enhancement_eval/consumer.py")
CONSUMER_RAW_SHA256 = "a693b2ab71876396fb3fd73f9dda40a665f8cd43981c31e6e61afadd17b3942e"
VISUAL_GUARD_REL = Path("scripts/audio_enhancement_eval/visual_motion_guard.py")
VISUAL_GUARD_RAW_SHA256 = "e154a46aa70e3c434ecc3ae9618e82e072d06fe36126eca19571404f6ca6968f"
VISUAL_FEATURE_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
ASR_BASELINE_WORKER_REL = Path(".codex/scratch/dev20_asr_repeat_baseline_calibration_attempt01/run_future_dev20_asr_baseline_calibration_attempt01.py")
ASR_BASELINE_WORKER_SHA256 = "00fc2273d8bcc8b3057f1a25add608a70b57cf83cf8b92ac28786fb0bdbc7efd"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
SPEAKERS = tuple(f"S{i:04d}" for i in range(102, 112))
SEEDS = (20261003, 20261004, 20261005)
SOURCE_METHODS = ("identity", "rms_fixed")
SUPPORTS = ("full", "guard20")
METRICS = ("Cbest", "D0", "Canchor")
CONTRASTS = ("candidate_minus_rms_fixed", "candidate_minus_identity")
ENDPOINT_ESTIMANDS = (*MODELS, "equal_model_mean")
EXPECTED_POLICY_SCHEMA = "cross_tfg_future_dev20_admission_policy_attempt01_v1"
EXPECTED_PRE_EFFECT_SCHEMA = "cross_tfg_future_dev20_pre_effect_freeze_v1"
EXPECTED_ASR_BASELINE_SCHEMA = "cross_tfg_future_dev20_asr_baseline_results_v1"
EXPECTED_ASR_PAYLOAD_SCHEMA = "cross_tfg_future_dev20_asr_payloads_v1"
EXPECTED_ASR_CANDIDATE_SCHEMA = "cross_tfg_future_dev20_asr_candidate_results_v1"
EXPECTED_ASR_CANDIDATE_PAYLOAD_SCHEMA = "cross_tfg_future_dev20_asr_candidate_payloads_v1"
EXPECTED_ASR_MODEL = "qwen3-asr-flash-2026-02-10"
EXPECTED_VISUAL_SCHEMA = "cross-tfg-visual-motion-guard-v1"
EXPECTED_EVAL_SCHEMA = "cross-tfg-audio-eval-v1"
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)


class AdmissionError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(canonical_json({key: item for key, item in value.items() if key != field})).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_regular(path: Path, *, max_bytes: int = 1 << 34) -> tuple[Path, bytes]:
    if path.is_symlink():
        raise AdmissionError(f"symlink input is not allowed: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise AdmissionError(f"input is not a regular file: {path}")
    size = resolved.stat().st_size
    if size < 0 or size > max_bytes:
        raise AdmissionError(f"input exceeds byte bound ({size}>{max_bytes}): {path}")
    raw = resolved.read_bytes()
    if len(raw) != size:
        raise AdmissionError(f"input changed while being read: {path}")
    return resolved, raw


def read_json(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> tuple[dict[str, Any], bytes, Path]:
    resolved, raw = read_regular(path, max_bytes=max_bytes)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AdmissionError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON root must be an object: {path}")
    return value, raw, resolved


def load_jsonl(path: Path, *, max_bytes: int = 1 << 34) -> tuple[list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    _resolved, raw = read_regular(path, max_bytes=max_bytes)
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            malformed.append({"line": line_number, "reason": "blank_line"})
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except Exception as exc:
            malformed.append({"line": line_number, "reason": f"invalid_json:{type(exc).__name__}"})
            continue
        if not isinstance(value, dict):
            malformed.append({"line": line_number, "reason": "row_not_object"})
            continue
        rows.append(value)
    return rows, raw, malformed


def source_pin(root: Path, rel: Path, expected_sha: str) -> dict[str, Any]:
    path, raw = read_regular(root / rel, max_bytes=256 * 1024 * 1024)
    actual = sha_bytes(raw)
    if actual != expected_sha:
        raise AdmissionError(f"pinned source digest mismatch: {rel}")
    return {"path": rel.as_posix(), "resolved_path": str(path), "bytes": len(raw), "sha256": actual}


def load_pinned_consumer(root: Path) -> Any:
    path = root / CONSUMER_REL
    if sha_bytes(read_regular(path, max_bytes=256 * 1024 * 1024)[1]) != CONSUMER_RAW_SHA256:
        raise AdmissionError("independent consumer source does not match the frozen SHA")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("future_dev20_pinned_consumer", path)
    if spec is None or spec.loader is None:
        raise AdmissionError("cannot construct pinned consumer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_policy(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, raw, _ = read_json(root / POLICY_REL)
    if sha_bytes(raw) != POLICY_RAW_SHA256:
        raise AdmissionError("machine policy raw bytes differ from the analyzer's frozen pin")
    if policy.get("schema") != EXPECTED_POLICY_SCHEMA or policy.get("status") != "FROZEN_PROSPECTIVE_NOT_RUN":
        raise AdmissionError("machine policy schema/status is not the frozen prospective policy")
    if policy.get("self_sha256") != payload_sha(policy, "self_sha256"):
        raise AdmissionError("machine policy embedded payload hash is invalid")
    protocol, protocol_raw, _ = read_json(root / PROTOCOL_REL)
    if sha_bytes(protocol_raw) != PROTOCOL_RAW_SHA256:
        raise AdmissionError("future cohort protocol raw bytes differ from its pin")
    if protocol.get("schema") != "cross_tfg_future_dev20_three_method_protocol_attempt06_v1":
        raise AdmissionError("future cohort source protocol schema differs")
    return policy, protocol


def verify_protocol_rows(protocol: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    design = protocol.get("primary_design")
    cohort = protocol.get("cohort_and_source")
    if not isinstance(design, dict) or not isinstance(cohort, dict):
        raise AdmissionError("future cohort protocol lacks primary_design/cohort_and_source")
    if design.get("total_planned_rows") != 540 or cohort.get("selected_speakers") != list(SPEAKERS):
        raise AdmissionError("future cohort denominator/speaker roster differs from policy")
    if tuple(design.get("renderer_seeds", [])) != SEEDS:
        raise AdmissionError("future cohort seed order differs from policy")
    source_rows = cohort.get("source_rows")
    if not isinstance(source_rows, list) or len(source_rows) != 20:
        raise AdmissionError("future cohort protocol must pin all 20 source rows")
    expected = {str(row.get("sample_id")): row for row in source_rows if isinstance(row, dict)}
    if len(expected) != 20:
        raise AdmissionError("future cohort source sample IDs are missing/duplicated")
    source_speaker_counts = Counter(str(row.get("speaker_id")) for row in source_rows if isinstance(row, dict))
    if set(source_speaker_counts) != set(SPEAKERS) or any(count != 2 for count in source_speaker_counts.values()):
        raise AdmissionError("frozen source rows are not exactly two clips for each of the ten speakers")
    if cohort.get("speaker_counts") != dict(source_speaker_counts):
        raise AdmissionError("protocol speaker-count summary differs from its exact source rows")
    plan_models = {str(item.get("model_id")): item for item in plan.get("models", []) if isinstance(item, dict)}
    protocol_models = {str(item.get("model_id")): item for item in design.get("models", []) if isinstance(item, dict)}
    if set(plan_models) != set(MODELS) or set(protocol_models) != set(MODELS):
        raise AdmissionError("future plan/protocol model roster differs from frozen three-model set")
    for model_id in MODELS:
        if plan_models[model_id].get("audio_frontend") != protocol_models[model_id].get("audio_frontend"):
            raise AdmissionError(f"plan model frontend differs from frozen protocol: {model_id}")
    candidate_id, _ = discover_candidate_method(plan)
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or len(plan_rows) != 540:
        raise AdmissionError("plan must contain exactly 540 planned rows")
    samples: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        if not isinstance(row, dict):
            raise AdmissionError("plan row is not an object")
        sid = row.get("sample_id")
        source = expected.get(sid)
        if source is None:
            raise AdmissionError(f"plan contains unexpected sample: {sid}")
        speaker = source.get("speaker_id")
        natural = row.get("natural_pcm")
        if (
            row.get("split") != "dev" or row.get("speaker_id") != speaker
            or row.get("face_id") != design.get("face_id")
            or not isinstance(natural, dict)
            or natural.get("file_sha256") != source.get("source_wav_sha256")
            or natural.get("pcm_sha256") != source.get("canonical_f32_sha256")
            or natural.get("sample_count") != source.get("sample_count_N")
            or natural.get("sample_rate_hz") != source.get("sample_rate_hz")
            or natural.get("channels") != 1
        ):
            raise AdmissionError(f"plan source identity differs from frozen protocol for {sid}")
        samples[sid] = {"speaker_id": speaker, **source}
    if set(samples) != set(expected):
        raise AdmissionError("plan does not cover the exact 20 source samples")
    job_list = design.get("run_jobs")
    if not isinstance(job_list, list) or len(job_list) != 27:
        raise AdmissionError("protocol must provide the complete ordered 27-job matrix")
    condition_method = {"identity": "identity", "rms_fixed": "rms_fixed", "direct_d0_step120": candidate_id}
    expected_jobs: list[tuple[str, str, int]] = []
    for job in job_list:
        condition = job.get("condition_id")
        method_id = condition_method.get(condition)
        if method_id is None or job.get("model_id") not in MODELS or job.get("seed") not in SEEDS:
            raise AdmissionError("protocol run-job roster contains an undeclared condition/model/seed")
        expected_jobs.append((method_id, str(job["model_id"]), int(job["seed"])))
    expected_job_set = {(method, model, seed) for method in condition_method.values() for model in MODELS for seed in SEEDS}
    if len(expected_jobs) != 27 or len(set(expected_jobs)) != 27 or set(expected_jobs) != expected_job_set:
        raise AdmissionError("protocol run-job matrix is not exactly 3 conditions × 3 models × 3 seeds")
    return samples


def verify_freeze_binding(path: Path, plan: Mapping[str, Any], plan_raw_sha: str, candidate_id: str) -> dict[str, Any]:
    freeze, raw, resolved = read_json(path, max_bytes=1 << 20)
    if freeze.get("schema") != EXPECTED_PRE_EFFECT_SCHEMA or freeze.get("status") != "FROZEN_BEFORE_DEV20_SCORE_ACCESS":
        raise AdmissionError("candidate freeze receipt does not attest the pre-effect frozen state")
    if freeze.get("self_sha256") != payload_sha(freeze, "self_sha256"):
        raise AdmissionError("candidate freeze receipt self-hash is invalid")
    method = next((item for item in plan.get("methods", []) if item.get("method_id") == candidate_id), None)
    if not isinstance(method, dict):
        raise AdmissionError("candidate method absent from frozen plan")
    if (
        freeze.get("plan_raw_sha256") != plan_raw_sha
        or freeze.get("candidate_method_id") != candidate_id
        or freeze.get("method_config_sha256") != method.get("config_sha256")
        or freeze.get("checkpoint_sha256") != method.get("checkpoint_sha256")
        or not isinstance(freeze.get("frozen_commit"), str) or not freeze.get("frozen_commit")
        or not isinstance(freeze.get("frozen_at_utc"), str) or not freeze.get("frozen_at_utc")
        or not SHA_RE.fullmatch(str(freeze.get("checkpoint_sha256", "")))
    ):
        raise AdmissionError("candidate freeze receipt does not bind this plan/method/config/checkpoint")
    return {"path": str(resolved), "bytes": len(raw), "sha256": sha_bytes(raw), "status": freeze["status"]}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalized if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "Z")))


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def _cer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if not ref or not hyp:
        raise AdmissionError("ASR normalized hypotheses must be nonempty")
    return _levenshtein(ref, hyp) / len(ref)


def verify_asr(
    protocol: Mapping[str, Any], source_rows: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any], result_rows: Sequence[Mapping[str, Any]], candidate_method_id: str,
    baseline_summary_path: Path, baseline_calls_path: Path, baseline_payload_path: Path,
    candidate_summary_path: Path, candidate_calls_path: Path, candidate_payload_path: Path,
    baseline_payload_root: Path, candidate_payload_root: Path,
) -> dict[str, Any]:
    base_summary, base_sum_raw, base_sum_resolved = read_json(baseline_summary_path, max_bytes=4 * 1024 * 1024)
    base_payload, base_payload_raw, base_payload_resolved = read_json(baseline_payload_path, max_bytes=16 * 1024 * 1024)
    candidate_summary, candidate_sum_raw, candidate_sum_resolved = read_json(candidate_summary_path, max_bytes=4 * 1024 * 1024)
    candidate_payload, candidate_payload_raw, candidate_payload_resolved = read_json(candidate_payload_path, max_bytes=16 * 1024 * 1024)
    base_calls, base_calls_raw, base_bad = load_jsonl(baseline_calls_path, max_bytes=16 * 1024 * 1024)
    candidate_calls, candidate_calls_raw, candidate_bad = load_jsonl(candidate_calls_path, max_bytes=16 * 1024 * 1024)
    if base_bad or candidate_bad:
        raise AdmissionError(f"ASR call ledgers contain malformed lines: baseline={len(base_bad)}, candidate={len(candidate_bad)}")
    if (
        base_summary.get("schema") != EXPECTED_ASR_BASELINE_SCHEMA
        or base_summary.get("status") != "COMPLETE_60_CALLS_PASS"
        or base_summary.get("planned_calls") != 60 or base_summary.get("calls_recorded") != 60
        or base_summary.get("passed_calls") != 60 or base_summary.get("failed_calls") != 0
        or base_summary.get("model") != EXPECTED_ASR_MODEL or base_summary.get("language") != "zh"
        or base_summary.get("enable_itn") is not False
        or base_summary.get("automatic_retry") is not False
        or base_summary.get("cache_or_deduplication") is not False
        or base_summary.get("results_jsonl_sha256") != sha_bytes(base_calls_raw)
        or base_summary.get("payload_manifest_raw_sha256") != sha_bytes(base_payload_raw)
    ):
        raise AdmissionError("natural/RMS ASR summary does not prove the frozen 60-call protocol")
    if base_payload.get("schema") != EXPECTED_ASR_PAYLOAD_SCHEMA or base_payload.get("status") != "PASS_SOURCE_PINNED_NATURAL_AND_RMS_PCM16_PAYLOADS":
        raise AdmissionError("natural/RMS ASR payload manifest is not the pinned source-only payload contract")
    if base_payload.get("api_calls", {}).get("performed") is not False or base_payload.get("api_calls", {}).get("expected_total") != 60:
        raise AdmissionError("natural/RMS payload manifest call boundary differs")
    if (
        candidate_summary.get("schema") != EXPECTED_ASR_CANDIDATE_SCHEMA
        or candidate_summary.get("status") != "COMPLETE_20_CALLS_PASS"
        or candidate_summary.get("planned_calls") != 20 or candidate_summary.get("calls_recorded") != 20
        or candidate_summary.get("passed_calls") != 20 or candidate_summary.get("failed_calls") != 0
        or candidate_summary.get("model") != EXPECTED_ASR_MODEL or candidate_summary.get("language") != "zh"
        or candidate_summary.get("enable_itn") is not False
        or candidate_summary.get("automatic_retry") is not False
        or candidate_summary.get("cache_or_deduplication") is not False
        or candidate_summary.get("results_jsonl_sha256") != sha_bytes(candidate_calls_raw)
        or candidate_summary.get("payload_manifest_raw_sha256") != sha_bytes(candidate_payload_raw)
    ):
        raise AdmissionError("candidate ASR summary does not prove the frozen 20-call protocol")
    if candidate_payload.get("schema") != EXPECTED_ASR_CANDIDATE_PAYLOAD_SCHEMA or candidate_payload.get("status") != "PASS_20_CANDIDATE_PCM16_PAYLOADS":
        raise AdmissionError("candidate ASR payload manifest has unexpected schema/status")

    base_payload_rows = base_payload.get("rows")
    candidate_payload_rows = candidate_payload.get("rows")
    if not isinstance(base_payload_rows, list) or not isinstance(candidate_payload_rows, list):
        raise AdmissionError("ASR payload manifests lack row arrays")
    base_by_id = {row.get("sample_id"): row for row in base_payload_rows if isinstance(row, dict)}
    cand_by_id = {row.get("sample_id"): row for row in candidate_payload_rows if isinstance(row, dict)}
    expected_samples = set(source_rows)
    if set(base_by_id) != expected_samples or set(cand_by_id) != expected_samples:
        raise AdmissionError("ASR payload sample roster differs from exact 20-clip cohort")

    def verify_pcm16_payload(payload_root: Path, ref: Mapping[str, Any], sample_count: int, where: str) -> None:
        rel_text = ref.get("path")
        if not isinstance(rel_text, str) or not rel_text or "\\" in rel_text:
            raise AdmissionError(f"unsafe or missing ASR payload path: {where}")
        rel = Path(rel_text)
        if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
            raise AdmissionError(f"ASR payload path is not a normalized relative path: {where}")
        root = payload_root.resolve(strict=True)
        path, wav_raw = read_regular(root / rel, max_bytes=16 * 1024 * 1024)
        if root not in path.parents or len(wav_raw) != ref.get("file_bytes") or sha_bytes(wav_raw) != ref.get("file_sha256"):
            raise AdmissionError(f"ASR PCM16 WAV bytes differ from its payload manifest: {where}")
        try:
            with wave.open(io.BytesIO(wav_raw), "rb") as handle:
                params = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes(), handle.getcomptype())
                pcm = handle.readframes(sample_count)
        except Exception as exc:
            raise AdmissionError(f"invalid PCM16 WAV payload {where}: {exc}") from exc
        if params != (1, 2, 16_000, sample_count, "NONE") or len(pcm) != sample_count * 2:
            raise AdmissionError(f"ASR PCM16 payload does not preserve the mono 16-kHz source sample clock: {where}")
        if sha_bytes(pcm) != ref.get("pcm16_sha256"):
            raise AdmissionError(f"ASR PCM16 payload sample bytes do not match manifest: {where}")

    result_pcm: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in result_rows:
        if row.get("status") != "PASS":
            continue
        ref = row.get("generator_pcm")
        if isinstance(ref, dict) and isinstance(ref.get("pcm_sha256"), str):
            result_pcm[(str(row.get("sample_id")), str(row.get("method_id")))].add(ref["pcm_sha256"])
    if any(len(result_pcm.get((sid, method), set())) != 1 for sid in expected_samples for method in ("identity", "rms_fixed", candidate_method_id)):
        raise AdmissionError("generator float32 PCM must be bit-identical across the nine model/seed rows per sample/method")
    for sid in expected_samples:
        sample_count = int(source_rows[sid]["sample_count_N"])
        payload_row = base_by_id[sid]
        natural_ref = payload_row.get("natural_payload")
        rms_ref = payload_row.get("rms_payload")
        if not isinstance(natural_ref, dict) or not isinstance(rms_ref, dict):
            raise AdmissionError(f"baseline payload rows lack natural/RMS WAVs: {sid}")
        verify_pcm16_payload(baseline_payload_root, natural_ref, sample_count, f"{sid}.natural")
        verify_pcm16_payload(baseline_payload_root, rms_ref, sample_count, f"{sid}.rms")
        if natural_ref.get("file_sha256") != source_rows[sid].get("source_wav_sha256"):
            raise AdmissionError(f"natural ASR payload is not original-source byte exact: {sid}")
        rms_pcm = payload_row.get("rms_intermediate", {}).get("pcm_float32_sha256")
        if rms_pcm != next(iter(result_pcm[(sid, "rms_fixed")])):
            raise AdmissionError(f"RMS ASR payload lineage differs from actual planned generator float32 PCM: {sid}")
        cand_ref = cand_by_id[sid]
        verify_pcm16_payload(candidate_payload_root, cand_ref, sample_count, f"{sid}.candidate")
        candidate_pcm = next(iter(result_pcm[(sid, candidate_method_id)]))
        if cand_ref.get("generator_pcm_float32_sha256") != candidate_pcm:
            raise AdmissionError(f"candidate ASR payload does not bind actual generated float32 PCM: {sid}")
        identity_pcm = next(iter(result_pcm[(sid, "identity")]))
        if identity_pcm != source_rows[sid].get("canonical_f32_sha256"):
            raise AdmissionError(f"identity generator PCM differs from canonical natural PCM: {sid}")

    calls_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in base_calls:
        if row.get("schema") != "cross_tfg_future_dev20_asr_baseline_call_v1":
            raise AdmissionError("unexpected baseline ASR call schema")
        condition = row.get("condition")
        replicate = row.get("replicate")
        key = (str(row.get("sample_id")), str(condition), int(replicate) if isinstance(replicate, int) else -1)
        if key in calls_by_key or key[0] not in expected_samples:
            raise AdmissionError(f"duplicate or unknown baseline ASR call: {key}")
        payload_row = base_by_id[key[0]]
        source = source_rows[key[0]]
        ref_key = "natural_payload" if condition == "natural" else "rms_payload" if condition == "rms_fixed" else None
        ref = payload_row.get(ref_key) if ref_key else None
        if (
            row.get("status") != "PASS" or not isinstance(row.get("hypothesis"), str) or not _normalize(row["hypothesis"])
            or row.get("speaker_id") != source.get("speaker_id")
            or row.get("model") != EXPECTED_ASR_MODEL or row.get("language") != "zh" or row.get("enable_itn") is not False
            or row.get("prompt_or_context_sent") is not False or row.get("ground_truth_sent") is not False
            or row.get("sample_count") != source.get("sample_count_N") or row.get("sample_rate_hz") != 16000
            or not isinstance(ref, dict) or row.get("input_wav_sha256") != ref.get("file_sha256")
            or row.get("input_pcm16_sha256") != ref.get("pcm16_sha256")
            or row.get("payload_kind") != ref_key
            or row.get("call_id") != (f"{key[0]}:natural_repeat_0{replicate}" if condition == "natural" else f"{key[0]}:rms_fixed")
        ):
            raise AdmissionError(f"baseline ASR call identity/payload mismatch for {key}")
        if condition == "natural" and row.get("input_wav_sha256") != source.get("source_wav_sha256"):
            raise AdmissionError(f"natural ASR payload is not the exact original PCM16 source: {key[0]}")
        calls_by_key[key] = row
    if len(calls_by_key) != 60 or any((sid, cond, rep) not in calls_by_key for sid in expected_samples for cond, reps in (("natural", (1, 2)), ("rms_fixed", (1,))) for rep in reps):
        raise AdmissionError("baseline ASR call set is not exactly 40 natural + 20 RMS calls")

    candidate_by_sample: dict[str, dict[str, Any]] = {}
    candidate_payload_by_id = cand_by_id
    for row in candidate_calls:
        if row.get("schema") != "cross_tfg_future_dev20_asr_candidate_call_v1":
            raise AdmissionError("unexpected candidate ASR call schema")
        sid = str(row.get("sample_id"))
        if sid in candidate_by_sample or sid not in expected_samples:
            raise AdmissionError(f"duplicate or unknown candidate ASR call: {sid}")
        payload = candidate_payload_by_id[sid]
        natural = source_rows[sid]
        if (
            row.get("status") != "PASS" or row.get("condition") != "direct_d0_step120" or row.get("replicate") != 1
            or row.get("call_id") != f"{sid}:candidate" or row.get("speaker_id") != natural.get("speaker_id")
            or not isinstance(row.get("hypothesis"), str) or not _normalize(row["hypothesis"])
            or row.get("model") != EXPECTED_ASR_MODEL or row.get("language") != "zh" or row.get("enable_itn") is not False
            or row.get("prompt_or_context_sent") is not False or row.get("ground_truth_sent") is not False
            or row.get("input_pcm16_sha256") != payload.get("pcm16_sha256")
            or row.get("input_wav_sha256") != payload.get("file_sha256")
            or row.get("generator_pcm_float32_sha256") != payload.get("generator_pcm_float32_sha256")
            or row.get("sample_count") != natural.get("sample_count_N") or row.get("sample_rate_hz") != 16000
        ):
            raise AdmissionError(f"candidate ASR call identity/payload mismatch: {sid}")
        candidate_by_sample[sid] = row
    if len(candidate_by_sample) != 20 or set(candidate_by_sample) != expected_samples:
        raise AdmissionError("candidate ASR call set is not exactly one per frozen clip")

    per_clip: dict[str, float] = {}
    for sid in expected_samples:
        n1 = calls_by_key[(sid, "natural", 1)]["hypothesis"]
        n2 = calls_by_key[(sid, "natural", 2)]["hypothesis"]
        rms = calls_by_key[(sid, "rms_fixed", 1)]["hypothesis"]
        candidate = candidate_by_sample[sid]["hypothesis"]
        per_clip[sid] = _cer(n1, candidate) - max(_cer(n1, n2), _cer(n1, rms))
    per_speaker: dict[str, float] = {}
    for speaker in SPEAKERS:
        clip_values = [per_clip[sid] for sid, source in source_rows.items() if source.get("speaker_id") == speaker]
        if len(clip_values) != 2:
            raise AdmissionError(f"ASR speaker does not have exactly two clips: {speaker}")
        per_speaker[speaker] = float(np.mean(np.asarray(clip_values, dtype=np.float64), dtype=np.float64))
    mean_excess = float(np.mean(np.asarray(list(per_speaker.values()), dtype=np.float64), dtype=np.float64))
    return {
        "status": "PASS" if mean_excess <= 0.020 and all(value <= 0.100 for value in per_clip.values()) else "FAIL",
        "baseline_summary": {"path": str(base_sum_resolved), "bytes": len(base_sum_raw), "sha256": sha_bytes(base_sum_raw)},
        "baseline_calls": {"path": str(baseline_calls_path.resolve()), "bytes": len(base_calls_raw), "sha256": sha_bytes(base_calls_raw)},
        "baseline_payload_manifest": {"path": str(base_payload_resolved), "bytes": len(base_payload_raw), "sha256": sha_bytes(base_payload_raw)},
        "candidate_summary": {"path": str(candidate_sum_resolved), "bytes": len(candidate_sum_raw), "sha256": sha_bytes(candidate_sum_raw)},
        "candidate_calls": {"path": str(candidate_calls_path.resolve()), "bytes": len(candidate_calls_raw), "sha256": sha_bytes(candidate_calls_raw)},
        "candidate_payload_manifest": {"path": str(candidate_payload_resolved), "bytes": len(candidate_payload_raw), "sha256": sha_bytes(candidate_payload_raw)},
        "planned_call_count": 80,
        "observed_call_count": len(base_calls) + len(candidate_calls),
        "equal_speaker_mean_excess_disagreement": mean_excess,
        "mean_threshold": 0.020,
        "per_clip_excess_disagreement": per_clip,
        "per_clip_max_threshold": 0.100,
        "per_speaker_excess_disagreement": per_speaker,
        "normalization": "NFKC+casefold; remove Unicode whitespace, separators, and punctuation; character Levenshtein / normalized first-hypothesis length",
        "api_calls_performed_by_analyzer": False,
    }


def verify_visual(
    manifest_path: Path, plan: Mapping[str, Any], result_manifest: Mapping[str, Any],
    plan_raw_sha: str, result_manifest_raw_sha: str, planned_rows: Sequence[Mapping[str, Any]],
    result_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest, manifest_raw, resolved_manifest = read_json(manifest_path, max_bytes=16 * 1024 * 1024)
    if (
        manifest.get("schema_version") != EXPECTED_VISUAL_SCHEMA or manifest.get("status") != "PASS"
        or manifest.get("run_id") != plan.get("run_id")
        or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
    ):
        raise AdmissionError("visual guard manifest schema/status/run/self-hash is not PASS")
    base = resolved_manifest.parent
    files = manifest.get("files")
    if not isinstance(files, list):
        raise AdmissionError("visual manifest file inventory is missing")
    refs: dict[str, dict[str, Any]] = {}
    for ref in files:
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or ref["path"] in refs:
            raise AdmissionError("visual manifest file inventory has malformed/duplicate paths")
        refs[ref["path"]] = ref
    required = ("audit.json", "visual_motion_results.jsonl")
    if any(name not in refs for name in required):
        raise AdmissionError("visual manifest omits its audit or row ledger")
    loaded: dict[str, bytes] = {}
    for name in required:
        rel = Path(name)
        target = base / rel
        path, raw = read_regular(target, max_bytes=512 * 1024 * 1024)
        ref = refs[name]
        if ref.get("bytes") != len(raw) or ref.get("sha256") != sha_bytes(raw):
            raise AdmissionError(f"visual manifest does not bind actual {name} bytes")
        loaded[name] = raw
    audit = json.loads(loaded["audit.json"].decode("utf-8"))
    if (
        audit.get("schema_version") != EXPECTED_VISUAL_SCHEMA or audit.get("status") != "PASS"
        or audit.get("run_id") != plan.get("run_id")
        or audit.get("plan_raw_sha256") != plan_raw_sha
        or audit.get("result_manifest_raw_sha256") != result_manifest_raw_sha
        or audit.get("feature_source_sha256") != VISUAL_FEATURE_SHA256
        or audit.get("landmarker_asset_sha256") != LANDMARKER_SHA256
        or audit.get("planned_rows") != 540 or audit.get("result_rows") != 540
        or audit.get("accounted_for_planned_rows") != 540
        or audit.get("visual_status_counts") != {"PASS": 540}
    ):
        raise AdmissionError("visual audit does not bind the complete actual result/plan and pinned extractor")
    rows, bad = _parse_jsonl_bytes(loaded["visual_motion_results.jsonl"])
    if bad or len(rows) != 540:
        raise AdmissionError("visual result ledger is malformed or not exactly 540 rows")
    expected = {str(row.get("row_id")): row for row in planned_rows}
    seen: set[str] = set()
    for observed in rows:
        row_id = observed.get("row_id")
        planned = expected.get(row_id)
        if planned is None or row_id in seen:
            raise AdmissionError("visual rows contain unexpected or duplicate row ID")
        seen.add(row_id)
        if any(observed.get(field) != planned.get(field) for field in IDENTITY_FIELDS):
            raise AdmissionError(f"visual row identity differs from plan: {row_id}")
        if observed.get("status") != "PASS":
            raise AdmissionError(f"visual row is not PASS: {row_id}")
        fraction = observed.get("metrics", {}).get("tracking_valid_fraction")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not math.isfinite(float(fraction)) or float(fraction) < 0.950:
            raise AdmissionError(f"visual tracking gate failed: {row_id}")
        visual_clock = observed.get("render_clock")
        source_result = result_by_id.get(row_id)
        render = source_result.get("render") if isinstance(source_result, Mapping) else None
        if not isinstance(visual_clock, dict) or not isinstance(render, Mapping):
            raise AdmissionError(f"visual row lacks pinned result/render clock peer: {row_id}")
        for field in ("fps_num", "fps_den", "frame_count"):
            if visual_clock.get(field) != render.get(field):
                raise AdmissionError(f"visual clock differs from producer's exact render clock: {row_id}.{field}")
        if visual_clock.get("timestamps_sha256") != render.get("timestamps_sha256"):
            raise AdmissionError(f"visual timestamps identity differs from exact result render: {row_id}")
        video = observed.get("video")
        if not isinstance(video, dict) or not isinstance(render, Mapping) or video.get("file_sha256") != render.get("silent_video_sha256"):
            raise AdmissionError(f"visual row video differs from planned row's exact silent render: {row_id}")
    if seen != set(expected):
        raise AdmissionError("visual rows fail to cover the exact planned 540 IDs")
    return {
        "status": "PASS",
        "manifest": {"path": str(resolved_manifest), "bytes": len(manifest_raw), "sha256": sha_bytes(manifest_raw)},
        "audit": {"bytes": len(loaded["audit.json"]), "sha256": sha_bytes(loaded["audit.json"])},
        "results": {"bytes": len(loaded["visual_motion_results.jsonl"]), "sha256": sha_bytes(loaded["visual_motion_results.jsonl"])},
        "planned_rows": 540,
        "minimum_tracking_valid_fraction": 0.950,
        "extractor_source_sha256": VISUAL_FEATURE_SHA256,
        "landmarker_asset_sha256": LANDMARKER_SHA256,
    }


def _parse_jsonl_bytes(raw: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
        except Exception as exc:
            errors.append({"line": line_number, "reason": type(exc).__name__})
    return rows, errors


def discover_candidate_method(plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    methods = plan.get("methods")
    if not isinstance(methods, list):
        raise AdmissionError("plan method table is missing")
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method in methods:
        if isinstance(method, dict):
            by_kind[str(method.get("kind"))].append(method)
    if len(by_kind.get("identity", [])) != 1 or len(by_kind.get("rms_fixed", [])) != 1 or len(methods) != 3:
        raise AdmissionError("future Dev plan must have exactly identity, fixed-RMS, and one frozen candidate")
    used = {by_kind["identity"][0].get("method_id"), by_kind["rms_fixed"][0].get("method_id")}
    candidates = [method for method in methods if method.get("method_id") not in used]
    if len(candidates) != 1 or not isinstance(candidates[0].get("method_id"), str):
        raise AdmissionError("cannot uniquely identify frozen candidate method")
    return str(candidates[0]["method_id"]), candidates[0]


def validate_native_analysis_protocol(plan: Mapping[str, Any]) -> None:
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise AdmissionError("future plan lacks frozen native evaluation protocol")
    syncnet = protocol.get("syncnet")
    roi = protocol.get("roi")
    if not isinstance(syncnet, dict) or not isinstance(roi, dict):
        raise AdmissionError("future plan lacks SyncNet/ROI policy")
    if (
        syncnet.get("lag_offsets_frames") != list(range(-15, 16))
        or syncnet.get("zero_lag_column") != 15
        or syncnet.get("anchor_column") != 15
        or roi.get("mode") != "native_official"
    ):
        raise AdmissionError("native 31-lag axis, zero-lag anchor, or official ROI differs from frozen admission policy")
    supports = {item.get("support_id"): item for item in plan.get("supports", []) if isinstance(item, dict)}
    if (
        set(supports) != set(SUPPORTS)
        or supports["full"].get("trim_each_end") != 0
        or supports["guard20"].get("trim_each_end") != 20
        or supports["full"].get("minimum_queries") != 1
        or supports["guard20"].get("minimum_queries") != 41
    ):
        raise AdmissionError("FULL/GUARD20 support definitions differ from frozen minimums")


def validate_complete_rows(
    plan: Mapping[str, Any], result_rows: Sequence[Mapping[str, Any]], source_rows: Mapping[str, Mapping[str, Any]], candidate_id: str,
) -> dict[str, Any]:
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != 540 or len(result_rows) != 540:
        raise AdmissionError("full result denominator is not exactly 540 planned/result rows")
    plan_ids = [row.get("row_id") for row in planned]
    result_ids = [row.get("row_id") for row in result_rows]
    if len(set(plan_ids)) != 540 or result_ids != plan_ids or len(set(result_ids)) != 540:
        raise AdmissionError("result row IDs/order differ from exact 540-row plan")
    methods = {"identity", "rms_fixed", candidate_id}
    expected_contexts: set[tuple[str, str, str, str, int]] = set()
    method_counts = Counter()
    result_counts = Counter()
    for p, r in zip(planned, result_rows):
        sid = p.get("sample_id")
        if p.get("method_id") not in methods or p.get("model_id") not in MODELS or p.get("seed") not in SEEDS:
            raise AdmissionError(f"plan includes non-policy method/model/seed: {p.get('row_id')}")
        source = source_rows.get(str(sid))
        if source is None or p.get("speaker_id") != source.get("speaker_id"):
            raise AdmissionError(f"plan speaker/sample pair differs from frozen protocol: {sid}")
        expected_contexts.add((str(sid), str(p["speaker_id"]), str(p["model_id"]), str(p["method_id"]), int(p["seed"])))
        method_counts[(p["model_id"], p["method_id"])] += 1
        result_counts[(r.get("model_id"), r.get("method_id"))] += 1
        for field in IDENTITY_FIELDS:
            if r.get(field) != p.get(field):
                raise AdmissionError(f"result identity differs from plan: {p.get('row_id')}.{field}")
        if r.get("status") != "PASS":
            raise AdmissionError(f"planned row is not PASS; retain denominator, reject admission: {p.get('row_id')}")
    expected_signatures = {
        (sid, str(source_rows[sid]["speaker_id"]), model, method, seed)
        for sid in source_rows for model in MODELS for method in methods for seed in SEEDS
    }
    if len(expected_contexts) != 540 or expected_contexts != expected_signatures or set(method_counts) != {(model, method) for model in MODELS for method in methods}:
        raise AdmissionError("plan is not exact 20×3×3×3 method/model/sample/seed roster")
    if any(method_counts[(model, method)] != 60 or result_counts[(model, method)] != 60 for model in MODELS for method in methods):
        raise AdmissionError("per-model/method rows must be exactly 60 with no denominator shrinkage")
    return {"planned_rows": 540, "result_rows": 540, "status_counts": dict(Counter(str(row.get("status")) for row in result_rows)), "rows_per_model_method": 60}


def summarize_supports(
    plan: Mapping[str, Any], rebuilt: Mapping[str, Mapping[str, Mapping[str, Any]]], candidate_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    contexts: dict[tuple[str, str, str, str, int], dict[str, str]] = defaultdict(dict)
    rows = plan["rows"]
    for row in rows:
        key = (str(row["sample_id"]), str(row["speaker_id"]), str(row["model_id"]), str(row["face_id"]), int(row["seed"]))
        if row["method_id"] in contexts[key]:
            raise AdmissionError(f"duplicate method in paired context: {key}/{row['method_id']}")
        contexts[key][str(row["method_id"])] = str(row["row_id"])
    expected_methods = {"identity", "rms_fixed", candidate_id}
    if len(contexts) != 180:
        raise AdmissionError("expected 180 paired sample/model/face/seed contexts")
    context_records: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {}
    for key, by_method in sorted(contexts.items()):
        if set(by_method) != expected_methods:
            raise AdmissionError(f"context lacks exact three-method roster: {key}")
        full_info: dict[str, Any] = {}
        for support_id in SUPPORTS:
            support_rows = {method: rebuilt.get(row_id, {}).get(support_id) for method, row_id in by_method.items()}
            if any(not isinstance(item, Mapping) for item in support_rows.values()):
                raise AdmissionError(f"support missing from rebuilt official consumer: {key}/{support_id}")
            signatures = {(item.get("support_mask_sha256"), item.get("query_count")) for item in support_rows.values() if isinstance(item, Mapping)}
            if len(signatures) != 1:
                raise AdmissionError(f"paired methods do not share the exact common support: {key}/{support_id}")
            mask_sha, query_count = next(iter(signatures))
            required = 1 if support_id == "full" else 41
            if not isinstance(query_count, int) or query_count < required:
                raise AdmissionError(f"support below frozen minimum: {key}/{support_id} count={query_count}")
            if any(item.get("eligible") is not True for item in support_rows.values()):
                raise AdmissionError(f"consumer marked mandatory support ineligible: {key}/{support_id}")
            full_info[support_id] = {"support_mask_sha256": mask_sha, "query_count": query_count}
            for contrast in CONTRASTS:
                ref_method = "rms_fixed" if contrast == "candidate_minus_rms_fixed" else "identity"
                for metric in METRICS:
                    cand_value = float(support_rows[candidate_id]["metrics"][metric])
                    ref_value = float(support_rows[ref_method]["metrics"][metric])
                    value = (ref_value - cand_value) if metric == "D0" else (cand_value - ref_value)
                    if not math.isfinite(value):
                        raise AdmissionError(f"non-finite paired improvement: {key}/{contrast}/{metric}/{support_id}")
                    k = f"{contrast}|{metric}|{support_id}|{key[2]}"
                    values.setdefault(k, []).append(value)
        context_records.append({"sample_id": key[0], "speaker_id": key[1], "model_id": key[2], "face_id": key[3], "seed": key[4], "supports": full_info})

    # Reaggregate each endpoint in the frozen hierarchy: seed mean per clip,
    # clip mean per speaker, then equal speaker mean within each model.
    endpoint_speakers: dict[str, list[float]] = {}
    for contrast in CONTRASTS:
        for metric in METRICS:
            for support_id in SUPPORTS:
                for model_id in MODELS:
                    key_prefix = f"{contrast}|{metric}|{support_id}|{model_id}"
                    per_context = values.get(key_prefix, [])
                    # Use the immutable plan keys rather than relying on the
                    # append order of a dictionary value list.
                    by_clip: dict[tuple[str, str], list[float]] = defaultdict(list)
                    methods_for_context: dict[tuple[str, str, str, int], dict[str, str]] = defaultdict(dict)
                    for row in rows:
                        if row["model_id"] == model_id:
                            ck = (str(row["sample_id"]), str(row["speaker_id"]), str(row["face_id"]), int(row["seed"]))
                            methods_for_context[ck][str(row["method_id"])] = str(row["row_id"])
                    for (sid, speaker, face, seed), methods in methods_for_context.items():
                        if set(methods) != expected_methods:
                            raise AdmissionError("paired method family incomplete while forming speaker estimand")
                        ref_method = "rms_fixed" if contrast == "candidate_minus_rms_fixed" else "identity"
                        cand_obj = rebuilt[methods[candidate_id]][support_id]["metrics"]
                        ref_obj = rebuilt[methods[ref_method]][support_id]["metrics"]
                        cv, rv = float(cand_obj[metric]), float(ref_obj[metric])
                        delta = rv - cv if metric == "D0" else cv - rv
                        by_clip[(sid, speaker)].append(delta)
                    clip_speaker_values: dict[str, list[float]] = defaultdict(list)
                    for (sid, speaker), seed_values in by_clip.items():
                        if len(seed_values) != 3:
                            raise AdmissionError(f"clip does not carry exact three-seed family: {sid}")
                        clip_speaker_values[speaker].append(float(np.mean(np.asarray(seed_values, dtype=np.float64), dtype=np.float64)))
                    speaker_values: list[float] = []
                    for speaker in SPEAKERS:
                        clip_values = clip_speaker_values.get(speaker, [])
                        if len(clip_values) != 2:
                            raise AdmissionError(f"speaker does not have exactly two clip estimates: {speaker}")
                        speaker_values.append(float(np.mean(np.asarray(clip_values, dtype=np.float64), dtype=np.float64)))
                    endpoint_id = f"{contrast}|{metric}|{model_id}|{support_id}"
                    endpoint_speakers[endpoint_id] = speaker_values
                # Equal-model estimand is formed within each speaker first.
                model_arrays = [np.asarray(endpoint_speakers[f"{contrast}|{metric}|{model}|{support_id}"], dtype=np.float64) for model in MODELS]
                equal_values = np.mean(np.stack(model_arrays, axis=0), axis=0, dtype=np.float64)
                endpoint_speakers[f"{contrast}|{metric}|equal_model_mean|{support_id}"] = equal_values.tolist()
    return context_records, endpoint_speakers


def make_endpoints(speaker_values: Mapping[str, Sequence[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = [f"{contrast}|{metric}|{model}|{support}" for contrast in CONTRASTS for metric in METRICS for model in ENDPOINT_ESTIMANDS for support in SUPPORTS]
    if len(expected) != 48 or set(expected) != set(speaker_values):
        raise AdmissionError("simultaneous family is not the exact frozen set of 48 endpoints")
    index = np.random.Generator(np.random.PCG64(20261009)).integers(0, 10, size=(100_000, 10), dtype=np.int64)
    index_le = np.ascontiguousarray(index, dtype=np.dtype("<i8"))
    index_raw = index_le.tobytes(order="C")
    index_sha = sha_bytes(index_raw)
    speaker_arrays: dict[str, list[float]] = {}
    endpoint_stats: list[dict[str, Any]] = []
    draw_stats: list[np.ndarray] = []
    for endpoint_id in expected:
        vals = np.asarray(speaker_values[endpoint_id], dtype=np.float64)
        if vals.shape != (10,) or not np.isfinite(vals).all():
            raise AdmissionError(f"endpoint lacks ten finite speaker values: {endpoint_id}")
        mean = float(np.mean(vals, dtype=np.float64))
        se = float(np.std(vals, ddof=1, dtype=np.float64) / math.sqrt(10.0))
        if se == 0.0:
            if not np.all(vals == vals[0]):
                raise AdmissionError(f"zero SE endpoint does not have ten identical speaker values: {endpoint_id}")
            stat = np.zeros(index.shape[0], dtype=np.float64)
        else:
            samples = vals[index]
            means = np.mean(samples, axis=1, dtype=np.float64)
            stat = np.abs((means - mean) / se)
        speaker_arrays[endpoint_id] = vals.tolist()
        draw_stats.append(stat)
        endpoint_stats.append({"endpoint_id": endpoint_id, "mean": mean, "standard_error": se, "speaker_values_float64": vals.tolist()})
    maximum = np.max(np.stack(draw_stats, axis=1), axis=1)
    critical = float(np.quantile(maximum, 0.95, method="higher"))
    for row in endpoint_stats:
        se = row["standard_error"]
        if se == 0.0:
            low = high = row["mean"]
        else:
            low = row["mean"] - critical * se
            high = row["mean"] + critical * se
        row.update({"simultaneous_95_ci": [float(low), float(high)], "critical_value": critical})
    return endpoint_stats, {
        "status": "PASS_48_ENDPOINT_SHARED_SPEAKER_MAX_T",
        "bit_generator": "PCG64", "seed": 20261009, "draw_count": 100_000,
        "speaker_order": list(SPEAKERS), "index_shape": [100_000, 10],
        "index_dtype": "<i8", "index_matrix_raw_bytes": len(index_raw),
        "index_matrix_contiguous_little_endian_sha256": index_sha,
        "critical_value_higher_0_95": critical,
        "family_endpoint_count": 48, "speaker_arrays_float64": speaker_arrays,
    }, index_le


def point_gate_summary(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {}
    for row in endpoints:
        contrast, metric, estimand, support = str(row["endpoint_id"]).split("|")
        by_key[(contrast, metric, estimand, support)] = float(row["mean"])
    checks: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for model in MODELS:
            value = by_key[(contrast, "Cbest", model, "full")]
            checks.append({"rule": "full_per_model_Cbest_gt_zero", "contrast": contrast, "model_id": model, "value": value, "pass": value > 0.0})
        equal_full = by_key[(contrast, "Cbest", "equal_model_mean", "full")]
        if contrast == "candidate_minus_rms_fixed":
            checks.append({"rule": "full_equal_model_Cbest_gte_0_050", "contrast": contrast, "value": equal_full, "threshold": 0.050, "pass": equal_full >= 0.050})
        for model in ENDPOINT_ESTIMANDS:
            guard = by_key[(contrast, "Cbest", model, "guard20")]
            checks.append({"rule": "guard20_Cbest_gt_zero", "contrast": contrast, "model_id": model, "value": guard, "pass": guard > 0.0})
            for metric in ("D0", "Canchor"):
                for support in SUPPORTS:
                    value = by_key[(contrast, metric, model, support)]
                    checks.append({"rule": "D0_Canchor_nonnegative", "contrast": contrast, "metric": metric, "model_id": model, "support_id": support, "value": value, "pass": value >= 0.0})
    return {"status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "checks": checks, "failed_count": sum(not row["pass"] for row in checks)}


def _file_ref(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha_bytes(raw)}


def _emit_report(out_dir: Path, report: dict[str, Any], bootstrap_index: np.ndarray | None) -> None:
    bootstrap_npy: bytes | None = None
    if bootstrap_index is not None:
        stream = io.BytesIO()
        np.save(stream, bootstrap_index, allow_pickle=False)
        bootstrap_npy = stream.getvalue()
        report["bootstrap"]["index_matrix_npy_bytes"] = len(bootstrap_npy)
        report["bootstrap"]["index_matrix_npy_sha256"] = sha_bytes(bootstrap_npy)
    report["self_sha256"] = payload_sha(report, "self_sha256")
    raw = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    target = out_dir / "admission_report.json"
    with target.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    if bootstrap_npy is not None:
        target_np = out_dir / "bootstrap_indices.npy"
        with target_np.open("xb") as handle:
            handle.write(bootstrap_npy)
            handle.flush()


def run(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve()
    if out_dir.exists():
        raise AdmissionError(f"output already exists; use a new directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve(strict=True)
    report: dict[str, Any] = {
        "schema": "cross_tfg_future_dev20_admission_analysis_attempt01_v1",
        "status": "INCOMPLETE_NOT_ADMITTED",
        "policy_status": "FROZEN_PROSPECTIVE_NOT_RUN",
        "input_pins": {}, "row_denominator": {}, "gates": {},
        "score_effects_read": False, "confirm_data_or_effects_accessed": False,
        "failures": [], "notes": ["Development-only admission analysis. No Confirm access; no score-based cohort selection."],
    }
    index_array: np.ndarray | None = None
    try:
        policy, protocol = verify_policy(root)
        report["input_pins"]["policy"] = _file_ref(root / POLICY_REL, (root / POLICY_REL).read_bytes())
        report["input_pins"]["protocol"] = source_pin(root, PROTOCOL_REL, PROTOCOL_RAW_SHA256)
        report["input_pins"]["consumer_source"] = source_pin(root, CONSUMER_REL, CONSUMER_RAW_SHA256)
        report["input_pins"]["visual_guard_source"] = source_pin(root, VISUAL_GUARD_REL, VISUAL_GUARD_RAW_SHA256)
        report["input_pins"]["visual_feature_source"] = source_pin(root, Path("scripts/experiments/lrs3_tts_visual_advantage/video_features.py"), VISUAL_FEATURE_SHA256)
        report["input_pins"]["landmarker_asset"] = source_pin(root, Path("checkpoints/mediapipe/face_landmarker.task"), LANDMARKER_SHA256)
        report["input_pins"]["asr_baseline_worker_source"] = source_pin(root, ASR_BASELINE_WORKER_REL, ASR_BASELINE_WORKER_SHA256)

        plan, plan_raw, plan_path = read_json(args.plan, max_bytes=128 * 1024 * 1024)
        results, results_raw, malformed = load_jsonl(args.results, max_bytes=1 << 34)
        result_manifest, manifest_raw, manifest_path = read_json(args.result_manifest, max_bytes=16 * 1024 * 1024)
        report["input_pins"].update({
            "plan": _file_ref(plan_path, plan_raw), "results": _file_ref(args.results, results_raw),
            "result_manifest": _file_ref(manifest_path, manifest_raw),
        })
        counts = Counter(str(row.get("status")) for row in results)
        report["row_denominator"] = {
            "planned_rows": len(plan.get("rows", [])) if isinstance(plan.get("rows"), list) else None,
            "parsed_result_rows": len(results), "malformed_jsonl_lines": malformed,
            "status_counts": {key: int(counts.get(key, 0)) for key in ("PASS", "FAILED", "INCOMPLETE", "OTHER")},
            "all_failures_retained": True,
        }
        if malformed:
            raise AdmissionError("results ledger contains malformed lines; full denominator cannot be verified")
        if plan.get("schema_version") != EXPECTED_EVAL_SCHEMA or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256"):
            raise AdmissionError("future plan schema/status/self-hash is not frozen")
        if result_manifest.get("schema_version") != EXPECTED_EVAL_SCHEMA or result_manifest.get("run_id") != plan.get("run_id"):
            raise AdmissionError("result manifest schema/run differs from plan")
        if result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256"):
            raise AdmissionError("result manifest self-hash is invalid")
        if result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw) or result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw):
            raise AdmissionError("result manifest does not bind the exact plan and result bytes")
        if len(plan.get("rows", [])) != 540 or len(results) != 540:
            raise AdmissionError("future plan/results must preserve the complete 540-row denominator")

        candidate_id, candidate_method = discover_candidate_method(plan)
        validate_native_analysis_protocol(plan)
        source_by_id = verify_protocol_rows(protocol, plan)
        report["gates"]["fixed_candidate"] = verify_freeze_binding(args.pre_effect_freeze, plan, sha_bytes(plan_raw), candidate_id)
        report["input_pins"]["pre_effect_freeze"] = report["gates"]["fixed_candidate"]
        coverage = validate_complete_rows(plan, results, source_by_id, candidate_id)
        report["gates"]["complete_540_rows"] = {"status": "PASS", **coverage}

        asr = verify_asr(
            protocol, source_by_id, plan, results, candidate_id,
            args.asr_baseline_summary, args.asr_baseline_calls,
            args.asr_baseline_payload_manifest, args.asr_candidate_summary,
            args.asr_candidate_calls, args.asr_candidate_payload_manifest,
            args.asr_baseline_payload_root, args.asr_candidate_payload_root,
        )
        report["gates"]["reference_free_asr"] = asr

        if counts.get("PASS", 0) != 540 or counts.get("FAILED", 0) != 0 or counts.get("INCOMPLETE", 0) != 0:
            raise AdmissionError("not all 540 planned renderer/score rows are PASS")
        if result_manifest.get("planned_rows") != 540 or result_manifest.get("result_rows") != 540 or result_manifest.get("status") != "PASS":
            raise AdmissionError("result manifest does not declare the complete 540-row PASS set")
        if asr.get("status") != "PASS":
            raise AdmissionError("reference-free ASR consistency gates failed")

        result_by_id = {str(row["row_id"]): row for row in results}
        visual = verify_visual(args.visual_manifest, plan, result_manifest, sha_bytes(plan_raw), sha_bytes(manifest_raw), plan["rows"], result_by_id)
        report["gates"]["independent_visual_tracking"] = visual
        if visual.get("status") != "PASS":
            raise AdmissionError("visual-motion availability/quality gate failed")

        consumer = load_pinned_consumer(root)
        namespace = argparse.Namespace(
            plan=str(args.plan), results=str(args.results), result_manifest=str(args.result_manifest),
            generation_repeat_qc=str(args.generation_repeat_qc), artifact_root=str(args.artifact_root),
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
        )
        state = consumer._load_and_validate_inputs(namespace)
        state["supports"] = {item["support_id"]: item for item in state["plan"]["supports"]}
        rebuilt = consumer._rebuild_all_supports(state)
        report["input_pins"]["generation_repeat_qc"] = _file_ref(args.generation_repeat_qc, read_regular(args.generation_repeat_qc)[1])
        if state["status_counts"] != {"PASS": 540, "FAILED": 0, "INCOMPLETE": 0}:
            raise AdmissionError("independent consumer did not validate all 540 rows as PASS")
        context_records, speaker_values = summarize_supports(plan, rebuilt, candidate_id)
        report["gates"]["independent_native_pcm_clock_support_consumer"] = {
            "status": "PASS", "consumer_source_sha256": CONSUMER_RAW_SHA256,
            "result_rows": 540, "paired_contexts": len(context_records),
            "all_original_pcm_mux_clock_native_matrix_checks": True,
            "support_contexts": context_records,
        }

        endpoints, bootstrap, index_array = make_endpoints(speaker_values)
        report["score_effects_read"] = True
        points = point_gate_summary(endpoints)
        report["candidate_method"] = {
            "method_id": candidate_id,
            "config_sha256": candidate_method.get("config_sha256"),
            "checkpoint_sha256": candidate_method.get("checkpoint_sha256"),
        }
        report["endpoints"] = endpoints
        report["bootstrap"] = bootstrap
        report["point_admission"] = points
        report["status"] = "PASS_DEVELOPMENT_ADMISSION" if points["status"] == "PASS" else "FAIL_POINT_ADMISSION"
    except Exception as exc:
        report["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        report["status"] = "INCOMPLETE_NOT_ADMITTED"
    _emit_report(out_dir, report, index_array)
    print(json.dumps({"status": report["status"], "out": str(out_dir), "planned_rows": report.get("row_denominator", {}).get("planned_rows"), "failure_count": len(report["failures"])}, sort_keys=True))
    return 0 if report["status"] == "PASS_DEVELOPMENT_ADMISSION" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--generation-repeat-qc", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pre-effect-freeze", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--asr-baseline-summary", type=Path, required=True)
    parser.add_argument("--asr-baseline-calls", type=Path, required=True)
    parser.add_argument("--asr-baseline-payload-manifest", type=Path, required=True)
    parser.add_argument("--asr-baseline-payload-root", type=Path, required=True)
    parser.add_argument("--asr-candidate-summary", type=Path, required=True)
    parser.add_argument("--asr-candidate-calls", type=Path, required=True)
    parser.add_argument("--asr-candidate-payload-manifest", type=Path, required=True)
    parser.add_argument("--asr-candidate-payload-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
