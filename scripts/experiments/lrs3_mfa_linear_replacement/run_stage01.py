from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .candidate_audio import (
    canonical_pcm_s16le,
    candidate_from_features,
    validate_wavlm_interface,
)
from .mfa_alignment import (
    build_frame_mapping,
    parsed_token_hash,
    parse_textgrid,
    prepare_mfa_corpus,
    run_mfa_alignment,
    transcript_sha256,
    UNKNOWN_LABELS,
)
from .protocol import assert_finite_json, canonical_sha256, file_sha256, load_json, write_json

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_20260824"
EXPECTED_RECORDS = 24
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
SAMPLE_RATE = 16_000


class AlignmentBatchError(ValueError):
    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        self.failures = [dict(failure) for failure in failures]
        super().__init__(f"{len(self.failures)} alignment records failed")


def stage_id_for_output(output_dir: Path) -> str:
    stage_id = output_dir.name
    if not stage_id.startswith("01_candidate_audio_retry"):
        raise ValueError(f"unexpected Stage01 output directory name: {stage_id}")
    return stage_id


def _read_pcm16(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"audio is not canonical 16 kHz mono PCM16: {path}")
        values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values


def _write_pcm16(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _alignment_rows(records: Sequence[Mapping[str, Any]], natural_grid_dir: Path, tts_grid_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        natural_grid = natural_grid_dir / f"{sample_id}.TextGrid"
        tts_grid = tts_grid_dir / f"{sample_id}.TextGrid"
        if not natural_grid.is_file() or not tts_grid.is_file():
            failures.append({"sample_id": sample_id, "side": "both", "error": "missing natural/TTS TextGrid"})
            continue
        natural_tokens = None
        tts_tokens = None
        try:
            natural_tokens = parse_textgrid(natural_grid)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "side": "natural", "error": str(exc), "textgrid_sha256": file_sha256(natural_grid)})
        try:
            tts_tokens = parse_textgrid(tts_grid)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "side": "tts", "error": str(exc), "textgrid_sha256": file_sha256(tts_grid)})
        if natural_tokens is None or tts_tokens is None:
            continue
        transcript = str(record.get("transcript", ""))
        rows.append({
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "natural_audio": str(record["natural_audio"]),
            "tts_audio": str(record["tts_audio"]),
            "natural_audio_sha256": str(record["natural_audio_sha256"]),
            "tts_audio_sha256": str(record["tts_audio_sha256"]),
            "transcript_sha256": transcript_sha256(transcript),
            "tts_transcript_sha256": str(record["tts_transcript_sha256"]),
            "natural_textgrid": str(natural_grid),
            "natural_textgrid_sha256": file_sha256(natural_grid),
            "tts_textgrid": str(tts_grid),
            "tts_textgrid_sha256": file_sha256(tts_grid),
            "natural_tokens": natural_tokens,
            "tts_tokens": tts_tokens,
            "natural_token_sha256": parsed_token_hash(natural_tokens),
            "tts_token_sha256": parsed_token_hash(tts_tokens),
        })
    if failures:
        raise AlignmentBatchError(failures)
    return rows


def _cohort_records_with_transcripts(stage00: Mapping[str, Any]) -> list[dict[str, Any]]:
    if stage00["cohort"].get("ordered_records_sha256") != canonical_sha256(stage00["cohort"]["records"]):
        raise ValueError("Stage00 cohort hash does not reproduce its frozen records")
    parent_files = stage00["parents"].get("parent_files", {})
    records = [dict(record) for record in stage00["cohort"]["records"]]
    source_path = Path(parent_files["source_manifest"]["path"])
    tts_meta_path = Path(parent_files["tts_meta"]["path"])
    if not source_path.is_absolute():
        source_path = REPO / source_path
    if not tts_meta_path.is_absolute():
        tts_meta_path = REPO / tts_meta_path
    if file_sha256(source_path) != parent_files["source_manifest"]["sha256"] or file_sha256(tts_meta_path) != parent_files["tts_meta"]["sha256"]:
        raise ValueError("Stage00 parent manifest hash changed")
    source = load_json(source_path)
    tts_meta = load_json(tts_meta_path)
    source_by_id = {str(row.get("sample_id")): row for row in source.get("records", [])}
    tts_by_id = {str(sample_id): row for sample_id, row in tts_meta.get("results", {}).items()}
    for record in records:
        sample_id = str(record["sample_id"])
        source_row = source_by_id.get(sample_id)
        tts_row = tts_by_id.get(sample_id)
        if not isinstance(source_row, Mapping) or not source_row.get("transcript") or not isinstance(tts_row, Mapping):
            raise ValueError(f"canonical transcript/TTS join missing for {sample_id}")
        if source_row.get("source_group") != record.get("source_group") or source_row.get("natural_audio_sha256") != record.get("natural_audio_sha256"):
            raise ValueError(f"canonical transcript join mismatch for {sample_id}")
        if tts_row.get("source_group") != record.get("source_group") or tts_row.get("canonical_audio_sha256") != record.get("tts_audio_sha256") or tts_row.get("reference_audio_sha256") != record.get("natural_audio_sha256"):
            raise ValueError(f"canonical TTS audio join mismatch for {sample_id}")
        natural_transcript = str(source_row["transcript"])
        tts_transcript = str(tts_row.get("tts_transcript", ""))
        if not tts_transcript or transcript_sha256(natural_transcript) != transcript_sha256(tts_transcript):
            raise ValueError(f"natural/TTS transcript mismatch for {sample_id}")
        record["transcript"] = natural_transcript
        record["tts_transcript"] = tts_transcript
        record["transcript_sha256"] = transcript_sha256(natural_transcript)
        record["tts_transcript_sha256"] = transcript_sha256(tts_transcript)
    return records


def _unknown_alignment_rows(output_dir: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for side in ("natural_textgrids", "tts_textgrids"):
        for path in sorted((output_dir / side).glob("*.TextGrid")):
            raw = path.read_text(encoding="utf-8").lower()
            labels = sorted(label for label in UNKNOWN_LABELS if f'text = "{label}"' in raw)
            if labels:
                failures.append({"side": side, "sample_id": path.stem, "unknown_labels": labels, "textgrid_sha256": file_sha256(path)})
    return failures


def prepare_and_align(stage00: Mapping[str, Any], output_dir: Path, *, stage_id: str, mfa_executable: str) -> list[dict[str, Any]]:
    records = _cohort_records_with_transcripts(stage00)
    if len(records) != EXPECTED_RECORDS:
        raise ValueError("Stage00 cohort is not n=24")
    expected_ids = [str(record["sample_id"]) for record in records]
    natural_input = output_dir / "_mfa_input" / "natural"
    tts_input = output_dir / "_mfa_input" / "tts"
    natural_output = output_dir / "natural_textgrids"
    tts_output = output_dir / "tts_textgrids"
    tts_records = [dict(record) for record in records]
    prepare_mfa_corpus(records, natural_input, audio_key="natural_audio", audio_hash_key="natural_audio_sha256", expected_records=records)
    prepare_mfa_corpus(tts_records, tts_input, audio_key="tts_audio", transcript_key="tts_transcript", audio_hash_key="tts_audio_sha256", expected_records=records)
    run_mfa_alignment(natural_input, natural_output, mfa_executable=mfa_executable, expected_sample_ids=expected_ids)
    run_mfa_alignment(tts_input, tts_output, mfa_executable=mfa_executable, expected_sample_ids=expected_ids)
    rows = _alignment_rows(records, natural_output, tts_output)
    write_json(output_dir / "alignment_manifest.json", {
        "schema_version": 1,
        "stage_id": stage_id,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "record_count": len(rows),
        "ordered_sample_ids_sha256": canonical_sha256(expected_ids),
        "records": rows,
        "mfa": {"executable": mfa_executable, "dictionary": "english_us_mfa", "acoustic_model": "english_mfa"},
    })
    return rows


def _load_adapter(device: str, local_knn_vc: Path, stage00: Mapping[str, Any]):
    from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

    adapter = WavLMKNNVCAdapter.load_pretrained(device=device, source=local_knn_vc, revision=KNN_VC_REVISION)
    validate_wavlm_interface(
        adapter.metadata(),
        revision=KNN_VC_REVISION,
        wavlm_checkpoint_sha256=stage00["assets"]["wavlm"]["sha256"],
        vocoder_checkpoint_sha256=stage00["assets"]["vocoder"]["sha256"],
    )
    return adapter


def generate_candidates(stage00: Mapping[str, Any], alignment_rows: Sequence[Mapping[str, Any]], output_dir: Path, *, device: str, local_knn_vc: Path) -> list[dict[str, Any]]:
    if len(alignment_rows) != EXPECTED_RECORDS:
        raise ValueError("candidate generation requires all 24 alignment rows")
    adapter = _load_adapter(device, local_knn_vc, stage00)
    import torch

    results: list[dict[str, Any]] = []
    for row in alignment_rows:
        sample_id = str(row["sample_id"])
        natural_values = _read_pcm16(row["natural_audio"])
        tts_values = _read_pcm16(row["tts_audio"])
        natural_features = adapter.extract(torch.from_numpy(natural_values).unsqueeze(0)).cpu().numpy()
        tts_features = adapter.extract(torch.from_numpy(tts_values).unsqueeze(0)).cpu().numpy()
        mapping, mapping_stats = build_frame_mapping(
            natural_features.shape[0],
            tts_features.shape[0],
            row["natural_tokens"],
            row["tts_tokens"],
        )
        candidate, metadata = candidate_from_features(
            natural_features=natural_features,
            tts_features=tts_features,
            mapping=mapping,
            natural_audio_samples=natural_values.size,
            vocode=lambda conditioning: adapter.vocode(torch.from_numpy(conditioning)).cpu().numpy(),
        )
        pcm, pcm_qc = canonical_pcm_s16le(candidate, natural_values.size)
        audio_path = output_dir / "audio" / f"{sample_id}.wav"
        _write_pcm16(audio_path, pcm)
        trace = [
            {
                "natural_frame_index": item.natural_frame_index,
                "natural_token_index": item.natural_token_index,
                "natural_label": item.natural_label,
                "natural_silence": item.natural_silence,
                "mapping_type": item.mapping_type,
                "tts_token_index": item.tts_token_index,
                "left_frame_index": item.left_frame_index,
                "right_frame_index": item.right_frame_index,
                "interpolation_alpha": item.interpolation_alpha,
                "fallback_reason": item.fallback_reason,
            }
            for item in mapping
        ]
        trace_path = output_dir / "traces" / f"{sample_id}.json"
        write_json(trace_path, {"sample_id": sample_id, "mapping_stats": mapping_stats, "frames": trace})
        results.append({
            "sample_id": sample_id,
            "source_group": row["source_group"],
            "natural_audio_sha256": row["natural_audio_sha256"],
            "tts_audio_sha256": row["tts_audio_sha256"],
            "natural_textgrid_sha256": row["natural_textgrid_sha256"],
            "tts_textgrid_sha256": row["tts_textgrid_sha256"],
            "trace_sha256": file_sha256(trace_path),
            "candidate_audio": str(audio_path),
            "candidate_audio_sha256": file_sha256(audio_path),
            "candidate_samples": int(candidate.size),
            "natural_samples": int(natural_values.size),
            "mapping": mapping_stats,
            "candidate": metadata,
            "pcm_qc": pcm_qc,
            "model_interface": adapter.metadata(),
        })
    return results


def assert_gpu_ready(*, nvidia_smi: str = "nvidia-smi", local_hour: int | None = None, runner=subprocess.run) -> dict[str, Any]:
    hour = time.localtime().tm_hour if local_hour is None else int(local_hour)
    if hour >= 23 or hour < 8:
        raise RuntimeError("GPU use is forbidden during the registered 23:00-08:00 window")
    result = runner([nvidia_smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit {result.returncode}")
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi output is malformed")
        index, utilization, memory_used, memory_total = (int(float(field)) for field in fields)
        if utilization != 0 or memory_used != 0:
            raise RuntimeError(f"GPU {index} is occupied: utilization={utilization}, memory_used={memory_used}")
        rows.append({"index": index, "utilization_gpu_percent": utilization, "memory_used_mib": memory_used, "memory_total_mib": memory_total})
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return {"checked": True, "gpus": rows, "local_hour": hour}


def run(*, stage00_path: Path, output_dir: Path, mfa_executable: str, align_only: bool, device: str, local_knn_vc: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty Stage01 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_id = stage_id_for_output(output_dir)
    stage00 = load_json(stage00_path)
    if stage00.get("status") != "complete" or stage00.get("decision") != "GO" or stage00.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Stage00 lock is not complete GO for this protocol")
    try:
        alignment_rows = prepare_and_align(stage00, output_dir, stage_id=stage_id, mfa_executable=mfa_executable)
    except Exception as exc:
        unknown_rows = _unknown_alignment_rows(output_dir)
        failure = {"schema_version": 1, "stage_id": stage_id, "status": "blocked", "decision": "BLOCKED", "error": str(exc), "alignment_errors": getattr(exc, "failures", []), "unknown_phone_records": unknown_rows, "unknown_phone_record_count": len({row["sample_id"] for row in unknown_rows}), "candidate_generation_started": False}
        write_json(output_dir / "alignment_failure.json", failure)
        summary = {"schema_version": 1, "stage_id": stage_id, "status": "blocked", "decision": "BLOCKED", "scientific_decision": "not_available", "record_count": 0, "expected_record_count": EXPECTED_RECORDS, "failure": failure, "media_access": {"fit_media_opened": True, "internal_dev_media_opened": False, "validation_media_opened": False, "test_media_opened": False, "gpu_used": False}}
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {"stage_id": stage_id, "engineering_decision": "BLOCKED", "scientific_decision": "not_available", "next_allowed_stage": None, "reason": str(exc)})
        return summary
    if align_only:
        summary = {"schema_version": 1, "stage_id": stage_id, "status": "alignment_complete_candidate_pending", "decision": "BLOCKED", "record_count": len(alignment_rows), "media_access": {"fit_media_opened": True, "internal_dev_media_opened": False, "validation_media_opened": False, "test_media_opened": False, "gpu_used": False}}
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {"stage_id": stage_id, "engineering_decision": "BLOCKED", "scientific_decision": "not_available", "reason": "alignment-only preflight; candidate generation intentionally not run"})
        return summary
    if device != "cuda":
        raise ValueError("candidate generation must use the registered CUDA device")
    gpu_gate = None
    try:
        gpu_gate = assert_gpu_ready()
        results = generate_candidates(stage00, alignment_rows, output_dir, device=device, local_knn_vc=local_knn_vc)
    except Exception as exc:
        partial_audio = sorted(str(path) for path in (output_dir / "audio").glob("*.wav"))
        failure = {"schema_version": 1, "stage_id": stage_id, "status": "blocked", "decision": "BLOCKED", "error": str(exc), "candidate_generation_started": gpu_gate is not None, "partial_audio_paths": partial_audio, "partial_audio_count": len(partial_audio)}
        write_json(output_dir / "candidate_failure.json", failure)
        summary = {"schema_version": 1, "stage_id": stage_id, "status": "blocked", "decision": "BLOCKED", "scientific_decision": "not_available", "record_count": 0, "expected_record_count": EXPECTED_RECORDS, "failure": failure, "gpu_used": gpu_gate is not None, "media_access": {"fit_media_opened": True, "internal_dev_media_opened": False, "validation_media_opened": False, "test_media_opened": False}}
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {"stage_id": stage_id, "engineering_decision": "BLOCKED", "scientific_decision": "not_available", "next_allowed_stage": None, "reason": str(exc)})
        return summary
    failures = [] if len(results) == EXPECTED_RECORDS else [{"error": "not all 24 candidates completed"}]
    status = "complete" if not failures else "incomplete"
    decision = "GO" if status == "complete" else "BLOCKED"
    summary = {"schema_version": 1, "stage_id": stage_id, "status": status, "decision": decision, "scientific_decision": "not_available", "record_count": len(results), "expected_record_count": EXPECTED_RECORDS, "failures": failures, "gpu_used": True, "media_access": {"fit_media_opened": True, "internal_dev_media_opened": False, "validation_media_opened": False, "test_media_opened": False}}
    write_json(output_dir / "candidate_manifest.json", {"schema_version": 1, "stage_id": stage_id, "status": status, "decision": decision, "record_count": len(results), "results": results})
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {"stage_id": stage_id, "engineering_decision": decision, "scientific_decision": "not_available", "next_allowed_stage": "02_wav2lip_strict_replacement_retry1" if decision == "GO" else None})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mfa-executable", default="mfa")
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-knn-vc", type=Path, default=Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358")
    args = parser.parse_args()
    result = run(stage00_path=args.stage00.resolve(), output_dir=args.output.resolve(), mfa_executable=args.mfa_executable, align_only=args.align_only, device=args.device, local_knn_vc=args.local_knn_vc.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("decision") == "GO" or args.align_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
