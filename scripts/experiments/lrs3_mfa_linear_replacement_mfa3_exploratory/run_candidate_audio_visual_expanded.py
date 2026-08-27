from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
    canonical_pcm_s16le,
    candidate_from_features,
    validate_wavlm_interface,
)
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    build_frame_mapping,
    trace_rows,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    KNN_VC_REVISION,
    PROTOCOL_ID,
    VOCODER_CHECKPOINT,
    WAVLM_CHECKPOINT,
    file_sha256,
    load_json,
    validate_stage00,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_candidate_audio import (
    _assert_exploratory_gpu_ready,
    _read_pcm16,
    _validate_candidate_assets,
    _write_pcm16,
)
from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

REPO = Path(__file__).resolve().parents[3]
STAGE_ID = "02_candidate_audio_visual_expanded_retry3"
EXPECTED_STAGE00_ID = "00_protocol_lock_expanded_retry3"
EXPECTED_STAGE00_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / EXPECTED_STAGE00_ID
    / "manifest.json"
)
EXPECTED_ALIGNMENT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / "01_mfa3_screen_expanded_retry2"
    / "alignment_manifest.json"
)
SAMPLE_RATE = 16_000


def _failure(
    output_dir: Path,
    error: str,
    *,
    gpu_gate: dict[str, Any] | None = None,
    partial_audio: list[str] | None = None,
) -> dict[str, Any]:
    paths = partial_audio or []
    payload = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error": error,
        "candidate_generation_started": gpu_gate is not None,
        "partial_audio_paths": paths,
        "partial_audio_count": len(paths),
        "gpu_gate": gpu_gate,
        "parents": {
            "stage00": {
                "path": str(EXPECTED_STAGE00_PATH),
                "sha256": file_sha256(EXPECTED_STAGE00_PATH) if EXPECTED_STAGE00_PATH.is_file() else None,
            },
            "alignment": {
                "path": str(EXPECTED_ALIGNMENT_PATH),
                "sha256": file_sha256(EXPECTED_ALIGNMENT_PATH) if EXPECTED_ALIGNMENT_PATH.is_file() else None,
            },
        },
    }
    write_json(output_dir / "candidate_failure.json", payload)
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "record_count": 0,
        "expected_record_count": 62,
        "candidate_generation_started": gpu_gate is not None,
        "gpu_used": gpu_gate is not None,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "decision.json",
        {
            "stage_id": STAGE_ID,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "next_allowed_stage": None,
            "reason": error,
        },
    )
    return summary


def run(
    stage00_path: Path,
    alignment_path: Path,
    output_dir: Path,
    *,
    local_knn_vc: Path,
) -> dict[str, Any]:
    stage00_path = stage00_path.resolve()
    alignment_path = alignment_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty candidate output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_gate: dict[str, Any] | None = None
    try:
        stage00 = load_json(stage00_path)
        validate_stage00(
            stage00_path,
            stage00,
            expected_stage_id=EXPECTED_STAGE00_ID,
            expected_path=EXPECTED_STAGE00_PATH,
        )
        _validate_candidate_assets(stage00)
        alignment = load_json(alignment_path)
        if alignment_path != EXPECTED_ALIGNMENT_PATH:
            raise ValueError("visual-expanded candidate requires the frozen expanded alignment path")
        if (
            alignment.get("protocol_id") != PROTOCOL_ID
            or alignment.get("stage_id") != "01_mfa3_screen_expanded_retry2"
            or alignment.get("status") != "partial"
            or alignment.get("clean_record_count") != 62
        ):
            raise ValueError("expanded MFA3 alignment is not the expected complete 62-record screen")
        records = list(alignment.get("records", []))
        if len(records) != 62:
            raise ValueError("expanded MFA3 alignment record count mismatch")
        stage00_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
        if any(str(row["sample_id"]) not in stage00_by_id for row in records):
            raise ValueError("alignment clean records are not a subset of expanded Stage00 pool")
        expected_ids_hash = hashlib.sha256(
            json.dumps(
                [str(row["sample_id"]) for row in records],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if alignment.get("clean_sample_ids_sha256") != expected_ids_hash:
            raise ValueError("alignment clean-record hash mismatch")
        gpu_gate = _assert_exploratory_gpu_ready()
        adapter = WavLMKNNVCAdapter.load_pretrained(
            device="cuda",
            source=local_knn_vc,
            revision=KNN_VC_REVISION,
        )
        validate_wavlm_interface(
            adapter.metadata(),
            revision=KNN_VC_REVISION,
            wavlm_checkpoint_sha256=file_sha256(WAVLM_CHECKPOINT),
            vocoder_checkpoint_sha256=file_sha256(VOCODER_CHECKPOINT),
        )
    except Exception as exc:
        return _failure(output_dir, str(exc), gpu_gate=gpu_gate)

    results: list[dict[str, Any]] = []
    try:
        for row in records:
            sample_id = str(row["sample_id"])
            source = stage00_by_id[sample_id]
            natural_values = _read_pcm16(Path(str(source["natural_audio"])))
            tts_values = _read_pcm16(Path(str(source["tts_audio"])))
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
            trace_path = output_dir / "traces" / f"{sample_id}.json"
            write_json(
                trace_path,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "mapping_stats": mapping_stats,
                    "frames": trace_rows(mapping),
                },
            )
            results.append(
                {
                    "sample_id": sample_id,
                    "source_group": source["source_group"],
                    "natural_audio_sha256": source["natural_audio_sha256"],
                    "tts_audio_sha256": source["tts_audio_sha256"],
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
                }
            )
    except Exception as exc:
        partial = sorted(str(path) for path in (output_dir / "audio").glob("*.wav"))
        return _failure(output_dir, str(exc), gpu_gate=gpu_gate, partial_audio=partial)

    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_visual_expanded_candidate_audio",
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "results": results,
        "gpu_gate": gpu_gate,
        "candidate_contract": stage00["candidate_contract"],
        "assets": stage00["assets"],
        "parents": {
            "stage00": {"path": str(stage00_path), "sha256": file_sha256(stage00_path)},
            "alignment": {"path": str(alignment_path), "sha256": file_sha256(alignment_path)},
        },
        "selection": {
            "rule": "all 62 records from the pre-score frozen clean MFA3 screen",
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "sealed_media_opened": False,
            "source_order_preserved": True,
        },
    }
    write_json(output_dir / "candidate_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "candidate_generation_started": True,
        "gpu_used": True,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "decision.json",
        {
            "stage_id": STAGE_ID,
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "next_allowed_stage": "03_visual_expanded_strict_render",
            "reason": "all clean MFA3 screen records generated exact-length candidate audio",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=EXPECTED_STAGE00_PATH)
    parser.add_argument("--alignment", type=Path, default=EXPECTED_ALIGNMENT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/02_candidate_audio_visual_expanded_retry3",
    )
    parser.add_argument(
        "--local-knn-vc",
        type=Path,
        default=Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358",
    )
    args = parser.parse_args()
    result = run(
        args.stage00.resolve(),
        args.alignment.resolve(),
        args.output.resolve(),
        local_knn_vc=args.local_knn_vc.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
