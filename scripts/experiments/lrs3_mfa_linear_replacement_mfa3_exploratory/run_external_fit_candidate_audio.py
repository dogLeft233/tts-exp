from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

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
    VOCODER_CHECKPOINT,
    WAVLM_CHECKPOINT,
    canonical_sha256,
    file_sha256,
    load_json,
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
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "02_external_fit_candidate_audio_retry4"
STAGE00_ID = "00_external_fit_supplement_protocol_retry4"
ALIGNMENT_ID = "01_external_fit_mfa3_screen_retry4"
ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE00_PATH = ROOT / STAGE00_ID / "manifest.json"
ALIGNMENT_PATH = ROOT / ALIGNMENT_ID / "alignment_manifest.json"
EXPECTED_OUTPUT_DIR = ROOT / STAGE_ID
REUSE_ALIGNMENT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / "01_mfa3_screen_expanded_retry2"
    / "alignment_manifest.json"
)
REUSE_CANDIDATE_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / "02_candidate_audio_visual_expanded_retry3"
    / "candidate_manifest.json"
)
REUSE_CANDIDATE_ROOT = REUSE_CANDIDATE_PATH.parent
EXPECTED_GROUP_COUNT = 23
PREVIOUS_MINIMUM_ELIGIBLE_COUNT = 57


def _binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    resolved = path.resolve()
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ValueError(f"{label} path mismatch")
    if not resolved.is_file() or file_sha256(resolved) != str(binding.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")


def _validate_inputs(
    stage00_path: Path,
    alignment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if stage00_path.resolve() != STAGE00_PATH.resolve():
        raise ValueError("external-fit candidate requires the frozen supplement Stage00 path")
    if alignment_path.resolve() != ALIGNMENT_PATH.resolve():
        raise ValueError("external-fit candidate requires the frozen external-fit alignment path")
    stage00 = load_json(stage00_path)
    if (
        stage00.get("protocol_id") != PROTOCOL_ID
        or stage00.get("stage_id") != STAGE00_ID
        or stage00.get("status") != "complete"
        or stage00.get("engineering_decision") != "GO"
    ):
        raise ValueError("external-fit Stage00 is not a complete engineering GO")
    if (stage00_path.parent / "manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(stage00_path):
        raise ValueError("external-fit Stage00 companion hash mismatch")
    _validate_candidate_assets(stage00)

    alignment = load_json(alignment_path)
    if (
        alignment.get("protocol_id") != PROTOCOL_ID
        or alignment.get("stage_id") != ALIGNMENT_ID
        or alignment.get("engineering_decision") != "GO"
        or alignment.get("scientific_decision") != "not_available"
        or alignment.get("next_allowed_stage") != STAGE_ID
    ):
        raise ValueError("external-fit alignment does not authorize candidate generation")
    _binding(alignment.get("parents", {}).get("stage00", {}), stage00_path, "alignment Stage00")
    if (alignment_path.parent / "alignment_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(alignment_path):
        raise ValueError("external-fit alignment companion hash mismatch")
    records = alignment.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("external-fit alignment clean records are missing")
    sample_ids = [str(row.get("sample_id")) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("external-fit alignment clean IDs are not unique")
    if alignment.get("clean_record_count") != len(records):
        raise ValueError("external-fit alignment clean count mismatch")
    if alignment.get("clean_sample_ids_sha256") != canonical_sha256(sample_ids):
        raise ValueError("external-fit alignment clean ID hash mismatch")
    if int(alignment.get("minimum_eligible_record_count", -1)) <= PREVIOUS_MINIMUM_ELIGIBLE_COUNT:
        raise ValueError("external-fit alignment did not raise the eligibility denominator")
    if alignment.get("clean_source_group_count") != EXPECTED_GROUP_COUNT or alignment.get("missing_clean_groups") != []:
        raise ValueError("external-fit alignment clean group coverage is incomplete")

    stage00_records = stage00.get("cohort", {}).get("records")
    if not isinstance(stage00_records, list):
        raise ValueError("external-fit Stage00 records are missing")
    stage00_by_id = {str(row["sample_id"]): row for row in stage00_records}
    if len(stage00_by_id) != len(stage00_records):
        raise ValueError("external-fit Stage00 IDs are not unique")
    if any(sample_id not in stage00_by_id for sample_id in sample_ids):
        raise ValueError("alignment clean records are not a Stage00 subset")
    if any(stage00_by_id[sample_id].get("protocol_split") != "fit" for sample_id in sample_ids):
        raise ValueError("alignment clean records crossed the fit split")
    for container in (stage00.get("test_lock"), alignment.get("test_lock")):
        if not isinstance(container, Mapping) or container.get("status") != "sealed_unvisited":
            raise ValueError("sealed split lock is missing")
        for key, value in container.items():
            if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
                raise ValueError(f"sealed split lock records access: {key}")
    return stage00, alignment, records, stage00_by_id


def _load_reusable_rows() -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    old_alignment = load_json(REUSE_ALIGNMENT_PATH)
    old_candidate = load_json(REUSE_CANDIDATE_PATH)
    if (
        old_alignment.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
        or old_alignment.get("stage_id") != "01_mfa3_screen_expanded_retry2"
        or old_alignment.get("clean_record_count") != 62
    ):
        raise ValueError("reusable alignment parent identity mismatch")
    if (
        old_candidate.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
        or old_candidate.get("stage_id") != "02_candidate_audio_visual_expanded_retry3"
        or old_candidate.get("status") != "complete"
        or old_candidate.get("engineering_decision") != "GO"
        or old_candidate.get("record_count") != 62
    ):
        raise ValueError("reusable candidate parent identity mismatch")
    _binding(old_candidate.get("parents", {}).get("alignment", {}), REUSE_ALIGNMENT_PATH, "reusable candidate alignment")
    if old_candidate.get("candidate_contract") is None or old_candidate.get("assets") is None:
        raise ValueError("reusable candidate contract or assets are missing")
    old_alignment_rows = old_alignment.get("records")
    old_candidate_rows = old_candidate.get("results")
    if not isinstance(old_alignment_rows, list) or not isinstance(old_candidate_rows, list):
        raise ValueError("reusable candidate rows are missing")
    alignment_by_id = {str(row["sample_id"]): row for row in old_alignment_rows}
    candidate_by_id = {str(row["sample_id"]): row for row in old_candidate_rows}
    if len(alignment_by_id) != 62 or len(candidate_by_id) != 62 or set(alignment_by_id) != set(candidate_by_id):
        raise ValueError("reusable candidate IDs do not match reusable alignment")
    return alignment_by_id, candidate_by_id


def _reusable_result(
    current_alignment: Mapping[str, Any],
    source: Mapping[str, Any],
    old_alignment: Mapping[str, Any] | None,
    old_result: Mapping[str, Any] | None,
    stage00: Mapping[str, Any],
) -> dict[str, Any] | None:
    if old_alignment is None or old_result is None:
        return None
    sample_id = str(current_alignment["sample_id"])
    if canonical_sha256(current_alignment["natural_tokens"]) != canonical_sha256(old_alignment["natural_tokens"]):
        return None
    if canonical_sha256(current_alignment["tts_tokens"]) != canonical_sha256(old_alignment["tts_tokens"]):
        return None
    if str(old_result.get("natural_audio_sha256")) != str(source["natural_audio_sha256"]):
        return None
    if str(old_result.get("tts_audio_sha256")) != str(source["tts_audio_sha256"]):
        return None
    if old_result.get("source_group") != source.get("source_group"):
        return None
    if old_result.get("mapping", {}).get("speech_fallback_frames") != 0:
        return None
    audio_path = Path(str(old_result.get("candidate_audio", ""))).resolve()
    if not audio_path.is_file() or file_sha256(audio_path) != str(old_result.get("candidate_audio_sha256", "")):
        return None
    natural_values = _read_pcm16(Path(str(source["natural_audio"])))
    candidate_values = _read_pcm16(audio_path)
    if candidate_values.size != natural_values.size or int(old_result.get("candidate_samples", -1)) != natural_values.size:
        return None
    validate_wavlm_interface(
        old_result.get("model_interface", {}),
        revision=KNN_VC_REVISION,
        wavlm_checkpoint_sha256=file_sha256(WAVLM_CHECKPOINT),
        vocoder_checkpoint_sha256=file_sha256(VOCODER_CHECKPOINT),
    )
    if stage00.get("candidate_contract") is None:
        raise ValueError("external-fit Stage00 candidate contract is missing")
    trace_path = REUSE_CANDIDATE_ROOT / "traces" / f"{sample_id}.json"
    if not trace_path.is_file() or file_sha256(trace_path) != str(old_result.get("trace_sha256", "")):
        return None
    result = dict(old_result)
    result["natural_textgrid_sha256"] = str(current_alignment["natural_textgrid_sha256"])
    result["tts_textgrid_sha256"] = str(current_alignment["tts_textgrid_sha256"])
    result["reuse"] = {
        "reused": True,
        "candidate_parent": {"path": str(REUSE_CANDIDATE_PATH), "sha256": file_sha256(REUSE_CANDIDATE_PATH)},
        "generation_alignment_parent": {"path": str(REUSE_ALIGNMENT_PATH), "sha256": file_sha256(REUSE_ALIGNMENT_PATH)},
        "generation_natural_textgrid_sha256": str(old_alignment["natural_textgrid_sha256"]),
        "generation_tts_textgrid_sha256": str(old_alignment["tts_textgrid_sha256"]),
        "current_alignment_token_parity": True,
        "trace_path": str(trace_path.resolve()),
    }
    return result


def _failure(
    output: Path,
    error: str,
    *,
    expected_count: int | None,
    gpu_gate: dict[str, Any] | None,
    completed_ids: list[str],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error": error,
        "expected_record_count": expected_count,
        "completed_record_count": len(completed_ids),
        "completed_sample_ids": completed_ids,
        "gpu_gate": gpu_gate,
        "parents": {
            "stage00": {"path": str(STAGE00_PATH), "sha256": file_sha256(STAGE00_PATH) if STAGE00_PATH.is_file() else None},
            "alignment": {"path": str(ALIGNMENT_PATH), "sha256": file_sha256(ALIGNMENT_PATH) if ALIGNMENT_PATH.is_file() else None},
        },
    }
    write_json(output / "candidate_failure.json", payload)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "record_count": len(completed_ids),
        "expected_record_count": expected_count,
        "candidate_generation_started": gpu_gate is not None,
        "gpu_used": gpu_gate is not None,
        "next_allowed_stage": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": None,
        "reason": error,
    })
    return summary


def run(stage00_path: Path, alignment_path: Path, output_dir: Path, *, local_knn_vc: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit candidate output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty external-fit candidate output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gpu_gate: dict[str, Any] | None = None
    expected_count: int | None = None
    completed_ids: list[str] = []
    try:
        stage00, alignment, records, stage00_by_id = _validate_inputs(
            stage00_path.resolve(),
            alignment_path.resolve(),
        )
        expected_count = len(records)
        old_alignment_by_id, old_candidate_by_id = _load_reusable_rows()
        reusable_by_id: dict[str, dict[str, Any]] = {}
        for row in records:
            sample_id = str(row["sample_id"])
            reusable = _reusable_result(
                row,
                stage00_by_id[sample_id],
                old_alignment_by_id.get(sample_id),
                old_candidate_by_id.get(sample_id),
                stage00,
            )
            if reusable is not None:
                reusable_by_id[sample_id] = reusable
        generation_ids = [str(row["sample_id"]) for row in records if str(row["sample_id"]) not in reusable_by_id]
        adapter = None
        if generation_ids:
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
        return _failure(
            output,
            str(exc),
            expected_count=expected_count,
            gpu_gate=gpu_gate,
            completed_ids=completed_ids,
        )

    results: list[dict[str, Any]] = []
    try:
        for row in records:
            sample_id = str(row["sample_id"])
            source = stage00_by_id[sample_id]
            result = reusable_by_id.get(sample_id)
            if result is None:
                if adapter is None:
                    raise RuntimeError("candidate adapter was not loaded for a generation record")
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
                audio_path = output / "audio" / f"{sample_id}.wav"
                _write_pcm16(audio_path, pcm)
                if _read_pcm16(audio_path).size != natural_values.size:
                    raise ValueError(f"candidate PCM readback length mismatch: {sample_id}")
                trace_path = output / "traces" / f"{sample_id}.json"
                write_json(trace_path, {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "mapping_stats": mapping_stats,
                    "frames": trace_rows(mapping),
                })
                result = {
                    "sample_id": sample_id,
                    "source_group": source["source_group"],
                    "natural_audio_sha256": source["natural_audio_sha256"],
                    "tts_audio_sha256": source["tts_audio_sha256"],
                    "natural_textgrid_sha256": row["natural_textgrid_sha256"],
                    "tts_textgrid_sha256": row["tts_textgrid_sha256"],
                    "trace_sha256": file_sha256(trace_path),
                    "candidate_audio": str(audio_path.resolve()),
                    "candidate_audio_sha256": file_sha256(audio_path),
                    "candidate_samples": int(candidate.size),
                    "natural_samples": int(natural_values.size),
                    "mapping": mapping_stats,
                    "candidate": metadata,
                    "pcm_qc": pcm_qc,
                    "model_interface": adapter.metadata(),
                    "reuse": {"reused": False},
                }
            results.append(result)
            completed_ids.append(sample_id)
            write_json(output / "records" / f"{sample_id}.json", result)
    except Exception as exc:
        return _failure(
            output,
            str(exc),
            expected_count=expected_count,
            gpu_gate=gpu_gate,
            completed_ids=completed_ids,
        )

    result_ids = [str(row["sample_id"]) for row in results]
    expected_ids = [str(row["sample_id"]) for row in records]
    if result_ids != expected_ids or len(set(result_ids)) != len(expected_ids):
        return _failure(
            output,
            "candidate result order or uniqueness mismatch",
            expected_count=expected_count,
            gpu_gate=gpu_gate,
            completed_ids=completed_ids,
        )
    reused_count = sum(bool(row.get("reuse", {}).get("reused")) for row in results)
    generated_count = len(results) - reused_count
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_candidate_audio",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "reused_record_count": reused_count,
        "generated_record_count": generated_count,
        "ordered_sample_ids_sha256": canonical_sha256(result_ids),
        "results": results,
        "gpu_gate": gpu_gate,
        "candidate_contract": stage00["candidate_contract"],
        "assets": stage00["assets"],
        "parents": {
            "stage00": {"path": str(stage00_path.resolve()), "sha256": file_sha256(stage00_path)},
            "alignment": {"path": str(alignment_path.resolve()), "sha256": file_sha256(alignment_path)},
            "reuse_alignment": {"path": str(REUSE_ALIGNMENT_PATH), "sha256": file_sha256(REUSE_ALIGNMENT_PATH)},
            "reuse_candidate": {"path": str(REUSE_CANDIDATE_PATH), "sha256": file_sha256(REUSE_CANDIDATE_PATH)},
        },
        "selection": {
            "rule": "all clean records from the frozen external-fit strict MFA3 screen",
            "alignment_clean_filter_required": True,
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "audio_quality_used_for_selection": False,
            "sealed_media_opened": False,
            "source_order_preserved": True,
        },
        "media_access": {
            "fit_audio_opened": True,
            "fit_video_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "syncnet_scores_created": False,
        },
        "test_lock": dict(stage00["test_lock"]),
    }
    write_json(output / "candidate_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "reused_record_count": reused_count,
        "generated_record_count": generated_count,
        "candidate_generation_started": generated_count > 0,
        "gpu_used": generated_count > 0,
        "next_allowed_stage": "03_external_fit_face_preflight_retry4",
        "media_access": manifest["media_access"],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": summary["next_allowed_stage"],
        "reason": "every frozen clean MFA3 record has exact-length candidate audio; token-identical prior candidates were reused without score-based selection",
    })
    (output / "candidate_manifest.sha256").write_text(
        file_sha256(output / "candidate_manifest.json") + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=STAGE00_PATH)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument("--output", type=Path, required=True)
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
    return 0 if result["engineering_decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
