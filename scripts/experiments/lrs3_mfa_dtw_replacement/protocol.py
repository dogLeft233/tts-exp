from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config


class ProtocolError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON root must be an object: {target}")
    return payload


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    _assert_finite(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return sha256_file(target)


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    return write_json(target, payload)


def bound_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config.REPO / path).resolve()


def verify_binding(name: str, binding: Mapping[str, Any]) -> Path:
    try:
        path = bound_path(str(binding["path"]))
        expected = str(binding["sha256"])
    except (KeyError, TypeError) as exc:
        raise ProtocolError(f"malformed binding: {name}") from exc
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        raise ProtocolError(f"invalid SHA-256 binding: {name}")
    if not path.is_file():
        raise ProtocolError(f"bound file is missing: {name}: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ProtocolError(f"bound file hash changed: {name}: {path}")
    return path


def _parent(name: str, path: Path) -> tuple[Path, dict[str, str]]:
    if not path.is_file():
        raise ProtocolError(f"parent artifact missing: {name}: {path}")
    actual = sha256_file(path)
    expected = config.PARENT_HASHES[name]
    if actual != expected:
        raise ProtocolError(f"parent artifact hash changed: {name}")
    return path.resolve(), {"path": str(path.resolve()), "sha256": actual}


def _record_file(name: str, path: str | Path, expected_hash: str) -> dict[str, str]:
    resolved = bound_path(path)
    if not resolved.is_file():
        raise ProtocolError(f"record file missing: {name}: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected_hash):
        raise ProtocolError(f"record file hash mismatch: {name}: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _score_rows(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    scores = manifest.get("scores")
    if not isinstance(scores, list) or len(scores) != config.EXPECTED_RECORD_COUNT * 4:
        raise ProtocolError("historical replacement score matrix is not complete")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scores:
        if not isinstance(row, Mapping):
            raise ProtocolError("historical score row is malformed")
        sample_id = str(row.get("sample_id", ""))
        cell = str(row.get("cell", ""))
        key = (sample_id, cell)
        if key in result or cell not in {"G_M_E_M", "G_M_E_N", "G_N_E_M", "G_N_E_N"}:
            raise ProtocolError(f"historical score matrix has duplicate or unknown cell: {key}")
        for field in ("sync_c", "sync_d"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ProtocolError(f"historical score is not finite: {key} {field}")
        result[key] = dict(row)
    return result


def build_stage00() -> dict[str, Any]:
    strict_path, strict_binding = _parent("strict_protocol", config.STRICT_PROTOCOL)
    replacement_path, replacement_binding = _parent("replacement_manifest", config.REPLACEMENT_MANIFEST)
    alignment_path, alignment_binding = _parent("alignment_manifest", config.ALIGNMENT_MANIFEST)
    analysis_path, analysis_binding = _parent("historical_analysis", config.HISTORICAL_ANALYSIS)
    legacy_path, legacy_binding = _parent("legacy_stage00", config.LEGACY_STAGE00)

    strict = load_json(strict_path)
    replacement = load_json(replacement_path)
    alignment = load_json(alignment_path)
    analysis = load_json(analysis_path)
    legacy = load_json(legacy_path)
    if strict.get("status") != "locked" or strict.get("manifest_type") != "lrs3_mfa3_exploratory_strict_replacement_protocol":
        raise ProtocolError("strict parent protocol identity mismatch")
    if replacement.get("status") != "complete" or replacement.get("manifest_type") != strict.get("manifest_type"):
        raise ProtocolError("replacement parent is not complete")
    if alignment.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825":
        raise ProtocolError("MFA3 alignment parent identity mismatch")
    if len(alignment.get("records", [])) != config.EXPECTED_ALIGNMENT_COUNT:
        raise ProtocolError("MFA3 alignment parent count changed")
    if analysis.get("record_count") != config.EXPECTED_RECORD_COUNT or not analysis.get("engineering_complete"):
        raise ProtocolError("historical analysis is incomplete")
    if legacy.get("status") != "complete" or legacy.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825":
        raise ProtocolError("legacy Stage00 identity mismatch")

    strict_records = strict.get("cohort", {}).get("records", [])
    replacement_records = replacement.get("cohort", {}).get("records", [])
    if len(strict_records) != config.EXPECTED_RECORD_COUNT or len(replacement_records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("frozen parent cohort count changed")
    ordered_ids = [str(row.get("sample_id", "")) for row in strict_records]
    if canonical_sha256(ordered_ids) != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("frozen ordered cohort hash changed")
    replacement_ids = [str(row.get("sample_id", "")) for row in replacement_records]
    if replacement_ids != ordered_ids or replacement.get("cohort", {}).get("ordered_sample_ids_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("replacement cohort does not match strict cohort")

    legacy_by_id = {str(row.get("sample_id", "")): row for row in legacy.get("cohort", {}).get("records", [])}
    alignment_by_id = {str(row.get("sample_id", "")): row for row in alignment.get("records", [])}
    if len(legacy_by_id) != len(legacy.get("cohort", {}).get("records", [])) or len(alignment_by_id) != config.EXPECTED_ALIGNMENT_COUNT:
        raise ProtocolError("parent sample ids are not unique")
    score_by_key = _score_rows(replacement)
    replacement_by_id = {str(row.get("sample_id", "")): row for row in replacement_records}
    if len(replacement_by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("replacement cohort sample ids are not unique")

    records: list[dict[str, Any]] = []
    for strict_row in strict_records:
        sample_id = str(strict_row.get("sample_id", ""))
        source_group = str(strict_row.get("source_group", ""))
        legacy_row = legacy_by_id.get(sample_id)
        alignment_row = alignment_by_id.get(sample_id)
        replacement_row = replacement_by_id.get(sample_id)
        if not legacy_row or not alignment_row or not replacement_row:
            raise ProtocolError(f"missing parent join for {sample_id}")
        if any(str(row.get("source_group", "")) != source_group for row in (legacy_row, alignment_row, replacement_row)):
            raise ProtocolError(f"source-group mismatch for {sample_id}")
        natural_grid = _record_file("natural_textgrid", alignment_row["natural_textgrid"], alignment_row["natural_textgrid_sha256"])
        tts_grid = _record_file("tts_textgrid", alignment_row["tts_textgrid"], alignment_row["tts_textgrid_sha256"])
        natural_audio = _record_file("natural_audio", strict_row["natural_audio"], strict_row["natural_audio_sha256"])
        tts_audio = _record_file("tts_audio", legacy_row["tts_audio"], legacy_row["tts_audio_sha256"])
        face_video = _record_file("face_video", strict_row["face_video"], strict_row["face_video_sha256"])
        linear_audio = _record_file("mfa_linear_audio", strict_row["candidate_audio"], strict_row["candidate_audio_sha256"])
        historical_scores: dict[str, dict[str, Any]] = {}
        for cell in ("G_M_E_M", "G_M_E_N", "G_N_E_M", "G_N_E_N"):
            score = score_by_key.get((sample_id, cell))
            if score is None:
                raise ProtocolError(f"missing historical score cell: {sample_id} {cell}")
            historical_scores[cell] = {
                "sync_c": float(score["sync_c"]),
                "sync_d": float(score["sync_d"]),
                "audio_sha256": str(score["audio_sha256"]),
                "video_sha256": str(score["video_sha256"]),
                "muxed_file": str(score["muxed_file"]),
                "muxed_file_sha256": str(score["verification"]["muxed_file_sha256"]),
                "score_row_sha256": canonical_sha256(score),
            }
        records.append({
            "sample_id": sample_id,
            "source_group": source_group,
            "transcript": strict_row["transcript"],
            "natural_audio": natural_audio,
            "tts_audio": tts_audio,
            "face_video": face_video,
            "natural_samples": int(strict_row["natural_samples"]),
            "tts_samples": int(legacy_row["tts_audio_samples"]),
            "natural_textgrid": natural_grid,
            "tts_textgrid": tts_grid,
            "natural_tokens": alignment_row["natural_tokens"],
            "tts_tokens": alignment_row["tts_tokens"],
            "mfa_linear_audio": linear_audio,
            "mfa_linear_mapping": strict_row["candidate_mapping"],
            "historical_scores": historical_scores,
            "alignment_record_sha256": canonical_sha256(alignment_row),
        })

    assets = {
        "wavlm_checkpoint": {"path": str(config.WAVLM_CHECKPOINT), "sha256": sha256_file(config.WAVLM_CHECKPOINT)},
        "vocoder_checkpoint": {"path": str(config.VOCODER_CHECKPOINT), "sha256": sha256_file(config.VOCODER_CHECKPOINT)},
        "wav2lip_python": {"path": str(config.WAV2LIP_PYTHON), "sha256": sha256_file(config.WAV2LIP_PYTHON)},
        "wav2lip_checkpoint": {"path": str(config.WAV2LIP_CHECKPOINT), "sha256": sha256_file(config.WAV2LIP_CHECKPOINT)},
        "wav2lip_inference": {"path": str(config.WAV2LIP_ROOT / "inference.py"), "sha256": sha256_file(config.WAV2LIP_ROOT / "inference.py")},
        "syncnet_python": {"path": str(config.SYNCNET_PYTHON), "sha256": sha256_file(config.SYNCNET_PYTHON)},
        "syncnet_model": {"path": str(config.SYNCNET_MODEL), "sha256": sha256_file(config.SYNCNET_MODEL)},
        "syncnet_pipeline": {"path": str(config.SYNCNET_ROOT / "run_pipeline.py"), "sha256": sha256_file(config.SYNCNET_ROOT / "run_pipeline.py")},
        "syncnet_score": {"path": str(config.SYNCNET_ROOT / "run_syncnet.py"), "sha256": sha256_file(config.SYNCNET_ROOT / "run_syncnet.py")},
        "ffmpeg": {"path": str(config.FFMPEG), "sha256": sha256_file(config.FFMPEG)},
        "ffprobe": {"path": str(config.FFPROBE), "sha256": sha256_file(config.FFPROBE)},
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_comparison_protocol",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "parents": {
            "strict_protocol": strict_binding,
            "replacement_manifest": replacement_binding,
            "alignment_manifest": alignment_binding,
            "historical_analysis": analysis_binding,
            "legacy_stage00": legacy_binding,
        },
        "cohort": {
            "record_count": len(records),
            "source_group_count": len({row["source_group"] for row in records}),
            "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in records]),
            "records": records,
            "selection_uses_outcomes": False,
            "sealed_validation_test_unvisited": True,
        },
        "dtw_contract": {
            "feature_source": "paired natural/TTS WavLM-L6",
            "distance": "float64 cosine distance",
            "phone_instance_scope": "matched non-silence MFA3 instance only",
            "band_ratio": config.BAND_RATIO,
            "frame_ownership_policy": config.DTW_FRAME_OWNERSHIP_POLICY,
            "frame_support_definition": "each frame has nominal support [center-half_stride, center+half_stride); positive-overlap phone sets may share boundary frames",
            "feature_tail_policy": "support outside the final nominal frame interval is unowned and never extrapolated",
            "endpoint_constrained": True,
            "predecessor_tie_order": ["vertical", "horizontal", "diagonal"],
            "natural_values_in_conditioning": False,
            "silence_policy": "reuse MFA-linear rows exactly",
        },
        "candidate_contract": {
            "repository": "bshall/knn-vc",
            "revision": config.KNN_VC_REVISION,
            "sample_rate_hz": 16000,
            "frame_stride_samples": 320,
            "feature_layer": 6,
            "feature_dim": 1024,
            "prematched_vocoder": True,
            "loudness_normalization": False,
            "exact_length_policy": "right_crop_or_right_zero_pad_only",
            "pcm_policy": "canonical PCM16 little-endian",
        },
        "assets": assets,
        "diagonal_gate": {
            "delta_C": "SyncC(DTW video, DTW audio)-SyncC(MFA-linear video, MFA-linear audio)",
            "delta_D": "SyncD(MFA-linear video, MFA-linear audio)-SyncD(DTW video, DTW audio)",
            "bootstrap": "source_group_cluster_bootstrap",
            "draws": config.BOOTSTRAP_DRAWS,
            "seed": config.SEED,
            "confidence": 0.95,
            "lower_bound_strictly_positive": True,
        },
        "media_access": {
            "fit_media_hashed": True,
            "fit_audio_decoded": False,
            "fit_features_created": False,
            "fit_scores_created": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
    }
    summary = {
        "schema_version": 1,
        "stage_id": "00_protocol",
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(records),
        "source_group_count": len({row["source_group"] for row in records}),
        "ordered_sample_ids_sha256": manifest["cohort"]["ordered_sample_ids_sha256"],
        "media_access": manifest["media_access"],
    }
    decision = {
        "schema_version": 1,
        "stage_id": "00_protocol",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "01_candidates",
        "reason": "parent artifacts and the ordered 133-record fit cohort are hash-bound without feature extraction or scoring",
    }
    return {"manifest": manifest, "summary": summary, "decision": decision}


def run_stage00(output_dir: str | Path = config.STAGE00) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing non-empty Stage 00 directory: {target}")
    artifacts = build_stage00()
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "summary", "decision"):
        write_json_once(target / f"{name}.json", artifacts[name])
    write_json_once(target / "manifest.sha256", {"sha256": sha256_file(target / "manifest.json")})
    return artifacts["summary"]
