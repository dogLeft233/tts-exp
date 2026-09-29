from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .common import file_sha256, verify_self_hashed_json
from .protocol import ProtocolError


def _load(path: Path) -> dict[str, Any]:
    payload = verify_self_hashed_json(path)
    if payload.get("protocol_id") != config.PROTOCOL_ID:
        raise ProtocolError(f"protocol ID mismatch: {path}")
    return payload


def validate_stage00(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    protocol = _load(root / "00_protocol" / "protocol.json")
    cohort = _load(root / "00_protocol" / "cohort.json")
    access = _load(root / "00_protocol" / "media_access.json")
    if cohort.get("status") != "complete" or cohort.get("record_count") != config.EXPECTED_RECORD_COUNT or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("cohort is incomplete")
    if access.get("cohort_audio_decode_count") != 0 or access.get("cohort_video_decode_count") != 0 or access.get("score_read_count") != 0 or access.get("sealed_media_accessed") is not False or access.get("selection_used_scores") is not False:
        raise ProtocolError("Stage 00 violated read-only preflight")
    checkpoint = Path(str(protocol["checkpoint"]["path"]))
    if file_sha256(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise ProtocolError("checkpoint binding changed")
    return {"status": "valid", "record_count": cohort["record_count"]}


def validate_stage01(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    _load(root / "00_protocol" / "protocol.json")
    manifest = _load(root / "01_audio" / "audio_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio stage is incomplete")
    for row in manifest["rows"]:
        sidecar_path = Path(str(row["dac"]["output_path"])).with_suffix(".json")
        sidecar = verify_self_hashed_json(sidecar_path)
        output = Path(str(sidecar["output_path"]))
        if file_sha256(output) != sidecar["output_sha256"] or sidecar["inference"]["n_quantizers"] is not None or sidecar["input"]["sample_count"] != sidecar["output_qc"]["sample_count"]:
            raise ProtocolError(f"DAC audio provenance mismatch: {row['sample_id']}")
    return {"status": "valid", "record_count": manifest["record_count"]}


def validate_stage02(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    fidelity = _load(root / "02_fidelity" / "fidelity.json")
    if fidelity.get("status") != "complete" or fidelity.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("fidelity stage is incomplete")
    for row in fidelity["rows"]:
        for arm in ("wavlm", "dac"):
            metrics = row[arm]
            if metrics.get("lag_search") or metrics.get("gain_alignment") or metrics.get("phase_alignment"):
                raise ProtocolError(f"fidelity metric corrected waveform: {row['sample_id']}/{arm}")
    return {"status": "valid", "record_count": fidelity["record_count"]}


def validate_stage03(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    videos = _load(root / "03_videos" / "videos_manifest.json")
    if videos.get("status") != "complete" or videos.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("video stage is incomplete")
    for row in videos["rows"]:
        geometries = {str(row["arms"][arm]["geometry_sha256"]) for arm in config.DRIVER_ARMS}
        if len(geometries) != 1:
            raise ProtocolError(f"driver geometry diverged: {row['sample_id']}")
        for arm in config.DRIVER_ARMS:
            sidecar = verify_self_hashed_json(Path(str(row["arms"][arm]["output"])).with_suffix(".json"))
            if sidecar.get("output_sha256") != file_sha256(Path(str(row["arms"][arm]["output"]))):
                raise ProtocolError(f"video hash mismatch: {row['sample_id']}/{arm}")
    return {"status": "valid", "record_count": videos["record_count"]}


def validate_stage04(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    matrix = _load(root / "04_matrix" / "matrix_manifest.json")
    expected = config.EXPECTED_RECORD_COUNT * len(config.MATRIX_CELLS)
    if matrix.get("status") != "complete" or matrix.get("cell_count") != expected:
        raise ProtocolError("matrix is incomplete")
    keys = {(str(row["sample_id"]), str(row["cell"])) for row in matrix["muxes"]}
    if len(keys) != expected or any(not row.get("audio_pcm_verified") or not row.get("video_stream_copy_verified") for row in matrix["muxes"]):
        raise ProtocolError("matrix media identity is invalid")
    return {"status": "valid", "cell_count": expected}


def validate_run(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    validate_stage00(root)
    validate_stage01(root)
    validate_stage02(root)
    validate_stage03(root)
    validate_stage04(root)
    analysis = _load(root / "05_analysis" / "analysis.json")
    final = _load(root / "06_final" / "final.json")
    if analysis.get("status") != "complete" or final.get("status") != "complete":
        raise ProtocolError("terminal stages are incomplete")
    if final.get("scientific_decision") != analysis.get("decisions", {}).get("scientific_decision"):
        raise ProtocolError("final decision differs from analysis")
    return {"status": "valid", "scientific_decision": final["scientific_decision"], "future_tts_alignment_experiment_eligible": final["future_tts_alignment_experiment_eligible"]}
