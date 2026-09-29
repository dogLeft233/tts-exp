from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    assert_not_sealed,
    canonical_json_sha256,
    file_sha256,
    sample_ids_sha256,
    verify_self_hashed_json,
)
from .scoring import expected_cells_for_sample


class ValidationError(RuntimeError):
    pass


def _artifact(path: Path, stage_id: str) -> dict[str, Any]:
    try:
        payload = verify_self_hashed_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid artifact: {path}: {exc}") from exc
    if payload.get("stage_id") != stage_id or payload.get("protocol_id") != config.PROTOCOL_ID:
        raise ValidationError(f"artifact identity mismatch: {path}")
    return payload


def _asset(value: Mapping[str, Any], name: str) -> None:
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    assert_not_sealed(path, config.NO_SEALED_MEDIA_TOKENS)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValidationError(f"bound asset changed: {name}: {path}")


def validate_stage00(root: Path) -> dict[str, Any]:
    protocol = _artifact(root / "00_protocol" / "protocol.json", "00_protocol")
    cohort = _artifact(root / "00_protocol" / "cohort.json", "00_protocol")
    media_access = _artifact(root / "00_protocol" / "media_access.json", "00_protocol")
    if protocol.get("config") != config.FrozenConfig().to_dict():
        raise ValidationError("serialized frozen config changed")
    parents = protocol.get("parents", {})
    expected_parents = {
        "parent_protocol": (config.PARENT_PROTOCOL, config.EXPECTED_PARENT_PROTOCOL_SHA256),
        "parent_replacement": (config.PARENT_REPLACEMENT, config.EXPECTED_PARENT_REPLACEMENT_SHA256),
        "discovery_final": (config.DISCOVERY_FINAL, config.EXPECTED_DISCOVERY_FINAL_SHA256),
    }
    for name, (path, expected) in expected_parents.items():
        binding = parents.get(name)
        if not isinstance(binding, Mapping) or Path(str(binding.get("path"))).resolve() != path.resolve() or binding.get("sha256") != expected:
            raise ValidationError(f"parent binding changed: {name}")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValidationError(f"parent file changed: {name}")
    records = cohort.get("records", [])
    if cohort.get("status") != "complete" or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ValidationError("Stage 00 cohort count is incomplete")
    ids = [str(row.get("sample_id")) for row in records]
    groups = [str(row.get("source_group")) for row in records]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups) or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise ValidationError("Stage 00 cohort identity changed")
    if cohort.get("selection") != "second ordered record per source group from parent protocol cohort" or cohort.get("discovery_disjoint") is not True:
        raise ValidationError("Stage 00 selection rule is invalid")
    for row in records:
        for name in ("natural_audio", "mfa_linear_audio", "face_video"):
            binding = row.get(name)
            if not isinstance(binding, Mapping):
                raise ValidationError(f"missing cohort asset: {row.get('sample_id')}/{name}")
            _asset(binding, name)
        if int(row.get("natural_sample_count", -1)) < config.MIN_AUDIO_SAMPLES:
            raise ValidationError(f"invalid sample count: {row.get('sample_id')}")
    if media_access.get("score_read_count") != 0 or media_access.get("selection_used_scores") is not False or media_access.get("sealed_media_accessed") is not False:
        raise ValidationError("Stage 00 scope ledger is invalid")
    return {"stage": "00_protocol", "status": "valid", "record_count": len(records)}


def validate_stage01(root: Path) -> dict[str, Any]:
    manifest = _artifact(root / "01_audio" / "audio_manifest.json", "01_audio")
    diagnostics = _artifact(root / "01_audio" / "diagnostics.json", "01_audio")
    access = _artifact(root / "01_audio" / "diagnostics_media_access.json", "01_audio")
    if manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or manifest.get("audio_cell_count") != config.EXPECTED_RECORD_COUNT * len(config.ARMS):
        raise ValidationError("Stage 01 audio count is incomplete")
    seen: set[tuple[str, str]] = set()
    for row in manifest.get("rows", []):
        sample_id = str(row.get("sample_id"))
        if tuple(str(cell.get("arm")) for cell in row.get("arms", [])) != config.ARMS:
            raise ValidationError(f"Stage 01 arm order is invalid: {sample_id}")
        for cell in row.get("arms", []):
            arm = str(cell.get("arm"))
            key = (sample_id, arm)
            if arm not in config.ARMS or key in seen:
                raise ValidationError(f"Stage 01 duplicate or unknown cell: {key}")
            seen.add(key)
            path = Path(str(cell.get("output")))
            if not path.is_file() or file_sha256(path) != cell.get("output_sha256"):
                raise ValidationError(f"Stage 01 output changed: {key}")
            _artifact(path.with_suffix(".json"), "01_audio")
    if len(seen) != config.EXPECTED_RECORD_COUNT * len(config.ARMS):
        raise ValidationError("Stage 01 cell identity is incomplete")
    if diagnostics.get("status") != "complete" or diagnostics.get("record_count") != config.EXPECTED_RECORD_COUNT or diagnostics.get("frozen_before_scoring") is not True or diagnostics.get("score_read_count") != 0:
        raise ValidationError("Stage 01 diagnostics freeze contract is invalid")
    if access.get("score_read_count") != 0 or access.get("sealed_media_accessed") is not False:
        raise ValidationError("Stage 01 diagnostics scope ledger is invalid")
    rows = diagnostics.get("rows", [])
    if len(rows) != config.EXPECTED_RECORD_COUNT:
        raise ValidationError("Stage 01 diagnostics record count is incomplete")
    for row in rows:
        if tuple(row.get("arms", {})) != config.ARMS:
            raise ValidationError(f"Stage 01 diagnostics arms are incomplete: {row.get('sample_id')}")
        bridge = row["arms"].get(config.BRIDGE_ARM, {})
        for field in ("progress", "orthogonal_ratio", "mel_mae_to_natural", "mel_mae_to_mfa_linear"):
            if not isinstance(bridge.get(field), (int, float)) or not math.isfinite(float(bridge[field])):
                raise ValidationError(f"Stage 01 movement is invalid: {row.get('sample_id')}/{field}")
    return {"stage": "01_audio", "status": "valid", "audio_cell_count": len(seen), "record_count": len(rows)}


def validate_stage02(root: Path) -> dict[str, Any]:
    manifest = _artifact(root / "02_videos" / "videos_manifest.json", "02_videos")
    if manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or manifest.get("video_count") != config.EXPECTED_VIDEO_COUNT:
        raise ValidationError("Stage 02 count is incomplete")
    count = 0
    for row in manifest.get("rows", []):
        geometry_hash = canonical_json_sha256(row["geometry"])
        if tuple(row.get("arms", {})) != config.ARMS:
            raise ValidationError(f"Stage 02 arms are incomplete: {row.get('sample_id')}")
        for arm, cell in row["arms"].items():
            path = Path(str(cell.get("output")))
            if cell.get("geometry_sha256") != geometry_hash or not path.is_file() or file_sha256(path) != cell.get("output_sha256"):
                raise ValidationError(f"Stage 02 media identity changed: {row.get('sample_id')}/{arm}")
            _artifact(path.with_suffix(".json"), "02_videos")
            count += 1
    if count != config.EXPECTED_VIDEO_COUNT:
        raise ValidationError("Stage 02 video count is incomplete")
    return {"stage": "02_videos", "status": "valid", "video_count": count}


def validate_stage03(root: Path) -> dict[str, Any]:
    manifest = _artifact(root / "03_scores" / "scores_manifest.json", "03_scores")
    if manifest.get("status") != "complete" or manifest.get("cell_count") != config.EXPECTED_CELL_COUNT or len(manifest.get("scores", [])) != config.EXPECTED_CELL_COUNT:
        raise ValidationError("Stage 03 count is incomplete")
    seen: set[tuple[str, str]] = set()
    for row in manifest["scores"]:
        key = (str(row.get("sample_id")), str(row.get("cell")))
        if key in seen or key[1] not in config.MATRIX_CELLS:
            raise ValidationError(f"Stage 03 duplicate or unknown cell: {key}")
        if not all(isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field])) for field in ("sync_c", "sync_d", "av_offset")):
            raise ValidationError(f"Stage 03 score is invalid: {key}")
        _artifact(Path(str(root / "03_scores" / "scores" / key[1].split("/")[0].removeprefix("V_") / key[1].split("/")[1].removeprefix("A_") / f"{key[0]}.json")), "03_scores")
        seen.add(key)
    if len({key[0] for key in seen}) != config.EXPECTED_RECORD_COUNT or len(seen) != config.EXPECTED_CELL_COUNT:
        raise ValidationError("Stage 03 score identity is incomplete")
    for sample_id in {key[0] for key in seen}:
        required = {(sample_id, item["cell"]) for item in expected_cells_for_sample(sample_id)}
        if required != {key for key in seen if key[0] == sample_id}:
            raise ValidationError(f"Stage 03 per-record matrix is incomplete: {sample_id}")
    return {"stage": "03_scores", "status": "valid", "cell_count": len(seen)}


def validate_stage04(root: Path) -> dict[str, Any]:
    final = _artifact(root / "04_final" / "final.json", "04_final")
    decision = str(final.get("scientific_decision"))
    allowed = {"BLOCKED", "CONTROL_FAILED", "NATURAL_TO_TTS_BRIDGE_CONFIRMED", "NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED"}
    if decision not in allowed:
        raise ValidationError("invalid terminal decision")
    if final.get("engineering_decision") == "BLOCKED":
        if decision != "BLOCKED" or final.get("reference_conditioned_audio_head_spec_eligible") is not False:
            raise ValidationError("blocked terminal result is invalid")
        return {"stage": "04_final", "status": "valid", "scientific_decision": decision}
    analysis = _artifact(root / "04_final" / "analysis.json", "04_analysis")
    if analysis.get("decisions", {}).get("scientific_decision") != decision:
        raise ValidationError("analysis and terminal decisions differ")
    if final.get("engineering_decision") != "GO" or final.get("record_count") != config.EXPECTED_RECORD_COUNT or final.get("cell_count") != config.EXPECTED_CELL_COUNT:
        raise ValidationError("terminal engineering result is invalid")
    expected_eligibility = decision == "NATURAL_TO_TTS_BRIDGE_CONFIRMED"
    if final.get("reference_conditioned_audio_head_spec_eligible") is not expected_eligibility:
        raise ValidationError("terminal eligibility mismatch")
    return {"stage": "04_final", "status": "valid", "scientific_decision": decision}


def validate_run(root: Path = config.RUN_ROOT) -> dict[str, Any]:
    results = [validate_stage00(root), validate_stage01(root), validate_stage02(root), validate_stage03(root), validate_stage04(root)]
    return {"status": "valid", "stages": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=config.RUN_ROOT)
    args = parser.parse_args()
    print(validate_run(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
