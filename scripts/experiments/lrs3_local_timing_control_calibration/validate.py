from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    CalibrationError,
    assert_not_sealed,
    file_sha256,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .control import construct_control, pcm16_to_float, read_pcm16
from .render import runtime_environment
from .scoring import parse_syncnet, runtime_bindings, verify_mux
from .stats import cluster_bootstrap


class ValidationError(RuntimeError):
    pass


EXPECTED_WAV2LIP_BINDINGS = {
    "wav2lip_python": config.WAV2LIP_PYTHON_SHA256,
    "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT_SHA256,
    "wav2lip_inference": config.WAV2LIP_INFERENCE_SHA256,
}


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}


def _artifact(path: Path, stage_id: str) -> dict[str, Any]:
    try:
        payload = verify_self_hashed_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid artifact: {path}: {exc}") from exc
    if payload.get("stage_id") != stage_id or payload.get("protocol_id") != config.PROTOCOL_ID:
        raise ValidationError(f"artifact identity mismatch: {path}")
    return payload


def _asset(value: Mapping[str, Any], name: str) -> Path:
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    assert_not_sealed(path)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValidationError(f"bound asset changed: {name}: {path}")
    return path


def validate_audit(root: Path) -> dict[str, Any]:
    audit = _artifact(root / "00_audit/audit.json", "00_audit")
    if audit.get("audit_decision") not in {"NO_DEFECT_FOUND", "DEFECT_FOUND", "INCONCLUSIVE"}:
        raise ValidationError("audit decision is invalid")
    if audit.get("record_count") != config.EXPECTED_RECORD_COUNT or audit.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT or audit.get("new_score_read_count") != 0:
        raise ValidationError("audit scope/count is invalid")
    items = audit.get("items")
    summary = audit.get("summary")
    if not isinstance(items, list) or not isinstance(summary, Mapping):
        raise ValidationError("audit items or summary is missing")
    counts = {name: sum(str(item.get("status", "")).lower() == name for item in items) for name in ("pass", "fail", "inconclusive")}
    if dict(summary) != counts:
        raise ValidationError("audit summary does not match its items")
    expected_decision = "INCONCLUSIVE" if counts["inconclusive"] else "DEFECT_FOUND" if counts["fail"] else "NO_DEFECT_FOUND"
    if audit.get("audit_decision") != expected_decision:
        raise ValidationError("audit decision does not match item evidence")
    return {"stage": "00_audit", "status": "valid", "audit_decision": expected_decision, "item_count": len(items)}


def validate_protocol(root: Path, audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = _artifact(root / "01_protocol/protocol.json", "01_protocol")
    if protocol.get("status") != "locked" or protocol.get("config") != config.FrozenConfig().to_dict():
        raise ValidationError("protocol is not the registered frozen protocol")
    source_hashes = protocol.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 for name, digest in source_hashes.items())
    ):
        raise ValidationError("protocol source hash snapshot is invalid")
    branch = str(protocol.get("branch", ""))
    control_arm = str(protocol.get("control_arm", ""))
    if branch not in config.BRANCHES or control_arm != config.control_arm_for_branch(branch) or protocol.get("arms") != list(config.arms_for_branch(branch)) or protocol.get("matrix_cells") != list(config.matrix_cells(control_arm)):
        raise ValidationError("protocol branch or matrix is invalid")
    audit = audit or _artifact(root / "00_audit/audit.json", "00_audit")
    if protocol.get("audit_sha256") != audit.get("artifact_sha256"):
        raise ValidationError("protocol is not bound to the audit artifact")
    history_hashes = {
        "final": config.HISTORY_FINAL_SHA256,
        "cohort": config.HISTORY_COHORT_SHA256,
        "audio_manifest": config.HISTORY_AUDIO_SHA256,
        "videos_manifest": config.HISTORY_VIDEOS_SHA256,
        "scores_manifest": config.HISTORY_SCORES_SHA256,
    }
    for name, expected in history_hashes.items():
        history = protocol.get("history")
        binding = history.get(name) if isinstance(history, Mapping) else None
        if not isinstance(binding, Mapping) or str(binding.get("sha256")) != expected:
            raise ValidationError(f"protocol historical binding is invalid: {name}")
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or file_sha256(path) != expected:
            raise ValidationError(f"historical binding changed: {name}")
    return {"stage": "01_protocol", "status": "valid", "branch": branch, "control_arm": control_arm}


def validate_cohort(root: Path) -> dict[str, Any]:
    cohort = _artifact(root / "01_protocol/cohort.json", "01_protocol")
    records = cohort.get("records")
    if cohort.get("status") != "complete" or not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ValidationError("cohort count is invalid")
    ids = [str(row.get("sample_id", "")) for row in records]
    groups = [str(row.get("source_group", "")) for row in records]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups) or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256 or cohort.get("sample_ids_sha256") != config.EXPECTED_SAMPLE_ID_SHA256:
        raise ValidationError("cohort identity is invalid")
    for row in records:
        for name in ("natural_audio", "mfa_linear_audio", "face_video"):
            if not isinstance(row.get(name), Mapping):
                raise ValidationError(f"cohort asset is missing: {row.get('sample_id')}/{name}")
            _asset(row[name], name)
    return {"stage": "01_protocol", "status": "valid", "record_count": len(records)}


def validate_audio(root: Path, cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _artifact(root / "02_audio/audio_manifest.json", "02_audio")
    branch = str(protocol["branch"])
    arms = config.arms_for_branch(branch)
    if manifest.get("status") != "complete" or manifest.get("branch") != branch or manifest.get("arms") != list(arms) or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or manifest.get("audio_cell_count") != config.expected_video_count(branch) or manifest.get("frozen_before_scoring") is not True or manifest.get("score_read_count") != 0:
        raise ValidationError("audio manifest is incomplete or mutable")
    audio_rows = {str(row.get("sample_id")): row for row in manifest.get("rows", [])}
    if len(audio_rows) != config.EXPECTED_RECORD_COUNT:
        raise ValidationError("audio row identity is incomplete")
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        row = audio_rows.get(sample_id)
        if row is None or tuple(str(item.get("arm")) for item in row.get("arms", [])) != arms:
            raise ValidationError(f"audio arm identity is invalid: {sample_id}")
        natural, natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
        if natural_meta["pcm_sha256"] != record["natural_audio"]["sha256"]:
            raise ValidationError(f"natural audio changed: {sample_id}")
        for item in row["arms"]:
            arm = str(item["arm"])
            output = Path(str(item["output"]))
            if not output.is_file() or file_sha256(output) != item.get("output_sha256"):
                raise ValidationError(f"audio output changed: {sample_id}/{arm}")
            sidecar = _artifact(output.with_suffix(".json"), "02_audio")
            values, metadata = read_pcm16(output)
            if metadata["pcm_sha256"] != item.get("output_sha256"):
                raise ValidationError(f"audio output changed: {sample_id}/{arm}")
            if sidecar.get("branch") != branch or sidecar.get("sample_id") != sample_id or sidecar.get("arm") != arm or sidecar.get("protocol_sha256") != file_sha256(root / "01_protocol/protocol.json") or sidecar.get("natural_audio_sha256") != natural_meta["pcm_sha256"]:
                raise ValidationError(f"audio sidecar identity is invalid: {sample_id}/{arm}")
            if arm in config.BASE_ARMS and values.tobytes() != natural.tobytes():
                raise ValidationError(f"natural copy is not byte-identical: {sample_id}/{arm}")
            if arm == config.control_arm_for_branch(branch):
                expected, _ = construct_control(natural, branch)
                if values.tobytes() != expected.tobytes():
                    raise ValidationError(f"control transform changed: {sample_id}/{arm}")
            if values.size != natural.size or not np.isfinite(pcm16_to_float(values)).all():
                raise ValidationError(f"audio output shape or values are invalid: {sample_id}/{arm}")
    return {"stage": "02_audio", "status": "valid", "audio_cell_count": manifest["audio_cell_count"]}


def validate_videos(root: Path, cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _artifact(root / "03_videos/videos_manifest.json", "03_videos")
    branch = str(protocol["branch"])
    arms = config.arms_for_branch(branch)
    if manifest.get("status") != "complete" or manifest.get("branch") != branch or manifest.get("arms") != list(arms) or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or manifest.get("video_count") != config.expected_video_count(branch):
        raise ValidationError("video manifest is incomplete")
    audio_manifest = _artifact(root / "02_audio/audio_manifest.json", "02_audio")
    audio_by_id = {str(row.get("sample_id")): row for row in audio_manifest["rows"]}
    rows = {str(row.get("sample_id")): row for row in manifest.get("rows", [])}
    if len(rows) != config.EXPECTED_RECORD_COUNT:
        raise ValidationError("video row identity is incomplete")
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        row = rows.get(sample_id)
        if row is None or tuple(str(arm) for arm in row.get("arms", {})) != arms:
            raise ValidationError(f"video arm identity is invalid: {sample_id}")
        face = _asset(record["face_video"], "face video")
        if row.get("face_sha256") != file_sha256(face):
            raise ValidationError(f"video face hash changed: {sample_id}")
        audio_by_arm = {str(item.get("arm")): item for item in audio_by_id[sample_id]["arms"]}
        for arm, item in row["arms"].items():
            output = Path(str(item.get("output")))
            sidecar = _artifact(output.with_suffix(".json"), "03_videos")
            if not output.is_file() or file_sha256(output) != item.get("output_sha256"):
                raise ValidationError(f"video output changed: {sample_id}/{arm}")
            expected_work_dir = root / "03_videos/work" / arm / sample_id
            if sidecar.get("branch") != branch or sidecar.get("sample_id") != sample_id or sidecar.get("arm") != arm or sidecar.get("protocol_sha256") != file_sha256(root / "01_protocol/protocol.json") or sidecar.get("face_sha256") != file_sha256(face) or sidecar.get("audio_sha256") != audio_by_arm[arm].get("output_sha256") or sidecar.get("checkpoint_sha256") != config.WAV2LIP_CHECKPOINT_SHA256 or sidecar.get("runtime_bindings") != EXPECTED_WAV2LIP_BINDINGS or sidecar.get("runtime_environment") != runtime_environment(expected_work_dir):
                raise ValidationError(f"video sidecar binding is invalid: {sample_id}/{arm}")
            command = [str(value) for value in sidecar.get("command", [])]
            if "--face" not in command or Path(command[command.index("--face") + 1]).resolve() != face.resolve() or "--audio" not in command or Path(command[command.index("--audio") + 1]).resolve() != Path(str(audio_by_arm[arm]["output"])).resolve():
                raise ValidationError(f"video command binding is invalid: {sample_id}/{arm}")
    return {"stage": "03_videos", "status": "valid", "video_count": manifest["video_count"]}


def validate_scores(root: Path, cohort: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _artifact(root / "04_scores/scores_manifest.json", "04_scores")
    branch = str(protocol["branch"])
    control_arm = str(protocol["control_arm"])
    cells = config.matrix_cells(control_arm)
    expected = config.expected_cell_count(branch)
    if manifest.get("status") != "complete" or manifest.get("branch") != branch or manifest.get("control_arm") != control_arm or manifest.get("cells") != list(cells) or manifest.get("cell_count") != expected or len(manifest.get("scores", [])) != expected or len(manifest.get("muxes", [])) != expected:
        raise ValidationError("score manifest is incomplete")
    try:
        bindings = runtime_bindings()
    except (OSError, CalibrationError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc
    audio_manifest = _artifact(root / "02_audio/audio_manifest.json", "02_audio")
    video_manifest = _artifact(root / "03_videos/videos_manifest.json", "03_videos")
    audio_by_id = {str(row.get("sample_id")): row for row in audio_manifest["rows"]}
    video_by_id = {str(row.get("sample_id")): row for row in video_manifest["rows"]}
    score_rows = {(str(row.get("sample_id")), str(row.get("cell"))): row for row in manifest["scores"]}
    mux_rows = {(str(row.get("sample_id")), str(row.get("cell"))): row for row in manifest["muxes"]}
    if len(score_rows) != expected or len(mux_rows) != expected:
        raise ValidationError("score cell identity is incomplete")
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        audio_by_arm = {str(item.get("arm")): item for item in audio_by_id[sample_id]["arms"]}
        video_by_arm = video_by_id[sample_id]["arms"]
        expected_for_record = {(sample_id, cell) for cell in cells}
        if {key for key in score_rows if key[0] == sample_id} != expected_for_record:
            raise ValidationError(f"per-record score matrix is incomplete: {sample_id}")
        for _, cell in sorted(expected_for_record):
            score = score_rows[(sample_id, cell)]
            mux_row = mux_rows[(sample_id, cell)]
            video_arm = cell.split("/")[0].removeprefix("V_")
            audio_arm = cell.split("/")[1].removeprefix("A_")
            video = Path(str(video_by_arm[video_arm]["output"]))
            audio = Path(str(audio_by_arm[audio_arm]["output"]))
            mux = Path(str(score["media"]))
            score_sidecar = _artifact(root / "04_scores/scores" / video_arm / audio_arm / f"{sample_id}.json", "04_scores")
            mux_sidecar = _artifact(mux.with_suffix(".json"), "04_scores")
            if not all(isinstance(score.get(field), (int, float)) and math.isfinite(float(score[field])) for field in ("sync_c", "sync_d", "av_offset")):
                raise ValidationError(f"score value is invalid: {sample_id}/{cell}")
            if score_sidecar.get("media_sha256") != file_sha256(mux) or score_sidecar.get("reference") != f"{sample_id}__{video_arm}_{audio_arm}":
                raise ValidationError(f"score sidecar identity is invalid: {sample_id}/{cell}")
            if score_sidecar.get("syncnet_model_sha256") != bindings["syncnet_model"] or score_sidecar.get("syncnet_python_sha256") != bindings["syncnet_python"] or score_sidecar.get("min_track") != config.MIN_TRACK or score_sidecar.get("runtime_bindings") != bindings:
                raise ValidationError(f"SyncNet runtime binding is invalid: {sample_id}/{cell}")
            if mux_row.get("video_source_sha256") != file_sha256(video) or mux_row.get("audio_source_sha256") != file_sha256(audio) or mux_sidecar.get("sha256") != file_sha256(mux):
                raise ValidationError(f"mux source identity is invalid: {sample_id}/{cell}")
            parsed = parse_syncnet(Path(str(score["score_log"])))
            if any(parsed[field] != score[field] for field in ("sync_c", "sync_d", "av_offset")):
                raise ValidationError(f"raw SyncNet log differs from score row: {sample_id}/{cell}")
            verify_mux(video, audio, mux)
    return {"stage": "04_scores", "status": "valid", "cell_count": expected}


def _recompute_controls(cohort: Mapping[str, Any], protocol: Mapping[str, Any], scores: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    control_arm = str(protocol["control_arm"])
    by_key = {(str(row["sample_id"]), str(row["cell"])): row for row in scores["scores"]}
    groups = [str(record["source_group"]) for record in cohort["records"]]
    baseline_offsets: list[int] = []
    repeat_c: list[float] = []
    repeat_d: list[float] = []
    own_c: list[float] = []
    own_d: list[float] = []
    damage_c: list[float] = []
    damage_d: list[float] = []
    repeat_offsets: list[int] = []
    own_offsets: list[int] = []
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        baseline = by_key[(sample_id, "V_N/A_N")]
        repeat = by_key[(sample_id, "V_N_REPEAT/A_N")]
        own = by_key[(sample_id, f"V_{control_arm}/A_{control_arm}")]
        replacement = by_key[(sample_id, f"V_{control_arm}/A_N")]
        baseline_offsets.append(int(baseline["av_offset"]))
        repeat_c.append(float(repeat["sync_c"]) - float(baseline["sync_c"]))
        repeat_d.append(float(baseline["sync_d"]) - float(repeat["sync_d"]))
        own_c.append(float(own["sync_c"]) - float(baseline["sync_c"]))
        own_d.append(float(baseline["sync_d"]) - float(own["sync_d"]))
        damage_c.append(float(own["sync_c"]) - float(replacement["sync_c"]))
        damage_d.append(float(replacement["sync_d"]) - float(own["sync_d"]))
        repeat_offsets.append(int(repeat["av_offset"]))
        own_offsets.append(int(own["av_offset"]))

    def metric(values: list[float], field: str) -> dict[str, Any]:
        result = cluster_bootstrap(values, groups)
        result["field"] = field
        result["wins"] = {"positive_count": int(sum(value > 0 for value in values)), "record_count": len(values)}
        return result

    def offset(values: list[int]) -> dict[str, Any]:
        differences = [abs(value - base) for value, base in zip(values, baseline_offsets)]
        count = sum(value <= config.OFFSET_TOLERANCE_FRAMES for value in differences)
        return {"tolerance_frames": config.OFFSET_TOLERANCE_FRAMES, "count": count, "required": config.MIN_OFFSET_AGREEMENT_RECORDS, "differences": differences, "passes": count >= config.MIN_OFFSET_AGREEMENT_RECORDS}

    repeat_gap_c = metric(repeat_c, "gap_c")
    repeat_gap_d = metric(repeat_d, "gap_d")
    own_gap_c = metric(own_c, "gap_c")
    own_gap_d = metric(own_d, "gap_d")
    repeat = {"gap_c": repeat_gap_c, "gap_d": repeat_gap_d, "offset_agreement": offset(repeat_offsets), "passes": bool(repeat_gap_c["ci95"][0] > config.REPLACEMENT_MARGIN and repeat_gap_d["ci95"][0] > config.REPLACEMENT_MARGIN and offset(repeat_offsets)["passes"]), "gate": "repeatability"}
    own = {"gap_c": own_gap_c, "gap_d": own_gap_d, "offset_agreement": offset(own_offsets), "passes": bool(own_gap_c["ci95"][0] > config.REPLACEMENT_MARGIN and own_gap_d["ci95"][0] > config.REPLACEMENT_MARGIN and offset(own_offsets)["passes"]), "gate": "own_audio_validity"}
    damage_c_stats = metric(damage_c, "damage_C")
    damage_d_stats = metric(damage_d, "damage_D")
    both_positive = sum(c > 0 and d > 0 for c, d in zip(damage_c, damage_d))
    sensitivity = {"damage_C": damage_c_stats, "damage_D": damage_d_stats, "both_damage_positive_count": both_positive, "required_positive_count": config.MIN_LOCAL_SENSITIVITY_RECORDS, "passes": bool(damage_c_stats["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD and damage_d_stats["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD and both_positive >= config.MIN_LOCAL_SENSITIVITY_RECORDS), "per_record_damage_C": damage_c, "per_record_damage_D": damage_d}
    controls = {"N_REPEAT": {"repeatability": repeat, "passes": repeat["passes"]}, control_arm: {"own_audio_validity": own, "sensitivity": sensitivity, "passes": bool(own["passes"] and sensitivity["passes"])}}
    decision = "CONTROL_CALIBRATED" if repeat["passes"] and own["passes"] and sensitivity["passes"] else "CONTROL_FAILED"
    return decision, {"controls": controls, "decision": decision}


def validate_analysis_and_final(root: Path, cohort: Mapping[str, Any], protocol: Mapping[str, Any], scores: Mapping[str, Any]) -> dict[str, Any]:
    analysis = _artifact(root / "05_final/analysis.json", "05_analysis")
    final = _artifact(root / "05_final/final.json", "05_final")
    decision, recomputed = _recompute_controls(cohort, protocol, scores)
    if analysis.get("decisions", {}).get("scientific_decision") != decision or analysis.get("controls") != recomputed["controls"]:
        raise ValidationError("analysis does not match independently recomputed controls")
    if (
        final.get("scientific_decision") != decision
        or final.get("controls") != recomputed["controls"]
        or final.get("bootstrap") != analysis.get("bootstrap")
        or final.get("reference_conditioned_audio_head_spec_eligible") is not False
        or final.get("engineering_decision") != "GO"
        or final.get("audit_decision") != protocol.get("audit_decision")
        or final.get("audit_item_count") != config.EXPECTED_RECORD_COUNT * 10
    ):
        raise ValidationError("final decision or eligibility is invalid")
    expected_hashes = {
        "audit_sha256": root / "00_audit/audit.json",
        "protocol_sha256": root / "01_protocol/protocol.json",
        "cohort_sha256": root / "01_protocol/cohort.json",
        "audio_manifest_sha256": root / "02_audio/audio_manifest.json",
        "videos_manifest_sha256": root / "03_videos/videos_manifest.json",
        "scores_manifest_sha256": root / "04_scores/scores_manifest.json",
        "analysis_sha256": root / "05_final/analysis.json",
    }
    for field, path in expected_hashes.items():
        if final.get(field) != file_sha256(path):
            raise ValidationError(f"final binding changed: {field}")
    if final.get("record_count") != config.EXPECTED_RECORD_COUNT or final.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT or final.get("cell_count") != config.expected_cell_count(str(protocol["branch"])):
        raise ValidationError("final counts are invalid")
    if not (root / "05_final/result.md").is_file():
        raise ValidationError("result.md is missing")
    return {"stage": "05_final", "status": "valid", "scientific_decision": decision}


def validate_blocked(root: Path) -> dict[str, Any]:
    final = _artifact(root / "05_final/final.json", "05_final")
    if final.get("status") != "blocked" or final.get("engineering_decision") != "BLOCKED" or final.get("scientific_decision") != "BLOCKED" or final.get("reference_conditioned_audio_head_spec_eligible") is not False:
        raise ValidationError("blocked terminal is invalid")
    return {"stage": "05_final", "status": "valid", "scientific_decision": "BLOCKED"}


def validate_run(root: Path) -> dict[str, Any]:
    final_path = root / "05_final/final.json"
    if final_path.is_file():
        final = verify_self_hashed_json(final_path)
        if final.get("status") == "blocked":
            results = []
            if (root / "00_audit/audit.json").is_file():
                results.append(validate_audit(root))
            if (root / "01_protocol/protocol.json").is_file():
                results.append(validate_protocol(root))
            results.append(validate_blocked(root))
            return {"status": "valid", "stages": results}
    audit_result = validate_audit(root)
    audit = _artifact(root / "00_audit/audit.json", "00_audit")
    protocol = _artifact(root / "01_protocol/protocol.json", "01_protocol")
    protocol_result = validate_protocol(root, audit)
    cohort = _artifact(root / "01_protocol/cohort.json", "01_protocol")
    cohort_result = validate_cohort(root)
    audio_result = validate_audio(root, cohort, protocol)
    video_result = validate_videos(root, cohort, protocol)
    scores = _artifact(root / "04_scores/scores_manifest.json", "04_scores")
    score_result = validate_scores(root, cohort, protocol)
    final_result = validate_analysis_and_final(root, cohort, protocol, scores)
    return {"status": "valid", "stages": [audit_result, protocol_result, cohort_result, audio_result, video_result, score_result, final_result]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate an LRS3 calibration run")
    parser.add_argument("--run-root", "--root", dest="run_root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    output = args.run_root / "05_final/validation.json"
    write_self_hashed_json(output, {"schema_version": 1, "stage_id": "05_validation", "protocol_id": config.PROTOCOL_ID, **result})
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
