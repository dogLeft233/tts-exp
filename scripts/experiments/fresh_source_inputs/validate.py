from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from .protocol import (
    GROUP_ID_RE,
    REPO_ROOT,
    ProtocolError,
    file_sha256,
    inspect_clip,
    load_self_hashed,
    scan_history,
    transcript_for,
    validate_visual_audit,
    write_json,
)


def _body_hash(value: dict) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _pcm16_identity(path: Path) -> tuple[np.ndarray, int, str]:
    """Read a mono PCM16 WAV and return samples, rate, and exact byte SHA."""

    values, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    values = np.asarray(values)
    digest = hashlib.sha256(values.astype(np.int16, copy=False).tobytes()).hexdigest()
    return values, int(sample_rate), digest


def _load_required(run_root: Path, name: str, errors: list[str]) -> dict:
    path = run_root / name
    if not path.is_file():
        errors.append(f"missing {name}")
        return {}
    try:
        return load_self_hashed(path)
    except ProtocolError as exc:
        errors.append(str(exc))
        return {}


def _load_root_shared(run_root: Path, errors: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    names = ("history_groups.json", "candidate_audit.json", "cohort.json", "inputs.json", "validation.json")
    root_values = {name: _load_required(run_root, name, errors) for name in names}
    shared = run_root / "run" / "shared"
    shared_values: dict[str, dict] = {}
    for name in names:
        path = shared / name
        if not path.is_file():
            errors.append(f"missing shared/{name}")
            continue
        try:
            shared_values[name] = load_self_hashed(path)
            if root_values.get(name) and _body_hash(root_values[name]) != _body_hash(shared_values[name]):
                errors.append(f"root/shared artifact mismatch: {name}")
        except ProtocolError as exc:
            errors.append(f"shared/{name}: {exc}")
    return root_values, shared_values


def _visual_groups_from_audit(run_root: Path, audit: dict, errors: list[str]) -> dict[str, dict]:
    audit_path_value = audit.get("visual_audit")
    if not audit_path_value:
        acquisition_path = run_root / "acquisition" / "visual_audit.json"
        audit_path = acquisition_path if acquisition_path.is_file() else None
    else:
        audit_path = Path(str(audit_path_value)).resolve()
    if audit_path is None:
        return {}
    try:
        value = load_self_hashed(audit_path)
        if audit_path_value and audit.get("visual_audit_sha256") != file_sha256(audit_path):
            errors.append("visual audit file SHA mismatch")
        return validate_visual_audit(value)
    except (OSError, ValueError, ProtocolError) as exc:
        errors.append(f"visual audit validation failed: {exc}")
        return {}


def _pool_context(run_root: Path, errors: list[str]) -> tuple[dict | None, dict | None]:
    acquisition = run_root / "acquisition"
    plan_path, pool_path = acquisition / "plan.json", acquisition / "pool.json"
    if not plan_path.exists() and not pool_path.exists():
        return None, None
    if not plan_path.is_file() or not pool_path.is_file():
        errors.append("acquisition plan and pool must be supplied together")
        return None, None
    try:
        plan = load_self_hashed(plan_path)
        pool = load_self_hashed(pool_path)
        if pool.get("source_plan_sha256") != file_sha256(plan_path):
            errors.append("acquisition pool is not bound to current plan")
        if plan.get("run_root") and Path(str(plan["run_root"])).resolve() != run_root.resolve():
            errors.append("acquisition plan belongs to another run root")
        data_root = REPO_ROOT / "data/dataset_samples/lrs3/pretrain"
        if int(pool.get("max_groups", 24)) > 24 or len(pool.get("groups", [])) > 24:
            errors.append("acquisition pool exceeds the 24-group cap")
        for entry in pool.get("groups", []):
            if not isinstance(entry, dict):
                errors.append("malformed acquisition pool group")
                continue
            group = str(entry.get("source_group", ""))
            clips = entry.get("clips", [])
            if not GROUP_ID_RE.fullmatch(group) or not isinstance(clips, list):
                errors.append(f"malformed acquisition pool group: {group}")
                continue
            pool_ids = [str(item.get("clip_id")) for item in sorted(clips, key=lambda item: str(item.get("clip_id", "")))[:8] if isinstance(item, dict)]
            local_ids = sorted(path.stem for path in (data_root / group).glob("*.mp4"))[:8] if (data_root / group).is_dir() else []
            if local_ids != pool_ids:
                errors.append(f"acquisition pool is not bound to the original first-eight clips: {group}")
            for clip in clips:
                if not isinstance(clip, dict):
                    continue
                path = Path(str(clip.get("mp4_path", "")))
                if not _under(path, data_root) or not path.is_file() or clip.get("mp4_sha256") != file_sha256(path):
                    errors.append(f"acquisition pool clip missing or SHA mismatch: {path}")
                txt = clip.get("txt_path")
                if txt and (not _under(Path(str(txt)), data_root) or not Path(str(txt)).is_file() or clip.get("txt_sha256") != file_sha256(Path(str(txt)))):
                    errors.append(f"acquisition pool transcript missing or SHA mismatch: {txt}")
        return plan, pool
    except ProtocolError as exc:
        errors.append(str(exc))
        return None, None


def _independent_clip_rows(run_root: Path, pool: dict | None, visual_groups: dict[str, dict], historical: set[str], errors: list[str]) -> tuple[list[dict], int, int]:
    data_root = REPO_ROOT / "data/dataset_samples/lrs3/pretrain"
    if pool is None:
        group_entries = [(path.name, [{"clip_id": item.stem, "mp4_path": str(item.resolve())} for item in sorted(path.glob("*.mp4"))[:8]]) for path in sorted(data_root.iterdir()) if path.is_dir() and GROUP_ID_RE.fullmatch(path.name)]
        pool_status = "READY"
    else:
        group_entries = [(str(item.get("source_group")), item.get("clips", [])) for item in pool.get("groups", []) if isinstance(item, dict)]
        pool_status = pool.get("status")
    discovered = len(group_entries)
    rows: list[dict] = []
    if pool is not None and pool_status != "READY":
        return rows, discovered, 0
    for group, pool_clips in group_entries:
        if group in historical:
            continue
        clips = []
        for pool_clip in sorted(pool_clips, key=lambda item: str(item.get("clip_id", "")))[:8]:
            path = Path(str(pool_clip.get("mp4_path", ""))).resolve()
            if not _under(path, data_root) or not path.is_file():
                errors.append(f"candidate clip is outside data root or missing: {path}")
                continue
            try:
                media = inspect_clip(path)
                transcript = transcript_for(path)
                media.update({"clip_id": path.stem, "transcript": transcript, "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(), "duration_ok": bool(np.isfinite(media.get("duration_s", np.nan)) and 6.0 <= float(media.get("duration_s", 0.0)) <= 10.0), "duration_pcm_ok": bool(96000 <= int(media.get("decoded_pcm_samples", 0)) <= 160000)})
                visual = visual_groups.get(group, {})
                media["visual_status"] = visual.get("status", "UNVERIFIED")
                media["visual_audit_verified"] = bool(visual.get("audit_verified") and visual.get("clip_sha256") == media.get("sha256"))
                media["input_ok"] = bool(media.get("has_video") and media.get("has_audio") and media.get("duration_ok") and media.get("duration_pcm_ok") and media.get("pts_ok") and transcript and media["visual_status"] == "PASS" and media["visual_audit_verified"])
                clips.append(media)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                errors.append(f"candidate media reinspection failed {path}: {type(exc).__name__}: {exc}")
        accepted = next((clip for clip in clips if clip.get("input_ok") is True), None)
        rows.append({"source_group": group, "historical": False, "clips_examined": clips, "accepted_clip": accepted, "duration_candidates": [clip for clip in clips if clip.get("duration_ok") and clip.get("duration_pcm_ok") and clip.get("pts_ok") and clip.get("transcript")], "quality_status": "candidate" if accepted else "VISUAL_UNVERIFIED"})
    return rows, discovered, len([row for row in rows if row.get("accepted_clip")])


def _validate_cohort(run_root: Path) -> dict:
    errors: list[str] = []
    root_values, _ = _load_root_shared(run_root, errors)
    cohort = root_values.get("cohort.json", {})
    audit = root_values.get("candidate_audit.json", {})
    history = root_values.get("history_groups.json", {})
    status = cohort.get("status")
    if status not in {"GO", "BLOCKED_NEW_SOURCE", "BLOCKED_HISTORY_COVERAGE", "BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET"}:
        errors.append(f"invalid cohort status: {status}")
    try:
        actual_history = scan_history(REPO_ROOT, ignore_run_root=run_root)
        if history.get("groups") != actual_history.get("groups"):
            errors.append("history group set differs from independent scan")
        if history.get("scan_file_manifest_sha256") != actual_history.get("scan_file_manifest_sha256"):
            errors.append("history file manifest differs from independent scan")
        _plan, pool = _pool_context(run_root, errors)
        visual_groups = _visual_groups_from_audit(run_root, audit, errors)
        independent_rows, discovered, eligible = _independent_clip_rows(run_root, pool, visual_groups, set(actual_history.get("groups", [])), errors)
        producer_rows = audit.get("rows", [])
        if pool is not None and pool.get("status") == "READY":
            expected_groups = sorted(row.get("source_group") for row in independent_rows)
            if sorted(str(row.get("source_group")) for row in producer_rows) != expected_groups:
                errors.append("candidate audit rows are not exactly the bounded pool group set")
        producer_by_group = {str(row.get("source_group")): row for row in producer_rows if isinstance(row, dict)}
        for expected_row in independent_rows:
            group = str(expected_row.get("source_group"))
            producer_row = producer_by_group.get(group)
            if producer_row is None:
                continue
            expected_clip = expected_row.get("accepted_clip") or {}
            actual_clip = producer_row.get("accepted_clip") or {}
            for key in ("path", "sha256", "clip_id", "input_ok", "visual_audit_verified", "duration_ok", "duration_pcm_ok", "pts_ok"):
                if actual_clip.get(key) != expected_clip.get(key):
                    errors.append(f"candidate audit field is not independently derived: {group}:{key}")
        if int(cohort.get("accepted_count", -1)) != eligible:
            errors.append("cohort eligible count does not match independent visual/media screening")
        accepted_rows = {str(row.get("source_group")): row for row in independent_rows if row.get("accepted_clip")}
        selected_formal = cohort.get("formal", [])
        selected_smoke = cohort.get("smoke", [])
        selected = [str(row.get("source_group")) for row in selected_formal + selected_smoke]
        if len(selected) != len(set(selected)):
            errors.append("formal and smoke cohorts contain duplicate source groups")
        if any(group not in accepted_rows for group in selected):
            errors.append("selected cohort contains a group not independently accepted")
        if status == "GO":
            if len(selected_formal) != 12 or len(selected_smoke) != 2:
                errors.append("GO cohort does not contain exactly 12 formal plus 2 smoke groups")
            if selected != sorted(accepted_rows)[:14]:
                errors.append("GO cohort is not the frozen sorted first 12 formal plus 2 smoke groups")
        elif selected:
            errors.append("blocked cohort must not select formal or smoke groups")
        upstream_coverage = list(history.get("coverage_blockers", [])) + list(history.get("pool_blockers", []))
        if status == "BLOCKED_HISTORY_COVERAGE" and eligible >= 14 and not upstream_coverage:
            errors.append("history coverage blocker is not recorded")
        if status in {"BLOCKED_NEW_SOURCE", "BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET"} and eligible >= 14 and not upstream_coverage:
            errors.append("blocked cohort has enough eligible groups without a recorded blocker")
        media_qualified = sum(1 for row in independent_rows for clip in row.get("clips_examined", []) if clip.get("duration_ok") and clip.get("duration_pcm_ok") and clip.get("pts_ok") and clip.get("transcript"))
        visual_reviewed = sum(1 for group in visual_groups.values() if group.get("status") in {"PASS", "FAIL"})
        final_counts = {"discovered": discovered, "discovered_groups": discovered, "historical_excluded": len(set(actual_history.get("groups", []))), "media_qualified": media_qualified, "visual_reviewed": visual_reviewed, "visually_audited": visual_reviewed, "accepted": eligible, "accepted_groups": eligible, "formal": len(selected_formal), "smoke": len(selected_smoke)}
        result_status = "COHORT_READY" if status == "GO" and not errors else (status if not errors else "NO_GO")
        visual_path = Path(str(audit.get("visual_audit"))).resolve() if audit.get("visual_audit") else run_root / "acquisition" / "visual_audit.json"
        result = {"schema_version": 1, "status": result_status, "integrity": "GO" if not errors else "NO_GO", "upstream_status": status, "inputs_status": "PENDING_CANDIDATE" if status == "GO" else status, "errors": errors, "counts": final_counts, "history_groups_sha256": history.get("scan_file_manifest_sha256"), "pool_sha256": file_sha256(run_root / "acquisition" / "pool.json") if (run_root / "acquisition" / "pool.json").is_file() else None, "visual_audit_sha256": file_sha256(visual_path) if visual_path.is_file() else None, "next": ["runner --stage candidate", "validate --phase inputs", "runner --stage freeze", "validate --phase inputs"] if result_status == "COHORT_READY" else []}
    except (OSError, ValueError, TypeError, KeyError, ProtocolError) as exc:
        result = {"schema_version": 1, "status": "NO_GO", "integrity": "NO_GO", "upstream_status": status, "errors": errors + [f"independent recomputation failed: {type(exc).__name__}: {exc}"]}
    write_json(run_root / "cohort_validation.json", result)
    acquisition_final = run_root / "acquisition" / "final.json"
    acquisition = run_root / "acquisition"
    bound = {}
    for name in ("plan.json", "history_snapshot.json", "source_manifest.json", "pool.json", "ledger.jsonl", "visual_audit.json", "clip_audit.jsonl"):
        path = acquisition / name
        if path.is_file():
            bound[name] = file_sha256(path)
    write_json(acquisition_final, {"schema_version": 1, "status": result.get("status"), "integrity": result.get("integrity"), "cohort_validation_sha256": file_sha256(run_root / "cohort_validation.json"), "bound_artifacts": bound, "counts": result.get("counts", {}), "errors": result.get("errors", [])})
    return result


def _validate_inputs(run_root: Path) -> dict:
    errors: list[str] = []
    names = ("history_groups.json", "candidate_audit.json", "cohort.json", "inputs.json", "validation.json")
    root_values = {name: _load_required(run_root, name, errors) for name in names}
    shared = run_root / "run" / "shared"
    shared_values: dict[str, dict] = {}
    for name in names:
        path = shared / name
        if not path.is_file():
            errors.append(f"missing shared/{name}")
            continue
        try:
            shared_values[name] = load_self_hashed(path)
            if root_values.get(name) and _body_hash(root_values[name]) != _body_hash(shared_values[name]):
                errors.append(f"root/shared artifact mismatch: {name}")
        except ProtocolError as exc:
            errors.append(f"shared/{name}: {exc}")

    cohort = root_values.get("cohort.json", {})
    status = cohort.get("status")
    if status not in {"GO", "BLOCKED_NEW_SOURCE", "BLOCKED_HISTORY_COVERAGE", "BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET"}:
        errors.append(f"invalid cohort status: {status}")
    if status in {"BLOCKED_NEW_SOURCE", "BLOCKED_HISTORY_COVERAGE", "BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET"} and cohort.get("accepted_count", 0) >= 14:
        errors.append("blocked cohort has enough eligible groups")

    history = root_values.get("history_groups.json", {})
    audit = root_values.get("candidate_audit.json", {})
    inputs = root_values.get("inputs.json", {})
    validation = root_values.get("validation.json", {})
    try:
        contracts = [
            value.get("execution_contract")
            for value in (audit, cohort, inputs, validation)
            if value.get("execution_contract") is not None
        ]
        for contract in contracts:
            if not isinstance(contract, dict):
                errors.append("execution contract is malformed")
                continue
            contract_body = dict(contract)
            actual_contract_sha = contract_body.pop("contract_sha256", None)
            if actual_contract_sha != hashlib.sha256(
                json.dumps(contract_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest():
                errors.append("execution contract self-hash mismatch")
        if contracts and any(contract != contracts[0] for contract in contracts[1:]):
            errors.append("execution contract differs across P artifacts")
        if status == "GO" and not contracts:
            errors.append("GO cohort is missing execution contract")
        # Re-scan independently. The current run is ignored by path; earlier
        # fresh-source runs remain historical evidence.
        actual_history = scan_history(REPO_ROOT, ignore_run_root=run_root)
        if sorted(history.get("groups", [])) != sorted(actual_history.get("groups", [])):
            errors.append("history group set differs from independent scan")
        if history.get("scan_file_manifest_sha256") != actual_history.get("scan_file_manifest_sha256"):
            errors.append("history file manifest differs from independent scan")
        if history.get("coverage") != actual_history.get("coverage"):
            errors.append("history coverage differs from independent scan")
        local_root = REPO_ROOT / "data/dataset_samples/lrs3/pretrain"
        _, pool = _pool_context(run_root, errors)
        if pool is not None:
            pool_groups = {str(item.get("source_group")) for item in pool.get("groups", []) if isinstance(item, dict)}
            recomputed_fresh = sorted(pool_groups - set(actual_history.get("groups", []))) if pool.get("status") == "READY" else []
            if len(pool_groups) > 24:
                errors.append("bounded acquisition pool exceeds 24 groups")
        else:
            local_groups = {path.name for path in local_root.iterdir() if path.is_dir() and GROUP_ID_RE.fullmatch(path.name)}
            recomputed_fresh = sorted(local_groups - set(actual_history.get("groups", [])))
        rows = audit.get("rows", [])
        if len(recomputed_fresh) != int(audit.get("fresh_group_count", -1)):
            errors.append("fresh group count does not match independent history exclusion set")
        if sorted(row.get("source_group") for row in rows) != recomputed_fresh:
            errors.append("candidate audit rows are not exactly the independent fresh group set")
        accepted_rows = {str(row.get("source_group")): row for row in rows if row.get("accepted_clip") is not None}
        recomputed_eligible = len(accepted_rows)
        if recomputed_eligible != int(cohort.get("accepted_count", -1)):
            errors.append("cohort eligible count does not match candidate audit")
        selected_formal = cohort.get("formal", [])
        selected_smoke = cohort.get("smoke", [])
        selected = [row.get("source_group") for row in selected_formal + selected_smoke]
        if len(selected) != len(set(selected)):
            errors.append("formal and smoke cohorts contain duplicate source groups")
        if any(group not in accepted_rows for group in selected):
            errors.append("selected cohort contains a group not accepted by candidate audit")
        if status == "GO" and (len(selected_formal) != 12 or len(selected_smoke) != 2):
            errors.append("GO cohort does not contain exactly 12 formal plus 2 smoke groups")
        if status != "GO" and selected:
            errors.append("blocked cohort must not select formal or smoke groups")
        if status == "GO":
            accepted_order = sorted(accepted_rows)
            selected_order = [str(row.get("source_group")) for row in selected_formal + selected_smoke]
            if selected_order != accepted_order[:14]:
                errors.append("GO cohort is not the frozen sorted first 12 formal plus 2 smoke groups")
        input_status = inputs.get("status")
        if status == "GO" and input_status not in {"PENDING_CANDIDATE", "GO"}:
            errors.append("GO cohort has an invalid inputs status")
        if status != "GO" and input_status != status:
            errors.append("blocked cohort and inputs statuses differ")

        visual_groups = _visual_groups_from_audit(run_root, audit, errors)

        data_root = local_root.resolve()
        for row in rows:
            for clip in row.get("clips_examined", []):
                path_value = clip.get("path")
                if not path_value:
                    continue
                path = Path(path_value)
                if not _under(path, data_root) or not path.is_file():
                    errors.append(f"candidate clip is outside data root or missing: {path}")
                    continue
                if clip.get("sha256") != file_sha256(path):
                    errors.append(f"candidate clip SHA mismatch: {path}")
                try:
                    measured = inspect_clip(path)
                    for key in ("duration_s", "fps", "width", "height", "video_frames", "has_video", "has_audio", "audio_sample_rate", "audio_channels", "pts_ok"):
                        expected_value = clip.get(key)
                        actual_value = measured.get(key)
                        if isinstance(expected_value, float) or isinstance(actual_value, float):
                            if expected_value is None or actual_value is None or abs(float(expected_value) - float(actual_value)) > 1e-3:
                                errors.append(f"candidate media field mismatch {key}: {path}")
                        elif expected_value != actual_value:
                            errors.append(f"candidate media field mismatch {key}: {path}")
                except (OSError, ValueError, ProtocolError, TypeError) as exc:
                    errors.append(f"candidate media reinspection failed {path}: {exc}")
                transcript = transcript_for(path)
                if clip.get("transcript") != transcript:
                    errors.append(f"candidate transcript content mismatch: {path}")
                if clip.get("transcript_sha256") != hashlib.sha256(transcript.encode("utf-8")).hexdigest():
                    errors.append(f"candidate transcript SHA mismatch: {path}")
                visual = visual_groups.get(str(row.get("source_group")), {})
                visual_verified = bool(
                    visual.get("audit_verified")
                    and visual.get("clip_sha256") == clip.get("sha256")
                    and float(visual.get("valid_fraction", 0.0)) >= 0.95
                    and float(visual.get("eye_distance_px", 0.0)) >= 40.0
                    and int(visual.get("frames_covered", 0)) >= 140
                    and visual.get("first_frame_face") is True
                    and visual.get("mouth_visible") is True
                )
                expected_input_ok = bool(
                    clip.get("has_video") and clip.get("has_audio") and clip.get("duration_ok")
                    and clip.get("duration_pcm_ok") and clip.get("pts_ok") and transcript
                    and visual.get("status") == "PASS" and visual_verified
                )
                if bool(clip.get("visual_audit_verified")) != visual_verified or bool(clip.get("input_ok")) != expected_input_ok:
                    errors.append(f"candidate quality gate is not independently reproducible: {path}")

        expected_validation = {
            "history_group_count": len(actual_history.get("groups", [])),
            "fresh_group_count": len(recomputed_fresh),
            "eligible_fresh_group_count": recomputed_eligible,
        }
        for key, value in expected_validation.items():
            if validation.get(key) != value:
                errors.append(f"validation.{key} is not independently derived")

        records = inputs.get("records", [])
        if inputs.get("status") == "GO":
            if len(records) != 14 or int(inputs.get("formal_count", -1)) != 12 or int(inputs.get("smoke_count", -1)) != 2:
                errors.append("GO inputs must contain exactly 12 formal plus 2 smoke records")
            if len({record.get("sample_id") for record in records}) != len(records):
                errors.append("GO inputs contain duplicate sample IDs")
            for record in records:
                if record.get("speaker_independence") not in {"verified", "unverified"}:
                    errors.append(f"invalid speaker independence status: {record.get('sample_id')}")
                group = str(record.get("source_group"))
                candidate_row = next((item for item in rows if str(item.get("source_group")) == group), None)
                candidate_clip = (candidate_row or {}).get("accepted_clip") or {}
                source_contract = record.get("source_media_contract") or {}
                for key in ("format_start_time", "first_video_pts", "first_audio_pts", "pts_ok"):
                    expected = candidate_clip.get(key)
                    actual = source_contract.get(key)
                    if isinstance(expected, float) or isinstance(actual, float):
                        try:
                            if expected is None or actual is None or abs(float(expected) - float(actual)) > 1e-6:
                                errors.append(f"source PTS contract mismatch: {record.get('sample_id')}:{key}")
                        except (TypeError, ValueError):
                            errors.append(f"source PTS contract malformed: {record.get('sample_id')}:{key}")
                    elif expected != actual:
                        errors.append(f"source media contract mismatch: {record.get('sample_id')}:{key}")
                if source_contract.get("stream_start_times") != candidate_clip.get("stream_start_times"):
                    errors.append(f"source stream PTS contract mismatch: {record.get('sample_id')}")
                expected_transcode = [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(record.get("source_video")),
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(record.get("natural_audio")),
                ]
                if source_contract.get("transcode_command") != expected_transcode:
                    errors.append(f"natural transcode command mismatch: {record.get('sample_id')}")
                assets = {
                    "source_video": record.get("source_video"),
                    "natural_audio": record.get("natural_audio"),
                    "real_video": record.get("real_video"),
                    "reference_image": record.get("reference_image"),
                }
                for field, value in assets.items():
                    asset = Path(str(value or ""))
                    if field == "source_video" and (not _under(asset, data_root) or not asset.is_file()):
                        errors.append(f"input source outside data root or missing: {asset}")
                    elif field != "source_video" and not asset.is_file():
                        errors.append(f"input asset missing: {field}={asset}")
                    elif asset.is_file() and record.get(f"{field}_sha256") != file_sha256(asset):
                        errors.append(f"input asset SHA mismatch: {field}={asset}")
                natural_values = np.asarray([], dtype=np.int16)
                try:
                    natural_values, natural_rate, natural_pcm_sha = _pcm16_identity(Path(str(record.get("natural_audio", ""))))
                    if record.get("natural_pcm_sha256") != natural_pcm_sha or record.get("natural_pcm_sample_count") != int(natural_values.size):
                        errors.append(f"natural decoded PCM binding failed: {record.get('sample_id')}")
                    natural_float = natural_values.astype(np.float32, copy=False)
                    if int(natural_rate) != 16000 or natural_values.ndim != 1 or not (96000 <= natural_values.size <= 160000) or not np.isfinite(natural_float).all() or float(np.sqrt(np.mean(np.square(natural_float)))) <= 0.0:
                        errors.append(f"natural audio PCM contract failed: {record.get('sample_id')}")
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(f"natural audio decode failed: {exc}")
                try:
                    real_media = inspect_clip(Path(str(record.get("real_video", ""))))
                    if real_media.get("width") != 512 or real_media.get("height") != 512 or real_media.get("video_frames") != 140 or abs(float(real_media.get("fps", 0.0)) - 25.0) > 0.01 or real_media.get("has_audio") or not real_media.get("pts_ok"):
                        errors.append(f"real video geometry/PTS contract failed: {record.get('sample_id')}")
                except (OSError, RuntimeError, ValueError, ProtocolError) as exc:
                    errors.append(f"real video reinspection failed: {exc}")
                try:
                    with Image.open(Path(str(record.get("reference_image", "")))) as reference:
                        if reference.size != (512, 512):
                            errors.append(f"reference image is not 512x512: {record.get('sample_id')}")
                except (OSError, ValueError) as exc:
                    errors.append(f"reference image inspection failed: {exc}")
                direct = record.get("direct_audio") or {}
                direct_path = Path(str(direct.get("output_path", "")))
                if not direct_path.is_file() or direct.get("output_sha256") != file_sha256(direct_path):
                    errors.append(f"direct audio output missing or SHA mismatch: {direct_path}")
                else:
                    try:
                        direct_values, direct_rate = sf.read(direct_path, dtype="int16", always_2d=False)
                        direct_values = np.asarray(direct_values)
                        direct_pcm_sha = hashlib.sha256(direct_values.astype(np.int16).tobytes()).hexdigest()
                        if int(direct_rate) != 16000 or direct_values.ndim != 1 or direct_values.size != int(natural_values.size) or not np.isfinite(direct_values).all() or float(np.sqrt(np.mean(np.square(direct_values.astype(np.float32))))) <= 0.0 or direct.get("decoded_pcm_sha256") != direct_pcm_sha:
                            errors.append(f"direct audio PCM contract failed: {record.get('sample_id')}")
                    except (OSError, RuntimeError, ValueError) as exc:
                        errors.append(f"direct audio decode failed: {exc}")
                for raw_field, sha_field in (("decoder_float_path", "decoder_float_sha256"), ("natural_feature_path", "natural_feature_sha256"), ("reencoded_feature_path", "reencoded_feature_sha256")):
                    raw_path = Path(str(direct.get(raw_field, "")))
                    if not raw_path.is_file() or direct.get(sha_field) != file_sha256(raw_path):
                        errors.append(f"direct raw artifact missing or SHA mismatch: {raw_path}")
                if (record.get("model_contract") or {}).get("seed") != 42:
                    errors.append("direct candidate model contract is not seed42")
                geometry = record.get("geometry_contract") or {}
                if geometry.get("frames_written") != 140 or geometry.get("resize_interpolation") != "INTER_LANCZOS4" or not geometry.get("pts_ok"):
                    errors.append(f"input geometry contract missing: {record.get('sample_id')}")
        elif records:
            errors.append("non-GO inputs must not contain scientific records")
        freeze_path = run_root / "run" / "shared" / "freeze.json"
        if status == "GO":
            if not freeze_path.is_file():
                errors.append("GO cohort is missing shared freeze manifest")
            else:
                frozen = load_self_hashed(freeze_path)
                if frozen.get("status") != "FROZEN":
                    errors.append("freeze manifest is not FROZEN")
                if frozen.get("inputs_sha256") != file_sha256(run_root / "inputs.json") or frozen.get("cohort_sha256") != file_sha256(run_root / "cohort.json"):
                    errors.append("freeze manifest does not bind current inputs/cohort")
                if frozen.get("record_count") != len(records):
                    errors.append("freeze record count mismatch")
        elif freeze_path.exists():
            errors.append("blocked cohort must not have a freeze manifest")
    except (OSError, ValueError, KeyError, TypeError, ProtocolError) as exc:
        errors.append(f"independent recomputation failed: {type(exc).__name__}: {exc}")

    result = {
        "schema_version": 1,
        "status": status if not errors else "NO_GO",
        "integrity": "GO" if not errors else "NO_GO",
        "errors": errors,
        "upstream_status": status,
    }
    write_json(run_root / "validation_independent.json", result)
    return result


def validate(run_root: Path, *, phase: str = "inputs") -> dict:
    """Independently validate either the cohort gate or frozen inputs.

    ``cohort`` deliberately stops before direct_v1/C/freeze.  ``inputs`` keeps
    the original full validator and is therefore the only phase that can
    authorize scientific cells.
    """

    run_root = Path(run_root).resolve()
    if phase == "cohort":
        return _validate_cohort(run_root)
    if phase != "inputs":
        raise ValueError("phase must be cohort or inputs")
    return _validate_inputs(run_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cohort", "inputs"), default="inputs")
    args = parser.parse_args(argv)
    result = validate(args.run_root.resolve(), phase=args.phase)
    print(result)
    return 0 if result.get("integrity") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
