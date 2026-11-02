#!/usr/bin/env python3
"""Freeze dev-only rows for a fit-selected shared-frequency profile shortlist.

This builder reads plan and receipt metadata only. It opens no audio, performs
no model inference, and emits candidate-only rows; previously scored controls
remain in their original immutable ledgers for later exact-binding joins.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_PLAN_SHA256 = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FRONTEND_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
PROFILE_SOURCE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
SELECTOR_SOURCE_SHA256 = "1e329d4b1d2f5918230e5cdcf2d3c3c91256daf7a4536e7a7424ad33ff609a6d"
PROTOCOL_SOURCE_SHA256 = "e79d33ce4e3338fafdc78ad52761817123fdac9070dace810fd8a10928be3cd2"
STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_FRONTEND_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
PROFILE_DISPATCHER_SHA256 = "dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660"
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
CONTROL_METHODS = ("identity", "rms_fixed", "localcompress")
RUN_ID = "cross_tfg_audio_head_20261003_shared_profile_shortlist_dev_attempt01"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha(value: dict[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def load_frozen_plan(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    if actual != expected_sha:
        raise ValueError(f"{label} raw SHA mismatch: {actual}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("status") != "FROZEN":
        raise ValueError(f"{label} is not a frozen plan")
    if value.get("plan_sha256") != payload_sha(value, "plan_sha256"):
        raise ValueError(f"{label} payload SHA mismatch")
    return value, raw


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dev-plan", type=Path, required=True)
    parser.add_argument("--frontend-fit-plan", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--expected-selection-report-raw-sha256", required=True)
    parser.add_argument("--raw-matrix-qc", type=Path, required=True)
    parser.add_argument("--expected-raw-matrix-qc-raw-sha256", required=True)
    parser.add_argument("--selector-source", type=Path, required=True)
    parser.add_argument("--protocol-source", type=Path, required=True)
    parser.add_argument("--profile-source", type=Path, required=True)
    parser.add_argument("--stagea-core", type=Path, required=True)
    parser.add_argument("--cross-frontend-runner", type=Path, required=True)
    parser.add_argument("--profile-dispatcher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base, base_raw = load_frozen_plan(args.base_dev_plan, BASE_PLAN_SHA256, "base dev plan")
    frontend, frontend_raw = load_frozen_plan(args.frontend_fit_plan, FRONTEND_PLAN_SHA256, "frontend fit plan")

    if base.get("schema_version") != "cross-tfg-audio-eval-v1":
        raise ValueError("base plan schema differs from the frozen Stage-A contract")
    if base.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ValueError("base dev seed family differs from the frozen paired seeds")
    if base.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("base plan unexpectedly exposes confirmation seeds")
    base_rows = base.get("rows")
    if not isinstance(base_rows, list) or len(base_rows) != 180:
        raise ValueError("base dev plan must contain the frozen 20 × 3-seed × 3-control roster")
    if tuple(method.get("method_id") for method in base.get("methods", [])) != CONTROL_METHODS:
        raise ValueError("base plan control roster changed")

    model_records = frontend.get("models")
    model_ids = tuple(model.get("model_id") for model in model_records or [])
    if model_ids != EXPECTED_MODELS:
        raise ValueError(f"qualified three-frontend roster changed: {model_ids}")
    if frontend.get("generation_seed_family", {}).get("confirm") != []:
        raise ValueError("frontend fit plan unexpectedly declares confirmation seeds")
    if len(frontend.get("rows", [])) != 360 or len(frontend.get("methods", [])) != 15:
        raise ValueError("frontend fit plan no longer binds the complete 360-row/15-method family")

    selector_path = args.selector_source.resolve(strict=True)
    protocol_path = args.protocol_source.resolve(strict=True)
    profile_path = args.profile_source.resolve(strict=True)
    core_path = args.stagea_core.resolve(strict=True)
    frontend_runner_path = args.cross_frontend_runner.resolve(strict=True)
    dispatcher_path = args.profile_dispatcher.resolve(strict=True)
    for path, expected, label in (
        (selector_path, SELECTOR_SOURCE_SHA256, "selector source"),
        (protocol_path, PROTOCOL_SOURCE_SHA256, "selector protocol"),
        (profile_path, PROFILE_SOURCE_SHA256, "profile source"),
        (core_path, STAGEA_CORE_SHA256, "Stage-A core"),
        (frontend_runner_path, CROSS_FRONTEND_RUNNER_SHA256, "cross-frontend runner"),
        (dispatcher_path, PROFILE_DISPATCHER_SHA256, "profile dispatcher"),
    ):
        if sha_file(path) != expected:
            raise ValueError(f"{label} source SHA mismatch")

    selection_path = args.selection_report.resolve(strict=True)
    selection_raw = selection_path.read_bytes()
    selection_sha = sha_bytes(selection_raw)
    if selection_sha != args.expected_selection_report_raw_sha256.lower():
        raise ValueError(f"selector report raw SHA mismatch: {selection_sha}")
    if len(selection_sha) != 64:
        raise ValueError("selector report SHA is not lowercase SHA-256")
    selection_report = json.loads(selection_raw.decode("utf-8"))
    if selection_report.get("schema") != "cross_tfg_shared_profile_selection_input_binding_v1":
        raise ValueError("selector report schema differs from the frozen fit-only adapter")
    if selection_report.get("role") != "fit_only_historical_exposed_descriptive_selection":
        raise ValueError("selector report does not declare fit-only descriptive selection")
    if selection_report.get("report_sha256") != payload_sha(selection_report, "report_sha256"):
        raise ValueError("selector report payload seal mismatch")
    if selection_report.get("consumer_issue_count") != 0 or selection_report.get("consumer_verified_rows") != 360:
        raise ValueError("selector report is not bound to the complete 360-row consumer audit")
    if selection_report.get("input_sha256", {}).get("plan") != FRONTEND_PLAN_SHA256:
        raise ValueError("selector report points at a different frozen fit plan")
    selection = selection_report.get("selection")
    if not isinstance(selection, dict) or selection.get("shortlist_status") != "READY_FOR_FIT_ONLY_REVIEW":
        raise ValueError("selector has no complete fit-only shortlist for review")
    selected_ids = selection.get("shortlist_profile_ids")
    if not isinstance(selected_ids, list) or not 1 <= len(selected_ids) <= 3:
        raise ValueError("selector shortlist must contain one to three unique profiles")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selector shortlist contains duplicates")

    raw_qc_path = args.raw_matrix_qc.resolve(strict=True)
    raw_qc_bytes = raw_qc_path.read_bytes()
    raw_qc_sha = sha_bytes(raw_qc_bytes)
    if raw_qc_sha != args.expected_raw_matrix_qc_raw_sha256.lower():
        raise ValueError(f"independent raw-matrix QC raw SHA mismatch: {raw_qc_sha}")
    if len(raw_qc_sha) != 64:
        raise ValueError("raw-matrix QC SHA is not lowercase SHA-256")
    raw_qc = json.loads(raw_qc_bytes.decode("utf-8"))
    raw_qc_status = str(raw_qc.get("status", raw_qc.get("decision", ""))).upper()
    if "PASS" not in raw_qc_status or "INCOMPLETE" in raw_qc_status or "FAIL" in raw_qc_status:
        raise ValueError(f"raw-matrix QC is not an explicit passing receipt: {raw_qc_status!r}")

    from scripts.audio_enhancement.shared_frequency_profile_v1 import PROFILES

    profile_ids = {profile.profile_id for profile in PROFILES}
    if any(profile_id not in profile_ids for profile_id in selected_ids):
        raise ValueError("selector returned a profile outside the pinned 12-profile family")
    profile_methods = {
        method["method_id"]: method
        for method in frontend["methods"]
        if method.get("method_id") in profile_ids
    }
    if set(profile_methods) != profile_ids:
        raise ValueError("frontend fit plan does not contain the complete pinned profile family")
    selected_results = {row.get("profile_id"): row for row in selection.get("candidate_results", [])}
    for profile_id in selected_ids:
        candidate = selected_results.get(profile_id)
        if not isinstance(candidate, dict) or candidate.get("status") != "COMPLETE":
            raise ValueError(f"selected profile is not complete on every declared fit context: {profile_id}")

    identity_rows = [row for row in base_rows if row.get("method_id") == "identity"]
    if len(identity_rows) != 60:
        raise ValueError("base dev plan must contain 60 identity source contexts")
    context_keys = [(row["sample_id"], row["speaker_id"], int(row["seed"])) for row in identity_rows]
    if len(context_keys) != len(set(context_keys)):
        raise ValueError("base identity context keys are duplicated")
    if {row.get("split") for row in identity_rows} != {"dev"}:
        raise ValueError("candidate inputs must come only from the frozen dev split")
    if len({row["sample_id"] for row in identity_rows}) != 20:
        raise ValueError("base identity rows no longer cover exactly 20 dev clips")
    if len({row["speaker_id"] for row in identity_rows}) != 10:
        raise ValueError("base identity rows no longer cover exactly 10 dev speakers")
    if {int(row["seed"]) for row in identity_rows} != {20261003, 20261004, 20261005}:
        raise ValueError("base identity rows no longer carry all three frozen paired seeds")
    for sample_id in {row["sample_id"] for row in identity_rows}:
        matching = [row for row in identity_rows if row["sample_id"] == sample_id]
        if len(matching) != 3 or len({row["speaker_id"] for row in matching}) != 1:
            raise ValueError(f"sample seed/speaker pairing is incomplete: {sample_id}")
        if len({canonical([row["natural_pcm"], row["score_audio"]]) for row in matching}) != 1:
            raise ValueError(f"natural/source-score PCM changed across seeds: {sample_id}")
    face_bindings = {canonical(row.get("source_face")) for row in identity_rows}
    video_bindings = {canonical(row.get("source_video")) for row in identity_rows}
    if len(face_bindings) != 1 or len(video_bindings) != 1:
        raise ValueError("base dev identities do not share the one frozen portrait and source-video binding")

    methods: list[dict[str, Any]] = []
    for profile_id in selected_ids:
        method = copy.deepcopy(profile_methods[profile_id])
        config = method.get("config")
        if not isinstance(config, dict) or config.get("profile_source_sha256") != PROFILE_SOURCE_SHA256:
            raise ValueError(f"selected method config does not pin the frequency transform: {profile_id}")
        method["selection_report_sha256"] = selection_sha
        method["raw_matrix_qc_sha256"] = raw_qc_sha
        method["selection_scope"] = "fit-only descriptive shortlist; frozen before candidate dev renders"
        methods.append(method)

    model_by_id = {model["model_id"]: model for model in model_records}
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_id = method["method_id"]
        for source_row in identity_rows:
            for model_id in EXPECTED_MODELS:
                model = model_by_id[model_id]
                row = copy.deepcopy(source_row)
                row["model_id"] = model_id
                row["audio_frontend"] = model["audio_frontend"]
                row["model_role"] = model.get("role", "candidate")
                row["method_id"] = method_id
                row["method_config_sha256"] = method["config_sha256"]
                row["checkpoint_sha256"] = None
                row["historical_exposed"] = False
                row["selection_report_sha256"] = selection_sha
                row["raw_matrix_qc_sha256"] = raw_qc_sha
                settings = copy.deepcopy(row["generation_settings"])
                settings.update({
                    "generator": model_id,
                    "input_audio": "shared_frequency_profile_condition_only; score_audio_is_original_natural_pcm",
                    "audio_transform": method_id,
                    "profile_source_sha256": PROFILE_SOURCE_SHA256,
                    "selection_report_sha256": selection_sha,
                    "seed": int(source_row["seed"]),
                })
                row["generation_settings"] = settings
                row["row_id"] = (
                    f"dev_{row['speaker_id']}_{row['sample_id']}_{model_id}_"
                    f"{row['face_id']}_{int(row['seed'])}_{method_id}"
                )
                rows.append(row)

    expected_rows = len(selected_ids) * 180
    if len(rows) != expected_rows:
        raise ValueError(f"candidate-only row count {len(rows)} differs from {expected_rows}")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("candidate-only plan has duplicate row IDs")
    for method_id in selected_ids:
        count = sum(row["method_id"] == method_id for row in rows)
        if count != 180:
            raise ValueError(f"method {method_id} does not have 180 dev rows")

    plan = copy.deepcopy(base)
    plan["run_id"] = RUN_ID
    plan["methods"] = methods
    plan["models"] = copy.deepcopy(model_records)
    plan["rows"] = rows
    plan["protocol"] = copy.deepcopy(frontend["protocol"])
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["simultaneous_family_ids"] = []
    plan["analysis"]["simultaneous_family_status"] = "NOT_APPLICABLE_CANDIDATE_ONLY_SUBRUN"
    plan["analysis"]["candidate_only_subrun"] = {
        "selected_profile_ids": list(selected_ids),
        "fit_selector_report_sha256": selection_sha,
        "independent_raw_matrix_qc_sha256": raw_qc_sha,
        "selection_scope": "fit-only historical data; descriptive shortlist; no inferential claim",
        "baseline_policy": "Reuse only previously completed PASS rows with exact PCM, face, model, seed, transform/config, score model, preprocessing, and normalized source-clock bindings; preserve original source run/row IDs.",
    }
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "shared_frequency_profile_candidate_only_development_render",
        "scope": "development-only; 20 fixed dev clips × 3 frozen render seeds × 3 qualified frontends × shortlisted profiles; confirmation remains closed",
        "planned_rows": expected_rows,
        "candidate_only": True,
        "fresh_baseline_rerender": False,
        "baseline_join": "pending exact input/face/model/seed/transform/score/clock verification after candidate completion",
        "selected_profile_ids": list(selected_ids),
        "selection_report_sha256": selection_sha,
        "raw_matrix_qc_sha256": raw_qc_sha,
        "stagea_core_sha256": STAGEA_CORE_SHA256,
        "stagea_runner_sha256": STAGEA_CORE_SHA256,
        "cross_frontend_runner_sha256": CROSS_FRONTEND_RUNNER_SHA256,
        "cross_frontend_dispatcher_sha256": PROFILE_DISPATCHER_SHA256,
        "profile_transform_source_sha256": PROFILE_SOURCE_SHA256,
        "generation_repeat_qc": "NOT_RUN_IN_CANDIDATE_ONLY_SUBRUN; prior control repeat records remain separate and are not relabeled as candidate repeats",
        "confirm_audio_read": False,
        "confirm_effects_read": False,
    }
    generator = plan["protocol"].get("generator")
    if not isinstance(generator, dict):
        raise ValueError("frontend fit plan has no generator protocol")
    generator["stagea_core_sha256"] = STAGEA_CORE_SHA256
    generator["cross_frontend_runner_sha256"] = CROSS_FRONTEND_RUNNER_SHA256
    generator["cross_frontend_dispatcher_sha256"] = PROFILE_DISPATCHER_SHA256
    generator["profile_transform_source_sha256"] = PROFILE_SOURCE_SHA256
    for model in plan["models"]:
        if isinstance(model.get("generator"), dict):
            model["generator"]["runner_sha256"] = STAGEA_CORE_SHA256
    plan["source_reference_plans"] = {
        "base_dev_plan": {
            "path": str(args.base_dev_plan),
            "raw_sha256": sha_bytes(base_raw),
            "plan_sha256": base.get("plan_sha256"),
            "planned_rows": len(base_rows),
        },
        "frontend_fit_plan": {
            "path": str(args.frontend_fit_plan),
            "raw_sha256": sha_bytes(frontend_raw),
            "plan_sha256": frontend.get("plan_sha256"),
            "model_ids": list(EXPECTED_MODELS),
        },
        "fit_selector_report": {
            "path": str(selection_path),
            "raw_sha256": selection_sha,
            "report_sha256": selection_report.get("report_sha256"),
            "shortlist_profile_ids": list(selected_ids),
        },
        "independent_raw_matrix_qc": {
            "path": str(raw_qc_path),
            "raw_sha256": raw_qc_sha,
            "status": raw_qc_status,
        },
        "selector_source": {"path": str(selector_path), "raw_sha256": sha_file(selector_path)},
        "protocol_source": {"path": str(protocol_path), "raw_sha256": sha_file(protocol_path)},
        "profile_source": {"path": str(profile_path), "raw_sha256": sha_file(profile_path)},
        "stagea_core": {"path": str(core_path), "raw_sha256": sha_file(core_path)},
        "cross_frontend_runner": {"path": str(frontend_runner_path), "raw_sha256": sha_file(frontend_runner_path)},
        "profile_dispatcher": {"path": str(dispatcher_path), "raw_sha256": sha_file(dispatcher_path)},
    }
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")

    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    plan_bytes = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    rows_bytes = b"".join(canonical(row) + b"\n" for row in rows)
    plan_path = args.output_dir / "plan.json"
    rows_path = args.output_dir / "planned_rows.jsonl"
    write_new(plan_path, plan_bytes)
    write_new(rows_path, rows_bytes)
    manifest = {
        "schema": "cross_tfg_shared_profile_shortlist_candidate_only_dev_plan_v1",
        "run_id": RUN_ID,
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": sha_bytes(plan_bytes),
        "planned_rows_sha256": sha_bytes(rows_bytes),
        "planned_rows": len(rows),
        "method_counts": {method_id: sum(row["method_id"] == method_id for row in rows) for method_id in selected_ids},
        "model_counts": {model_id: sum(row["model_id"] == model_id for row in rows) for model_id in EXPECTED_MODELS},
        "sample_count": 20,
        "speaker_count": 10,
        "render_seed_family": [20261003, 20261004, 20261005],
        "selected_profile_ids": list(selected_ids),
        "selection_report_sha256": selection_sha,
        "raw_matrix_qc_sha256": raw_qc_sha,
        "confirm_audio_read": False,
        "confirm_effects_read": False,
        "baseline_join_pending_exact_row_qualification": True,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path = args.output_dir / "plan_manifest.json"
    write_new(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({
        "status": "FROZEN",
        "run_id": RUN_ID,
        "plan_raw_sha256": sha_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "planned_rows_sha256": sha_file(rows_path),
        "rows": len(rows),
        "models": list(EXPECTED_MODELS),
        "profile_ids": list(selected_ids),
        "selection_report_sha256": selection_sha,
        "raw_matrix_qc_sha256": raw_qc_sha,
        "confirm_audio_read": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
