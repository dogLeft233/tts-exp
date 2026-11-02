#!/usr/bin/env python3
"""Freeze one fresh, paired 240-row development run including fit07 head."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any

BASE_PLAN_SHA = "bf68725df298d68b4205ef205236e4046b4ba180c8e6da622fcaa9b039d38585"
FIT_PLAN_SHA = "34b474fbc740103142b170d8d07e271b47b0bf0fc76fc5056666048fd9f0d956"
FIT_REPORT_SHA = "c5ea4c3877945719a1c680646629f65fb73b8e59883ad9a0c755adde1ffc58d0"
CHECKPOINT_SHA = "bc5aeec15f9197ca0c1d0d606eebe41ff2dbf6620767eee5bc37ca4b9395aaed"
BASE_METHODS = ("identity", "rms_fixed", "rms_localcompress")

def sha(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()
def canon(value: Any) -> bytes: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
def seal(value: dict[str, Any], field: str) -> str:
    body = dict(value); body.pop(field, None); return sha(canon(body))
def write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as f: f.write(raw); f.flush()

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--pipeline", type=Path, required=True)
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--fit-report-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    raw = a.base_plan.read_bytes()
    if sha(raw) != BASE_PLAN_SHA: raise ValueError("base plan raw SHA mismatch")
    if a.checkpoint_sha256.lower() != CHECKPOINT_SHA or a.fit_report_sha256.lower() != FIT_REPORT_SHA:
        raise ValueError("fit07 report/checkpoint pins mismatch")
    base = json.loads(raw)
    if base.get("status") != "FROZEN" or base.get("plan_sha256") != seal(base, "plan_sha256"):
        raise ValueError("base plan payload seal invalid")
    if len(base.get("rows", [])) != 180 or [m["method_id"] for m in base["methods"]] != list(BASE_METHODS):
        raise ValueError("base plan row/method inventory mismatch")
    runner_sha, pipeline_sha, heads_sha = (sha(x.read_bytes()) for x in (a.runner, a.pipeline, a.heads))
    architecture = {"class":"scripts.audio_enhancement.heads.SpectralResidualHead", "n_fft":512, "hop_length":128,
        "hidden_channels":24, "max_residual_db":3.0, "level_normalize":True, "frequency_center_mask":False,
        "frame_energy_normalize":False, "sample_rate_hz":16000,
        "identity_initialization":"final_conv_zero_weight_and_bias"}
    config = {"method":"spectral_head", "sample_rate_hz":16000, "use_global_rms":True,
        "use_local_envelope":False, "use_spectral_head":True, "post_rms_match":"target",
        "target_rms":0.0376838172675746, "rms_stage_peak_ceiling":0.99, "peak_ceiling":0.99,
        "local_window_ms":32.0, "local_ratio":0.8, "local_min_gain_db":-6.0, "local_max_gain_db":6.0,
        "head_architecture":architecture, "checkpoint_path":a.checkpoint_path,
        "checkpoint_sha256":CHECKPOINT_SHA, "fit_report_sha256":FIT_REPORT_SHA,
        "fit_plan_raw_sha256":FIT_PLAN_SHA, "fit_optimizer_steps":24,
        "training_objective":"31-lag median-minus-normalized-softmin tau=0.1; historical fit only",
        "transform_source_sha256":{"run_stagea_dev_attempt02.py":runner_sha,
            "audio_enhancement_pipeline.py":pipeline_sha,"audio_enhancement_heads.py":heads_sha}}
    method = {"method_id":"spectral_head_fit24", "kind":"learned_audio_head_candidate", "config":config,
        "config_sha256":sha(canon(config)), "checkpoint_sha256":CHECKPOINT_SHA,
        "fit_report_sha256":FIT_REPORT_SHA, "fit_plan_raw_sha256":FIT_PLAN_SHA,
        "training_steps":24, "selection_scope":"development_only; no confirmation audio opened"}

    # Force a stable paired order: context tuple, seed, then all four methods.
    by_context: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    context_order: list[tuple[Any, ...]] = []
    for r in base["rows"]:
        key = (r["split"],r["sample_id"],r["speaker_id"],r["model_id"],r["face_id"],r["seed"])
        if key not in by_context: by_context[key] = {}; context_order.append(key)
        if r["method_id"] in by_context[key]: raise ValueError("duplicate base row in context")
        by_context[key][r["method_id"]] = r
    rows: list[dict[str, Any]] = []
    for key in context_order:
        methods = by_context[key]
        if tuple(methods) != BASE_METHODS: raise ValueError(f"base methods/order mismatch for {key}")
        for method_id in BASE_METHODS:
            rows.append(copy.deepcopy(methods[method_id]))
        source = methods["identity"]
        head = copy.deepcopy(source)
        head["row_id"] = source["row_id"].rsplit("_identity",1)[0] + "_spectral_head_fit24"
        head["method_id"] = method["method_id"]
        head["method_config_sha256"] = method["config_sha256"]
        head["checkpoint_sha256"] = CHECKPOINT_SHA
        head["generation_settings"] = {**head["generation_settings"],
            "audio_transform":"fit07_spectral_head_full_audio", "fit_report_sha256":FIT_REPORT_SHA,
            "head_checkpoint_sha256":CHECKPOINT_SHA}
        rows.append(head)
    if len(rows) != 240: raise ValueError(f"expected 240 rows, got {len(rows)}")

    plan = copy.deepcopy(base)
    plan["run_id"] = "cross_tfg_audio_head_20261003_stageb_dev_attempt02"
    plan["methods"].append(method)
    plan["rows"] = rows
    plan["execution"] = {**plan["execution"], "stage":"B_combined_fresh_rerender",
        "scope":"dev-only; all four methods freshly rendered under one paired plan; no confirmation audio/effects read",
        "planned_rows":240, "stagea_reference_plan_raw_sha256":BASE_PLAN_SHA,
        "fresh_baseline_rows":180, "fit07_candidate_rows":60,
        "candidate_checkpoint_sha256":CHECKPOINT_SHA, "candidate_fit_report_sha256":FIT_REPORT_SHA,
        "combined_runner_sha256":runner_sha,
        "failure_history":"stagea_dev_attempt01 first row serialization failed before any sealed efficacy row; retained separately",
        "candidate_selection":"development_only_after_all_predeclared_rows_and_independent_consumer_audit"}
    plan["execution"]["stagea_runner_sha256"] = runner_sha
    plan["execution"]["primary_pcm16_lineage_code_sha256"] = "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"
    plan["execution"]["official_syncnet_evaluator_wrapper_sha256"] = "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b"
    contrasts = list(plan["contrasts"]) + [
        {"contrast_id":"spectral_head_fit24_minus_identity","method_a":"spectral_head_fit24","method_b":"identity"},
        {"contrast_id":"spectral_head_fit24_minus_rms_fixed","method_a":"spectral_head_fit24","method_b":"rms_fixed"},
        {"contrast_id":"spectral_head_fit24_minus_rms_localcompress","method_a":"spectral_head_fit24","method_b":"rms_localcompress"}]
    plan["contrasts"] = contrasts
    plan["analysis"]["contrasts"] = contrasts
    plan["analysis"]["summary_endpoints"] = [
        {"endpoint_id":f"dev_{c['contrast_id']}_{support}_Cbest", "split":"dev",
         "contrast_id":c["contrast_id"], "support_id":support, "metric":"Cbest",
         "model_scope":"model", "model_id":plan["models"][0]["model_id"],
         "roi_estimand":"native_official", "audio_estimand":"native_roi_original_pcm16", "exploratory":True}
        for c in contrasts for support in ("full","guard20")]
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = seal(plan, "plan_sha256")
    if a.output_dir.exists(): raise FileExistsError(f"refusing existing output: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    plan_bytes = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    rows_bytes = b"".join(canon(r)+b"\n" for r in rows)
    write_new(a.output_dir/"plan.json", plan_bytes)
    write_new(a.output_dir/"planned_rows.jsonl", rows_bytes)
    manifest = {"schema":"cross_tfg_stageb_combined_plan_v2", "run_id":plan["run_id"],
        "plan_sha256":plan["plan_sha256"], "plan_raw_sha256":sha(plan_bytes),
        "planned_rows_sha256":sha(rows_bytes), "planned_rows":len(rows),
        "method_counts":{m["method_id"]:sum(r["method_id"]==m["method_id"] for r in rows) for m in plan["methods"]},
        "base_stagea_reference_plan_raw_sha256":BASE_PLAN_SHA, "fit_plan_raw_sha256":FIT_PLAN_SHA,
        "fit_report_sha256":FIT_REPORT_SHA, "candidate_checkpoint_sha256":CHECKPOINT_SHA,
        "confirm_speaker_ids_in_plan":0, "development_audio_only":True, "confirm_audio_read":False}
    manifest["manifest_sha256"] = seal(manifest,"manifest_sha256")
    write_new(a.output_dir/"plan_manifest.json",(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
    print(json.dumps({"status":"FROZEN", "run_id":plan["run_id"], "plan_sha256":plan["plan_sha256"],
        "plan_raw_sha256":sha(plan_bytes), "rows":len(rows), "methods":manifest["method_counts"],
        "runner_sha256":runner_sha,"checkpoint_sha256":CHECKPOINT_SHA},sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
