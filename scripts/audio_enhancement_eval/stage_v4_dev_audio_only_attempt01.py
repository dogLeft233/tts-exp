#!/usr/bin/env python3
"""CPU-stage the frozen v4 waveform for the 20 development clips.

This is an audio-only readback. It does not render video, call SyncNet, or make
ASR requests. It replays the frozen Stage-A waveform transform on CPU, records
the RMS-stage and final peak-limit activity, and leaves a sealed F32 WAV pair
for each natural/candidate utterance. Confirmatory inputs are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-v4-dev-audio-stage-v1"
EXPECTED_RUN = "cross_tfg_audio_head_20261003_v4_dev180_attempt02"
EXPECTED_METHOD = "spectral_head_fit_v4_native_attempt01"
EXPECTED_CHECKPOINT = "3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4"
EXPECTED_PLAN_RAW = "70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c"


class StageError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical_json({k: v for k, v in value.items() if k != field}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageError(f"expected JSON object: {path}")
    return value


def safe_file(path: Path, where: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise StageError(f"{where} must be a regular non-symlink file: {path}")
    return path


def rms64(samples: Any) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype=np.float64)))))


def _source_map(repo_root: Path, transform_source_sha256: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "audio_enhancement_heads.py": repo_root / "scripts/audio_enhancement/heads.py",
        "audio_enhancement_pipeline.py": repo_root / "scripts/audio_enhancement/pipeline.py",
        "run_cross_frontend_dev_attempt03.py": repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py",
        "run_cross_frontend_fit_profiles_attempt02.py": repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_fit_profiles_attempt02.py",
        "run_stagea_dev_attempt03.py": repo_root / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        expected = transform_source_sha256.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise StageError(f"missing frozen transform source pin: {name}")
        safe_file(path, f"source {name}")
        actual = sha_file(path)
        if actual != expected:
            raise StageError(f"transform source SHA mismatch for {name}: {actual}")
        observed[name] = actual
    return observed


def stage(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = safe_file(Path(args.plan).resolve(strict=True), "plan")
    plan_raw = sha_file(plan_path)
    if plan_raw != args.expected_plan_raw_sha256 or plan_raw != EXPECTED_PLAN_RAW:
        raise StageError("frozen v4 plan raw SHA differs from both the CLI pin and source constant")
    plan = read_json(plan_path)
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise StageError("v4 plan schema/status is not frozen")
    if plan.get("run_id") != EXPECTED_RUN or plan.get("confirm_speaker_ids"):
        raise StageError("unexpected v4 run or confirmation roster is open")
    if plan.get("plan_sha256") != payload_sha(plan, "plan_sha256"):
        raise StageError("v4 plan embedded payload hash mismatch")
    seeds = plan.get("generation_seed_family", {}).get("dev")
    if seeds != [20261003, 20261004, 20261005] or plan.get("generation_seed_family", {}).get("confirm") != []:
        raise StageError("development seed family differs from frozen v4 roster")
    methods = plan.get("methods")
    if not isinstance(methods, list):
        raise StageError("plan method registry is missing")
    selected = [m for m in methods if isinstance(m, dict) and m.get("method_id") == EXPECTED_METHOD]
    if len(selected) != 1:
        raise StageError("frozen v4 method is missing or duplicated")
    method = selected[0]
    config = method.get("config")
    if not isinstance(config, dict):
        raise StageError("v4 transform config is missing")
    if method.get("kind") != "learned_audio_head_candidate" or method.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
        raise StageError("v4 method kind/checkpoint differs from the frozen candidate")
    if config.get("checkpoint_sha256") != EXPECTED_CHECKPOINT or config.get("method") != "spectral_head":
        raise StageError("v4 transform config does not bind the frozen checkpoint")
    if method.get("config_sha256") != "e1fb7fa010dbc6ea22e8362230f144f00dcea09bfd34500e0c2cdeed1fb08cd7":
        raise StageError("v4 transform config SHA differs from the frozen method")
    if len(plan.get("rows", [])) != 180:
        raise StageError("frozen v4 plan is not 180 rows")
    model_ids = {m.get("model_id") for m in plan.get("models", []) if isinstance(m, dict)}
    if model_ids != {"wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18"}:
        raise StageError("frozen v4 frontend roster differs")

    expected_rows = [r for r in plan["rows"] if r.get("method_id") == EXPECTED_METHOD]
    samples: dict[str, dict[str, Any]] = {}
    for row in expected_rows:
        if row.get("split") != "dev" or row.get("method_config_sha256") != method["config_sha256"]:
            raise StageError(f"unexpected row split/config: {row.get('row_id')}")
        sample_id = str(row.get("sample_id"))
        identity = {
            "sample_id": sample_id,
            "speaker_id": row.get("speaker_id"),
            "natural_pcm": row.get("natural_pcm"),
            "score_audio": row.get("score_audio"),
            "source_face": row.get("source_face"),
            "source_video": row.get("source_video"),
        }
        if identity["natural_pcm"] != identity["score_audio"]:
            raise StageError(f"score audio differs from natural PCM for {sample_id}")
        old = samples.setdefault(sample_id, identity)
        if old != identity:
            raise StageError(f"sample identity/natural PCM differs across method contexts: {sample_id}")
        if row.get("seed") not in seeds or row.get("model_id") not in model_ids:
            raise StageError(f"unexpected v4 planned context: {row.get('row_id')}")
        samples[sample_id].setdefault("planned_rows", []).append({
            "row_id": row["row_id"], "model_id": row["model_id"], "seed": row["seed"]
        })
    if len(samples) != 20 or any(len(s["planned_rows"]) != 9 for s in samples.values()):
        raise StageError("planned v4 audio roster must be 20 samples × 3 frontends × 3 seeds")
    if len({s["speaker_id"] for s in samples.values()}) != 10:
        raise StageError("planned v4 audio roster must contain 10 speakers")

    repo_root = Path(args.project_root).resolve(strict=True)
    transform_source_pins = config.get("transform_source_sha256")
    if not isinstance(transform_source_pins, dict):
        raise StageError("transform source SHA map is missing")
    source_shas = _source_map(repo_root, transform_source_pins)
    checkpoint = safe_file(Path(config["checkpoint_path"]).resolve(strict=True), "v4 checkpoint")
    if sha_file(checkpoint) != EXPECTED_CHECKPOINT or args.expected_checkpoint_sha256 != EXPECTED_CHECKPOINT:
        raise StageError("checkpoint bytes do not match the frozen v4 weight SHA")

    out = Path(args.out_dir).resolve()
    if out.exists():
        raise StageError(f"output must be a new path: {out}")
    out.mkdir(parents=True)
    natural_dir, enhanced_dir = out / "natural_pcm", out / "v4_candidate_pcm"
    natural_dir.mkdir(); enhanced_dir.mkdir()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import numpy as np
    import torch
    from scripts.audio_enhancement.audio_io import pcm_float32_sha256, read_canonical_wav, write_float32_wav
    from scripts.audio_enhancement.heads import EnhancementStack
    from scripts.experiments.cross_tfg_audio_head_20261003 import run_stagea_dev_attempt03 as stagea

    torch.set_num_threads(1)
    torch.manual_seed(0)
    head, loaded_checkpoint_sha = stagea._load_spectral_head(method)
    if loaded_checkpoint_sha != EXPECTED_CHECKPOINT:
        raise StageError("Stage-A loader returned a different checkpoint pin")
    stack = EnhancementStack(head).cpu().eval()
    rows_out: list[dict[str, Any]] = []
    for sample_id, sample in sorted(samples.items()):
        natural_ref = sample["natural_pcm"]
        natural_src = safe_file(Path(natural_ref["path"]).resolve(strict=True), f"natural source {sample_id}")
        natural_raw_sha = sha_file(natural_src)
        if natural_raw_sha != natural_ref["file_sha256"]:
            raise StageError(f"natural source file SHA mismatch: {sample_id}")
        natural, natural_info = read_canonical_wav(natural_src)
        if (
            natural_info.pcm_float32_sha256 != natural_ref["pcm_sha256"]
            or natural_info.frames != natural_ref["sample_count"]
            or natural_info.sample_rate != 16000
            or natural_info.source_channels != 1
            or natural_info.source_encoding != "FLOAT_32"
        ):
            raise StageError(f"canonical natural PCM metadata/hash mismatch: {sample_id}")
        natural_dst = natural_dir / f"{sample_id}.natural_f32.wav"
        shutil.copyfile(natural_src, natural_dst)
        if sha_file(natural_dst) != natural_raw_sha:
            raise StageError(f"staged natural copy differs byte-for-byte: {sample_id}")

        transformed, producer_diag = stagea._transform(natural, method)
        x = torch.from_numpy(np.ascontiguousarray(natural, dtype=np.float32))
        rms_ceiling = config.get("rms_stage_peak_ceiling")
        peak_ceiling = config.get("peak_ceiling")
        diag_kwargs = {
            "use_global_rms": bool(config.get("use_global_rms", False)),
            "use_local_envelope": bool(config.get("use_local_envelope", False)),
            "use_spectral_head": bool(config.get("use_spectral_head", True)),
            "local_window_ms": float(config.get("local_window_ms", 32.0)),
            "local_ratio": float(config.get("local_ratio", 0.8)),
            "local_min_gain_db": float(config.get("local_min_gain_db", -6.0)),
            "local_max_gain_db": float(config.get("local_max_gain_db", 6.0)),
            "post_rms_match": str(config.get("post_rms_match", "none")),
            "target_rms": float(config.get("target_rms", 0.0376838172675746)),
            "rms_stage_peak_ceiling": rms_ceiling,
            "peak_ceiling": peak_ceiling,
            "return_diagnostics": True,
        }
        with torch.inference_mode():
            stack_output, log_mask, torch_diag = stack.forward_with_mask(x, **diag_kwargs)
        stack_samples = np.ascontiguousarray(stack_output.detach().cpu().numpy(), dtype=np.float32)
        transformed = np.ascontiguousarray(transformed, dtype=np.float32)
        replay_delta = np.abs(stack_samples.astype(np.float64) - transformed.astype(np.float64))
        if not np.array_equal(stack_samples, transformed):
            raise StageError(f"diagnostic stack replay differs from Stage-A pipeline transform: {sample_id}")

        enhanced_dst = enhanced_dir / f"{sample_id}.v4_f32.wav"
        enhanced_ref = write_float32_wav(enhanced_dst, transformed, sample_rate=16000, peak_ceiling=None, overwrite=False)
        enhanced_read, enhanced_info = read_canonical_wav(enhanced_dst)
        if (
            enhanced_info.pcm_float32_sha256 != enhanced_ref.pcm_float32_sha256
            or enhanced_info.frames != natural_info.frames
            or not np.array_equal(enhanced_read, transformed)
        ):
            raise StageError(f"staged candidate WAV readback differs: {sample_id}")

        input_rms = rms64(natural)
        input_peak = float(np.max(np.abs(natural.astype(np.float64))))
        target_rms = float(config["target_rms"])
        requested_gain = target_rms / max(input_rms, 1e-12)
        rms_applied_gain = requested_gain
        if rms_ceiling is not None and input_peak > 0:
            rms_applied_gain = min(requested_gain, float(rms_ceiling) / input_peak)
        stage1_peak = input_peak * rms_applied_gain
        stage_rms_active = bool(rms_ceiling is not None and input_peak * requested_gain > float(rms_ceiling))
        final_peak_gain = float(torch_diag["final_peak_gain"].detach().reshape(-1)[0].item())
        final_peak_active = bool(final_peak_gain < 1.0 - 1e-12)
        mask_db = log_mask.detach().cpu().numpy().astype(np.float64) * (20.0 / math.log(10.0))
        rows_out.append({
            "sample_id": sample_id,
            "speaker_id": sample["speaker_id"],
            "split": "dev",
            "status": "PASS",
            "planned_row_ids": [r["row_id"] for r in sample["planned_rows"]],
            "planned_frontends": sorted({r["model_id"] for r in sample["planned_rows"]}),
            "planned_seeds": sorted({int(r["seed"]) for r in sample["planned_rows"]}),
            "natural_pcm": {
                "path": str(natural_dst), "source_path": str(natural_src),
                "file_sha256": sha_file(natural_dst), "source_file_sha256": natural_raw_sha,
                "staged_file_sha256": sha_file(natural_dst),
                "pcm_float32_sha256": natural_info.pcm_float32_sha256,
                "sample_rate_hz": natural_info.sample_rate, "sample_count": natural_info.frames,
            },
            "candidate_pcm": {
                "path": str(enhanced_dst), "file_sha256": enhanced_ref.file_sha256,
                "pcm_float32_sha256": enhanced_info.pcm_float32_sha256,
                "sample_rate_hz": enhanced_info.sample_rate, "sample_count": enhanced_info.frames,
            },
            "transform_replay": {
                "implementation": "pinned Stage-A _transform plus pinned EnhancementStack CPU diagnostic replay",
                "pipeline_vs_stack_exact_float32": True,
                "max_abs_delta": float(replay_delta.max(initial=0.0)),
                "producer_transform_diagnostics": producer_diag,
            },
            "peak_activity": {
                "rms_target": target_rms,
                "rms_stage_peak_ceiling": rms_ceiling,
                "requested_global_rms_gain": requested_gain,
                "applied_global_rms_gain": rms_applied_gain,
                "rms_stage_peak_before_limit": input_peak * requested_gain,
                "rms_stage_peak_after_limit": stage1_peak,
                "rms_stage_peak_limiter_active": stage_rms_active,
                "after_global_rms": float(torch_diag["after_global_rms"].reshape(-1)[0].item()),
                "after_head_rms": float(torch_diag["after_head_rms"].reshape(-1)[0].item()),
                "after_rms_match_rms": float(torch_diag["after_rms_match_rms"].reshape(-1)[0].item()),
                "final_peak_ceiling": peak_ceiling,
                "final_peak_gain": final_peak_gain,
                "final_peak_limiter_active": final_peak_active,
                "output_rms": rms64(transformed),
                "output_peak_abs": float(np.max(np.abs(transformed.astype(np.float64)))),
                "log_mask_db_min": float(mask_db.min()),
                "log_mask_db_max": float(mask_db.max()),
                "log_mask_db_mean": float(mask_db.mean()),
            },
        })

    ids = [r["sample_id"] for r in rows_out]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise StageError("audio staging did not produce exactly one unique row for each of 20 dev clips")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "source_run_id": plan["run_id"],
        "source_plan_path": str(plan_path),
        "source_plan_raw_sha256": plan_raw,
        "source_plan_payload_sha256": plan["plan_sha256"],
        "method_id": EXPECTED_METHOD,
        "method_config_sha256": method["config_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "project_root": str(repo_root),
        "transform_source_sha256": source_shas,
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": __import__("numpy").__version__,
            "torch_version": __import__("torch").__version__,
            "device": "cpu",
            "torch_num_threads": __import__("torch").get_num_threads(),
        },
        "planned_rows": 180,
        "planned_unique_samples": 20,
        "planned_speakers": 10,
        "planned_frontends": sorted(model_ids),
        "planned_seeds": list(seeds),
        "confirmation_audio_read": False,
        "video_rendered": False,
        "syncnet_called": False,
        "asr_called": False,
        "peak_ceiling_semantics": "Both 0.99 RMS-stage cap and final 0.99 peak cap are audited separately; the baseline rms_fixed method has no peak ceiling.",
        "rows": rows_out,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path = out / "audio_only_candidate_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    return {
        "status": manifest["status"], "manifest_path": str(manifest_path),
        "manifest_raw_sha256": sha_file(manifest_path), "manifest_payload_sha256": manifest["manifest_sha256"],
        "samples": 20, "speakers": 10, "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "asr_called": False, "confirmation_audio_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-raw-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        result = stage(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
