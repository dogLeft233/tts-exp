"""Fit-only CLI for a shared waveform head through injected frozen-model assets.

The dataset plan and file checks live here; a small runtime factory supplies
the pinned Wav2Lip/SyncNet weights plus a precomputed fixed-video context. The
factory is not permitted to select or open dev/confirmation audio.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio_io import read_canonical_wav
from .training import TrainStepInfo, sync_guided_train_step


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_checkpoint_atomic(torch, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _head_architecture(head: Any) -> dict[str, Any]:
    return {
        "class": f"{head.__class__.__module__}.{head.__class__.__qualname__}",
        "n_fft": int(head.n_fft),
        "hop_length": int(head.hop_length),
        "hidden_channels": int(head.mask_net[0].out_channels),
        "max_residual_db": float(head.max_residual_db),
        "level_normalize": bool(head.level_normalize),
        "frequency_center_mask": bool(head.frequency_center_mask),
        "frame_energy_normalize": bool(head.frame_energy_normalize),
        "sample_rate_hz": 16_000,
        "identity_initialization": "final_conv_zero_weight_and_bias",
    }


def _package_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "audio_io.py",
        "baselines.py",
        "differentiable_adapter.py",
        "fit_cli.py",
        "heads.py",
        "overdub.py",
        "pipeline.py",
        "training.py",
        "wav2lip_frontend.py",
    )
    return {name: _sha256_file(package / name) for name in names}


def _resolve_factory(reference: str) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError("adapter factory must use module.path:callable syntax")
    module_name, callable_name = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(factory):
        raise ValueError(f"adapter factory is not callable: {reference}")
    return factory


def _load_fit_rows(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode("utf-8"))
    if not isinstance(plan, dict) or plan.get("status") not in {"FROZEN", "FIT_ONLY_FROZEN"}:
        raise ValueError("fit CLI requires a frozen plan status")
    split_hash = str(plan.get("split_manifest_sha256", ""))
    if len(split_hash) != 64 or any(char not in "0123456789abcdef" for char in split_hash.lower()):
        raise ValueError("frozen plan must bind the split manifest SHA-256")
    fit_speakers = plan.get("fit_speaker_ids")
    dev_speakers = plan.get("dev_speaker_ids")
    confirm_speakers = plan.get("confirm_speaker_ids")
    if not all(isinstance(values, list) for values in (fit_speakers, dev_speakers, confirm_speakers)):
        raise ValueError("frozen plan must enumerate fit/dev/confirm speaker rosters")
    fit_set, dev_set, confirm_set = map(set, (fit_speakers, dev_speakers, confirm_speakers))
    if fit_set & dev_set or fit_set & confirm_set or dev_set & confirm_set:
        raise ValueError("speaker rosters overlap")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen plan has no rows")
    row_ids: set[str] = set()
    fit_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("plan rows must be objects")
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise ValueError("plan row IDs must be nonempty and unique")
        row_ids.add(row_id)
        split = row.get("split")
        speaker = row.get("speaker_id")
        if split == "fit":
            if speaker not in fit_set:
                raise ValueError(f"fit row speaker is outside the frozen fit roster: {speaker}")
            fit_rows.append(row)
        elif split == "dev" and speaker not in dev_set:
            raise ValueError(f"dev row speaker is outside the frozen dev roster: {speaker}")
        elif split == "confirm" and speaker not in confirm_set:
            raise ValueError(f"confirm row speaker is outside the frozen confirmation roster: {speaker}")
        elif split not in {"fit", "dev", "confirm"}:
            raise ValueError(f"unsupported split in frozen plan: {split}")
    if not fit_rows:
        raise ValueError("frozen plan contains no fit rows")
    # Only the fit subset is read below. Dev/confirm paths remain unopened.
    for row in fit_rows:
        natural = row.get("natural_pcm")
        if not isinstance(natural, dict):
            raise ValueError("fit rows must contain natural_pcm refs")
        for key in ("path", "file_sha256", "pcm_sha256", "sample_rate_hz", "channels", "sample_count"):
            if key not in natural:
                raise ValueError(f"fit natural_pcm ref is missing {key}")
    return plan, fit_rows, hashlib.sha256(raw).hexdigest()


def _read_fit_audio(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    record = row["natural_pcm"]
    path = Path(record["path"]).resolve(strict=True)
    actual_file_sha = _sha256_file(path)
    if actual_file_sha != record["file_sha256"]:
        raise ValueError(f"fit natural PCM file SHA mismatch for {row['row_id']}")
    audio, info = read_canonical_wav(path)
    if (
        info.sample_rate != 16_000
        or info.source_channels != 1
        or info.frames != int(record["sample_count"])
        or info.pcm_float32_sha256 != record["pcm_sha256"]
        or int(record["sample_rate_hz"]) != 16_000
        or int(record["channels"]) != 1
    ):
        raise ValueError(f"fit natural PCM metadata/float32 digest mismatch for {row['row_id']}")
    return audio, {
        "path": str(path),
        "file_sha256": actual_file_sha,
        "pcm_sha256": info.pcm_float32_sha256,
        "sample_count": int(info.frames),
        "sample_rate_hz": int(info.sample_rate),
        "channels": 1,
    }


def run_fit(
    *,
    plan_path: Path,
    adapter_factory: str,
    output_dir: Path,
    device: str = "cuda",
    fit_model_id: str | None = None,
    seed: int = 20261003,
    epochs: int = 1,
    max_steps: int | None = None,
    use_global_rms: bool = False,
    use_local_envelope: bool = False,
    local_window_ms: float = 32.0,
    local_ratio: float = 0.8,
    local_min_gain_db: float = -6.0,
    local_max_gain_db: float = 6.0,
    post_rms_match: str = "none",
    target_rms: float = 0.0376838172675746,
    anchor_lag_frames: int = 0,
    identity_weight: float = 0.02,
    spectral_smoothness_weight: float = 0.002,
    peak_ceiling: float | None = None,
    rms_stage_peak_ceiling: float | None = None,
    checkpoint_every_steps: int = 1,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote training runtime
        raise RuntimeError("fit CLI requires PyTorch") from exc
    if epochs <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("epochs/max_steps must be positive")
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be finite and positive")
    if not math.isfinite(local_window_ms) or not 20 <= local_window_ms <= 50:
        raise ValueError("local_window_ms must be finite and within 20--50 ms")
    if not math.isfinite(local_ratio) or not 0 < local_ratio <= 1:
        raise ValueError("local_ratio must be finite and in (0,1]")
    if (
        not math.isfinite(local_min_gain_db)
        or not math.isfinite(local_max_gain_db)
        or local_min_gain_db > 0
        or local_max_gain_db < 0
        or local_min_gain_db > local_max_gain_db
    ):
        raise ValueError("local gain bounds must be finite, ordered, and include 0 dB")
    if not isinstance(anchor_lag_frames, int) or not -15 <= anchor_lag_frames <= 15:
        raise ValueError("anchor_lag_frames must be an integer in [-15,15]")
    if peak_ceiling is not None and (not math.isfinite(peak_ceiling) or peak_ceiling <= 0):
        raise ValueError("peak_ceiling must be positive and finite")
    if rms_stage_peak_ceiling is not None and (
        not math.isfinite(rms_stage_peak_ceiling) or rms_stage_peak_ceiling <= 0
    ):
        raise ValueError("rms_stage_peak_ceiling must be positive and finite")
    plan, fit_rows, plan_sha = _load_fit_rows(plan_path.resolve(strict=True))
    if fit_model_id is not None:
        fit_rows = [row for row in fit_rows if row.get("model_id") == fit_model_id]
        if not fit_rows:
            raise ValueError(f"frozen fit plan has no rows for model {fit_model_id}")
    model_ids = {str(row.get("model_id", "")) for row in fit_rows}
    if len(model_ids) > 1 and fit_model_id is None:
        raise ValueError("fit rows span multiple models; pin --model-id explicitly")
    if any(row.get("method_id") == "identity" for row in fit_rows):
        fit_rows = [row for row in fit_rows if row.get("method_id") == "identity"]
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to reuse an existing fit output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    status_path = destination / "fit_status.json"
    status_record = {
        "schema": "cross_tfg_audio_head_fit_status_v1",
        "status": "RUNNING",
        "plan_path": str(plan_path.resolve()),
        "plan_raw_sha256": plan_sha,
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "fit_model_id": fit_model_id or (next(iter(model_ids)) if model_ids else None),
        "seed": int(seed),
        "planned_fit_rows": len(fit_rows),
        "epochs_requested": int(epochs),
        "max_steps": max_steps,
        "completed_steps": 0,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
    }
    _atomic_json(status_path, status_record)

    # Materialize and verify only fit-split canonical natural audio.
    fit_payloads = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in fit_rows:
        key = (str(row["sample_id"]), str(row["speaker_id"]))
        if key in seen_pairs:
            raise ValueError(f"duplicate fit sample/speaker row: {key}")
        seen_pairs.add(key)
        audio, binding = _read_fit_audio(row)
        fit_payloads.append((row, audio, binding))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    factory = _resolve_factory(adapter_factory)
    fit_plan_view = {key: value for key, value in plan.items() if key != "rows"}
    fit_plan_view["rows"] = fit_rows
    runtime = factory(plan=fit_plan_view, fit_rows=fit_rows, device=device, seed=seed)
    required = {"enhancement_stack", "optimizer", "adapter", "context_for_row"}
    if not isinstance(runtime, dict) or not required.issubset(runtime):
        raise ValueError(f"adapter factory must return {sorted(required)}")
    stack = runtime["enhancement_stack"]
    optimizer = runtime["optimizer"]
    adapter = runtime["adapter"]
    sync_score = runtime.get("sync_score", getattr(adapter, "sync_score", None))
    context_for_row = runtime["context_for_row"]
    if not callable(context_for_row) or not callable(getattr(adapter, "render_video", None)) or not callable(
        sync_score
    ):
        raise ValueError("runtime adapter/context callables do not match the frozen fit contract")

    architecture = _head_architecture(stack.head)
    transform_config = {
        "method": "spectral_head",
        "sample_rate_hz": 16_000,
        "use_global_rms": bool(use_global_rms),
        "target_rms": float(target_rms),
        "rms_stage_peak_ceiling": rms_stage_peak_ceiling,
        "use_local_envelope": bool(use_local_envelope),
        "local_window_ms": float(local_window_ms),
        "local_ratio": float(local_ratio),
        "local_min_gain_db": float(local_min_gain_db),
        "local_max_gain_db": float(local_max_gain_db),
        "use_spectral_head": True,
        "post_rms_match": post_rms_match,
        "peak_ceiling": peak_ceiling,
        "head_architecture": architecture,
    }
    score_config = runtime.get("score_config", {})
    if not isinstance(score_config, dict):
        raise ValueError("runtime score_config must be a JSON-compatible mapping")
    optimization_config = {
        **score_config,
        "score_proxy": score_config.get("score_proxy", "syncnet_canchor_median_minus_fixed_anchor_v1"),
        "anchor_lag_frames": int(anchor_lag_frames) if score_config.get("anchor_lag_frames_used", True) else None,
        "identity_weight": float(identity_weight),
        "spectral_smoothness_weight": float(spectral_smoothness_weight),
        "optimizer_class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
    }
    source_hashes = _package_source_hashes()
    runtime_source_pins = runtime.get("source_pins", {})
    if not isinstance(runtime_source_pins, dict):
        raise ValueError("adapter factory source_pins must be a JSON-compatible mapping")
    checkpoint_base = {
        "schema": "cross_tfg_audio_head_checkpoint_v1",
        "architecture": architecture,
        "transform_config": transform_config,
        "optimization_config": optimization_config,
        "source_hashes": source_hashes,
        "runtime_source_pins": runtime_source_pins,
        "plan_path": str(plan_path.resolve()),
        "plan_raw_sha256": plan_sha,
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "fit_model_id": fit_model_id or (next(iter(model_ids)) if model_ids else None),
        "seed": int(seed),
    }

    order_rng = random.Random(seed)
    epoch_rows: list[dict[str, Any]] = []
    step = 0
    steps_path = destination / "fit_steps.jsonl"
    for epoch in range(epochs):
        order = list(range(len(fit_payloads)))
        order_rng.shuffle(order)
        for row_index in order:
            row, audio, audio_binding = fit_payloads[row_index]
            context = context_for_row(row)
            audio_tensor = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
            if getattr(context, "expected_sample_count", audio.size) not in (None, int(audio.size)):
                raise ValueError(f"video context sample count differs from fit audio for {row['row_id']}")
            info: TrainStepInfo = sync_guided_train_step(
                enhancement_stack=stack,
                optimizer=optimizer,
                natural_audio=audio_tensor,
                video_context=context,
                render_video=adapter.render_video,
                sync_score=sync_score,
                use_global_rms=use_global_rms,
                use_local_envelope=use_local_envelope,
                local_window_ms=local_window_ms,
                local_ratio=local_ratio,
                local_min_gain_db=local_min_gain_db,
                local_max_gain_db=local_max_gain_db,
                use_spectral_head=True,
                post_rms_match=post_rms_match,
                target_rms=target_rms,
                anchor_lag_frames=anchor_lag_frames,
                peak_ceiling=peak_ceiling,
                rms_stage_peak_ceiling=rms_stage_peak_ceiling,
                identity_weight=identity_weight,
                spectral_smoothness_weight=spectral_smoothness_weight,
            )
            step += 1
            step_row = {
                "step": step,
                "epoch": epoch + 1,
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "natural_pcm": audio_binding,
                "proxy_median_minus_normalized_softmin": info.sync_score_mean,
                "loss": info.loss,
                "identity_penalty": info.identity_penalty,
                "head_gradient_l2": info.gradient_l2,
                "enhanced_audio_gradient_l2": info.audio_gradient_l2,
                "parameter_update_l2": info.parameter_update_l2,
                "batch_size": info.batch_size,
            }
            epoch_rows.append(step_row)
            _append_jsonl(steps_path, step_row)
            status_record.update({"completed_steps": step, "last_row_id": row["row_id"]})
            _atomic_json(status_path, status_record)
            if step % checkpoint_every_steps == 0:
                checkpoint_payload = {
                    **checkpoint_base,
                    "completed_steps": step,
                    "state_dict": stack.head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                _save_checkpoint_atomic(torch, destination / "checkpoint_latest.pt", checkpoint_payload)
                status_record["latest_checkpoint_sha256"] = _sha256_file(destination / "checkpoint_latest.pt")
                _atomic_json(status_path, status_record)
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    if step == 0:
        raise RuntimeError("fit loop completed with no optimizer steps")

    checkpoint_path = destination / "shared_audio_head.pt"
    checkpoint_payload = {
        **checkpoint_base,
        "completed_steps": step,
        "state_dict": stack.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    _save_checkpoint_atomic(torch, checkpoint_path, checkpoint_payload)
    checkpoint_sha = _sha256_file(checkpoint_path)
    report = {
        "schema": "cross_tfg_audio_head_fit_run_v1",
        "status": "FIT_ONLY_COMPLETE",
        "split": "fit",
        "plan_path": str(plan_path.resolve()),
        "plan_raw_sha256": plan_sha,
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "adapter_factory": adapter_factory,
        "device": device,
        "fit_model_id": fit_model_id or (next(iter(model_ids)) if model_ids else None),
        "seed": int(seed),
        "optimizer_steps": step,
        "planned_fit_rows": len(fit_rows),
        "used_fit_rows": len({row["row_id"] for row in epoch_rows}),
        "epochs_requested": int(epochs),
        "max_steps": max_steps,
        "use_global_rms": bool(use_global_rms),
        "use_local_envelope": bool(use_local_envelope),
        "local_window_ms": float(local_window_ms),
        "local_ratio": float(local_ratio),
        "local_min_gain_db": float(local_min_gain_db),
        "local_max_gain_db": float(local_max_gain_db),
        "post_rms_match": post_rms_match,
        "target_rms": float(target_rms),
        "peak_ceiling": peak_ceiling,
        "rms_stage_peak_ceiling": rms_stage_peak_ceiling,
        "anchor_lag_frames": optimization_config["anchor_lag_frames"],
        "identity_weight": float(identity_weight),
        "spectral_smoothness_weight": float(spectral_smoothness_weight),
        "transform_config": transform_config,
        "optimization_config": optimization_config,
        "architecture": architecture,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "effects_selected_on_development": False,
        "source_hashes": source_hashes,
        "runtime_source_pins": runtime_source_pins,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "steps_jsonl_path": str(steps_path),
        "steps_jsonl_sha256": _sha256_file(steps_path),
        "rows": epoch_rows,
    }
    report_path = destination / "fit_report.json"
    _atomic_json(report_path, report)
    report_sha = _sha256_file(report_path)
    status_record.update({
        "status": "FIT_ONLY_COMPLETE",
        "completed_steps": step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "report_path": str(report_path),
        "report_raw_sha256": report_sha,
        "steps_jsonl_path": str(steps_path),
        "steps_jsonl_sha256": _sha256_file(steps_path),
    })
    _atomic_json(status_path, status_record)
    report["report_raw_sha256"] = report_sha
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True, help="module.path:callable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", dest="fit_model_id")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--global-rms", action="store_true")
    parser.add_argument("--local-envelope", action="store_true")
    parser.add_argument("--local-window-ms", type=float, default=32.0)
    parser.add_argument("--local-ratio", type=float, default=0.8)
    parser.add_argument("--local-min-gain-db", type=float, default=-6.0)
    parser.add_argument("--local-max-gain-db", type=float, default=6.0)
    parser.add_argument("--post-rms-match", choices=("none", "input", "target"), default="none")
    parser.add_argument("--target-rms", type=float, default=0.0376838172675746)
    parser.add_argument("--anchor-lag-frames", type=int, default=0)
    parser.add_argument("--peak-ceiling", type=float)
    parser.add_argument("--rms-stage-peak-ceiling", type=float)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    parser.add_argument("--identity-weight", type=float, default=0.02)
    parser.add_argument("--spectral-smoothness-weight", type=float, default=0.002)
    args = parser.parse_args(argv)
    try:
        result = run_fit(
            plan_path=args.plan,
            adapter_factory=args.adapter_factory,
            output_dir=args.output_dir,
            device=args.device,
            fit_model_id=args.fit_model_id,
            seed=args.seed,
            epochs=args.epochs,
            max_steps=args.max_steps,
            use_global_rms=args.global_rms,
            use_local_envelope=args.local_envelope,
            local_window_ms=args.local_window_ms,
            local_ratio=args.local_ratio,
            local_min_gain_db=args.local_min_gain_db,
            local_max_gain_db=args.local_max_gain_db,
            post_rms_match=args.post_rms_match,
            target_rms=args.target_rms,
            anchor_lag_frames=args.anchor_lag_frames,
            peak_ceiling=args.peak_ceiling,
            rms_stage_peak_ceiling=args.rms_stage_peak_ceiling,
            checkpoint_every_steps=args.checkpoint_every_steps,
            identity_weight=args.identity_weight,
            spectral_smoothness_weight=args.spectral_smoothness_weight,
        )
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        status_path = output_dir / "fit_status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update({
                    "status": "FAILED",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                _atomic_json(status_path, status)
            except Exception:
                pass
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "report_path": str(args.output_dir / "fit_report.json"),
                      "checkpoint_path": result["checkpoint_path"],
                      "checkpoint_sha256": result["checkpoint_sha256"],
                      "optimizer_steps": result["optimizer_steps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
