"""CLI for the staged phoneme-separability/TTS-TFG association experiment."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from . import PROTOCOL_ID
from .analysis import benjamini_hochberg, build_block_differences, fit_primary, power_plan, sensitivity_analysis, summarize_gains
from .check import validate_run
from .features import compute_sample_features
from .generation import prepare_references, render_ditto, render_wav2lip
from .protocol import (
    CONDITIONS,
    PAIR_TYPES,
    ProtocolError,
    ResourceBusy,
    assert_gpu_clear,
    audit_inputs,
    freeze_protocol,
    read_json,
    read_jsonl,
    select_blocks,
    write_json,
    write_jsonl,
)
from .scoring import distance_from_embeddings, endpoint_rows, score_native, score_static_control, static_visual_embeddings, summarize_pair_matrices

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGES = ("audit", "features", "plan", "generate", "score", "analyze", "validate", "report")


def _load_cfg(repo_root: Path, config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("phoneme_tfg_association", payload)
    if not isinstance(section, dict):
        raise ProtocolError("phoneme_tfg_association config must be a mapping")
    cfg = dict(section)
    for key in ("cohort", "transfer_run", "visual_manifest", "face_detector_model", "wav2lip_checkpoint", "syncnet_model"):
        if key in cfg:
            path = Path(str(cfg[key]))
            cfg[key] = str(path if path.is_absolute() else (repo_root / path).resolve())
    return cfg


def _run_dir(repo_root: Path, run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise ProtocolError("run-id contains unsupported characters")
    return repo_root / "runs" / f"phoneme_tfg_association_{run_id}"


def _ensure_audit(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "00_protocol/input_audit.json"
    return read_json(path) if path.is_file() else audit_inputs(repo_root, run_dir, cfg)


def _ensure_features(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "01_features/features.jsonl"
    if path.is_file():
        return read_jsonl(path)
    audit = _ensure_audit(repo_root, run_dir, cfg)
    transfer_run = Path(str(cfg["transfer_run"]))
    result = compute_sample_features(transfer_run, audit, run_dir / "01_features")
    return list(result["rows"])


def _stage_audit(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    return audit_inputs(repo_root, run_dir, cfg)


def _stage_features(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    audit = _ensure_audit(repo_root, run_dir, cfg)
    result = compute_sample_features(Path(str(cfg["transfer_run"])), audit, run_dir / "01_features")
    return result["meta"]


def _stage_plan(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    audit = _ensure_audit(repo_root, run_dir, cfg)
    feature_rows = _ensure_features(repo_root, run_dir, cfg)
    reference_manifest = run_dir / "02_reference/reference_manifest.json"
    if reference_manifest.is_file():
        references = read_json(reference_manifest).get("rows", [])
    else:
        references = prepare_references(
            repo_root,
            run_dir,
            audit.get("rows", []),
            detector=str(cfg.get("detector", "haar")),
            detector_model=str(cfg.get("face_detector_model")) if cfg.get("face_detector_model") else None,
        )
    blocks = select_blocks(audit.get("rows", []), feature_rows, references, cfg, smoke=smoke)
    protocol = freeze_protocol(run_dir, blocks, cfg, smoke=smoke)
    power = power_plan(blocks, feature_rows, seed=int(cfg.get("seed", 20260921)), simulations=int(cfg.get("power_simulations", 2000)))
    write_json(run_dir / "00_protocol/power_plan.json", power)
    eligible = [row for row in feature_rows if row.get("eligible")]
    write_jsonl(run_dir / "00_protocol/eligibility.jsonl", eligible)
    return protocol


def _resolve_device(cfg: Mapping[str, Any], override: str | None) -> str:
    value = str(override or cfg.get("device", "cuda"))
    if value == "auto":
        try:
            import torch

            value = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            value = "cpu"
    if value not in {"cpu", "cuda"}:
        raise ProtocolError(f"unsupported device: {value}")
    return value


def _guard_gpu(device: str) -> None:
    if device == "cuda":
        # This is deliberately before any model import/load.  The caller sees
        # the competing PID instead of an opaque CUDA OOM.
        assert_gpu_clear()


def _stage_generate(run_dir: Path, cfg: Mapping[str, Any], *, smoke: bool = False, device_override: str | None = None) -> dict[str, Any]:
    protocol = read_json(run_dir / "00_protocol/protocol.json")
    device = _resolve_device(cfg, device_override)
    _guard_gpu(device)
    if str(protocol.get("config", {}).get("primary_tfg", "wav2lip")) != "wav2lip":
        raise ProtocolError("v1 generation requires wav2lip as primary_tfg")
    worker = None
    model = None
    if device == "cuda" or device == "cpu":
        from . import generation as generation_module

        worker = generation_module._wav2lip_worker()
        model = worker.load_model(Path(str(cfg["wav2lip_checkpoint"])).resolve(), device)
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_cells = [*protocol.get("cells", []), *protocol.get("ditto_cells", [])]
    if smoke:
        # Smoke operates on the already locked full protocol.  The first two
        # blocks are the frozen first 6 primary cells; no new protocol/hash is
        # created and successful receipts are reused by the full run.
        all_cells = list(protocol.get("cells", []))[:6]
    for cell in all_cells:
        succeeded = False
        last_failure: dict[str, Any] | None = None
        for attempt in range(1, int(cfg.get("max_render_attempts_per_cell", 2)) + 1):
            attempt_cell = {**cell, "attempt": attempt}
            started = time.monotonic()
            try:
                if str(cell.get("tfg")) == "ditto":
                    receipts.append(render_ditto(attempt_cell, run_dir, cfg))
                else:
                    receipts.append(render_wav2lip(attempt_cell, run_dir, cfg, model=model, device=device))
                succeeded = True
                break
            except Exception as exc:
                last_failure = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_hash": cell.get("protocol_hash"), "cell_key": cell["cell_key"], "sample_id": cell["sample_id"], "source_group": cell["source_group"], "arm": cell["arm"], "status": "failed", "reason": f"{type(exc).__name__}:{exc}", "attempt": attempt, "elapsed_seconds": float(time.monotonic() - started)}
        if not succeeded and last_failure is not None:
            write_json(run_dir / "03_video" / str(cell["tfg"]) / str(cell["sample_id"]) / f"{cell['arm']}.receipt.json", last_failure)
            failures.append(last_failure)
    write_json(run_dir / "03_video/generation_summary.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_hash": protocol.get("freeze_hash"), "device": device, "smoke": bool(smoke), "expected": len(all_cells), "completed": len(receipts), "failed": len(failures), "receipts": receipts, "failures": failures})
    return {"completed": len(receipts), "failed": len(failures), "device": device, "smoke": bool(smoke)}


def _stage_score(run_dir: Path, cfg: Mapping[str, Any], *, smoke: bool = False, device_override: str | None = None) -> dict[str, Any]:
    protocol = read_json(run_dir / "00_protocol/protocol.json")
    device = _resolve_device(cfg, device_override)
    _guard_gpu(device)
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

    scorer = SyncNetScorer(Path(str(cfg["syncnet_model"])).resolve(), device=device, batch_size=int(cfg.get("syncnet_batch_size", 20)), threads=int(cfg.get("syncnet_threads", 4)))
    receipts = {(str(row.get("sample_id")), str(row.get("arm"))): row for row in [read_json(path) for path in (run_dir / "03_video/wav2lip").glob("*/*.receipt.json")]} if (run_dir / "03_video/wav2lip").is_dir() else {}
    score_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    static_failures: list[dict[str, Any]] = []
    blocks = list(protocol.get("blocks", []))[:2] if smoke else list(protocol.get("blocks", []))
    for block in blocks:
        sid = str(block["sample_id"])
        arms = ("natural", *[str(item) for item in block["tts_arms"]])
        matrices: dict[str, np.ndarray] = {}
        endpoint_inputs: dict[str, Path] = {}
        try:
            for arm in arms:
                cell = next(cell for cell in protocol["cells"] if str(cell["sample_id"]) == sid and str(cell["arm"]) == arm)
                receipt = receipts.get((sid, arm))
                if receipt is None or receipt.get("status") != "complete":
                    raise ProtocolError(f"missing generation receipt: {sid}/{arm}")
                media = Path(str(receipt["output"])).resolve()
                audio = Path(str(cell["feature"]["arms"][arm]["audio"])).resolve()
                score_native(cell, media, audio, run_dir, cfg, scorer=scorer, device=device)
                matrix_path = run_dir / "04_syncnet" / str(cell["tfg"]) / sid / arm / "distance.npy"
                matrices[arm] = np.load(matrix_path, allow_pickle=False)
                endpoint_inputs[arm] = matrix_path
            endpoints = summarize_pair_matrices(matrices["natural"], {arm: matrices[arm] for arm in arms[1:]}, vshift=int(cfg.get("vshift", 15)), minimum_interior_windows=int(cfg.get("minimum_interior_windows", 25)))
            score_rows.append({"schema_version": 1, "protocol_id": PROTOCOL_ID, "sample_id": sid, "source_group": block["source_group"], "tfg": "wav2lip", "endpoints": endpoint_rows(sid, str(block["source_group"]), endpoints, tfg="wav2lip"), "matrix_paths": {arm: str(path.resolve()) for arm, path in endpoint_inputs.items()}, "status": "complete"})
            try:
                static_visual: dict[str, np.ndarray] = {}
                native_audio: dict[str, np.ndarray] = {}
                for arm in arms:
                    audio_embedding_path = run_dir / "04_syncnet" / "wav2lip" / sid / arm / "audio.npy"
                    native_audio[arm] = np.load(audio_embedding_path, allow_pickle=False)
                    static_visual[arm], _discarded_audio = static_visual_embeddings(
                        scorer,
                        Path(str(block["reference"]["crop"])).resolve(),
                        Path(str(block["feature"]["arms"][arm]["audio"])).resolve(),
                        int(native_audio[arm].shape[0]),
                        batch_size=int(cfg.get("syncnet_batch_size", 20)),
                    )
                    static_matrix = distance_from_embeddings(static_visual[arm], native_audio[arm], vshift=int(cfg.get("vshift", 15)))
                    static_path = run_dir / "04_syncnet" / "wav2lip" / sid / arm / "static_distance.npy"
                    np.save(static_path, static_matrix, allow_pickle=False)
                static_endpoints = score_static_control(static_visual, native_audio, vshift=int(cfg.get("vshift", 15)), minimum_interior_windows=int(cfg.get("minimum_interior_windows", 25)))
                static_rows.append({"schema_version": 1, "protocol_id": PROTOCOL_ID, "sample_id": sid, "source_group": block["source_group"], "tfg": "wav2lip_static", "endpoints": endpoint_rows(sid, str(block["source_group"]), static_endpoints, tfg="wav2lip_static"), "status": "complete"})
            except Exception as exc:
                static_failures.append({"sample_id": sid, "source_group": block["source_group"], "status": "failed", "reason": f"{type(exc).__name__}:{exc}"})
        except Exception as exc:
            failures.append({"sample_id": sid, "source_group": block["source_group"], "status": "failed", "reason": f"{type(exc).__name__}:{exc}"})
    write_jsonl(run_dir / "04_syncnet/scores.jsonl", score_rows)
    write_jsonl(run_dir / "04_syncnet/static_scores.jsonl", static_rows)
    write_json(run_dir / "04_syncnet/score_summary.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_hash": protocol.get("freeze_hash"), "device": device, "smoke": bool(smoke), "expected_blocks": len(blocks), "completed_blocks": len(score_rows), "failed_blocks": len(failures), "failures": failures, "static_completed_blocks": len(static_rows), "static_failed_blocks": len(static_failures), "static_failures": static_failures, "static_status": "COMPLETE" if len(static_rows) == len(score_rows) and not static_failures else "PARTIAL"})
    return {"completed_blocks": len(score_rows), "failed_blocks": len(failures), "static_completed_blocks": len(static_rows), "static_failed_blocks": len(static_failures), "device": device, "smoke": bool(smoke)}


def _stage_analyze(run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    protocol = read_json(run_dir / "00_protocol/protocol.json")
    feature_rows = read_jsonl(run_dir / "01_features/features.jsonl")
    score_rows = read_jsonl(run_dir / "04_syncnet/scores.jsonl")
    static_score_rows = read_jsonl(run_dir / "04_syncnet/static_scores.jsonl")
    blocks = protocol.get("blocks", [])
    differences = build_block_differences(blocks, feature_rows, score_rows, feature_name="primary_silhouette", support="EQUAL_COUNT", tfg="wav2lip")
    association = fit_primary(differences, feature_name="hubert_layer6_phoneme_silhouette", endpoint_name="wav2lip_interior_equal_count_sync_c", min_complete_blocks=int(cfg.get("primary_min_complete_blocks", 30)), min_pair_blocks=int(cfg.get("primary_min_pair_blocks", 4)), wild_draws=int(cfg.get("wild_bootstrap_draws", 9999)), seed=int(cfg.get("seed", 20260921)))
    gains = summarize_gains(blocks, score_rows, tfg="wav2lip", support="EQUAL_COUNT", draws=int(cfg.get("bootstrap_draws", 10000)), seed=int(cfg.get("seed", 20260921)))
    secondary: dict[str, Any] = {}
    secondary_specs = {
        "xlsr_l10_silhouette_to_C": ("xlsr_phoneme_silhouette", "wav2lip_interior_equal_count_sync_c"),
        "hubert_l6_viseme_silhouette_to_C": ("viseme_silhouette", "wav2lip_interior_equal_count_sync_c"),
        "hubert_l6_log_fisher_to_C": ("log_fisher", "wav2lip_interior_equal_count_sync_c"),
    }
    for name, (feature_name, endpoint_name) in secondary_specs.items():
        rows = build_block_differences(blocks, feature_rows, score_rows, feature_name=feature_name, support="EQUAL_COUNT", tfg="wav2lip")
        secondary[name] = fit_primary(rows, feature_name=feature_name, endpoint_name=endpoint_name, min_complete_blocks=int(cfg.get("primary_min_complete_blocks", 30)), min_pair_blocks=int(cfg.get("primary_min_pair_blocks", 4)), wild_draws=int(cfg.get("wild_bootstrap_draws", 9999)), seed=int(cfg.get("seed", 20260921)) + len(secondary) * 101)

    def _metric_fit(name: str, rows: Sequence[Mapping[str, Any]], y_key: str, endpoint_name: str) -> None:
        transformed = [{**row, "y": row.get(y_key)} for row in rows if row.get(y_key) is not None]
        secondary[name] = fit_primary(transformed, feature_name="hubert_layer6_phoneme_silhouette", endpoint_name=endpoint_name, min_complete_blocks=int(cfg.get("primary_min_complete_blocks", 30)), min_pair_blocks=int(cfg.get("primary_min_pair_blocks", 4)), wild_draws=int(cfg.get("wild_bootstrap_draws", 9999)), seed=int(cfg.get("seed", 20260921)) + len(secondary) * 101)

    _metric_fit("primary_silhouette_to_gain_D", differences, "gain_d", "wav2lip_interior_equal_count_gain_D")
    _metric_fit("primary_silhouette_to_delta_B", differences, "delta_b", "wav2lip_interior_equal_count_delta_B")
    for support in ("FULL", "INTERIOR"):
        rows = build_block_differences(blocks, feature_rows, score_rows, feature_name="primary_silhouette", support=support, tfg="wav2lip")
        _metric_fit(f"primary_silhouette_to_{support}_C", rows, "y", f"wav2lip_{support.lower()}_sync_c")
    static_difference_rows = build_block_differences(blocks, feature_rows, static_score_rows, feature_name="primary_silhouette", support="EQUAL_COUNT", tfg="wav2lip_static")
    static_by_group = {str(row["source_group"]): row for row in static_difference_rows}
    generated_static = [{**row, "y": float(row["y"]) - float(static_by_group[row["source_group"]]["y"])} for row in differences if row["source_group"] in static_by_group]
    _metric_fit("primary_silhouette_to_static_C", static_difference_rows, "y", "wav2lip_static_interior_equal_count_sync_c")
    _metric_fit("primary_silhouette_to_generated_minus_static_C", generated_static, "y", "wav2lip_minus_static_interior_equal_count_sync_c")
    q_values = benjamini_hochberg({name: value.get("p_wild") for name, value in secondary.items()})
    for name, value in secondary.items():
        value["q_fdr"] = q_values[name]
    sensitivity = sensitivity_analysis(
        differences,
        duration_ratio_min=float(cfg.get("duration_ratio_min", 0.5)),
        duration_ratio_max=float(cfg.get("duration_ratio_max", 1.5)),
        min_complete_blocks=int(cfg.get("primary_min_complete_blocks", 30)),
        min_pair_blocks=int(cfg.get("primary_min_pair_blocks", 4)),
    )
    write_json(run_dir / "05_analysis/association.json", association)
    write_json(run_dir / "05_analysis/gains.json", gains)
    write_json(run_dir / "05_analysis/secondary.json", {"family": "secondary_associations_v1", "count": len(secondary), "results": secondary})
    write_json(run_dir / "05_analysis/sensitivity.json", sensitivity)
    write_json(run_dir / "05_analysis/missingness.json", {"expected_blocks": len(blocks), "complete_blocks": len(differences), "missing_blocks": len(blocks) - len(differences), "score_rows": len(score_rows)})
    with (run_dir / "05_analysis/block_differences.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in differences for key in row}) if differences else ["block_id", "x", "y"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(differences)
    return association


def _write_review_pack(run_dir: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic, anonymized 12-source review package."""
    review_dir = run_dir / "06_review"
    media_dir = review_dir / "media"
    review_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    blocks = sorted(protocol.get("blocks", []), key=lambda row: str(row.get("block_id")))[:12]
    receipt_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in protocol.get("cells", []):
        receipt_path = run_dir / "03_video" / str(cell.get("tfg")) / str(cell.get("sample_id")) / f"{cell.get('arm')}.receipt.json"
        if receipt_path.is_file():
            try:
                receipt = read_json(receipt_path)
            except (OSError, ValueError):
                continue
            if receipt.get("status") == "complete":
                receipt_index[(str(cell.get("sample_id")), str(cell.get("tfg")), str(cell.get("arm")))] = receipt
    rows: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64(int(protocol.get("config", {}).get("seed", 20260921)) + 303))
    for block_index, block in enumerate(blocks, start=1):
        sid = str(block.get("sample_id"))
        tfg = str(protocol.get("config", {}).get("primary_tfg", "wav2lip"))
        arms = ("natural", *[str(item) for item in block.get("tts_arms", [])])
        receipts = [receipt_index.get((sid, tfg, arm)) for arm in arms]
        if len(receipts) != 3 or any(receipt is None for receipt in receipts):
            continue
        permutation = rng.permutation(len(arms)).tolist()
        blind_group = f"review_group_{block_index:02d}"
        for clip_index, arm_index in enumerate(permutation, start=1):
            receipt = receipts[arm_index]
            assert receipt is not None
            source = Path(str(receipt["output"])).resolve()
            if not source.is_file():
                continue
            blind_id = f"{blind_group}_clip_{clip_index}"
            link = media_dir / f"{blind_id}{source.suffix or '.mkv'}"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(source)
            rows.append({"blind_group": blind_group, "blind_id": blind_id, "media": str(link.relative_to(review_dir)), "status": "ready"})
            ratings.append({"blind_group": blind_group, "blind_id": blind_id, "sync_rating_1_5": "", "lip_readability_1_5": "", "notes": ""})
    status = "READY" if rows else "NOT_RUN"
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "blind_groups": len({row["blind_group"] for row in rows}),
        "clip_count": len(rows),
        "target_groups": 12,
        "rows": rows,
        "claim_boundary": "human ratings are supplementary and do not replace the preregistered SyncNet association",
    }
    write_json(review_dir / "blind_manifest.json", manifest)
    with (review_dir / "ratings_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_group", "blind_id", "sync_rating_1_5", "lip_readability_1_5", "notes"])
        writer.writeheader()
        writer.writerows(ratings)
    return {"status": status, "blind_groups": manifest["blind_groups"], "clip_count": len(rows)}


def _stage_report(run_dir: Path) -> dict[str, Any]:
    association = read_json(run_dir / "05_analysis/association.json")
    gains = read_json(run_dir / "05_analysis/gains.json")
    validation = read_json(run_dir / "validation.json") if (run_dir / "validation.json").is_file() else {}
    review = _write_review_pack(run_dir, read_json(run_dir / "00_protocol/protocol.json"))

    def _fmt(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "NA"

    def _fmt_ci(value: Any) -> str:
        if not isinstance(value, Sequence) or len(value) != 2:
            return "NA"
        return f"[{_fmt(value[0])}, {_fmt(value[1])}]"

    lines = [
        "# 音素可分度与 TTS→TFG 增益关联实验",
        "",
        f"- protocol: `{PROTOCOL_ID}`",
        f"- engineering_status: `{validation.get('engineering_status', 'NOT_VALIDATED')}`",
        f"- scientific_status: `{association.get('scientific_status')}`",
        f"- estimand: `{association.get('estimand')}`",
        f"- blocks: `{association.get('n_blocks')}`; source_groups: `{association.get('n_source_groups')}`",
        "",
        "## 主检验",
        "",
        f"β = `{_fmt(association.get('beta'))}` Sync-C / 0.1 silhouette; HC3 SE = `{_fmt(association.get('se_hc3'))}`; 95% CI = `{_fmt_ci(association.get('ci95_hc3'))}`; wild p = `{association.get('p_wild')}`.",
        "",
        "主检验是同句、同来源块、pair 固定效应后的关联；音素可分度没有被随机干预，因此不作因果解释。",
        "",
        "## TTS−natural 增益（描述性）",
        "",
        "|condition|estimate|95% CI|n groups|",
        "|---|---:|---|---:|",
    ]
    for condition, row in gains.get("summaries", {}).items():
        if condition == "source_mean_across_assigned_tts":
            continue
        lines.append(f"|{condition}|{_fmt(row.get('estimate'))}|{_fmt_ci(row.get('ci95'))}|{row.get('n_groups')}|")
    lines.extend(["", "## 状态边界", "", "- `INCONCLUSIVE` 不等于证明没有关联。", "- 静态对照、Ditto 和人工感知评价若未执行，均保持 `NOT_MEASURED`/`NOT_RUN`。", f"- blind review package: `{review['status']}` ({review['blind_groups']} groups / {review['clip_count']} clips).", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str((run_dir / "report.md").resolve()), "scientific_status": association.get("scientific_status"), "engineering_status": validation.get("engineering_status")}


def run_stage(stage: str, *, repo_root: Path, run_dir: Path, cfg: Mapping[str, Any], smoke: bool = False, device_override: str | None = None) -> dict[str, Any]:
    if stage == "audit":
        return _stage_audit(repo_root, run_dir, cfg)
    if stage == "features":
        return _stage_features(repo_root, run_dir, cfg)
    if stage == "plan":
        return _stage_plan(repo_root, run_dir, cfg, smoke=smoke)
    if stage == "generate":
        return _stage_generate(run_dir, cfg, smoke=smoke, device_override=device_override)
    if stage == "score":
        return _stage_score(run_dir, cfg, smoke=smoke, device_override=device_override)
    if stage == "analyze":
        return _stage_analyze(run_dir, cfg)
    if stage == "validate":
        return validate_run(run_dir, require_media=False, require_scores=False)
    if stage == "report":
        return _stage_report(run_dir)
    raise ProtocolError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the phoneme-separability/TTS-TFG association protocol")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "scripts/configs/phoneme_tfg_association_v1.yaml")
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = _run_dir(REPO_ROOT, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(REPO_ROOT, args.config.resolve())
    stages = STAGES if args.stage == "all" else (args.stage,)
    try:
        for stage in stages:
            result = run_stage(stage, repo_root=REPO_ROOT, run_dir=run_dir, cfg=cfg, smoke=args.smoke, device_override=args.device)
            if isinstance(result, Mapping):
                summary_keys = ("status", "completed", "failed", "completed_blocks", "failed_blocks", "static_completed_blocks", "eligible_sample_count", "eligible_source_group_count", "source_group_count", "block_count", "scientific_status", "engineering_status")
                compact = {key: result[key] for key in summary_keys if key in result}
                if not compact:
                    compact = {"keys": sorted(result)[:12]}
            else:
                compact = result
            print(f"[{stage}] {compact}", flush=True)
    except ResourceBusy as exc:
        print(f"[blocked/resource-busy] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"[{args.stage}] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
