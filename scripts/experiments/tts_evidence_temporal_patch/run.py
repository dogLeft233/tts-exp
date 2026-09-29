from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .analysis import summarize_evidence, summarize_patch
from .check import validate_run
from .evaluation import (
    build_native_pairs,
    export_blind_pack,
    import_official,
    import_ratings,
    score_patch_official,
    score_existing_temporal,
)
from .intervention import build_frame_map, continuous_core_blocks, mel_frame_clock
from .protocol import (
    DEFAULT_BINDINGS,
    DEFAULT_PARENT_RUN,
    DEFAULT_SEED,
    DIRECTIONS,
    FRAME_RATE,
    MAX_GPU_MINUTES,
    MAX_NEW_RENDER_CELLS,
    MEL_RATE,
    MODELS,
    NATURAL,
    PROTOCOL_ID,
    REVISION,
    RunPaths,
    audit_parent,
    canonical_hash,
    environment_fingerprint,
    file_sha256,
    finalize_artifact,
    load_yaml,
    read_json,
    resolve_path,
    run_paths,
    select_patch_cohort,
    write_json,
    write_jsonl,
)


REPO = Path(__file__).resolve().parents[3]
STAGES = ("prepare", "evaluate-existing", "export-blind", "import-ratings", "patch-preflight", "patch-smoke", "patch-render", "patch-evaluate", "analyze", "check")


def _config_value(config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name in config:
        return config[name]
    paths = config.get("paths")
    if isinstance(paths, Mapping) and name in paths:
        return paths[name]
    return default


def _parent_root(config: Mapping[str, Any], override: Path | None) -> Path:
    return (override or resolve_path(str(_config_value(config, "parent_run", DEFAULT_PARENT_RUN)))).resolve()


def _bindings_path(config: Mapping[str, Any], override: Path | None) -> Path:
    return (override or resolve_path(str(_config_value(config, "input_bindings", DEFAULT_BINDINGS)))).resolve()


def _mel_info(audio_path: Path) -> tuple[list[Any], dict[str, Any]]:
    """Use the registered Wav2Lip mel implementation on CPU during preflight."""

    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import mel_chunks

    return mel_chunks(audio_path)


def _strict_pair_support(group_payload: Mapping[str, Any], model: str) -> bool:
    arms = group_payload.get("arms", {})
    natural = arms.get(NATURAL)
    tts = arms.get(model)
    if not isinstance(natural, Mapping) or not isinstance(tts, Mapping):
        return False
    try:
        from scripts.experiments.tts_time_instance import parse_textgrid
        from scripts.experiments.tts_visual_timing_metrics import audio_time_map

        natural_tokens = parse_textgrid(Path(str(natural["textgrid"])))
        tts_tokens = parse_textgrid(Path(str(tts["textgrid"])))
        mapping_n_to_t = audio_time_map(natural_tokens, tts_tokens)
        mapping_t_to_n = audio_time_map(tts_tokens, natural_tokens)
        if mapping_n_to_t.get("status") != "COMPLETE" or mapping_t_to_n.get("status") != "COMPLETE":
            return False
        n_chunks, n_info = _mel_info(Path(str(natural["audio"])))
        t_chunks, t_info = _mel_info(Path(str(tts["audio"])))
        n_clock = mel_frame_clock(int(n_info["mel_shape"][1]), len(n_chunks))
        t_clock = mel_frame_clock(int(t_info["mel_shape"][1]), len(t_chunks))
        n_to_t = build_frame_map(
            [item.center_s for item in n_clock],
            [item.center_s for item in t_clock],
            mapping_n_to_t["segments"],
            direction="n_to_t",
            support_blocks=mapping_n_to_t.get("support_blocks", []),
            recipient_tail=[item.tail_reused for item in n_clock],
            donor_tail=[item.tail_reused for item in t_clock],
        )
        t_to_n = build_frame_map(
            [item.center_s for item in t_clock],
            [item.center_s for item in n_clock],
            mapping_t_to_n["segments"],
            direction="t_to_n",
            support_blocks=mapping_t_to_n.get("support_blocks", []),
            recipient_tail=[item.tail_reused for item in t_clock],
            donor_tail=[item.tail_reused for item in n_clock],
        )
        n_blocks = continuous_core_blocks(n_to_t, minimum_frames=25)
        t_blocks = continuous_core_blocks(t_to_n, minimum_frames=25)
        # ``audio_time_map`` defines coverage over speech-token support, not
        # over the complete audio duration.  Silence and mel tail padding are
        # intentionally outside the strict alignment denominator.
        n_coverage = float(mapping_n_to_t.get("n_speech_coverage", 0.0))
        t_coverage = float(mapping_n_to_t.get("t_speech_coverage", 0.0))
        return bool(
            n_coverage >= 0.80
            and t_coverage >= 0.80
            and n_blocks
            and t_blocks
            and sum(len(block) for block in n_blocks) + sum(len(block) for block in t_blocks) >= 50
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False


def _code_bindings() -> dict[str, str | None]:
    names = ("protocol.py", "intervention.py", "evaluation.py", "worker.py", "analysis.py", "run.py", "check.py")
    return {name: file_sha256(Path(__file__).with_name(name)) if Path(__file__).with_name(name).is_file() else None for name in names}


def _cache_inventory(parent_root: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for receipt in sorted((parent_root / "04_syncnet/wav2lip").glob("*/**/receipt.json")):
        try:
            value = read_json(receipt)
        except (OSError, ValueError):
            continue
        rows.append({"path": str(receipt.resolve()), "sha256": file_sha256(receipt), "sample_id": value.get("sample_id"), "arm": value.get("arm"), "status": value.get("status"), "matrix": value.get("matrix"), "visual": value.get("visual"), "audio_embedding": value.get("audio_embedding")})
    result = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "count": len(rows), "rows": rows}
    return finalize_artifact(result)


def prepare(paths: RunPaths, config: Mapping[str, Any], *, parent_override: Path | None = None, bindings_override: Path | None = None) -> dict[str, Any]:
    parent_root = _parent_root(config, parent_override)
    bindings = _bindings_path(config, bindings_override)
    inventory = audit_parent(parent_root, bindings_path=bindings, ffmpeg=resolve_path(str(_config_value(config, "ffmpeg", "/home/wjj/miniconda3/bin/ffmpeg"))))
    selection = select_patch_cohort(inventory, seed=int(_config_value(config, "seed", DEFAULT_SEED)), support_checker=_strict_pair_support)
    selection = finalize_artifact(selection)
    cache_inventory = _cache_inventory(parent_root, inventory)
    code = _code_bindings()
    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "revision": REVISION,
        "status": "FROZEN",
        "parent_run": str(parent_root),
        "input_bindings": str(bindings),
        "parent_inventory_sha256": inventory["artifact_sha256"],
        "selection_sha256": selection["artifact_sha256"],
        "cache_inventory_sha256": cache_inventory["artifact_sha256"],
        "design": {
            "seed": int(_config_value(config, "seed", DEFAULT_SEED)),
            "bootstrap_draws": int(_config_value(config, "bootstrap_draws", 20_000)),
            "layer": "audio_encoder.output",
            "lambda": 0.5,
            "dynamic_smoothing_frames": 5,
            "science_conditions": ["BASE", "COHERENT", "SCRAMBLED", "ERASE"],
            "directions": list(DIRECTIONS),
            "max_new_render_cells": 128,
            "max_gpu_minutes": 120,
            "stage_a_new_tfg_videos": 0,
        },
        "code": code,
        "environment": environment_fingerprint(),
        "config_sha256": canonical_hash(config),
        "created_at_epoch": time.time(),
        "artifact_sha256": None,
    }
    protocol = finalize_artifact(protocol)
    paths.ensure_dirs()
    write_json(paths.protocol / "parent_inventory.json", inventory)
    write_json(paths.protocol / "selection.json", selection)
    write_json(paths.protocol / "cache_inventory.json", cache_inventory)
    write_json(paths.protocol / "protocol.json", protocol)
    return {"protocol": protocol, "inventory": inventory, "selection": selection, "cache_inventory": cache_inventory}


def _load_frozen(paths: RunPaths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(paths.protocol / "protocol.json")
    inventory = read_json(paths.protocol / "parent_inventory.json")
    selection = read_json(paths.protocol / "selection.json")
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != "FROZEN":
        raise RuntimeError("run does not have a frozen protocol; run prepare first")
    return protocol, inventory, selection


def evaluate_existing(paths: RunPaths, config: Mapping[str, Any], *, visual_override: bool | None = None) -> dict[str, Any]:
    _protocol, inventory, _selection = _load_frozen(paths)
    parent_root = Path(str(inventory["parent_root"]))
    official = import_official(parent_root, destination=paths.evidence / "official.json")
    temporal = score_existing_temporal(parent_root, inventory, destination=paths.evidence / "temporal_rank.json")
    run_visual = bool(_config_value(config, "run_visual_extract", False)) if visual_override is None else bool(visual_override)
    visual = {"status": "PENDING_VISUAL_EXTRACTION", "reason": "explicit_cpu_acceptance_skip" if visual_override is False else "run_visual_extract=false_or_not_requested", "rows": [], "requested": run_visual}
    if run_visual:
        from .evaluation import extract_visual, score_visual_controls, score_visual_timing

        visual_rows: list[dict[str, Any]] = []
        technical_groups = {str(row["source_group"]) for row in _selection.get("rows", []) if row.get("stage") == "technical"}
        for row in inventory["cells"]:
            video = Path(str(row["video"]))
            extracted = extract_visual(video, paths.visual / row["sample_id"] / row["arm"], config=config)
            visual_result = score_visual_timing(extracted, width=float(row["reference"].get("crop_shape", [224, 224, 3])[1]), height=float(row["reference"].get("crop_shape", [224, 224, 3])[0]))
            controls = score_visual_controls(extracted, width=float(row["reference"].get("crop_shape", [224, 224, 3])[1]), height=float(row["reference"].get("crop_shape", [224, 224, 3])[0])) if str(row["source_group"]) in technical_groups else None
            visual_rows.append({"source_group": row["source_group"], "sample_id": row["sample_id"], "arm": row["arm"], "video_sha256": row["video_sha256"], "result": visual_result, "controls": controls, "status": extracted.get("status")})
        visual = {"status": "complete", "rows": visual_rows}
    write_json(paths.visual / "visual_timing.json", finalize_artifact(visual))
    result = {"schema_version": 1, "status": "complete", "official": official, "temporal": temporal, "visual": visual, "stage_a_new_tfg_videos": 0, "visual_override": visual_override}
    write_json(paths.evidence / "existing_evidence.json", finalize_artifact(result))
    return result


def export_blind(paths: RunPaths, config: Mapping[str, Any]) -> dict[str, Any]:
    _protocol, inventory, _selection = _load_frozen(paths)
    pairs = build_native_pairs(inventory)
    if len(pairs) != 72:
        raise RuntimeError(f"expected 72 natural/TTS blind pairs, got {len(pairs)}")
    ffmpeg = resolve_path(str(_config_value(config, "ffmpeg", "/home/wjj/miniconda3/bin/ffmpeg")))
    return export_blind_pack(
        pairs,
        paths.blind,
        seed=int(_protocol["design"]["seed"]),
        raters_per_pair=3,
        ffmpeg=ffmpeg,
        require_qc_media=True,
    )


def import_human_ratings(paths: RunPaths, ratings_path: Path) -> dict[str, Any]:
    _load_frozen(paths)
    return import_ratings(ratings_path, paths.blind)


def patch_preflight(paths: RunPaths) -> dict[str, Any]:
    _protocol, inventory, selection = _load_frozen(paths)
    by_key = {str(row["cell_key"]): row for row in inventory["cells"]}
    plans: list[dict[str, Any]] = []
    from scripts.experiments.tts_time_instance import parse_textgrid
    from scripts.experiments.tts_visual_timing_metrics import audio_time_map

    for selected in selection["rows"]:
        natural = by_key[str(selected["natural_cell_key"])]
        tts = by_key[str(selected["tts_cell_key"])]
        natural_tokens = parse_textgrid(Path(natural["textgrid"]))
        tts_tokens = parse_textgrid(Path(tts["textgrid"]))
        mappings = {
            "natural_from_tts": audio_time_map(natural_tokens, tts_tokens),
            "tts_from_natural": audio_time_map(tts_tokens, natural_tokens),
        }
        for direction in DIRECTIONS:
            mapping = mappings[direction]
            recipient = natural if direction == "natural_from_tts" else tts
            donor = tts if direction == "natural_from_tts" else natural
            r_chunks, r_info = _mel_info(Path(recipient["audio"]))
            d_chunks, d_info = _mel_info(Path(donor["audio"]))
            r_clock = mel_frame_clock(int(r_info["mel_shape"][1]), len(r_chunks))
            d_clock = mel_frame_clock(int(d_info["mel_shape"][1]), len(d_chunks))
            frame_map = build_frame_map(
                [item.center_s for item in r_clock],
                [item.center_s for item in d_clock],
                mapping.get("segments", []),
                direction="n_to_t" if direction == "natural_from_tts" else "t_to_n",
                support_blocks=mapping.get("support_blocks", []),
                recipient_tail=[item.tail_reused for item in r_clock],
                donor_tail=[item.tail_reused for item in d_clock],
            )
            support_blocks = continuous_core_blocks(frame_map, minimum_frames=25)
            plans.append(
                {
                    "source_group": selected["source_group"],
                    "sample_id": selected["sample_id"],
                    "tts_arm": selected["tts_arm"],
                    "direction": direction,
                    "recipient_cell_key": recipient["cell_key"],
                    "donor_cell_key": donor["cell_key"],
                    "recipient_frame_count": len(r_chunks),
                    "donor_frame_count": len(d_chunks),
                    "recipient_mel_shape": r_info["mel_shape"],
                    "donor_mel_shape": d_info["mel_shape"],
                    "mapping": mapping,
                    "frame_map": frame_map,
                    "continuous_block_lengths": [len(block) for block in support_blocks],
                    "support_status": "PASS" if support_blocks else "INSUFFICIENT_SUPPORT",
                }
            )
    result = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "rows": plans, "count": len(plans), "selection_sha256": selection["artifact_sha256"]})
    write_json(paths.patch_plan / "plan.json", result)
    return result


def _render_stage(paths: RunPaths, config: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    _protocol, inventory, selection = _load_frozen(paths)
    plan = read_json(paths.patch_plan / "plan.json")
    allowed_groups = {str(row["source_group"]) for row in selection["rows"] if row["stage"] == ("technical" if stage == "patch-smoke" else "science")}
    by_key = {str(row["cell_key"]): row for row in inventory["cells"]}
    checkpoint = resolve_path(str(_config_value(config, "wav2lip_checkpoint", "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth")))
    ffmpeg = resolve_path(str(_config_value(config, "ffmpeg", "/home/wjj/miniconda3/bin/ffmpeg")))
    if not checkpoint.is_file() or not ffmpeg.is_file():
        raise RuntimeError("patch renderer inputs are missing")
    from .protocol import gpu_lease
    from .worker import render_patch_cell

    rows: list[dict[str, Any]] = []
    target_uuid = _config_value(config, "gpu_uuid")
    conditions = ("BASE", "IDENTITY", "SHIFT_PLUS3", "SHIFT_MINUS3") if stage == "patch-smoke" else ("BASE", "COHERENT", "SCRAMBLED", "ERASE")
    existing_receipts = [read_json(path) for path in sorted(paths.patch_video.rglob("receipt.json"))]
    existing_render_count = sum(len(receipt.get("outputs", {})) for receipt in existing_receipts)
    existing_gpu_minutes = sum(float(receipt.get("elapsed_seconds", 0.0)) for receipt in existing_receipts) / 60.0
    new_render_count = 0
    if existing_render_count > MAX_NEW_RENDER_CELLS or existing_gpu_minutes > MAX_GPU_MINUTES:
        raise RuntimeError("frozen render budget is already exhausted before this stage")
    with gpu_lease(target_uuid=target_uuid, receipt_path=paths.patch_video / f"{stage}.gpu_lease.json"):
        for item in plan["rows"]:
            if str(item["source_group"]) not in allowed_groups:
                continue
            recipient = by_key[str(item["recipient_cell_key"])]
            donor = by_key[str(item["donor_cell_key"])]
            output_dir = paths.patch_video / item["source_group"] / item["tts_arm"] / item["direction"]
            receipt_path = output_dir / "receipt.json"
            if receipt_path.is_file():
                rows.append(read_json(receipt_path))
                continue
            if existing_render_count + new_render_count + len(conditions) > MAX_NEW_RENDER_CELLS:
                raise RuntimeError("new render cell budget would be exceeded")
            rows.append(render_patch_cell(recipient, donor, direction=item["direction"], output_dir=output_dir, checkpoint=checkpoint, ffmpeg=ffmpeg, device="cuda", mapping=item["mapping"], lam=0.5, smoothing_frames=5, batch_size=4, condition_names=conditions))
            new_render_count += len(conditions)
            from .protocol import assert_gpu_available

            assert_gpu_available(target_uuid=target_uuid)
    result = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "stage": stage, "receipt_count": len(rows), "rows": rows})
    write_json(paths.patch_video / f"{stage}.json", result)
    return result


def patch_evaluate(paths: RunPaths, config: Mapping[str, Any]) -> dict[str, Any]:
    """Leave scoring injectable; the default is explicit pending, never silent."""

    protocol, _inventory, _selection = _load_frozen(paths)
    receipts = [read_json(path) for path in sorted(paths.patch_video.rglob("receipt.json"))]
    if bool(_config_value(config, "run_patch_official_syncnet", False)):
        syncnet_root = resolve_path(str(_config_value(config, "syncnet_root", "third_party/syncnet_python")))
        syncnet_python = resolve_path(str(_config_value(config, "syncnet_python", "/home/wjj/.venvs/syncnet/bin/python")))
        syncnet_model = resolve_path(str(_config_value(config, "syncnet_model", "third_party/syncnet_python/data/syncnet_v2.model")))
        scored = score_patch_official(
            receipts,
            paths.patch_eval,
            syncnet_root=syncnet_root,
            syncnet_python=syncnet_python,
            syncnet_model=syncnet_model,
            min_track=int(_config_value(config, "min_track", 25)),
        )
        result = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": scored["status"], "metric_family": "official_patch_fulltrack", "official_endpoint": "confidence", "rows": scored["rows"], "official": scored.get("official"), "receipt_count": scored["receipt_count"]})
        write_json(paths.patch_eval / "scores.json", result)
        return result
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        for condition, output in receipt.get("outputs", {}).items():
            rows.append({"protocol_id": PROTOCOL_ID, "revision": protocol.get("revision"), "source_group": receipt["source_group"], "sample_id": receipt["sample_id"], "tts_arm": receipt["donor_arm"] if receipt["recipient_arm"] == NATURAL else receipt["recipient_arm"], "recipient_arm": receipt["recipient_arm"], "direction": receipt["direction"], "condition": condition, "endpoint": "custom_temporal_rank", "media": output["video"], "video_sha256": output["video_sha256"], "audio_pcm_sha256": receipt.get("recipient_audio_pcm_sha256"), "model_sha256": receipt.get("checkpoint_sha256"), "support_hash": receipt.get("support_hash"), "calibration_hash": None, "metric_family": "pending_custom_or_official", "metric": None, "status": "PENDING_SCORER", "reason": "patch-evaluate_requires_explicit_scorer"})
    result = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PENDING_SCORER" if rows else "EMPTY", "official_endpoint": "PENDING_OFFICIAL_SCORER", "rows": rows})
    write_json(paths.patch_eval / "scores.json", result)
    return result


def analyze(paths: RunPaths) -> dict[str, Any]:
    protocol, _inventory, _selection = _load_frozen(paths)
    evidence = read_json(paths.evidence / "existing_evidence.json") if (paths.evidence / "existing_evidence.json").is_file() else {"official": {"rows": []}, "temporal": {"rows": []}}
    ratings = None
    if (paths.blind / "ratings.json").is_file():
        ratings_value = read_json(paths.blind / "ratings.json")
        ratings = ratings_value.get("pair_rows", [])
    existing = summarize_evidence(evidence.get("official", {}).get("rows", []), evidence.get("temporal", {}).get("rows", []), ratings, draws=int(protocol["design"]["bootstrap_draws"]), seed=int(protocol["design"]["seed"]))
    patch_rows: list[dict[str, Any]] = []
    if (paths.patch_eval / "scores.json").is_file():
        patch_rows = read_json(paths.patch_eval / "scores.json").get("rows", [])
    patch = summarize_patch(patch_rows, None, draws=int(protocol["design"]["bootstrap_draws"]), seed=int(protocol["design"]["seed"]))
    result = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "engineering_status": "complete", "scientific_status": existing.get("conclusion", "PENDING_HUMAN"), "existing_evidence": existing, "patch": patch, "mediation_percentage": None})
    write_json(paths.analysis / "analysis.json", result)
    report = [
        "# TTS evidence and temporal representation patch",
        "",
        f"- protocol: `{PROTOCOL_ID}`",
        f"- official endpoint: full-track Confidence imported from the parent run; no TFG rerender in A",
        f"- human evidence: `{existing['human']['status']}`",
        f"- patch interpretation: `{patch['scientific_interpretation']}`",
        "- official full-track, custom temporal rank, visual aperture, and human judgments remain separate endpoint families.",
        "- no mediation percentage is computed.",
    ]
    paths.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def run_stage(paths: RunPaths, config: Mapping[str, Any], stage: str, *, parent_override: Path | None = None, bindings_override: Path | None = None, ratings_path: Path | None = None, visual_override: bool | None = None) -> dict[str, Any]:
    if stage == "prepare":
        return prepare(paths, config, parent_override=parent_override, bindings_override=bindings_override)
    if stage == "evaluate-existing":
        return evaluate_existing(paths, config, visual_override=visual_override)
    if stage == "export-blind":
        return export_blind(paths, config)
    if stage == "import-ratings":
        if ratings_path is None:
            raise ValueError("--ratings is required for import-ratings")
        return import_human_ratings(paths, ratings_path)
    if stage == "patch-preflight":
        return patch_preflight(paths)
    if stage in {"patch-smoke", "patch-render"}:
        return _render_stage(paths, config, stage=stage)
    if stage == "patch-evaluate":
        if bool(_config_value(config, "run_patch_official_syncnet", False)):
            from .protocol import assert_gpu_available, gpu_lease

            target_uuid = _config_value(config, "gpu_uuid")
            with gpu_lease(target_uuid=target_uuid, receipt_path=paths.patch_eval / "official.gpu_lease.json"):
                result = patch_evaluate(paths, config)
                assert_gpu_available(target_uuid=target_uuid)
                return result
        return patch_evaluate(paths, config)
    if stage == "analyze":
        return analyze(paths)
    if stage == "check":
        return validate_run(paths.root)
    raise ValueError(f"unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTS official-evidence audit and temporal activation patch protocol")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "scripts/configs/tts_evidence_temporal_patch_v1.yaml")
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--input-bindings", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--skip-visual", action="store_true", help="record visual timing as explicitly pending for a CPU acceptance run")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    paths = run_paths(args.run_id, repo_root=REPO)
    run_stage(paths, config, args.stage, parent_override=args.parent_run, bindings_override=args.input_bindings, ratings_path=args.ratings, visual_override=False if args.skip_visual else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
