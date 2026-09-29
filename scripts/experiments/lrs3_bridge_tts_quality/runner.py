"""Command line entry point for the staged bridge comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .audio import run_bridge_stage
from .common import (
    ProtocolError,
    assert_run_root_compatible,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import run_audit
from .quality import analyze_ratings, run_quality_pack
from .render import run_render_stage
from .scoring import run_score_stage
from .stats import write_analysis_stage
from .targets import run_targets_stage
from .tts import run_tts_stage

STAGES = ("audit", "tts", "targets", "bridge", "quality-pack", "render", "score", "analyze")


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _base(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = config.RunPaths(run_root)
    setup = _load(paths.protocol / "setup.json")
    cohort = _load(paths.protocol / "cohort.json")
    audit = _load(paths.protocol / "input_audit.json")
    if setup.get("status") != "complete" or cohort.get("status") != "complete" or audit.get("status") != "complete":
        raise ProtocolError("Stage 00 audit is not complete")
    return setup, cohort, audit


def _bound_files(value: Any, output: dict[str, dict[str, str]]) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _bound_files(item, output)
    elif isinstance(value, list):
        for item in value:
            _bound_files(item, output)
    elif isinstance(value, str):
        path = Path(value)
        try:
            is_file = path.is_file()
        except OSError:
            # JSON also contains transcripts and other arbitrary strings. A
            # long transcript can exceed the filesystem's filename limit when
            # pathlib probes it; such values are not file bindings.
            return
        if is_file:
            resolved = str(path.resolve())
            output.setdefault(resolved, {"path": resolved, "sha256": file_sha256(path)})


def _freeze_analysis_lock(run_root: Path) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    setup, cohort, _ = _base(run_root)
    required = {
        "setup": paths.protocol / "setup.json",
        "cohort": paths.protocol / "cohort.json",
        "input_audit": paths.protocol / "input_audit.json",
        "execution_order": paths.protocol / "execution_order.json",
        "tts": paths.tts / "tts_manifest.json",
        "targets_alignment": paths.targets / "alignment_manifest.json",
        "targets": paths.targets / "targets_manifest.json",
        "bridge_audio": paths.bridge / "audio_manifest.json",
        "bridge_diagnostics": paths.bridge / "diagnostics.json",
        "geometry": paths.bridge / "geometry.json",
        "quality_listening": paths.quality / "listening_manifest.json",
        "quality_blind_mapping": paths.quality / "blind_mapping.json",
        "quality_questionnaire": paths.quality / "questionnaire.md",
        "quality_template": paths.quality / "ratings_template.csv",
    }
    bindings: dict[str, dict[str, str]] = {}
    for name, path in required.items():
        if not path.is_file():
            raise ProtocolError(f"analysis lock input is missing: {path}")
        bindings[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    bound_files: dict[str, dict[str, str]] = {}
    for path in required.values():
        if path.suffix.lower() == ".json":
            _bound_files(_load(path), bound_files)
    for record in cohort["records"]:
        _bound_files(record, bound_files)
    lock = {
        "schema_version": 1,
        "stage_id": "03_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "frozen_before_scoring": True,
        "setup_sha256": file_sha256(paths.protocol / "setup.json"),
        "ordered_sample_ids": setup["cohort"]["ordered_sample_ids"],
        "record_count": len(cohort["records"]),
        "input_and_intermediate_bindings": bindings,
        "all_bound_file_hashes": sorted(bound_files.values(), key=lambda value: value["path"]),
        "video_hashes_prepopulated": False,
        "score_hashes_prepopulated": False,
        "quality_ratings_may_be_added_later": True,
        "score_read_count": 0,
        "selection_used_scores": False,
        "selection_used_quality": False,
    }
    existing = paths.bridge / "analysis_lock.json"
    if existing.is_file():
        old = _load(existing)
        if {key: value for key, value in old.items() if key != "artifact_sha256"} != lock:
            raise ProtocolError("analysis lock already exists with different frozen inputs")
        return old
    write_self_hashed_json(existing, lock)
    return _load(existing)


def _latest_analysis(paths: config.RunPaths) -> tuple[int, Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    base_final = paths.analysis / "final.json"
    if base_final.is_file():
        payload = _load(base_final)
        candidates.append((int(payload.get("analysis_version", 1)), base_final, payload))
    if paths.analysis_versions.exists():
        for directory in paths.analysis_versions.iterdir():
            if not directory.is_dir() or not directory.name.startswith("v") or not directory.name[1:].isdigit():
                continue
            final = directory / "final.json"
            if not final.is_file():
                raise ProtocolError(f"partial analysis version cannot be resumed: {directory}")
            payload = _load(final)
            candidates.append((int(payload.get("analysis_version", directory.name[1:])), final, payload))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], str(item[1])))


def _analysis_destination(
    paths: config.RunPaths,
) -> tuple[Path | None, int, str | None, dict[str, Any] | None]:
    latest = _latest_analysis(paths)
    quality_hash = file_sha256(paths.quality / "quality.json") if (paths.quality / "quality.json").is_file() else None
    if latest is None:
        return paths.analysis, 1, None, None
    version, final_path, final = latest
    if final.get("quality_input_sha256") == quality_hash:
        return None, version, None, final
    next_version = version + 1
    destination = paths.analysis_versions / f"v{next_version}"
    if destination.exists():
        raise ProtocolError(f"analysis version destination already exists: {destination}")
    return destination, next_version, file_sha256(final_path), None


def run_stage(run_id: str, stage: str, *, local_manifest: Path | None = None, ratings: Path | None = None) -> dict[str, Any]:
    config.validate_run_id(run_id)
    if stage not in STAGES and stage != "all":
        raise ValueError(f"unknown stage {stage}; expected one of {STAGES}")
    run_root = config.run_root_for(run_id)
    assert_run_root_compatible(run_root)
    if stage == "all":
        result: dict[str, Any] = {}
        for item in STAGES:
            result[item] = run_stage(run_id, item, local_manifest=local_manifest, ratings=ratings)
        return result
    if stage == "audit":
        return run_audit(run_root)
    setup, cohort, _ = _base(run_root)
    paths = config.RunPaths(run_root)
    if stage == "tts":
        return run_tts_stage(run_root, cohort, setup, local_manifest=local_manifest)
    if stage == "targets":
        return run_targets_stage(run_root, cohort, _load(paths.tts / "tts_manifest.json"))
    if stage == "bridge":
        return run_bridge_stage(run_root, cohort, _load(paths.targets / "targets_manifest.json"))
    if stage == "quality-pack":
        listening = run_quality_pack(run_root, cohort, _load(paths.tts / "tts_manifest.json"), _load(paths.targets / "targets_manifest.json"))
        lock = _freeze_analysis_lock(run_root)
        return {"status": listening.get("status"), "stimulus_count": listening.get("stimulus_count"), "analysis_lock_sha256": file_sha256(paths.bridge / "analysis_lock.json"), "lock": lock}
    if stage == "render":
        _load(paths.bridge / "analysis_lock.json")
        return run_render_stage(run_root, cohort)
    if stage == "score":
        _load(paths.bridge / "analysis_lock.json")
        return run_score_stage(run_root, cohort)
    if stage == "analyze":
        quality_path = paths.quality / "quality.json"
        quality = _load(quality_path) if quality_path.is_file() else None
        if ratings is not None:
            quality = analyze_ratings(run_root, ratings.resolve())
        destination, version, parent_hash, existing = _analysis_destination(paths)
        if existing is not None:
            return existing
        return write_analysis_stage(run_root, quality=quality, engineering_complete=True, output_dir=destination, analysis_version=version, parent_analysis_sha256=parent_hash)
    raise AssertionError(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description="LRS3 bridge comparison staged runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--local-manifest", type=Path, default=None, help="optional compatible existing local TTS manifest")
    parser.add_argument("--ratings", type=Path, default=None, help="blinded human ratings CSV; used only by analyze")
    args = parser.parse_args()
    try:
        result = run_stage(args.run_id, args.stage, local_manifest=args.local_manifest, ratings=args.ratings)
    except Exception as exc:  # noqa: BLE001 - CLI reports the concrete blocked stage
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
