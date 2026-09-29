"""Independent checker for the new run contract.

The checker does not import the runner's scoring, aggregation, support, or
selection functions.  It only validates serialized inputs, hashes, and basic
media invariants, then reports whether the run is complete or partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - exercised by the CLI smoke test
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.experiments.phone_gain_static_tfg_mfa.assets import FORBIDDEN_FIELDS
    from scripts.experiments.phone_gain_static_tfg_mfa.config import canonical_hash, file_sha256, read_json, write_json
else:
    from .assets import FORBIDDEN_FIELDS
    from .config import canonical_hash, file_sha256, read_json, write_json


def _finite(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_finite(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_finite(item, f"{path}[{index}]"))
    return errors


def _contains_forbidden(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_FIELDS:
                errors.append(f"{path}.{key}")
            errors.extend(_contains_forbidden(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_contains_forbidden(item, f"{path}[{index}]"))
    return errors


def check_run(run_dir: str | Path, *, require_complete: bool = False) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = ["protocol.json", "status.json", "00_protocol/protocol_lock.json", "00_protocol/registry.json", "00_protocol/support.json", "00_protocol/training_manifest.json", "00_protocol/inference_manifest.json", "00_protocol/render_manifest.json", "00_protocol/portraits.json", "00_protocol/audit_findings.json", "00_protocol/expected_artifacts.json"]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"MISSING:{relative}")
    if errors:
        return {"status": "FAIL", "run_dir": str(root), "errors": errors, "warnings": warnings}
    try:
        protocol = read_json(root / "protocol.json")
        status = read_json(root / "status.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "run_dir": str(root), "errors": [f"PROTOCOL_INVALID:{type(exc).__name__}"], "warnings": warnings}
    protocol_id = str(protocol.get("protocol_id", ""))
    if not protocol_id.startswith("phone_gain_static_tfg_mfa_"):
        errors.append("PROTOCOL_MISMATCH")
    if not str(protocol.get("measurement_version", "")).startswith("phone_gain_static_tfg_mfa_"):
        errors.append("MEASUREMENT_VERSION_MISMATCH")
    lock = read_json(root / "00_protocol/protocol_lock.json")
    actual_protocol_sha = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    if lock.get("protocol_sha256") != actual_protocol_sha:
        errors.append("PROTOCOL_LOCK_HASH_MISMATCH")
    if lock.get("protocol", {}).get("config_sha256") != protocol.get("config_sha256"):
        errors.append("PROTOCOL_LOCK_CONFIG_MISMATCH")
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            errors.extend(f"NONFINITE:{path.relative_to(root)}:{item}" for item in _finite(payload))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"JSON_INVALID:{path.relative_to(root)}:{type(exc).__name__}")
    for relative in ("00_protocol/training_manifest.json", "00_protocol/inference_manifest.json", "00_protocol/render_manifest.json"):
        payload = read_json(root / relative)
        errors.extend(f"FORBIDDEN_FIELD:{relative}:{item}" for item in _contains_forbidden(payload))
    portraits = read_json(root / "00_protocol/portraits.json")
    for portrait_id, meta in portraits.items():
        path = Path(str(meta.get("path", "")))
        if not path.is_file():
            errors.append(f"PORTRAIT_MISSING:{portrait_id}")
        elif file_sha256(path) != meta.get("container_sha256"):
            errors.append(f"PORTRAIT_HASH_MISMATCH:{portrait_id}")
        box = meta.get("generation_box_xyxy")
        if isinstance(box, list) and box == [0, 0, int(meta.get("width", 0)), int(meta.get("height", 0))]:
            errors.append(f"FULL_FRAME_GENERATION_BOX:{portrait_id}")
    support = read_json(root / "00_protocol/support.json")
    if not support.get("support_hash"):
        errors.append("SUPPORT_HASH_MISSING")
    else:
        expected_support_hash = canonical_hash({"labels": support.get("labels", []), "natural_primary": support.get("natural_primary", {}), "matched_nt": support.get("matched_nt", {}), "mixed_centroids": support.get("mixed_centroids", {})})
        if expected_support_hash != support.get("support_hash"):
            errors.append("SUPPORT_HASH_MISMATCH")
    if status.get("states", {}).get("train", {}).get("state") == "RESOURCE_WAIT":
        warnings.append("TRAIN_RESOURCE_WAIT")
    if status.get("states", {}).get("render", {}).get("state") == "RESOURCE_WAIT":
        warnings.append("RENDER_RESOURCE_WAIT")
    if not (root / "09_report/decision.json").is_file():
        warnings.append("REPORT_NOT_YET_WRITTEN")
    scientific = all((root / relative).is_file() for relative in ("05_phone/summary.json", "08_sync/summary.json"))
    registry = read_json(root / "00_protocol/registry.json")
    if registry.get("split_counts") != {"fit": 135, "dev": 65, "e_seen": 40}:
        errors.append("COHORT_SPLIT_COUNTS_MISMATCH")
    audio_path = root / "04_audio/manifest.json"
    if audio_path.is_file():
        audio = read_json(audio_path)
        rows = audio.get("rows", [])
        expected_e_seen = int(protocol.get("config", {}).get("cohort", {}).get("split_counts", {}).get("e_seen", 40))
        if audio.get("status") == "COMPLETE" and len(rows) != expected_e_seen:
            errors.append("AUDIO_EXPECTED_E_SEEN_ROWS_MISMATCH")
        for row in rows:
            if set(str(key) for key in row.get("paths", {})) != {"N", "T", "D", "A", "B", "C"}:
                errors.append(f"AUDIO_ARM_SET_MISMATCH:{row.get('pair_id')}")
            for arm, raw_path in row.get("paths", {}).items():
                path = Path(str(raw_path))
                if not path.is_file():
                    errors.append(f"AUDIO_MISSING:{row.get('pair_id')}:{arm}")
                    continue
                try:
                    with wave.open(str(path), "rb") as handle:
                        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
                            errors.append(f"AUDIO_PCM_FORMAT:{row.get('pair_id')}:{arm}")
                except (OSError, wave.Error):
                    errors.append(f"AUDIO_INVALID:{row.get('pair_id')}:{arm}")
        if audio.get("source") == "OLD_READ_ONLY":
            old_manifest = Path(str(audio.get("old_manifest", "")))
            if not old_manifest.is_file():
                errors.append("OLD_AUDIO_MANIFEST_MISSING")
            else:
                old = read_json(old_manifest)
                old_by_pair = {str(item.get("pair_id")): item for item in old.get("rows", [])}
                for row in rows:
                    old_row = old_by_pair.get(str(row.get("pair_id")))
                    if old_row is None:
                        errors.append(f"OLD_AUDIO_PAIR_MISSING:{row.get('pair_id')}")
                        continue
                    for arm, path in row.get("paths", {}).items():
                        if str(Path(str(path)).resolve()) != str(Path(str(old_row.get("paths", {}).get(arm, ""))).resolve()):
                            errors.append(f"OLD_AUDIO_PATH_CHANGED:{row.get('pair_id')}:{arm}")
    render_path = root / "07_tfg/manifest.json"
    if render_path.is_file():
        render = read_json(render_path)
        expected = int(render.get("expected_cells", 0))
        if render.get("status") == "COMPLETE" and expected and int(render.get("cells", -1)) != expected:
            errors.append("RENDER_EXPECTED_CELL_COUNT_MISMATCH")
        seen_cells: set[tuple[str, str, str, str]] = set()
        for cell in render.get("results", []):
            output = Path(str(cell.get("output", "")))
            if not output.is_file():
                errors.append(f"RENDER_OUTPUT_MISSING:{cell.get('pair_id')}:{cell.get('video_arm')}")
                continue
            if cell.get("output_sha256") != file_sha256(output):
                errors.append(f"RENDER_OUTPUT_HASH_MISMATCH:{cell.get('pair_id')}:{cell.get('video_arm')}")
            seen_cells.add((str(cell.get("pair_id")), str(cell.get("portrait_id")), str(cell.get("seed")), str(cell.get("video_arm"))))
        if render.get("status") == "COMPLETE" and int(render.get("cells", 0)) != len(seen_cells):
            errors.append("RENDER_SEMANTIC_CELL_DUPLICATE")
    sync_path = root / "08_sync/summary.json"
    if sync_path.is_file():
        sync = read_json(sync_path)
        if sync.get("status") == "COMPLETE" and int(sync.get("complete_cells", -1)) != int(sync.get("expected_cells", -2)):
            errors.append("SYNC_EXPECTED_CELL_COUNT_MISMATCH")
    stage_states = status.get("states", {})
    expected_states = {"audit", "calibrate", "infer", "quality", "phone", "render", "score", "official", "analyze", "report"}
    if protocol_id.endswith("repair_v2"):
        expected_states.update({"train", "lock"})
    missing_states = sorted(stage for stage in expected_states if stage not in stage_states)
    if missing_states:
        warnings.append("MISSING_STAGE_STATES:" + ",".join(missing_states))
    complete_states = {"COMPLETE", "NOT_APPLICABLE"}
    execution_complete = not missing_states and all(str(stage_states.get(stage, {}).get("state")) in complete_states for stage in expected_states)
    if require_complete and not execution_complete:
        errors.append("REQUIRED_SCOPE_INCOMPLETE")
    result = {"status": "PASS" if not errors and scientific and (execution_complete or not require_complete) else ("PARTIAL" if not errors else "FAIL"), "run_dir": str(root), "protocol": protocol_id, "measurement_version": protocol.get("measurement_version"), "scientific_outputs": scientific, "execution_complete": execution_complete, "errors": errors, "warnings": warnings, "independent": {"json_finite": not any(item.startswith("NONFINITE") for item in errors), "no_video_in_manifests": not any(item.startswith("FORBIDDEN_FIELD") for item in errors), "portrait_hashes": not any(item.startswith("PORTRAIT_HASH") for item in errors), "expected_scope": not bool(missing_states)}}
    write_json(root / "09_report/checker.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    result = check_run(args.run_dir, require_complete=bool(args.require_complete))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["check_run", "main"]
