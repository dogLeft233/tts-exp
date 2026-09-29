from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import audit, config
from .common import ProtocolError, file_sha256, load_self_hashed, read_json, write_json


def prepare(paths: config.RunPaths) -> None:
    if paths.protocol.exists() and paths.input_audit.exists(): load_self_hashed(paths.protocol); load_self_hashed(paths.input_audit); return
    source_paths = {
        "A": {"protocol": config.A / "protocol.json", "drivers": config.A / "drivers/manifest.json", "control_analysis": config.A / "control_analysis.json", "candidate_scores": config.A / "candidate_scores/manifest.json", "analysis": config.A / "analysis.json"},
        "B": {"protocol": config.B / "protocol.json", "reference": config.B / "reference_manifest.json", "control_scores": config.B / "control_scores/manifest.json", "control_analysis": config.B / "control_analysis.json", "final": config.B / "final.json"},
        "C": {"protocol": config.C / "protocol.json", "exposure": config.C / "exposure.json", "reused_manifest": config.C / "reused_manifest.json", "analysis": config.C / "analysis.json"},
    }
    file_key_map = {
        "A/protocol.json": ("A", "protocol"), "A/drivers/manifest.json": ("A", "drivers"), "A/control_analysis.json": ("A", "control_analysis"), "A/candidate_scores/manifest.json": ("A", "candidate_scores"), "A/analysis.json": ("A", "analysis"),
        "B/protocol.json": ("B", "protocol"), "B/reference_manifest.json": ("B", "reference"), "B/control_scores/manifest.json": ("B", "control_scores"), "B/control_analysis.json": ("B", "control_analysis"), "B/final.json": ("B", "final"),
        "C/protocol.json": ("C", "protocol"), "C/exposure.json": ("C", "exposure"), "C/reused_manifest.json": ("C", "reused_manifest"), "C/analysis.json": ("C", "analysis"),
    }
    for name, expected in config.FIXED_FILES.items():
        group, key = file_key_map[name]
        path = source_paths[group][key]
        if path is None or not path.is_file() or file_sha256(path) != expected: raise ProtocolError(f"fixed audit input changed: {name}")
    for paths_map, hashes in ((config.P_FILES, config.P_HASHES), (config.Q_FILES, config.Q_HASHES), (config.D_FILES, config.D_HASHES)):
        for key, path in paths_map.items():
            if not path.is_file() or file_sha256(path) != hashes[key]: raise ProtocolError(f"shared asset changed: {key}")
    protocol = {"schema_version": 1, "protocol_id": "wav2lip_parallel_evidence_contract", "status": "locked", "source_files": {name: {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in values.items()} for name, values in source_paths.items()}, "shared_files": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in {**config.P_FILES, **config.Q_FILES, **config.D_FILES}.items()}, "fixed_hashes": config.FIXED_FILES, "zero_model_budget": True, "candidate_authorized": False}
    write_json(paths.protocol, protocol); write_json(paths.input_audit, {"status": "complete", "fixed_inputs": protocol["source_files"], "shared_inputs": protocol["shared_files"], "no_parent_runner_called": True})


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = config.RunPaths(config.run_root_for(run_id)); paths.root.mkdir(parents=True, exist_ok=True)
    try:
        if any(paths.root.iterdir()) and not resume and not paths.protocol.exists(): raise ProtocolError(f"refusing non-empty output directory: {paths.root}")
        prepare(paths)
        if stage == "prepare": return 0
        result = audit.run_audit(); write_json(paths.per_record, {"A": result["A"]["records"], "B": result["B"]["records"], "C": result["C"]["records"]}); write_json(paths.recomputed, result); write_json(paths.discrepancies, {"contract_violations": result["contract_violations"], "A": result["A"].get("contrasts"), "B": {"legacy_offset_pass_count": result["B"]["legacy_offset_pass_count"], "matched_offset_pass_count": result["B"]["matched_offset_pass_count"], "f46_control_as_spec": result["B"]["f46_control_as_spec"]}, "C": result["C"]["stats"]})
        if stage == "analyze": return 0
        subprocess.run([sys.executable, "-m", "scripts.experiments.wav2lip_parallel_evidence_contract.validate", "--run-root", str(paths.root)], cwd=config.REPO, check=True)
        validation = load_self_hashed(paths.validation); review = write_json(paths.review, {"schema_version": 1, "status": "self_reviewed", "independent_validator": validation.get("independent"), "contract_audit": "A/B/C producer-validator contract defects recorded", "replacement_confirmed": False}); final = write_json(paths.final, {"schema_version": 1, "status": "complete", "engineering_decision": validation["engineering_decision"], "scientific_decision": validation["scientific_decision"], "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "review_sha256": review["artifact_sha256"], "validation_sha256": validation["artifact_sha256"]})
        paths.result.write_text(f"# Wav2Lip parallel evidence contract audit\n\n- engineering: `{final['engineering_decision']}`\n- scientific: `{final['scientific_decision']}`\n- B matched-domain control: `{validation['B']['f46_control_as_spec']}`\n- candidate authorization: `false`\n", encoding="utf-8"); paths.root.joinpath("error.json").unlink(missing_ok=True); return 0
    except Exception as exc:
        write_json(paths.root / "error.json", {"schema_version": 1, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}); return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--stage", choices=("prepare", "analyze", "all"), default="all"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(argv); return run(args.run_id, args.stage, args.resume)


if __name__ == "__main__": raise SystemExit(main())
