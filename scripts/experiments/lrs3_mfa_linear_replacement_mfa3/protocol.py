from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_mfa_linear_replacement import protocol as base_protocol

from .mfa3_alignment import (
    MFA_ACOUSTIC_MODEL,
    MFA_DEFAULT_ROOT,
    MFA_DICTIONARY,
    MFA_VERSION,
    NORMALIZATION_POLICY_ID,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_20260825"
STAGE_PREFIX = "00_protocol_lock_mfa3_retry"
EXPECTED_PARENT_STAGE00 = REPO / "runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json"
EXPECTED_PARENT_STAGE00_SHA256 = "feb1c1de09361f8b032f1a9e547ab5a69e09894d8334b3339ea9652157a057f6"


def stage_id_for_output(output_dir: Path) -> str:
    stage_id = output_dir.name
    if not stage_id.startswith(STAGE_PREFIX):
        raise ValueError(f"unexpected MFA3 Stage00 output directory name: {stage_id}")
    return stage_id


def next_stage_id(stage_id: str) -> str:
    return stage_id.replace("00_protocol_lock_mfa3", "01_candidate_audio_mfa3", 1)
EXPECTED_PARENT_STAGE00_SHA256 = "feb1c1de09361f8b032f1a9e547ab5a69e09894d8334b3339ea9652157a057f6"


def _bound_path(binding: Mapping[str, Any]) -> Path:
    value = Path(str(binding["path"]))
    return (value if value.is_absolute() else REPO / value).resolve()


def _bindings_to_paths(bindings: Mapping[str, Any]) -> dict[str, Path]:
    return {str(name): _bound_path(value) for name, value in bindings.items()}


def _contains_exact_model(output: str, model_name: str) -> bool:
    return model_name in re.findall(r"[A-Za-z0-9_.-]+", output)


def _contains_exact_version(output: str, version: str) -> bool:
    return version in re.findall(r"(?<![0-9])\d+\.\d+\.\d+(?![0-9])", output)


def _run_mfa_metadata(executable: Path, root_dir: Path) -> dict[str, Any]:
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(root_dir)
    environment["PATH"] = os.pathsep.join([str(executable.parent), environment.get("PATH", "")])
    commands = {
        "version": [str(executable), "version"],
        "acoustic_models": [str(executable), "model", "list", "acoustic"],
        "dictionary_models": [str(executable), "model", "list", "dictionary"],
    }
    results: dict[str, str] = {}
    for name, command in commands.items():
        result = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
        if result.returncode != 0:
            raise RuntimeError(f"MFA3 metadata command failed: {name}: exit {result.returncode}")
        results[name] = (result.stdout + result.stderr).strip()[-2000:]
    if not _contains_exact_version(results["version"], MFA_VERSION):
        raise RuntimeError(f"unexpected MFA version output: {results['version']!r}")
    if not _contains_exact_model(results["acoustic_models"], MFA_ACOUSTIC_MODEL):
        raise RuntimeError("MFA3 english_mfa acoustic model is not installed")
    if not _contains_exact_model(results["dictionary_models"], MFA_DICTIONARY):
        raise RuntimeError("MFA3 english_mfa dictionary is not installed")
    dictionary_path = root_dir / "pretrained_models/dictionary/english_mfa.dict"
    acoustic_path = root_dir / "pretrained_models/acoustic/english_mfa.zip"
    config_path = root_dir / "global_config.yaml"
    for path in (executable, dictionary_path, acoustic_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "version": MFA_VERSION,
        "metadata_commands": commands,
        "metadata_output": results,
        "root_dir": str(root_dir),
        "root_config": {"path": str(config_path), "sha256": base_protocol.file_sha256(config_path)},
        "executable": {"path": str(executable), "sha256": base_protocol.file_sha256(executable)},
        "dictionary": {
            "name": MFA_DICTIONARY,
            "path": str(dictionary_path),
            "sha256": base_protocol.file_sha256(dictionary_path),
        },
        "acoustic_model": {
            "name": MFA_ACOUSTIC_MODEL,
            "path": str(acoustic_path),
            "sha256": base_protocol.file_sha256(acoustic_path),
        },
    }


def build_stage00_from_parent(
    *,
    parent_stage00_path: Path,
    output_dir: Path,
    mfa_executable: Path,
    mfa_root_dir: Path,
    code_review_path: Path,
) -> dict[str, Any]:
    parent_stage00_path = parent_stage00_path.resolve()
    if parent_stage00_path != EXPECTED_PARENT_STAGE00.resolve():
        raise ValueError("MFA3 retry must use the immutable prior Stage00 parent path")
    parent_hash = base_protocol.file_sha256(parent_stage00_path)
    if parent_hash != EXPECTED_PARENT_STAGE00_SHA256:
        raise ValueError("immutable prior Stage00 parent hash changed")
    parent = base_protocol.load_json(parent_stage00_path)
    if parent.get("protocol_id") != "lrs3_mfa_linear_replacement_20260824":
        raise ValueError("unexpected prior Stage00 protocol")
    parent_files = parent["parents"]["parent_files"]
    parent_file_paths = _bindings_to_paths(parent_files)
    parent_visual_manifest = base_protocol.load_json(parent_file_paths["visual_manifest"])
    parent_visual_lock = base_protocol.load_json(parent_file_paths["visual_lock"])
    parent_motion_manifest = base_protocol.load_json(parent_file_paths["motion_manifest"])
    parent_motion_lock = base_protocol.load_json(parent_file_paths["motion_lock"])
    source_manifest = base_protocol.load_json(parent_file_paths["source_manifest"])
    tts_meta = base_protocol.load_json(parent_file_paths["tts_meta"])

    mfa_executable = mfa_executable.resolve()
    mfa_root_dir = mfa_root_dir.resolve()
    mfa_contract = _run_mfa_metadata(mfa_executable, mfa_root_dir)
    old_assets = _bindings_to_paths(parent.get("assets", {}))
    old_assets.update({
        "mfa_executable": mfa_executable,
        "mfa_dictionary": Path(mfa_contract["dictionary"]["path"]),
        "mfa_acoustic": Path(mfa_contract["acoustic_model"]["path"]),
    })
    old_sources = _bindings_to_paths(parent.get("source_manifest", {}))
    old_sources.update({
        "mfa3_protocol": Path(__file__),
        "mfa3_alignment": Path(__file__).with_name("mfa3_alignment.py"),
        "mfa3_stage01": Path(__file__).with_name("run_stage01_mfa3.py"),
    })
    old_tests = _bindings_to_paths(parent.get("test_manifest", {}))
    old_tests["mfa3_tests"] = REPO / "tests/experiments/lrs3_mfa_linear_replacement_mfa3/test_mfa3.py"
    stage_id = stage_id_for_output(output_dir)
    previous_protocol_id = base_protocol.PROTOCOL_ID
    previous_stage_id = base_protocol.STAGE_ID
    base_protocol.PROTOCOL_ID = PROTOCOL_ID
    base_protocol.STAGE_ID = stage_id
    try:
        artifacts = base_protocol.build_stage00_artifacts(
            parent_visual_manifest=parent_visual_manifest,
            parent_visual_lock=parent_visual_lock,
            parent_motion_manifest=parent_motion_manifest,
            parent_motion_lock=parent_motion_lock,
            source_manifest=source_manifest,
            tts_meta=tts_meta,
            asset_paths=old_assets,
            source_paths=old_sources,
            test_paths=old_tests,
            parent_files=parent_file_paths,
            review_files={"mfa3_code_review": code_review_path.resolve()},
        )
    finally:
        base_protocol.PROTOCOL_ID = previous_protocol_id
        base_protocol.STAGE_ID = previous_stage_id
    manifest = artifacts["manifest"]
    manifest["parents"]["prior_mfa2_stage00"] = {
        "path": str(parent_stage00_path.relative_to(REPO)),
        "sha256": parent_hash,
    }
    manifest["mfa_contract"] = mfa_contract
    manifest["normalization_policy"] = {
        "policy_id": NORMALIZATION_POLICY_ID,
        "number_tokens": "expand cardinal and ordinal decimal tokens to English words",
        "recognized_nonlexical_events": ["{LG}", "{LAUGHTER}", "[LAUGHTER]"],
        "recognized_nonlexical_event_action": "remove from MFA transcript; never map its phone to silence after alignment",
        "unsupported_bracketed_annotations": "engineering failure",
        "apostrophe_policy": "normalize curly apostrophes to ASCII; preserve lexical apostrophes and contractions",
        "unknown_phone_policy": "spn, unk, unknown, oov and related labels remain engineering failures",
    }
    manifest["stage_id"] = stage_id
    manifest["protocol_id"] = PROTOCOL_ID
    artifacts["summary"].update({"stage_id": stage_id, "mfa_contract": mfa_contract, "normalization_policy_id": NORMALIZATION_POLICY_ID})
    artifacts["decision"].update({"stage_id": stage_id, "next_allowed_stage": next_stage_id(stage_id)})
    return artifacts


def write_stage00_artifacts(output_dir: str | Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    base_protocol.write_stage00_artifacts(output_dir, artifacts)
    target = Path(output_dir)
    (target / "manifest.sha256").write_text(base_protocol.file_sha256(target / "manifest.json") + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-stage00", type=Path, default=EXPECTED_PARENT_STAGE00)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mfa-executable", type=Path, default=Path("/home/wjj/miniconda3/envs/mfa3/bin/mfa"))
    parser.add_argument("--mfa-root-dir", type=Path, default=MFA_DEFAULT_ROOT)
    parser.add_argument("--code-review", type=Path, required=True)
    args = parser.parse_args()
    artifacts = build_stage00_from_parent(
        parent_stage00_path=args.parent_stage00,
        output_dir=args.output,
        mfa_executable=args.mfa_executable,
        mfa_root_dir=args.mfa_root_dir,
        code_review_path=args.code_review,
    )
    write_stage00_artifacts(args.output, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
