from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(r"E:\Documents\tts-exp")
PACKET = Path(__file__).resolve().parent
DATA = WORKSPACE / ".codex/scratch/dev20_future_cohort_metadata_attempt06"
ROOT_REMOTE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/dev20_future_cohort_source_attempt01"
REMOTE_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"

BUNDLE = DATA / "future_dev20_source_bundle_attempt02.zip"
MANIFEST = DATA / "future_dev20_portable_source_manifest_attempt02.json"
BUILD_RECEIPT = DATA / "future_dev20_source_bundle_build_receipt_attempt02.json"
DOWNLOAD_RECEIPT = DATA / "aishell_future_dev20_download_readback_attempt01.json"
SOURCE_PLAN = DATA / "aishell_future_dev20_exact_source_download_plan_attempt01.json"
HEADER_RECEIPT = DATA / "aishell_future_dev20_header_eligibility_attempt01.json"
FIT_RECEIPT = DATA / "fit40_existing_source_hash_reconciliation_attempt01.json"
COHORT_PLAN = WORKSPACE / ".codex/scratch/dev20_future_cohort_metadata_attempt02/dev20_future_cohort_metadata_plan_attempt02.json"
README = DATA / "pinned_README_attempt01.md"
AUDIO_IO = WORKSPACE / "scripts/audio_enhancement/audio_io.py"
BUILDER = DATA / "build_dev20_portable_source_bundle_attempt02.py"
WORKER = PACKET / "extract_and_qc_dev20_source_stage_attempt02.py"
INDEX = PACKET / "request_index_attempt04.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_sha(obj: dict[str, Any]) -> str:
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def raw_pin(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha256(data)}


def write_create_only(path: Path, obj: dict[str, Any]) -> bytes:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return data


def main() -> None:
    if INDEX.exists():
        raise FileExistsError("request index already exists")
    receipt = json.loads(BUILD_RECEIPT.read_bytes())
    source_plan = json.loads(SOURCE_PLAN.read_bytes())
    download = json.loads(DOWNLOAD_RECEIPT.read_bytes())
    if receipt["status"] != "BUNDLE_CREATED_READBACK_VERIFIED_NOT_UPLOADED":
        raise RuntimeError("portable source bundle is not sealed/readback verified")
    if receipt["bundle"]["bytes"] != 3_548_546 or receipt["bundle"]["sha256"] != "6f9de4953da6fd52a4e916604c21b0fd187ab145fefc92ea47cd8e274a460d9a":
        raise RuntimeError("portable bundle pin differs from actual build receipt")
    if receipt["source_manifest"]["bytes"] != 24_072 or receipt["source_manifest"]["sha256"] != "fff18619b7be8f992002f99c33121ef3eb25acce6aa3caa3da55f398eb3faffa":
        raise RuntimeError("portable manifest pin differs from actual build receipt")
    if download["status"] != "ALL_20_DOWNLOADED_HASH_VERIFIED_AND_CANONICALIZED":
        raise RuntimeError("actual local download/readback receipt not complete")
    if len(source_plan["files"]) != 20 or len(download["files"]) != 20:
        raise RuntimeError("expected exact twenty-file source plan and download receipt")

    q_defs: list[tuple[str, dict[str, Any]]] = []
    remote_parent = ROOT_REMOTE.rsplit("/", 1)[0]
    old_python = "/root/autodl-tmp/envs/ditto/bin/python"
    mkdir_code = "\n".join(
        [
            "import json, os, sys",
            "from pathlib import Path",
            f"old=Path({json.dumps(old_python)})",
            f"selected=Path({json.dumps(REMOTE_PYTHON)})",
            f"parent=Path({json.dumps(remote_parent)})",
            f"p=Path({json.dumps(ROOT_REMOTE)})",
            "checks={",
            " 'old_ditto_interpreter':{'path':str(old),'exists':old.exists(),'is_file':old.is_file(),'executable':os.access(str(old),os.X_OK)},",
            " 'selected_interpreter':{'path':str(selected),'exists':selected.exists(),'is_file':selected.is_file(),'executable':os.access(str(selected),os.X_OK),'running_executable':sys.executable},",
            " 'research_namespace_parent':{'path':str(parent),'exists':parent.exists(),'is_dir':parent.is_dir(),'writable_searchable':os.access(str(parent),os.W_OK|os.X_OK)},",
            " 'remote_root':{'path':str(p),'exists':p.exists(),'is_symlink':p.is_symlink()},",
            "}",
            "if not (selected.exists() and selected.is_file() and os.access(str(selected),os.X_OK) and parent.is_dir() and os.access(str(parent),os.W_OK|os.X_OK) and not p.exists() and not p.is_symlink()):",
            " print(json.dumps({'status':'PREFLIGHT_FAILED_NO_MUTATION','checks':checks},sort_keys=True,separators=(',',':')));sys.exit(2)",
            "p.mkdir(mode=0o755,parents=False,exist_ok=False)",
            "print(json.dumps({'status':'REMOTE_SOURCE_DIRECTORY_CREATED_CREATE_ONLY','path':str(p),'checks':checks},sort_keys=True,separators=(',',':'))) ",
        ]
    )
    mkdir_code_b64 = base64.b64encode(mkdir_code.encode("utf-8")).decode("ascii")
    q_defs.append(
        (
            "00_create_remote_source_directory.json",
            {
                "op": "exec",
                "command": "\n".join(
                    [
                        "set -euo pipefail",
                        f"{REMOTE_PYTHON} -c 'import base64;exec(base64.b64decode(\"{mkdir_code_b64}\"))'",
                    ]
                ),
                "timeout": 60,
            },
        )
    )
    q_defs.append(
        (
            "01_upload_source_bundle.json",
            {
                "op": "upload",
                "local": str(BUNDLE.resolve()),
                "remote": f"{ROOT_REMOTE}/future_dev20_source_bundle_attempt02.zip",
                "expected_bytes": receipt["bundle"]["bytes"],
                "expected_sha256": receipt["bundle"]["sha256"],
                "timeout": 180,
            },
        )
    )
    q_defs.append(
        (
            "02_upload_portable_manifest.json",
            {
                "op": "upload",
                "local": str(MANIFEST.resolve()),
                "remote": f"{ROOT_REMOTE}/future_dev20_portable_source_manifest_attempt02.json",
                "expected_bytes": receipt["source_manifest"]["bytes"],
                "expected_sha256": receipt["source_manifest"]["sha256"],
                "timeout": 90,
            },
        )
    )
    worker_pin = raw_pin(WORKER)
    q_defs.append(
        (
            "03_upload_stage_qc_worker.json",
            {
                "op": "upload",
                "local": worker_pin["path"],
                "remote": f"{ROOT_REMOTE}/extract_and_qc_dev20_source_stage_attempt02.py",
                "expected_bytes": worker_pin["bytes"],
                "expected_sha256": worker_pin["sha256"],
                "timeout": 90,
            },
        )
    )
    remote_bundle = f"{ROOT_REMOTE}/future_dev20_source_bundle_attempt02.zip"
    remote_manifest = f"{ROOT_REMOTE}/future_dev20_portable_source_manifest_attempt02.json"
    remote_worker = f"{ROOT_REMOTE}/extract_and_qc_dev20_source_stage_attempt02.py"
    remote_stage = f"{ROOT_REMOTE}/source_stage_attempt01"
    shell_command = "\n".join(
        [
            "set -euo pipefail",
            f"test -x {REMOTE_PYTHON}",
            f"test -f {remote_bundle}",
            f"test -f {remote_manifest}",
            f"test -f {remote_worker}",
            f"test ! -e {remote_stage}",
            f"{REMOTE_PYTHON} {remote_worker} --bundle {remote_bundle} --manifest {remote_manifest} --destination {remote_stage}",
        ]
    )
    q_defs.append(
        (
            "04_extract_and_full_raw_f32_qc.json",
            {"op": "exec", "command": shell_command, "timeout": 300},
        )
    )

    request_pins: list[dict[str, Any]] = []
    for name, body in q_defs:
        q_path = PACKET / name
        request_raw = write_create_only(q_path, body)
        request_pins.append(
            {"name": name.removesuffix(".json"), "path": str(q_path.resolve()), "bytes": len(request_raw), "sha256": sha256(request_raw)}
        )

    evidence_paths = [
        BUNDLE,
        MANIFEST,
        BUILD_RECEIPT,
        DOWNLOAD_RECEIPT,
        SOURCE_PLAN,
        HEADER_RECEIPT,
        FIT_RECEIPT,
        COHORT_PLAN,
        README,
        AUDIO_IO,
        BUILDER,
        WORKER,
        Path(__file__).resolve(),
        *[Path(pin["path"]) for pin in request_pins],
    ]
    evidence: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(evidence_paths):
        key = f"source_{index:02d}_{path.name}"
        evidence[key] = raw_pin(path)
    for row in download["files"]:
        path = Path(row["target_path"])
        key = f"waveform_{row['sample_id']}"
        pin = raw_pin(path)
        if pin["bytes"] != row["expected_bytes"] or pin["sha256"] != row["expected_sha256"]:
            raise RuntimeError(f"actual source WAV differs from verified download receipt: {row['sample_id']}")
        evidence[key] = pin

    qc_protocol = {
        "purpose": "create-only staging of the future Dev20 natural-audio files and independent raw-F32 reproduction; no ASR, scores, model, training, or GPU",
        "source_cohort": "only the ten frozen future Dev speakers S0102-S0111, exactly two rows each",
        "source_integrity": "upload and worker check exact ZIP/manifest bytes and SHA; all 20 WAV member bytes/SHA are checked against the portable manifest and earlier pinned LFS metadata",
        "RIFF_gate": "parse full RIFF/WAVE; require PCM format code 1, mono, 16 kHz, 16 bits, block_align 2 and exact N from the local RIFF-header receipt",
        "canonical_reconstruction": "interpret each little-endian signed int16 sample, divide by 32768.0 in Python binary64, round/pack to little-endian IEEE float32, preserve exact N and 16 kHz clock; stream bytes are SHA-256 checked against the local audio_io canonical readback hashes",
        "expected_create_only_outputs_before_qc_receipt": {
            "source_wav": 20,
            "canonical_float32": 20,
            "evidence_files": 7,
            "source_manifest": 1,
            "total": 48,
        },
        "create_only": f"destination {remote_stage} must not exist; every source WAV, canonical .f32, evidence file, manifest, and QC receipt is opened with exclusive create; partial output is preserved on failure and no retry is preauthorized",
        "future_receipt": {
            "expected_relative_path": "source_stage_qc_attempt01.json",
            "bytes": None,
            "sha256": None,
            "rule": "obtain bytes/SHA from the actual successful exec response and later raw readback; never prefill a predicted receipt pin",
        },
    }
    index: dict[str, Any] = {
        "schema": "aishell_future_dev20_source_upload_packet_attempt04_v1",
        "status": "PREPARED_ONLY_NOT_DISPATCHED",
        "remote_root": ROOT_REMOTE,
        "remote_stage_root": remote_stage,
        "remote_python_absolute": REMOTE_PYTHON,
        "q00_readonly_diagnostic_before_create": {
            "checked_paths": ["/root/autodl-tmp/envs/ditto/bin/python", REMOTE_PYTHON, remote_parent, ROOT_REMOTE],
            "checks": ["old interpreter exists/is_file/executable", "selected interpreter exists/is_file/executable", "research namespace parent exists/is_dir/writable+searchable", "remote root absent and not a symlink"],
            "behavior": "emit one JSON object; mutate only by exclusive mkdir after the selected interpreter and parent checks pass and after confirming the destination is absent",
        },
        "source_evidence": evidence,
        "requests": request_pins,
        "qc_protocol": qc_protocol,
        "asr_or_scores_run": False,
        "model_or_gpu_run": False,
        "remote_upload_or_exec_dispatched": False,
    }
    index["self_sha256"] = compact_sha(index)
    raw = write_create_only(INDEX, index)
    print(
        json.dumps(
            {
                "status": index["status"],
                "index_path": str(INDEX.resolve()),
                "index_bytes": len(raw),
                "index_sha256": sha256(raw),
                "request_count": len(request_pins),
                "source_evidence_count": len(evidence),
                "zip_sha256": receipt["bundle"]["sha256"],
                "manifest_sha256": receipt["source_manifest"]["sha256"],
                "remote_dispatch": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
