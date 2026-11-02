#!/usr/bin/env python3
"""Build a fresh, immutable request packet for the corrected ROI readback."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".codex/scratch/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03"
PACKET = SCRATCH / "packet_attempt03"
COLLECTOR = ROOT / "scripts/audio_enhancement_eval/collect_full512_roi_geometry_attempt03.py"
REFERENCE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/export_engineering_context_attempt01.py"

REMOTE_CODE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03/code"
REMOTE_JOB = "/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03"
REMOTE_OUT = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03/output"
REMOTE_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"

SUMMARY_SHA = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
CONTEXT_NPZ_SHA = "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1"
CONTEXT_JSON_SHA = "46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450"
PIPELINE_SHA = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
REFERENCE_SHA = "e487d137f268d76f9dd68a1d27bdc1bb6d56fd8c9d760416c21596f5eecbe983"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> tuple[int, str]:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def local(path: Path) -> str:
    return path.resolve().as_posix()


def request(role: str, body: dict[str, Any]) -> dict[str, Any]:
    path = PACKET / f"{role}.json"
    size, digest = write_json(path, body)
    return {"role": role, "path": local(path), "bytes": size, "sha256": digest, "request": body}


def main() -> int:
    PACKET.mkdir(parents=True, exist_ok=False)
    collector_sha = sha(COLLECTOR)
    collector_bytes = COLLECTOR.stat().st_size
    if sha(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("the crop formula reference changed; stop and re-pin explicitly")

    namespace = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
for path in \"$CODE\" \"$JOB\" \"$OUT\"; do test ! -e \"$path\"; done
mkdir -p \"$CODE\" \"$JOB\" \"$(dirname \"$OUT\")\"
test ! -e \"$OUT\"
printf '%s\\n' 'FRESH_NAMESPACE_CREATED_OUTPUT_ABSENT'"""
    preflight = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{REMOTE_PYTHON}'
test -x \"$PY\"
test -f \"$CODE/collect_full512_roi_geometry_attempt03.py\"
test -f \"$CODE/export_engineering_context_attempt01.py\"
test ! -e \"$OUT\"
test ! -e \"$JOB/collector.stdout.log\"
test ! -e \"$JOB/collector.stderr.log\"
test \"$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json | awk '{{print $1}}')\" = '{SUMMARY_SHA}'
test \"$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.npz | awk '{{print $1}}')\" = '{CONTEXT_NPZ_SHA}'
test \"$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.json | awk '{{print $1}}')\" = '{CONTEXT_JSON_SHA}'
test \"$(sha256sum /root/autodl-tmp/wav2lip-syncnet/syncnet_python/run_pipeline.py | awk '{{print $1}}')\" = '{PIPELINE_SHA}'
PYTHONDONTWRITEBYTECODE=1 \"$PY\" -B -c 'import cv2,numpy as np; print("PREFLIGHT_PASS_NO_MODEL_FORWARD numpy=%s cv2=%s"%(np.__version__,cv2.__version__))'"""
    run = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{REMOTE_PYTHON}'
set +e
PYTHONDONTWRITEBYTECODE=1 \"$PY\" -B \"$CODE/collect_full512_roi_geometry_attempt03.py\" --output-dir \"$OUT\" >\"$JOB/collector.stdout.log\" 2>\"$JOB/collector.stderr.log\"
rc=$?
set -e
printf 'collector_exit_code=%s\\n' \"$rc\"
if [ -s \"$JOB/collector.stdout.log\" ]; then cat \"$JOB/collector.stdout.log\"; fi
if [ -s \"$JOB/collector.stderr.log\" ]; then cat \"$JOB/collector.stderr.log\"; fi
exit \"$rc\""""

    requests = [
        request("00_namespace", {"op": "exec", "timeout": 60, "command": namespace}),
        request("01_upload_collector", {
            "op": "upload", "local": local(COLLECTOR), "remote": f"{REMOTE_CODE}/{COLLECTOR.name}",
            "expected_bytes": collector_bytes, "expected_sha256": collector_sha,
        }),
        request("02_upload_crop_geometry_reference", {
            "op": "upload", "local": local(REFERENCE),
            "remote": f"{REMOTE_CODE}/{REFERENCE.name}",
            "expected_bytes": 23332, "expected_sha256": REFERENCE_SHA,
        }),
        request("03_preflight", {"op": "exec", "timeout": 120, "command": preflight}),
        request("04_run_read_only_collector", {"op": "exec", "timeout": 900, "command": run}),
        request("05_download_receipt_after_pass", {
            "op": "download", "remote": f"{REMOTE_OUT}/roi_geometry_receipt.json",
            "local": local(ROOT / "runs/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03_readback/roi_geometry_receipt.json"),
        }),
        request("06_download_selected_images_after_pass", {
            "op": "download", "remote": f"{REMOTE_OUT}/roi_geometry_selected_frames.npz",
            "local": local(ROOT / "runs/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03_readback/roi_geometry_selected_frames.npz"),
        }),
    ]

    index = {
        "schema": "cross_tfg_full512_roi_geometry_readback_packet_v3",
        "status": "PREPARED_FOR_ROOT_REVIEW_NOT_DISPATCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope": "Read-only extraction from frozen full-512 fit parity artifacts: two historical fit samples, q0 plus offsets 0/8/16/24/31; exact source-clock lookup into the full 141-frame context export, exact run_pipeline constant-110 padding and processed-track crop; no model import/forward, no score recomputation, no dev/confirmation input.",
        "scientific_source_pins": {
            "parity_summary_raw_sha256": SUMMARY_SHA,
            "parity_summary_status": "COMPLETE",
            "fit_context_npz_raw_sha256": CONTEXT_NPZ_SHA,
            "fit_context_metadata_raw_sha256": CONTEXT_JSON_SHA,
            "run_pipeline_sha256": PIPELINE_SHA,
            "crop_formula_reference_sha256": REFERENCE_SHA,
            "collector_sha256": collector_sha,
            "sample_ids": ["S0052_BAC009S0052W0121", "S0055_BAC009S0055W0122"],
            "frame_offsets_from_q0": [0, 8, 16, 24, 31],
        },
        "remote_paths": {"code": REMOTE_CODE, "job": REMOTE_JOB, "output": REMOTE_OUT,
                         "python": REMOTE_PYTHON},
        "source_files": [
            {"role": "collector", "local": local(COLLECTOR), "remote": f"{REMOTE_CODE}/{COLLECTOR.name}",
             "bytes": collector_bytes, "sha256": collector_sha},
            {"role": "crop_geometry_reference", "local": local(REFERENCE), "remote": f"{REMOTE_CODE}/{REFERENCE.name}",
             "bytes": 23332, "sha256": REFERENCE_SHA},
            {"role": "parity_summary", "local": "runs/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03_readback_attempt01/parity_summary.json",
             "remote": "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json",
             "bytes": 171086, "sha256": SUMMARY_SHA},
            {"role": "fit_context_npz", "local": "not_downloaded_local", "remote": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.npz",
             "bytes": 15596178, "sha256": CONTEXT_NPZ_SHA},
            {"role": "fit_context_json", "local": "not_downloaded_local", "remote": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.json",
             "bytes": 11218, "sha256": CONTEXT_JSON_SHA},
            {"role": "official_run_pipeline", "local": "server_source_not_downloaded", "remote": "/root/autodl-tmp/wav2lip-syncnet/syncnet_python/run_pipeline.py",
             "bytes": None, "sha256": PIPELINE_SHA},
        ],
        "requests": requests,
        "gates": [
            "dispatch requests 00-04 only after source and exact official padding/crop review",
            "namespace leaves exact output directory absent because collector refuses pre-existing output",
            "dispatch downloads 05-06 only after code 0 and PASS_READ_ONLY_SAVED_FRAMES receipt",
            "input files rehashed before and after extraction; any mismatch fails closed",
            "no model forward, inference, SyncNet forward, or effects recomputation",
        ],
    }
    write_json(PACKET / "request_index_attempt03.json", index)
    print(json.dumps({"status": index["status"], "packet": local(PACKET),
                      "collector_sha256": collector_sha, "collector_bytes": collector_bytes,
                      "request_count": len(requests),
                      "index_sha256": sha(PACKET / "request_index_attempt03.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
