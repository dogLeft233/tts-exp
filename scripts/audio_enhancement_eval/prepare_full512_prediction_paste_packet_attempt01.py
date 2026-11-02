#!/usr/bin/env python3
"""Prepare a no-forward remote readback for saved Wav2Lip face-patch pixels."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".codex/scratch/cross_tfg_audio_head_20261003/proxy_native_full512_prediction_paste_attempt01"
PACKET = SCRATCH / "packet_attempt01"
READBACK = ROOT / "runs/cross_tfg_audio_head_20261003/proxy_native_full512_prediction_paste_attempt01_readback"
SOURCE = ROOT / "scripts/audio_enhancement_eval/verify_full512_prediction_paste_attempt01.py"
REMOTE_CODE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/proxy_native_full512_prediction_paste_attempt01/code"
REMOTE_JOB = "/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/proxy_native_full512_prediction_paste_attempt01"
REMOTE_OUT = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_prediction_paste_attempt01/output"
PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"
SUMMARY = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json"
SUMMARY_SHA = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
ROI_RECEIPT = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05/output/roi_geometry_receipt.json"
ROI_RECEIPT_SHA = "c5be31bfaa2f3fe5e1b937a63720292db85858399d91d39bc4ddbe79d5688523"
ROI_NPZ = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05/output/roi_geometry_selected_frames.npz"
ROI_NPZ_SHA = "6c14ebce3c2704f34aae39e82a3c345988b83e8af78577d921bc51dbf7a960ea"
ROI_NPZ_BYTES = 15603354
W2L_INFERENCE = "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py"
W2L_INFERENCE_SHA = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> tuple[int, str]:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def local(path: Path) -> str:
    return path.resolve().as_posix()


def add_request(packet: Path, role: str, body: dict[str, Any]) -> dict[str, Any]:
    path = packet / f"{role}.json"
    size, digest = write_json(path, body)
    return {"role": role, "path": local(path), "bytes": size, "sha256": digest, "request": body}


def main() -> int:
    if PACKET.exists() or READBACK.exists():
        raise FileExistsError("refusing to reuse a prediction-paste namespace or local readback")
    source_sha = sha(SOURCE)
    source_bytes = SOURCE.stat().st_size
    PACKET.mkdir(parents=True, exist_ok=False)
    READBACK.mkdir(parents=True, exist_ok=False)

    namespace = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
for p in "$CODE" "$JOB" "$OUT"; do test ! -e "$p"; done
mkdir -p "$CODE" "$JOB" "$(dirname "$OUT")"
test ! -e "$OUT"
printf '%s\\n' 'FRESH_SAVED_FRAME_PASTE_PROBE_NAMESPACE_CREATED'"""
    preflight = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{PYTHON}'
test -x "$PY"
test ! -e "$OUT"
test ! -e "$JOB/probe.stdout.log"
test ! -e "$JOB/probe.stderr.log"
test "$(sha256sum '{SUMMARY}' | awk '{{print $1}}')" = '{SUMMARY_SHA}'
test "$(sha256sum '{ROI_RECEIPT}' | awk '{{print $1}}')" = '{ROI_RECEIPT_SHA}'
test "$(sha256sum '{ROI_NPZ}' | awk '{{print $1}}')" = '{ROI_NPZ_SHA}'
test "$(stat -c %s '{ROI_NPZ}')" = '{ROI_NPZ_BYTES}'
test "$(sha256sum '{W2L_INFERENCE}' | awk '{{print $1}}')" = '{W2L_INFERENCE_SHA}'
test "$(sha256sum "$CODE/{SOURCE.name}" | awk '{{print $1}}')" = '{source_sha}'
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -c 'import ast,cv2,numpy as np; from pathlib import Path; ast.parse(Path("{REMOTE_CODE}/{SOURCE.name}").read_text()); print("PREFLIGHT_PASS_READ_ONLY numpy=%s cv2=%s"%(np.__version__,cv2.__version__))'"""
    run = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{PYTHON}'
set +e
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$CODE/{SOURCE.name}" --output-dir "$OUT" >"$JOB/probe.stdout.log" 2>"$JOB/probe.stderr.log"
rc=$?
set -e
printf 'probe_exit_code=%s\\n' "$rc"
if [ -s "$JOB/probe.stdout.log" ]; then cat "$JOB/probe.stdout.log"; fi
if [ -s "$JOB/probe.stderr.log" ]; then cat "$JOB/probe.stderr.log"; fi
exit "$rc"
"""
    requests = [
        add_request(PACKET, "00_namespace", {"op": "exec", "timeout": 60, "command": namespace}),
        add_request(PACKET, "01_upload_probe_source", {"op": "upload", "local": local(SOURCE),
            "remote": f"{REMOTE_CODE}/{SOURCE.name}", "expected_bytes": source_bytes, "expected_sha256": source_sha}),
        add_request(PACKET, "02_preflight", {"op": "exec", "timeout": 120, "command": preflight}),
        add_request(PACKET, "03_run_saved_pixel_probe", {"op": "exec", "timeout": 180,
            "command": run}),
        add_request(PACKET, "04_download_probe_receipt", {"op": "download",
            "remote": f"{REMOTE_OUT}/prediction_paste_probe_receipt.json",
            "local": local(READBACK / "prediction_paste_probe_receipt.json")}),
    ]
    index = {
        "schema": "cross_tfg_saved_prediction_paste_probe_packet_v1",
        "status": "PREPARED_FOR_ROOT_REVIEW_NOT_DISPATCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope": "Compare ten already-saved 96px predictions against the same pixels in already-saved full512 frames. No model load/forward, video rendering, SyncNet scoring, or effects recomputation.",
        "source_pins": {
            "probe_source": {"path": local(SOURCE), "bytes": source_bytes, "sha256": source_sha},
            "parity_summary": {"path": SUMMARY, "bytes": 171086, "sha256": SUMMARY_SHA},
            "roi_collector_receipt": {"path": ROI_RECEIPT, "bytes": 73632, "sha256": ROI_RECEIPT_SHA},
            "roi_geometry_npz": {"path": ROI_NPZ, "bytes": ROI_NPZ_BYTES, "sha256": ROI_NPZ_SHA},
            "wav2lip_inference_source": {"path": W2L_INFERENCE, "sha256": W2L_INFERENCE_SHA},
        },
        "runtime": {"python": PYTHON, "opencv": "pinned server environment; actual version in preflight receipt"},
        "remote_paths": {"code": REMOTE_CODE, "job": REMOTE_JOB, "output": REMOTE_OUT},
        "requests": requests,
        "gates": [
            "dispatch 00-03 after all exact source paths and raw pins are reviewed",
            "source NPZ/receipt/summary/inference.py are SHA- and size-checked before and after the probe",
            "probe is limited to two historical fit samples and five saved points per sample",
            "cv2.resize of the pinned saved uint8 96px patch is compared with the same full512 face-box slice; channel-swapped pixels are diagnostic only",
            "dispatch receipt download only after run code 0 and PASS_READ_ONLY_SAVED_FRAME_PASTE_COMPARISON",
        ],
    }
    index_path = PACKET / "request_index_attempt01.json"
    index_size, index_sha = write_json(index_path, index)
    print(json.dumps({"status": index["status"], "packet": local(PACKET), "source_bytes": source_bytes,
        "source_sha256": source_sha, "request_count": len(requests), "index_bytes": index_size,
        "index_sha256": index_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
