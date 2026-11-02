#!/usr/bin/env python3
"""Prepare a fresh bounded-chunk readback for the frozen ROI geometry probe."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".codex/scratch/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05"
PACKET = SCRATCH / "packet_attempt05"
READBACK = ROOT / "runs/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05_readback"
COLLECTOR = ROOT / "scripts/audio_enhancement_eval/collect_full512_roi_geometry_attempt04.py"
SPLITTER = ROOT / "scripts/audio_enhancement_eval/split_roi_geometry_payload_attempt01.py"
REASSEMBLER = ROOT / "scripts/audio_enhancement_eval/reassemble_roi_geometry_payload_attempt01.py"
REFERENCE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/export_engineering_context_attempt01.py"

REMOTE_CODE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05/code"
REMOTE_JOB = "/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05"
REMOTE_OUT = "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt05/output"
REMOTE_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"

SUMMARY_SHA = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
CONTEXT_NPZ_SHA = "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1"
CONTEXT_JSON_SHA = "46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450"
PIPELINE_SHA = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
REFERENCE_SHA = "e487d137f268d76f9dd68a1d27bdc1bb6d56fd8c9d760416c21596f5eecbe983"
MAX_PART_BYTES = 6_500_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> tuple[int, str]:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return len(data), hashlib.sha256(data).hexdigest()


def local(path: Path) -> str:
    return path.resolve().as_posix()


def request(role: str, body: dict[str, Any]) -> dict[str, Any]:
    path = PACKET / f"{role}.json"
    size, digest = write_json(path, body)
    return {"role": role, "path": local(path), "bytes": size, "sha256": digest, "request": body}


def main() -> int:
    if PACKET.exists():
        raise FileExistsError(f"refusing to overwrite fresh packet directory: {PACKET}")
    if READBACK.exists():
        raise FileExistsError(f"refusing to overwrite fresh readback directory: {READBACK}")
    if sha(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("frozen crop formula reference changed")
    if sha(SPLITTER) == "" or sha(REASSEMBLER) == "":
        raise RuntimeError("missing transport helper source")
    collector_sha = sha(COLLECTOR)
    collector_bytes = COLLECTOR.stat().st_size
    splitter_sha = sha(SPLITTER)
    splitter_bytes = SPLITTER.stat().st_size
    reassembler_sha = sha(REASSEMBLER)
    reassembler_bytes = REASSEMBLER.stat().st_size

    PACKET.mkdir(parents=True, exist_ok=False)
    READBACK.mkdir(parents=True, exist_ok=False)
    (READBACK / "parts").mkdir(exist_ok=False)

    namespace = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
for path in "$CODE" "$JOB" "$OUT"; do test ! -e "$path"; done
mkdir -p "$CODE" "$JOB" "$(dirname "$OUT")"
test ! -e "$OUT"
printf '%s\\n' 'FRESH_NAMESPACE_CREATED_OUTPUT_ABSENT'"""
    preflight = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{REMOTE_PYTHON}'
test -x "$PY"
test -f "$CODE/collect_full512_roi_geometry_attempt04.py"
test -f "$CODE/export_engineering_context_attempt01.py"
test -f "$CODE/split_roi_geometry_payload_attempt01.py"
test ! -e "$OUT"
test ! -e "$JOB/collector.stdout.log"
test ! -e "$JOB/collector.stderr.log"
test ! -e "$JOB/splitter.stdout.log"
test ! -e "$JOB/splitter.stderr.log"
test "$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json | awk '{{print $1}}')" = '{SUMMARY_SHA}'
test "$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.npz | awk '{{print $1}}')" = '{CONTEXT_NPZ_SHA}'
test "$(sha256sum /dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.json | awk '{{print $1}}')" = '{CONTEXT_JSON_SHA}'
test "$(sha256sum /root/autodl-tmp/wav2lip-syncnet/syncnet_python/run_pipeline.py | awk '{{print $1}}')" = '{PIPELINE_SHA}'
test "$(sha256sum "$CODE/collect_full512_roi_geometry_attempt04.py" | awk '{{print $1}}')" = '{collector_sha}'
test "$(sha256sum "$CODE/export_engineering_context_attempt01.py" | awk '{{print $1}}')" = '{REFERENCE_SHA}'
test "$(sha256sum "$CODE/split_roi_geometry_payload_attempt01.py" | awk '{{print $1}}')" = '{splitter_sha}'
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -c 'import ast,cv2,numpy as np; from pathlib import Path; [ast.parse(Path(p).read_text()) for p in ["{REMOTE_CODE}/collect_full512_roi_geometry_attempt04.py","{REMOTE_CODE}/split_roi_geometry_payload_attempt01.py"]]; print("PREFLIGHT_PASS_NO_MODEL_FORWARD numpy=%s cv2=%s"%(np.__version__,cv2.__version__))'"""
    run_collector = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{REMOTE_PYTHON}'
set +e
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$CODE/collect_full512_roi_geometry_attempt04.py" --output-dir "$OUT" >"$JOB/collector.stdout.log" 2>"$JOB/collector.stderr.log"
rc=$?
set -e
printf 'collector_exit_code=%s\\n' "$rc"
if [ -s "$JOB/collector.stdout.log" ]; then cat "$JOB/collector.stdout.log"; fi
if [ -s "$JOB/collector.stderr.log" ]; then cat "$JOB/collector.stderr.log"; fi
exit "$rc"
"""
    split_payload = f"""set -euo pipefail
CODE='{REMOTE_CODE}'
JOB='{REMOTE_JOB}'
OUT='{REMOTE_OUT}'
PY='{REMOTE_PYTHON}'
test -f "$OUT/roi_geometry_receipt.json"
test -s "$OUT/roi_geometry_receipt.json"
test ! -e "$OUT/roi_geometry_transfer_manifest.json"
set +e
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$CODE/split_roi_geometry_payload_attempt01.py" --output-dir "$OUT" --max-part-bytes {MAX_PART_BYTES} >"$JOB/splitter.stdout.log" 2>"$JOB/splitter.stderr.log"
rc=$?
set -e
printf 'splitter_exit_code=%s\\n' "$rc"
if [ -s "$JOB/splitter.stdout.log" ]; then cat "$JOB/splitter.stdout.log"; fi
if [ -s "$JOB/splitter.stderr.log" ]; then cat "$JOB/splitter.stderr.log"; fi
exit "$rc"
"""

    requests = [
        request("00_namespace", {"op": "exec", "timeout": 60, "command": namespace}),
        request("01_upload_collector", {
            "op": "upload", "local": local(COLLECTOR), "remote": f"{REMOTE_CODE}/{COLLECTOR.name}",
            "expected_bytes": collector_bytes, "expected_sha256": collector_sha,
        }),
        request("02_upload_crop_geometry_reference", {
            "op": "upload", "local": local(REFERENCE), "remote": f"{REMOTE_CODE}/{REFERENCE.name}",
            "expected_bytes": 23332, "expected_sha256": REFERENCE_SHA,
        }),
        request("03_upload_bounded_chunk_splitter", {
            "op": "upload", "local": local(SPLITTER), "remote": f"{REMOTE_CODE}/{SPLITTER.name}",
            "expected_bytes": splitter_bytes, "expected_sha256": splitter_sha,
        }),
        request("04_preflight", {"op": "exec", "timeout": 120, "command": preflight}),
        request("05_run_saved_frame_collector", {"op": "exec", "timeout": 900, "command": run_collector}),
        request("06_split_verified_npz_copy", {"op": "exec", "timeout": 180, "command": split_payload}),
        request("07_download_collector_receipt", {
            "op": "download", "remote": f"{REMOTE_OUT}/roi_geometry_receipt.json",
            "local": local(READBACK / "roi_geometry_receipt.json"),
        }),
        request("08_download_transfer_manifest", {
            "op": "download", "remote": f"{REMOTE_OUT}/roi_geometry_transfer_manifest.json",
            "local": local(READBACK / "roi_geometry_transfer_manifest.json"),
        }),
    ]
    for index in range(3):
        filename = f"roi_geometry_selected_frames.part{index:03d}"
        requests.append(request(f"{9 + index:02d}_download_npz_part_{index:03d}", {
            "op": "download", "remote": f"{REMOTE_OUT}/roi_geometry_transfer_parts/{filename}",
            "local": local(READBACK / "parts" / filename),
        }))

    source_files = [
        {"role": "collector", "local": local(COLLECTOR), "remote": f"{REMOTE_CODE}/{COLLECTOR.name}",
         "bytes": collector_bytes, "sha256": collector_sha},
        {"role": "crop_geometry_reference", "local": local(REFERENCE), "remote": f"{REMOTE_CODE}/{REFERENCE.name}",
         "bytes": 23332, "sha256": REFERENCE_SHA},
        {"role": "bounded_chunk_splitter", "local": local(SPLITTER), "remote": f"{REMOTE_CODE}/{SPLITTER.name}",
         "bytes": splitter_bytes, "sha256": splitter_sha},
        {"role": "local_reassembler", "local": local(REASSEMBLER), "remote": None,
         "bytes": reassembler_bytes, "sha256": reassembler_sha},
        {"role": "parity_summary", "local": "not_copied; pinned remote input",
         "remote": "/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/output/parity_summary.json",
         "bytes": 171086, "sha256": SUMMARY_SHA},
        {"role": "fit_context_npz", "local": "not_copied; pinned remote input",
         "remote": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.npz",
         "bytes": 15596178, "sha256": CONTEXT_NPZ_SHA},
        {"role": "fit_context_json", "local": "not_copied; pinned remote input",
         "remote": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.json",
         "bytes": 11218, "sha256": CONTEXT_JSON_SHA},
        {"role": "official_run_pipeline", "local": "server source; not copied",
         "remote": "/root/autodl-tmp/wav2lip-syncnet/syncnet_python/run_pipeline.py",
         "bytes": None, "sha256": PIPELINE_SHA},
    ]
    index = {
        "schema": "cross_tfg_full512_roi_geometry_chunked_readback_packet_v2",
        "status": "PREPARED_FOR_ROOT_REVIEW_NOT_DISPATCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope": "Rerun the exact read-only saved-frame extraction from the same frozen historical fit inputs under fresh namespace04, then copy the verified 15.6MB NPZ into <=6.5MB chunks for bridge transfer. No model import/forward, no SyncNet scoring, no metric recomputation, no dev/confirmation input.",
        "scientific_source_pins": {
            "parity_summary_raw_sha256": SUMMARY_SHA,
            "parity_summary_status": "COMPLETE",
            "fit_context_npz_raw_sha256": CONTEXT_NPZ_SHA,
            "fit_context_metadata_raw_sha256": CONTEXT_JSON_SHA,
            "run_pipeline_sha256": PIPELINE_SHA,
            "crop_formula_reference_sha256": REFERENCE_SHA,
            "collector_sha256": collector_sha,
            "splitter_sha256": splitter_sha,
            "reassembler_sha256": reassembler_sha,
            "source_samples": ["S0052_BAC009S0052W0121", "S0055_BAC009S0055W0122"],
            "frame_offsets_from_q0": [0, 8, 16, 24, 31],
        },
        "remote_paths": {"code": REMOTE_CODE, "job": REMOTE_JOB, "output": REMOTE_OUT,
                         "python": REMOTE_PYTHON},
        "local_paths": {"readback": local(READBACK), "parts": local(READBACK / "parts"),
                         "restored_npz": local(READBACK / "roi_geometry_selected_frames.npz"),
                         "restore_receipt": local(READBACK / "local_restore_receipt.json")},
        "source_files": source_files,
        "requests": requests,
        "gates": [
            "dispatch 00-05 only after exact source/collector review; output must be absent at collection start",
            "collector04 removes only the preferred-size exception; all inputs and image/crop checks remain unchanged",
            "dispatch 06 only after collector exit 0 and PASS_READ_ONLY_SAVED_FRAMES receipt; it copies the sealed NPZ without altering it",
            "dispatch 07-11 only after splitter exit 0 and transfer manifest reports exactly three contiguous parts each <= 6,500,000 bytes",
            "run local reassembler only after all 5 downloads and verify collector raw SHA, canonical manifest, every part SHA/size, and whole NPZ SHA/size",
            "attempt03's >7MB failure/output remains preserved as historical evidence; this packet uses fresh namespace04",
        ],
        "failed_attempt03_reference": {
            "response": local(ROOT / ".codex/scratch/cross_tfg_audio_head_20261003/proxy_native_full512_roi_geometry_attempt03/packet_attempt03/04_run_read_only_collector.json.response.json"),
            "reason": "collector wrote the 15,603,354-byte NPZ then exited on an unnecessary preferred-7MB transfer cap; preserve, do not reinterpret as geometry failure",
        },
    }
    index_path = PACKET / "request_index_attempt05.json"
    index_bytes, index_sha = write_json(index_path, index)
    print(json.dumps({"status": index["status"], "packet": local(PACKET), "request_count": len(requests),
                      "collector_sha256": collector_sha, "collector_bytes": collector_bytes,
                      "splitter_sha256": splitter_sha, "splitter_bytes": splitter_bytes,
                      "reassembler_sha256": reassembler_sha, "reassembler_bytes": reassembler_bytes,
                      "index_bytes": index_bytes, "index_sha256": index_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
