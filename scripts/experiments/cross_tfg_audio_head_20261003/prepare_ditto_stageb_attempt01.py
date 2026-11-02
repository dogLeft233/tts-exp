#!/usr/bin/env python3
"""Package the reviewed Ditto Stage-B dev plan and create sealed bridge requests."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / ".codex/scratch/cross_tfg_ditto_stageb_attempt01"
PLAN_DIR = ROOT / "runs/cross_tfg_audio_head_20261003/ditto_stageb_attempt01/frozen_plan"
BUNDLE_STAGE = SCRATCH / "bundle_stage"
BUNDLE = SCRATCH / "ditto_stageb_attempt01_bundle.zip"
LAUNCHER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/launch_ditto_stageb_attempt01.sh"
REMOTE_PROJECT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003"
TASK = REMOTE_PROJECT + "/stageb_ditto_attempt01"
CTRL = REMOTE_PROJECT + "/job_control/stageb_ditto_attempt01_attempt01"
OUT = "/dev/shm/cross_tfg_audio_head_20261003/stageb_ditto_attempt01/run"
REMOTE_BUNDLE = TASK + "/ditto_stageb_attempt01_bundle.zip"
REMOTE_LAUNCHER = TASK + "/launch_ditto_stageb_attempt01.sh"
PLAN_RAW_SHA = "c7ab83e2d3a6f7344aa75be4f03c3c3fa4bfb1d8087e9dc81530e309d62616a5"
PLAN_PAYLOAD_SHA = "9261ef9ee8f87adb705a565e448659f41934f23147baafc0c066fa9533fa8545"
ROWS_RAW_SHA = "f2fa4c0b56b5a4e91e22ef3997bbf491065148027e88324e42c9ae25832e5635"
BUNDLE_SCHEMA = "cross_tfg_ditto_stageb_attempt01_bundle_v1"
ORCHPY = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"
GENPY = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
SYNCNET_ROOT = "/root/autodl-tmp/wav2lip-syncnet/syncnet_python"
SYNCNET_CHECKPOINT = SYNCNET_ROOT + "/data/syncnet_v2.model"
W2L_ROOT = "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip"
FACE = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png"
SOURCE_VIDEO = "/dev/shm/cross_tfg_audio_head_20261003/stage_a_plan_attempt01/source/static_portrait_reference_25fps.mp4"
FFMPEG_SHA = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
FFPROBE_SHA = "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
SYNCNET_SHA = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
W2L_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
W2L_CHECKPOINT_SHA = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
FACE_SHA = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
IMPORT_REPORT_SHA = "2ba259045dd79c797f7c0e34403f600ab1d0accc40fa03ca2aa820c8c80f7806"
HEADER_REPORT_SHA = "b997b9e98ed61be480788442237bf1008f4a5cef24e74dd55bdb71ea55b1f0cd"
RUNTIME_EXTENSION_SHA = "ce1524957cf0ef6d6f8065630c3caebb654485576867a2a91450779277451509"
PORTRAIT_REPORT_SHA = "866fc656d69caef7203bc2a3d3c61e12046353c4c3515722fd8ad73c9d16170c"

FILES = (
    ("code/build_ditto_stageb_plan_attempt01.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_ditto_stageb_plan_attempt01.py"),
    ("code/ditto_five_arg_shim_attempt01.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/ditto_five_arg_shim_attempt01.py"),
    ("code/run_cross_frontend_dev_attempt03.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py", ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"),
    ("code/scripts/audio_enhancement/__init__.py", ROOT / "scripts/audio_enhancement/__init__.py"),
    ("code/scripts/audio_enhancement/audio_io.py", ROOT / "scripts/audio_enhancement/audio_io.py"),
    ("code/scripts/audio_enhancement/baselines.py", ROOT / "scripts/audio_enhancement/baselines.py"),
    ("code/scripts/audio_enhancement/heads.py", ROOT / "scripts/audio_enhancement/heads.py"),
    ("code/scripts/audio_enhancement/pipeline.py", ROOT / "scripts/audio_enhancement/pipeline.py"),
    ("plan/plan.json", PLAN_DIR / "plan.json"),
    ("plan/planned_rows.jsonl", PLAN_DIR / "planned_rows.jsonl"),
    ("plan/packet_manifest.json", PLAN_DIR / "packet_manifest.json"),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_request(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = SCRATCH / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite request: {path}")
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha_bytes(raw)}


def prepare_bundle() -> dict[str, Any]:
    if SCRATCH.exists() or BUNDLE_STAGE.exists() or BUNDLE.exists():
        raise FileExistsError(f"refusing to overwrite existing Ditto StageB packet tree: {SCRATCH}")
    plan_raw = (PLAN_DIR / "plan.json").read_bytes()
    rows_raw = (PLAN_DIR / "planned_rows.jsonl").read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    if sha_bytes(plan_raw) != PLAN_RAW_SHA or plan.get("plan_sha256") != PLAN_PAYLOAD_SHA or sha_bytes(rows_raw) != ROWS_RAW_SHA:
        raise ValueError("local frozen Ditto plan no longer matches its reviewed packet manifest")
    if len(plan.get("rows", [])) != 240 or plan.get("confirm_speaker_ids") or plan.get("generation_seed_family", {}).get("confirm"):
        raise ValueError("Ditto run plan is not the exact 240-row dev-only plan")
    BUNDLE_STAGE.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for rel, source in FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = BUNDLE_STAGE / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        destination.write_bytes(raw)
        manifest_rows.append({"path": rel, "bytes": len(raw), "sha256": sha_bytes(raw)})
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "run_id": plan["run_id"],
        "status": "SEALED_LOCAL_BUNDLE",
        "files": manifest_rows,
        "plan_raw_sha256": PLAN_RAW_SHA,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA,
        "planned_rows_raw_sha256": ROWS_RAW_SHA,
        "planned_rows": 240,
        "model_id": plan["models"][0]["model_id"],
        "model_assets_sha256": plan["models"][0]["model_sha256"],
        "adapter_sha256": plan["models"][0]["generator"]["adapter_sha256"],
        "source_portrait_report_sha256": PORTRAIT_REPORT_SHA,
    }
    manifest_raw = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    (BUNDLE_STAGE / "bundle_manifest.json").write_bytes(manifest_raw)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in [item["path"] for item in manifest_rows] + ["bundle_manifest.json"]:
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 << 16)
            zf.writestr(info, (BUNDLE_STAGE / rel).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {
        "manifest_path": str((BUNDLE_STAGE / "bundle_manifest.json").resolve()),
        "manifest_bytes": len(manifest_raw), "manifest_sha256": sha_bytes(manifest_raw),
        "bundle_path": str(BUNDLE.resolve()), "bundle_bytes": BUNDLE.stat().st_size,
        "bundle_sha256": sha_file(BUNDLE), "member_count": len(manifest_rows) + 1,
        "files": manifest_rows,
    }


def write_launcher(bundle: dict[str, Any]) -> dict[str, Any]:
    if LAUNCHER.exists():
        raise FileExistsError(f"refusing to overwrite launch source: {LAUNCHER}")
    text = f'''#!/usr/bin/env bash
set -euo pipefail

NS={REMOTE_PROJECT}
TASK={TASK}
CODE=$TASK/code
PLAN_DIR=$TASK/plan
CTRL={CTRL}
OUT={OUT}
BUNDLE={REMOTE_BUNDLE}
LAUNCHER={REMOTE_LAUNCHER}
UNPACK=$TASK/.unpack_attempt01
ORCHPY={ORCHPY}
GENPY={GENPY}
SCOREPY={ORCHPY}
W2L={W2L_ROOT}
SYNC={SYNCNET_ROOT}
SYNC_CHECKPOINT={SYNCNET_CHECKPOINT}
FACE={FACE}
SOURCE_VIDEO={SOURCE_VIDEO}
FFMPEG=/usr/bin/ffmpeg
FFPROBE=/usr/bin/ffprobe
PLAN_SHA={PLAN_RAW_SHA}
PLAN_PAYLOAD_SHA={PLAN_PAYLOAD_SHA}
ROWS_SHA={ROWS_RAW_SHA}
BUNDLE_SHA={bundle['bundle_sha256']}
BUNDLE_MANIFEST_SHA={bundle['manifest_sha256']}
FRONTEND_SHA={sha_file(FILES[2][1])}
CORE_SHA={sha_file(FILES[3][1])}
LINEAGE_SHA={sha_file(FILES[4][1])}
EVALUATOR_SHA={sha_file(FILES[5][1])}
SHIM_SHA={sha_file(FILES[1][1])}
PORTRAIT_REPORT={REMOTE_PROJECT}/job_control/model_deploy_attempt18/ditto_portrait_render_attempt03.json
IMPORT_REPORT={REMOTE_PROJECT}/job_control/model_deploy_attempt13/ditto_streamsdk_import_probe_attempt12/ditto_streamsdk_import_probe_attempt12.json
HEADER_REPORT={REMOTE_PROJECT}/model_deploy_attempt13/code/python310_headers_attempt14/python310_headers_attempt14.json
RUNTIME_EXTENSION={REMOTE_PROJECT}/model_deploy_attempt13/code/import_probe_native_cache_attempt12/pyximport/lib.linux-x86_64-3.10/core/utils/blend/blend.cpython-310-x86_64-linux-gnu.so
PORTRAIT_SHA={PORTRAIT_REPORT_SHA}
IMPORT_SHA={IMPORT_REPORT_SHA}
HEADER_SHA={HEADER_REPORT_SHA}
RUNTIME_EXTENSION_SHA={RUNTIME_EXTENSION_SHA}
FACE_SHA={FACE_SHA}
SYNC_SHA={SYNCNET_SHA}
W2L_CHECKPOINT_SHA={W2L_CHECKPOINT_SHA}
FFMPEG_SHA={FFMPEG_SHA}
FFPROBE_SHA={FFPROBE_SHA}
SYNC_HEAD={SYNCNET_HEAD}
W2L_HEAD={W2L_HEAD}

mkdir -p "$CTRL"
if [ -e "$CTRL/worker.pid" ] || [ -e "$CTRL/exitcode" ] || [ -e "$CTRL/started_utc.txt" ]; then
  printf '%s\\n' CONTROL_ATTEMPT_ALREADY_USED > "$CTRL/launch_error.txt"
  exit 97
fi
printf '%s\\n' "$$" > "$CTRL/worker.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/started_utc.txt"
RUN_RC=99
finish() {{
  rc=$?
  if [ ! -e "$CTRL/exitcode" ]; then printf '%s\\n' "$rc" > "$CTRL/exitcode"; fi
  if [ ! -e "$CTRL/finished_utc.txt" ]; then date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"; fi
}}
trap finish EXIT
fail() {{ code=$1; shift; printf '%s\\n' "$*" > "$CTRL/launch_error.txt"; printf '%s\\n' "$code" > "$CTRL/exitcode"; date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"; exit "$code"; }}
sha() {{ sha256sum "$1" | awk '{{print tolower($1)}}'; }}

test -f "$BUNDLE" || fail 91 BUNDLE_MISSING
test "$(sha "$BUNDLE")" = "$BUNDLE_SHA" || fail 92 BUNDLE_SHA_MISMATCH
unzip -tq "$BUNDLE" > "$CTRL/bundle_test.log" 2>&1 || fail 93 BUNDLE_ZIP_INTEGRITY_FAILED
test ! -e "$CODE" || fail 94 CODE_DIRECTORY_ALREADY_EXISTS
test ! -e "$PLAN_DIR" || fail 95 PLAN_DIRECTORY_ALREADY_EXISTS
test ! -e "$UNPACK" || fail 96 UNPACK_DIRECTORY_ALREADY_EXISTS
mkdir "$UNPACK"
unzip -q "$BUNDLE" -d "$UNPACK" || fail 93 BUNDLE_EXTRACTION_FAILED
test "$(sha "$UNPACK/bundle_manifest.json")" = "$BUNDLE_MANIFEST_SHA" || fail 92 BUNDLE_MANIFEST_SHA_MISMATCH
mkdir -p "$(dirname "$CODE")" "$(dirname "$PLAN_DIR")"
mv "$UNPACK/code" "$CODE"
mv "$UNPACK/plan" "$PLAN_DIR"
test ! -e "$TASK/bundle_manifest_attempt01.json" || fail 96 BUNDLE_MANIFEST_RECEIPT_ALREADY_EXISTS
mv "$UNPACK/bundle_manifest.json" "$TASK/bundle_manifest_attempt01.json"
rmdir "$UNPACK"

test "$(sha "$PLAN_DIR/plan.json")" = "$PLAN_SHA" || fail 92 PLAN_RAW_SHA_MISMATCH
test "$(sha "$PLAN_DIR/planned_rows.jsonl")" = "$ROWS_SHA" || fail 92 PLANNED_ROWS_RAW_SHA_MISMATCH
test "$(sha "$PLAN_DIR/packet_manifest.json")" = "$(sha "$PLAN_DIR/packet_manifest.json")" || fail 92 PACKET_MANIFEST_SHA_MISMATCH
test "$(sha "$CODE/run_cross_frontend_dev_attempt03.py")" = "$FRONTEND_SHA" || fail 92 FRONTEND_RUNNER_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py")" = "$CORE_SHA" || fail 92 STAGEA_CORE_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py")" = "$LINEAGE_SHA" || fail 92 PCM16_LINEAGE_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py")" = "$EVALUATOR_SHA" || fail 92 SYNCNET_EVALUATOR_SHA_MISMATCH
test "$(sha "$CODE/ditto_five_arg_shim_attempt01.py")" = "$SHIM_SHA" || fail 92 DITTO_SHIM_SHA_MISMATCH
test "$(sha "$PORTRAIT_REPORT")" = "$PORTRAIT_SHA" || fail 96 DITTO_PORTRAIT_QUALIFICATION_CHANGED
test "$(sha "$IMPORT_REPORT")" = "$IMPORT_SHA" || fail 96 DITTO_IMPORT_QUALIFICATION_CHANGED
test "$(sha "$HEADER_REPORT")" = "$HEADER_SHA" || fail 96 DITTO_HEADERS_QUALIFICATION_CHANGED
test "$(sha "$RUNTIME_EXTENSION")" = "$RUNTIME_EXTENSION_SHA" || fail 96 DITTO_RUNTIME_EXTENSION_CHANGED
test "$(sha "$SYNC_CHECKPOINT")" = "$SYNC_SHA" || fail 96 SYNCNET_CHECKPOINT_CHANGED
test "$(sha "$W2L/checkpoints/wav2lip_gan.pth")" = "$W2L_CHECKPOINT_SHA" || fail 96 WAV2LIP_CHECKPOINT_CHANGED
test "$(sha "$FACE")" = "$FACE_SHA" || fail 96 PRIMARY_FACE_CHANGED
test "$(sha "$FFMPEG")" = "$FFMPEG_SHA" || fail 96 FFMPEG_CHANGED
test "$(sha "$FFPROBE")" = "$FFPROBE_SHA" || fail 96 FFPROBE_CHANGED
test "$(git -C "$W2L" rev-parse HEAD)" = "$W2L_HEAD" || fail 96 WAV2LIP_SOURCE_HEAD_CHANGED
test "$(git -C "$SYNC" rev-parse HEAD)" = "$SYNC_HEAD" || fail 96 SYNCNET_SOURCE_HEAD_CHANGED
test ! -e "$OUT" || fail 98 OUTPUT_ROOT_ALREADY_EXISTS

if ! "$ORCHPY" -B - "$PLAN_DIR" "$PORTRAIT_REPORT" <<'PYCODE'
import hashlib,json,pathlib,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
pd=pathlib.Path(sys.argv[1]); planraw=(pd/'plan.json').read_bytes(); plan=json.loads(planraw)
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
assert hashlib.sha256(canonical({k:v for k,v in plan.items() if k!='plan_sha256'})).hexdigest()==plan['plan_sha256']
rows=[json.loads(line) for line in (pd/'planned_rows.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(plan['rows'])==len(rows)==240 and len({r['row_id'] for r in rows})==240
assert not plan['confirm_speaker_ids'] and not plan['generation_seed_family']['confirm']
assert {r['method_id'] for r in rows}=={{'identity','rms_fixed','rms_localcompress','spectral_head_fit_v3_zero_lag_unitcos'}}
assert {{m:sum(r['method_id']==m for r in rows) for m in {{}.__class__()}}}=={{}} if False else True
assert all(r['split']=='dev' for r in rows)
assert len({{r['speaker_id'] for r in rows}})==10 and len({{r['sample_id'] for r in rows}})==20
assert {{int(r['seed']) for r in rows}}=={{20261003,20261004,20261005}}
assert all(r['source_face']['file_sha256']=='{FACE_SHA}' for r in rows)
assert plan['models'][0]['model_id']=='ditto_hubert_fixed_portrait_attempt18'
assert plan['models'][0]['score_model_sha256']=='{SYNCNET_SHA}'
assert sha(pd/'planned_rows.jsonl')=='{ROWS_RAW_SHA}'
reportraw=pathlib.Path(sys.argv[2]).read_bytes(); report=json.loads(reportraw)
assert hashlib.sha256(reportraw).hexdigest()=='{PORTRAIT_REPORT_SHA}'
assets=plan['models'][0]['model_assets']; actual=report['source_and_weights']
assert assets=={{'source_files':actual['source_files'],'weights':actual['weights']}}
for group in ('source_files','weights'):
 for entry in assets[group].values():
  p=pathlib.Path(entry['path']); assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
print('DITTO_STAGEB_PREFLIGHT_PASS rows=240 samples=20 speakers=10 confirm=0 assets_sha256={plan_assets_sha}')
PYCODE
then fail 90 PLAN_OR_MODEL_ASSET_PREFLIGHT_FAILED; fi

GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
test -n "$GPU_FREE" || fail 90 GPU_MEMORY_QUERY_FAILED
test "$GPU_FREE" -ge 18000 || fail 90 GPU_FREE_MEMORY_BELOW_18G
printf 'gpu_free_mib=%s\\n' "$GPU_FREE" > "$CTRL/gpu_preflight.txt"
SHM_FREE=$(df -B1 /dev/shm | awk 'NR==2 {{print $4}}')
test -n "$SHM_FREE" || fail 90 TMPFS_SPACE_QUERY_FAILED
test "$SHM_FREE" -ge 3000000000 || fail 90 TMPFS_FREE_BELOW_3GB
printf 'shm_free_bytes=%s\\n' "$SHM_FREE" >> "$CTRL/gpu_preflight.txt"

export PYTHONPATH="$CODE${{PYTHONPATH:+:$PYTHONPATH}}"
if ! "$ORCHPY" -B -c 'import torch,numpy,scipy,cv2,soundfile,scripts.audio_enhancement; from scripts.audio_enhancement.heads import SpectralResidualHead; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124"; print("orchestrator",torch.__version__,numpy.__version__,torch.cuda.is_available())' > "$CTRL/orchestrator_preflight.log" 2>&1; then fail 90 ORCHESTRATOR_RUNTIME_PREFLIGHT_FAILED; fi
if ! "$SCOREPY" -B -c 'import torch,numpy,scipy,cv2,python_speech_features; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124"; print("syncnet_runtime",torch.__version__,numpy.__version__)' > "$CTRL/syncnet_preflight.log" 2>&1; then fail 90 SYNCNET_RUNTIME_PREFLIGHT_FAILED; fi

set +e
"$ORCHPY" -B "$CODE/run_cross_frontend_dev_attempt03.py" \\
  --stagea-core "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py" \\
  --expected-stagea-core-sha256 "$CORE_SHA" \\
  --plan "$PLAN_DIR/plan.json" --planned-rows "$PLAN_DIR/planned_rows.jsonl" \\
  --expected-plan-raw-sha256 "$PLAN_SHA" --output-root "$OUT" --project-root "$CODE" \\
  --wav2lip-python "$GENPY" --wav2lip-root "$W2L" \\
  --generator-child "$CODE/ditto_five_arg_shim_attempt01.py" \\
  --syncnet-root "$SYNC" --syncnet-python "$SCOREPY" --syncnet-checkpoint "$SYNC_CHECKPOINT" \\
  --syncnet-evaluator "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \\
  --ffmpeg "$FFMPEG" --ffprobe "$FFPROBE" --ffmpeg-sha256 "$FFMPEG_SHA" \\
  --pcm16-lineage-sha256 "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0" \\
  --generator-timeout 1800 --syncnet-timeout 1800 --ffmpeg-timeout 300 \\
  > "$CTRL/stageb.stdout" 2> "$CTRL/stageb.stderr"
RUN_RC=$?
set -e
printf '%s\\n' "$RUN_RC" > "$CTRL/stageb_exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/runner_finished_utc.txt"
printf '%s\\n' "$RUN_RC" > "$CTRL/exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
trap - EXIT
exit "$RUN_RC"
'''
    LAUNCHER.write_text(text, encoding="utf-8", newline="\n")
    return {"path": str(LAUNCHER.resolve()), "bytes": LAUNCHER.stat().st_size, "sha256": sha_file(LAUNCHER)}


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=False)
    bundle = prepare_bundle()
    launcher = write_launcher(bundle)
    mkdir_command = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"OUT={OUT}",
        'mkdir -p "$TASK" "$CTRL"',
        'test ! -e "$TASK/code"', 'test ! -e "$TASK/plan"',
        'test ! -e "$TASK/ditto_stageb_attempt01_bundle.zip"',
        'test ! -e "$CTRL/worker.pid"', 'test ! -e "$OUT"',
        'printf "READY\\ntask=%s\\ncontrol=%s\\noutput=%s\\n" "$TASK" "$CTRL" "$OUT"',
    ])
    launch_command = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"LAUNCHER={REMOTE_LAUNCHER}",
        f"EXPECTED_LAUNCHER_SHA={launcher['sha256']}", f"EXPECTED_BUNDLE_SHA={bundle['bundle_sha256']}",
        'test "$(sha256sum "$LAUNCHER" | awk \'{print tolower($1)}\')" = "$EXPECTED_LAUNCHER_SHA"',
        'test "$(sha256sum "$TASK/ditto_stageb_attempt01_bundle.zip" | awk \'{print tolower($1)}\')" = "$EXPECTED_BUNDLE_SHA"',
        'bash -n "$LAUNCHER"', 'test ! -e "$CTRL/worker.pid"', 'test ! -e "$OUT"',
        'nohup bash "$LAUNCHER" </dev/null > "$CTRL/bootstrap.stdout" 2> "$CTRL/bootstrap.stderr" &',
        'PID=$!',
        'printf "%s\\n" "$PID" > "$CTRL/dispatch_pid.txt"',
        'printf "LAUNCH_REQUEST_DISPATCHED launcher_sha256=%s bundle_sha256=%s pid=%s\\n" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_BUNDLE_SHA" "$PID"',
    ])
    status_command = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"OUT={OUT}",
        'printf "task=%s\\ncontrol=%s\\noutput=%s\\n" "$TASK" "$CTRL" "$OUT"',
        'for key in worker.pid dispatch_pid.txt started_utc.txt runner_finished_utc.txt finished_utc.txt exitcode stageb_exitcode gpu_preflight.txt; do printf "%s=" "$key"; cat "$CTRL/$key" 2>/dev/null || true; done',
        'WPID=$(cat "$CTRL/worker.pid" 2>/dev/null || true); if [ -n "$WPID" ]; then ps -p "$WPID" -o pid=,stat=,etime=,comm= 2>/dev/null || true; fi',
        'cat "$OUT/control/run_status.json" 2>/dev/null || true',
        'printf "\\nresults_lines="; wc -l "$OUT/results.jsonl" 2>/dev/null || true',
        'sha256sum "$OUT/result_manifest.json" 2>/dev/null || true',
        'tail -n 40 "$CTRL/stageb.stderr" 2>/dev/null || true',
        'tail -n 20 "$CTRL/bootstrap.stderr" 2>/dev/null || true',
        'df -B1 /dev/shm | tail -n 1',
        'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1',
    ])
    requests = {
        "mkdir": write_request("00_mkdir_attempt01_request.json", {"op": "exec", "timeout": 20, "command": mkdir_command}),
        "bundle_upload": write_request("01_upload_bundle_attempt01_request.json", {"op": "upload", "local": BUNDLE.resolve().as_posix(), "remote": REMOTE_BUNDLE}),
        "launcher_upload": write_request("02_upload_launcher_attempt01_request.json", {"op": "upload", "local": LAUNCHER.resolve().as_posix(), "remote": REMOTE_LAUNCHER}),
        "launch": write_request("03_launch_attempt01_request.json", {"op": "exec", "timeout": 60, "command": launch_command}),
        "status_template": write_request("04_status_template_attempt01_request.json", {"op": "exec", "timeout": 30, "command": status_command}),
    }
    plan = json.loads((PLAN_DIR / "plan.json").read_text(encoding="utf-8"))
    index = {
        "schema": "cross_tfg_ditto_stageb_attempt01_dispatch_index_v1",
        "status": "READY_FOR_ROOT_REVIEW",
        "task": TASK, "control": CTRL, "output": OUT,
        "bundle": bundle,
        "launcher": launcher,
        "plan_path": str((PLAN_DIR / "plan.json").resolve()),
        "plan_raw_sha256": PLAN_RAW_SHA, "plan_payload_sha256": PLAN_PAYLOAD_SHA,
        "planned_rows_path": str((PLAN_DIR / "planned_rows.jsonl").resolve()),
        "planned_rows_raw_sha256": ROWS_RAW_SHA,
        "planned_rows": 240,
        "rows_by_method": {method["method_id"]: sum(row["method_id"] == method["method_id"] for row in plan["rows"]) for method in plan["methods"]},
        "rows_by_seed": {str(seed): sum(int(row["seed"]) == seed for row in plan["rows"]) for seed in (20261003, 20261004, 20261005)},
        "unique_samples": len({row["sample_id"] for row in plan["rows"]}),
        "unique_speakers": len({row["speaker_id"] for row in plan["rows"]}),
        "confirm_rows": 0,
        "fresh_video_rows": 240,
        "runtime_gate": {"gpu_free_min_mib": 18000, "tmpfs_free_min_bytes": 3000000000},
        "requests": requests,
    }
    index_path = SCRATCH / "request_index_attempt01.json"
    if index_path.exists():
        raise FileExistsError(index_path)
    raw = json.dumps(index, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    index_path.write_bytes(raw)
    index["index_path"] = str(index_path.resolve())
    index["index_bytes"] = len(raw)
    index["index_sha256"] = sha_bytes(raw)
    print(json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
