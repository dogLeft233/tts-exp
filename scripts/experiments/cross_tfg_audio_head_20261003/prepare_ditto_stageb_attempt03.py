#!/usr/bin/env python3
"""Build an immutable bundle and bridge requests for the fresh Ditto Stage-B run."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003"
SCRATCH = ROOT / ".codex/scratch/cross_tfg_ditto_stageb_attempt03"
PLAN_DIR = ROOT / "runs/cross_tfg_audio_head_20261003/ditto_stageb_attempt03/frozen_plan"
BUNDLE_STAGE = SCRATCH / "bundle_stage"
BUNDLE = SCRATCH / "ditto_stageb_attempt03_bundle.zip"
LAUNCHER = CODE_DIR / "launch_ditto_stageb_attempt03.sh"

REMOTE_PROJECT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003"
TASK = REMOTE_PROJECT + "/stageb_ditto_attempt03"
CTRL = REMOTE_PROJECT + "/job_control/stageb_ditto_attempt03_attempt01"
OUT = "/dev/shm/cross_tfg_audio_head_20261003/stageb_ditto_attempt03/run"
REMOTE_BUNDLE = TASK + "/ditto_stageb_attempt03_bundle.zip"
REMOTE_LAUNCHER = TASK + "/launch_ditto_stageb_attempt03.sh"

ORCHPY = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"
GENPY = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
W2L = "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip"
SYNC = "/root/autodl-tmp/wav2lip-syncnet/syncnet_python"
SYNC_CHECKPOINT = SYNC + "/data/syncnet_v2.model"
FACE = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png"
SOURCE_VIDEO = "/dev/shm/cross_tfg_audio_head_20261003/stage_a_plan_attempt01/source/static_portrait_reference_25fps.mp4"
DITTO_SOURCE = REMOTE_PROJECT + "/model_sources_attempt02/ditto-talkinghead"
MODEL_ROOT = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/ditto_pytorch"
CONFIG = MODEL_ROOT + "/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
IMPORT_REPORT = REMOTE_PROJECT + "/job_control/model_deploy_attempt13/ditto_streamsdk_import_probe_attempt12/ditto_streamsdk_import_probe_attempt12.json"
HEADER_REPORT = REMOTE_PROJECT + "/model_deploy_attempt13/code/python310_headers_attempt14/python310_headers_attempt14.json"
RUNTIME_EXTENSION = REMOTE_PROJECT + "/model_deploy_attempt13/code/import_probe_native_cache_attempt12/pyximport/lib.linux-x86_64-3.10/core/utils/blend/blend.cpython-310-x86_64-linux-gnu.so"
PORTRAIT_REPORT = REMOTE_PROJECT + "/job_control/model_deploy_attempt18/ditto_portrait_render_attempt03.json"

PLAN_RAW_SHA = "1fe71d82d59512e49d687690556ec55aedbfca0cad046c2ec1e54923254de987"
PLAN_PAYLOAD_SHA = "97e6c61dcaafc165a28084b8399bf48faa4f2a51498fb4e0f42eaf50a9a36cdf"
ROWS_RAW_SHA = "f2fa4c0b56b5a4e91e22ef3997bbf491065148027e88324e42c9ae25832e5635"
PACKET_RAW_SHA = "REPLACED_AFTER_PLAN"
SHIM_SHA = "848c7ddd5905eb262935b87197ac13c0f5ed1e7cb760ed091cf8b0e4fc8e1a67"
RUNNER_SHA = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
CORE_SHA = "30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc"
LINEAGE_SHA = "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"
EVALUATOR_SHA = "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b"
PORTRAIT_SHA = "866fc656d69caef7203bc2a3d3c61e12046353c4c3515722fd8ad73c9d16170c"
IMPORT_SHA = "2ba259045dd79c797f7c0e34403f600ab1d0accc40fa03ca2aa820c8c80f7806"
HEADER_SHA = "b997b9e98ed61be480788442237bf1008f4a5cef24e74dd55bdb71ea55b1f0cd"
HEADER_TREE_SHA = "e9dcde923150324ed29f3e8104d18f9ca61f61b7b32b07a6a1874e47c15a2a6e"
EXTENSION_SHA = "ce1524957cf0ef6d6f8065630c3caebb654485576867a2a91450779277451509"
CONFIG_SHA = "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146"
FACE_SHA = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
SYNC_SHA = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG_SHA = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
FFPROBE_SHA = "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
W2L_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
SYNC_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
W2L_CHECKPOINT_SHA = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"

FILES = (
    ("code/build_ditto_stageb_plan_attempt03.py", CODE_DIR / "build_ditto_stageb_plan_attempt03.py"),
    ("code/ditto_five_arg_shim_attempt03.py", CODE_DIR / "ditto_five_arg_shim_attempt03.py"),
    ("code/run_cross_frontend_dev_attempt03.py", CODE_DIR / "run_cross_frontend_dev_attempt03.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py", CODE_DIR / "run_stagea_dev_attempt02.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py", CODE_DIR / "stagea_pcm16_lineage.py"),
    ("code/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py", CODE_DIR / "stagea_syncnet_evaluator.py"),
    ("code/scripts/audio_enhancement/__init__.py", ROOT / "scripts/audio_enhancement/__init__.py"),
    ("code/scripts/audio_enhancement/audio_io.py", ROOT / "scripts/audio_enhancement/audio_io.py"),
    ("code/scripts/audio_enhancement/baselines.py", ROOT / "scripts/audio_enhancement/baselines.py"),
    ("code/scripts/audio_enhancement/heads.py", ROOT / "scripts/audio_enhancement/heads.py"),
    ("code/scripts/audio_enhancement/overdub.py", ROOT / "scripts/audio_enhancement/overdub.py"),
    ("code/scripts/audio_enhancement/pipeline.py", ROOT / "scripts/audio_enhancement/pipeline.py"),
    ("plan/plan.json", PLAN_DIR / "plan.json"),
    ("plan/planned_rows.jsonl", PLAN_DIR / "planned_rows.jsonl"),
    ("plan/packet_manifest.json", PLAN_DIR / "packet_manifest.json"),
)


LAUNCH_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail
NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003
TASK=@@TASK@@
CODE=$TASK/code
PLAN_DIR=$TASK/plan
CTRL=@@CTRL@@
OUT=@@OUT@@
BUNDLE=@@REMOTE_BUNDLE@@
REMOTE_LAUNCHER=@@REMOTE_LAUNCHER@@
UNPACK=$TASK/.unpack_attempt03
ORCHPY=@@ORCHPY@@
SYNCNET_PY=$ORCHPY
GENPY=@@GENPY@@
W2L=@@W2L@@
SYNC=@@SYNC@@
SYNC_CHECKPOINT=@@SYNC_CHECKPOINT@@
FACE=@@FACE@@
SOURCE_VIDEO=@@SOURCE_VIDEO@@
DITTO_SOURCE=@@DITTO_SOURCE@@
MODEL_ROOT=@@MODEL_ROOT@@
CONFIG=@@CONFIG@@
IMPORT_REPORT=@@IMPORT_REPORT@@
HEADER_REPORT=@@HEADER_REPORT@@
RUNTIME_EXTENSION=@@RUNTIME_EXTENSION@@
PORTRAIT_REPORT=@@PORTRAIT_REPORT@@
RENDERER=@@RENDERER@@
PLAN_SHA=@@PLAN_RAW_SHA@@
PLAN_PAYLOAD_SHA=@@PLAN_PAYLOAD_SHA@@
ROWS_SHA=@@ROWS_RAW_SHA@@
PACKET_SHA=@@PACKET_RAW_SHA@@
BUNDLE_SHA=@@BUNDLE_SHA@@
BUNDLE_MANIFEST_SHA=@@BUNDLE_MANIFEST_SHA@@
SHIM_SHA=@@SHIM_SHA@@
RUNNER_SHA=@@RUNNER_SHA@@
CORE_SHA=@@CORE_SHA@@
LINEAGE_SHA=@@LINEAGE_SHA@@
EVALUATOR_SHA=@@EVALUATOR_SHA@@
PORTRAIT_SHA=@@PORTRAIT_SHA@@
IMPORT_SHA=@@IMPORT_SHA@@
HEADER_SHA=@@HEADER_SHA@@
HEADER_TREE_SHA=@@HEADER_TREE_SHA@@
EXTENSION_SHA=@@EXTENSION_SHA@@
RENDERER_SHA=@@RENDERER_SHA@@
CONFIG_SHA=@@CONFIG_SHA@@
FACE_SHA=@@FACE_SHA@@
SYNC_SHA=@@SYNC_SHA@@
FFMPEG_SHA=@@FFMPEG_SHA@@
FFPROBE_SHA=@@FFPROBE_SHA@@
W2L_HEAD=@@W2L_HEAD@@
SYNC_HEAD=@@SYNC_HEAD@@
W2L_CHECKPOINT_SHA=@@W2L_CHECKPOINT_SHA@@

mkdir -p "$CTRL"
if [ -e "$CTRL/worker.pid" ] || [ -e "$CTRL/exitcode" ] || [ -e "$CTRL/started_utc.txt" ]; then
  printf '%s\n' CONTROL_ATTEMPT_ALREADY_USED > "$CTRL/launch_error.txt"
  exit 97
fi
printf '%s\n' "$$" > "$CTRL/worker.pid"
"$ORCHPY" -B -c 'import pathlib,sys; s=pathlib.Path("/proc",sys.argv[1],"stat").read_text(); print(s[s.rfind(")")+2:].split()[19])' "$$" > "$CTRL/worker_start_ticks.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/started_utc.txt"
finish() {
  rc=$?
  if [ ! -e "$CTRL/exitcode" ]; then printf '%s\n' "$rc" > "$CTRL/exitcode"; fi
  if [ ! -e "$CTRL/finished_utc.txt" ]; then date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"; fi
}
trap finish EXIT
fail() { code=$1; shift; printf '%s\n' "$*" > "$CTRL/launch_error.txt"; printf '%s\n' "$code" > "$CTRL/exitcode"; date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"; exit "$code"; }
sha() { sha256sum "$1" | awk '{print tolower($1)}'; }

test -f "$BUNDLE" || fail 91 BUNDLE_MISSING
test "$(sha "$BUNDLE")" = "$BUNDLE_SHA" || fail 92 BUNDLE_SHA_MISMATCH
unzip -tq "$BUNDLE" > "$CTRL/bundle_test.log" 2>&1 || fail 93 BUNDLE_ZIP_INTEGRITY_FAILED
test ! -e "$CODE" || fail 94 CODE_DIRECTORY_ALREADY_EXISTS
test ! -e "$PLAN_DIR" || fail 95 PLAN_DIRECTORY_ALREADY_EXISTS
test ! -e "$UNPACK" || fail 96 UNPACK_DIRECTORY_ALREADY_EXISTS
mkdir "$UNPACK"
unzip -q "$BUNDLE" -d "$UNPACK" || fail 93 BUNDLE_EXTRACTION_FAILED
test "$(sha "$UNPACK/bundle_manifest.json")" = "$BUNDLE_MANIFEST_SHA" || fail 92 BUNDLE_MANIFEST_SHA_MISMATCH
if ! "$ORCHPY" -B - "$UNPACK" "$BUNDLE_MANIFEST_SHA" <<'PYCODE'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); raw=(root/'bundle_manifest.json').read_bytes()
def sha(b): return hashlib.sha256(b).hexdigest()
m=json.loads(raw.decode('utf-8'))
assert sha(raw)==sys.argv[2] and m['status']=='SEALED_BUNDLE'
listed=set(m['members'])
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='bundle_manifest.json'}
assert actual==listed
for name,item in m['members'].items():
 p=root/name; data=p.read_bytes(); assert len(data)==item['bytes'] and sha(data)==item['sha256'],name
print('DITTO_BUNDLE_MEMBER_READBACK_PASS',len(listed))
PYCODE
then fail 93 BUNDLE_MEMBER_READBACK_FAILED; fi
mkdir -p "$(dirname "$CODE")" "$(dirname "$PLAN_DIR")"
mv "$UNPACK/code" "$CODE"
mv "$UNPACK/plan" "$PLAN_DIR"
test ! -e "$TASK/bundle_manifest_attempt03.json" || fail 96 BUNDLE_MANIFEST_RECEIPT_ALREADY_EXISTS
mv "$UNPACK/bundle_manifest.json" "$TASK/bundle_manifest_attempt03.json"
rmdir "$UNPACK"

test "$(sha "$PLAN_DIR/plan.json")" = "$PLAN_SHA" || fail 92 PLAN_RAW_SHA_MISMATCH
test "$(sha "$PLAN_DIR/planned_rows.jsonl")" = "$ROWS_SHA" || fail 92 ROWS_RAW_SHA_MISMATCH
test "$(sha "$PLAN_DIR/packet_manifest.json")" = "$PACKET_SHA" || fail 92 PACKET_RAW_SHA_MISMATCH
test "$(sha "$CODE/ditto_five_arg_shim_attempt03.py")" = "$SHIM_SHA" || fail 92 DITTO_SHIM_SHA_MISMATCH
test "$(sha "$CODE/run_cross_frontend_dev_attempt03.py")" = "$RUNNER_SHA" || fail 92 CROSS_FRONTEND_RUNNER_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py")" = "$CORE_SHA" || fail 92 STAGEA_CORE_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py")" = "$LINEAGE_SHA" || fail 92 PCM16_LINEAGE_SHA_MISMATCH
test "$(sha "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py")" = "$EVALUATOR_SHA" || fail 92 SYNCNET_EVALUATOR_SHA_MISMATCH
test "$(sha "$PORTRAIT_REPORT")" = "$PORTRAIT_SHA" || fail 96 DITTO_PORTRAIT_QUALIFICATION_CHANGED
test "$(sha "$RENDERER")" = "$RENDERER_SHA" || fail 96 DITTO_NATIVE_RENDERER_CHANGED
test "$(sha "$IMPORT_REPORT")" = "$IMPORT_SHA" || fail 96 DITTO_STREAMSDK_IMPORT_QUALIFICATION_CHANGED
test "$(sha "$HEADER_REPORT")" = "$HEADER_SHA" || fail 96 DITTO_PYTHON_HEADERS_QUALIFICATION_CHANGED
test "$(sha "$RUNTIME_EXTENSION")" = "$EXTENSION_SHA" || fail 96 DITTO_NATIVE_EXTENSION_CHANGED
test "$(sha "$SYNC_CHECKPOINT")" = "$SYNC_SHA" || fail 96 SYNCNET_CHECKPOINT_CHANGED
test "$(sha "$FACE")" = "$FACE_SHA" || fail 96 PRIMARY_FACE_CHANGED
test "$(sha /usr/bin/ffmpeg)" = "$FFMPEG_SHA" || fail 96 FFMPEG_CHANGED
test "$(sha /usr/bin/ffprobe)" = "$FFPROBE_SHA" || fail 96 FFPROBE_CHANGED
test "$(git -C "$W2L" rev-parse HEAD)" = "$W2L_HEAD" || fail 96 WAV2LIP_SOURCE_HEAD_CHANGED
test "$(git -C "$SYNC" rev-parse HEAD)" = "$SYNC_HEAD" || fail 96 SYNCNET_SOURCE_HEAD_CHANGED
test "$(sha "$W2L/checkpoints/wav2lip_gan.pth")" = "$W2L_CHECKPOINT_SHA" || fail 96 WAV2LIP_CHECKPOINT_CHANGED
test "$(git -C "$DITTO_SOURCE" rev-parse HEAD)" = "c3e47eee2e626500017a0556b470d6d4182f85e8" || fail 96 DITTO_SOURCE_COMMIT_CHANGED
test ! -e "$OUT" || fail 98 OUTPUT_ROOT_ALREADY_EXISTS

if ! "$ORCHPY" -B - "$PLAN_DIR" "$PORTRAIT_REPORT" "$PLAN_SHA" "$PLAN_PAYLOAD_SHA" "$ROWS_SHA" "$PORTRAIT_SHA" "$IMPORT_SHA" "$EXTENSION_SHA" "$CONFIG_SHA" "$FACE_SHA" <<'PYCODE'
import collections,hashlib,json,pathlib,re,sys
pd=pathlib.Path(sys.argv[1]); plan_raw=(pd/'plan.json').read_bytes(); rows_raw=(pd/'planned_rows.jsonl').read_bytes(); report_raw=pathlib.Path(sys.argv[2]).read_bytes()
sha=lambda b: hashlib.sha256(b).hexdigest()
canonical=lambda x: json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
plan=json.loads(plan_raw.decode('utf-8')); rows=[json.loads(x) for x in rows_raw.decode('utf-8').splitlines() if x]
report=json.loads(report_raw.decode('utf-8'))
expected_raw,expected_payload,expected_rows,expected_report,expected_import,expected_ext,expected_config,expected_face=sys.argv[3:]
assert sha(plan_raw)==expected_raw and sha(rows_raw)==expected_rows and sha(report_raw)==expected_report
assert plan['status']=='FROZEN' and sha(canonical({k:v for k,v in plan.items() if k!='plan_sha256'}))==plan['plan_sha256']==expected_payload
assert len(rows)==len(plan['rows'])==240 and [r['row_id'] for r in rows]==[r['row_id'] for r in plan['rows']]
assert all(r['split']=='dev' for r in rows) and not plan['confirm_speaker_ids'] and not plan['generation_seed_family']['confirm']
assert len({r['sample_id'] for r in rows})==20 and len({r['speaker_id'] for r in rows})==10
assert set(r['seed'] for r in rows)=={20261003,20261004,20261005}
assert dict(collections.Counter(r['method_id'] for r in rows))=={'identity':60,'rms_fixed':60,'rms_localcompress':60,'spectral_head_fit_v3_zero_lag_unitcos':60}
assert all(r['source_face']['file_sha256']==expected_face for r in rows)
assert plan['models'][0]['generator']['renderer_sha256']=='1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0'
assert report['status']=='PASS_ENGINEERING_ONLY' and report['profile']=='ditto_hubert_onnx_cpu_renderer_torch_cuda'
assert report['runtime']['ditto_streamsdk_import_report']['raw_sha256']==expected_import
assert report['runtime']['cython_cache']['blend_extension_sha256_after']==expected_ext
assert report['source_and_weights']['weights']['ditto_cfg/v0.4_hubert_cfg_pytorch.pkl']['sha256']==expected_config
assert report['face']['file_sha256']==expected_face and report['silent_video']['probe']['fps']=='25/1'
assert plan['models'][0]['model_assets']=={'source_files':report['source_and_weights']['source_files'],'weights':report['source_and_weights']['weights']}
for group in ('source_files','weights'):
 for item in report['source_and_weights'][group].values():
  p=pathlib.Path(item['path']); assert p.is_file() and p.stat().st_size==item['bytes']
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1<<20),b''):h.update(block)
  assert h.hexdigest()==item['sha256'],str(p)
assert len(report['source_and_weights']['weights'])==12
sha_values=[report['source_and_weights']['weights'][k]['sha256'] for k in report['source_and_weights']['weights']]
assert all(re.fullmatch('[0-9a-f]{64}',x) for x in sha_values)
print('DITTO_PLAN_AND_12_WEIGHT_READBACK_PASS rows=240 confirm=0')
PYCODE
then fail 90 PLAN_OR_PORTRAIT_ASSET_PREFLIGHT_FAILED; fi

GPU_WAIT_START=$(date +%s)
GPU_LAST_REPORT=0
while true; do
  GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)
  NOW=$(date +%s)
  if [[ "$GPU_FREE" =~ ^[0-9]+$ ]] && [ "$GPU_FREE" -ge 18000 ]; then
    printf 'status=PASS_HEADROOM\ngpu_free_mib=%s\nwait_seconds=%s\n' "$GPU_FREE" "$((NOW-GPU_WAIT_START))" > "$CTRL/gpu_launch_gate.txt"
    break
  fi
  if [ $((NOW-GPU_WAIT_START)) -ge 1800 ]; then
    printf 'status=BOUNDED_WAIT_EXPIRED\ngpu_free_mib=%s\nwait_seconds=%s\n' "${GPU_FREE:-unavailable}" "$((NOW-GPU_WAIT_START))" > "$CTRL/gpu_launch_gate.txt"
    fail 90 GPU_HEADROOM_WAIT_EXPIRED_BEFORE_STAGEA
  fi
  if [ $((NOW-GPU_LAST_REPORT)) -ge 30 ]; then
    printf 'status=WAITING_FOR_HEADROOM\ngpu_free_mib=%s\nwait_seconds=%s\n' "${GPU_FREE:-unavailable}" "$((NOW-GPU_WAIT_START))" > "$CTRL/gpu_launch_gate.txt"
    GPU_LAST_REPORT=$NOW
  fi
  sleep 15
done
SHM_FREE=$(df -B1 /dev/shm | awk 'NR==2 {print $4}')
test -n "$SHM_FREE" && [ "$SHM_FREE" -ge 3000000000 ] || fail 90 TMPFS_FREE_BELOW_3GB
printf 'shm_free_bytes=%s\n' "$SHM_FREE" >> "$CTRL/gpu_launch_gate.txt"

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
if ! "$ORCHPY" -B -c 'import torch,numpy,scipy,cv2,soundfile,scripts.audio_enhancement; from scripts.audio_enhancement.heads import SpectralResidualHead; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124"; print("orchestrator",torch.__version__,numpy.__version__,torch.cuda.is_available())' > "$CTRL/orchestrator_preflight.log" 2>&1; then fail 90 ORCHESTRATOR_RUNTIME_PREFLIGHT_FAILED; fi
if ! env PYTHONPATH="$NS/model_deploy_attempt13/code/overlays/setuptools_overlay_attempt12:$NS/model_deploy_attempt13/code/overlays/cython_overlay_attempt12:$NS/model_deploy_attempt13/code/overlays/filetype_overlay:$NS/model_deploy_attempt10/code/future_overlay_attempt02:$NS/model_deploy_attempt09/code/overlay_python310_attempt02:$DITTO_SOURCE" "$GENPY" -B -c 'import torch,numpy,cv2,onnxruntime; assert str(torch.__version__) == "2.5.1+cu124" and torch.cuda.is_available(); print("ditto_runtime",torch.__version__,onnxruntime.__version__,torch.cuda.get_device_name(0),torch.cuda.is_available())' > "$CTRL/ditto_runtime_preflight.log" 2>&1; then fail 90 DITTO_RUNTIME_PREFLIGHT_FAILED; fi
if ! "$SYNCNET_PY" -B -c 'import torch,numpy,scipy,cv2,python_speech_features; torch.set_num_threads(1); assert str(torch.__version__) == "2.5.1+cu124"; print("syncnet_runtime",torch.__version__,numpy.__version__,torch.get_num_threads())' > "$CTRL/syncnet_preflight.log" 2>&1; then fail 90 SYNCNET_RUNTIME_PREFLIGHT_FAILED; fi

set +e
"$ORCHPY" -B "$CODE/run_cross_frontend_dev_attempt03.py" \
  --stagea-core "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py" \
  --expected-stagea-core-sha256 "$CORE_SHA" \
  --plan "$PLAN_DIR/plan.json" --planned-rows "$PLAN_DIR/planned_rows.jsonl" \
  --expected-plan-raw-sha256 "$PLAN_SHA" --output-root "$OUT" --project-root "$CODE" \
  --wav2lip-python "$GENPY" --wav2lip-root "$W2L" \
  --generator-child "$CODE/ditto_five_arg_shim_attempt03.py" \
  --syncnet-root "$SYNC" --syncnet-python "$ORCHPY" --syncnet-checkpoint "$SYNC_CHECKPOINT" \
  --syncnet-evaluator "$CODE/scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py" \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe --ffmpeg-sha256 "$FFMPEG_SHA" \
  --pcm16-lineage-sha256 "$LINEAGE_SHA" \
  --generator-timeout 3600 --syncnet-timeout 1800 --ffmpeg-timeout 300 \
  > "$CTRL/stageb.stdout" 2> "$CTRL/stageb.stderr"
RUN_RC=$?
set -e
printf '%s\n' "$RUN_RC" > "$CTRL/stageb_exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/runner_finished_utc.txt"
printf '%s\n' "$RUN_RC" > "$CTRL/exitcode"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CTRL/finished_utc.txt"
trap - EXIT
exit "$RUN_RC"
'''


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_request(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = SCRATCH / name
    if path.exists():
        raise FileExistsError(path)
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha_bytes(raw)}


def main() -> None:
    if SCRATCH.exists() or BUNDLE_STAGE.exists() or BUNDLE.exists() or LAUNCHER.exists():
        raise FileExistsError("attempt03 packet target already exists; refusing overwrite")
    plan_raw = (PLAN_DIR / "plan.json").read_bytes()
    rows_raw = (PLAN_DIR / "planned_rows.jsonl").read_bytes()
    packet_raw = (PLAN_DIR / "packet_manifest.json").read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    assert sha_file(PLAN_DIR / "plan.json") == PLAN_RAW_SHA and plan["plan_sha256"] == PLAN_PAYLOAD_SHA
    assert sha_file(PLAN_DIR / "planned_rows.jsonl") == ROWS_RAW_SHA and len(plan["rows"]) == 240
    assert sha_file(PLAN_DIR / "packet_manifest.json") == sha_bytes(packet_raw)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    BUNDLE_STAGE.mkdir(parents=True, exist_ok=False)
    members: dict[str, dict[str, Any]] = {}
    for relative, source in FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = BUNDLE_STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        target.write_bytes(raw)
        members[relative] = {"bytes": len(raw), "sha256": sha_bytes(raw)}
    manifest = {
        "schema": "cross_tfg_ditto_stageb_bundle_v1", "status": "SEALED_BUNDLE",
        "run_id": plan["run_id"], "plan_raw_sha256": PLAN_RAW_SHA,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA, "planned_rows_raw_sha256": ROWS_RAW_SHA,
        "members": members,
    }
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (BUNDLE_STAGE / "bundle_manifest.json").write_bytes(manifest_raw)
    with zipfile.ZipFile(BUNDLE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in BUNDLE_STAGE.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(BUNDLE_STAGE).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 << 16)
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(BUNDLE, "r") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(members) | {"bundle_manifest.json"}
        for rel, expected in members.items():
            data = archive.read(rel)
            assert len(data) == expected["bytes"] and sha_bytes(data) == expected["sha256"], rel

    tokens = {
        "@@TASK@@": TASK, "@@CTRL@@": CTRL, "@@OUT@@": OUT,
        "@@REMOTE_BUNDLE@@": REMOTE_BUNDLE, "@@REMOTE_LAUNCHER@@": REMOTE_LAUNCHER,
        "@@ORCHPY@@": ORCHPY, "@@GENPY@@": GENPY, "@@W2L@@": W2L, "@@SYNC@@": SYNC,
        "@@SYNC_CHECKPOINT@@": SYNC_CHECKPOINT, "@@FACE@@": FACE, "@@SOURCE_VIDEO@@": SOURCE_VIDEO,
        "@@DITTO_SOURCE@@": DITTO_SOURCE, "@@MODEL_ROOT@@": MODEL_ROOT, "@@CONFIG@@": CONFIG,
        "@@IMPORT_REPORT@@": IMPORT_REPORT, "@@HEADER_REPORT@@": HEADER_REPORT,
        "@@RUNTIME_EXTENSION@@": RUNTIME_EXTENSION, "@@PORTRAIT_REPORT@@": PORTRAIT_REPORT,
        "@@RENDERER@@": REMOTE_PROJECT + "/model_deploy_attempt18/code/render_silent_video_attempt07.py",
        "@@PLAN_RAW_SHA@@": PLAN_RAW_SHA, "@@PLAN_PAYLOAD_SHA@@": PLAN_PAYLOAD_SHA,
        "@@ROWS_RAW_SHA@@": ROWS_RAW_SHA, "@@PACKET_RAW_SHA@@": sha_bytes(packet_raw),
        "@@BUNDLE_SHA@@": sha_file(BUNDLE), "@@BUNDLE_MANIFEST_SHA@@": sha_bytes(manifest_raw),
        "@@SHIM_SHA@@": SHIM_SHA, "@@RUNNER_SHA@@": RUNNER_SHA, "@@CORE_SHA@@": CORE_SHA,
        "@@LINEAGE_SHA@@": LINEAGE_SHA, "@@EVALUATOR_SHA@@": EVALUATOR_SHA,
        "@@PORTRAIT_SHA@@": PORTRAIT_SHA, "@@IMPORT_SHA@@": IMPORT_SHA,
        "@@HEADER_SHA@@": HEADER_SHA, "@@HEADER_TREE_SHA@@": HEADER_TREE_SHA,
        "@@EXTENSION_SHA@@": EXTENSION_SHA,
        "@@RENDERER_SHA@@": "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0",
        "@@CONFIG_SHA@@": CONFIG_SHA,
        "@@FACE_SHA@@": FACE_SHA, "@@SYNC_SHA@@": SYNC_SHA,
        "@@FFMPEG_SHA@@": FFMPEG_SHA, "@@FFPROBE_SHA@@": FFPROBE_SHA,
        "@@W2L_HEAD@@": W2L_HEAD, "@@SYNC_HEAD@@": SYNC_HEAD,
        "@@W2L_CHECKPOINT_SHA@@": W2L_CHECKPOINT_SHA,
    }
    launcher = LAUNCH_TEMPLATE
    for key, value in tokens.items():
        launcher = launcher.replace(key, value)
    if "@@" in launcher:
        raise ValueError("unfilled launcher template token remains")
    LAUNCHER.write_text(launcher, encoding="utf-8", newline="\n")
    launcher_sha = sha_file(LAUNCHER)

    mkdir_cmd = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"OUT={OUT}",
        'mkdir -p "$TASK" "$CTRL"',
        'test ! -e "$TASK/code"', 'test ! -e "$TASK/plan"',
        'test ! -e "$TASK/ditto_stageb_attempt03_bundle.zip"',
        'test ! -e "$CTRL/worker.pid"', 'test ! -e "$OUT"',
        'printf "READY\\ntask=%s\\ncontrol=%s\\noutput=%s\\n" "$TASK" "$CTRL" "$OUT"',
    ])
    launch_cmd = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"OUT={OUT}", f"LAUNCHER={REMOTE_LAUNCHER}",
        f"EXPECTED_LAUNCHER_SHA={launcher_sha}", f"EXPECTED_BUNDLE_SHA={sha_file(BUNDLE)}",
        'test "$(sha256sum "$LAUNCHER" | awk \'{print tolower($1)}\')" = "$EXPECTED_LAUNCHER_SHA"',
        'test "$(sha256sum "$TASK/ditto_stageb_attempt03_bundle.zip" | awk \'{print tolower($1)}\')" = "$EXPECTED_BUNDLE_SHA"',
        'bash -n "$LAUNCHER"', 'test ! -e "$CTRL/worker.pid"', 'test ! -e "$OUT"',
        'nohup bash "$LAUNCHER" </dev/null > "$CTRL/bootstrap.stdout" 2> "$CTRL/bootstrap.stderr" &',
        'PID=$!', 'printf "%s\\n" "$PID" > "$CTRL/dispatch_pid.txt"',
        'printf "LAUNCH_REQUEST_DISPATCHED launcher_sha256=%s bundle_sha256=%s pid=%s\\n" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_BUNDLE_SHA" "$PID"',
    ])
    status_cmd = "\n".join([
        "set -euo pipefail", f"TASK={TASK}", f"CTRL={CTRL}", f"OUT={OUT}",
        'printf "task=%s\\ncontrol=%s\\noutput=%s\\n" "$TASK" "$CTRL" "$OUT"',
        'for key in worker.pid worker_start_ticks.txt dispatch_pid.txt started_utc.txt runner_finished_utc.txt finished_utc.txt exitcode stageb_exitcode gpu_launch_gate.txt ditto_gpu_gate_current.json; do printf "%s=" "$key"; cat "$CTRL/$key" 2>/dev/null || true; done',
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
        "mkdir": write_request("00_mkdir_attempt03_request.json", {"op": "exec", "timeout": 20, "command": mkdir_cmd}),
        "bundle_upload": write_request("01_upload_bundle_attempt03_request.json", {"op": "upload", "local": BUNDLE.resolve().as_posix(), "remote": REMOTE_BUNDLE}),
        "launcher_upload": write_request("02_upload_launcher_attempt03_request.json", {"op": "upload", "local": LAUNCHER.resolve().as_posix(), "remote": REMOTE_LAUNCHER}),
        "launch": write_request("03_launch_attempt03_request.json", {"op": "exec", "timeout": 120, "command": launch_cmd}),
        "status_template": write_request("04_status_template_attempt03_request.json", {"op": "exec", "timeout": 30, "command": status_cmd}),
    }
    index = {
        "schema": "cross_tfg_ditto_stageb_attempt03_dispatch_index_v1",
        "status": "READY_FOR_ROOT_REVIEW", "task": TASK, "control": CTRL, "output": OUT,
        "bundle": {"path": str(BUNDLE.resolve()), "bytes": BUNDLE.stat().st_size, "sha256": sha_file(BUNDLE),
                   "manifest_sha256": sha_bytes(manifest_raw), "member_count": len(members) + 1, "members": members},
        "launcher": {"path": str(LAUNCHER.resolve()), "bytes": LAUNCHER.stat().st_size, "sha256": launcher_sha},
        "plan": {"path": str((PLAN_DIR / "plan.json").resolve()), "raw_sha256": PLAN_RAW_SHA,
                 "payload_sha256": PLAN_PAYLOAD_SHA, "planned_rows_raw_sha256": ROWS_RAW_SHA,
                 "packet_raw_sha256": sha_bytes(packet_raw), "rows": 240, "confirm_rows": 0},
        "ditto_profile": {"portrait_report_raw_sha256": PORTRAIT_SHA, "renderer_sha256": "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0",
                          "model_assets_sha256": plan["models"][0]["model_sha256"], "weight_count": 12,
                          "import_report_sha256": IMPORT_SHA, "runtime_extension_sha256": EXTENSION_SHA},
        "gpu_gate": {"minimum_free_mib": 18000, "wait_timeout_seconds": 1800, "row_poll_seconds": 15,
                     "row_progress_seconds": 30, "control_events_path": CTRL + "/ditto_gpu_gate_events.jsonl"},
        "requests": requests,
    }
    index_path = SCRATCH / "request_index_attempt03.json"
    index_raw = (json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    index_path.write_bytes(index_raw)
    index["index_path"] = str(index_path.resolve())
    index["index_bytes"] = len(index_raw)
    index["index_sha256"] = sha_bytes(index_raw)
    print(json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
