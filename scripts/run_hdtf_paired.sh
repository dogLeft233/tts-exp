#!/usr/bin/env bash
# Run ASR -> Qwen3-TTS -> Ditto -> SyncNet for one prepared HDTF run.

set -euo pipefail

RUN_ID=""
CONFIG=""
SMOKE=0

while (($#)); do
  case "$1" in
    --run-id|--run_id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --config)
      CONFIG="${2:?missing value for --config}"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    -h|--help)
      printf '%s\n' "Usage: $0 --run-id RUN_ID --config CONFIG [--smoke]"
      exit 0
      ;;
    *)
      printf '[hdtf-run] unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$RUN_ID" ] || [ -z "$CONFIG" ]; then
  printf '%s\n' "--run-id and --config are required" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
DITTO_PYTHON="${DITTO_PYTHON:-/root/autodl-tmp/envs/ditto/bin/python}"
if [ ! -x "$DITTO_PYTHON" ]; then
  DITTO_PYTHON="$(command -v python3)"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDNN_LIB="/root/autodl-tmp/envs/ditto/opt/cudnn8/lib"
if [ -d "$CUDNN_LIB" ]; then
  export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
fi

SMOKE_ARGS=()
if [ "$SMOKE" -eq 1 ]; then
  SMOKE_ARGS=(--smoke)
fi

"$DITTO_PYTHON" scripts/01_asr.py \
  --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}" || exit $?
"$DITTO_PYTHON" scripts/02_tts.py \
  --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}" || exit $?
"$DITTO_PYTHON" scripts/03_ditto.py \
  --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}" || exit $?

EVAL_RC=0
if "$DITTO_PYTHON" scripts/04_eval.py \
  --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}"; then
  EVAL_RC=0
else
  EVAL_RC=$?
fi

"$DITTO_PYTHON" scripts/05_report.py --run_id "$RUN_ID" || exit $?
exit "$EVAL_RC"
