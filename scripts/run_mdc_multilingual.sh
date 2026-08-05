#!/usr/bin/env bash
# Run the legacy self-clone TTS -> Ditto -> SyncNet protocol for MDC languages.

set -uo pipefail

LANGUAGE_LIST="en,de,it,es,ko"
SMOKE=0
MANIFEST_PATH="data/mdc_tts/manifest.json"
SOURCE_LABEL="MDC"
RISK_LABEL=""
RUN_PREFIX="mdc"
TARGET_IDS=""
EXCLUDE_SOURCE_IDS=""
MIN_TEXT_CHARS_PER_S="0"
MAX_NEW_TOKENS=""

while (($#)); do
  case "$1" in
    --languages)
      LANGUAGE_LIST="${2:?missing value for --languages}"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --manifest)
      MANIFEST_PATH="${2:?missing value for --manifest}"
      shift 2
      ;;
    --source-label)
      SOURCE_LABEL="${2:?missing value for --source-label}"
      shift 2
      ;;
    --risk-label)
      RISK_LABEL="${2:?missing value for --risk-label}"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="${2:?missing value for --run-prefix}"
      shift 2
      ;;
    --target-ids)
      TARGET_IDS="${2:?missing value for --target-ids}"
      shift 2
      ;;
    --exclude-sample-ids)
      EXCLUDE_SOURCE_IDS="${2:?missing value for --exclude-sample-ids}"
      shift 2
      ;;
    --min-text-chars-per-s)
      MIN_TEXT_CHARS_PER_S="${2:?missing value for --min-text-chars-per-s}"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="${2:?missing value for --max-new-tokens}"
      shift 2
      ;;
    -h|--help)
      printf '%s\n' \
        "Usage: $0 [--languages ...] [--manifest path] [--smoke]"
      exit 0
      ;;
    *)
      printf '[mdc-run] unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

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

RUN_INDEX="$REPO_DIR/runs/mdc_run_ids.txt"
FAILED_INDEX="$REPO_DIR/runs/mdc_failed_runs.txt"
LAST_RUN_ID=""
mkdir -p "$REPO_DIR/runs"

run_language() {
  local language="$1"
  local sample_count=13
  if [ -n "$TARGET_IDS" ]; then
    local target_id_count
    IFS=',' read -r -a target_id_array <<< "$TARGET_IDS"
    target_id_count="${#target_id_array[@]}"
    sample_count="$target_id_count"
  fi
  if [ "$SMOKE" -eq 1 ]; then
    sample_count=1
  fi

   local run_id="${RUN_PREFIX}_${language}_$(date -u +%Y%m%dT%H%M%SZ)"
  LAST_RUN_ID="$run_id"
  local config="runs/$run_id/config_override.yaml"
  printf '[mdc-run] language=%s run_id=%s samples=%s python=%s\n' \
    "$language" "$run_id" "$sample_count" "$DITTO_PYTHON"

  "$DITTO_PYTHON" scripts/prepare_mdc_pairs.py \
     --run-id "$run_id" \
     --language "$language" \
     --manifest "$MANIFEST_PATH" \
     --sample-count "$sample_count" \
     --source-label "$SOURCE_LABEL" \
     --min-text-chars-per-s "$MIN_TEXT_CHARS_PER_S" \
     ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"} \
     ${RISK_LABEL:+--risk-label "$RISK_LABEL"} \
     ${TARGET_IDS:+--target-ids "$TARGET_IDS"} \
     ${EXCLUDE_SOURCE_IDS:+--exclude-sample-ids "$EXCLUDE_SOURCE_IDS"} \
     || return $?

  "$DITTO_PYTHON" scripts/00_datacheck.py \
    --run_id "$run_id" \
    --config "$config" \
    || return $?
  "$DITTO_PYTHON" scripts/02_tts.py \
    --run_id "$run_id" \
    --config "$config" \
    || return $?
  "$DITTO_PYTHON" scripts/03_ditto.py \
    --run_id "$run_id" \
    --config "$config" \
    || return $?
  local eval_rc=0
  if "$DITTO_PYTHON" scripts/04_eval.py \
    --run_id "$run_id" \
    --config "$config"; then
    eval_rc=0
  else
    eval_rc=$?
  fi
  "$DITTO_PYTHON" scripts/05_report.py \
    --run_id "$run_id" \
    || return $?

  if [ "$eval_rc" -ne 0 ]; then
    return "$eval_rc"
  fi
  printf '%s\n' "$run_id" >> "$RUN_INDEX"

  printf '[mdc-run] completed language=%s run_id=%s\n' "$language" "$run_id"
}

IFS=',' read -r -a LANGUAGES <<< "$LANGUAGE_LIST"
FAILURES=()
for language in "${LANGUAGES[@]}"; do
  if run_language "$language"; then
    continue
  else
    rc=$?
  fi
  printf '[mdc-run] failed language=%s rc=%s\n' "$language" "$rc" >&2
  FAILURES+=("$language:$rc")
  printf '%s\t%s\t%s\n' "$LAST_RUN_ID" "$language" "$rc" >> "$FAILED_INDEX"
  if [ "$SMOKE" -eq 1 ]; then
    exit "$rc"
  fi
done

if [ "${#FAILURES[@]}" -gt 0 ]; then
  printf '[mdc-run] failures: %s\n' "${FAILURES[*]}" >&2
  exit 1
fi

printf '[mdc-run] all requested languages complete\n'
