#!/usr/bin/env bash
# run_all.sh - Sequential pipeline runner (issue #1)
# Usage:
#   ./run_all.sh             # full run, 10 samples (Q20)
#   ./run_all.sh --smoke     # smoke test, 1 sample (Q25)
# Per Q11 of plan: fail-fast-free (failures recorded, run continues).
# Per Q21: each step is independently rerunnable via --run_id reuse.

set -uo pipefail

SMOKE_FLAG=""
if [ "${1:-}" = "--smoke" ]; then
  SMOKE_FLAG="--smoke"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_ID

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

RUN_DIR="$REPO_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "[run_all] RUN_ID=$RUN_ID"
echo "[run_all] RUN_DIR=$RUN_DIR"
echo "[run_all] smoke=$SMOKE_FLAG"

# Snapshot config.yaml + git commit for reproducibility (Q22)
cp scripts/config.yaml "$RUN_DIR/config.yaml"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || echo "unknown" > "$RUN_DIR/git_commit.txt"

STEPS=("00_datacheck.py" "01_asr.py" "02_tts.py" "03_ditto.py" "04_eval.py" "05_report.py")
FAILS=()

for step in "${STEPS[@]}"; do
  echo ""
  echo "===== $step$SMOKE_FLAG ====="
  if python "scripts/$step" --run_id "$RUN_ID" $SMOKE_FLAG; then
    echo "[ok] $step"
  else
    rc=$?
    echo "[fail] $step (exit $rc)"
    FAILS+=("$step($rc)")
  fi
done

echo ""
echo "============================================================"
if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "[run_all] ALL OK: $RUN_DIR"
  exit 0
else
  echo "[run_all] FAILURES: ${FAILS[*]}"
  echo "[run_all] artifacts at: $RUN_DIR"
  exit 1
fi