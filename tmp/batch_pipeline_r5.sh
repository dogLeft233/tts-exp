#!/bin/bash
# batch_pipeline_r5.sh — F5-TTS → Ditto → SyncNet
# Run on server :20398 (RTX 4080 SUPER)
# TTS audio already generated locally in runs/r5_f5_tts/02_tts/
set -euo pipefail

REPO=/root/autodl-tmp/repos/tts-exp
RUN_ID=r5_f5_tts
cd "$REPO"

echo "=== F5-TTS Pipeline at $(date) ==="
echo "Run ID: $RUN_ID"

# Check TTS files exist
TTS_COUNT=$(ls "$REPO/runs/$RUN_ID/02_tts/"*.wav 2>/dev/null | wc -l)
echo "TTS files: $TTS_COUNT"

if [ "$TTS_COUNT" -lt 13 ]; then
    echo "ERROR: Missing TTS files. Copy from local: runs/r5_f5_tts/02_tts/*.wav"
    exit 1
fi

# 03_ditto
if [ -f "$REPO/runs/$RUN_ID/03_ditto/ditto_meta.json" ]; then
    echo "[03_ditto] CACHED"
else
    echo "[03_ditto] starting ($(date '+%H:%M:%S'))..."
    python scripts/03_ditto.py --run_id "$RUN_ID" || { echo "DITTO FAILED"; exit 1; }
    echo "[03_ditto] done ($(date '+%H:%M:%S'))"
fi

# 04_eval
if [ -f "$REPO/runs/$RUN_ID/04_eval/eval_meta.json" ]; then
    echo "[04_eval] CACHED"
else
    echo "[04_eval] starting ($(date '+%H:%M:%S'))..."
    python scripts/04_eval.py --run_id "$RUN_ID" || { echo "SYNCNET FAILED"; exit 1; }
    echo "[04_eval] done ($(date '+%H:%M:%S'))"
fi

# 05_report
echo "[05_report]..."
python scripts/05_report.py --run_id "$RUN_ID" || echo "REPORT FAILED (non-fatal)"
echo "[05_report] done"

echo ""
echo "=== R5 pipeline complete at $(date) ==="
echo "Results: runs/$RUN_ID/04_eval/syncnet.json"
echo "         runs/$RUN_ID/05_report/report.md"

# Download back to local:
#   scp -P <port> root@connect.westd.seetacloud.com:/root/autodl-tmp/repos/tts-exp/runs/r5_f5_tts/04_eval/tts_raw/*/syncnet.json runs/r5_f5_tts/04_eval/tts_raw/
