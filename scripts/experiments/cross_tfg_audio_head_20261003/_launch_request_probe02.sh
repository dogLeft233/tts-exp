set -euo pipefail
TASK=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stageb_musetalk_attempt04
CTRL=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/stageb_musetalk_attempt04_attempt01
OUT=/dev/shm/cross_tfg_audio_head_20261003/stageb_musetalk_attempt04/run
LAUNCHER=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stageb_musetalk_attempt04/launch_musetalk_stageb_attempt04.sh
EXPECTED_LAUNCHER_SHA=876f0b4889e7f8b615b221d98b589b664ecbeb77b2b9c7496e2ca9c066ff4c8c
EXPECTED_BUNDLE_SHA=76fbd3ff3feb5eb17ff7f863246327c172dc5d34c5d3b70b349f87285904681c
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA"
test "$(sha256sum "$TASK/musetalk_stageb_attempt04_bundle.zip" | awk '{print $1}')" = "$EXPECTED_BUNDLE_SHA"
bash -n "$LAUNCHER"
test ! -e "$CTRL/worker.pid"
test ! -e "$OUT"
nohup bash "$LAUNCHER" </dev/null > "$CTRL/bootstrap.stdout" 2> "$CTRL/bootstrap.stderr" &
PID=$!
printf "%s\n" "$PID" > "$CTRL/dispatch_pid.txt"
printf "LAUNCH_REQUEST_DISPATCHED launcher_sha256=%s bundle_sha256=%s pid=%s\n" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_BUNDLE_SHA" "$PID"