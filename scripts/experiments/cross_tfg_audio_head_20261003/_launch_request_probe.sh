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
if test -e "$CTRL/worker.pid"; then printf "%s\n" CONTROL_ALREADY_USED; exit 97; fi
if test -e "$OUT"; then printf "%s\n" OUTPUT_ALREADY_EXISTS; exit 98; fi
nohup bash "$LAUNCHER" </dev/null > "$CTRL/bootstrap.stdout" 2> "$CTRL/bootstrap.stderr" &
PID=$!
printf "%s\n" "$PID" > "$CTRL/dispatch_pid.txt"
for i in $(seq 1 32); do test -f "$CTRL/worker.pid" && break; test -f "$CTRL/launch_error.txt" && break; sleep 0.25; done
if test -f "$CTRL/launch_error.txt"; then STATE=FAILED_BEFORE_RUNNER; elif test -f "$CTRL/worker.pid" && test ! -f "$CTRL/exitcode"; then STATE=STARTED; elif test -f "$CTRL/exitcode"; then STATE=EXITED; else STATE=UNKNOWN; fi
printf "state=%s\nlauncher_sha256=%s\nbundle_sha256=%s\ndispatch_pid=%s\n" "$STATE" "$EXPECTED_LAUNCHER_SHA" "$EXPECTED_BUNDLE_SHA" "$PID"
if test -f "$CTRL/worker.pid"; then printf "worker_pid="; tr -d "\n" < "$CTRL/worker.pid"; printf "\n"; fi
if test -f "$CTRL/launch_error.txt"; then printf "launch_error="; tr -d "\n" < "$CTRL/launch_error.txt"; printf "\n"; fi
if test -f "$CTRL/exitcode"; then printf "outer_exit="; tr -d "\n" < "$CTRL/exitcode"; printf "\n"; fi