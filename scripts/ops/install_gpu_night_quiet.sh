#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ $EUID != 0 ]]; then
    echo 'Run: sudo bash install_gpu_night_quiet.sh' >&2
    exit 1
fi
if [[ -e /etc/systemd/system/gpu-night-quiet.service || -e /etc/gpu-night-quiet.json || -e /usr/local/sbin/gpu-night-quiet ]]; then
    echo 'Existing installation found; refusing to overwrite. Review/update explicitly.' >&2
    exit 1
fi
/usr/bin/python3 gpu_night_quiet.py --config gpu-night-quiet.json --dry-run --once
for schedule in '2026-10-11T22:59:59+08:00' '2026-10-11T23:00:00+08:00' '2026-10-12T07:59:59+08:00' '2026-10-12T08:00:00+08:00'; do
    /usr/bin/python3 gpu_night_quiet.py --config gpu-night-quiet.json --dry-run --once --at "$schedule"
done
install -o root -g root -m 0755 gpu_night_quiet.py /usr/local/sbin/gpu-night-quiet
install -o root -g root -m 0644 gpu-night-quiet.json /etc/gpu-night-quiet.json
install -o root -g root -m 0644 gpu-night-quiet.service /etc/systemd/system/gpu-night-quiet.service
systemd-analyze verify /etc/systemd/system/gpu-night-quiet.service
systemctl daemon-reload
/usr/bin/python3 /usr/local/sbin/gpu-night-quiet --config /etc/gpu-night-quiet.json --once
systemctl enable --now gpu-night-quiet.service
sleep 2
systemctl is-active --quiet gpu-night-quiet.service
systemctl is-enabled gpu-night-quiet.service
journalctl -u gpu-night-quiet.service -n 8 --no-pager
/usr/bin/nvidia-smi --query-gpu=uuid,temperature.gpu,temperature.memory,power.draw,power.limit --format=csv
