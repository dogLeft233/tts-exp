#!/usr/bin/env bash
set -euo pipefail

CODE=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/visual_grid_attempt01/code
JOB=/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03_visual_grid_attempt01
OUT=/dev/shm/cross_tfg_audio_head_20261003/proxy_native_full512_parity_attempt03/visual_grid_attempt01/output
PY=/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python
SCRIPT=$CODE/render_full512_parity_grid_attempt01.py
INPUT=$CODE/visual_grid_inputs_attempt01.json

test -d "$CODE"
test -d "$JOB"
test ! -e "$OUT"
printf '%s  %s\n' '33289897a7ae4fd88c7317bfea318612aebbed5bd15501bc32e9b232c33bbfd6' "$SCRIPT" | sha256sum -c -
printf '%s  %s\n' '43665d4c3678ea4199743c16d2fe4ea1346863e10db83b582a28beac1184648d' "$INPUT" | sha256sum -c -
PYTHONDONTWRITEBYTECODE=1 "$PY" -B - "$INPUT" <<'PY'
import hashlib
import json
import pathlib
import sys

import cv2
import numpy as np

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

assert sha(manifest["summary_path"]) == manifest["summary_raw_sha256"]
assert len(manifest["rows"]) == 2
assert sha(manifest["source_portrait_path"]) == manifest["source_portrait_sha256"]
for row in manifest["rows"]:
    ref = row["visual_arrays"]
    p = pathlib.Path(ref["npz_path"])
    assert p.is_file() and p.stat().st_size == ref["npz_bytes"]
    assert sha(p) == ref["npz_sha256"]
print(json.dumps({"status":"PREFLIGHT_PASS","python":sys.version.split()[0],"numpy":np.__version__,"opencv":cv2.__version__,"rows":len(manifest["rows"])},sort_keys=True,separators=(",",":")))
PY
