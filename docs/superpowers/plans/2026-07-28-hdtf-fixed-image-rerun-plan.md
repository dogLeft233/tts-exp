# HDTF Fixed-Image Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run HDTF Ditto and SyncNet evaluation with the existing fixed images used by the other experiments while reusing the completed HDTF v2 audio and TTS artifacts.

**Architecture:** Create an isolated run on the GPU staging workspace. Its config points audio and TTS inputs to `hdtf_paired_full_v2`, but points `image_dir` to `data/data/image`. Run only Ditto, SyncNet, and report generation, then verify source hashes, image paths, pair completeness, and the original run remains unchanged.

**Tech Stack:** Bash, Python/YAML config, Ditto TRT adapter, SyncNet V2, `scripts/03_ditto.py`, `scripts/04_eval.py`, `scripts/05_report.py`.

---

### Task 1: Create the isolated fixed-image run configuration

**Files:**
- Create on the remote staging workspace: `runs/$RUN_ID/config_override.yaml`
- Create on the remote staging workspace: `runs/$RUN_ID/run_metadata.json`
- Create on the remote staging workspace: `runs/$RUN_ID/pair_manifest.json`

- [ ] **Step 1: Verify source inputs and all 13 fixed images exist**

Run on the GPU workspace:

```bash
export RUN_ID="hdtf_paired_fixed_image_$(date -u +%Y%m%dT%H%M%SZ)"
test -f runs/hdtf_paired_full_v2/run_metadata.json
test -f runs/hdtf_paired_full_v2/00_pairs/natural/1.wav
test -f runs/hdtf_paired_full_v2/02_tts/1.wav
for i in $(seq 1 13); do test -f "data/data/image/$i.png"; done
```

Expected: all commands exit successfully.

- [ ] **Step 2: Write the isolated config and provenance metadata**

Generate the files with this Python program after exporting `RUN_ID`:

```python
import copy
import hashlib
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
source_id = "hdtf_paired_full_v2"
repo = Path.cwd()
source_run = repo / "runs" / source_id
run_dir = repo / "runs" / run_id
run_dir.mkdir(parents=True, exist_ok=False)

source_metadata = json.loads(
    (source_run / "run_metadata.json").read_text(encoding="utf-8")
)
records = copy.deepcopy(source_metadata["records"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for record in records:
    sample_id = int(record["sample_id"])
    image = repo / "data" / "data" / "image" / f"{sample_id}.png"
    record["image_protocol"] = "fixed_existing_images"
    record["source_image"] = str(image.relative_to(repo))
    record["image_sha256"] = sha256(image)
    tts_audio = source_run / "02_tts" / f"{sample_id}.wav"
    record["tts_audio"] = str(tts_audio.relative_to(repo))
    record["tts_audio_sha256"] = sha256(tts_audio)

metadata = copy.deepcopy(source_metadata)
metadata.update({
    "run_id": run_id,
    "source_run_id": source_id,
    "design": "hdtf_paired_self_clone_fixed_existing_images",
    "image_protocol": "fixed_existing_images",
    "image_dir": "data/data/image",
    "sample_description": (
        "HDTF v2 paired source with fixed existing images, "
        "13 distinct speakers, centered six-second crops"
    ),
    "records": records,
})
(run_dir / "run_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(run_dir / "pair_manifest.json").write_text(
    json.dumps({
        "schema_version": 1,
        "design": metadata["design"],
        "source_run_id": source_id,
        "image_protocol": "fixed_existing_images",
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(run_dir / "config_override.yaml").write_text(
    """paths:
  audio_dir: runs/hdtf_paired_full_v2/00_pairs/natural
  tts_audio_dir: runs/hdtf_paired_full_v2/02_tts
  image_dir: data/data/image
samples:
  count: 13
  ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  smoke_id: 1
tts:
  language: English
  seed: 42
  retry: 1
ditto:
  loudnorm: false
  conditions: [natural_raw, tts_raw]
  retry: 1
  seed: 42
eval:
  require_complete_pairs: true
""",
    encoding="utf-8",
)
source_tts_meta = source_run / "02_tts" / "tts_meta.json"
if source_tts_meta.exists():
    tts_meta_dir = run_dir / "02_tts"
    tts_meta_dir.mkdir()
    (tts_meta_dir / "tts_meta.json").write_bytes(source_tts_meta.read_bytes())
```

- [ ] **Step 3: Validate config and provenance before inference**

Run:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import yaml

run = Path("runs") / os.environ["RUN_ID"]
cfg = yaml.safe_load((run / "config_override.yaml").read_text())
meta = json.loads((run / "run_metadata.json").read_text())
assert cfg["paths"]["image_dir"] == "data/data/image"
assert cfg["paths"]["audio_dir"] == "runs/hdtf_paired_full_v2/00_pairs/natural"
assert cfg["paths"]["tts_audio_dir"] == "runs/hdtf_paired_full_v2/02_tts"
assert meta["image_protocol"] == "fixed_existing_images"
assert len(meta["records"]) == 13
assert all(r["source_image"] == f"data/data/image/{r['sample_id']}.png" for r in meta["records"])
print("fixed-image config and provenance valid")
PY
```

Expected: `fixed-image config and provenance valid`.

### Task 2: Run Ditto with fixed images

**Files:**
- Create on the remote staging workspace: `runs/$RUN_ID/03_ditto/`
- Create on the remote staging workspace: `runs/$RUN_ID/03_ditto/ditto_meta.json`

- [ ] **Step 1: Run the existing Ditto stage without regenerating ASR/TTS**

Run on the GPU workspace:

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="/root/autodl-tmp/envs/ditto/opt/cudnn8/lib:${LD_LIBRARY_PATH:-}"
DITTO_PYTHON="${DITTO_PYTHON:-/root/autodl-tmp/envs/ditto/bin/python}"
"$DITTO_PYTHON" scripts/03_ditto.py \
  --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml" \
  --no-cache
```

Expected: 26 videos, 13 each for `natural_raw` and `tts_raw`, with no missing fixed images.

- [ ] **Step 2: Verify Ditto metadata records the fixed-image path**

Run:

```bash
test "$(find "runs/$RUN_ID/03_ditto/natural_raw" -name '*.mp4' | wc -l)" -eq 13
test "$(find "runs/$RUN_ID/03_ditto/tts_raw" -name '*.mp4' | wc -l)" -eq 13
```

Expected: both counts equal `13`.

### Task 3: Run SyncNet and generate the report

**Files:**
- Create on the remote staging workspace: `runs/$RUN_ID/04_eval/`
- Create on the remote staging workspace: `runs/$RUN_ID/05_report/`

- [ ] **Step 1: Run SyncNet evaluation**

Run:

```bash
DITTO_PYTHON="${DITTO_PYTHON:-/root/autodl-tmp/envs/ditto/bin/python}"
"$DITTO_PYTHON" scripts/04_eval.py \
  --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml" \
  --no-cache
```

Expected: `complete cases=13/13`; if not, retain `eval_meta.json` and failure details.

- [ ] **Step 2: Generate the fixed-image report**

Run:

```bash
DITTO_PYTHON="${DITTO_PYTHON:-/root/autodl-tmp/envs/ditto/bin/python}"
"$DITTO_PYTHON" scripts/05_report.py --run_id "$RUN_ID"
```

Expected: `report.md`, `summary.csv`, `report_stats.json`, and figures are written under `runs/$RUN_ID/05_report/`.

### Task 4: Verify comparison and preserve the original run

**Files:**
- Read: `runs/$RUN_ID/05_report/report.md`
- Read: `runs/$RUN_ID/05_report/report_stats.json`
- Read: `runs/hdtf_paired_full_v2/run_metadata.json`

- [ ] **Step 1: Check fixed-image provenance and source hashes**

Run:

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run = Path("runs") / os.environ["RUN_ID"]
new = json.loads((run / "run_metadata.json").read_text())
old = json.loads(Path("runs/hdtf_paired_full_v2/run_metadata.json").read_text())
assert new["source_run_id"] == "hdtf_paired_full_v2"
assert [r["source_audio_sha256"] for r in new["records"]] == [r["source_audio_sha256"] for r in old["records"]]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert all(sha256(Path(r["tts_audio"])) == r["tts_audio_sha256"] for r in new["records"])
assert all(r["source_image"].startswith("data/data/image/") for r in new["records"])
print("fixed-image provenance and source audio/TTS hashes verified")
PY
```

Expected: `fixed-image provenance and source audio/TTS hashes verified`.

- [ ] **Step 2: Compare the new means and direction with the HDTF v2 report**

Run:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
for label, path in [
    ("source-frame", Path("runs/hdtf_paired_full_v2/05_report/report_stats.json")),
    ("fixed-image", Path("runs") / run_id / "05_report/report_stats.json")),
]:
    stats = json.loads(path.read_text())
    print(json.dumps({
        "protocol": label,
        "paired_c": stats.get("paired_c"),
        "paired_d": stats.get("paired_d"),
        "direction_ok": stats.get("direction_ok"),
    }, ensure_ascii=False, sort_keys=True))
PY
```

Expected: one line per protocol containing Sync-C/Sync-D paired n, p-values,
effect sizes, and direction judgement. Interpret any change as an
image-protocol sensitivity result, not as a strict unseen generalization
result, because the HDTF/Ditto evaluation overlap remains.

- [ ] **Step 3: Confirm the original run is untouched**

Run:

```bash
test -f "runs/$RUN_ID/run_metadata.json"
test -f "runs/$RUN_ID/config_override.yaml"
test -f runs/hdtf_paired_full_v2/run_metadata.json
test -f runs/hdtf_paired_full_v2/config_override.yaml
```

Expected: both commands exit successfully and no files under the original run are modified.
