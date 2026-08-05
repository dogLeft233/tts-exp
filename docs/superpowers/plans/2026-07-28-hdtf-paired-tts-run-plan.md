# HDTF Paired TTS-vs-Natural Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare 13 matched HDTF audio/video sources, transcribe their fixed six-second clips, and run the Chinese-baseline Qwen3-TTS self-clone versus natural comparison through Ditto and SyncNet.

**Architecture:** A NAS downloader selects matched WRA/WDA clips by decoded audio duration and unique speaker key, downloads only the selected videos, and writes provenance. A local preprocessor crops matched audio and extracts source frames into a numeric run directory. The remote runner reuses the existing TTS/TFG stages, adding only the configurable ASR stage and using a new isolated staging directory.

**Tech Stack:** Python 3.10, ffprobe/ffmpeg, Synology FileStation API, PyYAML, soundfile, NumPy, DashScope Qwen3-ASR Flash, local Qwen3-TTS 0.6B, Ditto TRT, SyncNet V2, Bash, pytest, rsync/sshpass.

---

## File Map

- Create `scripts/download_hdtf_paired_videos.py`: enumerate NAS video files, match local HDTF audio by stem, select 13 unique speakers, download selected MP4 files, and write a video provenance manifest.
- Create `scripts/prepare_hdtf_pairs.py`: crop matched six-second audio, extract source frames, validate canonical assets, write transcripts directory/config metadata scaffolding, and create the run input manifest.
- Create `scripts/run_hdtf_paired.sh`: run ASR, TTS, Ditto, SyncNet, and report for one prepared run ID.
- Create `tests/test_hdtf_pair_selection.py`: verify matching, WRA/WDA filtering, speaker deduplication, deterministic duration ordering, and config values.
- Modify `scripts/01_asr.py:100-147`: accept a config override and use configured audio paths/sample IDs while preserving the existing Qwen3-ASR Flash request.
- Reuse `scripts/03_ditto.py`, `scripts/04_eval.py`, and metadata-aware `scripts/05_report.py` without changing their defaults.
- Create `docs/superpowers/specs/2026-07-28-hdtf-paired-tts-run-design.md`: approved design record.
- Create `docs/superpowers/plans/2026-07-28-hdtf-paired-tts-run-plan.md`: this implementation plan.

No credentials are stored in source, manifests, run metadata, or shell scripts. Do not modify the default Chinese config.

### Task 1: Add Selection Tests

**Files:**
- Create: `tests/test_hdtf_pair_selection.py`
- Test: `scripts/download_hdtf_paired_videos.py` after Task 2

- [ ] **Step 1: Write the failing selection test**

```python
def test_select_pairs_filters_media_and_deduplicates_speakers():
    audio = [
        {"stem": "WRA_Alice0", "duration_s": 20.0},
        {"stem": "WRA_Alice1", "duration_s": 18.0},
        {"stem": "WDA_Bob", "duration_s": 25.0},
        {"stem": "RD_Radio1", "duration_s": 10.0},
    ]
    video_stems = {"WRA_Alice0", "WRA_Alice1", "WDA_Bob", "RD_Radio1"}
    selected = hdtf.select_pairs(audio, video_stems, count=2)
    assert [item["stem"] for item in selected] == ["WRA_Alice1", "WDA_Bob"]
```

- [ ] **Step 2: Write the failing config test**

```python
def test_prepare_config_uses_hdtf_paths_and_english_protocol():
    config = prepare.build_config_override("hdtf_run", sample_count=13)
    assert config["paths"]["audio_dir"] == "runs/hdtf_run/00_pairs/natural"
    assert config["paths"]["image_dir"] == "runs/hdtf_run/00_pairs/images"
    assert config["tts"]["language"] == "English"
    assert config["ditto"]["conditions"] == ["natural_raw", "tts_raw"]
    assert config["eval"]["require_complete_pairs"] is True
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_hdtf_pair_selection.py`

Expected: FAIL because the two new modules and their public functions do not exist.

### Task 2: Implement NAS Pair Download

**Files:**
- Create: `scripts/download_hdtf_paired_videos.py`

- [ ] **Step 1: Reuse the existing FileStation authentication primitives**

Import the existing NAS helpers from `scripts/download_hdtf_english_audio.py`:
`login`, `list_folder`, `is_directory`, and `download_file`. Add a video enumerator that accepts `.mp4`, `.mkv`, `.mov`, `.avi`, and `.webm` and returns remote paths under `/dhg_mm/HDTF/_videos_raw`.

- [ ] **Step 2: Implement pure deterministic matching and selection**

Use these rules:

```python
def speaker_key(stem: str) -> str:
    return re.sub(r"[0-9]+$", "", stem)


def select_pairs(audio_records, video_stems, count=13):
    candidates = []
    for record in audio_records:
        stem = Path(record["local_path"]).stem
        if stem not in video_stems:
            continue
        if not (stem.startswith("WRA_") or stem.startswith("WDA_")):
            continue
        candidates.append({
            "stem": stem,
            "speaker_key": speaker_key(stem),
            "duration_s": float(record["duration_s"]),
            "audio_path": record["local_path"],
        })
    best_by_speaker = {}
    for candidate in sorted(candidates, key=lambda item: (item["duration_s"], item["stem"])):
        best_by_speaker.setdefault(candidate["speaker_key"], candidate)
    selected = sorted(best_by_speaker.values(), key=lambda item: (item["duration_s"], item["stem"]))
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} unique matched WRA/WDA speakers")
    return selected[:count]
```

- [ ] **Step 3: Download and record provenance**

The CLI accepts `--audio-manifest`, `--remote-root`, `--output-dir`, `--count`, and `--dry-run`. Read `NAS_DHG_MM_PASSWORD` from the environment, query the NAS, download only the selected MP4 files, and write `video_manifest.json` containing selected speaker/stem, local path, remote path, duration, file size, and SHA256. `--dry-run` must not download files.

- [ ] **Step 4: Run selection tests**

Run: `pytest -q tests/test_hdtf_pair_selection.py`

Expected: PASS.

### Task 3: Implement Matched Audio/Frame Preparation

**Files:**
- Create: `scripts/prepare_hdtf_pairs.py`

- [ ] **Step 1: Implement deterministic six-second cropping**

For each selected audio/video record, use `ffprobe` to read audio duration and set:

```python
crop_duration = 6.0
crop_start = max(0.0, (audio_duration - crop_duration) / 2.0)
frame_time = crop_start + crop_duration / 2.0
```

Run ffmpeg for audio:

```python
[
ffmpeg, "-y", "-v", "error", "-ss", f"{crop_start:.3f}",
"-t", "6.0", "-i", str(audio_source),
"-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(natural_dst),
]
```

Run ffmpeg for the source frame with `-ss frame_time -frames:v 1`. Reject missing, empty, or unreadable outputs. Record source and output SHA256, crop start, duration, and frame time.

- [ ] **Step 2: Implement run input and config outputs**

`prepare_pairs(repo, run_id, video_manifest, audio_manifest, count=13)` writes:

- `runs/<run_id>/00_pairs/natural/1..13.wav`;
- `runs/<run_id>/00_pairs/images/1..13.png`;
- `runs/<run_id>/pair_manifest.json`;
- `runs/<run_id>/run_metadata.json` with `design: "hdtf_paired_self_clone"`, `language: "English"`, `sample_count`, source provenance, and `strict_unseen: false`;
- `runs/<run_id>/config_override.yaml` with natural/image/TTS paths, IDs `1..13`, English Qwen language, no loudnorm, raw conditions, and complete-pair evaluation.

Do not create transcript contents at this stage; ASR must produce them from the exact cropped natural audio.

- [ ] **Step 3: Implement one-sample smoke mode**

Accept `--count 1` and `--dry-run`. The smoke input must use the first deterministic candidate and preserve the same crop/frame rules as full mode. If Ditto later rejects the frame, rerun preparation with the next candidate and record the replacement in `pair_manifest.json`.

- [ ] **Step 4: Run preparation tests and compile checks**

Run: `pytest -q tests/test_hdtf_pair_selection.py`

Run: `python -m compileall -q scripts/download_hdtf_paired_videos.py scripts/prepare_hdtf_pairs.py`

Expected: PASS and no compiler output.

### Task 4: Parameterize ASR and Add the HDTF Runner

**Files:**
- Modify: `scripts/01_asr.py:100-147`
- Create: `scripts/run_hdtf_paired.sh`

- [ ] **Step 1: Make ASR use the run config**

Add `--config` to `01_asr.py`. Load the merged config with `utils.load_config`, pass it to `detect_sample_ids`, and resolve `paths.audio_dir` with `resolve_repo_path`. Preserve the existing `qwen3-asr-flash` request, retry behavior, and `DASHSCOPE_API_KEY` environment lookup.

- [ ] **Step 2: Implement the stage runner**

The runner accepts `--run-id`, `--config`, and `--smoke`, resolves the Ditto Python environment, builds `SMOKE_ARGS=()` or `SMOKE_ARGS=(--smoke)`, and executes these stages in order:

```bash
python scripts/01_asr.py --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}"
python scripts/02_tts.py --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}"
python scripts/03_ditto.py --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}"
python scripts/04_eval.py --run_id "$RUN_ID" --config "$CONFIG" "${SMOKE_ARGS[@]}"
python scripts/05_report.py --run_id "$RUN_ID"
```

The runner must generate the report even if `04_eval.py` returns incomplete-pair status, then return the evaluation failure code. It must never print or persist `DASHSCOPE_API_KEY`.

- [ ] **Step 3: Add shell validation**

Run: `bash -n scripts/run_hdtf_paired.sh`

Expected: exit 0.

### Task 5: Local Download, Preparation, and Smoke Inputs

**Files:**
- Temporary only: `/tmp/opencode/hdtf_paired_videos/`
- Temporary only: `/tmp/opencode/hdtf_video_manifest.json`

- [ ] **Step 1: Run NAS dry-run selection**

Run with `NAS_DHG_MM_PASSWORD` injected at runtime:

```bash
NAS_DHG_MM_PASSWORD='<runtime-only>' \
python scripts/download_hdtf_paired_videos.py \
  --audio-manifest data/hdtf_audio_en/manifest.json \
  --output-dir /tmp/opencode/hdtf_paired_videos \
  --count 13 --dry-run
```

Expected: 13 unique WRA/WDA speaker selections and no downloads.

- [ ] **Step 2: Download only the selected videos**

Run the same command without `--dry-run`, writing a manifest under `/tmp/opencode/hdtf_paired_videos`. Verify 13 MP4 files, matching stems, and nonzero SHA256 values.

- [ ] **Step 3: Prepare a one-sample run input**

Run:

```bash
python scripts/prepare_hdtf_pairs.py \
  --run-id hdtf_paired_smoke \
  --video-manifest /tmp/opencode/hdtf_paired_videos/video_manifest.json \
  --audio-manifest data/hdtf_audio_en/manifest.json \
  --count 1
```

Verify one 24 kHz mono PCM16 WAV, one nonempty PNG, one provenance record, and a config override with `natural_raw`/`tts_raw`.

- [ ] **Step 4: Inspect the input frame before remote sync**

Use the image analyzer on `runs/hdtf_paired_smoke/00_pairs/images/1.png` and confirm a visible talking-face source. If no usable face is present, select the next candidate, rerun preparation, and retain the replacement record.

### Task 6: Remote Isolated Smoke and Full Run

**Files:**
- Remote create: `/root/autodl-tmp/tts-exp-hdtf-<timestamp>/`
- Remote outputs: `runs/hdtf_paired_*`

- [ ] **Step 1: Create a new remote staging directory**

Reuse the remote `third_party` symlink, `/root/autodl-tmp/envs`, and `/root/autodl-tmp/checkpoints`. Sync current `scripts/`, the prepared run input, and only the required HDTF manifest. Do not copy all 871 MB of HDTF audio or raw videos if the prepared six-second WAVs are sufficient.

- [ ] **Step 2: Sync the smoke run and execute ASR/TTS/TFG**

Run the remote runner with `DASHSCOPE_API_KEY` injected only into that process environment. Verify ASR produces `01_transcript/1.txt`, TTS produces one WAV, Ditto produces natural/TTS MP4s, and SyncNet writes both JSON results.

- [ ] **Step 3: Start the full 13-sample run only after smoke succeeds**

Prepare/sync the full run input and run the full runner detached with an external log. Verify `run_metadata.json`, `pair_manifest.json`, `eval_meta.json`, `report.md`, `summary.csv`, and `report_stats.json` exist.

- [ ] **Step 4: Handle face or SyncNet failures explicitly**

If a candidate cannot generate a valid Ditto video, do not silently drop it. Replace it with the next deterministic candidate, rerun preparation for that numeric ID, and record the old/new source IDs. If TTS audio is too short for SyncNet, preserve the failure and report the valid paired `n` rather than treating it as a complete 13-sample result.

### Task 7: Verify and Report Results

- [ ] **Step 1: Re-read all run metadata and evaluation metadata**

Check that selected sources have distinct speaker keys, all run assets have SHA256/provenance, and `complete_case_ids` matches the number used in paired statistics.

- [ ] **Step 2: Extract paired metrics**

Report per-run Natural/TTS means, `TTS - Natural` Sync-C, `TTS - Natural` Sync-D, paired `n`, p-values, and failed IDs. Do not pool the HDTF result with the five MDC languages.

- [ ] **Step 3: State the interpretation boundary**

Label the result `HDTF paired-source exploratory`, `self_clone`, and `strict_unseen: false` in the final report. Keep the known Ditto/HDTF evaluation overlap visible.

No commit is included because the worktree contains unrelated user changes and the user did not request a commit.
