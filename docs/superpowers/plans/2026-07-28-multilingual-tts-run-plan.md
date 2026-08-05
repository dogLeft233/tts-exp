# Multilingual TTS-vs-Natural Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run five isolated 13-sample MDC language experiments with the same self-clone TTS, Ditto, and SyncNet protocol used by the final 13-sample Chinese baseline.

**Architecture:** A new manifest-driven preparation script selects the longest 13 records per language, canonicalizes them to 24 kHz mono PCM16 WAV, writes gold transcripts and a per-run config, and leaves the existing sample-ID based stages intact. A shell runner executes one language per run ID in a separate remote staging repository, reusing only the remote third-party repositories, environments, and checkpoints.

**Tech Stack:** Python 3.10, PyYAML, soundfile, NumPy, ffmpeg, local Qwen3-TTS 0.6B, Ditto TRT offline inference, SyncNet V2, Bash, pytest, rsync/sshpass.

---

## File Map

- Create `scripts/prepare_mdc_pairs.py`: manifest loading, deterministic 13-sample selection, audio canonicalization/QC, transcript creation, run metadata, and config override generation.
- Create `scripts/run_mdc_multilingual.sh`: per-language run orchestration with smoke mode and failure isolation.
- Create `tests/test_prepare_mdc_pairs.py`: pure selection and generated-config tests using temporary fixtures.
- Modify `scripts/00_datacheck.py:76-90`: accept a config override and use configured audio/image paths.
- Modify `scripts/02_tts.py:123-157`: use the configured natural-audio directory instead of the hardcoded Chinese directory.
- Modify `scripts/05_report.py:165-233,287-339`: read optional run metadata and report language, design, and actual sample description.
- Create `docs/superpowers/specs/2026-07-28-multilingual-tts-run-design.md`: approved protocol record.
- Create `docs/superpowers/plans/2026-07-28-multilingual-tts-run-plan.md`: this implementation plan.

Do not modify the default Chinese values in `scripts/config.yaml`. Do not overwrite the remote repository, which has unrelated uncommitted changes.

### Task 1: Add Preparation Tests

**Files:**
- Create: `tests/test_prepare_mdc_pairs.py`
- Test: `scripts/prepare_mdc_pairs.py` after Task 2

- [ ] **Step 1: Write the failing selection test**

```python
def test_select_samples_uses_longest_records_and_stable_manifest_order():
    records = [
        {"sample_id": "en_001", "language_code": "en", "duration_s": 4.0},
        {"sample_id": "en_002", "language_code": "en", "duration_s": 8.0},
        {"sample_id": "en_003", "language_code": "en", "duration_s": 8.0},
        {"sample_id": "de_001", "language_code": "de", "duration_s": 99.0},
    ]
    selected = prepare.select_samples(records, "en", count=3)
    assert [row["sample_id"] for row in selected] == ["en_002", "en_003", "en_001"]
```

- [ ] **Step 2: Write the failing config/metadata test**

```python
def test_build_config_uses_numeric_ids_and_language_settings():
    override = prepare.build_config_override("de", "mdc_de_run", sample_count=13)
    assert override["samples"]["ids"] == list(range(1, 14))
    assert override["tts"]["language"] == "German"
    assert override["ditto"]["conditions"] == ["natural_raw", "tts_raw"]
    assert override["eval"]["require_complete_pairs"] is True
    assert override["paths"]["audio_dir"] == "runs/mdc_de_run/00_pairs/natural"
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_prepare_mdc_pairs.py`

Expected: FAIL because `scripts/prepare_mdc_pairs.py` and its public functions do not exist yet.

### Task 2: Implement Manifest Preparation

**Files:**
- Create: `scripts/prepare_mdc_pairs.py`

- [ ] **Step 1: Implement the language table and pure selection functions**

Use this exact mapping and stable sort behavior:

```python
LANGUAGE_CONFIG = {
    "en": {"label": "English", "qwen": "English"},
    "de": {"label": "German", "qwen": "German"},
    "it": {"label": "Italian", "qwen": "Italian"},
    "es": {"label": "Spanish", "qwen": "Spanish"},
    "ko": {"label": "Korean", "qwen": "Korean"},
}


def select_samples(records: list[dict], language_code: str, count: int = 13) -> list[dict]:
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("language_code") == language_code
    ]
    if len(candidates) < count:
        raise ValueError(f"{language_code}: need {count} records, found {len(candidates)}")
    ranked = sorted(
        candidates,
        key=lambda item: (-float(item[1].get("duration_s", 0.0)), item[0]),
    )
    return [record for _, record in ranked[:count]]
```

- [ ] **Step 2: Implement canonical audio conversion and QC**

For each selected source path from `file_path`, run ffmpeg with:

```python
command = [
    ffmpeg_bin, "-y", "-v", "error", "-i", str(source),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
]
```

Read the result with `soundfile.info` and `soundfile.read(dtype="float32", always_2d=True)`. Reject empty, silent, non-finite, or out-of-range audio. Record source and canonical duration, sample rate, channels, subtype, source SHA256, and canonical SHA256. Set `shorter_than_5s` from the canonical duration.

- [ ] **Step 3: Implement per-run outputs**

`prepare_language(repo, run_id, language_code, manifest_path, image_dir, sample_count=13, ffmpeg_bin="ffmpeg")` must create:

- `runs/<run_id>/00_pairs/natural/1.wav` through `13.wav`;
- `runs/<run_id>/01_transcript/1.txt` through `13.txt`;
- `runs/<run_id>/01_transcript/transcript.json` with integer `sample_id` keys in `results` values, matching `02_tts.py:41-45`;
- `runs/<run_id>/00_pairs/pair_manifest.json` with selected source rows and QC;
- `runs/<run_id>/run_metadata.json` with `design: "self_clone"`, language, sample count, short-audio flags, image mapping, and manifest SHA256;
- `runs/<run_id>/config_override.yaml` from `build_config_override`.

Validate `data/data/image/{1..13}.png` before writing a complete run manifest. Use the manifest text as both `ref_text` and target text; do not invoke ASR.

- [ ] **Step 4: Implement CLI and dry-run support**

The CLI must accept `--run-id`, `--language`, `--manifest`, `--sample-count`, `--image-dir`, `--ffmpeg`, and `--dry-run`. `--dry-run` prints selected IDs and durations without creating audio outputs. Exit nonzero on missing source files, missing images, decode/QC failure, or fewer than 13 language records.

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_prepare_mdc_pairs.py`

Expected: PASS.

### Task 3: Parameterize Existing Stages and Reports

**Files:**
- Modify: `scripts/00_datacheck.py:76-90`
- Modify: `scripts/02_tts.py:95-157`
- Modify: `scripts/05_report.py:165-233,287-339`

- [ ] **Step 1: Parameterize data-check paths**

Add `--config` to `00_datacheck.py`, load it with `utils.load_config`, pass the merged config to `detect_sample_ids`, and resolve `paths.audio_dir` and `paths.image_dir` with `resolve_repo_path`. Preserve the default Chinese paths when no override is provided.

- [ ] **Step 2: Parameterize the TTS natural-audio path**

In `02_tts.py`, replace the hardcoded assignment:

```python
audio_dir = repo / "data" / "data" / "audio"
```

with:

```python
from utils import resolve_repo_path

audio_dir = resolve_repo_path(
    repo,
    cfg.get("paths", {}).get("audio_dir", "data/data/audio"),
)
```

Keep `ref_text=text` and the existing provider/seed behavior unchanged.

- [ ] **Step 3: Add metadata-aware report method text**

Extend `generate_report` with an optional `run_metadata: dict | None = None`. Use backward-compatible defaults, then add these method lines before the existing sample line:

```python
metadata = run_metadata or {}
report_lines.extend([
    f"- **Design**: {metadata.get('design', 'legacy')}",
    f"- **Language**: {metadata.get('language', 'unknown')}",
    f"- **Samples**: {metadata.get('sample_description', 'legacy Chinese sample set')}",
])
```

In `main`, load `run_dir / "run_metadata.json"` when present and pass it to `generate_report`. Do not change paired statistics or the natural/TTS direction calculation.

- [ ] **Step 4: Add a report metadata regression test**

Add a test that calls `generate_report` with one metadata dictionary and asserts `report.md` contains `self_clone`, the language label, and the 13-sample description. Preserve existing report behavior when metadata is absent.

- [ ] **Step 5: Run stage tests and compile checks**

Run: `pytest -q tests/test_prepare_mdc_pairs.py tests/test_eval_stage.py tests/test_english_pair_prep.py`

Run: `python -m compileall -q scripts/00_datacheck.py scripts/02_tts.py scripts/05_report.py scripts/prepare_mdc_pairs.py`

Expected: PASS and no compiler output.

### Task 4: Add the Multilingual Runner

**Files:**
- Create: `scripts/run_mdc_multilingual.sh`

- [ ] **Step 1: Implement argument parsing and environment setup**

Support `--languages en,de,it,es,ko` and `--smoke`. Resolve the repository from the script location, use `/root/autodl-tmp/envs/ditto/bin/python` when executable, export `HF_ENDPOINT=https://hf-mirror.com`, and prepend Ditto's cuDNN library directory when present.

- [ ] **Step 2: Implement one isolated language run**

For each language, create `RUN_ID=mdc_<language>_<UTC timestamp>` and execute:

```bash
python scripts/prepare_mdc_pairs.py \
  --run-id "$RUN_ID" \
  --language "$LANGUAGE" \
  --manifest data/mdc_tts/manifest.json \
  --sample-count "$COUNT"
python scripts/00_datacheck.py --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml"
python scripts/02_tts.py --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml"
python scripts/03_ditto.py --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml"
python scripts/04_eval.py --run_id "$RUN_ID" \
  --config "runs/$RUN_ID/config_override.yaml"
python scripts/05_report.py --run_id "$RUN_ID"
```

Skip `01_asr.py` because preparation writes the manifest's gold text. In smoke mode use `COUNT=1` and run only the requested language. The runner must record failed language IDs and continue full runs for other languages; smoke failure must exit nonzero before full execution.

- [ ] **Step 3: Add shell validation**

Run: `bash -n scripts/run_mdc_multilingual.sh`

Expected: exit 0.

### Task 5: Local Data and Runner Verification

**Files:**
- No additional source files.

- [ ] **Step 1: Check actual sample selections without writing outputs**

Run:

```bash
for language in en de it es ko; do
  python scripts/prepare_mdc_pairs.py \
    --run-id check_$language \
    --language "$language" \
    --manifest data/mdc_tts/manifest.json \
    --sample-count 13 \
    --dry-run
done
```

Expected: exactly 13 selected records for each language; output includes the short-audio flags, and Italian records show one source speaker where metadata provides it.

- [ ] **Step 2: Run the complete local test suite relevant to the pipeline**

Run: `pytest -q tests/test_prepare_mdc_pairs.py tests/test_eval_stage.py tests/test_english_pair_prep.py`

Expected: all tests pass.

- [ ] **Step 3: Inspect the intended diff before remote sync**

Run: `git diff -- docs/superpowers/specs/2026-07-28-multilingual-tts-run-design.md docs/superpowers/plans/2026-07-28-multilingual-tts-run-plan.md scripts/prepare_mdc_pairs.py scripts/run_mdc_multilingual.sh scripts/00_datacheck.py scripts/02_tts.py scripts/05_report.py tests/test_prepare_mdc_pairs.py`

Expected: only the planned files are changed; unrelated pre-existing worktree changes remain untouched.

### Task 6: Isolated Remote Sync and English Smoke

**Files:**
- Remote create: `/root/autodl-tmp/tts-exp-mdc-<timestamp>/`
- Remote reuse: `/root/autodl-tmp/repos/tts-exp/third_party`, `/root/autodl-tmp/envs`, and `/root/autodl-tmp/checkpoints`

- [ ] **Step 1: Create a new remote staging directory**

Use the supplied SSH credential through `SSHPASS`/`sshpass -e`; never put the password in a file or command script. Create only a new directory under `/root/autodl-tmp`, then link the existing remote `third_party` directory into it.

- [ ] **Step 2: Sync only required files**

Use `rsync` for `scripts/`, `data/mdc_tts/`, and `data/data/image/`. Do not sync `.git`, `runs`, unrelated `data`, or the remote repository's uncommitted files.

- [ ] **Step 3: Run English one-sample smoke**

From the new staging directory run:

```bash
bash scripts/run_mdc_multilingual.sh --languages en --smoke
```

Verify the smoke run has one selected sample, one TTS WAV, one natural MP4, one TTS MP4, two parseable `syncnet.json` files, and `complete_case_ids` containing sample `1`.

- [ ] **Step 4: Stop on smoke failure**

If any smoke artifact is missing or SyncNet parsing fails, do not start the full matrix. Capture the exact stage log and error for diagnosis.

### Task 7: Start and Monitor the 5×13 Matrix

**Files:**
- Remote outputs: `<staging>/runs/mdc_en_*`, `mdc_de_*`, `mdc_it_*`, `mdc_es_*`, `mdc_ko_*`
- Remote log: `<staging>/mdc_full.log`

- [ ] **Step 1: Start the full run detached after smoke passes**

Run the full command with `nohup` and a remote log, preserving the returned PID:

```bash
nohup bash scripts/run_mdc_multilingual.sh \
  --languages en,de,it,es,ko > mdc_full.log 2>&1 < /dev/null &
printf '%s\n' "$!"
```

- [ ] **Step 2: Poll progress without restarting the job**

Inspect `mdc_full.log`, each `run_metadata.json`, `02_tts/tts_meta.json`, `03_ditto/ditto_meta.json`, `04_eval/eval_meta.json`, and `05_report/report_stats.json`. Check that every completed language has 13 selected samples and paired natural/TTS results.

- [ ] **Step 3: Record final remote paths and status**

Report the PID, staging directory, per-language run IDs, completed/incomplete languages, failed sample IDs, and exact report paths. Do not claim a language is complete until `eval_meta.json` has complete pairs and `05_report/report_stats.json` exists.

No commit is included in this plan because the worktree already contains unrelated user changes and the user did not request a commit. If a commit is later requested, stage only the planned source/tests/docs files.
