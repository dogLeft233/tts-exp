# TFG/THG and Multilingual Experiment Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise Chinese experiment summary covering all scored TFG/THG models, the HDTF English experiments, and the five-language self-clone experiment.

**Architecture:** Create one standalone report under `docs/experiments/` rather than expanding the existing long reports. Separate results by protocol, identify the exact source/year for every scored model, and identify both the original dataset purpose and this project's experimental use for every dataset.

**Tech Stack:** Markdown, existing repository experiment reports, OpenCode session records, SyncNet V2 summary tables.

---

### Task 1: Write and Verify the Consolidated Experiment Report

**Files:**
- Create: `docs/experiments/07-tfg-thg-multilingual-summary.md`
- Reference: `docs/results/SYNC-C-SCORES.md`
- Reference: `docs/experiments/04-english-replication.md`
- Reference: `docs/reference/dataset-reference.md`
- Reference: `docs/reference/multilingual-datasets.md`

- [x] **Step 1: Add the report scope and protocol boundary**

  State that Sync-C is higher-is-better and Sync-D is lower-is-better. Define TFG/THG as the project umbrella for audio-driven portrait/talking-face/talking-head generation, but preserve separate labels for I2V, V2V, and streaming protocols so incompatible inputs are not presented as one fair ranking.

- [x] **Step 2: Add the dataset-purpose table**

  Include AISHELL-1, HDTF, LibriSpeech test-clean, and the exact five session-recorded dataset names:

  - `Effect AI Scripted Speech 1.0 - English`
  - `Common Voice Spontaneous Speech 4.0 - German`
  - `Italian TTS - female voice`
  - `Common Voice Spontaneous Speech 4.0 - Spanish`
  - `Common Voice Scripted Speech 26.0 - Korean`

  For each row, state the dataset's original purpose and its use here. State that the five-language run uses 13 selected samples per language, self-clone TTS, Ditto, and within-language TTS-minus-Natural comparisons.

- [x] **Step 3: Add all scored model comparisons**

  Use the final SyncNet table for the original nine-model batch and the separate 2026 model records for SoulX-FlashHead Pro, AVTR-1, AvatarForcing, and LeapTalk. Include model source/paper and year, protocol label, sample count, Natural/TTS Sync-C, ΔSync-C, Natural/TTS Sync-D where available, and ΔSync-D. Mark Hallo2's no-separator result and EchoMimic V2's synthetic-pose limitation.

- [x] **Step 4: Add attempted-but-not-ranked models**

  List EchoMimic V1, AniPortrait, official Hallo2 default-pipeline result, ACTalker, StableAvatar, AptAvatar, and LipForcing with their status and the reason they are not part of the unified scored ranking. Do not invent scores for models without a completed SyncNet evaluation.

- [x] **Step 5: Add HDTF and multilingual result sections**

  Report the HDTF 6-second center-crop and fixed-image results from the English replication document, then add the session-recorded full-audio result as a separate protocol. Report the five-language table exactly as recorded in the OpenCode session, including valid pairs, deltas, and the Korean nominal p-values. Explicitly note incomplete English/Spanish pairs and the single-speaker Italian condition.

- [x] **Step 6: Add concise conclusions and limitations**

  Summarize the Chinese multi-model positive response, English/HDTF protocol dependence, multilingual within-language pattern, and the prohibition on pooling absolute SyncNet scores across languages or incompatible generation protocols.

- [x] **Step 7: Self-review and verify the report**

  Run:

  ```bash
  rg -n "Effect AI Scripted Speech 1\.0 - English|Common Voice Spontaneous Speech 4\.0 - German|Italian TTS - female voice|Common Voice Spontaneous Speech 4\.0 - Spanish|Common Voice Scripted Speech 26\.0 - Korean|HDTF|SoulX-FlashHead|AVTR-1|AvatarForcing|LeapTalk" docs/experiments/07-tfg-thg-multilingual-summary.md
  git diff --check -- docs/experiments/07-tfg-thg-multilingual-summary.md
  ```

  Confirm the exact five dataset names appear, each scored model has a source/year, the HDTF protocols are separated, and no unscored model is given a fabricated numeric result.
