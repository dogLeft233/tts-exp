---
title: Codex checkpoint - 2026-10-14T04-06-47Z - Cross-TFG CPU coordinates qualified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t04-06-47-z-cross-tfg-cpu-coordinates-qualified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 04:06:47+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-14T04-06-47Z - Cross-TFG CPU coordinates qualified

## Summary
The corrected CPU calibration now covers Wav2Lip, MuseTalk, and Ditto on the same FIT corpus; all three native entry points are mechanically reproducible, but ENV remains unavailable and no factor intervention has yet been scored.

## Story
The active objective is to test SEP, STAB, and ENV independently in each TFG native input space while fixing RMS, clock, content, and final natural PCM, then compare SyncNet with mouth aperture/velocity. After the bounded Wav2Lip STAB result failed the two-speaker/motion expansion gate, the work stayed CPU-only to close coordinate provenance before spending more GPU time.

Corrected speaker-partitioned MFA retry02 passed ten per-speaker commands (40 TextGrids, MFA v2.2.17, pinned model/dictionary assets, exact command receipts). Wav2Lip calibration passed on S0053-S0061 (18 pairs/36 rows/9 speakers, 631+631 temporal segments, 58/84 SEP phones eligible, ENV diagnostic unavailable at sign agreement .5300). MuseTalk and Ditto calibration attempt03 then passed on the same 36 rows with CPU provider/source guards: each has 53/79 SEP phones eligible. MuseTalk natural/TTS centroid norms are 118.4600/119.8225 and adjacent cosine costs .177417/.178629; Ditto values are 9.3290/9.3420 and .173334/.185897. Their ENV diagnostics are unavailable (sign agreement .5566/.5118), and their native scales are not interchangeable. Initial Muse import and Ditto frame-rule guard failures were preserved as engineering receipts; final retry corrected both. No renderer, SyncNet, tests, or GPU forwards ran during calibration.

## Working State
Canonical BM experiment/task/research-map/HANDOFF now record the corrected MFA and three-model CPU calibration. Remote final output is `/dev/shm/native_axis_musetalk_ditto_fit_calibration_20261014_attempt03`; Wav2Lip calibration is `.codex/scratch/native_axis_fit_calibration_20261014_attempt02_retry02_calibration/`. A local compact summary and the Muse blocked-Ditto copy are under `.codex/scratch/`; large final Muse/Ditto arrays remain on remote and are bound by their metrics/receipts. Remote V100 is idle.

## Verification
- MFA retry02 PASS: 40 TextGrids, ten commands return 0, complete source/model/command provenance.
- Wav2Lip CPU calibration PASS; corrected binding receipt binds MFA hashes and exact commands.
- MuseTalk/Ditto calibration PASS: 36 rows/18 pairs/9 speakers, GPU 0%, no renderer/SyncNet/tests.
- ENV diagnostic statuses: Wav2Lip .5300, MuseTalk .5566, Ditto .5118; no ENV direction was promoted.
- No pytest or implementation tests were run.

## Observations
- [decision] Treat all CPU calibration as coordinate qualification only; it is not a factor effect or replacement result.
- [decision] Keep ENV stopped until the protocol uses the frozen TTS-minus-natural local trajectory and passes support/purity gates.
- [result] Native coordinate scales are model-specific: Wav2Lip mel, MuseTalk Whisper sequence, and Ditto HuBERT/ONNX cannot share numeric doses.
- [blocker] Factor intervention adapter and SyncNet/mouth-motion rendering are still pending, beginning with Wav2Lip SEP purity.
- [next_step] Review and execute the bounded Wav2Lip SEP CPU purity packet; launch at most two Wav2Lip renders only after target/off-target gates pass.

## Relations
- continues [[Codex checkpoint - 2026-10-14T02-13-19Z - CPU calibration bound and provenance retry required]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[跨TFG原生输入三因素因果pilot Spec 2026-10-13]]