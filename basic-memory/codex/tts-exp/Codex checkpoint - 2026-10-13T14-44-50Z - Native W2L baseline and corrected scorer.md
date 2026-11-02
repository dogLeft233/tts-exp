---
title: Codex checkpoint - 2026-10-13T14-44-50Z - Native W2L baseline and corrected
  scorer
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t14-44-50-z-native-w2-l-baseline-and-corrected-scorer
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 14:44:50+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
---

# Codex checkpoint - 2026-10-13T14-44-50Z - Native W2L baseline and corrected scorer

## Summary
The bounded native-axis pilot has moved from protocol freeze to a real Wav2Lip no-op frontend run; scientific scoring is still pending a corrected 89-frame audio-video binding.

## Story
The user requires four causal arms in each TFG native input space while fixing RMS, clock, and content, followed by mouth aperture and velocity checks. To control time cost, the root agent froze a qualification-first pilot in the canonical tts-exp Basic Memory structure: two unseen speakers, three TFGs, four arms, with a two-of-three-model expansion rule. The pilot keeps SyncNet evidence separate from motion evidence and does not reopen the retired large center or Dev/Confirm branches.

A Luna/max subagent selected clips BAC009S0044W0266 and BAC009S0045W0283 and created a fresh remote scratch namespace. Wav2Lip native no-op capture on S0044 completed with 89 frames and a deterministic [89,512,1,1] feature array. The first two direct SyncNet attempts were integration failures: the renderer's AAC duration caused 91 frames, and a video-only tracker input failed because the tracker requires an audio stream. These are preserved as engineering receipts, not scientific negative results. The active repair creates an elementary-stream-identical 89-frame video with exactly 56960 samples of silent tracker audio, then obtains the native crop and scores the original natural PCM window.

Historical native archives are FIT-only and have no video or SyncNet evidence. The Ditto archive is now verified as an actual CPU ONNX collection with 56 source-condition rows, but it remains historical FIT-only and cannot be reported as unseen holdout evidence.

## Working State
Durable: the canonical protocol is in `basic-memory/Experiments/`, the indexed experiment note is `native-axis-factor-pilot-20261013`, the active task is `tts-exp/tasks/native-axis-factor-pilot-20261013`, and the research map links the pilot. The protocol fixes 16 kHz mono, RMS 0.0376838172675746, exact natural clock, transcript and MFA boundaries, original natural PCM for scoring, model-native Wav2Lip mel, MuseTalk Whisper sequence, and Ditto HuBERT sequence.

Fragile: the working tree is already heavily dirty from earlier research; this turn did not intentionally modify tracked source. Remote scratch is `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01`. The active subagent is repairing attempt04; no MuseTalk, Ditto, factor arm, or mouth-metric run has started.

## Changed Files
- Canonical Basic Memory protocol, task, research map, and HANDOFF pilot section were created or revised before this checkpoint.
- No new repository source change was made by root in this turn.

## Verification
- Read the bm-checkpoint instructions and the existing same-session checkpoint.
- Confirmed repository root, branch, commit, identity, and current working directory directly.
- Wav2Lip no-op hook receipt: adapter SHA `dcd536eacb38b66b8ff91333d8f2e01507e508b572d9b2dc52920bfb4b95b578`; generator PID 2251774 rc 0; video SHA `cc663e33397b147f39dfc0389ef89d18a49db9a371c380c06875a8d041870500`; feature shape `[89,512,1,1]`; determinism guard passed.
- Attempt01 SyncNet direct full-frame scoring failed with a 51200-to-512 linear shape mismatch and is retained as an integration failure.
- Attempt02 exposed the 91-frame AAC-duration mismatch; attempt03 exposed the tracker no-audio contract. Attempt04 is not yet complete.
- No pytest or implementation test was run.

## Observations
- [result] The native Wav2Lip no-op frontend can execute deterministically on the selected unseen clip and produces the expected 89-frame feature tensor.
- [decision] Keep the first scientific execution bounded and qualify the scorer contract before any factor arm or cross-model expansion.
- [decision] Treat SyncNet C/B/D and mouth motion as separate endpoints; a SyncNet increase alone is not a mouth-truth claim.
- [blocker] The corrected 89-frame tracker and SyncNet receipt is not yet available; the current preliminary score is unqualified.
- [result] Ditto historical feature collection is real CPU ONNX data, but it is FIT-only with no video or SyncNet and cannot substitute for the unseen pilot.
- [next_step] Finish attempt04 with 89-frame silent tracker audio, verify the complete scorer receipt and post-run process state, then run only the next predeclared bounded baseline or qualification row.

## Relations
- continues [[Codex checkpoint - 2026-10-13T04-02-19Z - Native axis factor pilot planned]]
- relates_to [[native-axis-factor-pilot-20261013]]