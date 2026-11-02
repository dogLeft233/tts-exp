---
title: Codex checkpoint - 2026-10-13T16-11-06Z - Factor cohort baseline qualified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t16-11-06-z-factor-cohort-baseline-qualified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 16:11:06+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
---

# Codex checkpoint - 2026-10-13T16-11-06Z - Factor cohort baseline qualified

## Summary
The two-speaker Wav2Lip factor cohort is now qualified with source-clock SyncNet and mouth-motion baselines; the first causal STAB arm is the only next execution.

## Story
The user requested fixed RMS, clock, and content tests for phoneme separation, local segment stability, and local energy or spectral envelope in each TFG native input space, with mouth aperture and velocity checks. The qualification-first protocol was kept small and the scorer contract was repaired before factor work.

S0044 (`BAC009S0044W0266`) has a valid 89-frame Wav2Lip no-op baseline, 84/84 valid SyncNet windows, and 89/89 valid MediaPipe mouth tracks. Its SyncNet Cbest is 6.9585318565, official AV offset -3 frames. S0052 (`BAC009S0052W0386`) is the second factor-cohort speaker after CPU MFA screening at 84.905660 percent known-phone coverage of the non-silence phone region. Its no-op baseline has 100 frames, 95/95 valid SyncNet windows, Cbest 6.3298373222, D0 13.7421436301, official AV offset -3, and 100/100 valid mouth tracks. Both runs preserve original natural PCM for scoring and have post-run idle V100 evidence.

S0045 was also scored as a valid 97-frame Wav2Lip baseline, but its known-phone coverage is 50 percent, so it is retained as scorer qualification and excluded from SEP/STAB. ADR 012 records the rule that `spn`, blank, and silence intervals are left at RMS identity; no phone label is guessed.

## Working State
Canonical protocol, experiment note, task, research map, ADR 012, and HANDOFF record the final factor cohort S0044 plus S0052 and the eligibility rule. Local small readbacks are under `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/artifacts/`. Remote scorer receipts are under `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/`. No factor arm, MuseTalk, Ditto, or ENV run has started.

## Verification
- S0044 no-op hook feature shape `[89,512,1,1]`; SyncNet 84/84; mouth valid 89/89.
- S0045 scorer-only SyncNet 92/92 and mouth valid 97/97; excluded from SEP/STAB by 50 percent known-phone coverage.
- S0052 no-op hook feature shape `[100,512,1,1]`; SyncNet 95/95; mouth valid 100/100; output MP4 SHA `766f3a9f72653c60b2a9ac0ed97185458747e83292bd53c8300e5893eb493d54`; feature SHA `624da6f8c6d6b15b5cb1a7f908b5f47a2ee390ab3cadea5e15957f92d499fc10`.
- CPU MFA TextGrid coverage, endpoint, no-gap, and transcript checks passed for S0044 and S0052; S0052 retains one excluded `spn` interval.
- No implementation tests or pytest were run.

## Observations
- [result] The scorer and motion gates are now valid for both factor-cohort speakers.
- [decision] Keep S0045 out of factor arms; retain it only as engineering/scorer qualification.
- [decision] Run STAB first because it needs verified segment masks but no fitted phone prototypes; do not spend MuseTalk/Ditto budget before a two-speaker Wav2Lip contrast.
- [next_step] Run exactly two Wav2Lip STAB rows on S0044 and S0052 using known-phone masks and the fixed 25 percent residual smoothing operator, then apply the predeclared stop/purity/motion gates.

## Relations
- continues [[Codex checkpoint - 2026-10-13T15-29-43Z - Valid W2L native baseline]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[012-native-axis-known-phone-mask-rule]]