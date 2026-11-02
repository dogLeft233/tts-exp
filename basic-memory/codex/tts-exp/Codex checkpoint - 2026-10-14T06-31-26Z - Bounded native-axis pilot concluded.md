---
title: Codex checkpoint - 2026-10-14T06-31-26Z - Bounded native-axis pilot concluded
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t06-31-26-z-bounded-native-axis-pilot-concluded
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 06:31:26+00:00
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

# Codex checkpoint - 2026-10-14T06-31-26Z - Bounded native-axis pilot concluded

## Summary

The bounded native-axis factor pilot is concluded: CPU coordinates are reproducible but the small causal checks found no common SyncNet-plus-mouth-motion benefit.

## Story

The user asked for fixed RMS, fixed natural time axis, fixed content, native TFG input interventions, and visual mouth checks while controlling GPU cost. The work therefore qualified coordinates on CPU, then spent a small predeclared budget on two Wav2Lip SEP_RD rows and one MuseTalk plus one Ditto ENV row, each paired with a no-op baseline and original natural PCM scoring.

Wav2Lip SEP_RD CPU qualification reached 0.5 FIT SD actual rival-distance movement on both speakers. The two GPU rows passed source, clock, mux, SyncNet execution, and MediaPipe quality, but the response was heterogeneous: S0044 Delta Cbest -0.080363 and Delta Dmin +0.019251; S0052 Delta Cbest +0.074784 and Delta Dmin -0.092860. Velocity RMS declined on both. SEP_RD is closed without a benefit claim.

MuseTalk and Ditto ENV coordinates passed a NumPy-only purity replay. One common S0053/W0121 utterance was then rendered as ENV/no-op pairs for both models. Engineering and tracking gates passed, but both response gates failed: MuseTalk Delta Cbest -0.252169, Delta Dmin +0.277637, Delta D0 +0.127375; Ditto -0.106188, +0.063300, +0.063300. Motion consistency also failed in both. The packet recommends no further rows.

The earlier Wav2Lip STAB rows remain pre-calibration and already failed the two-speaker motion consistency gate. The bounded task now closes with model-native coordinate evidence but no common causal direction; the parent enhancement-head task remains active for a future mechanism with a new budget.

## Working State

- Canonical experiment, task, research map, and HANDOFF contain the final results and source pointers.
- Final Wav2Lip packet: .codex/scratch/native_axis_injection_pilot_20261013_attempt01/sep_rd_gpu_readback_attempt01/sep_rd_two_row_gate_attempt01.json, SHA 972967be...
- Final ENV packet: .codex/scratch/native_axis_env_gpu_causal_rows_20261014_attempt01/causal_summary.json; run receipt SHA 00937b88...
- NumPy ENV purity packet: .codex/scratch/native_axis_numpy_only_env_preflight_20261014_attempt01/, receipt 0df85b1b..., summary d475346e...
- Blocked dependency attempt is preserved separately at .codex/scratch/native_axis_one_speaker_env_preflight_20261014_attempt01_blocked_receipt.json.
- No old four-center, Dev, Confirm, or broad matrix was reopened. No tests were run.

## Changed Files

- basic-memory/Experiments/native-axis-factor-pilot-20261013.md
- basic-memory/Tasks/native-axis-factor-pilot-20261013.md
- HANDOFF.md
- research evidence map under basic-memory/Research/

## Verification

- bm reindex --search -p tts-exp completed after final document edits (918 observed, 1 indexed, 0 deleted).
- All four ENV renders and both Wav2Lip SEP_RD renders returned zero and passed engineering source/mux/clock/evaluator execution gates.
- Scientific response and mouth-motion gates were evaluated from the final compact receipts and failed.
- Task native-axis-factor-pilot-20261013 was marked concluded; parent enhancement-head task remains active.
- No tests were run.

## Observations

- [result] Model-native coordinate purity did not translate into a common score or mouth-motion benefit.
- [result] SEP_RD was negative/heterogeneous on Wav2Lip; ENV was negative on both MuseTalk and Ditto.
- [result] STAB remains a pre-calibration negative motion result and was not promoted.
- [decision] Close this bounded factor pilot and do not add GPU rows without a new mechanism and explicit budget.
- [blocker] A reusable universal enhancement head is not supported by this factor map; model-specific adapters remain a separate future question.
- [next_step] Return to the parent task with a new mechanism hypothesis or hold the enhancement-head branch; do not reopen these factor arms.

## Relations

- continues [[Codex checkpoint - 2026-10-14T05-03-36Z - SEP_RD two-row response closed]]
- relates_to [[native-axis-factor-pilot-20261013]]