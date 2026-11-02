---
title: Codex checkpoint - 2026-10-14T05-03-36Z - SEP_RD two-row response closed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t05-03-36-z-sep-rd-two-row-response-closed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 05:03:36+00:00
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

# Codex checkpoint - 2026-10-14T05-03-36Z - SEP_RD two-row response closed

## Summary

The two authorized Wav2Lip SEP_RD GPU rows completed with intact source and visual QA but failed both the predeclared SyncNet response gate and the no-regression mouth-velocity gate; the next work is CPU-only MuseTalk/Ditto ENV preflight.

## Story

The pilot holds RMS, natural clock, content, and final scoring PCM fixed while testing separate native coordinates. CPU qualification succeeded for all three frontends, including a pure MuseTalk/Ditto ENV candidate and a distance-directed Wav2Lip SEP_RD coordinate. The bounded Wav2Lip GPU check was deliberately capped at two rows to control cost.

Both Wav2Lip renders used the final CPU operator and completed on the V100. Natural PCM mux, fixed 25 fps clock, source pins, frame counts, SyncNet evaluator execution, and MediaPipe tracking quality all passed. The scientific response was not consistent: S0044 had Delta Cbest -0.080363 and Delta Dmin +0.019251; S0052 had Delta Cbest +0.074784 and Delta Dmin -0.092860. Best lag and official AV offset remained +3/-3. Mouth velocity RMS declined for both rows by -0.00364849 and -0.00821861 IPD/s. The response and motion gates therefore failed, and no additional Wav2Lip SEP_RD rows are authorized.

The result closes one reproducible coordinate without claiming a common factor. A CPU-only MuseTalk/Ditto ENV intervention preflight is now being built, one speaker bounded, before any GPU authorization.

## Working State

- Canonical docs and BM task were updated with the CPU coordinate evidence and the negative/heterogeneous Wav2Lip SEP_RD result.
- Final gate packet: .codex/scratch/native_axis_injection_pilot_20261013_attempt01/sep_rd_gpu_readback_attempt01/sep_rd_two_row_gate_attempt01.json, latest SHA 972967be...
- Wav2Lip render outputs: S0044 SHA 05f84eed...; S0052 SHA 584242025...
- CPU ENV candidate: .codex/scratch/native_axis_env_coordinate_candidate_20261014_attempt03/, 49 available phones per MuseTalk/Ditto model and 28 explicit unavailable.
- Wav2Lip STAB rows remain pre-calibration and fail the two-speaker motion consistency gate; do not relabel or repeat them.
- Retired four-center, Dev, Confirm, and broad audit branches remain closed.

## Changed Files

- basic-memory/Experiments/native-axis-factor-pilot-20261013.md
- basic-memory/Research/research-evidence-map.md
- basic-memory/Tasks/native-axis-factor-pilot-20261013.md
- HANDOFF.md

## Verification

- bm reindex --search -p tts-exp completed successfully after the result and doc updates (917 observed, 3 indexed, 0 deleted).
- Both GPU rows had return code 0; SyncNet and MediaPipe receipts passed their execution/quality checks.
- The scientific response and motion endpoint gates were checked from the final packet and failed as described.
- No tests were run.

## Observations

- [result] A native coordinate can be mechanically valid yet produce heterogeneous SyncNet and motion outcomes; coordinate purity is not a causal benefit.
- [decision] Close Wav2Lip SEP_RD after the two rows and do not spend more GPU budget on it.
- [decision] Keep ENV investigation CPU-first and one-speaker-bounded for MuseTalk/Ditto.
- [blocker] A common cross-TFG score/motion direction remains unconfirmed.
- [next_step] Review the MuseTalk/Ditto ENV CPU preflight and authorize at most one speaker row per model only if source and operator gates pass.

## Relations

- continues [[Codex checkpoint - 2026-10-14T04-35-56Z - SEP_RD CPU gate and bounded render]]
- relates_to [[native-axis-factor-pilot-20261013]]