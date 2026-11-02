---
title: Codex checkpoint - 2026-10-12T18-12-00Z - Trainable holdout closed and route
  boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t18-12-00-z-trainable-holdout-closed-and-route-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 18:12:00+00:00
username: laptop-e133gsle\dogle
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

# Codex checkpoint - 2026-10-12T18-12-00Z - Trainable holdout closed and route boundary

## Summary

The single bounded trainable-checkpoint holdout is complete; it is near zero and heterogeneous, so the production boundary remains model-aware routing with RMS fallback.

## Story

The user asked to reduce audit overhead while keeping enough reliability to make progress. I preserved the existing 10-speaker three-model holdout and executable route, then allowed exactly one additional GPU action to close the fit-only checkpoint question. The run used Wav2Lip only, ten unseen speakers, candidate versus RMS, original natural PCM and clock, lossless mux, and official SyncNet. It completed 20/20 paired rows and exited cleanly. No old center, Confirm, shutdown, or archive workflow was started.

The trainable checkpoint result is full mean delta Cbest +0.003285, median +0.028225, wins/ties/losses 5/0/5. Guard20 excludes ineligible S0048 and gives n=9, mean +0.003601, median -0.010525, wins/ties/losses 4/0/5. This is close to zero with mixed per-speaker signs. It does not justify productionizing the checkpoint or launching a larger audit matrix.

## Working State

Checkpoint: runs/cross_tfg_audio_head_20261003/fit_v2_actual_evidence_attempt01/shared_audio_head.pt
Checkpoint SHA256: 768c5b1157139281b4d62d73ed6be19b9f69e37401e9298395f469dc4501c943
Remote run root: /dev/shm/fit40_queue_remote_package_attempt02/project/trainable_w2l_holdout_20261013_attempt01
Runner PID 2204334, start_ticks 31235573, returncode 0, GPU idle after exit.
Plan SHA256: 764c394475b383fd9bb1d3109890181165509152fbf3b4f5aea82c23e127ea5d
Results SHA256: 152e4aea1b8f1ee49ce4ec7b26c5d385a3831dc62921f456433b2488bdef21b5
Corrected compact stats SHA256: 98f5e0c09b3acb4eed0e93f2ce6d8edaa90f50cfc1e5b5699eeb90c6afd4a229
Local JSON bundle: .codex/scratch/trainable_w2l_holdout_20261013_attempt01/

The usable route remains:
- MuseTalk explicit11 local spectral shape when runtime, source pins, shape digest, and finite checks pass.
- Wav2Lip, Ditto, unknown, or provider failure use rms_fixed.
- Final output replaces generated audio with untouched natural PCM.

## Changed Files

- HANDOFF.md updated with the bounded holdout and current boundary.
- Basic Memory active task appended with results and decision.
- Basic Memory experiment note created: Bounded unseen-speaker validation of the trainable enhancement checkpoint.
- Existing route files and receipts remain as recorded in the prior checkpoint.

## Verification

- The bounded remote run passed all 20 paired rows.
- All 20 generator and SyncNet return codes were zero.
- All 20 mux audio streams matched natural PCM; video identity was preserved.
- Ten paired source PCM/video checks passed.
- No implementation tests were run.
- No old center or Confirm work was started.

## Observations

- [result] The trainable checkpoint is not validated as a cross-TFG production head.
- [decision] Keep the speed-balanced model-aware route with RMS fallback and stop expanding the audit matrix.
- [decision] Further GPU work requires a new concrete scientific hypothesis and an explicit budget.
- [next_step] Use the existing route, method note, HANDOFF, receipts, and local JSON bundle as the current delivery boundary.

## Relations

- continues [[Codex checkpoint - 2026-10-12T17-47-05Z - Balanced route closure and trainable holdout checkpoint]]
- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Bounded unseen-speaker validation of the trainable enhancement checkpoint]]