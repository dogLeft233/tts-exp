---
title: Codex checkpoint - 2026-10-14T04-35-56Z - SEP_RD CPU gate and bounded render
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t04-35-56-z-sep-rd-cpu-gate-and-bounded-render
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 04:35:56+00:00
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

# Codex checkpoint - 2026-10-14T04-35-56Z - SEP_RD CPU gate and bounded render

## Summary

The bounded native-axis pilot now has a pure CPU coordinate for Wav2Lip phoneme separation and pure-by-construction ENV candidates for MuseTalk and Ditto; only two Wav2Lip SEP_RD GPU rows are authorized next.

## Story

The project is testing whether the shared directions already observed in TTS-versus-natural comparisons can be expressed as reproducible coordinates inside each TFG's own renderer input while holding RMS, natural time axis, content, and final natural PCM fixed. The user asked for reliability and speed to be balanced, so qualification stays CPU-first and the GPU budget is capped before any cross-model expansion.

The corrected speaker-partitioned MFA retry02 and CPU native calibrations are complete for Wav2Lip, MuseTalk, and Ditto on the 9-speaker FIT subset. MuseTalk and Ditto now each have a separate ENV coordinate candidate with 49 available phones and 28 explicit unavailable phones; the paired natural-to-TTS log-norm operator is pure by construction and has zero excluded-frame change. The earlier same-sign ENV diagnostics remain unavailable and are not being treated as causal evidence.

The first Wav2Lip prototype-direction SEP packet moved the actual centroid-to-rival distance by only 0.104493/0.237511 FIT SD, so it is retained as a failed coordinate attempt. The separately named SEP_RD attempt04 then passed CPU-only on S0044 and S0052 with 0.500000 SD actual rival-distance movement on both rows, unchanged source/RMS/PCM/clock and masks, off-target means inside the two-SD limits, and exact renderer chunk counts 89/100. No GPU, SyncNet, or mouth-motion result exists yet.

## Working State

- The active BM task is native-axis-factor-pilot-20261013; the canonical experiment note and research map were updated with the ENV candidate and SEP_RD result.
- Existing Wav2Lip STAB renders remain pre-calibration and fail the two-speaker motion consistency gate; they are not being relabeled.
- The old four-center, Dev, Confirm, and broad audit branches remain closed.
- The next authorized action is at most two Wav2Lip SEP_RD GPU rows, S0044 and S0052, using existing no-op baselines, natural PCM, natural clock, SyncNet C/B/D scoring, and MediaPipe aperture/velocity. MuseTalk/Ditto renders require a later explicit gate review.
- CPU-only artifact roots: .codex/scratch/native_axis_env_coordinate_candidate_20261014_attempt03/ and .codex/scratch/native_axis_injection_pilot_20261013_attempt01/native_sep_rd_cpu_qualification_attempt04/.
- Remote experiment root is /dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01; no credentials are recorded here.

## Changed Files

- basic-memory/Experiments/native-axis-factor-pilot-20261013.md
- basic-memory/Research/research-evidence-map.md
- basic-memory/Tasks/native-axis-factor-pilot-20261013.md
- HANDOFF.md

## Verification

- Basic Memory Startup Router and experiment instructions were read before note/document updates.
- bm reindex --search -p tts-exp completed successfully (915 observed, 3 indexed, 0 deleted).
- CPU calibration and SEP_RD receipts were supplied by the Luna/max subagents; no tests were run and no GPU render was started in this checkpoint turn.
- Repository was dirty before this turn with many pre-existing working-tree changes; this checkpoint does not claim ownership of unrelated files.

## Observations

- [result] All three TFG native frontends have model-specific CPU calibration; no common numeric coordinate is assumed.
- [result] MuseTalk/Ditto ENV candidates are available only where paired support reaches the frozen speaker threshold; unavailable phones remain explicit.
- [decision] Treat SEP_RD as the only current GPU candidate and keep its budget to two rows.
- [decision] Keep STAB, MuseTalk/Ditto renders, and retired broad matrices closed until the bounded response and motion gate is read.
- [blocker] SyncNet response and mouth-motion correspondence for SEP_RD are still unmeasured.
- [next_step] Run the two Wav2Lip SEP_RD GPU rows and score both SyncNet and mouth-motion endpoints.

## Relations

- continues [[Codex checkpoint - 2026-10-14T04-06-47Z - Cross-TFG CPU coordinates qualified]]
- relates_to [[native-axis-factor-pilot-20261013]]