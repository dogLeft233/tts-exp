---
title: Codex checkpoint - 2026-10-14T10-58-47Z - Native factor endpoints withheld
  for mask repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t10-58-47-z-native-factor-endpoints-withheld-for-mask-repair
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 10:58:47.542900+00:00
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

# Codex checkpoint - 2026-10-14T10-58-47Z - Native factor endpoints withheld for mask repair

## Summary

The 16 in-cohort MuseTalk/Ditto endpoint rows are complete as engineering readback but withheld as scientific factor evidence after a frozen frame-mask mismatch was found.

## Story

The bounded native-axis pilot rendered fixed-RMS, fixed-clock, same-content no-op/SEP/STAB/ENV rows for S0044 and S0052 in MuseTalk and Ditto. GPT-6 Luna/max scoring produced official SyncNet and MediaPipe readback using raw-original natural PCM, exact 25 fps, and 100 percent tracking. The round-mask screening showed one apparent Ditto S0044 ENV pass and no replicated cross-speaker/model factor.

A reliability audit found the intervention/materializer used rounded TextGrid boundaries, while the FIT calibration owns native frame j when j/25 is in [xmin,xmax). The mismatch is material: S0044 SEP 57/57 frames with XOR 4, S0044 STAB 57/54 XOR 5, S0052 SEP 61/57 XOR 4, S0052 STAB 64/61 XOR 7. Current rows remain engineering/exploratory and are not admitted as frozen scientific cells. No new GPU render was started.

## Working State

- Experiment and map: basic-memory/Experiments/native-axis-factor-pilot-20261013.md and basic-memory/Research/Research map.
- Engineering score packet: .codex/scratch/native_axis_pilot_score_20261014_attempt02/; compact summary SHA 07B16AA32DEBBA1B3E0ED31F890720974DB541B7D131484EC3C16E6F4C057FDA, raw-array index SHA 6883a5ace6caade9d355ebdd319ced8f12BA71a7AE8f6bf9372a8b1c50efb7bf.
- Coverage manifests: attempt02 records endpoint rows; attempt03 downgrades all 16 intervention cells to EXPLORATORY_MASK_MISMATCH and admits zero scientific cells.
- Remote server is idle; no active renderer/scorer process and GPU utilization is zero. Existing media and score artifacts remain unchanged.

## Verification

All 16 engineering score rows passed source/raw-PCM/clock/evaluator/MediaPipe checks. No tests were run. The CPU mask audit is read-only and the next corrected materialization remains pending. Documentation, Task, HANDOFF, and Research map were updated and Basic Memory was reindexed.

## Observations

- [result] Round-mask endpoint readout is not a canonical factor result.
- [decision] Use calibration ownership j/25 >= xmin and j/25 < xmax; bind per-factor mask hashes and actual float32 arrays before any rerender.
- [decision] Keep velocity/aperture descriptive; do not equate a velocity increase/decrease with mouth accuracy.
- [blocker] Corrected CPU rematerialization and a budget decision for any minimal rerender remain.
- [next_step] Run the CPU-only canonical mask rematerialization and compare corrected arrays before authorizing at most one small factor rerender pair.

## Relations

- continues [[Codex checkpoint - 2026-10-14T10-04-23Z - In-cohort native factors rendered pending endpoints]]
- relates_to [[native-axis-factor-pilot-20261013]]