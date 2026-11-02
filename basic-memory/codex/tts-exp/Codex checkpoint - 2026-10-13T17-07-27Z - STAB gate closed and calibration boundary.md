---
title: Codex checkpoint - 2026-10-13T17-07-27Z - STAB gate closed and calibration
  boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t17-07-27-z-stab-gate-closed-and-calibration-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 17:07:27+00:00
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

# Codex checkpoint - 2026-10-13T17-07-27Z - STAB gate closed and calibration boundary

## Summary
The Wav2Lip STAB contrast is archived as a bounded partial result and the next GPU spend is stopped because the two-speaker and motion gates fail and no frozen FIT SD calibration exists.

## Story
The fixed-RMS native-axis pilot ran the predeclared STAB intervention on S0044 and S0052, preserving natural PCM and clock for scoring. S0044 showed a real distance-improving SyncNet pattern; S0052 was near-null, B-linked, and had worse Dmin. Mouth velocity RMS fell for both, with no consistent motion support. Exact adapter binding was closed with CPU attempt04 and render SHA `3bdd7aeb…`; failed preflight/temp/ffprobe engineering receipts remain preserved.

A review of available FIT-only archives found no frozen STAB SD estimator, speaker weighting, clock/exclusion binding, ddof choice, or matching source pins. A CPU recalculation would require a new protocol and cannot retroactively qualify this run. The current result is therefore retained as bounded `NO_COORDINATE`/`SCORE_RESPONSE` evidence; no second-TFG STAB, SEP/ENV GPU, or larger cohort is authorized.

## Working State
BM experiment, task, research map, and `HANDOFF.md` now record the partial conclusion and calibration boundary. Remote operator manifest: `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/stab_w2l_operator_manifest.json`, SHA prefix `db4d69e6`; local paired readback under `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/local_readback_attempt01/`. V100 idle.

## Verification

- S0044/S0052 CPU attempt04 algebraic purity PASS and exact adapter binding.
- Both GPU renders PASS; both source-clock SyncNet and MediaPipe motion receipts PASS.
- S0044 Cbest delta `+0.038277`, D0/Dmin deltas `-0.470227/-0.263139`, B delta `-0.224862`.
- S0052 Cbest delta `+0.002663`, D0/Dmin deltas `-0.038471/+0.031153`, B delta `+0.033815`.
- Mouth velocity RMS deltas `-0.009740` and `-0.017112`.
- No pytest or implementation tests were run.

## Observations

- [decision] Treat STAB as a bounded partial result; do not upgrade it to a cross-TFG factor.
- [decision] Any FIT calibration is a new CPU-only protocol, not a retroactive qualification.
- [next_step] If research continues, freeze a separate FIT calibration protocol before any new GPU factor or TFG run.

## Relations

- continues [[Codex checkpoint - 2026-10-13T17-03-05Z - Bounded Wav2Lip STAB result]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[012-native-axis-known-phone-mask-rule]]