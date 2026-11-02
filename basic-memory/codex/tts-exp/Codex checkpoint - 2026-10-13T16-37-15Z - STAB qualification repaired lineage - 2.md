---
title: Codex checkpoint - 2026-10-13T16-37-15Z - STAB qualification repaired lineage
  - 2
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t16-37-15-z-stab-qualification-repaired-lineage-2
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 16:37:15+00:00
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

# Codex checkpoint - 2026-10-13T16-37-15Z - STAB qualification repaired lineage - 2

## Summary
The bounded native-axis pilot remains in qualification: both Wav2Lip factor-cohort baselines are valid, while STAB CPU qualification is being repaired for exact source-byte lineage before any GPU factor render.

## Story
The user asked for a small, reliable test of phoneme separation, local segment stability, and local energy or spectral envelope in each TFG native input space, with fixed RMS, clock, and content plus independent mouth-motion descriptors. The frozen protocol limits the first scientific budget and requires CPU/native purity checks before rendering. The current route starts with STAB on Wav2Lip because it needs only known-phone masks and avoids fitting phone prototypes.

S0044 has a valid 89-frame no-op Wav2Lip baseline with 84/84 valid SyncNet windows and 89/89 valid mouth tracks. S0052 has a valid 100-frame baseline with 95/95 valid SyncNet windows and 100/100 valid mouth tracks; S0045 remains scorer-only because its known-phone coverage is 50 percent. The final factor cohort is S0044 plus S0052, with `spn`, blank, and silence excluded from SEP/STAB.

The STAB adapter is isolated in machine-local scratch and passed py_compile. Its S0044 CPU qualification changed only known-phone mel columns, retained 75 percent residuals, reduced adjacent-cosine cost, and left excluded intervals unchanged. The receipt currently exposes an aggregate centroid-restored flag that conflicts with per-segment shifts of about 1e-6 and status PASS; this is being clarified rather than silently accepted. The first S0052 CPU attempt used an altered WAV at the canonical path: its actual file hash `a9e677...` and PCM hash `d936063d...` do not match the authoritative no-op hashes `44124149...` and `17a6809b...`. The operator is reproducing the pinned RMS recipe to a fresh path and will require exact file and PCM matches before qualification.

## Working State

Durable protocol and task state is in `basic-memory/Experiments/跨TFG原生输入三因素因果pilot Spec 2026-10-13.md`, `basic-memory/Experiments/native-axis-factor-pilot-20261013.md`, `basic-memory/Tasks/native-axis-factor-pilot-20261013.md`, ADR 012, the research map, and `HANDOFF.md`. Remote namespace is `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01`; the V100 on `wjj@192.168.50.2:22` is currently idle. The current scientific status is qualification in progress; no SEP, ENV, MuseTalk, Ditto, or factor GPU render has started.

Machine-local STAB code and receipts are under `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/artifacts/` and the remote qualification namespace. These scratch files are not treated as durable source changes.

## Verification

- Read Startup Router, experiment instructions, research map, active task, HANDOFF, and prior same-chat checkpoint.
- Read authoritative S0044/S0052 baseline and STAB CPU receipts over SSH.
- Confirmed remote GPU idle: V100 5 MiB, 0 percent utilization, no compute applications.
- Confirmed S0044 STAB CPU receipt status PASS, known-only changes and excluded-interval identity; centroid purity wording requires repair.
- Confirmed S0052 STAB CPU receipt is an engineering failure caused by altered input bytes, not a scientific negative.
- No pytest or implementation tests were run.

## Observations

- [result] The existing two-speaker Wav2Lip baseline cohort and scorer/motion gates remain valid.
- [decision] Do not start a factor render until S0052 is regenerated to the authoritative file and PCM hashes and S0044 centroid-restoration reporting is reconciled.
- [decision] Keep the budget at exactly two Wav2Lip STAB rows before deciding whether any MuseTalk, Ditto, SEP, or ENV spend is justified.
- [blocker] S0052 current canonical-path WAV is not the authoritative no-op byte stream; re-generation and hash verification are required.
- [next_step] Finish exact S0052 source regeneration and CPU qualification, repair the S0044 purity receipt, then run exactly the predeclared two-row Wav2Lip STAB contrast only if both CPU gates pass.

## Relations

- continues [[Codex checkpoint - 2026-10-13T16-11-06Z - Factor cohort baseline qualified]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[012-native-axis-known-phone-mask-rule]]