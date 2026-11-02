---
title: Codex checkpoint - 2026-10-13T17-03-05Z - Bounded Wav2Lip STAB result
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t17-03-05-z-bounded-wav2-lip-stab-result
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 17:03:05+00:00
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

# Codex checkpoint - 2026-10-13T17-03-05Z - Bounded Wav2Lip STAB result

## Summary
The first bounded causal factor arm is complete: Wav2Lip STAB ran on the frozen S0044/S0052 cohort with valid natural-PCM SyncNet and mouth-motion receipts, but it does not meet the two-speaker or motion-consistency gate for cross-TFG expansion.

## Story
The user asked for fixed RMS, clock, and content tests that isolate phoneme separation, local segment stability, and local energy or spectral envelope in each TFG native input space, while checking whether SyncNet changes track mouth motion. To control cost, the frozen pilot starts with CPU qualification and a two-speaker Wav2Lip STAB arm.

Both STAB renders passed return code, frame count, exact post-intervention mel delivery, and algebraic purity. S0044 improved Cbest by `+0.038277` while D0 and Dmin decreased and B decreased, with lag unchanged. S0052 changed Cbest by only `+0.002663`; Dmin worsened and B rose, so the small score change is a B-linked response. Mouth velocity RMS decreased for both speakers (`-0.009740`, `-0.017112`), without directional motion support. The FIT reference gate remains UNQUALIFIED because the pilot has no frozen FIT SD calibration receipt; a CPU-only attempt04 now binds the exact render adapter SHA and preserves the stale-source attempt03 for provenance.

## Working State

Canonical BM experiment/task/map and HANDOFF were updated with the partial STAB conclusion. Full pilot remains incomplete: SEP, ENV, MuseTalk, and Ditto have not run; S0045 remains scorer-only. The updated remote operator manifest is `/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/stab_w2l_operator_manifest.json` with SHA prefix `db4d69e6`; local paired readback is `.codex/scratch/native_axis_injection_pilot_20261013_attempt01/local_readback_attempt01/`. The V100 is idle at 5 MiB/0 percent utilization.

## Verification

- CPU attempt04 for S0044/S0052: algebraic PASS, known-phone-only changes, excluded intervals unchanged, residual fraction `0.75`, centroid shifts below `5e-6`; exact render adapter SHA bound.
- GPU render receipts for both rows: return code 0, 89/100 frames, exact post-STAB chunks received.
- Source-clock SyncNet audits: S0044 84/84 and S0052 95/95 valid queries, no invalid windows, lag/offset unchanged.
- MediaPipe motion receipts: 89/89 and 100/100 valid tracks.
- Remote GPU idle after run.
- No pytest or implementation tests were run.

## Observations

- [result] S0044 STAB is a distance-improving single-speaker response, not a shared factor result.
- [result] S0052 STAB is near-null and B-linked with worse Dmin.
- [decision] Do not expand STAB to a second TFG or start SEP/ENV GPU rows until the low-cost FIT calibration question is resolved.
- [decision] Keep SyncNet, mouth motion, and FIT purity as separate evidence gates; do not call the score change a mouth-shape improvement.
- [blocker] No frozen FIT SD calibration receipt is available, so the full coordinate purity gate remains unqualified.
- [next_step] Determine whether an auditable FIT calibration can be derived CPU-only; then decide whether any SEP/ENV or MuseTalk/Ditto screening is worth a new bounded budget.

## Relations

- continues [[Codex checkpoint - 2026-10-13T16-37-15Z - STAB qualification repaired lineage - 2]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[012-native-axis-known-phone-mask-rule]]