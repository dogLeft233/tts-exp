---
title: Codex checkpoint - 2026-10-12T18-35-50Z - Minimal cross-TFG probe closed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t18-35-50-z-minimal-cross-tfg-probe-closed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 18:35:50+00:00
username: laptop-e133gsle\\dogle
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

# Codex checkpoint - 2026-10-12T18-35-50Z - Minimal cross-TFG probe closed

## Summary
The speed-balanced six-row trainable probe is closed: Wav2Lip and MuseTalk both lose to RMS on the shared unseen sample, while Ditto is blocked by an adapter-interface mismatch; the production route remains model-aware with RMS fallback.

## Story
The user asked to reduce audit overhead because the current learned scheme may not work. I allowed one independent one-speaker cross-model probe and stopped once it answered the route question. The fit_v2 checkpoint was applied to candidate/RMS controls for Wav2Lip, MuseTalk, and Ditto. Wav2Lip and MuseTalk completed four rows with original natural PCM, 25 fps, and SyncNet. Trainable-minus-RMS Cbest was -0.060789/-0.166985 (full/guard20) for Wav2Lip and -0.095057/-0.139113 for MuseTalk. Ditto never rendered: its repaired adapter is a Stage-A block CLI but the new wrapper passes a silent-video CLI contract, so argparse rejected missing Stage-A arguments. This is an implementation blocker, not a Ditto scientific result.

Together with the prior 180-row fit_v4 development aggregate (all three models below RMS) and the bounded 20-row unseen Wav2Lip check (near-zero and heterogeneous), the evidence does not support a universal learned replacement. The usable delivery stays the frozen model-aware route: MuseTalk explicit11 only when its checks pass; Wav2Lip, Ditto, unknown, and provider failure use RMS; final scoring and mux restore untouched natural PCM and original clock.

## Working State
- Minimal probe plan SHA256: 8f4ee6cc84971807baf7687d79fd5cf9983bf1491fea21d4eb15639343aaa1f.
- Raw plan SHA256: 4a8340da03adc45abb5dda83046b9edb8faa24b61b0457c5840cf1ba8d2bba90.
- Planned rows SHA256: 8fc56b39bc22296bb0c0fffbaddf9354c6e54a79dde91c27bc8ce258eb7d73e9.
- Local source/report: .codex/scratch/trainable_cross_tfg_oneid_attempt01/, including REPORT.md.
- Remote namespaces exited; no cross-TFG runner remains and NVIDIA compute apps are empty.
- Server stays running; retired old matrix, shutdown/archive branch, and Confirm remain parked.

## Changed Files
- HANDOFF.md now records the six-row probe, hashes, per-model deltas, and Ditto interface blocker.
- Basic Memory active task and experiment note record the same boundary.

## Verification
- Read back four PASS rows and per-row provenance: 59135 samples at 16 kHz, mux matched natural canonical PCM, and video stream identity was verified.
- Read back Ditto failure rows: render FAILED, mux NOT_STARTED, no score.
- Confirmed both bounded namespaces exited and GPU idle.
- No implementation tests were run.

## Observations
- [result] The trainable checkpoint is below RMS on both models that executed in the minimal shared-sample probe.
- [blocker] Ditto needs a separately frozen adapter contract before a trainable score can be obtained.
- [decision] Stop GPU expansion and deliver model-aware routing with RMS fallback; do not claim three-model trainable validation.
- [decision] Any future GPU run must answer a new scientific question with an explicit small budget.
- [next_step] Maintain the documented route and wait for a concrete new hypothesis before scheduling another renderer or holdout.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unpushed HEAD 5688784dd5c6c1272074f0c4b6c5ff79f117c907.
- HANDOFF.md
- scripts/audio_enhancement/model_routing.py
- scripts/audio_enhancement/README.md

## Relations
- continues [[Codex checkpoint - 2026-10-12T18-26-44Z - Speed-balanced cross-TFG route and minimal trainable check]]
- relates_to [[Minimal cross-TFG trainable checkpoint probe, one unseen speaker]]
- relates_to [[Developing a cross-TFG natural audio enhancement head]]