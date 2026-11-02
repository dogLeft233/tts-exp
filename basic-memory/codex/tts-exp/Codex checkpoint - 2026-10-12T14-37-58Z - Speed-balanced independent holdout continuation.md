---
title: Codex checkpoint - 2026-10-12T14-37-58Z - Speed-balanced independent holdout
  continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t14-37-58-z-speed-balanced-independent-holdout-continuation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 14:37:58+00:00
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

# Codex checkpoint - 2026-10-12T14-37-58Z - Speed-balanced independent holdout continuation

## Summary

After two complete explicit11 initial360 sweeps, the work is bounded to one independent holdout of a fixed model-aware routing head before any further GPU sweep.

## Story

The project is testing a portable natural-audio enhancement head that can improve TFG lip-sync while the final scored audio remains the original natural PCM and original clock. The frozen nine-parameter route completed and was rejected by its admission gates, so this turn evaluated two explicit eleven-parameter local-power/spectral variants. The first zero-envelope run improved equal-model full candidate-minus-RMS by +0.008278 but was -0.010243 on guard20, with Ditto negative. The second anti-symmetric transition-envelope variant (+0.25 dB for positive flux and -0.25 dB for negative flux) improved full to +0.014598 and guard20 to -0.002323; Ditto guard20 improved from -0.029825 to -0.012072, but Wav2Lip guard20 became -0.024014. Neither variant is a universal win.

To balance reliability and time, the next action is a single disjoint holdout rather than another broad audit or an immediate third GPU sweep. A fixed routing hypothesis is frozen before seeing holdout scores: Wav2Lip uses the zero-envelope candidate, MuseTalk uses the anti-symmetric envelope candidate, and Ditto uses RMS fallback. The delegated Luna/max agent has identified ten holdout clips from disjoint speakers under the remote fresh AISHELL directory and is preparing one 60-row run (three models 脳 ten candidate/RMS pairs) with the original 25 fps clock, natural PCM scoring audio, static portrait, model checkpoints, and SyncNet pins.

## Working State

Durable:
- Original natural PCM, original clock, 40-sample development cohort, model checkpoints, and scoring gates remain frozen.
- Nine-parameter route is terminally NOT_ADMITTED and must not be rerun.
- Explicit11 run 1 and run 2 are complete CPU-recovered results; no GPU rerun is needed for their serialization tail.
- Routing holdout is the only current science execution authorized before deciding whether to spend another GPU sweep.

Machine-local / fragile:
- Remote queue/source root: /dev/shm/fit40_queue_remote_package_attempt02/project.
- Run 1 output root: /dev/shm/native11_fallback_initial360_attempt04_final_attempt02_byteshex.
- Run 2 output root: /dev/shm/native11_env_pos025_neg025_initial360_attempt02.
- Independent holdout source directory: /home/wjj/tts-audio/tts-exp/data/fresh_aishell_disjoint_ref_target_20260929/audio.
- The current experiment agent is /root/native11_fastpath; it is preparing the holdout and has not reported a new terminal result yet.
- Parked archive/shutdown work remains out of scope.

## Changed Files

- HANDOFF.md: added the two explicit11 terminal result sections and current routing conclusion.
- Basic Memory active task: recorded both run aggregates and the independent holdout decision.
- Basic Memory decision note: `tts-exp/docs/adr/explicit11-routing-before-another-gpu-sweep`.

## Verification

- Run 1 complete: 120/120 seed and 360/360 model rows, CPU finalization; results SHA `94ac3ad1694ec09bbe7139c1435bb06c397a9e694e51acfb3c97f0f3a3e9eac6`; equal-model full +0.0082782785, guard20 -0.0102425893.
- Run 2 complete: 120/120 seed and 360/360 model rows, CPU finalization; results SHA `d34c6a08c2cb6a6a9a440751a31a875723b1953080e14bcd24fd50e0fc946a55`; equal-model full +0.0145976702, guard20 -0.0023231586.
- Run 2 Ditto guard20 delta improved to -0.0120717764; Wav2Lip guard20 was -0.0240141988.
- No implementation tests were run, per project instruction and user request.

## Observations

- [result] Two explicit11 variants provide a plausible mechanism signal but do not yet establish a cross-model universal head.
- [decision] Use one fixed, pre-registered model-aware routing holdout to control GPU/time cost.
- [blocker] Holdout terminal metrics are not available yet.
- [next_step] Let /root/native11_fastpath launch and finish the 60-row independent holdout, then read its aggregate and make one bounded go/no-go decision.

## Relations

- continues [[Codex checkpoint - 2026-10-12T10-49-43Z - Speed-balanced explicit11 initial360 continuation]]
- relates_to [[Explicit11 routing before another GPU sweep]]