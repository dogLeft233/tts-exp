---
title: Codex checkpoint - 2026-10-12T18-26-44Z - Speed-balanced cross-TFG route and
  minimal trainable check
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t18-26-44-z-speed-balanced-cross-tfg-route-and-minimal-trainable-check
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 18:26:44+00:00
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

# Codex checkpoint - 2026-10-12T18-26-44Z - Speed-balanced cross-TFG route and minimal trainable check

## Summary
The route is now intentionally speed-balanced: keep the executable model-aware method and its evidence boundary, and spend at most one small cross-model trainable check before deciding whether a new hypothesis merits GPU time.

## Story
The active objective is to build a natural-audio enhancement head usable across TFG models while restoring untouched natural PCM in the final video. The latest user direction explicitly reduces audit overhead because the current method may fail and time is limited. I therefore preserved the already completed production route and real trainable evidence instead of reopening the retired nine-parameter/four-center/Confirm matrix. The existing fit_v4 checkpoint executed on Wav2Lip, MuseTalk, and Ditto development renders (180/180 producer rows passed) but was below RMS on every model: -0.039777, -0.029347, and -0.063774 Cbest delta, equal-model -0.044300. A bounded unseen Wav2Lip checkpoint check was near zero and heterogeneous (+0.003285 full; +0.003601 guard20; 5/5 and 4/5 wins/losses). The practical method remains model-aware routing with RMS fail-closed fallback: MuseTalk may use the explicit11 local spectral candidate only when runtime/source/shape checks pass; Wav2Lip, Ditto, unknown models, or provider failure use RMS. Enhanced audio controls generation, while SyncNet scoring and final mux retain untouched natural PCM and original clock.

To answer the speed/reliability tradeoff, the remaining work is limited to one independent six-row true trainable checkpoint check (one unseen speaker x Wav2Lip/MuseTalk/Ditto x candidate/RMS). It must use a new namespace and source contract and must not mutate or relaunch the old runner. If it is informative, record the result; if it is blocked, record the exact source-contract blocker and stop expanding the audit.

## Working State
- Production route config SHA256: 886f75c03977ab7d8ac7a5caecd568421c9a46cbf455d6b9cef87a9bf24fb00.
- Fit_v4 checkpoint SHA256: 3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4.
- Bounded unseen Wav2Lip bundle: .codex/scratch/trainable_w2l_holdout_20261013_attempt01/.
- The LAN V100 server remains running; shutdown and archive branches were cancelled by the user and are parked.
- GPU is otherwise idle; no old runner, center, or Confirm matrix is to be relaunched.
- The worktree is intentionally dirty with many prior experiment artifacts; do not clean or reset it.

## Changed Files
- HANDOFF.md and CONTEXT.md record the current route and trainable boundary.
- scripts/audio_enhancement/model_routing.py and configs/model_aware_routing_v1.json/md define the frozen model-aware dispatch.
- scripts/audio_enhancement/cli.py and README.md document route receipts, fail-closed behavior, original-PCM muxing, and trainable research-only status.
- Local scratch JSON recovery evidence is under .codex/scratch/trainable_w2l_holdout_20261013_attempt01/.

## Verification
- Read-only inspection confirmed fit_v4 report, checkpoint hash, source pins, aggregate metrics, and producer manifest (180/180 PASS; repeat QC not run).
- Bounded unseen Wav2Lip holdout had 20/20 PASS, zero generator/SyncNet return-code failures, natural PCM mux equality, preserved video identity, and ten paired source checks.
- Existing engineering smoke confirmed finite route outputs, 16 kHz/sample-count preservation, lossless original-audio mux, and official SyncNet execution.
- No implementation tests were run, as requested.
- No new GPU run was launched by root in this turn; one delegated minimal runner is preparing a bounded six-row check.

## Observations
- [result] The executable cross-model trainable checkpoint is real but underperforms RMS in the 180-row development aggregate.
- [result] The unseen Wav2Lip trainable result is near zero and heterogeneous, so it cannot justify a universal learned replacement.
- [decision] Keep model-aware routing with RMS fallback as the usable method; keep trainable checkpoints research-only until a materially different objective has positive evidence.
- [decision] Balance reliability and speed by allowing only one tiny new cross-model check, then stop GPU expansion unless it changes the route decision.
- [blocker] The six-row check is not yet complete; its independent runner must finish or report a concrete source-contract blocker.
- [next_step] Read the delegated six-row runner result, record its hashes and per-model deltas if it completes, and then either close this bounded branch or define one materially different objective without reopening retired matrices.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unpushed HEAD: `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.
- HANDOFF.md, scripts/audio_enhancement/model_routing.py, scripts/audio_enhancement/README.md.

## Relations
- continues [[Codex checkpoint - 2026-10-12T18-30-00Z - Cross-TFG trainable boundary confirmed]]
- relates_to [[Developing a cross-TFG natural audio enhancement head]]