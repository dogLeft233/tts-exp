---
title: Bounded unseen-speaker validation of the trainable enhancement checkpoint
type: experiment
permalink: tts-exp/experiments/bounded-unseen-speaker-validation-of-the-trainable-enhancement-checkpoint
status: concluded
project: tts-exp
tags:
- experiment
- tts
- tfg
---

# Bounded unseen-speaker validation of the trainable enhancement checkpoint

## Summary

One pre-budgeted Wav2Lip-only unseen-speaker check was completed for the FIT_ONLY_COMPLETE checkpoint. The effect is near zero and heterogeneous, so the checkpoint is not promoted into the production route.

## Protocol

Only one runner was started on 192.168.50.2: PID 2204334, start_ticks 31235573, returncode 0. Model ID was wav2lip_gan_bac9a81e. Ten unseen AISHELL speakers each received candidate and RMS conditions, 20 paired rows total. The original natural PCM, 25 fps, and original clock were preserved; generated video was muxed back to natural PCM before SyncNet scoring. No old session, center matrix, or Confirm was started.

Checkpoint SHA256: 768c5b1157139281b4d62d73ed6be19b9f69e37401e9298395f469dc4501c943.
Plan SHA256: 764c394475b383fd9bb1d3109890181165509152fbf3b4f5aea82c23e127ea5d.
Results SHA256: 152e4aea1b8f1ee49ce4ec7b26c5d385a3831dc62921f456433b2488bdef21b5.
Corrected compact stats SHA256: 98f5e0c09b3acb4eed0e93f2ce6d8edaa90f50cfc1e5b5699eeb90c6afd4a229.

## Results

Full Cbest candidate minus RMS over n=10: mean +0.003285, median +0.028225, wins/ties/losses 5/0/5. Guard20 excludes ineligible S0048, n=9: mean +0.003601, median -0.010525, wins/ties/losses 4/0/5. All 20 rows passed generation and SyncNet returncode checks; mux audio matched natural PCM and video identity was preserved.

## Conclusion and boundary

This small unseen-speaker Wav2Lip check does not show a stable trainable-checkpoint gain and does not establish cross-TFG generalization. The checkpoint remains research-only. The usable delivery boundary is the frozen model-aware route: MuseTalk explicit11 candidate when runtime and source conditions pass, Wav2Lip/Ditto/unknown RMS fallback, and final natural PCM replacement. Further GPU work requires a new concrete hypothesis; do not restart old center or Confirm matrices.

Local JSON-only recovery bundle: .codex/scratch/trainable_w2l_holdout_20261013_attempt01/.


## Relations

- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Model-aware natural-audio routing and lossless track replacement qualification]]
