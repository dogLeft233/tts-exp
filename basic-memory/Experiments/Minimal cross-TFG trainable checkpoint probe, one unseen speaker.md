---
title: Minimal cross-TFG trainable checkpoint probe, one unseen speaker
type: experiment
permalink: tts-exp/experiments/minimal-cross-tfg-trainable-checkpoint-probe-one-unseen-speaker
status: concluded
project: tts-exp
tags:
- experiment
- tts
- tfg
---

# Minimal cross-TFG trainable checkpoint probe, one unseen speaker

## Question
Does the existing trainable fit_v2 checkpoint beat fixed RMS when the same unseen natural utterance is sent through several TFG frontends, using the original natural PCM for scoring and final mux?

## Design
To control time cost, this was a six-row independent probe: one unseen speaker (`BAC009S0044W0266`), Wav2Lip, MuseTalk, and Ditto, with trainable and RMS controls. The runner used a new namespace and source contract and did not mutate or relaunch the retired matrix. The checkpoint SHA256 was `768c5b1157139281b4d62d73ed6be19b9f69e37401e9298395f469dc4501c943`.

## Result
The plan SHA256 was `8f4ee6cc84971807baf7687d79fd5cf9983bf1491fea21d4eb15639343aaa1f`; raw plan SHA256 `4a8340da03adc45abb5dda83046b9edb8faa24b61b0457c5840cf1ba8d2bba90`; planned rows SHA256 `8fc56b39bc22296bb0c0fffbaddf9354c6e54a79dde91c27bc8ce258eb7d73e9`.

Wav2Lip and MuseTalk completed 4/4 rows with original PCM, 25 fps, and SyncNet. Wav2Lip trainable minus RMS Cbest was full `-0.060789` (7.146176 versus 7.206965) and guard20 `-0.166985` (7.732951 versus 7.899936). MuseTalk was full `-0.095057` (4.257451 versus 4.352508) and guard20 `-0.139113` (5.143403 versus 5.282516).

Ditto's two rows failed before rendering because the repaired adapter points to a Stage-A block CLI that requires `--stagea-core`, parent plan/rows, and runtime binding, while the independent wrapper supplies the silent-video CLI contract. The resulting argparse failure is an implementation interface blocker, not a Ditto score.

## Interpretation
This small true execution agrees with the larger fit_v4 development aggregate: the current trainable checkpoint is not a drop-in replacement for RMS on the models that executed. It does not establish a universal learned head, and it does not justify reopening the retired center or Confirm matrices. A future Ditto trainable result requires a separately frozen adapter contract and an explicit new budget.

## Provenance
Remote roots:
- `/dev/shm/fit40_queue_remote_package_attempt02/project/trainable_cross_tfg_oneid_20261013_attempt01`
- `/dev/shm/fit40_queue_remote_package_attempt02/project/trainable_cross_tfg_oneid_20261013_retry_ditto_attempt01`

Local readback and source bundle: `.codex/scratch/trainable_cross_tfg_oneid_attempt01/`.

## Relations
- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Fit v4 trainable enhancement head cross-TFG development result]]