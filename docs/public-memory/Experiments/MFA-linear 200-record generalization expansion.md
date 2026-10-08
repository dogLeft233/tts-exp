---
title: MFA-linear 200-record generalization expansion
type: report
permalink: tts-exp/experiments/mfa-linear-200-record-generalization-expansion
tags:
- mfa-linear
- syncnet
- generalization
- tts-only
- experiment
status: blocked
---

# MFA-linear 200-record generalization expansion

## Context

A new OpenSpec change `scale-mfa-linear-sync-generalization-200` was created to test whether a TTS-only MFA-linear waveform adapter generalizes better when trained on exactly 200 records. The planned evaluation uses 40 source-group-disjoint fit-only records, with at least 30 of 40 successes and unchanged per-record and median SyncNet gains.

## Protocol

- The adapter accepts only one exact `[1,1,61440]` MFA-linear TTS waveform.
- Natural audio is excluded from model input and training loss; it is used only for detached coordinate calibration and, conditionally, natural-audio replacement evaluation.
- Model architecture, frozen SyncNet, target-margin loss, 31-offset official curves, `q=0.001`, minimum gain `0.003`, target gap `0.002`, audio QC, fresh seed, and 100-step budget remain fixed.
- The 200 training records and 40 evaluation source groups must be score-independent and locked before training. Wav2Lip replacement is allowed only after a complete real-video transfer pass, with 360 required matrix cells.
- The previous four-record and eight-record run roots remain immutable and are not reused as mutable parents.

## Execution result on 2026-09-03

- Focused predecessor regression suite: 89 tests passed.
- New scale protocol suite: 10 tests passed.
- OpenSpec strict validation: passed.
- New run: `runs/lrs3_mfa_linear_sync_generalization_200_20260903/`.
- Structural scan: 102 eligible records across 10 source groups.
- Required scale: 200 training records and 40 evaluation source groups.
- Terminal status: `BLOCKED_DATASET_SCALE`.
- Training, candidate generation, real-video scoring, and Wav2Lip replacement did not run.
- Terminal artifact validation: valid; no denominator was reduced and no record was duplicated or substituted.

## Interpretation

The 200-record generalization experiment cannot begin with the currently locked fit-only inventory. The repository has more raw policy/MFA files, but after excluding the predecessor fit groups and enforcing the existing MFA, video, exact-length, hash, and train-split contracts, only 102 records and 10 source groups remain. This is a data-inventory blocker, not evidence that a 200-record adapter succeeds or fails.

To continue, expand the permitted fit-only asset inventory to at least 200 eligible training records plus 40 additional source groups, then start a new immutable run. Do not use sealed validation/test records, duplicate existing rows, weaken QC, alter thresholds, or modify the blocked run.

## Relations

- implements [[MFA-linear real-video SyncNet prototype plan]]
- relates_to [[tts-exp|Natural reference audio available at inference]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
