---
title: Codex checkpoint - 2026-10-12T17-47-05Z - Balanced route closure and trainable
  holdout checkpoint
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t17-47-05-z-balanced-route-closure-and-trainable-holdout-checkpoint
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 17:47:05+00:00
username: laptop-e133gsle\dogle
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

# Codex checkpoint - 2026-10-12T17-47-05Z - Balanced route closure and trainable holdout checkpoint

## Summary

This turn closed a speed-balanced, executable model-aware audio route while preserving the one material evidence gap: the trainable checkpoint still needs one bounded unseen-speaker check.

## Story

The original objective is a reusable natural-audio enhancement head for several TFG models. The latest user direction is to reduce audit overhead and balance reliability with time, so this turn did not repeat the old nine-parameter matrix, four-center recursion, or old Confirm. It consolidated the existing three-model holdout, lossless audio replacement and SyncNet engineering evidence, and the trained checkpoint into an executable route.

The route uses exact model_id, then case-folded aliases, then a default. MuseTalk selects the explicit 11-parameter anti-symmetric local spectral shape. Wav2Lip, Ditto, and unknown models use fixed RMS. The explicit shape safely falls back to RMS when PyTorch, source pins, shape digest, or finite-value checks fail. The CLI now separates requested method from applied method, and writes WAV only after a successful transform. The renderer and final natural PCM replacement workflow are documented.

The independent ten-speaker holdout gives an equal-weight routed candidate delta of about +0.004688 full and +0.008456 guard20 versus RMS. Wav2Lip is negative, MuseTalk positive, and the repaired Ditto batch is zero; this is model-heterogeneous and does not support a universal cross-TFG winner. The remote Wav2Lip/SyncNet smoke proves generation, lossless original PCM mux, and official SyncNet execution, but is engineering qualification rather than new unseen-speaker effect evidence. The remaining scientific gap is evaluation of shared_audio_head.pt on unseen speakers and more than one TFG; it is not in the production route.

## Working State

Durable route files:
- scripts/audio_enhancement/model_routing.py
- scripts/audio_enhancement/configs/model_aware_routing_v1.json
- scripts/audio_enhancement/configs/model_aware_routing_v1.md
- scripts/audio_enhancement/cli.py
- scripts/audio_enhancement/README.md

Routing config SHA256: 886f75c03977ab7d8ac7a5caecd568421c9a46cbf455d6b9cef87a9bf24fb00.

Pinned sources:
- shared_recipient_local_power_spectral_head_v1.py: 8f22db628ec84f4b67d01e6e9c1bce12edf9280cca88e52cfc55d32d7b94dcfe
- shared_recipient_transition_head_v3.py: 77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87
- baselines.py: 7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2

Trainable checkpoint:
- runs/cross_tfg_audio_head_20261003/fit_v2_actual_evidence_attempt01/shared_audio_head.pt
- SHA256 768c5b1157139281b4d62d73ed6be19b9f69e37401e9298395f469dc4501c943
- FIT_ONLY_COMPLETE, 96 steps, 8 fit rows. Its report says development and confirmation audio were not opened and effects were not selected on development.

Remote qualification:
- /home/wjj/tts-audio/fit40_w2l_syncnet_qualification_outputs_attempt02/engineering_smoke_manifest.json
- SHA256 61399ad57c082727ba756b9ceff6300c6b3e45202f5854f4262664b796492d9b
- Server 192.168.50.2 remains running and currently has no experiment worker; GPU is idle.

## Changed Files

Relevant dirty-worktree files from this turn:
- CONTEXT.md
- scripts/audio_enhancement/README.md
- scripts/audio_enhancement/__init__.py
- scripts/audio_enhancement/cli.py
- scripts/audio_enhancement/model_routing.py
- scripts/audio_enhancement/configs/model_aware_routing_v1.json
- scripts/audio_enhancement/configs/model_aware_routing_v1.md
- HANDOFF.md

The worktree contains many prior dirty and untracked experiment files; do not clean or reset it.

## Verification

- Ran current-route AST/JSON and git diff --check static checks. No pytest or implementation test was run.
- Local contract smokes: 56320 samples at 16 kHz; MuseTalk, Wav2Lip, and unknown receipts all rc0. MuseTalk fell back to RMS when its runtime condition was unavailable; Wav2Lip and unknown used RMS.
- Remote CPU-only true-torch 2.5.1+cu121 probe: MuseTalk applied explicit11; Wav2Lip and unknown used RMS. Receipts retained current config SHA and untouched natural PCM/original clock policy.
- Read-only checked the existing Wav2Lip generation, original PCM mux, and official SyncNet engineering smoke; it was not counted as scientific holdout.
- Reversibly moved 172 non-scientific checkpoint HTML scratch files to C:/Users/dogle/.codex/scratch/tts-exp-migrated-20261013 to restore E-drive space; scientific artifacts were not changed.

## Observations

- [result] The executable model-aware route and original natural PCM replacement contract are ready; current cross-model evidence supports model-aware routing with RMS as conservative fallback.
- [result] Independent ten-speaker three-model routed aggregate is about +0.004688 full and +0.008456 guard20; model directions are heterogeneous and do not establish a universal winner.
- [decision] Use a speed-balanced boundary: do not repeat the old nine-parameter/four-center/old Confirm work; allow one bounded unseen-speaker check for the trainable checkpoint.
- [blocker] shared_audio_head.pt is still fit-only. Without one bounded unseen-speaker evaluation, the trainable head cannot be called a validated production method.
- [next_step] Read the checkpoint_unseen_holdout agent result; if feasible, run and record one Wav2Lip-only 20-row checkpoint-versus-RMS unseen holdout; otherwise record the concrete blocker and keep the task open.

## Relations

- relates_to [[Developing a cross-TFG natural audio enhancement head]]
- relates_to [[Model-aware natural-audio routing and lossless track replacement qualification]]
- relates_to [[Model-aware routing with RMS fallback]]