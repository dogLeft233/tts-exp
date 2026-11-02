---
title: Codex checkpoint - 2026-10-12T17-00-03Z - Speed-balanced executable route integration
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t17-00-03-z-speed-balanced-executable-route-integration
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 17:00:03+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- '["codex"'
- '"checkpoint"]'
---

# Codex checkpoint - 2026-10-12T17-00-03Z - Speed-balanced executable route integration

## Summary

This checkpoint narrows the work to an executable model-aware routing interface and one low-cost smoke, while preserving the completed three-model independent holdout result and avoiding another broad audit or GPU matrix.

## Story

The project aims to deliver a natural-audio enhancement head for multiple TFGs: enhanced audio drives video generation, while scoring restores the original natural PCM and original clock. The strongest current evidence is a 10-unseen-speaker Wav2Lip/MuseTalk/Ditto holdout: Wav2Lip and Ditto use RMS fallback, MuseTalk uses the anti-symmetric explicit11 candidate, and the equal-model routed contribution is about +0.004688 full and +0.008456 guard20. The route config is currently declarative JSON only; production pipeline/CLI do not consume model_id and do not implement the anti-symmetric candidate. This turn therefore limits work to the smallest executable contract plus a CPU/audio and lossless-mux smoke.

## Working State

Durable: the nine-parameter route is fully rejected; two explicit11 360-row runs are CPU-recovered; the independent holdout completed 60 planned rows (Wav2Lip 20/20 PASS, MuseTalk 20/20 PASS, Ditto pin-fix 20/20 PASS, with Ditto candidate equal to RMS); the routing decision is MuseTalk candidate, Wav2Lip/Ditto/unknown RMS fail-closed. The trainable waveform head remains fit-only evidence and is excluded from delivery.

Machine-local/fragile: full holdout JSONL/aggregate/source pins remain mainly on remote /dev/shm; the local tree has summaries and hashes but not the full body. The route implementation and smoke are not complete. The server remains running; do not resume shutdown, archive, or old-route work.

## Changed Files

Current untracked route artifacts are scripts/audio_enhancement/configs/model_aware_routing_v1.json and scripts/audio_enhancement/configs/model_aware_routing_v1.md. The next agent may add a pure-CPU routing/provider module and CLI selector without changing scientific originals.

## Verification

Read-only checks confirmed repository branch/SHA/status and parsed the route JSON. No implementation tests, GPU training, or broad matrix were run. Agents confirmed that pipeline/CLI do not yet consume the route JSON and that overdub.py already contains PCM, video-stream, and timestamp checks.

## Observations

- [result] The holdout supports a practical model-aware prototype, not one universal candidate.
- [decision] To control time cost, prioritize an executable route and one reproducible smoke; do not launch another GPU sweep for a declarative config.
- [blocker] The anti-symmetric explicit11 waveform provider is not yet in the production audio transform API, and the full holdout body is not locally reproducible.
- [next_step] Accept the minimal route resolver/provider implementation and run one CPU audio transform plus lossless-mux smoke.

## Relations

- continues [[Codex checkpoint - 2026-10-12T14-37-58Z - Speed-balanced independent holdout continuation]]
- relates_to [[Model-aware routing with RMS fallback]]