---
title: Codex checkpoint - 2026-10-12T17-12-40Z - Executable model-aware route smoke
  complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-12-t17-12-40-z-executable-model-aware-route-smoke-complete
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-12 17:12:40+00:00
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

# Codex checkpoint - 2026-10-12T17-12-40Z - Executable model-aware route smoke complete

## Summary

The model-aware natural-audio route is now executable from the repository CLI, with fail-closed RMS behavior and a completed local CPU smoke; no new GPU sweep was started.

## Story

The completed science result is a three-model, ten-unseen-speaker holdout: MuseTalk benefits from the anti-symmetric explicit11 candidate, Wav2Lip is safer with RMS, and Ditto is unchanged by the candidate. The route is therefore model-aware rather than universal. This turn converted that decision into a usable interface while honoring the user's request to control audit time.

## Working State

`model_routing.py` performs exact model ID then case-folded alias dispatch, source-pin checks, explicit11 candidate application when the runtime supports it, and RMS fallback for unknown models or provider errors. `cli.py` exposes `--model-id` and `--routing-config`; receipts preserve the untouched natural PCM/original clock policy and record route/config hashes, fallback state, and exact sample counts. `README.md`, route markdown, `HANDOFF.md`, `CONTEXT.md`, and the BM decision/task now describe the executable route.

A bundled-Python CPU smoke used one 16 kHz WAV with 56320 samples. Wav2Lip alias, MuseTalk alias, and an unknown model all returned finite same-length output. MuseTalk correctly recorded a fallback to RMS because local PyTorch is unavailable. No implementation tests, renderer runs, SyncNet runs, or GPU jobs were added in this turn.

## Changed Files

- `scripts/audio_enhancement/model_routing.py`
- `scripts/audio_enhancement/cli.py`
- `scripts/audio_enhancement/__init__.py`
- `scripts/audio_enhancement/configs/model_aware_routing_v1.json`
- `scripts/audio_enhancement/configs/model_aware_routing_v1.md`
- `scripts/audio_enhancement/README.md`
- `CONTEXT.md`
- `HANDOFF.md`

## Observations

- [result] The route is usable as a generation-control interface and safely falls back to RMS.
- [result] The scientific routed holdout contribution remains approximately +0.004688 full and +0.008456 guard20, with MuseTalk positive, Wav2Lip negative, and Ditto zero relative to RMS.
- [decision] Stop the current turn at the executable prototype and smoke; do not spend another GPU run until a concrete renderer-level question justifies it.
- [blocker] The full holdout body remains remote/fragile, and the local machine cannot execute the explicit11 provider without PyTorch.
- [next_step] If needed, run one pinned renderer engineering smoke on the LAN host; otherwise use the CLI route directly and keep the original natural WAV for mux/scoring.

## Relations

- continues [[Codex checkpoint - 2026-10-12T17-00-03Z - Speed-balanced executable route integration]]
- relates_to [[Model-aware routing with RMS fallback]]