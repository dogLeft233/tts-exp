---
title: Codex checkpoint - 2026-10-13T04-02-19Z - Native axis factor pilot planned
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t04-02-19-z-native-axis-factor-pilot-planned
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 04:02:19+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
---

# Codex checkpoint - 2026-10-13T04-02-19Z - Native axis factor pilot planned

## Summary
The research-map-guided native-axis factor pilot is now frozen as a qualification-first experiment, while no GPU or scientific score has been started.

## Story
The user asked to test the three shared directions already identified in the TTS-TFG research map: phoneme representation separation, local segment stability, and local energy or spectral envelope. The root agent read the Startup Router, the canonical research map, the evidence synthesis, HANDOFF, and the latest route boundary. A bounded protocol was written under the Basic Memory Experiments structure and indexed with an ASCII Basic Memory experiment note. The design fixes RMS, natural sample clock, transcript, MFA boundaries, and final natural PCM scoring, and limits the first execution to two unseen speakers, three TFGs, and four arms per context after CPU/native-front-end qualification.

The protocol is model-native: Wav2Lip uses its mel chunk, MuseTalk uses its pinned Whisper/audio sequence, and Ditto uses its pinned HuBERT/ONNX sequence. Ditto hidden dimensions are not treated as frequency bands. SyncNet C/B/D/fixed-lag distance and normalized mouth aperture and velocity are separate endpoints; without human or real-video labels, motion is only reported as motion-consistent or not.

## Working State
The durable protocol is in basic-memory/Experiments/ under the dated Chinese title. The indexed ASCII experiment note is native-axis-factor-pilot-20261013. The active Task is tts-exp/tasks/native-axis-factor-pilot-20261013. The research map in basic-memory/Research/ now links the new pilot and its two-speaker, two-of-three-model stop rule.

Existing native encoder evidence was checked before planning execution: Wav2Lip and MuseTalk have a completed historical FIT-only probe for eight clips and seven audio conditions with no video or SyncNet; Ditto feature collection remains SOURCE_ONLY_NOT_EXECUTED. These assets cannot be used as unseen-speaker results. The server wjj@192.168.50.2:22 is reachable and GPU idle; no new GPU job was launched.

## Changed Files
- basic-memory/Experiments/dated native-axis protocol file: created and revised so ENV uses true mel/log-mel bands only where the frontend has a frequency axis, and Ditto uses frame-norm envelope only.
- basic-memory/Experiments/native-axis-factor-pilot-20261013.md: indexed experiment note.
- basic-memory/Tasks/native-axis-factor-pilot-20261013.md: active task with steps, blocker, and next action.
- basic-memory/Research/Research map: added the planned pilot entry.

## Verification
- Read the canonical Startup Router and experiment instructions from local Basic Memory.
- Read the canonical research map, evidence synthesis, CONTEXT.md, and HANDOFF.md.
- Read back the indexed experiment note and active task from Basic Memory.
- Confirmed the Wav2Lip and MuseTalk historical native probe summaries and its no-video/no-score scope.
- Confirmed the remote V100 is idle with no matching training or pilot process.
- No implementation tests, GPU render, SyncNet score, or visual metric run was performed in this turn.

## Observations
- [decision] Keep qualification first and cap the first scientific execution at 24 generated units; do not reopen retired four-center or Confirm branches.
- [decision] Never call a PCA or hidden dimension a spectral band for Ditto; mark an unavailable spectral coordinate instead.
- [blocker] The requested GPT-6 Luna/max execution slots were repeatedly capacity-rejected earlier in the turn; one pending Luna/max subagent is now being retried for the CPU/native qualification.
- [next_step] Have the Luna/max subagent confirm native hook entry points and run the CPU qualification, then only launch the predeclared two-speaker pilot if all three models have a valid bounded adapter.

## Relations
- continues [[Codex checkpoint - 2026-10-13T01-54-44Z - Research map and startup routing]]
- relates_to [[native-axis-factor-pilot-20261013]]