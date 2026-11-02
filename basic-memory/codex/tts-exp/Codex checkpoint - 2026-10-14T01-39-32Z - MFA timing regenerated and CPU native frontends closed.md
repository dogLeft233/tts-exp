---
title: Codex checkpoint - 2026-10-14T01-39-32Z - MFA timing regenerated and CPU native
  frontends closed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t01-39-32-z-mfa-timing-regenerated-and-cpu-native-frontends-closed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 01:39:32+00:00
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

# Codex checkpoint - 2026-10-14T01-39-32Z - MFA timing regenerated and CPU native frontends closed

## Summary
The fixed-RMS native-axis pilot now has regenerated MFA timing and authoritative CPU-only native frontend receipts for Wav2Lip, MuseTalk, and Ditto; the next bounded action is FIT calibration on S0053-S0061 before any new factor GPU rows.

## Story
The active objective is to identify reproducible native-input coordinates for phoneme separation, local stability, and energy/spectral-envelope interventions while keeping RMS, clock, and content fixed and checking SyncNet against mouth-motion descriptors. The preceding Wav2Lip STAB run remains a bounded two-speaker result: S0044 improved distance, S0052 was near-null with worse Dmin, and motion changes were inconsistent, so it is not a cross-TFG factor claim. To improve speed/reliability balance, the current turn closed the structural CPU boundary and rebuilt the missing calibration timing rather than launching more GPU rows.

The historical FIT archive had WAV and .lab files but no TextGrids. Forty TextGrids were regenerated with CPU MFA using mandarin_china_mfa and mandarin_mfa for S0052-S0061. Current CPU-only native frontend probes are now pinned for Wav2Lip, MuseTalk, and Ditto; Ditto required a scratch-only CPU adapter because its source configuration forced cuda/ori. These probes establish finite native tensors and phone coverage, but they are not factor interventions or SyncNet results. S0052 attempt01 used an obsolete PCM and is engineering-only; attempt02/attempt06 use recovered PCM `17a6809b...` where applicable.

## Working State
The canonical protocol and task remain `basic-memory/Experiments/native-axis-factor-pilot-20261013.md` and `basic-memory/Tasks/native-axis-factor-pilot-20261013.md`. The research map and `HANDOFF.md` carry the same boundary. Remote V100 at 192.168.50.2 is idle; no new GPU factor run is authorized yet. Fragile receipts are under `.codex/scratch/` locally and `/dev/shm/` remotely; the scientific source and natural PCM/clock rules remain unchanged.

## Changed Files
- `basic-memory/Experiments/native-axis-factor-pilot-20261013.md` updated with CPU frontend and MFA receipts.
- `basic-memory/Tasks/native-axis-factor-pilot-20261013.md` updated with the next calibration gate.
- `basic-memory/Research/研究主线与证据地图.md` and `HANDOFF.md` updated with the same boundary.
- `.codex/scratch/native_axis_fit_mfa_20261014_attempt01/` contains the copied TextGrids and MFA logs.
- `.codex/scratch/native_axis_cpu_wav2lip_receipt_20261014.json`, `.codex/scratch/native_axis_cpu_frontends_probe_20261014_attempt02/`, and `.codex/scratch/native_axis_cpu_frontends_probe_20261014_attempt06_ditto_receipt.json` are machine-local receipts.

## Verification
- CPU MFA alignment completed with exit code 0; 40 TextGrids were produced and copied locally.
- Wav2Lip, MuseTalk, and authoritative Ditto CPU probes completed with finite tensors and the pinned source/provider receipts described above.
- No pytest or implementation tests were run, and no new GPU factor intervention was launched.

## Observations
- [result] Structural native input spaces are now closed for CPU probing: Wav2Lip mel, MuseTalk Whisper conditioning, and Ditto HuBERT/ONNX sequence.
- [decision] Treat regenerated MFA calibration as a new protocol; do not retroactively qualify the earlier STAB result.
- [decision] Exclude S0052 from FIT calibration to avoid pilot leakage; use speaker-equal metrics with known-phone masks, explicit clock mapping, and ddof=1.
- [blocker] FIT calibration metrics and replayable SEP/STAB/ENV coordinate estimates are still missing; historical archives do not contain a frozen estimator.
- [next_step] Have the Luna/max calibration agent create and run the CPU-only FIT calibration on S0053-S0061 using the current Wav2Lip source pins and the 40 regenerated TextGrids, then inspect the receipt before authorizing any new GPU factor rows.

## Relations
- continues [[Codex checkpoint - 2026-10-13T17-07-27Z - STAB gate closed and calibration boundary]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[跨TFG原生输入三因素因果pilot Spec 2026-10-13]]