---
title: Codex checkpoint - 2026-10-14T12-05-50Z - Canonical two-model factor screen
  and Wav2Lip boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t12-05-50-z-canonical-two-model-factor-screen-and-wav2-lip-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 12:05:50+00:00
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

# Codex checkpoint - 2026-10-14T12-05-50Z - Canonical two-model factor screen and Wav2Lip boundary

## Summary

The corrected native-axis screen completed 12 Ditto/MuseTalk factor rows; none of SEP, STAB or ENV passes both speakers in either model.

## Story

The user wants fixed RMS, unchanged natural timing and content, native-space separation/stability/envelope interventions and independent mouth motion endpoints, with small experiments before scaling. The earlier round-mask packet disagreed with FIT frame ownership. The canonical repair now uses j/25 >= xmin and j/25 < xmax with FIT-supported phone masks, float32 arrays and source/calibration/TextGrid hashes.

Six corrected rows per model were rendered against existing no-op baselines. Generation uses fixed-RMS driver audio; final mux and scoring use untouched natural PCM. Official SyncNet and complete MediaPipe tracks passed engineering checks. Ditto passes only S0044 SEP/ENV, MuseTalk only S0052 SEP, and STAB fails every corrected row. These coordinates have no two-speaker replication. Global aperture and velocity are descriptive, not articulation-accuracy labels.

Wav2Lip CPU materialization now has canonical SEP/STAB chunks for both speakers; ENV remains unavailable (sign agreement 0.530017 < 0.60). ROOT assigned GPT-6 Luna/max wav2lip_minimal_gpu only four fresh factor rows to complete the each-TFG comparison. The worker confirmed an idle GPU and no competing process at 12:05:23Z; renderer launch was pending at this snapshot. No broad matrix, old four-center, Dev, Confirm, archive or shutdown branch is reopened.

## Working State

The artifacts below are machine-local/remote scratch and not Git backups.

- Ditto packet: .codex/scratch/native_axis_pilot_ditto_canonical_score_20261014_attempt01/, manifest SHA 83b784e5741967dc3065e18fe924d0f6eed41b60b92c5392fe3b30a5.
- MuseTalk packet: .codex/scratch/native_axis_pilot_musetalk_canonical_score_20261014_attempt01/, manifest SHA 7bac2949f257fa71fcad86074079de0e827a4ce80fc91ee90b95e69743588b72.
- Combined summary: .codex/scratch/native_axis_canonical_factor_summary_20261014_attempt01.json, SHA d0bedebcd8a95c1ffcce65dc677f8471e6f5da3d712b9554b0c2ccba316e111e.
- Canonical array root: .codex/scratch/native_axis_pilot_mask_audit_20261014_attempt02/; source SHA 3c1e47e03640155d5d6e667ffb9b765850e82dd0b20039a510f9e3d0ba4b8f5e.
- Wav2Lip CPU receipt: .codex/scratch/native_axis_wav2lip_canonical_cpu_20261014_attempt01/materialized/receipt.json, SHA 0e310ba7bdcb444a4023de9c7aa93a152831b49432f0e9aae74ded7a9b4ef753. Supported chunks 59/89 and 58/100; ENV arrays_created=false.
- Remote host wjj@192.168.50.2:22 remains running. Old Wav2Lip STAB is PRE_CALIBRATION_UNQUALIFIED; SEP_RD is CLOSED_VARIANT_DO_NOT_REUSE.
- Git tree remains heavily dirty from earlier research: nine tracked modifications and many untracked sources/artifacts. This turn changes BM, HANDOFF and scratch evidence, not those pre-existing tracked source modifications.

## Changed Files

- basic-memory/Experiments/native-axis-factor-pilot-20261013.md: corrected canonical table and historical round-mask boundary.
- basic-memory/Tasks/native-axis-factor-pilot-20261013.md: mask repair completion and remaining Wav2Lip cells.
- basic-memory/Research/研究主线与证据地图.md: canonical two-model screen and partial three-TFG status.
- HANDOFF.md: canonical packet pointers and bounded Wav2Lip follow-up.

## Verification

Local score manifests and combined summary were inspected. All 12 corrected rows have official SyncNet and valid MediaPipe endpoints; no factor replicates across both speakers. Wav2Lip CPU receipt PASS and source/array bindings were read directly. No implementation tests were run. The canonical GitHub repository URL was verified via public API; current SHA was not resolvable there and is treated as local/unpushed. GitHub CLI is unavailable, so PR state was not established.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Current local/unpushed Git SHA: `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.

## Observations

- [result] Canonical Ditto/MuseTalk response fails two-speaker replication for every tested factor.
- [decision] Fixed-RMS generation and untouched natural score/mux PCM remain separate.
- [decision] Aperture/velocity remain descriptive and separate from articulation accuracy.
- [decision] Stop Ditto/MuseTalk factor expansion; limit Wav2Lip to four canonical SEP/STAB rows.
- [blocker] Wav2Lip ENV is unavailable; canonical SEP/STAB score and visual endpoints are pending.
- [next_step] Collect the minimal Wav2Lip canonical SEP/STAB endpoints and synthesize the bounded three-TFG result before any new intervention design.

## Relations

- continues [[Codex checkpoint - 2026-10-14T10-58-47Z - Native factor endpoints withheld for mask repair]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]