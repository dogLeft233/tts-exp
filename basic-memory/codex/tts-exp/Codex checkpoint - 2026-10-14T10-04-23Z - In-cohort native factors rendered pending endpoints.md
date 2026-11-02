---
title: Codex checkpoint - 2026-10-14T10-04-23Z - In-cohort native factors rendered
  pending endpoints
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t10-04-23-z-in-cohort-native-factors-rendered-pending-endpoints
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 10:04:23.213543+00:00
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

# Codex checkpoint - 2026-10-14T10-04-23Z - In-cohort native factors rendered pending endpoints

## Summary

The frozen S0044/S0052 MuseTalk and Ditto cohort now has qualified native arrays and sixteen factor/no-op renders; SyncNet and visual endpoints are the next bounded work.

## Story

The user wants reproducible coordinates for phoneme separation, local stability, and local energy or spectral envelope across TFG models at fixed RMS, content, and natural clock, then independent mouth-motion measurements. The previous bounded closure did not cover the requested matrix: MuseTalk/Ditto ENV used a FIT-exposed S0053 utterance, while Wav2Lip STAB was pre-calibration and SEP_RD was a variant. The Task was reopened rather than treating those checks as complete coverage. The parent usable enhancement-head objective remains active.

CPU extraction materialized the correct S0044/S0052 native tensors for MuseTalk and Ditto. SEP and STAB operators passed algebraic checks on those inputs, and all four conditions per model/speaker were materialized. Sixteen small renders then completed with original natural PCM muxing. This is engineering completion only; none of the new sixteen rows has a score or mouth endpoint yet. Old negative results remain valid for their stated cohorts and variants, not replacements for missing cells.

## Working State

- Authoritative task and experiment: basic-memory/Tasks/native-axis-factor-pilot-20261013.md and basic-memory/Experiments/native-axis-factor-pilot-20261013.md. Research map and HANDOFF need the new in-cohort results reconciled after endpoint collection.
- CPU native arrays: .codex/scratch/native_axis_cpu_frontends_materialize_20261014_attempt01/ (MuseTalk PASS, initial Ditto provider failure preserved) and .codex/scratch/native_axis_ditto_materialize_20261014_attempt01/ (Ditto PASS).
- CPU operator receipt: .codex/scratch/native_axis_pilot_operator_qualification_20261014_attempt01/. SEP directional movement is 0.5 FIT SD; actual rival-distance movement is 0.131/0.161 MuseTalk and 0.178/0.199 Ditto. STAB residual ratio is 0.75 and centroids are unchanged. Algebraic purity must not be confused with full empirical metric independence.
- Remote feature packet: /home/wjj/tts-audio/tts-exp/.codex/scratch/native_axis_pilot_feature_materialize_20261014_attempt01/ (16 rows PASS).
- Remote Ditto renders: /home/wjj/tts-audio/tts-exp/.codex/scratch/native_axis_pilot_ditto_render_20261014_attempt02/; both speakers x noop/SEP/STAB/env, 93/103 frames. Remote MuseTalk renders: /home/wjj/tts-audio/tts-exp/.codex/scratch/native_axis_pilot_musetalk_render_20261014_attempt01/; both speakers x noop/SEP/STAB/ENV.
- Remote runtime and /dev/shm are fragile machine-local state on wjj@192.168.50.2. SSH BatchMode works. No shutdown is authorized now. No credentials are stored.
- Git is dirty with substantial pre-existing modified and untracked files; current work is in scratch/BM/HANDOFF. HEAD is local/unpublished (GitHub SHA lookup returned 422); GitHub branch PR query returned none.

## Changed Files

- .codex/scratch/native_axis_pilot_operator_qualification_20261014_attempt01.py
- .codex/scratch/native_axis_pilot_feature_materialize_20261014_attempt01.py
- Scratch MuseTalk and Ditto runners adapted for variable sequence length and conditions.
- Task, experiment, research map and HANDOFF previously corrected partial coverage; final endpoint reconciliation remains pending.

## Verification

In-cohort arrays are finite and source bound. MuseTalk shapes are [92,50,384] and [102,50,384]; Ditto shapes are [93,1024] and [103,1024]. Four model/speaker SEP/STAB operator receipts pass shape/mask/centroid/residual checks. Sixteen condition feature rows and all sixteen renders completed, with natural PCM mux equality. No pytest or implementation tests ran. Earlier empty-output interruption and CUDA visibility failure were retained as separate engineering attempts; Ditto attempt02 is the valid render root.

The current model availability problem was GPT-6 Luna/max capacity during the previous execution, so ROOT made fallback operational progress. Two fresh GPT-6 Luna/max agents have now been dispatched: native_axis_scoring owns endpoints; native_axis_design_review performs a capped read-only purity/visual-estimand review.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Current local HEAD: `5688784dd5c6c1272074f0c4b6c5ff79f117c907` (not resolved by GitHub).
- Strict previous coverage manifest: .codex/scratch/native_axis_factor_pilot_coverage_manifest_20261014_attempt01.json (PARTIAL_COVERAGE).

## Observations

- [result] Sixteen in-cohort MuseTalk/Ditto factor/no-op videos exist; their scientific endpoints are pending.
- [decision] Keep the pilot small and score existing rows before any further render or expansion.
- [decision] Do not infer accurate mouth articulation from a larger mouth-velocity RMS alone; retain score and independent motion endpoints separately.
- [blocker] Wav2Lip ENV is unavailable at the frozen calibration gate; canonical Wav2Lip SEP/STAB coverage and new MuseTalk/Ditto endpoints remain incomplete.
- [next_step] Score the sixteen current rows with canonical natural-PCM SyncNet and mouth aperture/velocity, compare each factor with its matched no-op, then reconcile the evidence map and coverage matrix.

## Relations

- continues [[Codex checkpoint - 2026-10-14T06-31-26Z - Bounded native-axis pilot concluded]]
- relates_to [[native-axis-factor-pilot-20261013]]