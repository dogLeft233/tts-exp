---
title: Codex checkpoint - 2026-10-14T02-13-19Z - CPU calibration bound and provenance
  retry required
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t02-13-19-z-cpu-calibration-bound-and-provenance-retry-required
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 02:13:19+00:00
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

# Codex checkpoint - 2026-10-14T02-13-19Z - CPU calibration bound and provenance retry required

## Summary
The first bounded Wav2Lip FIT calibration passed its mechanical gates and exposed a low-cost provenance gap; no GPU factor rows were launched, and the next action is a speaker-partitioned MFA retry plus CPU calibration preparation for MuseTalk and Ditto.

## Story
The current research question is whether SEP, STAB, and ENV can become reproducible native-input coordinates across TFGs under fixed RMS, clock, and content, with SyncNet and mouth-motion endpoints kept separate. After the bounded Wav2Lip STAB contrast failed the two-speaker/motion expansion gate, a new CPU MFA/TextGrid source and Wav2Lip calibration were run to reduce uncertainty before spending GPU time.

Calibration retry01 passed on S0053-S0061: 18 utterance pairs, 36 rows, 9 speakers, 631 natural and 631 TTS temporal segments, all known-phone coverage 1.0, endpoint error 0 samples, and speaker-equal ddof=1 summaries. Fifty-eight of 84 phone labels have at least three independent FIT speakers for exploratory SEP prototypes. ENV is unavailable under the conservative sign gate (0.5283 < 0.60). Pilot support is partial (S0044 20/22 labels; S0052 17/20). The original short-segment failure is preserved separately.

Independent review found that MFA aligned a flat corpus as one speaker and the receipt omitted complete executable/version, acoustic-model/dictionary hashes, and exact command provenance. Therefore retry01 is only a mask/clock qualification and exploratory reference; it does not qualify the earlier STAB rows and cannot authorize a GPU intervention by itself. MuseTalk and Ditto still have structural CPU receipts but no FIT coordinate calibration.

## Working State
Canonical experiment/task/research-map/HANDOFF files now record the retry01 result, the provenance gap, and the speed-balanced stop rule. Local artifacts are under `.codex/scratch/native_axis_fit_calibration_20261014_attempt02_retry01/`; remote retry01 remains at `/dev/shm/native_axis_fit_mfa_20261014_attempt02_retry01`. Remote V100 is idle. No tests or GPU/model forwards were run in this stage.

## Verification
- Remote retry01 exit code 0 and receipt status PASS.
- Receipt SHA `4b3673566cbf564cbcced831befb42f92c222093acdd508881b55e26b397288b`; metrics SHA `5e3997d01a37b6f77c895cbb8f09524ed916e045f1d75fc77afb0b5486a269d3`; mel arrays SHA `2b76181ba651556828e09f79de8849058e73740d39e2c4cc2334efc6fae76b31`.
- ENV status `UNAVAILABLE`; no direction fabricated.
- No pytest or implementation tests were run.

## Observations
- [decision] Keep the earlier STAB rows bounded and pre-calibration; do not retroactively upgrade them.
- [decision] Do not launch GPU until MFA speaker binding and tool/model provenance are complete enough for a repeatable coordinate reference.
- [result] Wav2Lip SEP is technically available but only partial on the two pilot speakers; unsupported phone labels remain identity.
- [blocker] Speaker-partitioned MFA retry and equivalent MuseTalk/Ditto CPU FIT calibration are pending.
- [next_step] Finish the speaker-bound MFA retry with complete pins, then run bounded CPU calibration for MuseTalk and Ditto before reconsidering one Wav2Lip SEP GPU arm.

## Relations
- continues [[Codex checkpoint - 2026-10-14T01-39-32Z - MFA timing regenerated and CPU native frontends closed]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[跨TFG原生输入三因素因果pilot Spec 2026-10-13]]