---
title: Codex checkpoint - 2026-10-13T15-29-43Z - Valid W2L native baseline
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-13-t15-29-43-z-valid-w2-l-native-baseline
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-13 15:29:43+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
---

# Codex checkpoint - 2026-10-13T15-29-43Z - Valid W2L native baseline

## Summary
The Wav2Lip native-axis scorer contract is now valid on one unseen clip, so the pilot can move to a second baseline and motion qualification without opening factor arms yet.

## Story
The user requested fixed RMS, clock, and content causal tests for separation, local stability, and energy or envelope in each TFG native input space, plus mouth aperture and velocity checks. The root agent kept the qualification-first two-speaker, three-model, four-arm protocol in canonical Basic Memory and asked a GPT-6 Luna/max subagent to repair the scorer before using GPU time for factor rows.

The original SyncNet pipeline converted a correct 89-frame source to a 91-frame AVI because `-async 1 -r 25` inherited container timing. The subagent made a fresh remote scratch copy with passthrough rational clock conversion. The valid Wav2Lip S0044 no-op baseline now uses contiguous source frames 0..88, a 56960-sample natural PCM16 window, and an 89-frame crop. The official evaluator completed all 84 of 84 five-frame queries with no invalid windows. This is a baseline/scorer qualification result, not an enhancement result.

Target transcripts for both unseen clips were verified in the AISHELL transcript provenance file, but no target-specific phone boundaries or prototypes are available yet. A CPU-only forced-alignment qualification is being attempted; SEP/STAB remain withheld until it produces a verified TextGrid. No MuseTalk, Ditto, SEP, STAB, or ENV factor arm has run.

## Working State
Durable protocol and task are in `basic-memory/Experiments/` and `basic-memory/Tasks/`; the research map and `HANDOFF.md` now record qualification in progress. The canonical indexed experiment note remains scientific_status NOT_RUN.

Remote baseline receipt:
`/dev/shm/cross_tfg_native_axis_injection_pilot_20261013_attempt01/score_corrected_w2l_s0044_attempt05/score_corrected_audit.json`
Remote scratch remains machine-local. The V100 is idle after scoring. The working tree remains dirty from earlier research; root intentionally changed only HANDOFF and Basic Memory task/map/experiment state in this turn.

## Verification
- Wav2Lip native hook generator rc 0; feature tensor shape `[89,512,1,1]`; no-op determinism guard passed.
- Corrected scorer receipt status PASS; source/crop/primary MKV 89 frames at 25 fps; natural PCM16 window 56960 samples, SHA `f82c14bca1a3403c485e9629a108e7e37229f83376a549efd6ffe51fbfe606bd`; valid queries 84/84; matrix 84x31; Cbest `6.9585318565`; Canchor `1.1476526260`; B `13.7897262573`; D0 `12.6420736313`; Dmin `6.8311944008`; best lag `+3` frames; official AV offset `-3` frames; evaluator rc 0.
- Remote post-run GPU: Tesla V100-SXM2-16GB, 5 MiB used, 0 percent utilization, no matching renderer or SyncNet process.
- Alignment search found target WAVs and transcripts but no target-specific TextGrid/prototype; no tests were run.

## Observations
- [result] A valid native Wav2Lip no-op baseline and scorer contract exists for S0044.
- [decision] Keep the preliminary score separate from factor deltas and motion claims; do not use the invalid 91-frame receipt.
- [decision] Do not spend renderer budget on MuseTalk/Ditto until the second baseline and alignment/motion gates are resolved.
- [blocker] Target phone boundaries and prototypes remain unavailable; SEP/STAB cannot be run with invented masks.
- [next_step] Complete the bounded S0045 Wav2Lip no-op baseline and extract mouth aperture/velocity for the two baseline crops.

## Relations
- continues [[Codex checkpoint - 2026-10-13T14-44-50Z - Native W2L baseline and corrected scorer]]
- relates_to [[native-axis-factor-pilot-20261013]]