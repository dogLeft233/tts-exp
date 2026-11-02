---
title: Codex checkpoint - 2026-10-16T04-10-21Z - Local clock pilot completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t04-10-21-z-local-clock-pilot-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: '2026-10-16T04:10:21Z'
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

# Codex checkpoint - 2026-10-16T04-10-21Z - Local clock pilot completed

## Summary

The authorized four-source fixed-video local-clock pilot is complete: all five technically qualified pairs improved Sync-C, while19/24 maps exceeded frozen strain limits, so no eight-source extension ran.

## Story

The project began with the mechanism of TTS talking-face gains and a transferable natural-audio head. The human paused that head, questioned natural/TTS crossed-score attribution, and report44 reproduced large ordinary within-class instance losses. The latest request authorized four existing-video sources with a known timing positive control and sham, extending only after controls were effective, with ROOT theory/design/BM/actual QA and GPT-6 Luna/max implementation and execution.

ROOT froze actual waveform phase-vocoder remapping, endpoint-only affine sham of the same output length/RMS, identity shams, independent MFA timing and original cached video features/support/self-lag. Three of four known240ms controls qualified: actual clock recovery91.3–97.9%, fixed-lag score-loss recovery85.1–98.9%, identity drift about1e-6. S0008 repair TextGrid was missing after disk exhaustion and remained unmeasurable, despite its available waveform and score.

The24 predeclared main pairs yielded five qualified operations from two speakers: S0003 four rows and S0007 one. Local versus affine Sync-C gains were+1.770 to+5.495, all positive; actual center MAE241.8–594.7ms fell to7.7–16.3ms. Natural within-class, TTS within-class, and both cross-class directions are represented. Nineteen true strain violations remained, seventeen involving internal intervals, and no source met complete within/cross coverage for both reference videos. The pilot stopped at four without selecting by score signs or claiming a balanced population mean/CI. Video pixels never changed; the estimand is the combined MFA-map plus PV evaluation-audio operation.

## Working State

Authoritative report: `tts-exp/docs/experiments/45-local-clock-crossover`, file `basic-memory/docs/experiments/45-local-clock-crossover.md`. Task `tts-exp/codex/tts-exp/local-clock-crossover-20261016` is done; spec concluded, evidence map/index/CONTEXT updated (99lines). Head, old training/centers/Dev/Confirm and closed audits remain paused. No pending experiment is authorized; every recorded launcher, scorer and MFA CLI PID was confirmed absent. LAN wjj@192.168.50.2:22 stays running.

Actual38 audio forwards,0 new TFG/TTS/ASR or installs/tests. Source commit `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223` contains only five new dedicated scripts. Nine inherited tracked changes and1004 untracked entries remain untouched except the intended CONTEXT update; commit is local/unverified remotely, no PR. GitHub REST confirms canonical repository, empty exact-head PR array and commit lookup422.

Machine-local data: `runs/local_clock_crossover_20261016_attempt02/provenance/score_recovery/` (28 controls); `runs/local_clock_crossover_20261016_main_attempt01/` (8 main scores); `runs/local_clock_crossover_20261016_boundary_completion_attempt01/` (2 previously unexecuted boundary scores). Main delivery has a derived24-row join preserving original receipts, five qualified rows/19 unsupported, raw source counts and null balanced means/CI. White-background five-pair plot was visually checked and copied to BM report assets; SVG92124ba3…/PNG55749181… remain linked to actual data. Local copies are not a certified cloud or full-machine backup; SHM is fragile.

## Changed Files

Five committed sources under `scripts/experiments/local_clock_crossover_20261016/`: pilot.py, score_worker.py, pilot_main.py, boundary_completion.py, analyze_delivery.py. BM report45/spec/done Task/map/index and report image assets, CONTEXT and short HANDOFF are local project documentation. Root independent numerical readers are preserved in main run's `root_qa_sources/` and scratch.

## Verification

Control ROOT actual QA381bd121b8b3a8fd2a4016243084a651986437c729aa462df5da2bff5e55245c checked28 physicalWAV/84arrays and original MFA math; main/addendum QA98e78a93046d90f96f7a52a9fe80513e78ef79687b82aa7cfaa828fd356f8ba5 checked10forwards/30arrays and all output MFA. Float32 distance reconstruction max error3.815e-6; metric and clock differences0. All actual support rows/31lags/self-index/RMS/PCM inputs passed. Original report44 QA was not repeated.

Failures stayed visible: attempt01 rate/t_out API conflict produced0WAV/0MFA/0scores; corrected attempt02 then hit ENOSPC after28WAV/11grids. Its original RUNNING receipt is historical, parent9272/239317 and final-MFA13243/261862 exited. Score-only recovery inSHM reused exact waves,28/28PASS; its worker15951 lacked captured start ticks due a shell logging defect, explicitly retained, never rerun. Only this task's transient MFA cache390files/963886811B was preserved inSHM before removing the exact old cache path, freeing755MB. Scientific originals and old runs stayed unchanged.

Main was rc0/66.18s, parent20334/383559/scorer21460/389855 gone. One mathematically inclusive4.0 map was rejected at4.000000000000024; a1e-12 floating-boundary correction completed only that previously unexecuted pair under the original[.25,4] range. No existing output was rerun. Derived delivery counts and all five D/C/clock values match ROOT QA; plot readback is legible. No implementation tests were added/run. BM readback confirms Task done with exactly one status/Observations, report concluded and spec CONCLUDED.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely commit `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223`.
- Report45, root_control_admission.json and root_actual_qa.json are the primary scientific resume pointers.

## Observations

- [result] Five qualified physical local-clock operations all improve C+1.770–+5.495; this is a two-speaker conditional subset, with19/24 mapping violations and0/4 complete-source eligibility.
- [decision] End the bounded pilot at four; keep missing/failed cases and frozen dose/range; no head or old experiments resumed.
- [decision] Future clock work should improve executable mapping coverage within the same four sources before adding scale; smooth/soft constrained maps are a proposal only.
- [next_step] Use report45 and the evidence map for the next user-directed bounded clock-method or paper-writing task; no new experiment is pending.

## Relations

- continues [[Codex checkpoint - 2026-10-16T02-47-05Z - Bounded local clock crossover authorization]]
- relates_to [[45-local-clock-crossover]]
- relates_to [[local-clock-crossover-20261016]]
- relates_to [[研究主线与证据地图]]