---
title: Codex checkpoint - 2026-10-16T04-17-47Z - Bounded local clock result and coverage
  limit
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t04-17-47-z-bounded-local-clock-result-and-coverage-limit
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c6c1e7bd005abeaa4a2cc52991391ef2aaec5223
started: 2026-10-16 04:17:47+00:00
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

# Codex checkpoint - 2026-10-16T04-17-47Z - Bounded local clock result and coverage limit

## Summary

The authorized four-source local-clock pilot is complete: five qualified pairs improve Sync-C, but mapping coverage fails the planned extension gate, so the experiment ends at four sources.

## Story

The original project asks why TTS can increase talking-face synchronization scores and sought a transferable natural-audio head. The user subsequently paused that head, excluded LibriSpeech from independent conclusions, and challenged whether crossed natural/TTS scores merely reflect different utterance timing. Report44 reproduced large ordinary within-class instance mismatch. The latest request was a bounded clock experiment using four existing-video sources, a known misalignment positive control and sham, with expansion only after control and coverage success. ROOT owned theory, design, Basic Memory and actual numerical QA; GPT-6 Luna/max agents implemented and executed it.

Report45 now records actual waveform MFA phone-center remapping against an endpoint-affine sham with identical output length and RMS. Video features, original full31-lag support and original self-lag stayed fixed. Three of four known240ms controls passed both timing and score gates; identity score drift was about1e-6. S0008 repair audio was scored, but its fresh repair TextGrid was lost to ENOSPC and remains unmeasurable.

Five of24 planned main pairs qualified, from S0003 (four) and S0007 (one), spanning natural within-class, TTS within-class and both cross-class directions. Every qualified pair improved C by+1.770 to+5.495 and fixed-lag distance by4.967 to7.784; actual center MAE fell from241.8–594.7ms to7.7–16.3ms. Nineteen maps exceeded frozen[.25,4] strain limits, seventeen involving internal intervals. Complete within/cross coverage for both reference videos was0/4 sources, so no next-four expansion ran. These five rows are not five independent people; no balanced cohort mean or CI is claimed. Pixels never changed: this is the conditional effect of MFA mapping plus phase-vocoder rendering on evaluation audio, not evidence of better mouth motion or a transferable head.

## Working State

Primary scientific pointer: [[45-local-clock-crossover]], file `basic-memory/docs/experiments/45-local-clock-crossover.md`. [[local-clock-crossover-20261016]] is done; spec CONCLUDED; [[研究主线与证据地图]], numbered index, CONTEXT (99lines) and short HANDOFF reflect the result. No authorized experiment remains pending. Audio-head work, old training/centers/Dev/Confirm and closed audits stay paused.

Source commit `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223` is local/unverified remotely, contains only five new pilot scripts, and has no PR. Current direct git evidence: nine inherited tracked dirty entries and1004 untracked entries; preserve them. GitHub read-only canonical lookup confirms repository and exact-head empty PR array. Do not push or invent a commit URL.

Machine-local evidence lives in `runs/local_clock_crossover_20261016_attempt02/provenance/score_recovery/`, `runs/local_clock_crossover_20261016_main_attempt01/`, and `runs/local_clock_crossover_20261016_boundary_completion_attempt01/`. Main delivery includes24 derived rows, five qualified and19 unsupported, null balanced means/CI and a visually verified five-pair figure. Local readback is not a full-machine/cloud backup; remote SHM is volatile. LAN `wjj@192.168.50.2:22` stays running. All recorded launcher/scorer/MFA CLI PIDs were confirmed absent. Secrets never enter files, argv, BM or logs.

## Changed Files

Five committed files under `scripts/experiments/local_clock_crossover_20261016/`: `pilot.py`, `score_worker.py`, `pilot_main.py`, `boundary_completion.py`, `analyze_delivery.py`. Project documentation updated report45, spec, done Task, evidence map, experiment index, report figure assets, CONTEXT and HANDOFF. Independent ROOT readers are preserved under the main run's `root_qa_sources/`.

## Verification

Actual38 audio forwards, zero new TFG/TTS/ASR, zero installs or implementation tests. Control QA `root_control_admission.json` SHA381bd121b8b3a8fd2a4016243084a651986437c729aa462df5da2bff5e55245c verified28WAV/84arrays and actual MFA mathematics. Main/addendum `root_actual_qa.json` SHA98e78a93046d90f96f7a52a9fe80513e78ef79687b82aa7cfaa828fd356f8ba5 verified10forwards/30arrays and all actual output MFA. Float32 distance reconstruction maximum discrepancy3.815e-6; metric/clock discrepancies0. RMS, PCM, frozen support and original lag checks passed. The white-background plot was visually inspected. Report44 mathematics was not repeated.

Failures remain preserved: attempt01 API conflict produced zero scientific outputs; corrected attempt02 produced28WAV but ENOSPC interrupted S0008 repair MFA. Its RUNNING receipt is historical, its processes exited. Score-only recovery reused exact waves and passed28/28; worker startticks are unavailable due shell logging and were not fabricated or rerun. Only this task's transient MFA cache390files/963886811B was preserved inSHM before removing that exact old cache, freeing755MB; scientific originals and old runs stayed unchanged. A mathematically inclusive4.0 map rejected at4.000000000000024 was completed once with1e-12 boundary semantics, only the previously unexecuted pair; existing outputs and scientific limits stayed unchanged. Derived delivery counts match actual ROOT QA.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely commit `c6c1e7bd005abeaa4a2cc52991391ef2aaec5223`.
- Report45 and the two ROOT QA receipts above are authoritative; HANDOFF is only a resume snapshot.

## Observations

- [result] Known-clock controls pass3/4; five qualified pairs from two speakers improve Sync-C+1.770–+5.495; nineteen maps are unsupported and complete-source coverage is0/4.
- [decision] End this bounded pilot at four and preserve missing/failed cases; do not automatically resume head work or expand the cohort.
- [blocker] Executable local-clock mapping coverage, rather than score direction in qualified cases, limits a balanced conclusion with this implementation.
- [next_step] Use report45 and the evidence map for the next user-directed small mapping-coverage or paper-writing task; no further experiment is currently pending.

## Relations

- continues [[Codex checkpoint - 2026-10-16T04-10-21Z - Local clock pilot completed]]
- relates_to [[45-local-clock-crossover]]
- relates_to [[local-clock-crossover-20261016]]
- relates_to [[研究主线与证据地图]]