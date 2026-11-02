---
title: Codex checkpoint - 2026-10-16T02-47-05Z - Bounded local clock crossover authorization
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t02-47-05-z-bounded-local-clock-crossover-authorization
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5a8d006892aae7ce937c4b1dc48fc8e64ec48c24
started: '2026-10-16T02:47:05+00:00'
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

# Codex checkpoint - 2026-10-16T02-47-05Z - Bounded local clock crossover authorization

## Summary

The user authorized a four-source local-clock pilot with existing videos, a known timing positive control and sham, extending to eight only when controls work; implementation is assigned to GPT-6 Luna/max and no new scientific output exists yet.

## Story

The project originally sought the mechanism of TTS talking-face gains and a transferable natural-audio head. The human paused the head. Report44 then showed that ordinary independent natural and TTS realizations reproduce large crossed-score losses, so source preference cannot be inferred from the old natural/TTS matrix alone. The latest authorization now targets whether correcting local clock mismatch improves scores on fixed existing videos, with strict time-cost control.

ROOT read the BM router, experiment/task instructions, evidence map, report44 and historical timing boundaries. Two GPT-6 Luna/max agents completed read-only feasibility and design review. ROOT selected actual waveform remapping through a phase vocoder and official SyncNet audio forward, with cached video features. Cached-feature reindexing is not the physical positive control. Main comparison is ordered phone-center remapping versus endpoint-only affine sham; both share reference length/RMS. Identity shams estimate no-op processing damage. Known N1 distortion uses output-to-input g(t)=t+0.240*sin(2*pi*t/D), with repair estimated from fresh MFA of the distorted waveform rather than fitting its known inverse. This is a combined MFA-map plus waveform-rendering operation, not a guaranteed pure-clock intervention or human mouth-quality test.

## Working State

Authoritative completed baseline remains `tts-exp/docs/experiments/44-repeated-take-crossover` and `runs/repeated_take_crossover_20261015_attempt01/`. Natural/TTS within-class Sync-C losses3.650/4.275 and cross losses3.963/4.708 motivate this bounded probe. All report44 jobs/QA are closed; do not rerun them. LibriSpeech remains excluded from independent inference and the cross-TFG head remains paused.

New dedicated paths assigned but not yet created: `scripts/experiments/local_clock_crossover_20261016/` and `runs/local_clock_crossover_20261016_attempt01/`. First four sources S0003,S0006,S0007,S0008; next four S0014,S0015,S0016,S0020. Fixed reference videos N1/T1 per source, original report44 common real31lag support, primary lag frozen from each original self cell. Per source at most19 new audio forwards (four identity, twelve local/affine, three positive-control waves), no new TFG/TTS/ASR calls. A concise BM experiment/spec and Task are the next documentation work; current HANDOFF/map still show completed44 and need the new authorization at the top.

LAN wjj@192.168.50.2:22 public-key SSH and `/home/wjj/tts-audio/tts-exp/`; no shutdown. Existing wav2lip librosa1.0 supports arbitrary t_out; DSP planned n_fft512/hop80/Hann/centered STFT and exact-length ISTFT. Phase advances once per output hop without multiplying by local warp slope, as required for approximate pitch preservation. Scoring env has official MFCC dependencies. MFA2.2.17 requires its bin directory prepended to PATH, as the existing successful helper does; an agent's bare version probe omitted that PATH and failed initdb lookup, not an alignment failure. No installs, media DSP, scoring forward or new MFA run occurred in this turn at checkpoint time.

Machine-local inherited Git state is nine tracked dirty files and1004 untracked entries; current local HEAD is unchanged and unverified on GitHub. Do not stage or delete unrelated work. Runs/BM/scratch are local artifacts, not a certified cloud backup. Secrets never enter files/argv/BM/logs.

## Verification

Read-only imports/version/signature checks succeeded in the named LAN wav2lip/syncnet/MFA environments. Reviewer identified that identity shams do not match nonidentity artifacts; ROOT retained a conditional combined-pipeline interpretation and added endpoint-only affine comparators. Source/code implementation has not been reviewed or executed yet. No tests were added or run. Public GitHub REST confirmed the canonical repository and no exact-head PR; SHA lookup returned422, so no remote commit link is asserted. Same-chat searches paged386 matches and legacy0; direct prior note frontmatter confirms exact agent/session and started timestamp.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely commit `5a8d006892aae7ce937c4b1dc48fc8e64ec48c24`.
- `basic-memory/docs/experiments/44-repeated-take-crossover.md` and original arrays/receipts are the completed baseline.

## Observations

- [decision] Run only four sources first; expand to the remaining fixed four on positive/sham/actual-clock validity, never on the signs of the main alignment effects.
- [decision] Reuse fixed videos and original effective support, synthesize actual waveform time maps, and separate endpoint-only affine and local phone-center remapping.
- [decision] Keep head, old training/centers/Dev/Confirm and closed audits paused; stop this study at eight or a failed control gate.
- [next_step] Finish the concise BM protocol/Task, review the agent's dedicated source once, then execute the authorized four-source controls and pilot.

## Relations

- continues [[Codex checkpoint - 2026-10-15T11-37-36Z - Repeated take crossover completed]]
- relates_to [[44-repeated-take-crossover]]
- relates_to [[研究主线与证据地图]]