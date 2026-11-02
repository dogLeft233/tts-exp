---
title: Codex checkpoint - 2026-10-16T11-21-59Z - Amplitude six-link validation completed
  with responder coverage
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-16-t11-21-59-z-amplitude-six-link-validation-completed-with-responder-coverage
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-16 11:21:59+00:00
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

# Codex checkpoint - 2026-10-16T11-21-59Z - Amplitude six-link validation completed with responder coverage

## Summary

The six-link amplitude study is complete: all20 speakers and160 video/score/mouth conditions are retained; fixed RMS has14 positive and6 negative effects with a primary interval crossing zero, while the predeclared+3dB contrast has15 positive effects and a positive simultaneous interval.

## Story

The original TTS-TFG project sought a transferable natural-audio head. The user later paused that head, excluded LibriSpeech from independent conclusions, and requested bounded experiments and less audit overhead. After repeated-take report44 and local-clock report45, the proposal delegation prioritized six amplitude links: pure scalar controls, new20 RAW/FIXED repeats, old40 internal mechanisms, full lag decomposition, old15 native TTS level gaps and a fixed five-level dose curve. P2 pause repair was parked independently. ROOT owned theory/freeze/BM and actual QA; the original GPT-6 Luna/max agents implemented and executed the experiments and focused tests.

The latest user asks whether score gains are relative to natural self-drive/self-evaluation and whether too few samples improve. The answer must distinguish estimands. Report45's+1.770–+5.495 is local versus endpoint-affine evaluation audio from another utterance instance on a fixed video; its5/24 executable pairs come from two people, with complete-source coverage0/4. It is not improvement over natural self. Report46 amplitude explicitly compares S(G(gN),N) against S(G(N),N), with identical original natural PCM evaluation, fixed crop/support and no clock change.

Report46 now closes all six amplitude links. In the new20 cohort, primary FIXED repeat-mean minus RAW repeat-mean is+0.0489019893, speaker-bootstrap95%[-0.0052390117,+0.1060568489],14/20 positive and6/20 negative. Both repeats have the same effect sign for every person. Four predeclared dose means at-6/-3/+3/+6dB are-0.283382/-0.100889/+0.066948/+0.064959, positive counts1/4/15/15 of20. The+3dB max-|t| simultaneous95% interval is[+0.012781,+0.121115];+6 crosses zero. Do not select an optimum or switch the primary to full sensitivity, whose FIXED effect is+0.079147[+0.020371,+0.140784].

The score decomposition and actual motion are separate: FIXED mean deltaB+0.046550, deltaDmin-0.002352 and deltaD0+0.074296. Mean mouth opening/absolute opening speed increase+0.00226070/+0.01976806 per second with positive descriptive intervals. All160 mouth rows pass100% common support, and ROOT reconstructs all17,240 condition-frame formulas exactly. This establishes real generated motion changes and bounded score sensitivity; it is not a perceptual or articulation-accuracy endpoint. Greater motion at+6dB does not produce a clearly greater C benefit.

## Working State

Authoritative report: [[46-amplitude-gain-independent-validation]], `basic-memory/docs/experiments/46-amplitude-gain-independent-validation.md`. Four PNG/SVG figures and asset manifest are in its assets/46-amplitude-gain directory, visually inspected. [[研究主线与证据地图]], numbered index, [[proposal-validation-p0-p1-p2-20261016]], Task Board and CONTEXT (99lines) were updated. The same proposal `basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md` now contains E16 and targeted amplitude changes; original cross-TFG/TTS/dataset section3.1 and appendixE remain. Its Chinese title was restored via edit-note after write-note temporarily replaced it with the filename; its permalink/path are unchanged.

The Task remains in-progress only because original P2 is a parked independent todo; phase amplitude_complete_p2_parked. No amplitude GPU units remain. The original goal tool stays externally paused with its original P0/P1/P2 objective; do not claim it complete or create a duplicate goal. Head training, old centers/Dev/Confirm, mapping expansion and broad scans remain paused. No permission request is pending.

LAN wjj@192.168.50.2:22 remains on, public-key SSH, repo `/home/wjj/tts-audio/tts-exp`. Unique amplitude run `/dev/shm/p1_gain_dose_response_20261016_attempt02` is terminal. Render160/160 and score160/160 exited rc0. Final query found only the existing gnome-remote-desktop-daemon PID2543/313MiB, which was left unchanged. AutoDL westd:18625 is historical; do not replay old tasks. SHM is volatile. Scientific readback is local, not a cloud or full-machine backup.

Canonical local closure: `runs/p1_gain_dose_response_20261016_attempt02_readback/final/`. Original score SHA29865f15c7859cd601f8c2e417d06e41cace193856b20d790d8f79fe256f0769 is immutable. Correct full score SHA56ce5546… is additive. Final joined input is `analysis_input_attempt02/score_results_joined_for_analysis.json` SHA5567f0393f40dc8ae77fc61fe66883b8d00431f9cf3f205b67169fe72e450509; canonical analysis is `analysis_results_attempt02/analysis_summary.json` SHA7b9ef0a1e8ce9c6efaa6c828168ba190a6df196b2c3a6dc453cfdc78e59d60c9. Final addendum09da85ba… names all repairs and sources. Source manifest rawf3cc98a1… differs from its canonical internal fingerprint9e44024f…; do not conflate them.

Old40 final evidence: `runs/recovered_20260930/independent_validation/amplitude_evidence_reconstruction_20261016_attempt03/`; old15: `.codex/scratch/native_level_gap_audit_attempt01/derived_attempt09/`. Direction C+0.161, mel interior+0.171; native T−N gap+0.438 to FIXEDq11+0.254. These are reconstructed historical evidence, not new independent replication. RMS consumer07 is the real RAW→FIXED manipulation; source_axis08 is different. Old148 evaluation videos were intentionally not retained; MEL original full-curve payload remains an explicit gap. Do not launch old video recovery or experiments to fill it automatically.

## Changed Files

Local commit `2b71639c3b555293375d248350999377168d515d` adds17 source/test files: the eight scripts under scripts/experiments/p1_gain_dose_response_20261016, old evidence reconstruction script, and their eight test files. No push/PR. Nine inherited tracked dirty entries remain untouched apart from the already-authorized CONTEXT documentation change; unrelated untracked artifacts remain. Raw runs, BM notes, helper readers and staging sources are machine-local/ignored, not all Git-backed.

## Verification

ROOT actual scalar QA a1ef9d50… verifies20 source PCM plus100 saved gain WAVs, exact float32 scalar samples, clock/RMS/spectrum and no clipping. `root_final_gain_actual_qa.json` SHA52a62bf9e36052d1fedf34f7ba392d193625a84f82a07120a1edb48cc57cd8c8 verifies160 video bodies,160 primary and160 corrected-full matrices, curves/anchors/primary and simultaneous intervals; independent distance maxerror3.815e-6, metric residual0. `root_final_mouth_actual_qa.json` SHA30f974e1742d122f3ff1106cf559c1fb48176973335ad9f36becc37f307cea8f verifies160 actual landmark trajectories and geometry/clock/speed formulas, residual0. `root_analysis_delivery_qa.json` SHA8ef77f6e9b915bf1e15aafe8070c7046ce413e550c418274d7b59164f761c263 verifies the repaired160-row mouth join and all28 dose-component means/intervals, residual0. Readers are under `.codex/scratch/proposal_amplitude_20261016/`. ROOT reused old accepted QA and independently checked actual P0 arrays without rebootstrap or model reruns.

Agent-run focused fixtures passed: source/runner/analyzer17, mouth8, full-context4, two derived renderers3 each, plus old reconstruction6. No whole-repository tests or new installation. New scalar audio uses existing models and original PCM; no new TTS/ASR/MFA/training. Existing source/spec/model pins stay frozen.

Material failures remain: first missing log parent0renderer child; first missing encoder helper0forward; full repair's initial dtype mismatch0array writes. The original distance helper truncatedA to min(A,V), leaving384 full-tail cells incorrectly padded; data-only full repair fixes them while primary overlap is bitwise equal. First mouth join edited copies and left NOT_MEASURED; canonical attempt02 repairs references and replays only data analysis. Primary/dose JSON and speaker_primary.csv are exact across both analyses. No TFG/SyncNet was rerun for these repairs. Renderer continuation parent identity was not separately captured; per-child identities and rc0 are recorded without fabrication.

BM readback previously lagged direct file edits, so current notes were explicitly indexed. A new report already written on disk caused write-note conflict; only that newly-created report was staged in scratch, then created through BM, returning action created at the intended path. No scientific content was discarded. Existing notes returned updated; no knowledge duplicated into main.

## References

- [Repository](https://github.com/dogLeft233/tts-exp), canonical read-only API verified. Exact-branch PR endpoint returned[].
- Local unpushed commit `2b71639c3b555293375d248350999377168d515d`; no unverified commit URL.
- [[46-amplitude-gain-independent-validation]] contains all denominators, CI types, four final figures, actual source references and replay command.

## Observations

- [result] All20 people/160 conditions completed; FIXED improves14/20 but primary CI crosses zero, while predeclared+3dB improves15/20 with positive four-dose simultaneousCI.
- [decision] Keep natural-self, native TTS paired scoring and mismatched-instance clock recovery distinct; preserve negative, missing and support denominators without changing primary/doses after outcomes.
- [decision] Amplitude task is complete; no automatic best-dose head, cohort expansion or restart of paused cross-TFG training follows.
- [blocker] P2 originalM and8 annotated pause spans are not fully frozen; this independent parked task keeps the original broader objective incomplete.
- [next_step] Use report46 and the updated proposal for paper claim refinement; keep P2 separately tracked and current GPU experiments stopped.

## Relations

- continues [[Codex checkpoint - 2026-10-16T10-15-49Z - Amplitude experiment launched and clock baseline clarified]]
- relates_to [[46-amplitude-gain-independent-validation]]
- relates_to [[proposal-validation-p0-p1-p2-20261016]]
- relates_to [[研究主线与证据地图]]
- relates_to [[从 TTS 高分到可控驱动：唇形生成的输入机制与迁移边界]]