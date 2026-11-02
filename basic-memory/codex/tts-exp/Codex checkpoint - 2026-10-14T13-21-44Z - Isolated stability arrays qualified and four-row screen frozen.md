---
title: Codex checkpoint - 2026-10-14T13-21-44Z - Isolated stability arrays qualified
  and four-row screen frozen
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t13-21-44-z-isolated-stability-arrays-qualified-and-four-row-screen-frozen
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 13:21:44.049299+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-14T13-21-44Z - Isolated stability arrays qualified and four-row screen frozen

## Summary

A small stability intervention now separates within-phone angular dispersion from frame norm and unit-centroid phoneme margin; four actual CPU arrays passed ROOT review and only four serial renders are admitted.

## Story

The original goal is a natural-audio head that improves TFG generation while restoring original natural PCM for final scoring. The user now prioritizes mechanism discovery at fixed RMS, content and natural clock: separation, local stability, local energy/envelope, and independent mouth motion, with GPT-6 Luna/max agents doing implementation and ROOT maintaining theory, design and BM. Reliability should not consume a broad audit budget before efficacy is established.

The available canonical operator matrix is complete (22/24 cells: 6 baselines and 16 factor endpoints; Wav2Lip ENV unavailable), with no factor passing both speakers in any model. Euclidean operators also changed native norms, so the cleaner angular/radial coordinates matter. The completed SEP_ANG_ACTIVE05 four-row probe preserves native norms and within-token Gram while increasing actual nearest-rival margins. Wav2Lip and Ditto both yield only about +0.006 Cbest on S0044, but regress -0.170/-0.080 on S0052; expansion is closed. Actual-nearest LOSO FIT means differ from fixed-rival means: Wav2Lip -0.000249, MuseTalk +0.003932, Ditto +0.003140. Tone stripping does not restore a common positive actual-margin direction. Existing mouth localization is weak and variable; aperture/velocity are motion measurements, not accuracy labels.

The next bounded hypothesis tests local stability independently. STAB_ANG_FIT contracts each token's unit vectors toward its original unit centroid, rotates their centroid direction back, then restores each original frame radius. Fixed rounded FIT-scale reductions are 0.004 rad Wav2Lip and 0.008 rad MuseTalk. The unit-centroid direction and actual margin remain unchanged, while dispersion falls. The common rotation does not preserve an arbitrary raw centroid; feature norm is not physical acoustic energy. Ditto is excluded because its observed FIT stability direction differs.

## Working State

Authoritative CPU packet: .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/. Executed source.py SHA 42a6e69b49fdf8f8609324e84afb65ba042436709bff2389e05168d44a75cd06; receipt.json SHA 13f0e8dc6b49ef98412ebbda09289da73743da3526fa9bfe2609adb3e46fd2cf; root_array_qa.json SHA 55d732257e3dc25abbc67b2576881c98f7c19d222993e012f9284a60429227aa. Four float32 NPY candidate paths and hashes are in the receipt. Every sample has 18/16 eligible tokens, all applied; norm relative error <=9.67e-9, unit-centroid drift <=1.96e-8, actual-margin drift <=7.86e-10, unsupported/short/unowned frames bitwise identity. Boundary collateral <=0.292 degrees is disclosed.

root_four_row_admission.json SHA f02c77c1fde62dab6109249583a3093e31fbd5f21d6998e737ec6a9921618b91 freezes four new renders only, Wav2Lip S0044/S0052 then MuseTalk S0044/S0052 serially. Reuse canonical no-op baselines. A speaker passes only when dCbest>0 and dDmin<=0 and dD0<=0; both speakers must pass per model, both models for a candidate spanning these models. Report motion separately, no n=2 inferential claim. No automatic dose sweep, new cohort, Ditto, SEP expansion, training, old centers, Dev or Confirm.

Machine-local execution: server wjj@192.168.50.2:22; user cancelled shutdown. Wav agent wav2lip_minimal_gpu has ROOT authorization to create a minimal copied launcher and launch once in fresh /dev/shm/cross_tfg_native_axis_wav2lip_stab_ang_fit_20261014_attempt01, candidate input root /dev/shm/native_axis_stab_ang_fit_candidate_inputs_20261014_attempt01. At this checkpoint no new PID is yet received: inspect actual job-control first and never duplicate an unknown launch. Its preflight UTC13:16:41 reports util0/5MiB/no compute apps; canonical Wav source hashes and checkpoint pins match. Muse agent musetalk_canonical_followup is preparing baseline/source/adapter equality only and cannot launch until ROOT releases it after Wav exit/idle. angular_radial_cpu_feasibility is complete. All reused agents are original GPT-6 Luna/max. Secrets stay in private runtime only.

Canonical prior packets, corrected manifest hashes and exact raw PCM/source pins are in the experiment note and previous checkpoint. In particular Ditto canonical score manifest actual SHA is 83b784e574b5893e4e2697dc3065e18fe924d0f6eed41b60b92c5392fe3b30a5; the immutable 12:05 checkpoint had a truncated transcription. Ditto angular compact readback is .codex/scratch/native_axis_sep_ang_active05_ditto_readback_20261014_attempt01/ (manifest dd5858fb...). Missing OS rc/S0052 worker PID remain null and do not trigger rerender for metadata alone. Its actual dynamic runner source cf84ac09... is valid; a local same-named fixed104-frame runner is stale.

## Changed Files

ROOT updated basic-memory/Research/研究主线与证据地图.md first, then basic-memory/Experiments/native-axis-factor-pilot-20261013.md, basic-memory/Tasks/native-axis-factor-pilot-20261013.md and HANDOFF.md with closed SEP results, qualified STAB arrays and the four-row admission. Scratch scientific QA/admission JSONs are create-only. No production code was edited by ROOT. Basic-memory and HANDOFF are ignored/local durable memory, not pushed code. The branch retains nine pre-existing tracked modifications and many untracked experiment/test scripts, recovered media and scratch artifacts; do not overwrite or claim them as this change. No commit or PR was created here.

## Verification

ROOT read AGENTS, project Startup Router, experiment instructions, active task, research map and current HANDOFF. CodeGraph provided executed STAB source. Independent CPU QA loaded all four actual arrays and their source tensors, checked source/file/raw hashes, float32 shape/finite values, per-frame norms, token mean-angle dose, unit-centroid preservation and bitwise identity masks; PASS. Initial QA schema lookup used token_results instead of tokens and failed before any output; corrected read succeeded without rerunning candidate construction. No pytest or implementation tests were run. BM search reindex completed after documentation edits. Checkpoint lineage search paged all 368 matches; no legacy matches, newest earlier same-chat checkpoint directly read and agent/session identity verified.

## References

- [Verified repository](https://github.com/dogLeft233/tts-exp) from read-only GitHub query earlier in this turn.
- Current local/unpushed SHA 5688784dd5c6c1272074f0c4b6c5ff79f117c907 was unresolved by GitHub (422); no commit URL is fabricated. PR state was not established, gh unavailable.
- .codex/scratch/native_axis_angular_radial_cpu_20261014_attempt01/sep_ang_active05/root_four_row_summary.json SHA 357b987e...: closed four-row SEP outcome.
- .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/root_four_row_admission.json: current finite GPU budget and controls.

## Observations

- [result] Canonical factors and norm-preserving angular SEP do not replicate across both speakers; their expansion is closed, not the long-term research goal.
- [result] STAB_ANG_FIT supplies four independently checked float32 arrays with isolated angular dispersion reduction, original native radii and preserved unit-centroid margins.
- [decision] Reuse valid baselines and collect source/actual input/PCM/clock/endpoints; do not rerender completed science to fill incidental job metadata.
- [decision] Limit the next screen to four short renders, serialize GPU users, and require two-speaker agreement before considering any expansion.
- [blocker] MuseTalk candidate source must match the actual canonical no-op forward source, beyond shape equality, before release; no efficacy endpoint exists yet for STAB_ANG_FIT.
- [next_step] Read Wav STAB actual controller/job-control and terminal endpoints once; after confirmed exit/idle and Muse source equality, release its two rows and evaluate the frozen four-row screen.

## Relations

- continues [[Codex checkpoint - 2026-10-14T13-11-23Z - Canonical factors and isolated angular SEP closed]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- relates_to [[开发跨 TFG 的自然音频增强头]]