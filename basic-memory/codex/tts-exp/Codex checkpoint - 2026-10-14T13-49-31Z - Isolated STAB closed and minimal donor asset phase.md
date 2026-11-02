---
title: Codex checkpoint - 2026-10-14T13-49-31Z - Isolated STAB closed and minimal
  donor asset phase
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t13-49-31-z-isolated-stab-closed-and-minimal-donor-asset-phase
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 13:49:31.751947+00:00
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

# Codex checkpoint - 2026-10-14T13-49-31Z - Isolated STAB closed and minimal donor asset phase

## Summary

The four fixed-dose angular-stability rows are complete with zero joint C/D passes; subsequent work is limited to two donor assets and one S0052 spoken-text check, with no new renderer or training.

## Story

The user wants a usable natural-audio enhancer and a mechanism for TTS advantage, and now asks for fixed RMS/content/natural-clock native separation, stability and energy/envelope interventions with independent mouth metrics. ROOT maintains theory/design/BM; original GPT-6 Luna/max agents implement and execute, with small efficacy gates before scale. The available canonical factors and angular SEP screens are already closed without two-speaker replication. This turn completed the cleaner STAB_ANG_FIT coordinate: spherical contraction with original frame radii and restored unit-centroid direction, using fixed FIT-scale reductions0.004 rad Wav2Lip and0.008 rad MuseTalk.

All four actual candidates passed ROOT CPU source/hash/norm/centroid/identity review before the four-render admission. Wav2Lip two rows then MuseTalk two rows completed serially on V100, reusing canonical no-ops. Original natural PCM and25fps were retained in final scoring. ROOT recomputed raw query×lag and mouth track arrays: Wav S0044 dC/dDmin/dD0=-0.038/-0.081/-0.238; Wav S0052=+0.028/-0.071/+0.046; Muse S0044=-0.046/+0.071/-0.067; Muse S0052=-0.079/+0.091/-0.001. All fail dC>0 && dDmin<=0 && dD0<=0. Mouth opening/speed change measurably but with no universal accuracy ordering. One Wav C increase accompanies worse original-clock D0, while the other C decrease accompanies lower D0; score contrast and timing distance must remain separate. The fixed-dose candidate is closed, not the long-term goal.

## Working State

Authoritative root .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/. source.py SHA42a6e69b49fdf8f8609324e84afb65ba042436709bff2389e05168d44a75cd06, CPU receipt SHA13f0e8dc6b49ef98412ebbda09289da73743da3526fa9bfe2609adb3e46fd2cf, root_array_qa SHA55d732257e3dc25abbc67b2576881c98f7c19d222993e012f9284a60429227aa, four-row admission SHAf02c77c1fde62dab6109249583a3093e31fbd5f21d6998e737ec6a9921618b91. Maximum norm relative drift9.67e-9, centroid drift1.96e-8, nearest-rival margin drift7.86e-10; unsupported frames bitwise identity. Boundary collateral<=0.292degrees.

ROOT final scientific summary root_four_row_stability_summary.json SHAe56f5aa1106457716ae1bc79508005a65dd2fef8cb057dc36e0e71bdf4c42f2c. Wav compact manifest local_readback_wav2lip_attempt01/compact_readback_manifest.json SHAe1a1fd636dd80657b9fbed2b5fe526fec7a78babe88d9c52ef418fad482f3978. Muse readback_compact_attempt02/compact_readback.json SHA58759f95a17e1481ab8909310447fe46fb735433f57303460e5cfa991929e49e; candidate score manifest eca600f505b3c3b7982b2f6671ee61250232054f0b971556acf493e3b48bb135. Wav forward arrays exactly match candidates; Muse direct-prompt path/hash/source binds actual arrays. All tracks complete; raw PCM S0044=7ddcdeb9… and S0052=451711ed…. Renderer mux driver PCM b59ec…/17a680… is distinct from final raw-original score PCM. Raw float32 reductions differ from receipt scalars by a few e-6; three-decimal values/gates agree.

Machine-local server wjj@192.168.50.2:22 stays on; shutdown cancelled. Wav controller2476069/ticks46888376 and workers2476076/ticks46888389,2476727/ticks46890689 exited rc0. Muse renderers2478690/ticks46934378 and2479386/ticks46943806 exited, OS rc unknown/null; no rerender for metadata. Scorer2480139 exited rc0, startticks unavailable. Latest actual snapshot UTC13:38:23 V1000%/5MiB/no compute apps. No scientific GPU process remains admitted. Old four-center, Dev/Confirm, training/archive branches stay closed.

Next hypothesis asks whether a structured TTS-native donor can provide a positive transfer upper bound before more heuristics. Prefer MuseTalk first; historical Wav phone-aligned25% TTS mel residual on30 LRS3 groups was negative and is not equivalent to complete donor unit directions at natural frame radii. Proposed structured vs within-phone phase-shuffled sham is not yet a tensor/render admission. Each current sample lacks located paired TTS donor. next_asset_phase_admission.json admits at most2TTS requests,2optionalASR calls and2voice enrollments, no downloads/new models/training/renderers. User already authorized cloud TTS/ASR. Inherited process/user/machine DASHSCOPE_API_KEY variables are absent. angular_radial_cpu_feasibility prepares a scratch private no-echo getpass PTY; ROOT will provide the user's existing key at runtime, never argv/file/BM/log.

Spoken-text review found S0044 official/MFA text agree. S0052 official, .lab and word-tier all say “此前的战绩是三零胜六负”; no existing ASR result was found. Forced alignment cannot independently resolve actual digit pronunciation. Latest ROOT instruction is exactly one S0052 Qwen-ASR check first, disable numeric normalization only if supported by the official API, retain raw response, and report before TTS. A normalized “30” alone does not distinguish 零/十. No transcript error is established. Agent is preparing only this ASR runtime; do not auto-synthesize or construct tensors until ROOT reviews the result.

## Changed Files

ROOT maintained basic-memory/Research/研究主线与证据地图.md, basic-memory/Experiments/native-axis-factor-pilot-20261013.md, basic-memory/Tasks/native-axis-factor-pilot-20261013.md and HANDOFF.md; added create-only scientific QA/admission/summary JSONs in scratch. Agents created isolated candidate/runners/readbacks, no ROOT production code edit. Nine tracked modifications and many untracked research files pre-existed; Git SHA/branch unchanged, no commit/PR created. BM/HANDOFF, scratch and remote SHM are machine-local/ignored and require preservation.

## Verification

AGENTS/Router/map/experiment instructions/task/handoff read; executed STAB source read through CodeGraph. ROOT independently loaded all4candidate arrays, checked hashes/shape/float32/finite/radii/centroid/dose/identity; checked score matrices and mouth NPZs, raw PCM/clock/source bindings. Scientific gate0/4; engineering outputs valid. Initial ROOT CPU QA used wrong token_results key, failed before output; corrected tokens field succeeded. No implementation tests/pytest run. BM full-text reindex completed after main results; latest asset-phase prose may need one final reindex. Lineage search newest verified agent/session checkpoint was13:21:44, directly read; legacy search earlier returned0.

## References

- [Verified repository](https://github.com/dogLeft233/tts-exp), read-only GitHub identity verified earlier in turn.
- Current SHA5688784dd5c6c1272074f0c4b6c5ff79f117c907 is local/unpushed (GitHub422), not linked as a remote commit. PR state not established, gh unavailable.
- .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/root_four_row_stability_summary.json is current four-row evidence.

## Observations

- [result] Norm-preserving angular stability at these fixed doses changes mouth motion but has zero joint C/D passes across4rows; expansion closed.
- [decision] Preserve score, timing-distance and motion endpoints separately; native L2 radius is not physical acoustic power, and overlapping chunks may be waveform-unrealizable.
- [decision] Continue with a small structured-donor upper-bound question, first resolving content/assets, before admitting another renderer budget or training.
- [blocker] Runtime key is absent from inherited environments; private PTY preparation is underway. This is credential plumbing, not missing user authorization.
- [next_step] Read Luna's prepared ASR source/private no-echo PTY session, provide existing credential only via private stdin, then review the single S0052 ASR result before freezing donor text.

## Relations

- continues [[Codex checkpoint - 2026-10-14T13-21-44Z - Isolated stability arrays qualified and four-row screen frozen]]
- relates_to [[tts-exp/tasks/native-axis-factor-pilot-20261013]]
- relates_to [[tts-exp/experiments/native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]