---
title: Codex checkpoint - 2026-10-14T14-35-01Z - Native radial evidence recovered
  and bounded missing-cell preparation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t14-35-01-z-native-radial-evidence-recovered-and-bounded-missing-cell-preparation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 14:35:01.617428+00:00
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

# Codex checkpoint - 2026-10-14T14-35-01Z - Native radial evidence recovered and bounded missing-cell preparation

## Summary

Recovered existing canonical radial ENV evidence instead of repeating four renders, completed one S0052 ASR check and independent-reference assets, and prepared six missing native model-factor candidates on CPU while repairing an unavailable cloud TTS model.

## Story

The original goal is to explain TTS's TFG advantage and build an enhancer whose generated video is scored and delivered with unmodified natural audio. The user's latest direction is a small, fixed-RMS/content/clock native-space factor experiment with independent mouth motion, using original GPT-6 Luna/max agents for code and execution. ROOT owns theory, design, BM and one independent actual-array check. Reliability must be sufficient for the next decision rather than expanding an unconfirmed method or repeating broad audits.

Canonical available factors, isolated angular SEP and isolated angular STAB lack two-speaker replication; their old stop rules remain closed. A source-linkage correction matters: MuseTalk/Ditto canonical ENV already uses the v2 PURE radial coordinates downstream. The coordinate generator's renderer=false was not evidence of absent downstream runs. ROOT loaded the four actual canonical NPYs and source NPZs, confirming unit-direction drift <=1.45e-8 and total native energy drift <=2.87e-10. This prevents four redundant renders. Radial norm is a native-feature coordinate, not physical spectral envelope. Clean native coordinates cover six of nine model-factor cells; physical acoustic spectral-envelope cells remain untested.

ROOT admitted CPU-only preparation of the other three cells: MuseTalk SEP with the existing actual-rival rotation rule, Ditto stability increase by0.008 rad explicitly counter to observed TTS direction, and separately named Wav2Lip radial ENV v2. Agent has written six float32 arrays and receipt; its final process rc1 is a post-write GBK console print error, preserved in a create-only completion repair without regenerating arrays. Agent readback passes; ROOT's independent geometry review and any <=6-render admission are still pending. Wav ENV FIT directions are heterogeneous (48/49 eligible phones have mixed speaker signs); historical v1 sign gate0.530017/UNAVAILABLE is unchanged. This fills untested model scope, not retroactive reconsideration of old four-row gates.

## Working State

Authoritative navigation is basic-memory/Research/研究主线与证据地图.md, active Task and HANDOFF.md. HANDOFF was reduced to a current75-line snapshot; its full old content is preserved at .codex/scratch/handoff_snapshot_before_donor_20261014.md (SHA ad66f3d7efbb381e39df4f5c9f11d74080f3296a83cc757a15567a9fc0b1f032), historical only. Nine tracked modifications and many untracked research files pre-exist; no commit or PR created. Scratch, ignored BM and remote SHM are machine-local and fragile.

Server wjj@192.168.50.2:22 stays on; shutdown cancelled. Latest actual GPU snapshot UTC13:38:23 is idle0%/5MiB/no compute apps; no new GPU task is admitted. Revalidate current owners before launch. Retired training/four-center/Dev/Confirm/archive branches stay parked.

Missing-cell CPU root: .codex/scratch/native_axis_model_factor_scope_cpu_20261014_attempt01/. Receipt SHA0738b04b8f748252b0395b192b7fd2066de20ced9c9453e771f64a200c194061; manifest SHAb41c2b8c096e6a0dad227341fcc816bc0272759c4bf026b978cd3db2768b4d5b; completion repair SHAb83ab1fe2a2ece6f9c00dcd4cfacfc67c86ab78b7b4f8291a11b5d3b02049b6b. Geometry source and per-array file/raw hashes are in receipt and manifest, not reconstructed from summaries. Scope remains CPU-only until ROOT review.

Canonical ENV independent actual-array verification: .codex/scratch/native_axis_pilot_mask_audit_20261014_attempt02/root_env_radial_readback_qa.json. Last completed STAB four-row raw/mouth summary remains .codex/scratch/native_axis_stab_ang_fit_20261014_attempt01/root_four_row_stability_summary.json SHAe56f5aa1106457716ae1bc79508005a65dd2fef8cb057dc36e0e71bdf4c42f2c, zero joint passes. Mouth amplitude/speed is descriptive, without real-video/human accuracy target.

S0052 one-shot Qwen ASR returned HTTP200 with enable_itn=false and verbatim “此前的战绩是三零胜六负。”; strip only terminal punctuation. Receipt SHAad918f21017b099de30caca9c84a4ea08d77b5d55b48d534aae73c0943190e36 in .codex/scratch/qwen_s0052_asr_check_20261014_attempt01/. S0044 text remains “用户可以设置和管理预设信息回复”. No further ASR is admitted. ROOT used the agent-authored script in a private ROOT PTY with F:/Apps/MiniConda/python.exe because agent sessions are scoped and bundled NumPy Python lacks requests. The secret never echoed or entered files/argv/BM/logs.

Independent same-speaker references are W0161(S0044,6.971s,SHAc2fa45b058ce62491541d055e098e161e4a4018314a166482a0b9bfa167851ed) and W0495(S0052,6.312s,SHAc0b1ab831fe5a4c04891a5c72d94c18f0105c13525524ff458221f199f3692ad). S0044 single-speaker tar was downloaded from fixed AISHELL revision bbe295d530192a4cd41644b711c9aecd087df653,30,157,135B, and exactly one eligible reference extracted by the frozen sorted rule. These are independent-reference clones, not self-clones. Input manifest .codex/scratch/qwen_independent_donors_20261014_attempt01/input_manifest.json SHA1668c859d03febd4296ae1671815a2ad7dfcccc9b5731298f38e76229e6d94f3.

The first HTTP worker is terminal rc1: S0044 enrollment200 followed by synthesis404 InvalidParameter/Model not exist for qwen3-tts-vc-2026-01-22. No audio exists; S0052 was never called. Preserve cloud_attempt01; never retry that worker/output root or reuse its old-model voice for another model. This is configuration unavailability, not a scientific negative. Decision037 selects documented qwen3-tts-vc-realtime-2026-01-15 with official SDK: new budget2 enrollments+2 WS sessions, no retries, same references/texts, raw PCM/WAV retained unchanged. Luna is preparing source and isolated pure-wheel runtime only; no new cloud request yet. Downloaded wheelhouse9.416MiB; no-compile isolated deps approved, no global install/NumPy mixing. Worker review and private getpass credential injection remain ROOT's next cloud gate.

## Changed Files

ROOT maintained map, active Experiment/Task and HANDOFF; added create-only ENV scientific QA, reference acquisition/input manifest and decision037. Luna created missing-cell CPU sources/arrays/receipts and cloud source/runtime proposals. Production implementation unchanged by ROOT; no unit tests added or run.

## Verification

Read project Router, experiment instructions, map, active Task/HANDOFF and bm-writing/checkpoint skills. Verified same-chat lineage via both paginated metadata searches (370 shared-identity results, legacy0) and directly read previous checkpoint13:49:31 with exact agent/session. Current git branch/root/SHA directly checked; nine tracked changes remain. GitHub read-only canonical repository query succeeded; matching branch PR query returned0. SHA previously returns422 on GitHub, so label local/unpushed rather than link it. Actual ENV arrays were independently checked; ASR and HTTP404 raw outputs exist. CPU six-array checks are currently agent-reported, not ROOT accepted. No implementation tests/pytest, new TFG render or training.

## References

- [Verified repository](https://github.com/dogLeft233/tts-exp); current local/unpushed SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.
- [Official realtime SDK](https://help.aliyun.com/en/model-studio/qwen-tts-realtime-python-sdk).
- [Voice enrollment API](https://help.aliyun.com/zh/model-studio/voice-clone-design-http-api).

## Observations

- [result] Canonical Muse/Ditto ENV already supplies four pure-radial endpoints; repeated rendering is unnecessary.
- [decision] Native norm/radial energy and physical spectral envelope remain separate estimands; the full user's scope is still active.
- [decision] Missing model cells get one CPU qualification and separately frozen <=6 short renders, not broad matrix reopening.
- [decision] Preserve old cloud404 and use a documented realtime SDK repair in a new create-only attempt with2+2 calls/no retries.
- [blocker] No donor audio yet; reviewed realtime worker and isolated dependencies are pending. This is engineering work, not missing user authorization.
- [next_step] Independently review the six actual CPU arrays once and the prepared realtime worker, then freeze the smallest corresponding execution budget.

## Relations

- continues [[Codex checkpoint - 2026-10-14T13-49-31Z - Isolated STAB closed and minimal donor asset phase]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- relates_to [[037-qwen-realtime-donor-assets-after-http-model-unavailable]]