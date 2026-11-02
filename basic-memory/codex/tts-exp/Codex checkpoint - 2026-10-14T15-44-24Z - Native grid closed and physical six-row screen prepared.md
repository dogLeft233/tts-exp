---
title: Codex checkpoint - 2026-10-14T15-44-24Z - Native grid closed and physical six-row
  screen prepared
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t15-44-24-z-native-grid-closed-and-physical-six-row-screen-prepared
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 15:44:24.322289+00:00
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

# Codex checkpoint - 2026-10-14T15-44-24Z - Native grid closed and physical six-row screen prepared

## Summary

Completed the clean native 9-cell grid and six physical-waveform native CPU encodings, preserved their independent actual-artifact checks, and froze a six-render physical spectral-shape screen without enlarging the cohort.

## Story

The original goal is to explain why TTS sometimes improves talking-face generation and turn that mechanism into a cross-TFG natural-audio enhancer. The user's current experiment holds generation RMS, natural content and sample clock fixed while separating native phoneme margin, local stability and local energy/spectral shape, with independent mouth motion. ROOT owns theory, design, BM and one practical actual-artifact review; original GPT-6 Luna/max agents write and execute code. The user explicitly asks to balance reliability and time cost, so existing baselines are reused and weak factors do not trigger larger searches.

The clean native-space grid is complete: 9 model-factor cells and 18 candidate rows against six reused no-ops. Four individual rows pass Cbest>0, Dmin<=0 and D0<=0, but zero cells reproduce in both speakers. The standalone tested SEP, STAB and radial-norm doses therefore stop expanding. Native radii are not acoustic energy or spectral envelope, and arbitrary contextual/overlapping feature edits need not correspond to a waveform. This leaves a meaningful physical-input question rather than a reason to repeat the grid.

A new waveform candidate uses existing 9-speaker/18-pair FIT audio and MFA, fitting phone-conditioned smooth log-envelope TTS-minus-natural corrections with equal speaker weighting, half signed median, clip3dB and at least3-speaker support. It retains original N and global RMS and compensates edited STFT-frame power. Actual OLA phase and local-power collateral is measured rather than declared zero. The two target candidates pass ROOT's disk-WAV control review, and all six existing Linux native CPU encoders now pass ROOT's actual-array source/shape/finite/clock review. The shared physical correction produces mixed native margins and stability, not a common response across models. This diagnostic does not select samples or tune dose.

A six-row video screen is frozen before effects: two speakers times Wav2Lip/MuseTalk/Ditto, no new baseline renders/scores, doses, seeds or training. Progression requires at least two models with both speakers passing the joint distance gate and mean delta C>=0.050 across those model rows. Only then consider a small opposite/active control and independent speakers. Larger motion is not automatically more accurate.

## Working State

Authoritative project navigation remains basic-memory/Research/研究主线与证据地图.md, active Task native-axis-factor-pilot-20261013 and HANDOFF.md. They currently describe CPU encoding as pending and must be reconciled with this newer checkpoint before another resumption. Existing formal numbered reports are read-only. Scientific scratch, untracked BM and remote SHM are machine-local and fragile; this checkpoint is not a complete backup.

Native full-grid ROOT aggregate: .codex/scratch/native_axis_model_factor_scope_render_20261014_attempt02/root_complete_native_coordinate_grid_summary.json, SHA 9e03f9c50dee1bb5300952572a3b88f9ac84fe97b7fc6c9679fa5bcb6449cc4f. Source raw-matrix and mouth summaries are linked inside. All old six missing-cell render/scorer jobs are terminal; do not rerun for missing OS return codes.

Physical candidate root .codex/scratch/physical_phone_spectral_shape_fit_20261014_attempt04/: source SHA89c644de802bd8273c792671e12ab27df42dc96decdb24288cb310106ffd24d0; receipt2d1e685b463304f8c3bcb31a8fd202631e12ac912079fe154d6cf809373ab24e; actual disk readback f3e443e8c37be4e0f3b1307fd0e1902a4355872b242c597160c3b5408b1546ad; ROOT WAV QA d6713acf080bcf3c2a370b44390c22df170937ed6319911079e6f99b532d6347. S0044/S0052 candidate WAV file SHA0ffdc74037d487f32fcade58840724308350c7f67b5ebce4f5200a79b1575b53 / 2c1208a7cd323fd7e4b888324b17a02af348739165dd42320c4f57c7174b6523, N59135/65855, 16k mono FLOAT32, RMS0.0376838173 within3.71e-10 relative error. Roundtrip WAV difference<=1.32e-17, not bitwise identity.

CPU encodings live in the same root/native_reencode_cpu_attempt02/. Executed source9486dac710b668e87b6227866809657a4454803c525551e4830f8b49a4c156cb; receipt2464541eacec8036d33d6c318ffaf44a0e8364527308695bd7261a544b9dfd2e; ROOT six-array QA0f991e8622305df1b9d7f46299ac40e578c94eac4582cc29d993fcfffb7f2042. All6 candidate file/raw pins are in these artifacts. Remote outputs /dev/shm/cross_tfg_physical_phone_native_reencode_20261014_attempt02/outputs. Actual PID2500176/startticks47652946 is terminal success in22.12s. Earlier attempt01 exited before encoding on a missing metadata constant;0arrays, original failure retained. New02 is a separate repair, not retrying unknown work.

CPU geometry receipt1dc5711e3473dd27361931b0435b295523ec5f1c125cceebf515f49a589a0374, summarya46cbcc5e6e24fa2e144cbc1ae5b20556a6c0c5c14738925d70c02ccee6ba6d6. Delta actual-rival margin S0044/S0052: Wav-.000704/-.000232; Muse+.000202/-.000471; Ditto-.000154/-.0000347. Dispersion also mixed. Encoder context halo beyond edited centers is expected for physical reencoding and not the old latent mask contract.

Independent exploratory vowel-opening contrast root .codex/scratch/native_axis_vowel_opening_contrast_20261014_attempt01/result/receipt.json SHA8063808fc654ba00b2d1ebb9532cf15a4eec04df499d0a5085fa981d835fc50e. Twelve existing tracks only; no GPU. Exact a/i/u token-centered contrast is UNMEASURABLE for S0044 because its sole a has no central20-80% frame; S0052 has only one central a frame. S0052 contrasts candidate-minus-noop: Muse SEP+.006714 despite C-.036; Ditto STAB-.000925 and C-.064; Wav ENV-.011890 despite C+.034 and worseD0. Descriptive support-limited dissociation, not ground-truth accuracy or a retroactive gate.

Frozen physical screen protocol root_physical_screen_protocol.json SHA6ca540bf8c7f7133445b827d69554fca472694a587d278b733968d2447a175b8, pending source admission. Same qualified CPU-encode-to-native-injection branch as reused no-op, physical WAV bound to candidate native array, exact Wav forward capture, original natural PCM for finalscore,25fps, same face/seeds/profile. Muse agent prepared .codex/scratch/physical_phone_spectral_shape_render_20261014_attempt01/ sources, not launched: source_pin_attempt01.json900f871cacf5e588f956d9fa7f8ed0db8d823e9c9b8c6603794bbc3455f4e637; Muse launcher19125639415aa62dc1486e62f863be1c90f19f004f035d6859246c59c41d70d2; Ditto launcher42b5f734e015905cbc4f0dd63624bf434122ac3781ff0654a816fea53f3c7b92. Ditto derived runner23bdd08758adcd233b1fbfc49b7a4e91b1ff79503a60efe41d0ead1433a1f4a2 only adds PHYSICAL CLI choice to actual dynamic v2 cf84ac09b2a8bc3c9442867c47a12abeb6eb4c3d9fa75d4dd2f68bb9854e304e. ROOT source review is next, no GPU launch admitted yet. wav2lip_minimal_gpu has been explicitly assigned separate two-row controller preparation. Muse coordinates serial pairs to avoid duplicate launch; angular agent is idle after CPU delivery.

Server wjj@192.168.50.2:22 remains on; shutdown cancelled. Agent read-only snapshot reports empty compute owners; fresh successful owner query remains mandatory immediately before a launch. Never resume old training/four centers/Dev/Confirm/archive jobs. Secrets stay private runtime/no-echo stdin, never files/argv/BM/log. Cloud donor branch is parked after one realtime session.created-only240s timeout,0PCM; no further ASR/API/debugging budget. Existing FIT data unblocks the physical line.

## Changed Files

ROOT created physical waveform/native actual-array QA and frozen screen protocol in scientific scratch. Original Luna agents added physical candidate builder/receipts/NPYs, CPU geometry, independent vowel diagnostic and prepared renderer sources. ROOT has not modified production implementation or committed anything. Nine pre-existing tracked modifications remain; many scratch/docs are untracked.

## Verification

Loaded Router, experiment instructions, map/Task/HANDOFF and checkpoint/writing skills. Same-chat identity search completed all371shared-session notes across2pages, legacy0, and verified the previous14:35:01Z checkpoint's exact agent/session pair. Direct git branch/root/SHA/status checked. Read-only GitHub canonical repo verified; current SHA422 means local/unpushed. Exact-head PR query returned an empty response (PowerShell array wrapping had briefly counted the empty collection as1; no PR exists). ROOT actual raw scores/mouth, disk4WAV and6candidateNPY checks ran once and passed their stated controls; no pytest or implementation tests. Physical renderer has not run yet.

## References

- [Verified repository](https://github.com/dogLeft233/tts-exp), local/unpushed SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907`.
- Current Task: tts-exp/tasks/native-axis-factor-pilot-20261013.

## Observations

- [result] Native SEP/STAB/radial grid has0/9two-speaker joint passes, despite4/18individual rows; standalone tested doses stop expanding.
- [result] Physical candidate and six real CPU encoder outputs qualify for the bounded renderer screen; native geometry response is mixed across frontends.
- [decision] Reuse no-op generation/score/mouth and review actual scientific outputs once; no full environment re-audit or automatic retries.
- [decision] Physical six-row progression is frozen before effects:>=2models both-speaker joint pass and mean deltaC>=0.050 before any controlled follow-up.
- [blocker] Renderer sources still need ROOT's narrow source admission; no physical video-effect evidence exists yet.
- [next_step] Review Muse/Ditto pair launchers once, admit and start the first serial physical pair, while Wav agent prepares its isolated two-row controller.

## Relations

- continues [[Codex checkpoint - 2026-10-14T14-35-01Z - Native radial evidence recovered and bounded missing-cell preparation]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- relates_to [[037-qwen-realtime-donor-assets-after-http-model-unavailable]]