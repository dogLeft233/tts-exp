---
title: Codex checkpoint - 2026-10-04T17-08-06Z - Running shared-model enhancement
  and queued encoder measurements
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t17-08-06-z-running-shared-model-enhancement-and-queued-encoder-measurements
hostname: LAPTOP-E133GSLE
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
git_sha: 41332dae20743b1d5f2554094c6a0c8d6a7575d2
username: dogle
status: open
project: tts-exp
started: '2026-10-04T17:08:06.8702015+00:00'
model: gpt-6.1-sol
agent: codex
branch: codex/cross-tfg-audio-head-20261003
capture: deliberate
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T17-08-06Z - Running shared-model enhancement and queued encoder measurements

## Summary

Launched the frozen 540-row three-model enhancement experiment after verified space recovery, staged the 168-unit native encoder probe, and kept the next historical-fit redesign separate from sealed confirmation.

## Story

The original question was why TTS improves talking-face generation. The user now wants a usable portable natural-audio enhancement head: h(x) drives the generator, while the delivered video restores the original natural PCM and source clock. Root maintains theory, experimental protocol, Basic Memory, private dispatch and independent checks; experiment/model code is delegated to GPT-6 Luna/max. Continue through failed candidates to a complete method. Qwen ASR/TTS is authorized when useful, with credentials only in private runtime memory.

V4 improved over identity on the full three-model development comparison but lost to fixed RMS in every model (equal-model Cbest difference −.044299629). The method therefore continues with three fit-only shared local spectral profiles, followed by a finite fit40 transition-gated redesign if needed. Actual frontend measurements show gain produces almost a common Whisper log-mel offset but also changes Wav2Lip spectral shape through its fixed floor. Native encoder direction and generated visual motion are separate measurements; input results alone do not establish those later effects.

Storage recovery is complete. The old Ditto run and its exact duplicate remote tar are both retired after C/F whole-hash recovery, strict restore and independent protected-path checks. The frozen shared540 packet actually passed CPU preflight and GPU admission and was launched once. Its latest read-only status is RUNNING with 17 PASS rows, no stderr and matching launcher/runner identities. Do not dispatch the launcher again. Postrun score/content/visual analysis must bind actual terminal artifacts, not this partial status.

## Working State

Authoritative task is [[开发跨 TFG 的自然音频增强头]]; experiment is [[跨 TFG 自然音频增强头的训练与原音换轨验证]]. Task47/Experiment46 have actual launch/recovery state. Protocol lives in `openspec/changes/develop-cross-tfg-audio-head-20261003/`. Development is 20 clips/10 male speakers × three seeds × Wav2Lip/MuseTalk/Ditto. Confirmation 80 clips/40 unseen speakers and reserved LatentSync remain sealed. Full cohorts, failed rows, RMS comparator, original PCM/clock, and independent content/visual checks remain required. Historical fit40 (20 speakers, fixed endpoint selection, five speaker folds, ≤6 new candidates plus identity/RMS) is qualified; no new coefficients are frozen.

Git branch `codex/cross-tfg-audio-head-20261003`, local unpublished HEAD `41332dae20743b1d5f2554094c6a0c8d6a7575d2`, no PR. Existing five dirty source/test files and many untracked historical artifacts remain preserved. `HANDOFF.md`, `.codex/scratch/` and most `runs/` data are machine-local/fragile. HANDOFF/CONTEXT still need the actual launch and tar-retirement state reconciled; do not read their stale NOTRUN lines as current execution truth.

Server endpoint `root@connect.weste.seetacloud.com:12349`, hostpin `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`. NS is `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`; SHM is `/dev/shm/cross_tfg_audio_head_20261003`. Private PTY50274, bridge06 port10887/PID252640 and SecureStrings were rechecked present. They are fragile; secrets never belong in files or memory.

Shared540 packet07 requests00–05 actually passed; status06 is read-only. Producer output is `SHM/shared_profile_shortlist_dev_attempt03_execution_attempt01/run`; controls `SHM/job_control/shared_profile_shortlist_dev_attempt03_execution_attempt01`. Launcher PID549953/startticks770442756, runner550013/770442788, wrapper550016. Frozen plan SHA `43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`; rows SHA `64a41fb0c76943ac535d38afa9347d9ca0e91ad075e54bb514be262945a63abb`. CPU preflight receipt7018B SHA `61d95357c72746e61981b960181721891b6e3a38208288986c67bb540bb05c9a`. Admission measured SHM6256889856B/GPU32230MiB/0apps and all29 source pins; every renderer/tracker/evaluator child retains ≥18GiB/two-sample/unknown-application gate.

The prior DH checkpoint admission path is wrong: correct index is `.codex/scratch/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/admission_snapshot_attempt04/requests_attempt04/request_index_attempt04.json`, SHA `f7bd978334ed677dcc77002cb3eb6b05a6e37b54d58cb007cedd06d53a8a12ea`. Actual admission outer SHA `f2f2dfa940052a0c5b96a6b46c9a3638692a3b70da8e2ade15227db82633e703`.

Native encoder packet04 requests00–03 actually staged PASS; requests04/05/06 remain unrun. Index `runs/cross_tfg_audio_head_20261003/native_encoder_probe_attempt01/request_index_attempt04.json` SHA `657eff15a7f26e4feda809522abcc4e9fb18b4d6edbcc9f8262fbdac1b540299`. Runner44757B SHA `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c`; --plan is original frozen360 fitplan997947B SHA `1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7`, separate 7-condition sidecar21529B SHA `f205263a89cc135dad52a7db5b11efca927aab457ee640c694bd050863af811f`. Eight fit WAVs × seven conditions × three native encoders=168. W2L/Muse GPU and Ditto CPU all wait for shared540 terminal; ONNX thread-pool contention has not been bounded. Readback packer is prepared at `native_encoder_probe_readback_attempt01/request_index_attempt01.json`, SHA `acd4d9db004ae89a853b01d47c904dce955763859dfb8bdbfee35042f776d8fe`, unreviewed by root/unrun.

900-row inputs are staged, analysis unrun. ASR private client08 is statically qualified, new API calls unrun; its README09/map09 use actual packet07 status06. Actual terminal response needs complete EOF/exit status and source hashes before ASR60 or postrun launch. Fresh postrun04 stage packet03 supersedes preserved packet02: index SHA `e2b5213f3603f273008442cf52892260a5f0578b3ff82e88e09a900f85cbf542`. Root has read the index; still must read all requests, worker03 diff (especially visual returncode), and verify ZIP closure before staging. Worker27758B SHA `0fe46e3bcd749d4a8d26b8c82f9bb1ad0ea87b3b75dc9cd43de7cbda3f90f53c`. No candidate visual production/join yet. Prior terminal discovery03 is prepared read-only, not root fully reviewed.

V5 flux-gated contrast scaffold was fully read by root and remains preparation, not an evaluated candidate: `v5_fit_expansion_preparation_attempt02/v5_local_transition_transform_scaffold_v1.py`, SHA `b34450397c5fa18b8cd4f820ae095b356499ecabb96b3c6a814b676642e63cbd`. Before freezing, retain mask/gate/contrast/centers arrays, use one common κ across ADR011 ablation cells, and report actual physical frame-power changes. Original coefficient phase during mask construction does not guarantee identical phase after ISTFT/reanalysis. Agents are preparing exact r090 compression-only control, source-bound fit40 materialization and finite local-transition hypotheses. No new fit or final parameter selection has occurred.

## Changed Files

HEAD contains measured frontend probe03/preflight03/runtime contract02, CONTEXT and transfer-iteration explanation. New native/postrun/retirement/scaffold preparations are local untracked artifacts. Task47/Experiment46 were updated and read back. No historical dirty code was overwritten.

## Verification

Root frontend numeric QC remains PASS: 640 recomputations/max2.657e−8; summary SHA `6af5b7a66a73f8a40138fb3cac2d98adcb33751327cce4bc5c75b8d77eaad888`, NPZ SHA `143daf5014c15ff4f59157744b809592f5dcfbbe4f70b8fef2bcb6c902232665`. The frozen relative_rms label denotes absolute RMS difference, not a ratio.

Old Ditto archive1615391171B SHA `d0b48b51c4a0c0ddf9a173fd14b0a4139069dfe48adac7a3a2726055813ae6ed` is preserved in C/F and F strict restore61562 files/4557 directories. Actual run-retirement receipt8146B SHA `504fb355aeb91a19fb718196e02b3b0f912d51f1c9cd4dd9afb04782a8f83cb2`. Actual duplicate-tar retirement receipt11025B SHA `edb6d32d3b75ff920de85d509b6773d1537e6a5b984cc94959a016647eef8c78`. Independent postQC03 actual PASS/outer SHA `69db5113d1654311ff36745fa60f1c3c3645d3b12ea27fb75b46af4876435aeb`: exact targets absent,31 remaining protected identities/eight controls/six C/F proofs unchanged. Do not rerun retirement or recovery.

Latest actual producer status outer SHA `b1be62a1d52cc77aa283acae1f5d5b06e8d7e1d7dc1c0437c84c8aed67245189`: RUNNING17/540 PASS, launcher/runner alive and exact startticks, no stderr, exitcodes null; partial results not a scientific endpoint. Source request06 SHA `2b90a8401a9406ca3150376424f003327a632bba6148a6bd0baf3f7456da0662`. Prior statuses preserved in `.codex/scratch/shared540_actual_status_readbacks/`.

Native runner/source/requests/pins and byte-preserving shell syntax were reviewed; all four staging responses passed. No encoder forward has run. `git diff --cached --check` passed for current HEAD commit. No pytest/unit tests were added or run.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local unpublished commit `41332dae20743b1d5f2554094c6a0c8d6a7575d2`.
- Actual postrun preparation: `runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/postrun_chain_attempt04/request_index_attempt03.json`.

## Observations

- [result] Frozen shared540 generation is actually running,17 PASS rows, after verified recovery and fresh resource admission; scientific comparison awaits terminal data.
- [result] Native encoder probe168 is staged and source-qualified, but no forward or video effect is claimed from frontend inputs.
- [decision] Keep full cohorts, RMS comparison, original PCM/clock and sealed confirmation; use historical fit40 for finite redesign after shared-profile readout.
- [decision] Keep native encoder forwards queued until shared540 terminates; CPU ONNX threading is not yet bounded for safe concurrent scheduling.
- [blocker] Actual terminal score/content/visual/encoder evidence, further candidate iteration, confirmation and usable delivery remain outstanding.
- [next_step] Follow the running shared540 to its actual terminal state and bind producer hashes for the900-row/content/visual readout, while qualifying the postrun staging packet.

## Relations

- continues [[Codex checkpoint - 2026-10-04T16-27-18Z - Measured frontend response and recovered shared-model execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]