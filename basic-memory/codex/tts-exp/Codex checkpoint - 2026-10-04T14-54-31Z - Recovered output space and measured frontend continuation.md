---
title: Codex checkpoint - 2026-10-04T14-54-31Z - Recovered output space and measured
  frontend continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t14-54-31-z-recovered-output-space-and-measured-frontend-continuation
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
username: dogle
git_sha: 10bbea2f3808c31d0fd36bd30d00f81268ca9221
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
project: tts-exp
agent: codex
repo_root: E:/Documents/tts-exp
status: open
started: '2026-10-04T14:54:31.9289893+00:00'
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T14-54-31Z - Recovered output space and measured frontend continuation

## Summary
The failed-fit output tree is now recoverable on two local disks and has been retired from shared memory; the next candidate generation is awaiting recovery of one old Ditto run while the actual Muse frontend runtime is being measured.

## Story
The original question was why TTS improves Chinese talking-face synchronization. The current objective is a usable natural-audio enhancement head that transfers across Wav2Lip, MuseTalk and Ditto, generates video from enhanced audio, and restores the original natural PCM and clock for primary SyncNet Cbest scoring. The user authorized continued experiments, Qwen ASR/TTS and local or network data; engineering failures must be repaired and unsuccessful candidates must lead to redesign.

Root maintains the theory, frozen evaluation protocol, independent evidence checks and Basic Memory. GPT-6 Luna at max effort owns experimental code and model execution preparation. The expanded fit cohort has 40 independently verified endpoint clips from 20 speakers. Confirmation data remain sealed. The three shared spectral profiles have fit evidence only; their 540 planned development renders and 900-row comparison have been staged, not executed. The latest qualified V4 ASR check remains 20 clips, 297 characters and zero errors.

Storage recovery has advanced concretely: the failed-fit archive was fully transferred to C and F, strictly restored and independently checked. Exact retirement attempt02 removed only its output run tree; post-QC independently verified all 33 protected outside paths. The remote duplicate tar remains present. Its new cleanup packet fixes the previously identified generic inode check and will undergo read-only preflight before its single exact unlink. A fresh inventory qualified the real old Ditto retry04 path and its 239-pass/1-failed history; it has not yet been archived or retired. This sequence preserves recovery and historical failures while reopening shared-memory capacity for generation.

The CPU frontend probe exposed an environment assumption: base Python lacks Transformers, and the plan-pinned Muse overlay does not match the assumed 4.44.2 distribution. Those two actual failures occurred before audio features were produced. A fresh read-only runtime discovery queries the deployed module version, source and callable signature without a version gate, audio reads or encoder forward; its measured result will govern the repaired probe. Ditto fixed-affine input constants have already been independently qualified, but they describe the input adapter rather than HuBERT encoder behavior.

## Working State
Durable protocol and code are on branch `codex/cross-tfg-audio-head-20261003`, local unpushed HEAD `10bbea2f3808c31d0fd36bd30d00f81268ca9221`. Task snapshot44 and experiment snapshot43 were updated and directly read back in local tts-exp. Historical dirty files remain untouched; no unit tests were added or run.

Machine-local and fragile: private PTY 50274 retains runtime SecureStrings; bridge attempt06 is on port10887. Credentials must stay in runtime memory. Server endpoint `root@connect.weste.seetacloud.com:12349` is durable, and its host pin is recorded in existing access evidence. Shared namespace is `/dev/shm/cross_tfg_audio_head_20261003`; system namespace is `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`. Last measured shared free space is 3,205,025,792 bytes, below the unchanged 5GiB generation admission gate. No shared candidate run or terminal result exists yet.

## Changed Files
- `CONTEXT.md` and `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`: committed qualified expanded-fit and frontend input findings.
- `scripts/experiments/cross_tfg_audio_head_20261003/inspect_ditto_hubert_input_constants_attempt05.py`: committed deployed ONNX input constant inspector.
- `HANDOFF.md`: ignored local 50-line current snapshot.
- `.codex/scratch/failedfit_output_retirement_exec_attempt02/`, `failedfit_output_retirement_postqc_attempt01/` and `ditto_retry04_archive_inventory_attempt03/`: actual local retirement and inventory evidence.
- `.codex/scratch/failedfit_archive_tar_retirement_attempt02/` and `cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt01/request_index_attempt05.json`: prepared, unexecuted next packets.

## Verification
Actual failed-fit run retirement receipt SHA `f56efc6d8a5ccd9f9ee01233c2464477fcfbb6ec1ae7e18b400ff176222479a4`; fresh independent C/F full archive hashes both equal `234006a3cc5efef762f1a943b79529fe4443751fad7f56a76cf718579e888638`. Post-QC response SHA `ab2a5ab501807eab015f7ac7b4afaa8c671566526d31fb7f597bf2e4c9a3824b` confirms target absent and all 33 preserved-path identities. Actual old Ditto inventory SHA `b19eaaf3973706c63d5ec232046c7cb387d2744c5954346de116764cdb0aa0a2` verifies 61,562 files and 1,823,420,416 allocated bytes. Root statically reviewed ASR guard04 and visual join02 including canonical seals and baseline raw pins; neither postrun consumer has executed. Actual CPU inventory03 failed on a version assumption; runtime discovery04 is prepared and unchanged request SHA is `c24900e2174f89446ceb1f85ceefa5290474d98b622e610bc3304ff922ca1022`.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/develop-cross-tfg-audio-head-20261003/`: authoritative transfer protocol and iteration design. The predecessor checkpoint had a shortened incorrect directory label; this exact path is the current authority.

## Observations
- [result] Failed-fit run retirement and protected-path post-QC passed; the archive remains recoverable on C and F with a complete strict restore.
- [decision] Preserve historical failed attempts and require full recovery plus exact-path retirement before releasing remote output space.
- [decision] Derive frontend version and source contracts from the actually deployed runtime; do not install a guessed older version to make a probe pass.
- [decision] Primary endpoint remains original-audio Cbest, equal weighting across speakers and model families, with failed planned cells retained; confirmation remains sealed.
- [blocker] Shared generation is waiting for sufficient admitted shared-memory capacity; the CPU feature probe requires its actual runtime contract.
- [next_step] Execute the reviewed read-only Muse runtime discovery and failed-fit tar preflight, then recover and retire the qualified old Ditto output to launch the frozen 540-render development run.

## Relations
- continues [[Codex checkpoint - 2026-10-04T14-13-01Z - Qualified input constants and recovered experiment continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]