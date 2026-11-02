---
title: Codex checkpoint - 2026-10-05T10-36-35Z - Qualified complete fit40 geometry
  for frozen local spectral head
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t10-36-35-z-qualified-complete-fit40-geometry-for-frozen-local-spectral-head
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
git_sha: b20149b4a63063a6c1622899dd46cb37f08bf16a
model: gpt-6.1-sol
agent: codex
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
trigger: compact
hostname: LAPTOP-E133GSLE
project: tts-exp
username: dogle
started: '2026-10-05T10:36:35.9676326+00:00'
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T10-36-35Z - Qualified complete fit40 geometry for frozen local spectral head

## Summary

The 40-utterance RMS geometry producer completed and passed independent root qualification, providing the actual frozen inputs needed for the nine-parameter natural-audio enhancement head.

## Story

The project began by explaining why TTS sometimes improves Chinese TFG Sync-C. The user now requests a usable enhancement head that preserves natural speech and transfers across generators: enhanced audio drives video generation, and the exact original natural PCM and clock are restored for evaluation and delivery. Root maintains theory, the frozen protocol, independent qualification, Git and Basic Memory; GPT-6 Luna/max subagents implement and run experiments. Qwen ASR/TTS, reruns, public data and Baidu storage remain authorized. The existing ASR60 qualification has zero CER/disagreement, so the latest ASR authorization does not require retranscription.

All-utterance RMS is the qualified baseline, with equal-model Dev20 improvement +0.153602 over natural audio. Three fixed local spectral profiles failed the preset additional cross-model improvement criterion; this motivates learning a constrained local spectral head. The frozen head uses nine parameters for four frequency bands and static/rising/falling components, bounded to ±3 dB, with the same full-utterance RMS and unchanged source clock. Training uses fit40, 20 speakers, 120 predetermined Adam updates. Dev20 evaluation uses Wav2Lip, MuseTalk and Ditto, three seeds, complete failure denominators and D0/Canchor/content/visual guards. Confirm80 and unseen LatentSync remain sealed until method freeze.

Producer04 completed all 40 inputs in 328.596 seconds after fixing an actual 95-AVI/97-JPEG frame correspondence and reading eligibility fields from the renderer's tracking return object. Root downloaded the terminal producer receipt, worker state, geometry manifest and cache manifest and qualified all 40 rows and 1,280 target/prediction frame correspondences against the frozen PCM and clock. This is a usable training-input milestone. Numeric NPZ contents and the all-40 same-GPU zero-head gate have not run locally or remotely yet; no optimizer updates, learned weights, terminal rescore, candidate Dev videos or confirmation results exist.

Training attempt04 source/config is actually staged. Final-rescore attempt06 revision02 and binder07 are fresh source-only proposals against that staged configuration. The immediate execution dependency is recovering 284,491,776 bytes to restore the unchanged 4 GiB SHM admission gate; source-only inventory/relocation work is assigned to local_recovery. Preserve all training-critical source audio, caches, tracks, portraits, mappings and full512 outputs through zero-head qualification. Use exact unused artifacts, verified backups and process/queue reference checks for storage recovery.

Ditto's separate 56-cell native-encoder experiment stopped during construction, producing zero arrays. A read-only actual configuration/source audit identified device=cuda and force_ori_type=true inside parse_cfg's nested Hubert configuration. Execution ignores the diagnostic probe-plan override. Before changing this route, mechanism_audit prepared a read-only frozen-commit wrapper-source request, because the force_ori wrapper may perform extra preprocessing. Constructor failure is an engineering failure and contributes no scientific effect estimate.

## Working State

Durable tracked change is local unpushed commit `b20149b4a63063a6c1622899dd46cb37f08bf16a`, on `codex/cross-tfg-audio-head-20261003`: CONTEXT.md and the immutable producer04/supervisor04 sources. Five historical dirty source/test files remain untouched by this commit. No current PR was found by a fresh GitHub head-branch query. Required local Git status is preserved in `.codex/scratch/checkpoint_EK_git_status.txt`.

Machine-local evidence lives in ignored/untracked `.codex/scratch/` and runs. Remote namespace is `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`; SHM namespace is `/dev/shm/cross_tfg_audio_head_20261003`. The authorized endpoint is root@connect.weste.seetacloud.com:12349, pinned host fingerprint `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`. Password/API key exist only in private runtime variables; never place them in files, BM, command arguments or output. Private PTY50274 and bridge06 port10887 are fragile dispatch handles, not resumable authorization. Producer controller/child are gone; training and Ditto workers are not active. V4 memfd holder was read-only qualified, expires around 10:53 UTC, and has zero SHM allocation benefit; do not extend or signal it to solve SHM.

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_rms_geometry_producer_attempt04.py`: renderer field repair and measured frame mapping, SHA `0ecfd1f3be25f6c56d76fa6fecd1ab5dd9b1868e03b5394f355939e74dc14df7`.
- `scripts/experiments/cross_tfg_audio_head_20261003/supervise_fit40_native_rms_geometry_producer_attempt04.py`: attempt04 supervisor, SHA `62af6b7be96520241c11c1238f436b7c2b328cd750278ac683dfb6744b0b41bd`.
- `CONTEXT.md`: completed fit40 qualification, pending training/storage/Ditto route, under 100 lines. Ignored HANDOFF.md and Task70/Experiment69 still need the completed-40 snapshot.

## Verification

Root's actual `.codex/scratch/fit40_native_rms_geometry_packet_attempt04_revision02/terminal_readback_attempt01/root_terminal_manifest_qc_attempt01.json` passed; 2,580 bytes, SHA `cde4ed6d9ae3811200d2c722b4fbccad79a230b42a70f282496b15e1a15dc5a0`. It checks all four downloaded raw pins, canonical geometry payload, exact frozen roster/20 speakers, 40 complete PASS ledgers, original PCM16/float hashes, natural-clock queries, 31-lag limits, contiguous 32-frame windows, monotone prediction mapping, source RMS and mel parity. Maximum mel error was 9.5367431640625e-6; RMS error 1.880704883494566e-9. Canonical geometry payload SHA is `6accff30fa72755b9d7f430ea942cc8cd69c5892bd530d9a8cdf4fa4f3ab0857`.

The same terminal_readback_attempt01 directory contains producer receipt 22,215 B/SHA `2d1eccd05f95e7cb75dfb3b707216251a1b7c9a0641511fcdfca3366ba2f1994`, worker state 9,855 B/SHA `5817ebaf13241f144e22d7b4c03b247cdd8aef2a3265bf617b7ade20dc71cfa0`, raw geometry 1,414,953 B/SHA `7d70d3d691b3b11b23aaee97c436f7a7896a28bc6a6a41c68ddb5105ef898e1d`, cache manifest 218,125 B/SHA `3d2b4012ab958a1c20ed37e9b7416bad16e7487805bb043404deed03a3437c2a`. Execution_state is remote-observed only and must be rehashed remotely before use.

Training actual source stage receipt is 9,880 B/SHA `0deec08370bfac9e08522bfe2de2e6a13fd14672986291f07c7e4f731551d1c5`, at namespace `job_control/fit40_transition_head_v3_attempt04/source_stage_receipt_attempt04.json`; its stage response is `.codex/scratch/fit40_transition_head_training_packet_attempt04_revision01/root_stage_attempt01/05_validate_training_stage_attempt04_request.json.response.json`, SHA `15bf5f829ba3a8e09188bfa8e84875586aeb9868d72ca615a9ff2999940dcd2de`. Config SHA is `3c80d3ad5c93bd3d1f0d9e96572f65396aa6f7ff84a7acb86b3eeaecb34364b2`. Actual late binding and model execution remain pending; use the fresh poststage_bind_attempt01 request, not old producer02/config02 packets.

Ditto actual config audit response: `runs/cross_tfg_audio_head_20261003/native_encoder_ditto_config_route_audit_attempt08/requests_attempt01/`, raw SHA `8d2b1a6cf4f3fcc51687e17ee3e1d51ec8fda2aa7df4a4bca102e731b9fbb89a1`, PASS_CONFIG_PARSE_ONLY. It returned complete pinned cfg/wav2feat/hubert/load_model source and parsed nested fields without importing models or running audio. The next wrapper-source request is source-only, index SHA `6d9d19ba67ed0312cca1f4dec3f28751d87d1b21dcb4e06ebcc1ee93fd2f8813`. No general tests or pytest were run.

## References

- [Verified repository](https://github.com/dogLeft233/tts-exp).
- [Last directly verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97). Current `b20149b4a63063a6c1622899dd46cb37f08bf16a` is local/unpushed.
- Frozen training/rescore packets, terminal readback and root QC paths above are the authoritative execution evidence; proposals remain source-only until actual receipts exist.

## Observations

- [result] Producer04 completed and independently qualified 40/40 fit inputs, with the original natural PCM/clock and 1,280 mapped windows preserved.
- [decision] Retain the frozen RMS, ±3 dB, 120-update, full denominator and 4 GiB SHM gates; resolve actual storage and renderer/config defects without weakening scientific acceptance.
- [blocker] Training admission requires 284,491,776 additional free SHM bytes; exact qualified unused-artifact recovery is being prepared.
- [blocker] Ditto native features require auditing the frozen force_ori wrapper and applying one shared effective CPU-ONNX route to preflight and execution.
- [next_step] Recover enough SHM using exact unused artifacts with verified backup/reference safety, then actual-bind and execute the all-40 zero-head gate and frozen 120-step training.

## Relations

- continues [[Codex checkpoint - 2026-10-05T09-59-43Z - Audited fit40 clock and return schema for learned head]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]