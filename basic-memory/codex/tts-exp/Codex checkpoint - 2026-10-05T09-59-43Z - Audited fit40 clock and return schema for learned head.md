---
title: Codex checkpoint - 2026-10-05T09-59-43Z - Audited fit40 clock and return schema
  for learned head
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t09-59-43-z-audited-fit40-clock-and-return-schema-for-learned-head
started: '2026-10-05T09:59:43.7439051+00:00'
agent: codex
capture: deliberate
repo_root: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
status: open
project: tts-exp
model: gpt-6.1-sol
hostname: LAPTOP-E133GSLE
git_sha: 16270c228a1dcd3419e67186d28a0335bf9c8dc3
repository: dogLeft233/tts-exp
username: dogle
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T09-59-43Z - Audited fit40 clock and return schema for learned head

## Summary

The fit40 frame mapping is independently qualified, and a fresh producer04 repairs the next renderer return-field mismatch; trained enhancement and unseen confirmation remain pending.

## Story

The user expanded the TTS mechanism investigation into a reusable natural-audio enhancement head across TFG models: generate with transformed audio, restore exact original PCM and timing, and improve Sync-C. Root owns hypotheses, frozen evaluation, private dispatch and independent artifact qualification; GPT-6 Luna/max agents implement and run experiments. Qwen ASR/TTS and data acquisition remain authorized. The already-qualified 60-item ASR batch has zero CER/disagreement, so no retranscription was introduced.

RMS remains a measured usable baseline: equal-model Dev gain 0.154 across Wav2Lip, MuseTalk and Ditto. Three fixed local spectral profiles failed the stronger three-positive-model and +0.050-over-RMS criterion. Qualified native encoder measurements on eight fit clips show much stronger local-profile response in MuseTalk than global-gain response, with Wav2Lip showing the reverse. These measurements motivate the frozen nine-parameter static/rising/falling band head, trained on independent fit40 for exactly 120 updates. They do not establish a paired local-profile video gain.

The first producer02 row exposed 95 decoded AVI frames versus 97 JPEG observations. A read-only diagnostic now closes this: the full512 and PCM16 mux clocks have 95 frames at 25 Hz; the pipeline starts 0,0.12,0.16 seconds and yields 95 decoded frames, while its JPEG observations contain two repeated frames. Root checked the complete monotone 95-to-97 mapping, endpoints, steps and exact repeat positions. The existing mapper already supports these repetitions, so producer03 removed only the redundant count equality check. Producer03 passed source/GPU/storage admission and ran, then failed in 16.102 seconds on a schema access: eligible_full_lag_track_count exists under tracking, while row assembly read video. It completed 0/40 with one failed row and 39 not run. Root downloaded and raw-hash-qualified the entire terminal receipt, execution state, stderr, worker state and failure. This is preparation failure, with no optimizer run or scientific negative result.

The implementation agent audited every renderer video/tracking/PCM/mel/face-crop/context access against the pinned helper and prepared immutable producer04. Its only behavioral repair moves that eligible-count access to tracking; source identities and supervisor receipt names move to attempt04. All 40 rows will rerun in a fresh namespace. Training and final rescore must be rebound to actual successful producer04 receipts after completion.

## Working State

Durable repository: branch codex/cross-tfg-audio-head-20261003, local/unpushed HEAD `16270c228a1dcd3419e67186d28a0335bf9c8dc3`. Last verified pushed commit is c4a1998. No branch PR exists. CONTEXT.md's current committed entry precedes producer03 failure; project Task69/Experiment68 and ignored HANDOFF.md capture the earlier clock-diagnosed phase and need their next meaningful reconciliation after fresh execution. Historical five dirty source/test files and many untracked sources remain preserved; no broad staging or general tests.

Frozen Dev20/10 male speakers × three seeds × Wav2Lip/MuseTalk/Ditto; Confirm80/40 unseen speakers plus LatentSync stays sealed until method freeze. Keep D0, Canchor, content and visual guards, complete failure denominator, exact natural PCM/clock scoring, and GUARD20 null. No trained checkpoint, final120 fit rescore, learned Dev180, Confirm or report26 exists.

Machine-local runtime: private PTY50274/bridge06 port10887; server root@connect.weste.seetacloud.com:12349, pinned fingerprint SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0. Secrets remain runtime-only. Remote root `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`, SHM root `/dev/shm/cross_tfg_audio_head_20261003`. Producer03 is terminal and no GPU experiment is active at this snapshot. Preserve its partial artifacts as failed evidence.

Authoritative evidence pointers:
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt02/frame_mismatch_diagnostic_attempt04/`: actual full clock/decoded mapping response 88,989 B, SHA `97cb279ad8b05fae0549fe50a41f3739ecd0c827892d5c0530c3420c9ca1c26c`; root_all_frame_map_qc_attempt01.json qualifies complete mapping and endpoints.
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt03/failure_readback_attempt01/`: actual 9,204 B producer receipt SHA `df456c1fdbe54c680e5c02bce2f2af25bb19ea3ae08836b39ce586c2cfcd9ce2`, full stderr and failure; complete 0/40 denominator retained.
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt03/compact_status_attempt04/`: actual terminal response SHA `27c9767b05201284a2646b484aa1bae9419fdce897f52db44dd9aeec71454b0df`; statusreader03 was never dispatched because its receipt basename was stale.
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt04/sealed_packet_attempt04/request_index_attempt04.json`: prepared/unexecuted seven-request packet, SHA `010f16a8de48445d6a643025807f9aebd9c7fe202f3ca22e36d35dcddb503922`. ZIP 148,340 B SHA `f2036db2aad8424e036a822a37c7f46bc7c74e838f3609d79f686cd9f17931b2`; producer04 62,375 B SHA `0ecfd1f3be25f6c56d76fa6fecd1ab5dd9b1868e03b5394f355939e74dc14df7`; supervisor04 35,021 B SHA `62af6b7be96520241c11c1238f436b7c2b328cd750278ac683dfb6744b0b41bd`. Root read the full minimal source diffs; remaining packet review is next.
- `.codex/scratch/native_encoder_ditto_retry_root_stage_attempt01/`: actual four-source Ditto staging and CPU preflight response 11,533 B SHA `3f1aa6f0fb8b92b9f59c63048d100790dae8fee4145c37ca885b9e4fc1e9479f3`. Effective nested HuBERT ONNX asset 1,460,740,880 B SHA `04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697`; CPU-only, no forward/video/score. Async packet01 remains unexecuted: root caught wrong runner metadata, undefined SHM variable, multiple JSON stdout and overescaped cap/logging. Mechanism agent prepares fresh packet02 for 56 fit-only CPU encoder forwards.
- `.codex/scratch/cross_tfg_audio_head_20261003/v4_memfd_holder_runtime_probe_attempt02/`: actual read-only holder/reference/resource response SHA `83588315abf891eaaf5842189f0554f6913aed4e0b7bee95b90d4fb824c6192c`, all 12 checks pass. Holder629596/start774776231/FD3, sealed 1,435,576,320 B memfd, only holder consumer, no retired-output references; expiry about 10:53 UTC. Memfd creation consumed cgroup memory but zero measured SHM capacity. SYS_ADMIN absent, so tmpfs enlargement is unavailable. Let TTL expire; no release or extension.
- `.codex/scratch/cross_tfg_audio_head_20261003/fit40_intermediate_retirement_source_contract_attempt01.json`: source-only backup candidate audit; its old producer binding and zero-head-before-archive ordering need fresh04 contract02. Consumers need caches/RMS/source WAVs/portrait/tracks/embedded geometry. Unused JPEG/intermediate AVI could be backed to local F with full byte/SHA strict restore and zero references if actual terminal40 leaves insufficient trainer reserve. No deletion authorized from source-only scope.

V4 and native W/M archive/strict-restore/exact retirement are closed and should not be repeated. SHM last measured free 4,311,363,584 B; cgroup headroom ~31.35 GB. Local F is qualified native fixed NTFS/Samsung NVMe, free 9,399,627,776 B; C ~6.5 GB, E ~1.3 GB. Baidu CloudDrive still requires whole-file SHA reread before recovery use. Training admission retains 4 GiB SHM/8 GiB cgroup/two fresh empty-GPU samples. Measure actual completed producer usage before choosing any further backed retirement.

## Changed Files

New immutable producer04/supervisor04 and local source/config packet are untracked, unexecuted. Task69/Experiment68 and ignored HANDOFF were written/read back with HEAD162. Learned Dev dispatcher/builder remain source-only; builder needs exact reused-model/score/preprocessing/clock evidence before qualification. Final-rescore26-source staging is already closed, but future training producer/runner constants must receive a fresh operational rebind.

## Verification

Root checked the entire actual frame-map diagnostic, source pins, all failure downloads, and actual Ditto CPU asset/config preflight. Root read complete producer03-to04 and supervisor03-to04 diffs; agent audited all renderer return accesses. Fresh GitHub repository/last pushed commit/branch PR query were read directly for this checkpoint. No pytest, training, Ditto56 forward, final rescore, Dev candidate or Confirm was executed in this segment.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local/unpushed commit: `16270c228a1dcd3419e67186d28a0335bf9c8dc3`.

## Observations

- [result] Exact first-row mapping accommodates two repeated JPEG observations without retiming source PCM or video scoring clocks.
- [result] Producer03 remains terminal preparation failure 0/40; every failed/not-run row is retained.
- [decision] Fresh04 reruns all 40 rows after the minimal renderer-field repair; all future closures bind actual receipts rather than old failed namespaces.
- [decision] Preserve frozen cohorts, original PCM/clock, scientific guards and fixed120 updates; the trained method is still pending.
- [decision] Let sealed memfd holder expire naturally because its release does not recover SHM capacity; qualify local F for future exact backups.
- [blocker] Full40 successful geometry and fresh training source closure are required before any optimizer update or method evaluation.
- [next_step] Complete root packet review, stage/preflight and launch producer04 in a fresh namespace, then bind successful full40 geometry to training120 and final rescore while completing Ditto56 CPU encoder evidence.

## Relations

- continues [[Codex checkpoint - 2026-10-05T09-18-35Z - Verified storage recovery and fit40 clock repair]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]