---
title: 014-sealed-overlay-v4-recovery
type: decision
permalink: tts-exp/docs/adr/014-sealed-overlay-v4-recovery
decided: '2026-10-05T14:10:22.4723422+08:00'
status: open
project: tts-exp
tags:
- codex
- decision
- storage
- recovery
---

# 014-sealed-overlay-v4-recovery

## Choice and context

Use the first completely qualified persistent V4 backup and strict restoration before exact source retirement. Prefer the independent-host C archive now that its full transfer and independent member qualification have completed; the additional overlay copy remains a capacity-gated fallback. The user authorized recoverable storage work to keep enhancement-head training moving. This choice applies only to V4 dev180 and preserves ADR012's separate historical mirrors.

The raw overlay preflight passed holder/receipt/source/quiescence/reserve checks but failed capacity by 91,785,216 bytes, so no copy or deletion occurred. Read-only inventory found no sufficiently large eligible duplicate, with retained depth errors; it did not establish that all possible cloud mounts are absent. The separate C transfer subsequently completed. Root qualified its 1,435,576,320-byte whole SHA and all 49,297 member paths/types/modes/full-file hashes independently. Windows restoration04 failed because text-mode writes expanded newlines; the intact tar, failed tree and remote source were preserved. Fresh helper05 adds explicit binary exclusive writes and restores from the existing C tar into a new exact tree, without retransmission. Its actual preflight passed with 8,292,577,280 bytes free and a 4 GiB reserve; complete restoration is still pending.

## Rationale and alternatives

Whole-archive SHA plus complete typed-member and restored-file closure establishes recoverability. C provides the already-transferred independent-host copy; creating another server archive now would consume the overlay space needed by training sources without adding a prerequisite. A sealed memfd alone is transient. Removing V4 before restoration closes would lose the current evidence. Lowering the overlay reserve or deleting unbacked outputs remains rejected. A gzip overlay copy can be considered only if C recovery remains blocked and its compressed/decompressed whole-byte SHA, full restoration and fresh capacity gates pass.

## Consequences and affected work

Keep the C tar, its whole-member QC, inventory and successful exact restore. Preserve failed recovery attempts. Require full strict restoration, independent root qualification, source/holder/terminal pins, anchored exact inventory and two fresh zero process/queue-reference scans before source retirement. Retirement removes only the reviewed V4 run, preserving task code, controls, archives and siblings. Derive every raw pin from actual files; do not manually transcribe digests. If the overlay fallback is later used, retain 512 MiB overlay reserve, restore allocation plus 1 GiB SHM reserve and 8 GiB cgroup reserve. Recheck the fixed 4 GiB production admission after retirement; independently backed native112 output can be retired if it leaves training just below this gate. Scientific cohorts, score tolerance, training schedule, natural scoring PCM and sealed confirmation remain frozen.

## Evidence

- C:/v4-a03/root_c_archive_full_hash_qc_attempt01.json: actual independent archive/member PASS, raw723B/SHA ce59f8d0d1152966ef5bb42cb9a7e0b5cf23fb5c9fd700fa445eaac65fa2c38d.
- .codex/scratch/cross_tfg_audio_head_20261003/v4_memfd_restore_attempt05/: pinned binary-safe helper, actual preflight and running restoration; do not infer a completed receipt before exit.
- .codex/scratch/cross_tfg_audio_head_20261003/v4_memfd_direct_stream_attempt03/root_actual_stderr.txt: retained Windows restoration failure.
- .codex/scratch/cross_tfg_audio_head_20261003/v4_overlay_recovery_attempt03/: actual raw-overlay capacity failure without mutation.
- openspec/changes/develop-cross-tfg-audio-head-20261003/: frozen scientific protocol.

## Observations

- [decision] Prefer the first fully qualified independent C archive/strict restore for exact V4 retirement; overlay backup is a gated fallback.
- [rationale] The C whole tar and all members already pass; duplicate overlay writing would consume useful space while strict recovery can use the existing tar.
- [alternative] Use persistent compressed overlay backup only after whole compressed/decompressed hashes, full restoration and original reserve gates close.
- [consequence] V4 source stays intact until complete strict restoration and fresh exact-retirement gates pass; failed Windows restore is preserved.

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[012-verified-local-mirror-for-artifact-retirement]]