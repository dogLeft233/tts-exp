---
title: 012-verified-local-mirror-for-artifact-retirement
type: note
permalink: tts-exp/docs/adr/012-verified-local-mirror-for-artifact-retirement
status: open
project: tts-exp
decided: '2026-10-04T08:46:41.7577002+00:00'
---

# 012-verified-local-mirror-for-artifact-retirement

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## Context
跨TFG增强头的共享540开发运行需要至少5 GiB SHM。旧fit输出完整可恢复归档后才能退休，避免丢失失败分母、原始矩阵、媒体、代码及来源。第一次归档完成1,279,971,156字节SFTP传输，并严格恢复47,105文件。B盘百度网盘挂载副本的即时Python读回曾匹配全档SHA，但主代理稍后独立Get-FileHash得到不同SHA；文件大小相同，首1 MiB匹配，原因未确定。即时挂载读回因此不足以作为长期恢复证据。

## Choice and rationale
本次采用两个独立本地卷上的完整封印归档（C主档、F第二副本），加上C上逐文件严格恢复，作为指定旧输出退休的恢复前置。主代理已分别重算C、F全档SHA及大小，均为cd567ec15fade9d8f1d1ce6c0d1ae043e0c1d8042aa7178caf546c69b0f69039。这个替代保留可恢复性，不必依赖当前读回不稳定的网盘挂载。

上述前置只准许准备退休；真正移除前还必须用新鲜只读审计验证指定远端输出目录的精确路径、完整文件树及科学绑定、终态，并检查活动进程的命令、cwd、FD和待执行包没有必需引用。先前静态计划不替代这一检查。每次退休绑定自己的回执，不泛化到其他目录。

## Alternatives and consequences
等B挂载稳定后才能继续会延误540运行，且尚无可靠完成信号；单份C档或仅保留汇总分数不能抵御本地损坏或重建原始证据；删除缓存/原始媒体而不逐文件恢复会破坏证据完整性。这些方案不采用。

C和F占用更多本地空间，但只使用已核验的完整副本。B副本保持保护、标记UNRESOLVED，不能宣称云端可靠归档或直接删除它。此次仅可能退休 `/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_fit_recovery_attempt04/run`，父目录、代码、控制器及远端封印归档保留。旧Muse/Ditto基线目录另有当前评分/视觉引用，不能由本决定提前移除。

## Evidence pointers
归档主路径 `${USER_HOME}/.codex/tts-exp-artifacts/recoverable-output-archive-20261004-attempt01/`；第二副本 `F:/tts-exp-archives/recoverable-output-archive-20261004-attempt01-secondary/`。归档文件为 `shared_frequency_fit_recovery_attempt04_output_attempt01.tar.gz`。

恢复回执 `.codex/scratch/recoverable_output_archive_attempt02/full_restore_receipt_attempt01.json` SHA4f6483c0ad6399a97839a8afc8ed63b108563f5b60a10424d99dcd3d5fff423f，RESTORED_ALL_MEMBERS_SHA256_VERIFIED/47,105文件/3,537目录/1,395,866,176逻辑字节；内部manifest SHA f7c02fcb03aaf3cf9631fd4b1d031e789cf4e2578981353513b2e0cba852660a。

F副本回执 `.codex/scratch/recoverable_output_archive_attempt02/f_mirror_receipt_attempt01.json` SHA670e0f9388830e99b05249f3805d98e68b2ba6efad3edf9429bff10e18da76f1。原B回执仍保留，但后续B全档SHA9f7b957a1992815d08ab9f3c05310a66c53c912b327aef8ae1dbce882e505908与封印不同，不能用于退休门槛。

## Observations
- [decision] 指定旧输出退休可使用C/F两个独立本地全档封印加逐文件恢复，B当前不计入可靠恢复副本。
- [rationale] 两卷全档SHA经主代理独立复算匹配；严格恢复全部文件通过，保留科学及工程证据。
- [alternative] 不采用单份归档、只有分数汇总、或未核验云端挂载即时缓存作为恢复保证。
- [consequence] 每次退休仍须新鲜精确路径、文件树、科学绑定及活动/待执行引用审计；本决定本身未执行删除。
- [blocker] B挂载后续全档散列不一致，原因及远端云存储完整性未确认，副本保持保护。

## Relations
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]