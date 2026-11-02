---
title: Codex checkpoint - 2026-10-19T18-47-36Z - Ubuntu cloud retirement with bounded
  SMB restoration
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t18-47-36-z-ubuntu-cloud-retirement-with-bounded-smb-restoration
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
username: LAPTOP-E133GSLE/dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1543e-4ce2-7401-bd94-9dfddaf8204c
trigger: compact
model: gpt-6.1-sol
started: 2026-10-19T18-47-36Z
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-19T18-47-36Z - Ubuntu cloud retirement with bounded SMB restoration

## Summary

历史runs的可恢复归档仍在进行，已确认只释放4.383754730224609 GiB；60 GiB目标尚未达成。

## Story

用户要整理分散实验数据、统一百度网盘目录，并把Ubuntu暂时不用的旧runs迁到网盘，尽量腾出60G以上。已统一12个云目录到B:/tts-exp，路径映射在_index/path_mapping.json，AGENTS.md与ADR049记录新根。当前选定7根原分配约61.456 GiB，保留10月/May活动实验、原始数据、模型环境，并排除存在外部引用的两根。源manifest逐文件SHA及全部POSIX元数据、gzip/PAX分片全流SHA、归档后源树一致性和删除前fresh源树/进程引用检查共同约束退休。Windows从B云端读取，逐文件实际落盘、重新打开做SHA，完整通过才允许精确unlink和空目录rmdir，源路径保留报告与CLOUD_ARCHIVE.json。

Windows的C/E/F都接近满盘，原本在F盘逐文件恢复的消费者被安全暂停，已恢复约25.25 GiB但这只是未闭合partial，不能作为删除证据。现采用Windows独立B读取、Ubuntu SMB Z仅作有限恢复临时空间的全量重试；Z不是备份，也不读取Ubuntu源或上传缓存来代替云端。SMB复用同名临时文件发生FileExistsError后，冻结新版本按每个成员使用唯一数字临时文件名，保留所有旧失败及原件。为避免串行耗时，两个消费者拥有不相交的批次，主聚合器只有全7根的闭合证据且实际净df增加至少60 GiB才宣布完成。

## Working State

持久导航：Task ubuntu-runs-cloud-retirement-60g-20261019和ADR050；云目录B:/tts-exp/ubuntu-runs-archive/20261019/batch02/<run>/attempt01，根README.md及批次catalog.json是可刷新导航。首根lrs3_phone_rules_full_20260921_proxy已全量恢复并退休，4.383754730224609 GiB真实释放。第二根phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3已完整归档36543219212字节/545片，335781成员；source-after通过但独立恢复尚未闭合，不可删除。余5根仍等待或在归档/恢复。

机器局部权威：Z:/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_controls_20261019/active_route.json。主coordinator_v9.py PID274316，client_attempt09，仅验证前两根，后五根只等闭合退休回执。coordinator_future_v10.py PID272072，client_attempt10，仅负责原索引02至06并有RAM门；两者均使用verify_live_smb_v2.py，SHA f7288a6c7fa196cc273772c529832a303b8e047841ce91cbce58caa401c69fc0。元数据目录为数字00至06。2026-10-19T18:45:59Z二者存活无failure，主第二根已恢复556537374字节/2070成员，未来第三根140489038字节/2086成员。first4.38外无其他已释放证据。

Ubuntu控制根/home/wjj/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_20261019_attempt02，冻结archive_remote_v3.py SHA d6e905893ca41bbd9096ecfed474e02d72657d2332aaaebfe07037fdbf6351cc。prefetch_future_archives_v1.py PID280863只串行归档后五根，不删除。upload_cache_guard.py PID161076仅暂停本次严格身份匹配的归档生产者，根空间小于3GiB暂停、大于4GiB恢复；完成后创建cache_guard.stop使finally恢复其所暂停的生产者。不得操作CloudDrive进程/配置/DB/缓存。

旧coordinator_v4.py PID265156仍暂停保留，一个NtSuspend计数由本任务持有，FILETIME134368934664940606。BASE/preserved_v4_attempt01/state.json记录身份、partial及SHA。新第二根闭合退休后再按身份终止旧消费者并标记TERMINATED_REPLACED，释放RAM。若用户清理F到至少1GiB并希望恢复旧进度，必须先停新主第二根验证器，再只NtResume一次，保留future分区；旧client_attempt04第三根目录已有PARTITION_STOP.json防止越界接管第三至七根，之后需纯聚合调用器。不得同时保留两个第二根退休owner。当前无该用户回复。

## Changed Files

本任务的代码与验证位于.codex/scratch/ubuntu_runs_archive_20261019；当前主SHA bfed9b81e2b86425d482875ca6cc578a3a3d4385f821b84f32d4a99b6f16de00、未来SHA e1589c7ba81d01841639137494c2cbc6c11a0f3bf4cae084909ce1471df63f69。冻结源已发布CONTROL/_verification_sources/parallel_unique_smb_v2/sources_manifest.json；云相同相对目录须独立B回读确认。AGENTS.md云根说明、ADR049/050、Task为本任务文档；仓库大量其他会话dirty/untracked研究代码必须保留。本地SHA未提交到GitHub，无本次commit/PR。

## Verification

先前已实际通过归档roundtrip/边界/无proof禁止退休3项、UTF8路由2项、Linux guard scope1项、live分片与端到端恢复4项、8MiB独立云读回、首根1446成员全量恢复与fresh源删除。新增SMB唯一文件名1024次真实写入/重开SHA/unlink含空文件通过；test_verify_live_smb_unique_v2.py两项和坏body/坏whole拒绝子场景通过；test_restore_partition_routing_v1.py两项通过，不相交所有权且全7处理不因rawsum超过65而提前退出。旧v7同名SMB失败保留在client_attempt07/failure.json，没有第二根退休。没有跑科学训练/推理或整仓测试。GitHub只读确认仓库、当前分支无openPR，当前SHA API422不能解析，因此只标本机SHA。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local-only Git SHA: 2b71639c3b555293375d248350999377168d515d
- Current route: Z:/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_controls_20261019/active_route.json
- Navigation: B:/tts-exp/ubuntu-runs-archive/README.md

## Observations

- [result] 云目录统一已完成；当前60GiB清理目标未完成，只有首根4.383754730224609GiB具备闭合退休证据。
- [decision] SMB仅作有限物理恢复scratch，独立输入仍为Windows B云端，全部原有SHA/成员/源树门保留；分区聚合不改变验收标准。
- [blocker] Windows可用空间不足，原25.25GiB恢复进度暂停且不可拼接当闭合证明，当前完整SMB重试较慢。
- [next_step] 继续当前09/10消费者和预归档队列，解决实际故障、发布闭合journals并终止被替代的旧消费者，直到全7闭合和实际df净增至少60GiB；若不足须fresh资格额外旧根，不能以盘点或重叠free差值代替净释放。

## Relations

- continues [[Codex checkpoint - 2026-10-19T15-10-38Z - Ubuntu runs archival active with independent restore]]
- relates_to [[ubuntu-runs-cloud-retirement-60g-20261019]]
- relates_to [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]