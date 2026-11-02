---
title: Codex checkpoint - 2026-10-19T19-26-05Z - Verified cloud retirement continues
  with resilient readers
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t19-26-05-z-verified-cloud-retirement-continues-with-resilient-readers
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
started: 2026-10-19T19-26-05Z
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-19T19-26-05Z - Verified cloud retirement continues with resilient readers

## Summary

Ubuntu旧runs迁云工作仍在后台执行，仅首根实际释放4.383754730224609 GiB；至少60 GiB的用户目标尚未完成。

## Story

用户要求找清分散的实验数据、把百度网盘项目目录统一，再把Ubuntu暂时不用的历史runs迁到网盘以腾出60G以上。12个云目录已归入B:/tts-exp，AGENTS.md与ADR049、_index/path_mapping.json记录映射。当前7个无保留树incoming symlink的旧根原分配约61.456GiB；保留原始数据、环境、模型、10月/May活动数据，另两根因外部引用排除。科学任务不恢复或重跑。

归档采用Linux逐文件SHA与POSIX元数据manifest和64MiB有序gzip/PAX云分片。Windows从独立B云挂载解析每个成员、验证全部逐片与全流SHA，普通文件真实写入有限scratch、关闭并重开SHA；归档后和退休前各完整重核源树、检查精确进程引用。只有闭合proof才精确unlink manifest成员并rmdir空目录，源原路径保留小报告与CLOUD_ARCHIVE.json。不能以上传完成、盘点体积或partial日志称已释放。

Windows C/E/F几乎满盘使旧F恢复读者暂停。SMB每文件操作偏慢，因此改成固定源码SHA的单SSH帧通道：独立Windows仍负责B读回/解包/hash，Ubuntu helper仅接收真实文件body并在自建数字scratch上xb写、fsync、关闭、重开SHA、unlink、返回计数与双SHA；它不读取原件或上传缓存，scratch不算备份。CloudDrive一度让两只Windows B句柄同时报OSError22，旧11/12失败保留。新reader有1200秒时限，重开同一不可变分片、按已返回给hash的字节count seek；底层已读但未返回的字节也不遗漏。13/14生成时默认Windows编码损坏中文路径，被相等门立即拦截，0新增退休；15/16改显式UTF8源读取与源码ASCII转义，真实中文路由/越界测试通过。

## Working State

持久权威：Task ubuntu-runs-cloud-retirement-60g-20261019、ADR050与每run闭合manifest/proof/retirement receipt。云根B:/tts-exp/ubuntu-runs-archive/20261019/batch02/<run>/attempt01；根README与批次catalog为可刷新导航。仅lrs3_phone_rules_full_20260921_proxy已完成，1446成员、4.383754730224609GiB真实释放；第二根phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3完整归档36543219212字节/545片、源after通过，但335781成员全恢复尚未闭合，原件保留。

当前机器局部路由BASE/active_route.json，BASE=Z:/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_controls_20261019（Z为UbuntuSMB）。主coordinator_v15.py PID274680，client_attempt15，仅验证前二；coordinator_future_v16.py PID263804，client_attempt16，仅原索引02至06且有RAM门。子目录用00至06数字。两者均使用verify_live_pipe_v2.py SHA41d2bb51a2903e48cd007848a94348d498fc9fdc93bf5eb51dcd31c85378d360，helper/server b50e4ee4b033c0352d5591a4764ea552fbaf5a8761d84934a28e718c58a03a20、client 8df67190e6f852adb8db194ebc16e81c2101adb4b918d953798a40e141066f00。19:24:56Z两者存活无失败，第二根恢复560754467字节/2093成员，第三根358966013字节/7718成员；都是新完整重试而非拼接旧partial。

finalize_retirement_v3.py PID265596，BASE/finalizer_attempt03，SHA1ef87069ce04c9085e2839708fe660f6c6736dc03d9dfa958f99e45fae1ed969，只接当前15/16。第二根绑定闭合proof且退休后，它按严格PID/FILETIME终止旧暂停reader；全7闭合且净df>=60GiB后，发布SHA绑定闭合journals、停止本任务Linuxcacheguard、更新目录导航，再从WindowsB回读全部source/transport/proof/retirement/journal SHA。completion_receipt.json的VERIFIED_DELIVERY_COMPLETE才是交付闭合证据；任何失败写failure.json并保留原件，目标不足亦停止待处理。最终还需fresh df、核本任务guard/queue退出、Task done/ADR结果。

旧coordinator_v4.py PID265156仍暂停，FILETIME134368934664940606，恰好1次本任务manual NtSuspend；BASE/preserved_v4_attempt01/state.json为权威。它约25.25GiB/6975成员的partial解压状态不能作删除准入。若新主先完成第二根，finalizer按新闭合proof终止它并标记TERMINATED_REPLACED；若用户腾F约1GiB希望恢复旧进度，先停止新主第二根owner、保持future16，再仅NtResume一次，旧第三根PARTITION_STOP边界有效，之后须纯proof聚合器并换finalizer路由。用户尚未回复可选F空间问题，不重复请求权限。

Ubuntu CONTROL=/home/wjj/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_20261019_attempt02，冻结archive_remote_v3.py SHA d6e905893ca41bbd9096ecfed474e02d72657d2332aaaebfe07037fdbf6351cc。prefetch_future_archives_v1.py PID280863只串行归档后五，Linux upload_cache_guard.py PID161076只管理严格匹配的本次生产者，小于3GiB暂停、大于4GiB恢复；完成后cache_guard.stop使finally恢复它所暂停的生产者。19:06根可用3550142464字节，旧根未退休与上传缓存使净增不足目标。不得改/停CloudDrive配置/DB/缓存/服务。

## Changed Files

任务自有工具均在.codex/scratch/ubuntu_runs_archive_20261019（机器局部/未提交），包括当前coordinators、pipe reader/helper、bounded retry、最终journal publisher/catalog/finalizer与测试。冻结当前12份源码已发布CONTROL及cloud/_verification_sources/bounded_cloud_retry_pipe_v2/sources_manifest.json，SHA9476583561d38da5aa1d1c0118246fceb8060c7f9381bfd1aacacde578397f75。AGENTS.md云根说明与Basic Memory ADR049/050/Task为本任务文档。仓库还有大量其他会话dirty/untracked研究代码；保持，不提交、不删除。无本次commit/PR。

## Verification

既有归档/边界/拒绝退休、UTF8路径、guard scope、live分片与真实恢复测试及首根4.38GiB实际闭合保持有效。新增真实B输入1024次物理写入/fsync/重开SHA/unlink含空文件通过3.52s（SMB23.44s）；新云handle重开三项故障注入通过；新reader真实物理恢复/坏body/坏whole拒绝套件4项通过；15/16不相交全7聚合2项与真实中文route/越界拒绝通过；旧reader终止需有效绑定退休proof、已替代幂等2项通过。19:24两只15/16与finalizer03无failure，但大批全恢复尚未通过，不推断成功。保留11/12读错误、13/14编码拒绝、旧SMB同名错误及所有partial，无科学训练/推理/整仓测试。

GitHub只读确认仓库与本分支无openPR；当前SHA在GitHub API422不可解析，故仅标本机SHA。当前git tracked dirty包括AGENTS/CONTEXT、audio_enhancement、portable_source_axis与对应测试，其他会话修改不归本次。两个前一checkpoint同chat搜索去重后以18:47:36Z为最新，直接读确认agent/session精确匹配。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local-only SHA: 2b71639c3b555293375d248350999377168d515d
- Current cursor: BASE/active_route.json（上述BASE）
- Navigation: B:/tts-exp/ubuntu-runs-archive/README.md
- Final delivery authority: BASE/finalizer_attempt03/completion_receipt.json

## Observations

- [result] 已统一云根；当前只有首根4.383754730224609GiB退休，目标仍未完成。
- [decision] 独立WindowsB全内容验证不变，Ubuntu仅作有限物理scratch；分片重连不重置或跳过hash，失败按已返回count重开并保持最终全流绑定。
- [decision] 不相交的前二/后五分区与finalizer只有全7闭合、实际净df>=60GiB和云端闭合metadata回读后才可完成；raw并行free差值不能求和当实际回收。
- [blocker] Windows近满盘、网盘读回缓慢；需两台机器在线完成长传输。后台在继续执行，不把该等待包装为已达到目标。
- [next_step] 接着检查15/16和finalizer03实际进展/故障，直到最终闭合交付与fresh净df>=60GiB，发布最终证据、清理本任务旧进程并更新Task/ADR；不足60须fresh资格额外旧根。

## Relations

- continues [[Codex checkpoint - 2026-10-19T18-47-36Z - Ubuntu cloud retirement with bounded SMB restoration]]
- relates_to [[ubuntu-runs-cloud-retirement-60g-20261019]]
- relates_to [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]