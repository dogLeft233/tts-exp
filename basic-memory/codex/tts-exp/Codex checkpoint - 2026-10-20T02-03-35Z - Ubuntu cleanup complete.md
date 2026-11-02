---
title: Codex checkpoint - 2026-10-20T02-03-35Z - Ubuntu cleanup complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t02-03-35-z-ubuntu-cleanup-complete
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
username: LAPTOP-E133GSLE/dogle
hostname: LAPTOP-E133GSLE
started: 2026-10-20 02:03:35+00:00
capture: deliberate
agent: codex
session_id: 01a1543e-4ce2-7401-bd94-9dfddaf8204c
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-20T02-03-35Z - Ubuntu cleanup complete

## Summary

Ubuntu七个已上传历史run已清理完成，实际净腾出61.107574GiB；用户为节约流量明确豁免剩余五根的完整下载核验，所有相关worker均已退出。

## Story

用户希望找清历史实验数据并归拢百度网盘，随后要求把Ubuntu暂时不用的旧runs转到网盘，尽量腾出60G以上。此前已将12个散落的云项目目录整合到B:/tts-exp，内部结构保留，路径映射由ADR049与云根_index/path_mapping.json管理。旧runs采用64MiB gzip/PAX流分片上传，逐文件源manifest保留恢复路径和POSIX元数据。

原方案独立从Windows网盘挂载下载并逐文件物理恢复，两根已验证后清理；其余下载恢复耗时并消耗流量。用户先说暂停核验，再明确“直接清理ubuntu空间吧，我网络的流量不是很足了”。这条最新授权覆盖先前完整回读准入与挂起恢复计划。四只旧Windows管理/校验worker均严格核对PID和FILETIME后终止，暂停计数随退出失效，绝不可再恢复。

新的retire_uploaded_only_v1.py没有伪造旧的完整恢复proof。它只查询云分片名称和大小，在Ubuntu本地核对已完成producer记录、源after、fresh全文件SHA/身份、保留树链接及候选活动引用，再按精确manifest unlink与空目录rmdir。剩余五根全部完成，分别留下retirement_without_download_receipt.json，明确independent_restore_verified=false与verification_waived_by_user=true。此次最终清理下载归档payload为0；仅发布几KB源码、凭据与目录导航。原位置保留报告和CLOUD_ARCHIVE.json，恢复方式仍是按transport顺序拼接后解到新空目录。

最终本地审计2026-10-20T02:01:11Z：起点空闲5,570,371,584字节，最终71,184,130,048字节，实际净增65,613,758,464字节（61.107574GiB），当前可用66.295387GiB。完成7/7；2根全内容恢复核验通过，5根用户豁免。这个净df结果而非原占用或并行free差值之和满足目标。Task已done，ADR050已accepted并记载用户修改；云README/catalog与项目网盘入口已更新。

## Working State

持久入口是Task ubuntu-runs-cloud-retirement-60g-20261019、ADR049/050及云批次 `B:/tts-exp/ubuntu-runs-archive/20261019/batch02/`。Linux对应 `/home/wjj/百度网盘/tts-exp/ubuntu-runs-archive/20261019/batch02/`。每run原source/transport manifest、producer source_after与不同验证模式的退休回执保持，不改冻结历史凭据。云 `_retirement_sources/user_requested_no_download_v1/` 保存本次小型源码、5项测试结果、授权、completion及final_local_audit；这些本地审计不能替代五根尚未执行的独立云内容验证。

机器局部权威控制根为 `/home/wjj/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_20261019_attempt02/`，新子目录 `retirement_without_download_20261020_attempt01/` 中的completion_receipt.json、published_catalog.json、final_local_audit.json和每run删除journal可追溯。Windows Z映射同一Ubuntu磁盘，不是独立备份。控制根 ubuntu_runs_archive_controls_20261019/active_route.json 为COMPLETE、无active_worker、禁止重启旧下载；verification_terminated_by_user_v1.json记录4只终止身份。新Ubuntu PID338966也已退出；最后审计没有restore helper、producer prefetch或upload cache guard，Windows相关PID亦不存在。旧partial日志保留但不能当完整proof。

选中七根：lrs3_phone_rules_full_20260921_proxy；phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3；mfa_linear_intraphone_time_20260925；aishell1_qwen_mfa_linear_n100_20260820；mfa_linear_video_retiming_v1；local_tts_spectral_repair_20260925；lrs3_mfa_scorer_effect_20260923。带1520外链的旧LRS3队列与912 incoming引用的external audit仍排除。原始数据集、模型环境、10月与May当前资产保留；CloudDrive服务/配置/数据库/缓存未干预；没有重跑科学实验。

## Changed Files

本任务未提交的机器局部工具在 `.codex/scratch/ubuntu_runs_archive_20261019/`：retire_uploaded_only_v1.py、test_retire_uploaded_only_v1.py、publish_catalog_v3.py、audit_retirement_without_download_v1.py；Linux部署副本及真实执行凭据在上述控制根。项目记录为AGENTS.md的统一云根说明、Basic Memory Task和ADR049/050。本轮收尾更新Task/ADR050和云导航。工作区另有10个tracked修改与1058个untracked入口（01:56的git status），大部分来自其他研究会话，全部保留；无本任务commit或PR。

## Verification

5项真实Linux隔离测试通过：禁止打开cloud archive body的正常清理，源变化拒绝、活动引用拒绝、分片缺失拒绝、名单外根拒绝。生产前fresh扫描7972个保留树symlink，0命中、0错误；五根fresh本地SHA/身份逐项等于上传清单。最终7个stub的文件集合、全部71个保留报告SHA、locator路由及两种回执状态检查通过；stub总逻辑内容3,782,864字节。最后实际净df>=60GiB通过。publisher成功，Windows已读取新批次README的小型文字导航。完整云payload核验对五根明确跳过；未宣称云副本全部独立恢复验证通过。

## References

[项目仓库](https://github.com/dogLeft233/tts-exp)，由GitHub只读get_repo确认；当前分支的all-state PR检索无结果。当前SHA `2b71639c3b555293375d248350999377168d515d` 为local-only，先前GitHub解析未找到，不构造不可用commit链接。

## Observations

- [result] 七根全部归档清理，净空闲增加61.107574GiB，达到用户60GiB以上目标；当前可用66.295387GiB。
- [decision] 用户节约网络流量的明确要求覆盖原完整回读要求；两根完整验证与五根用户豁免必须分开记录，不能伪造proof。
- [decision] 旧worker全部退出，今后不得按历史暂停笔记恢复下载；取用归档由新需求触发。
- [next_step] 将来需要旧run时沿CLOUD_ARCHIVE.json，在空目录拼接并恢复相应分片；本任务已完成，不自动继续下载。

## Relations

- continues [[Codex checkpoint - 2026-10-19T19-26-05Z - Verified cloud retirement continues with resilient readers]]
- closes [[ubuntu-runs-cloud-retirement-60g-20261019]]
- records [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]
- follows [[049-unified-baidu-project-root]]