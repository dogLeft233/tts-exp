---
title: ubuntu-runs-cloud-retirement-60g-20261019
type: task
permalink: tts-exp/codex/tts-exp/ubuntu-runs-cloud-retirement-60g-20261019
status: done
project: tts-exp
repository: dogLeft233/tts-exp
started: '2026-10-19'
completed: '2026-10-20T02:01:11Z'
description: Ubuntu 七个已归档旧 runs 已释放；实际净腾出61.1076 GiB，用户要求跳过剩余五根的完整下载核验。
current_step: 7
---

# Ubuntu 旧 runs 归档清理完成

用户最初要求归拢历史实验数据和百度网盘，然后要求把 Ubuntu 暂时不用的旧 runs 转到网盘，尽量腾出60G以上。最终已清理7/7个选定目录。2026-10-20T02:01:11Z本地验收：起点空闲5,570,371,584字节，最终71,184,130,048字节，实际净增65,613,758,464字节（61.107574 GiB）；当前可用66.295GiB。各批空闲差可能重叠，不叠加作为验收。

## 用户修改与证据边界

用户最新明确“直接清理ubuntu空间吧，我网络的流量不是很足了”。此前两根已经独立下载、逐文件物理恢复验证后清理；剩余五根全部上传完成，按用户授权跳过独立全内容恢复核验。新流程只读云分片元数据，在Ubuntu本地复核原producer清单、源after、fresh逐文件SHA和身份、保留树链接与精确活动引用，再按manifest unlink、rmdir空目录。此次最终清理下载归档payload为0字节。五根单独的 retirement_without_download_receipt.json 明确 independent_restore_verified=false、verification_waived_by_user=true；没有伪造完整恢复凭据。

## 归档与清理范围

统一云批次为 `B:/tts-exp/ubuntu-runs-archive/20261019/batch02/`，Ubuntu对应 `/home/wjj/百度网盘/tts-exp/ubuntu-runs-archive/20261019/batch02/`。已上传压缩内容合计56.373428 GiB。云端README/catalog列明每根状态；项目网盘根README已有批次入口。每根源位置保留 CLOUD_ARCHIVE.json 和小报告，7根残留报告/指针合计约3.8MB，文件SHA最终本地复核通过。

| 旧 runs 根 | 最终核验状态 |
|---|---|
| lrs3_phone_rules_full_20260921_proxy | 完整独立恢复通过后清理 |
| mfa_linear_intraphone_time_20260925 | 完整独立恢复通过后清理 |
| phone_gain_static_tfg_mfa_repair_20260923_sync_c_v3 | 已上传，依用户要求跳过完整下载核验后清理 |
| aishell1_qwen_mfa_linear_n100_20260820 | 同上 |
| mfa_linear_video_retiming_v1 | 同上 |
| local_tts_spectral_repair_20260925 | 同上 |
| lrs3_mfa_scorer_effect_20260923 | 同上 |

带1520外链的 lrs3_qwen_cloud_n500_20260818 和被912个链接引用的 phoneme_tfg_association_external_visual_cpu_audit_v1_20260922 仍排除。原始数据集、模型/环境、10月与May当前资产保留，没有重新运行科学实验或干预其他会话训练。

## 权威凭据与恢复

Ubuntu控制根：`/home/wjj/tts-audio/tts-exp/.codex/scratch/ubuntu_runs_archive_20261019_attempt02/`。每run的 source_manifest.jsonl.gz、transport_manifest.json、producer source_after 和不同模式的退休回执是权威证据；原凭据不改。新控制子目录 `retirement_without_download_20261020_attempt01/` 包含授权、fresh源清单、逐成员删除日志、completion_receipt.json、published_catalog.json、final_local_audit.json。云端 `_retirement_sources/user_requested_no_download_v1/` 保存小型源码、测试结果、授权、完成凭据及最终本地审计；没有为此重新上传或下载大归档。

恢复时跟随源locator，按transport清单顺序拼接 archive.partNNNNN，以gzip/PAX tar解到新建空目录。分片不是独立压缩包，不能直接逐片在线解压。若需要独立确认五根的云副本内容，日后有流量时可另外校验；本任务不自动启动下载。

## 验收与终局进程

5项Linux隔离测试通过：不打开云payload的正常清理，源变化、活动引用、缺分片及名单外根均拒绝。最新保留树扫描7972个符号链接，0命中、0错误。生产五根fresh源全部与上传清单一致；最终7个源stub集合、保留报告SHA、locator和两种凭据状态检查通过；净空闲>=60GiB。

旧Windows fast manager、v4、v15、future16已精确身份终止，记录在 `ubuntu_runs_archive_controls_20261019/verification_terminated_by_user_v1.json`；旧暂停计数因进程退出无效，不得恢复。新Ubuntu清理PID338966已退出。active_route.json记录终局；旧partial恢复日志保留但不作为完整通过。CloudDrive服务/配置/数据库/缓存未干预。

## Observations

- [result] 7/7旧目录清理完成，实际净增61.107574GiB，达到60GiB目标。
- [decision] 用户明确节约网络流量，剩余五根跳过独立下载核验；与已全内容验证的两根分开记录。
- [next_step] 将来确需旧run时按CLOUD_ARCHIVE.json恢复到空目录，不自动重启旧下载或清理worker。

## Relations

- implements [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]
- follows [[049-unified-baidu-project-root]]