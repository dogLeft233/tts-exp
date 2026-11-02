---
title: CloudDrive 实验归档目录
type: tool
permalink: tts-exp/tool-notes/cloud-drive-实验归档目录
status: active
tool: CloudDrive2
date: '2026-09-27'
---

# CloudDrive 实验归档目录

## Observations

- [tool] CloudDrive2现有用户服务clouddrive.service已active；既有CloudFS fuse rw挂载/home/wjj/百度网盘，无需重挂/重启。
- [authorization] 用户明确授权空间不足时使用本地clouddrive；本次只复用现有认证与挂载、小探针，不搬旧实验、不改已封存路径。
- [archive] 可用项目独立目录：/home/wjj/百度网盘/tts-exp-experiment-archive。
- [probe] 2026-09-27仅4096B自建独占探针，write/fsync→读回SHA→单调用rename/source消失→目录fsync→等待15s后重开hash全PASS；只删自己探针。挂载表面语义通过，不等于断网持久性/跨客户端云端原子事务保证，读回可能来自cache。
- [capacity] statvfs报告云端可用1288536883200B，约1.172TiB；不当作本地空闲。探针后transfer_tasks只读计数0，无读取其他任务内容。
- [cache] 非敏感配置file_buffer_disk_cache_max_bytes=536870912（512MiB）、Lru，upload_delay10s。服务资料与已有约136MiB目录缓存位于/home/wjj/Waytech/CloudDrive2，与项目根盘同设备；未清理。payload缓存精确路径尚未从设置/小探针确证；TMPDIR/TMP/TEMP未设，/tmp也在根盘，不能假定独立缓存盘。
- [budget] 本次实测仅4KiB；本地高于5GiB底线约58MiB，root最新决定主实验维持本地40MiB方案。CloudDrive只作可选归档，单次≤8MiB串行并每批复查free/任务/读回hash、5GiB硬停，缓存累积即停；这是保守运行预算，不是已验证GB级吞吐或无条件持久保证。缓存上限512MiB不得直接当安全可填满额度。
- [retention] 新大产物可另选云端目录，但本地保留manifest/hashes/来源和上传读回完成凭证；不修改旧sealed路径，不将云端未独立验证的数据作为唯一副本。无需为已有认证再次确认；缺失凭据时具体报缺口，不猜测。
- [safety] 未重启/卸载/改配置、未动用户文件、未大上传、无GPU；只删除自建探针；凭据/token未输出或入笔记，新增≤1MiB。
- [report] runs/clouddrive_storage_audit_20260927/report.md；probe_receipt.json、storage_summary.json保留非敏感测量。目录可用性与空间预算须使用前实时复核。

## Relations

- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 复用既有CloudDrive挂载，创建项目归档目录并4KiB读写/hash/rename探针通过；缓存与本地预算限制留证 | September 27, 2026 | user explicit CloudDrive authorization / parent storage audit |

| root最终约束：主实验本地40MiB，云端仅≤8MiB串行可选归档+watchdog，缓存累积即停；无需大probe或服务改动 | September 27, 2026 | parent final storage decision |
