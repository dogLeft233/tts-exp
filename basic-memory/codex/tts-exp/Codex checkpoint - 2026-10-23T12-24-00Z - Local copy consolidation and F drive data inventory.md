---
title: Codex checkpoint - 2026-10-23T12-24-00Z - Local copy consolidation and F drive
  data inventory
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t12-24-00-z-local-copy-consolidation-and-f-drive-data-inventory
status: open
project: tts-exp
cwd: E:\Documents\tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T12:24:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1687e-853c-7641-9916-7d13b80177db
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T12-24-00Z - Local copy consolidation and F drive data inventory

## Summary
已将历史本地副本的独有视频评估代码和55.2 GB缺失资产整合进当前项目，并确认F盘另有约105.9 GB项目相关文件；旧副本物理清理仍受阻。

## Story
用户起初要求整理百度网盘挂载的项目目录，让agent理解各目录用途，随后发现E:/Documents/tts-audio/tts-exp是独立旧Git副本，要求先合并独有视频评估代码、修复指向副本的路径、验证独有数据备份，再清理副本。先比较代码，保留当前更新版本；将独有41–45评估链与测试合并，并用统一路径解析器兼容旧Windows/WSL路径。先完整备份并逐成员及整包校验，再以同卷rename迁移当前缺失的数据，不覆盖当前同路径不同内容。最新问题是F盘是否仍分布实验数据；只读目录统计确认仍有大量独立资产，不能把本次E副本恢复包视为覆盖F盘。

## Working State
当前唯一工作入口为E:/Documents/tts-exp。Git分支codex/cross-tfg-audio-head-20261003，当前本地HEAD为2b71639c3b555293375d248350999377168d515d。Git status --short已读，存在大量原有修改和未跟踪研究脚本；本轮没有提交，不能把全部dirty内容归入本次整合。GitHub只读查询确认dogLeft233/tts-exp，当前分支搜索未返回PR。gh CLI不可用。

权威说明为basic-memory/docs/reference/local-copy-and-video-metrics.md；机器本地证据在.codex/scratch/local-copy-consolidation-20261023/，包括retirement_result.json、code_validation.json、migration_actions.jsonl、backup_verification.json与f_drive_inventory.json。这些scratch证据、迁入数据、尚未提交代码和C盘恢复包均为机器本地状态。

完整恢复包C:/Users/dogle/tts-exp-archives/local-copy-retirement-20261023-attempt03/old-checkout.tar为56298711040字节，SHA256 6c2c3d142d0c4da8cfcdbb2a898618a84582340d574b0a3ecfeee6e25a0ad123。原96925普通文件、507个WSL LX reparse链接均逐成员校验，整包读回PASS。恢复工具及清单随包保存。此为C盘本地备份，不是云备份，C/E物理设备独立性未核验；外部链接目标内容未包含。attempt01/02是不完整失败中间TAR，共约32.2 GB，仍留在C盘，不能用于恢复。

91435条目迁入当前，普通数据55220635384字节，506内部WSL链接重定向当前入口；146条已相同，568同路径不同内容未覆盖，5283其它条目仅在完整恢复包保留。旧目录E:/Documents/tts-audio/tts-exp仍存在，未改名、未删除，已用RETIRED.md和退役AGENTS.md引导当前入口。原AGENTS保存在恢复包。

最新F盘只读统计（逻辑文件字节，非物理占用；仅扫描根目录命名tts-*、Datasets、codex_scratch、BaiduNetdiskDownload，非全盘任意目录内容搜索）：
- F:/Datasets/tts-exp：80740800605字节，354033文件。主要synctalk_multispeaker_full_20261017 23.793 GB、synctalk_gain_training_20261017 11.020 GB，另有May噪声/归一化/训练、编码器、metrics_20261021等。包含媒体、特征、检查点/交付资料及运行环境，不能全称纯实验媒体。
- F:/tts-research-archives：15709194240字节，为20261001/world_v6_attempt07_v6_attempt01/world_v6_attempt07_v6.tar。
- F:/tts-exp-archives：4177008709字节、117284文件，包含Ditto/Muse/failedfit恢复目录与FIT40归档。
- F:/codex_scratch：4086339911字节、1451文件，主要world_v5_attempt06.tar 4043694080字节及Ditto读回产物。
- F:/tts-exp-p1-gain-dose-response-20261016-source-attempt02：1166859953字节，63文件，含30归档与30音频。
- archive-transfer约1.1 MB；p1 attempt03仅18590字节JSON；p1 attempt01和BaiduNetdiskDownload本次无文件。
合计105881360213字节、472974文件，扫描无报错。F盘未迁移/删除，未进行逐文件备份覆盖或SHA复核。新42/44脚本仍默认F:/Datasets/tts-exp，可用--data-root覆盖。

## Changed Files
本次合并新增scripts/41_video_quality_metrics.py、42_build_video_metric_manifests.py、43_aggregate_video_metrics.py、44_extract_sync_scores.py、45_render_video_metric_report.py、project_paths.py与requirements-video-quality.txt，及5个测试文件。
兼容解析器修复8个rhythm/pause/cohort/knn/diagnostics/vsr路径用户，3个VFHQ下载脚本改为源码定位repo；保留无关远端/home/wjj/tts-audio/tts-exp路径。41支持--repo-root，42/44支持--repo-root和--data-root。
当前AGENTS.md及reference说明已更新。百度挂载B:/tts-exp已创建AGENTS、_index/navigation/CATALOG.md、14个目录指南及1008条浅层inventory和receipt；原归档未移动或删除。历史云端路径须经_index/path_mapping.json转换。

## Verification
31个新合并/路径/真实MJPG解码相关测试通过；迁移后cohort集成4通过1跳过（原生成cohort不存在）。17源文件语法检查通过，按CRLF设置的git diff --check通过。GPU IQA/LPIPS/FID/FVD未完整重跑；依赖仅安装在Temp测试目录。已有FID/FVD代理指标限制保留。
旧副本完整备份逐成员SHA与整包SHA通过；剩余5997条在删除尝试前核验。历史云端源manifest匹配1460文件366710380字节，但未读取云payload，不能替代完整备份验证。迁移后506链接扫描无旧目标、无外部目标、无报错。
F盘大小来自元数据遍历，未hash，也未认定所有归档均已云备份。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- basic-memory/docs/reference/local-copy-and-video-metrics.md
- .codex/scratch/local-copy-consolidation-20261023/retirement_result.json
- .codex/scratch/local-copy-consolidation-20261023/f_drive_inventory.json
- C:/Users/dogle/tts-exp-archives/local-copy-retirement-20261023-attempt03/README.md

## Observations
- [result] 当前项目已接收独有视频评估链和约55.2 GB缺失数据，旧副本完整恢复包通过SHA读回。
- [decision] 不覆盖当前同路径不同内容，不改写冻结历史凭据；旧路径通过解析器或存储映射处理。
- [blocker] 自动审批拒绝旧目录及失败中间包删除，仅返回blocked by policy；不得换API绕过。可撤销目录改名又被Windows文件占用阻止，.git改名access denied，未杀进程。
- [result] F盘识别目录仍有约105.9 GB项目相关文件，并非本次E旧副本备份覆盖范围。
- [next_step] 若继续整理F盘，先按f_drive_inventory.json逐目录核对当前代码依赖与B:/tts-exp归档覆盖，再提出可迁移/可清理清单；不要直接删除。
