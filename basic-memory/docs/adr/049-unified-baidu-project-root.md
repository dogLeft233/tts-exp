---
title: 049-unified-baidu-project-root
type: note
permalink: tts-exp/docs/adr/049-unified-baidu-project-root
status: accepted
decided: '2026-10-19'
project: tts-exp
decision: 百度网盘项目资产统一存入 B:/tts-exp，保持原目录内部相对路径。
rationale: 用户要求把散落在百度网盘根目录的历史实验数据归到一个文件夹，降低查找与恢复成本。
alternative:
- 保留分散目录并只新增索引；不能满足用户希望根目录收拢的要求。
- 同时重排每批归档内部层级；会扩大来源路径和恢复清单变更。
consequence:
- 所有新增项目云端输出必须位于统一根目录。
- 历史绝对路径通过独立映射转换，冻结凭据保留原执行位置。
- 未建立旧位置兼容链接；旧脚本重新执行前必须转换云端路径。
---

# 百度网盘项目资产统一根目录

用户发现网盘数据分散，要求集中保存。本次将 12 个项目目录整体移入 `B:/tts-exp/`，Ubuntu 对应 `/home/wjj/百度网盘/tts-exp/`。内部相对路径和已有归档内容保持不变，不合并同名批次，也不删除失败尝试。

完整入口为 `B:/tts-exp/README.md`，路径对照为 `B:/tts-exp/_index/path_mapping.json`。目录包括 tts-exp-project、tts-exp-archives、tts-exp-research-models、experiment-archive、tts-research-archives、tts-exp-archive-transfer、tts-exp-audio-head-20261003、tts-exp-local-w2l-syncnet-attempt01，以及 fit40_session09_poweroff_archive_attempt11/12/16/17 的四个完整原名目录。

## 来源与验证

本地操作凭据在 `.codex/scratch/cloud_consolidation_20261019/`。移动前保存每目录的直接子项名称、类型、文件字节数和一个小文件 SHA；移动后在 Windows 核对，并从 Ubuntu 独立 CloudDrive 挂载复核新位置。大归档未全量重哈希，本次核验不替代原来的完整归档验证，也不改变原包成功或失败的状态。

WORLD 目录移动时挂载返回“不支持该请求”，初次新路径检查尚不可见；随后两端看到新目录，旧位置消失。未重复执行移动，独立保存 reconciliation receipt，原错误 receipt 保留。

## 恢复边界

历史脚本、包内 manifest 和 cloud_restore_locator 可能仍包含旧云端绝对路径。按 `_index/path_mapping.json` 将原根级目录映射到统一根目录后恢复，不能因旧路径不存在推断数据丢失；冻结历史凭据不回写。AGENTS.md 已记录新入口，后续脚本执行前必须转换旧前缀，不再创建旧根级项目目录。

这次仅归拢已有云端目录，没有把 E/F/C/Ubuntu 所有实验数据上传。编码器 survey 与 normalization ablation 的本地原件仍按各自存储记录确认，不能推断已云同步。

## Observations

- [decision] 百度网盘项目资产统一根为 B:/tts-exp/；Ubuntu 同目录为 /home/wjj/百度网盘/tts-exp/。
- [constraint] 历史数据内部结构与冻结凭据保持原样，旧路径通过独立映射转换。
- [verification] 目录命名空间和小文件读取验证与完整归档 SHA 验证分开记录。

## Relations

- affects [[编码器扩展的执行环境与资产存储]]
- affects [[server-access]]
- relates_to [[研究交接总览]]