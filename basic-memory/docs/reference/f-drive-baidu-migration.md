---
title: F盘资料迁移到百度网盘的路径与恢复方式
type: note
status: active
permalink: tts-exp/docs/reference/f-drive-baidu-migration
---

# F盘资料迁移到百度网盘的路径与恢复方式

## F盘资料迁移（2026-10-23）

🔄 上传进行中；F盘原件保留。状态更新时间：2026-10-25T18:56:06.058067+08:00。

| 原位置 | 网盘位置 | 存放方式 |
|---|---|---|
| `F:/tts-research-archives/` | `B:/tts-exp/tts-research-archives/` | WORLD v6原TAR补入原批次目录 |
| `F:/tts-exp-archives/` | `B:/tts-exp/tts-exp-archives/` | FIT40原TAR直接存放；三个restore_attempt01恢复树为下列不压缩TAR |
| `F:/codex_scratch/` | `B:/tts-exp/codex_scratch/` | 目录与文件原样保存，含WORLD v5原TAR、Ditto读回文件 |
| `F:/tts-exp-p1-gain-dose-response-20261016-source-attempt02/` | `B:/tts-exp/tts-exp-p1-gain-dose-response-20261016-source-attempt02/` | 30个已有tar.gz、30个WAV和3份JSON原样保存 |

原始范围：118799个普通文件、25139402813字节（约25.1 GB）。三个恢复树共117283个文件，为避免十多万个深层小文件逐个上传，分别封装成无压缩TAR；包内从`restore_attempt01/`开始，保留相对路径。已有TAR/TAR.GZ未重新压缩。

三个恢复包均位于原对应的`recoverable-output-archive-20261004-attempt02-muse`、`attempt04-failedfit-parallel4`、`attempt06-ditto-parallel4`批次目录，文件名统一为`restore_attempt01_from_F_20261023.uncompressed.tar`。旧网盘`restore_attempt01/`目录可能不完整，恢复时优先使用本批新TAR并解包到新建空目录。

按用户要求未下载网盘payload逐文件计算哈希。核对范围为来源清单、普通文件数/相对路径/字节数、无压缩包成员元数据，以及CloudDrive上传队列。上传队列未完成前，挂载文件存在不等于云端持久化成功。

原始历史receipt/locator不回写。F盘旧路径先查`B:/tts-exp/_index/f_drive_path_mapping_20261023.json`；restore路径按archive_overrides解包，不能直接把旧文件路径拼到网盘目录中使用。Ubuntu网盘根为`/home/wjj/百度网盘/tts-exp/`。

F:/Datasets/tts-exp约80.7 GB未纳入本次迁移，42/44视频评估脚本的数据默认入口仍在那里。F盘其他目录、空的p1 attempt01和仅含JSON的attempt03也未处理。

本地任务与清单：`./.codex/scratch/f-to-baidu-20261023/`；可恢复工作状态见`task_state.json`，云端结果见`B:/tts-exp/_index/f_drive_transfer_20261023/receipt.json`。临时上传缓存从F盘切到C盘，队列完成后恢复原设置。清理状态：已完成三组约9.43 GB具备清理条件，但自动审批拒绝F原件和C临时TAR删除；本次释放0字节。

### 已完成的三组与可清理范围（2026-10-25）

`F:/tts-exp-archives/`、`F:/codex_scratch/`、`F:/tts-exp-p1-gain-dose-response-20261016-source-attempt02/`已确认无待上传任务，1518个目标文件大小一致；三个无压缩TAR的117283个成员路径和大小与冻结来源清单一致。本地118798个原件的路径、大小和修改时间也一致，合计9430208573字节（约9.43 GB）。未下载网盘payload重验哈希。

上述三处可由用户手动清理；当前agent的递归删除操作被自动审批拒绝，原件仍在。C盘三个本批生成的无压缩临时TAR约4.06 GB也具备清理条件，但单文件删除同样被自动审批拒绝，当前仍保留。本次没有实际释放空间。

WORLD v6的`F:/tts-research-archives/`不在可清理范围：最新检查仍有15.7 GB包处于上传队列。不得把挂载文件大小完整视为上传成功。三组核对凭据见`_index/f_drive_transfer_20261023/completed_groups_cleanup_gate.json`，删除拦截记录见同目录`cleanup_attempt_20261025.json`。
