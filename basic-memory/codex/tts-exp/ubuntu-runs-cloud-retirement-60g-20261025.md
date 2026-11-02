---
title: ubuntu-runs-cloud-retirement-60g-20261025
type: note
permalink: tts-exp/codex/tts-exp/ubuntu-runs-cloud-retirement-60g-20261025
status: in_progress
project: tts-exp
repository: dogLeft233/tts-exp
started: '2026-10-25'
completed: '2026-10-25T15:25:00+08:00'
description: 把 Ubuntu 研究机 runs/ 下 2026-09-09 之前的历史 run 以未压缩 TAR 搬到网盘并删除原目录，实际移出 63.37
  GiB、服务器可用空间 15 G→84 G；上传从本机缓存继续。按用户要求不压缩、不做回读复核。
current_step: 4
---

# Ubuntu 旧 runs 归档清理（20261025）

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 用户要求

- 清理 `wjj@192.168.50.2`（Ubuntu/V100 研究机）上的 tts-exp 项目空间，约 60 GB。
- 先对照服务器历史数据与本地重复 / 暂时用不到的目录，再用百度网盘挂载上传后清理。
- **不压缩上传**，**不需要复核**；同时维护网盘侧文档与文件结构。

## 环境事实（2026-10-25）

- 服务器：`/home/wjj/tts-audio/tts-exp/`，根分区 468 G，已用 429 G，可用约 15–16 G。
- 其中 `runs/` 约 178 GiB（558 个 run），`data/` 约 16 G，`models/` 6.6 G，`.venv` 6.4 G。
- 网盘挂载：`B:` = CloudDrive 挂载的百度网盘（`/百度网盘`），项目根 `B:/tts-exp/`。
- 上传实测：CloudDrive 走本机 Clash 的系统代理，但国内规则为 DIRECT；**聚合上传约 1 MB/s**，并发 4–6 路可到 1–2.7 MB/s，60 GB 量级需要十几小时。
- CloudDrive 会话缓存目录 `%LOCALAPPDATA%\CloudDrive.WinUI\session-upload-cache-tts-20261023`，写 B: 先落本地缓存再异步上传；另有 2026-10-23 遗留的一个 15.7 GB 未完成 `world_v6_attempt07_v6.tar` 任务（块已传完但 finalize 失败），未处理。

## 选取范围

- 规则：`runs/` 下最新文件时间**早于 2026-09-09** 的 run（2026-07–09-08，LRS3 masked-TTS、MFA-linear、Wav2Lip replacement / timing-transfer 等已结论阶段），共约 63.3 GiB。
- 保留：`data/` 原始数据集、`models/`、conda/venv 环境、9 月 9 日之后的 run、以及被大量外部链接引用的目录（本次仅被 link 的仍按 symlink 归档）。
- 本地重复（未删除，仅核对）：`data/hdtf_audio_en`、`multilingual_tts`、`data/data`、`mdc_tts`、`gsv-tts` 与本地 `./data` 同文件同名（约 1.06 GB），本批未动，可作后续快速清理项。

## 执行方式

- 每个 run 一个 `tar -cf -` 流式传输（不压缩、不落服务器临时盘），直接写入 `B:/tts-exp/ubuntu-runs-archive/20261025/batch01/<run>.tar`。
- 用户中途确认目标：先把数据搬到本机挂机上传，服务器空间立即释放。据此改成 **缓存优先删除**：TAR 完整写入本机 CloudDrive 上传缓存并核对字节数后立即 `rm -rf` 服务器原目录，上传从缓存继续；缓存文件在传完前是唯一副本，不能删。
- 控制脚本：`${USER_HOME}\tts-exp-cleanup-20261025\transfer_manager.py`（可恢复，状态 `state.json`，日志 `transfer.log`）；收尾/发布脚本 `finalize_watch.py`；状态对齐脚本 `reconcile_state.py`、`promote_uploading.py`。
- 本机中转容量受 `C:` 剩余空间限制（设为 20 G 安全线、约 36 GiB 在途上限）；2026-10-23 遗留的 15.7 GB `world_v6_attempt07_v6.tar` 未完成上传任务仍占着缓存，未处理。

## 结果（2026-10-25 15:25）

- 服务器可用空间：**15 G → 84 G**（`/dev/nvme0n1p2 468G 361G 84G 82%`）。
- 已移出服务器 **211 个 run**、**63.37 GiB / 68,042,035,599 字节**（超过 60 GB 目标）；对应云端 TAR 合计 73,776,271,360 字节（含 TAR 头/对齐开销）。
- 其中 11 个 run 已确认上传完成；其余 200 个的 TAR 已在网盘目录并继续从本机缓存上传，完成后状态由 `cached` 变 `uploaded`。
- 抽查：从网盘读回已归档 TAR 的前 512 字节，TAR 头正常、条目名为对应 run 目录。
- 清点：`batch01/` 下 TAR 文件数与 state 中已移出的 run 数一致（212 个），字节总和与逐 run 记录的 tar_bytes 之和完全相等（73,776,281,600，差 0），无缺失 TAR。
- 记录：`batch01/manifest.tsv`、`batch01/_cloud_archive_20261025.json`、`batch01/state_snapshot.json`、服务器 `runs/_cloud_archive_20261025.json`。

## 文档与结构

- 网盘批次说明：`B:/tts-exp/ubuntu-runs-archive/20261025/README.md`（含计数与恢复方法）。
- 归档根 README 与 `_index/navigation/CATALOG.md`、`folders/ubuntu-runs-archive.md` 已补充 20261025 批次、未压缩 TAR 格式与缓存优先删除策略。
- 服务器侧记录：`/home/wjj/tts-audio/tts-exp/runs/_cloud_archive_20261025.json`。

## 状态

已完成（2026-10-25T22:15+08:00）：212 个 run 全部上传完成（`uploads_pending=0`），收尾脚本已重发布最终记录，本机 CloudDrive 缓存中属于本批的数据已全部释放（只剩 2026-10-23 那个 15.7 GB 旧任务）。清理时服务器可用空间 15 G → 84 G；之后另一条 `audio200_nl17_*` 工作线在服务器上持续产出新 run，夜间回落到 68 G，属正常占用。`data/` 下与本地重复的 1.06 GB 未删除，留作后续可选项。

## Relations

- follows [[ubuntu-runs-cloud-retirement-60g-20261019]]
- implements [[050-ubuntu-runs-cloud-retirement-after-independent-restore]]
