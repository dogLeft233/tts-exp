---
title: Codex checkpoint - 2026-10-28T10-04-52Z - SparseSync official parity pipeline
  prepared and weights downloading
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t10-04-52-z-sparse-sync-official-parity-pipeline-prepared-and-weights-downloading
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-28 10:04:52+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-28T10-04-52Z - SparseSync official parity pipeline prepared and weights downloading

## Summary
用户已授权部署SparseSync并给既有云TTS/自然视频评分；官方输入检查通过，权重仍下载，真实官方输出对照和评分尚未运行。

## Story
原版VocaLiST完整评分已结束，权威结论保留报告71。用户进一步要求SparseSync评分，且强调评分前确认与官方方法一致。选择官方LRS3 NoFaceCrop 21类22-07-13T22-25-49，严格5秒原配置；主要端点为官方softmax零时移概率，属于项目连续偏好端点而非官方数据集accuracy。批量前执行官方scripts/example.py main，完整strict权重载入，仅省略重建可视化；捕获输入、logits和概率对照直接调用，必须通过才进入控制和评分。

## Working State
Ubuntu部署 /home/wjj/av-sync-eval-20261028/SparseSync-source，官方GitHub main a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f；经GitHub recursive tree下载31个源码/配置文件，逐个git blob SHA1核对。未改官方源码。独立venv /home/wjj/.venvs/sparsesync-20261028 使用共享runtime .pth，只读引用synchformer、cogenav及cross_tfg torch251环境，torch2.5.1+cu121/torchvision0.20.1/PyAV12.3。导入及CUDA可用通过，非2022锁版本重现。extractor初始化ckpt_path=None避免额外预训练权重，随后完整同步模型strict加载。

正式run runs/sparsesync_official_cloud_rescore_20261028_attempt01 已冻结protocol/inputs/sync_pairs/source_snapshot。沿用cloud_mtd_synchformer_20261028_attempt01清单和PCM MOV：80媒体中28可用5秒，10自然/18TTS，18完整配对、6独立来源；52短媒体不补齐、不拉伸。此前RMS保持不变，无交叉音轨，无最佳时移修正。28媒体官方读取器与原WAV逐样本相等，官方transform输出video[1,125,3,224,224]、audio[1,1,257,626]，preprocessing_validation.json PASS。

权重目标 /home/wjj/av-sync-eval-20261028/weights/sparsesync/22-07-13T22-25-49.pt，官方总长653037857。单线程下载太慢且已终止自有PID77449，4MiB块32线程主下载当前工具exec session30824，进程77932；前约20MiB已复用。辅助512KiB分块64线程session86057负责第70大块以后，最终并入chunks供主下载组装；10:04约52大块完成，128辅助小块。偶发IncompleteRead会重试，绝不把.part当完整权重。原完整git clone已终止，仅以已验证source目录为入口；Windows多余源码ZIP下载自有进程429380已停止。未删除归档或改网络设置。

执行链exec session98614正等待最终.pt，然后freeze刷新源码快照、run官方parity及20物理±400ms控制/28原生评分、audit、对应pytest。chain_status.json WAITING_FOR_OFFICIAL_CHECKPOINT。若控制P0下降不足80%，保留CONTROL_FAILED，不将偏好当同步真值。独立NumPy审计从原logits复算softmax、端点和严格配对，源bootstrap10000 seed20261028。所有会话进程是机器本地易失状态；如恢复先检查链状态，避免重复启动。Ubuntu剩余约8.8GB；外来GPU约5.6GB保留，未终止别人任务。

## Changed Files
- scripts/experiments/sparsesync_official_cloud_rescore_20261028.py（新、未跟踪）：冻结、官方main对照门控、输入精确检查、时移控制、原生评分、独立审计和统计。
- tests/test_sparsesync_official_cloud_rescore_20261028.py（新、未跟踪）：只读真实artifact集成检查，权重/官方输出对照、支持集分母、真实控制和审计。
大量原有dirty/untracked保留，无commit/push/reset。协议和原数据未改。最终报告计划basic-memory/docs/experiments/72-sparsesync-official-native-cloud-preference.md，尚未创建；部署指南尚待完成验证后保存。

## Verification
已运行：官方31源码blob验证、模型模块导入、CUDA探测、28媒体逐样本PCM/官方预处理全覆盖PASS、本地和Ubuntu py_compile PASS。权重下载未完成，因此尚未strict模型加载、GPU前向、官方logit/probability对照、时移控制、原生评分或pytest；不得提前宣称这些通过。Git branch/root/SHA/status只读核对；GitHub REST确认canonical仓库和当前分支open PR=[]，SHA未远端证明，仅本地SHA。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [SparseSync official source](https://github.com/v-iashin/SparseSync)
- [Official inference example](https://github.com/v-iashin/SparseSync/blob/a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f/scripts/example.py)
- runs/sparsesync_official_cloud_rescore_20261028_attempt01/{protocol,preprocessing_validation,chain_status}.json

## Observations
- [decision] 坚持官方5秒原配置、18配对/6来源，不用短片填充或改长度冒称官方。
- [decision] P0和negative expected absolute offset为预先固定连续端点；保留全部21类原logits以供复算。
- [result] 28媒体输入精确核验通过；正式模型前向尚未开始。
- [blocker] 官方653MB权重下载慢，仍在续传。
- [next_step] 等权重完整，观察session98614官方对照门控及评分链；通过后拉回全部小产物、跑本地审计测试、保存报告72并向用户报告胜率/平均差/Sync-C冲突及控制限制。

## Relations
- continues [[Codex checkpoint - 2026-10-28T09-40-13Z - Original VocaLiST scoring complete and SparseSync availability checked]]
- relates_to [[71-vocalist-official-native-cloud-preference]]