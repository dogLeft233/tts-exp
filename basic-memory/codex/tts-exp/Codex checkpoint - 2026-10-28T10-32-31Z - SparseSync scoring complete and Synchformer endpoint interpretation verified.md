---
title: Codex checkpoint - 2026-10-28T10-32-31Z - SparseSync scoring complete and Synchformer
  endpoint interpretation verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t10-32-31-z-sparse-sync-scoring-complete-and-synchformer-endpoint-interpretation-verified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T10:32:31Z'
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

# Codex checkpoint - 2026-10-28T10-32-31Z - SparseSync scoring complete and Synchformer endpoint interpretation verified

## Summary
SparseSync官方对照、评分和审计完成；最新核对确认Synchformer的TTS优势属于官方模型零时移置信度差异，全部原生媒体最佳时移均为零。

## Story
研究目标是区分TTS驱动视频的真实同步改善与SyncNet评分偏好。用户要求SparseSync先确认官方一致再评分。现已完成：报告72和部署指南是权威入口，旧10:04下载中checkpoint已被完成结果取代。最新用户问Synchformer是否官方流程、为何它与SyncNet支持TTS。已读取报告68、部署指南、实际评分脚本和保存的28组输出，并查官方README/example.py。回答须区别官方模型/预处理/前向与项目连续P0端点，不能将置信度当作物理同步误差或以模型票数判真值。

## Working State
权威报告 basic-memory/docs/experiments/72-sparsesync-official-native-cloud-preference.md；部署 basic-memory/docs/deployment/sparsesync.md；run runs/sparsesync_official_cloud_rescore_20261028_attempt01。Windows及Ubuntu均保存小结果和源码快照，Ubuntu部署根 /home/wjj/av-sync-eval-20261028，SparseSync-source为有效入口。653037857字节完整权重严格加载54348652参数；官方commit a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f，31源码文件gitblob验证。独立venv /home/wjj/.venvs/sparsesync-20261028，Torch2.5.1/PyAV12.3，FP32；非2022锁版本逐位重现。下载缓存还在，勿重复下载/评分，勿杀外来GPU任务。

SparseSync保留原RMS和原生音轨、不交叉、不补短片、不搜最佳lag。官方5秒支持28媒体（10N/18T），18对/6来源。P0 T胜6/18=33.3%，平均T−N −0.00931124百分点，来源bootstrapCI [−0.02314945,+0.00018742]百分点跨0；与同18对Sync-C冲突12/18=66.7%，全为SyncT/SparseN。全部28预测零时移，P0>99.95%，不可将细小尾概率排序当自然口型优越。Synchformer与SparseSync同支持18对：P0 T胜15/18，均差+1.29367百分点；自然均值98.4113%、T99.7050%。本轮直接读取原始scores确认自然10/10、T18/18全部零时移类；N P0范围92.6779–99.8888%，T98.7601–99.9799%。

Synchformer用了官方LRS3 23-12-23T18-33-57，官方patch_config/get_transforms/prepare_inputs/forward和cfg半精度，严格载入完整模型，28音轨解码与原WAV逐点相等，20/20真实±400ms P0控制通过；P0为项目比较端点，作者benchmark是Acc@1/±1类。仅6来源且中心5秒对比旧全片Sync-C。没有找到像SparseSync那样实际官方main与批处理逐logit数值对照记录，不得声称已做。PyAV12.3与作者8/9不同，官方README明确版本会影响概率；当前不可宣称旧环境bitexact。原因尚未因果定位，特征/时间支持/训练任务/声学域均只能作为待验证假设。

## Changed Files
前序新、未跟踪：scripts/experiments/sparsesync_official_cloud_rescore_20261028.py 与 tests/test_sparsesync_official_cloud_rescore_20261028.py；报告72和部署指南已保存读回及复制Ubuntu。本轮仅只读核对，没有新增评分或修改代码。大量继承dirty/untracked保留，无commit/push/reset/clean。GitSHA为本地证据，未核验GitHub可解析该SHA。

## Verification
SparseSync：真实官方scripts/example.py main先运行，音频/视频/logit最大差0，概率差5.55e−17；28媒体PCM逐点相等；20/20±400ms P0下降且相对类别时移准确；独立NumPy复算28softmax/两端点与36配对端点PASS，最大概率差1.09e−7；FP64事后复算18/18方向一致。前序Ubuntu和Windows各4真实artifact测试PASS，py_compile PASS。pytest初缺只安装独立venv/本地cache后补测，无GPU重跑。本轮实际读回chain_status COMPLETE_OFFICIAL_PARITY_SCORES_AUDIT_TESTS及official_parity PASS。当前Synchformer代码/结果与报告核对、GitHub REST确认仓库canonical及当前branch open PR=[]；无新GPU测验。CodeGraph先调用但未返回相关新文件，随后读取已知未索引脚本。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Synchformer official source](https://github.com/v-iashin/Synchformer)
- [Synchformer official example](https://github.com/v-iashin/Synchformer/blob/b66668a1521d7567cc760e5544b2b5b53179b687/example.py)
- runs/cloud_mtd_synchformer_20261028_attempt01/synchformer_scores.json
- runs/sparsesync_official_cloud_rescore_20261028_attempt01/{chain_status,official_parity,saturation_precision_diagnostic}.json

## Observations
- [result] SparseSync请求已全部完成，无下载/评分待办；报告72与最终run状态替代旧下载中checkpoint。
- [result] Synchformer及SparseSync对同28原生媒体全部预测零时移；相反偏好来自置信度排序，不是最佳时移类不同。
- [decision] P0连续端点与官方benchmark accuracy分开陈述；官方组件复用不等于已做官方main逐logit对照。
- [decision] 不将模型票数、置信度饱和或粗时移控制通过作为SyncNet偏置或感知真值证明。
- [next_step] 回答用户官方流程与偏好分歧的解释；若继续实验，优先做Synchformer官方main数值一致性审计及统一5秒/来源支持的对比，再研究小幅已知时移剂量响应，尚未启动新实验。

## Relations
- continues [[Codex checkpoint - 2026-10-28T10-04-52Z - SparseSync official parity pipeline prepared and weights downloading]]
- relates_to [[72-sparsesync-official-native-cloud-preference]]
- relates_to [[68-cloud-mtd-synchformer-native-preference]]
