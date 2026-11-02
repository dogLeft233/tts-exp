---
title: Codex checkpoint - 2026-10-24T08-47-00Z - WIMHF design and audio-only explanation
  scope
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-24-t08-47-00-z-wimhf-design-and-audio-only-explanation-scope
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-24T08:47:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-24T08-47-00Z - WIMHF design and audio-only explanation scope

## Summary
WIMHF式局部差异解释方案已保存并核对，最新用户把重点收窄到音频；推荐音频-only特征发现与解释，未来以冻结视频验证评分端依赖，仍然只设计、不训练。

## Story
研究目的是解释自然音频、Qwen0.6B和1.7B的SyncNet评分差异，尤其同协议下换来源后优势反转。旧HuBERT全句mean/std→PCA8→Ridge探针没有增量预测价值，但该简化是项目选择，非WIMHF原版，负结果不能否定所有时序或SAE解释。重查WIMHF论文确认它采用embedding差、带符号稀疏字典、原样本描述、独立描述忠实度与偏好关联；PCA为比较项，Matryoshka前8特征不是先PCA8。

用户要求先设计更忠实的音视频方案，不运行。已形成SPEC_ONLY计划：冻结实际SyncNet两支路最终Linear前512维，11个原生40ms格点保序拼接、同文字occurrence各自在原生时间定位、配对差→signed SAE→局部概念后句级汇总与偏好头。保留未压缩差表示与双输入非线性对照，分开重建、可读忠实、留出预测、原模型内部干预、物理原因/人类质量五层证据。窗口约600–615ms，不能称保留所有长程顺序；20来源仅探索，未来120来源划分是成本示例，不是功效保证或执行授权。

最新用户问能否不借视频证据、主要需要音频解释。方案第11节新增范围修订建议：发现器只用音频512维×11格点，音频原片段/波形/谱图用于命名与独立核验，不输入视频或视频特征。视频可能给联合解释器提供捷径，也可能承载真实互动，不能称必然干扰。原生分数C(A,G(A))仍包含生成视频变化，移除输入不能自动消除这条路径；只用音频预测原生分差首先是整链相关线索。以后检验评分端时，固定每条自己的原生视频，只改同一发声的候选音频内部方向或受控波形因素，再用原SyncNet检查。禁止直接将N/0.6/1.7不同发声全换到同一视频来声称音频纯效应，时间失配是已知替代解释。

## Working State
权威计划：basic-memory/Experiments/WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24.md，BM permalink tts-exp/experiments/wimhf-式局部视听差异与-sync-net-偏好解释方案-2026-10-24。状态planned/SPEC_ONLY，training_started=false，execution_authorized=false。第11节是最新音频优先方向；前面的联合表示设计保留作历史设计，不应默认执行。

已完成旧探针与独立复核见自然音频与双规模TTS的SyncNet偏好探针 2026-10-24及runs/syncnet_preference_probe_20261024_attempt01/。旧主MAE M1 .580837、M3 .597757，未增益；不改冻结结果或原停止门。新设计另立，非重写失败为成功。

工作区E:/Documents/tts-exp是唯一活动本机入口。大量继承修改与未跟踪文件仍在，未清理/提交。当前Git SHA只作为本地标识；前次GitHub解析422，本轮未重试commit，不构造在线commit链接。本轮只读GitHub仓库身份与当前branch PR查询，未找到PR。旧140媒体、缓存可用性和远程进程状态未重新检查，不据历史状态称当前正在运行。

## Changed Files
- basic-memory/Experiments/WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24.md：前阶段新建详细方案，本阶段加入第11节音频优先范围建议和版本元数据。
- basic-memory/Research/研究主线与证据地图.md：加入WIMHF设计导航及音频-only边界。
- 无实验代码修改，无新训练/特征提取/评分/生成/人评/媒体上传。本checkpoint为新的不可变handoff，不编辑旧checkpoint。

## Verification
前阶段已核对WIMHF原论文、官方SyncNetModel及本机源码的512中间层/1024输出，验证设计文件存在并读回。本阶段重读canonical计划及地图；重查官方SyncNetInstance.py确认双分支匹配、时间窗与median-minus-min评分。两次设计/地图编辑成功后read_note读回确认新段存在。
本轮git status、branch/root/SHA、用户/主机读取成功；GitHub连接器确认dogLeft233/tts-exp且head PR结果为空。没有运行科学测试或独立复算新实验，因为当前只设计。当前助手没有音频直接听辨能力，描述核验不得假称听过。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [WIMHF paper](https://arxiv.org/html/2510.26202v2)
- [Official SyncNet scoring](https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py)
- [Official SyncNet model](https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py)

## Observations
- [decision] 无前置整句avg pool或PCA8；按原生时间保留局部结构，再检验SAE压缩与描述忠实度。
- [decision] 根据最新用户重点推荐音频-only发现器；视频只在以后原评分验证作冻结参照，不作解释器输入或特征命名证据。
- [decision] 完全不使用视频的验证最多支持音频特征与既有原生分数的可预测关联，不能证明纯评价器偏好原因。
- [decision] 原SyncNet音频内部干预只支持模型条件依赖；真实声学原因仍需可实现波形控制，人类质量需独立证据。
- [blocker] 512缓存覆盖、完整140条文字定位/原生时钟映射和可测支持尚未审计；SAE/偏好实现和资源未验证；独立来源仅20，不能把窗口数当样本量。
- [next_step] 继续按用户反馈收敛音频优先方案；只有用户后续要求执行时才先审计缓存与对齐支持并冻结最小技术验证，不自动启动训练。

## Relations
- continues [[Codex checkpoint - 2026-10-24T08-03-15Z - Preference probe negative and bounded SyncNet mechanism route]]
- summarizes [[WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24]]
- relates_to [[自然音频与双规模TTS的SyncNet偏好探针 2026-10-24]]
- relates_to [[研究主线与证据地图]]
