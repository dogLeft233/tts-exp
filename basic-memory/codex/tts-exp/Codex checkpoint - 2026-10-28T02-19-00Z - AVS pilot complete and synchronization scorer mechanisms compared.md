---
title: Codex checkpoint - 2026-10-28T02-19-00Z - AVS pilot complete and synchronization
  scorer mechanisms compared
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t02-19-00-z-avs-pilot-complete-and-synchronization-scorer-mechanisms-compared
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T02:19:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-28T02-19-00Z - AVS pilot complete and synchronization scorer mechanisms compared

## Summary
中文AVS_u先导完成并独立重放；跨评分器分歧用于提出机制假说，最新问题转向比较评分原理与寻找更独立的同步测量。

## Story
原目标是区分TTS生成口型改善与SyncNet评价偏好。云Qwen20来源自然/60TTS原生比较已完成：Sync-C倾向TTS，CoGenAV、SyncReward、AVS_u没有一致的平均TTS优势，但这些不是独立真值投票。用户不倾向依赖缺乏专家的人评，并要求检查BM历史实验。本轮回顾发现历史编码链确有因果混淆，但统一PCM后另有剩余增益，生成端音量干预也确实能改变嘴部运动；不能归结为全部评分偏置。随后提出时间局部曲线分解作为新假说检验，尚未执行。
最新用户明确PEAVS先不管，询问另外三种（按CoGenAV、SyncReward、AVS_u理解）的架构/原理相似度，以及更不同的TFG同步指标。本轮直接核对论文与官方源码：CoGenAV和AVS_u在AV-HuBERT家族及跨模态特征cos上较近；CoGenAV增加逐帧正负对学习和文本生成目标；SyncReward用PE-AV编码、跨模态Transformer及人类评分回归/排序。SyncNet V2的Sync-C是时移距离曲线median减minimum，不能与直接cos等同。候选互补路径为PhoVis显式音素-视素空间一致性、VocaLiST跨模态同步分类、Synchformer时移分类及发音事件时间误差。后三类仍有学习表征/域偏好；PhoVis代码权重可用性尚未核实。ATR新论文输入静音视频和候选文本，不直接检验实际音轨的起止时刻，不能作为同文本不同时间结构的替代评分。

## Working State
权威AVS run仍在Ubuntu /home/wjj/tts-audio/tts-exp/runs/avs_u_zh_pilot_20261027_attempt01；本机可用最终小产物在C:/Users/dogle/.codex/tts-exp-artifacts/avs-u-zh-pilot-20261027/。E盘本轮只读确认Free=0；E最终JSON副本可能零字节，禁止当有效结果。完整新run未全归档。保留暂停生产和取消交叉音轨，不重启或关闭外来任务。
AVS COMPLETE80，40真实PCM时移控制32下降，正好达到冻结门槛。all20自然.1163565950、TTS.1088280157、差−.0075285793；T胜23/60，自然37/60，冲突43/60（SyncT/AVSN34，反向9）。strict12差−.0066387291，T胜11/36，冲突28/36。来源bootstrap区间跨0；20来源而非60独立样本。它是论文定义复现，原文精确checkpoint未确认，不是作者官方scorer。
SyncReward官方BF16已知时移仅14/40下降，控制失败；历史完整云原生分数仍FP32，没有全云BF16重评。PEAVS依最新用户范围搁置。
当前仓库1302条dirty/untracked状态，均保留；本轮未改代码、未新生成或评分，未提交推送。项目配置使用coding profile，用户级basic-memory.json不存在。

## Changed Files
前序新增未提交scripts/experiments/avs_u_zh_pilot_20261027.py及tests/test_avs_u_zh_pilot_20261027.py；本轮仅创建此handoff，未改实验源码。AVS现有Task最终结果与E盘更正已全文读回。

## Verification
前序独立重算全部80 embedding主分max误差0、60 Sync-C差值逐项一致，三个直接测试函数通过，未pytest。本轮读取C副本status.json确认COMPLETE80及parent_final_statistics.json确认all20/strict12和max误差0；没有重跑测试。读取BM官方审计及AVS任务全文，验证旧结论的范围。直接读SyncNet上游SyncNetInstance.py确认conf=median(mdist)-minval。核对CoGenAV Eq1、AVS Eq4/5/6、SyncReward两阶段及VocaLiST分类结构。
本轮只读git证据root E:/Documents/tts-exp、branch codex/cross-tfg-audio-head-20261003、SHA 2b71639c3b555293375d248350999377168d515d；GitHub REST确认canonical repository及该分支open pulls=[]。SHA未核验GitHub可解析，不构造commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV](https://arxiv.org/html/2505.03186v1)
- [AVS metrics](https://arxiv.org/html/2405.04327v1)
- [SyncReward](https://arxiv.org/html/2610.16338v1)
- [PhoVis](https://www.amazon.science/publications/perceptual-synchronization-scoring-of-dubbed-content-using-phoneme-viseme-agreement)
- [VocaLiST](https://arxiv.org/html/2204.02090)
- [Synchformer official models](https://github.com/v-iashin/Synchformer)
- runs/avs_u_zh_pilot_20261027_attempt01/parent_final_statistics.json
- 历史因果证据：Experiments/中的TTS原生增益音轨审计与时间表交叉实验、SyncNet编码偏差的码率频谱与表征机制拆解、统一无损音轨后的跨模型剩余优势与自然音频替换审计（均2026-09-26）；固定视频与实际评分PCM留来源谱形干预2026-10-25为窄范围阴性，勿重复包装成新实验。

## Observations
- [result] AVS_u全部80原生评分完成，T胜38.3%，与Sync-C冲突71.7%；均值差区间跨0，不证明自然总体更好。
- [decision] 先搁置PEAVS；架构不同和多个指标一致都不足以建立同步真值，先明确内容匹配、时间定位与评分背景三个端点。
- [decision] 不把普通唇读CER/WER当局部时间准确度，也不能在评分前用自由DTW消除待测时差。
- [blocker] E盘已满，完整AVS媒体/embedding仍仅以Ubuntu为权威；PhoVis即用官方实现未核实，中文事件/视素可靠性需控制验证。
- [next_step] 回答三评分器机制差异及互补指标选择；如用户授权落地，优先核实PhoVis官方代码权重并用已知时移和保持时间结构的声学扰动验证候选，不直接重做已完成的全局EQ或预测探针。

## Relations
- continues [[Codex checkpoint - 2026-10-28T01-39-29Z - Mandarin AVS weight verified and native scoring running]]
- relates_to [[AVS_u中文域同步控制与自然TTS原生比较 2026-10-27]]
- relates_to [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]
