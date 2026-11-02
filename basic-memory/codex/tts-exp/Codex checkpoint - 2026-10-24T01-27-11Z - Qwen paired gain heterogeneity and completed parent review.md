---
title: Codex checkpoint - 2026-10-24T01-27-11Z - Qwen paired gain heterogeneity and
  completed parent review
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-24-t01-27-11-z-qwen-paired-gain-heterogeneity-and-completed-parent-review
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-24T01:27:11Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-24T01-27-11Z - Qwen paired gain heterogeneity and completed parent review

## Summary
独立复核完成；1.7B优势依赖来源，现有正负配对的评分构成可定位，但有界声学检查尚未找到跨两批稳健的上游解释。

## Story
研究目标是解释归一化后TTS/1.7B的Sync-C优势，而非先训练方法。用户要求子代理执行双规模条件预测机制实验，随后要求父代理独立复核并解释换10来源后优势消失。本轮恢复前，父代理已完成A/B/C直接数组与统计复核，报告runs/qwen_prediction_root_review_20261024_attempt01/。旧10来源L17−L06+.656，新10−.201；旧共同剔两条停止异常仍+.392，整剔两来源+.339；新所有基线EOS/时长合格、逐删任一来源仍负均值。两批实际官方pipeline→syncnet命令、权重、脸、RMS、PCM和采样参数检查未发现足以解释反转的流程差异。来源/文本/参考/seed数值同时变化，不能归因单独身份或短句。用户问“两次都是官方流程吗”已回答是。

最新用户追问能否定位有增益和无增益样本区别与成因。按L17−L06定义，CPU检查20来源×3draw×2模型120条已有norm音频，六项固定声学特征及参考字数/时长；不做新合成、TFG、SyncNet前向、训练或特征/阈值扩搜。旧S0002/S0007三draw全正，C差+.956/+.403，最佳距离项−.152/−.124，但偏移中位距离项+1.108/+.527；新S0013三draw全负−.220，最佳项+.155、中位项−.375；新S0014三draw全负−1.221，最佳项−.881、中位项−.340。正负内部并非一种变化，Sync-C不是嘴型准确性。

能量变异系数与分差来源间相关全draw旧/新均−.673，去旧两条预定不合格draw后旧−.091、新−.673；旧删一来源相关可变号。谱变化资格后旧+.430、新−.624也不一致。当前没有跨批稳健声学解释。已有共同历史细码熵降低仍验证，但旧熵—分差强相关去异常后−.195；C温度操纵1.7未全达目标，P1/P2均未支持，不能称熵因果机制已否证。尚无新人工或独立视觉验证。

## Working State
权威新输出runs/qwen_native_pair_features_20261024_attempt03/{report.md,analysis.json,readback.json}，本机E和Ubuntu /home/wjj/tts-audio/tts-exp均有；analysis SHA bf3b92f3c8f6738cf3f97d9d8c5d617f873134ea114c5175b288b9f0b7006d37。完整原子代理run为qwen_prediction_mechanism_20261023_attempt02；3GB tar在C:/tts-exp-experiment-backups/，父代理没有再做完整tar复核，不声称云备份。

工作区大量继承modified/untracked，git status本轮直接读取；不清理/提交。当前分支codex/cross-tfg-audio-head-20261003，HEAD本地2b71639c3b555293375d248350999377168d515d，未验证GitHub该SHA可解析，不构造commit链接。GitHub只读确认canonical repo，当前分支PR搜索为空。gh命令未安装，用GitHub连接器完成只读查询。默认Windows Python无numpy，使用已有远端Qwen Python CPU环境；无安装。没有授权启动的新GPU任务。

## Changed Files
新增未跟踪scripts/experiments/qwen_prediction_mechanism_20261023/root_pair_features_20261024.py。此前父代理新增root_review_20261024.py与root_cohort_diagnosis_20261024.py，HANDOFF.md载有独立复核摘要，本轮未改HANDOFF。BM正式实验《Qwen双规模实验独立复核与来源差异诊断 2026-10-24》追加配对定位，研究地图同步并读回。原封存科学run未改。

## Verification
此前父代理直接核验A原生/TF数组、编码与波形；B140trace概率/熵、源聚合与bootstrap；C120唯一评分加旧60norm曲线共180track，602冻结pins、22共享pins、60规范化PCM重放及P1/P2统计通过。它们不是新独立模型前向。当前CPU脚本成功退出0，120音频/20来源计数检查、20项C分解恒等式与从逐音频特征重聚合的共同资格相关重算通过；不称独立第二声学测量。未运行完整pytest。

本轮attempt01错误使用仅新增音频audio_manifest，缺复用0.6B norm路径而KeyError，未写结论；改comparison_index后attempt02完成，attempt03追加共同资格敏感性，保留所有目录不覆盖。前轮PCM float64-floor模拟曾遇1LSB写入边界差，固定libsndfile内存写入后60条一致，不是数据异常。无新的科学结果因失败而被丢弃。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/qwen_prediction_root_review_20261024_attempt01/report.md
- runs/qwen_native_pair_features_20261024_attempt03/report.md

## Observations
- [result] 正负配对内部有不同最佳距离/偏移背景变化；“有增益”不等于嘴型更清晰。
- [decision] 保留当前效应条件依赖判断；表面能量/谱相关未过跨批稳健性，不晋升因果解释，不训练。
- [blocker] 同一来源只有一句且seed数值随来源变化；缺少稳定正负案例局部独立视觉/发音事件检查，不能分离语音、生成端与评分表征。
- [next_step] 若继续定位，使用已有稳定正旧S0002/S0007和稳定负新S0013/S0014完整音视频检查局部距离与闭唇/释放事件；标记为事后案例定位，不作为独立确认收益集。

## Relations
- continues [[Codex checkpoint - 2026-10-23T13-10-41Z - Generator hypotheses for normalized Sync-C gain]]
- summarizes [[Qwen双规模实验独立复核与来源差异诊断 2026-10-24]]
- informs [[研究主线与证据地图]]
