---
title: Codex checkpoint - 2026-10-29T02-44-31Z - Scorer mechanism audit complete and
  no retraining validation design
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t02-44-31-z-scorer-mechanism-audit-complete-and-no-retraining-validation-design
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T02:44:31Z'
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

# Codex checkpoint - 2026-10-29T02-44-31Z - Scorer mechanism audit complete and no retraining validation design

## Summary
评分器训练负例与取分机制审计、共同窗口分解和AVS边界验证均已完成；当前转向设计不重训SyncNet的验证，区分普遍特征亲和、时间对应可辨识性与独立嘴部时序改善。

## Story
用户持续追问TTS驱动视频的Sync-C优势是否代表更好口型，已要求实际部署并用官方组件核查多个评分器。当前中文云Qwen/Wav2Lip GAN队列是20自然+60TTS、20来源、每来源3个TTS draw，保留RMS归一化；严格同参考12来源另报。不能把60配对当60独立来源，也不能将旧模型投票视为感知真值。

权威新入口是 [[scorer-negative-sampling-and-score-mechanisms-20261029]]。原论文SyncNet同clip时移负例通过margin对比损失塑造特征；目前V2完整训练配方不公开，不能将原论文训练履历完全套给V2。推理eval时5视频帧/20MFCC步的局部双塔不读取远处窗口；其他时间位置的训练影响与当前推理依赖是两件事。零时移原始距离不显式使用其他lag，但仍使用训练后的几何。因此训练负例影响表征有方法证据，造成TTS偏好尚无因果证据。

Synchformer与SparseSync均为21类、±2秒/200ms，实际配置训练jitter最多±50ms；同28原生媒体全部argmax零，连续P0是置信度排序。新增CPU检查发现Synchformer零类相对最强非零类margin的T−N为+1.553928，18/18正；每条logit向量去均值/RMS标准化后仍+0.243177、15/18正，不能归为纯统一logit尺度。两模型相同分类网格却偏好相反，反对“时移范围单独解释”。其他模型训练/端点及公开代码与论文差异见审计报告，不重复列整张表。

最新用户问：不重训SyncNet，能否用现有模型和本组数据区分真实同步与负样本塑造出的偏好。当前正在给出方案，尚未启动新实验。建议先用现有特征做严格保留窗口集合的异步配对null，比较原始零时移差、null差及对应特异收益，并用局部排名/已知时移恢复作辅助；这些仍只是评价器机制证据。要支持实际时间误差更小，需独立嘴部运动/发音事件的时间测量，并先验证对已知时移和空间干预的响应。历史已有打乱、排名、音素负池和纯视觉参照，不能包装成从未做过的新路线，也不能跳过独立标尺校准。

## Working State
可持久恢复的完成产物：
- basic-memory/docs/experiments/scorer-negative-sampling-and-score-mechanisms-20261029.md：本次论文/官方代码/部署配置审计最终报告，BM读回核对完成。
- basic-memory/docs/experiments/73-common-window-cloud-curve-decomposition.md 与 runs/common_window_curves_cloud_20261028_attempt01/：80媒体共同窗、原始特征、31 lag曲线和独立审计。SyncNet零时移ΔS0=+0.184，42/60 T胜；contrast最佳贡献+0.247、背景贡献+0.304，总+0.551。背景减最佳贡献的CI跨0，不宣称背景主导。新的canonical音频重算修正旧自然AVI+1105样本约69ms的音轨基准，旧新差不能全归窗口。
- basic-memory/docs/experiments/74-avs-boundary-context-vs-silence-mouth-state.md 与 runs/avs_boundary_content_cloud_20261028_attempt01/：固定目标输入、改变边界/上下文可大幅改变AVS分数；非低能量张嘴目标也出现。不把AVS全片或中央评分当真值。
- runs/cloud_mtd_synchformer_20261028_attempt01/synchformer_scores.json 与 runs/sparsesync_official_cloud_rescore_20261028_attempt01/scores.json：已有logits，28媒体/18配对/6来源。不要重启旧SparseSync下载或评分；报告72已结项。

机器本地或未提交的状态：
- .codex/scratch/scorer_training_audit_20261029/：confidence_diagnostic.py、confidence_diagnostic.json、independent_check.json和两个阶段官方配置，当前文件存在。两个研究子代理草稿为 .codex/scratch/sync_negative_training_audit_20261029.md 和 representation_negative_training_audit_20261029.md；已完成，无需再派发。
- Ubuntu仍以 ssh -o BatchMode=yes wjj@192.168.50.2、repo /home/wjj/tts-audio/tts-exp 为已有执行入口。此次未查询GPU/磁盘实时状态，没有新GPU或TTS/TFG作业；不要将旧资源快照当现在空闲。
- git status --short 当前21条修改、1320条未跟踪（目录可能折叠），大量继承工作属于其他任务；本轮不改、不清理、不提交、不推送。不要将新出现的其他任务脚本当作本会话成果。EARS的新任务状态本轮未核实。

## Changed Files
前序本会话新增研究CPU脚本及结果在 .codex/scratch/scorer_training_audit_20261029/，最终审计报告由BM保存。报告73/74对应 scripts/experiments/common_window_curves_20261028.py、common_window_curves_independent_audit_20261028.py、avs_boundary_content_20261028.py、avs_boundary_content_independent_audit_20261028.py 及各自同名tests，保持未提交状态。本次最新问题仅只读核对和方案推理；除本checkpoint外没有修改评分代码、原媒体、权重或冻结协议。

## Verification
前序实际完成：报告73和74各有ROOT独立重建及6项本地artifact测试PASS，详见报告与run中的审计receipt。本次重新确认两份独立审计文件及CPU材料存在；没有重新运行这些测试或GPU前向。

CPU置信度诊断使用已有两模型28+28组logits；独立stdlib审计不导入producer，56媒体检查PASS，最大误差8.881784197001252e-16。独立检查不等于独立复算全部bootstrap。诊断JSON SHA256 a8df98a5693eb08ee12ea1d8839dcff3c0e6a3a46a2349b27baa5523a90e08d3。

本次已全文读取Startup Router、实验指令、新审计、报告73/74、2026-09-26共同音素负池、2026-09-17时间辨识/实例交叉及机制研究枢纽。旧负池结果依赖lag，未支持简单负池/范数单因子解释；旧独立配对null还保留粗位置且短域有偶然对应；旧GRID完整几何校准仅3/8通过，相关性开发7/8通过，故不能直接复用为通用嘴型时间真值。

Git根/branch/SHA、用户名和主机只读核实。GitHub插件及REST确认canonical仓库；当前branch open PR搜索为空。REST查询本地SHA返回422，未确认其可从GitHub解析，保留本地SHA不构造远程commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地SHA：`2b71639c3b555293375d248350999377168d515d`，远程可解析性未确认。
- [SyncNet official inference](https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py)
- [SyncNet official model](https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py)

## Observations
- [result] 共同窗口下原始零时移仍偏TTS，因而搜索与背景中位数不是完整解释；不能由此直接推出真实口型更准。
- [result] AVS当前复现有明确边界/上下文敏感性，其他评分器不能按票数构成独立真值。
- [decision] 不重训验证应报告“与时间对应无关的特征亲和”“对应可辨识性”“独立可见事件时间误差”三个不同证据层；null扣除和单位化属于派生诊断，不替换官方分数。
- [decision] 主端点保留物理零时移；若需固定偏移校准，只能独立控制样本预定，不能逐样本搜峰后宣称原生零误差。
- [limitation] 现有模型和数据可检验部分评分偏好解释，但无法单独识别哪一种历史训练负采样导致偏好；两个机制也可能并存。
- [next_step] 向用户完成不重训的分层验证方案说明；若后续要求执行，先冻结本组80媒体的完整窗口异步null及已知时移设计，独立时序标尺须先过校准，不将当前方案当已运行结果。

## Relations
- continues [[Codex checkpoint - 2026-10-28T10-32-31Z - SparseSync scoring complete and Synchformer endpoint interpretation verified]]
- relates_to [[scorer-negative-sampling-and-score-mechanisms-20261029]]
- relates_to [[73-common-window-cloud-curve-decomposition]]
- relates_to [[74-avs-boundary-context-vs-silence-mouth-state]]
- relates_to [[原生 TTS 优势的共同音素负样本池干预 2026-09-26]]
- relates_to [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]
