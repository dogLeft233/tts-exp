---
title: Codex checkpoint - 2026-10-25T05-45-30Z - Four source WIMHF training smoke
  completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t05-45-30-z-four-source-wimhf-training-smoke-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T05:45:30Z'
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

# Codex checkpoint - 2026-10-25T05-45-30Z - Four source WIMHF training smoke completed

## Summary
40人技术复核/200人600源验收后，用户追加授权的4人音频解释小训练已完成：训练保存重载与独立矩阵读回通过，但单人留出偏好损失变差，不支持科学解释结论。

## Story
原任务是解释归一化后自然、0.6B和1.7B的SyncNet偏好。完整40人官方特征复核与600自然源已验收，严格七臂共同支持35/40而非37/40。用户现问“能先用少量数据验证下流程吗？”，随后明确选择“包含小规模试训练”，覆盖先前不训练限制的这一小试部分。采用既有fit来源、已有冻结音频512缓存，既避开昂贵视频生成，也保留11行原生时间结构，检查配对差学习能否连接到句对标签。完成后停止，不自动扩大训练。

## Working State
权威新结果 [[WIMHF音频解释流程的4人试训练 2026-10-25]]，runs/wimhf_audio_smoke_20261025_attempt01/report.md及protocol/result/pair_inputs/pilot.pt/learned_codes/readback_audit。11项3306725B双端SHA通过；finalmanifest SHA478dc380be519f4655b3c0a920668bd6cfc18e65d7b2d40176d42c5378a61baa。本机E:/Documents/tts-exp与Ubuntu /home/wjj/tts-audio/tts-exp同相对run都有完整本次小试；底层完整40特征仍远端。

S0002/S0003/S0004拟合、S0005工程留出，按来源ID前4，不看分数；28媒体60pairs858差补丁，所有人都是正式fit，未揭正式dev/confirm。CPU2threads、seed20261025，5632维=11×512，无初始avg/PCA或视频输入。SAE64/BatchTopK平均4/无偏置/列归一，128步batch32；尺度和推理阈值仅fit。冻结后事件平均码→无截距偏好头128步。|ΔC|≥.1，fit27主pairs/hold8；同模型pairs保留并前向，未用于此次头训练。

SAE fitMSE1.001520→.452967，hold.854260→.603499；偏好fitBCE.693147→.045996，hold.693147→1.472428。hold5/8=62.5%，多数类6/8=75%，fit100%。这提示小样本过拟合，不能判断科学路线成败或宣称已解释SyncNet。未完成稠密/混杂/三seed/命名忠实度/因果等正式验证，不是完整WIMHF复现。

前一checkpoint中的训练未授权仅为当时快照：现仅本次小试获授权并结束；全量训练无授权。4200媒体仍存储HOLD，600源100fit/40dev/60confirm冻结，完整40复核技术证据仍有效。新小试未改TTS/TFG/SyncNet，未新生成音视频或重评分。无运行中job。

## Changed Files
新增未跟踪 scripts/experiments/wimhf_audio_smoke_20261025.py 及 tests/test_wimhf_audio_smoke_20261025.py；新run约3.3MB。实验note写入并读回。继承dirty仓库未清理或提交；本地HEAD未被本轮修改。

## Verification
实际unittest2项PASS：signedTopK/阈值奇对称，保序与标签反号。128+128更新finitegrad；保存重载码/logit exact。另编audit.py不导入producer，重读28缓存SHA、全部858差输入并以矩阵复算码/汇总/60logit、官方标签差exact。该复算仍是同父会话非盲化复现。原manifest/patchindex未变。主运行3.411秒。双端11项完整SHA通过；一个无关SSH print诊断引号失败，不是训练失败，用stdin完成核验，无训练重试。
GitHub本轮只读确认dogLeft233/tts-exp；当前branch无PR；HEAD422不可远端解析，作为本机SHA保留，不构造commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local-only HEAD `2b71639c3b555293375d248350999377168d515d`。
- 40复核run runs/syncnet_audio_dataset40_parent_audit_20261025_attempt01/report.md；200源 runs/audio_data_inventory200_20261024_attempt05/report.md。

## Observations
- [result] 小规模完整学习/存储链通过，但一名工程留出的偏好损失恶化且准确率低于样本多数类。
- [decision] 只完成已授权128步SAE+128步头，不调参扩训；未产概念命名或因果结论。
- [blocker] 科学验证尚缺来源分组的稠密/稀疏对照及混杂基线；全4200存储/归档条件不足。
- [next_step] 等用户决定是否执行旧40fit来源的冻结分组探索验证，优先稠密/稀疏及混杂基线，不自动揭封dev/confirm或启动全量训练。

## Relations
- continues [[Codex checkpoint - 2026-10-25T05-40-21Z - Full40 independent audit and source200 accepted]]
- summarizes [[WIMHF音频解释流程的4人试训练 2026-10-25]]
- relates_to [[完整40人SyncNet特征独立复核与200人自然语料验收 2026-10-25]]
