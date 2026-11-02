---
title: TTS 机制确认分离生成端与评分端
type: decision
permalink: tts-exp/docs/adr/tts-机制确认分离生成端与评分端
status: open
project: tts-exp
decided: 2026-09-30 15:50:00+00:00
tags:
- tts
- tfg
- causal
- protocol
---

# TTS 机制确认分离生成端与评分端

## Context

历史同PCM Wav2Lip/Qwen配对仍有较高Sync-C，但原生N/T同时改变声学内容、时长、生成器输入和评分器输入。旧9/27部分数值未恢复，现有WORLD缓存缺V与producer。用户授权自主重跑、维护主线并用BM持续记录。

## Rationale

采用新说话人、先固定规则的20条校准/80条评估，能够独立确认来源方向是否可转移。固定评分自然PCM、时钟、face和support，把视频侧干预与评分音频侧干预区分开。SyncNet已知延迟控制显示最优lag分数可维持高分，因此同时使用cal-only固定anchor。音量严格正交干预只排除实测level secant；独立视觉与后续物理source/filter另取证，保持结论边界明确。

## Affected work
来源、WORLD/overlay、female及direction-radius协议均在 openspec/changes/；对应独立runner与消费者在 scripts/experiments/。当前branch codex/tts-mechanism-20260930，HEAD为本地未推送072c260，当前分支GitHub查询无PR。已推送baseline仍c4a199890bd529928ec34b9be6da86a10a737c97；不得把本地commit构造为可访问GitHub链接。
## 视觉分类与端点资格版本

首读正式效果前，CAL-only资格检查发现冻结旧visual normalizer没有剥除普通话IPA声调，因此a˥等元音未进入open-vowel类，旧版本CAL=0 open-vowel支持。选择新建CAL-only visual版本：adapter新增步骤只移除U+02E5至U+02E9的五个tone bars，随后调用既有normalizer。既有规则会剥ː、ʰ、ʲ、ʷ、̪，因此最终class classifier共剥十个字符；先前“保持其他标记”的绝对表述在源码复核后更正。保存raw-to-base表、旧快照与新源码SHA，实际冻结映射和数值不改。物理phone identity仍为NFC/strip且保留声调，不共享该宽类映射。

主协议将phone-class aperture separation作为主要视觉端点，将全轨迹extrema timing列为探索性。旧实现把event控制资格绑定到phone primary，产生错误耦合。新的预效果版本分开资格：phone endpoint使用结构、分类、共同支持与CAL冻结latency/count；event_metric_calibrated保留原rematching和shuffle控制失败，不作事件同步改善主张。固定同一detected indices的±5 timestamp变换只证明工程变换正确，不能替代rematching控制。额外CAL static/100 shuffle/zero-latency类分离只是诊断，不新增阳性门槛；EVAL zero-latency列为secondary，主值继承冻结latency。所有修改、原规则结果与新冻结均在正式effect首读前保存。

## 实际 forward 状态与可复用桥接

CAL后记录的cuDNN benchmark=true，与人脸检测前请求false不同；detector构造会改变该flag。旧EVAL未逐次记录forward状态，因此不能沿用配置字段将其称为false。选择保留原protocol/axes/产物封印，新增不可变execution correction和固定前两CAL样本true/false数值敏感性验证。

新sham和WORLD执行在人脸检测后重置benchmark=false与deterministic，并逐forward记录真实状态。sham q00重新生成，与q10/q01/q11使用相同runtime，避免把原true缓存与新false干预相减。WORLD RAW/FIXED bridge只在PCM/模型/ROI/前端/执行状态/特征相容时复用；不相容时在新run重生成。无论CAL数值差异是否很小，都保留这项运行来源说明。

## Observations

- [decision] 先做RAW/FIXED的生成×评分四格，再将cal-only TTS−N表示方向移植到FIXED自然生成输入，所有方向条件均以原RAW自然PCM评分；99%联合门槛在新效果前固定。
- [rationale] 单次原生T−N差不能辨认生成器与评分器路径；相同时钟和PCM上的直接干预提供可解释的路径响应。
- [alternative] 仅恢复旧统计可补历史，但不作为本次确认的唯一证据。
- [alternative] 直接以TTS视频替换自然视频同时改变时序，不能取代来源表示干预。
- [alternative] 仅使用自动找峰Sync-C易把偏移校正收益与固定时序混淆。
- [consequence] 100条自然输入按manifest冻结；TTS失败、随机剂量不匹配和视觉缺失保留在分母中，不静默删除。
- [consequence] 每帧随机方向的实际投影位移匹配orth_plus；搜索[0,64]并记录不可达，保证对照剂量明确。
- [consequence] SyncNet、独立嘴部运动和主观感知分别形成结论；抽象表示结果后继续物理声学机制。

## Relations

- affects [[TTS 来源方向与声学机制确认]]
- affects [[解释 TTS 的 TFG 与 SyncNet 优势]]
## 短语音端点与运行续接

2026-10-01正式retry2在一条TTS的共同SyncNet支持=40时停止；guard20按冻结定义剩0窗口，但原自然评分的来源方向仍有足够支持。效果汇总尚未读取。选择保留原音频、行计划、guard/anchor及失败run；新的显式continuation只修复整clip中断，支持>0的矩阵/full/valid/guard15照常存，guard20=too_short明确不可测。

新run沿用原protocol/rows字节，附parent与reused artifact哈希，执行代码另行封存SHA/attempt，不能让旧production hash冒充新执行代码。全部条件/双脸产物封存后才汇总。主要来源门槛按其必要行及40speaker×2支持判断；T四格或native gap的缺失单独标incomplete，不用它制造主要因果阴性。保留整个意向分母、失败和可测子集估计。未采用改guard或重抽TTS来修补主端点。

## 独立消费的剂量依赖和搜索诊断

原08消费者把search_gain写为C，次要诊断必须更正为D0−D（zero-lag列15减31列最小距离），而C仍为B−D。保留08不变，新增独立CPU supplement重算此项及full/valid/guard15预设敏感性；不变更C/anchor主门、不重跑GPU或按效应挑支持。随机剂量仅按每arm实际存储F32公差判定，原nonrandom主门完整80保留；new equal-dose辅助在已封存同vectors/seed/dose下fresh重跑，不放宽1e-6。
## 观测与置换的音频边界保持一致

原生cached-null实现审阅发现观测沿旧共同A cutoff=query_n重建、null却以真实full A长度作边界，差异会混入observed-minus-null。正式null结果读取前，选择同时保留两个完整、预先声明的口径：legacy口径观测/null均用query_n有效A边界；full-A敏感性观测/null均用真实len(A)边界。两种口径都均匀置换完整A行集合、保持共同V查询前缀、相同四支持域、4cell/双脸/N-T接收者/raw-unit和固定100个π种子；不能按结果挑一种口径。旧score NPZ parity只要求legacy观测重建匹配，full-A口径为新增边界诊断。新annex和consumer另存源码SHA，原spec与所有failed attempts不改。

- [decision] 每个cached pairing诊断的观测与null必须共享相同音频边界；原生分析全报legacy与full-A预先声明敏感性。
- [rationale] 仅置换完整A多集合并不能保证零填充域相同，混用边界会把padding差异当成时间配对差异。
- [alternative] 只修改null边界而省略full-A敏感性，会缺少对完整音频尾域的明确诊断。
- [alternative] 只改observed为full-A并称为原score复现，会破坏与已封存历史评分口径的对应。
- [consequence] 两套观测/null、相同意向cohort/缺失和统计全部报告，原主端点与stored parity继续可追溯。