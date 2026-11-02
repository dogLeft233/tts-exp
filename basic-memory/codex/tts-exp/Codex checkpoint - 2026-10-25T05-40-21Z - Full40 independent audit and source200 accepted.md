---
title: Codex checkpoint - 2026-10-25T05-40-21Z - Full40 independent audit and source200
  accepted
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t05-40-21-z-full40-independent-audit-and-source200-accepted
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T05:40:21Z'
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

# Codex checkpoint - 2026-10-25T05-40-21Z - Full40 independent audit and source200 accepted

## Summary
完整40人280媒体独立技术复核与200人600自然源双端验收已完成，严格七臂同文字支持修正为35/40；用户现要求少量数据验证下游解释流程，训练尚未授权。

## Story
研究希望解释自然、0.6B和1.7B在归一化后的SyncNet偏好，而非用Sync-C代替人类口型质量。用户要求完整40人独立复核并检查200人进度，不训练。父侧独立CPU实现核对官方曲线、512投影、MFCC、处理延迟与时间索引，另以标准库在Windows和Ubuntu逐个重读600自然源。原后台补数均已完成，先前checkpoint的运行中PID不是当前状态。

权威结论见 [[完整40人SyncNet特征独立复核与200人自然语料验收 2026-10-25]] 及 runs/syncnet_audio_dataset40_parent_audit_20261025_attempt01/report.md。技术通过不代表FA真值、可听概念、模型解释或纯评价器因果已验证。最新用户“能先用少量数据验证下流程吗？”；先采用旧40按来源ID前4人，7臂各保留，用现有缓存验证实际配对/保序窗口/批读取/前向，不训练。可选小规模训练问题已提出，尚未收到答复，时间不能替代授权。

## Working State
父audit run final_manifest SHA f61866c269f87c5ab8d6c0453c001d6d4b0ec278171d4aff23b3991bc5cfe0f9；10项内容在双端完整SHA通过。完整feature仍远端 /home/wjj/tts-audio/tts-exp，不能称本机完整镜像。原child runs/syncnet_audio_dataset40_20261024_attempt03 冻结finalstatus的PENDING不回写，新父run是后来审核层。

40人280条、29485窗，2540路径SHA、全31lag缓存exact、独立NumPy最大误差5.722e-6在冻结容差、512投影8.345e-7、实际PCM到MFCC exact、FFT延迟exact；4172字符12516变体窗口索引通过。37/40是每条各自覆盖门，严格七臂同字符交集仅35/40；600固定pairs其中360主比较352支持，240同模型draw；不能误扩成全部crossdraw。低支持S0006/S0019/S0032/S0034/S0041保留标记，整段评分均保留。

runs/audio_data_inventory200_20261024_attempt05 的600源已完成，200固定speaker每人3条，100fit/40dev/60confirm，旧40全fit；源manifest SHA3ef28c8076aba23e64e2bc1a22963e24dba1e9a89ed9e2aeaf7fdaae6821e608，split SHA3ffb6d516470d5185a71e18d5bd8390b3180de7d6178640a2726c5eaef6fc749。双端600原字节/PCM/资格/名单/分组PASS，179700文本pair无预定近重复、PCM无重复。119210100B/3724.490625秒。原空WAV失败现场保留，attempt05按原资格排除10空条后完成，未换speaker或已选186源。

存储观察约16.56GB free，4200全活动预算约57.21GB；完整4200生成仍HOLD_STORAGE_AND_ARCHIVE_POLICY。小云盘TAR跨host还原通过不等于provider durable flush，不能删资料。当前无授权训练。源级确认数据技术QA不等于揭封模型评估。Git大量继承dirty/untracked未清理；本轮脚本未提交，SHA仅本机，GitHub 422不可链接。当前head PR查询为空。

## Changed Files
- scripts/experiments/syncnet_dataset40_independent_audit_20261025.py；tests/test_syncnet_dataset40_independent_audit_20261025.py。
- scripts/experiments/audio200_parent_readback_20261025.py；tests/test_audio200_parent_readback_20261025.py。
- 父audit run中的pair_support_audit.py、pair_set_and_common_support.json、双端600readback、报告及封存清单。
- BM新验收note、研究主线与证据地图及旧技术note的后续指针已读回。

## Verification
实际远端unittest2项PASS（边缘padding/epsilon、RF中心/40ms漂移/越界），本机unittest2项PASS（manifest/roster漂移前置拒绝）。完整40审计13.504秒；双端600读回约16秒。pair检查最初默认GBK及错误全crossdraw模板失败，未写成功凭据，读取原冻结protocol后修正父侧模板通过，未修改原实验。新下游小试尚未运行；勿当作已训练或科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，GitHub read-only metadata确认dogLeft233/tts-exp。
- Local-only current SHA `2b71639c3b555293375d248350999377168d515d`；远端不可解析，不构造commit链接。

## Observations
- [result] 40人280媒体官方特征/曲线/窗口技术复核通过，600自然源双端验收通过。
- [decision] 严格共同支持35/40与各条自身37/40分开；全体整段标签保留，不按分数选样。
- [decision] 训练保持关闭，小试优先复用旧fit来源及音频512×11原生有序窗口，视频不进入发现器。
- [blocker] 4200全量活动盘及持久归档条件仍不足；FA真值和概念解释未验证。
- [next_step] 按最新请求完成前4来源的配对保序输入与无训练前向小试，记录缺失和资源，并等待用户对可选试训练的明确答复。

## Relations
- continues [[Codex checkpoint - 2026-10-24T09-50-00Z - SyncNet replay passed and bounded data supplementation running]]
- relates_to [[完整40人SyncNet特征独立复核与200人自然语料验收 2026-10-25]]
- relates_to [[WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24]]
