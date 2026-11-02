---
title: Codex checkpoint - 2026-10-15T00-21-11Z - Bounded waveform control and grouped
  basis readback
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t00-21-11-z-bounded-waveform-control-and-grouped-basis-readback
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-15T00:21:11.928157+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-15T00-21-11Z - Bounded waveform control and grouped basis readback

## Summary

真实波形16参数Jacobian和有界LP已关闭，稳定性幅度是当前局部瓶颈；分组32参数唯一CPU计算已rc0读回，等待一次原数组QA和固定尺度比较。

## Story

用户要在相同RMS、内容和原时钟下，分别控制各TFG原生空间的分离、局部稳定、局部能量/谱形，并用独立口部运动检查SyncNet；长期目标是可用的跨模型自然音频增强头。用户强调先小有效性后规模、减少重复审计。ROOT维护理论/协议/BM和一次关键实际QA，原GPT-6 Luna/max代理负责代码与实验。已完成报告39–42保留其因果和视觉边界：Muse正确输出所有者坐标能双向控制口部功能，但正式原自然音轨同步收益未出现。现在转向真正可合成波形的控制坐标，而非继续任意native旋转或盲拟合p4方向。

16参数真实frontend实验仅6encoder/6VJP，原数组QA通过。saved-J box-LP在相同±0.25dB系数约束下，SEP和ENV达到两句raw目标1e-4，STAB最大归一化幅度0.090980小于所需0.273843；这是该局部线性族的幅度限制。下一小假设只松开a与i/y/u强制相反滤波，成为[2,8,2]的32参数严格超集，不增数据、剂量或模型。32唯一worker已真正完成并取回六NPZ；尚未QA其Jacobian或宣布三目标通过。

## Working State

权威入口为地图→专题[[接收端局部构音响应坐标]]→Task23→HANDOFF.md；本checkpoint晚于旧23:22快照。probe=`.codex/scratch/native_attribute_response_probe_20261015_attempt01/`。这些scratch/SHM及未提交文档是机器局部、易失产物，不是完整备份。LAN wjj@192.168.50.2:22保持运行、公钥SSH；用户已取消关机。旧训练/四中心/Dev/Confirm/归档/cloud donor/visual teacher均停放。当前没有准入TFG/GPU/新训练，不运行实现tests/pytest/compileall/install；秘密不文件/argv/BM/log。

16实际产物：`realizable_coordinate_jacobian_attempt04/readback_attempt01/`，ROOT QA b4230b2e6ae3b4f7d8a924e429c4d9b3b7cb9fb537ad151fa66a910fd763e5f9；box-LP结果2a6aaaa7e66ec27564d728d2afa6e6556d4e57bbee1cc92f759cc44a7cc2d3cb，ROOT原/对偶QA363961f65d31917bcc2ac24b2c62ad0c57fde7cb7b5f15dface73ddef5968d41。自然9FIT校准JSON ee45a753…/NPZ f289185b…和原CPU6/parity复用，不重聚合或重encode。

32实际产物：`realizable_grouped_waveform_jacobian_attempt02/readback_attempt01/`，local manifest SHA77beec89d147a266c14e5e1fa70f2daae2f965f724f135465451a824d140530f，16/16 selected body bytes/hash匹配。receipt a38198283745fdb99b390a42bcc161f01cc948d8364aa005a06c1b4d96455f99，COMPLETE_READOUT/6forward/6VJP/2clips/noerror；worker2590233/50800792、supervisor2590232/50800788均退出，terminal rc0 UTC2026-10-15T00:13:38.538108Z。原output `/dev/shm/cross_tfg_realizable_grouped_coordinate_jacobian_20261015_attempt02`，成功control加`_control_retry01`；不得重launch。

32scientific source/input保持：module dd06576a…/contract5031f315…，runner02 80bf0506…/input02 83be7c46…，admission0dbc8b3e9ea13f364b12c9dcb7c58fbcb4dbbe57695d8a9d8319122e16c177a9。runner01错contract接口在dispatch前retire；首dispatcher长度门rc2/0spawn/0modelcall，旧partial control保留。ROOT授权只修UTF8 byte传输，retry实际config1421B/b0e3fcac…远端长度/hash通过；不是第二次科学worker。

## Changed Files

地图、专题、Task、HANDOFF和CONTEXT维护16结果与32计划/准入；32当前实际终局待写入。ROOT新增冻结条件有限剂量协议`realizable_grouped_waveform_coordinate_attempt01/conditional_finite_three_scalar_control_protocol_attempt01.json` SHA400a4ce143f423f3837f70727c467977114ec9a0fb3a6fa99b249ffbb9926271，仍NOT_EXECUTION_APPROVAL。若32全部通过，未来仅6新encoder/0VJP/0baseline/0TFG，三个LP轴各两句一次，原RMS/clock/剂量与target≥1e-4、预测误差≤20%、offtarget ratio≤10%，另featureextractor-only parity和native collateral记录；三scalar近似隔离不等于完整Gram/noncore纯度。

原Angular刚准备LP source04 `realizable_coordinate_jacobian_attempt04/box_lp_attempt01/box_lp_attempt04_fixed_reference.py` SHA3da1d19e12c82816cd405fb699eeff2e7c02ed4b355cce1351809ae9e28195d4，ROOT已读81行，未执行；主self-normalized32分析保留，secondary固定旧16 row-norm。它绑定旧actual_phi_j_attempt04.npz 5eba675e…与旧LP原件；0新modelcalls。

Git branch/HEAD见metadata；git status9tracked dirty及1003untracked entries，绝大多数继承既有工作，当前新源码/产物在scratch。未commit/PR，不整理或删除他人变更。

## Verification

16 ROOT一次actual PCM/native/clock/phi/J/FD检查：theta0逐值同原基线，phi误差≤9.99e-16；FD最大relative0.001235，RMSrelative≤1.05e-7、PCMrelativeL2≤0.001201。shared normalizedJ rank6/minsv0.130988。LP原/对偶/KKT误差≤1.12e-16。32仅完成真实终局和body readback；尚未独立QA六NPZ，不复用16QA冒充32通过。此前元数据/clock/reader换行/dispatcher失败均保留为工程失败。无实现测试。

## References

- [Repository](https://github.com/dogLeft233/tts-exp) 已read-only API确认200；精确head的PR查询为空。
- HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907` 为local/unpushed，GitHub commit API422，不构造commit链接。
- `HANDOFF.md`、`basic-memory/Tasks/native-axis-factor-pilot-20261013.md` 和上述probe原数组/receipt是恢复依据。

## Observations

- [result] 16basis中SEP的min-L2不足来自选解保守；STAB即box最优也未达原局部共同目标幅度。梯度/线性可达不是有限剂量或head效果。
- [decision] 新32family的Jgroup0−Jgroup1须从实际数组复现旧16J，maxabs≤2e-8/relativeL2≤1e-4，无额外encoder。
- [decision] 不直接比较各自row-normalized16/32的alpha；尺度和跨句raw比会改变。另用J32/旧16 row-norm固定尺度3LP验证新增自由度，保留primary32门不替换。
- [blocker] 32原数组及三目标有界可行性尚未独立核验，有限剂量阶段尚无执行准入；这是下一计算阶段，不是外部阻塞或方法失败。
- [next_step] ROOT一次实际32 phi/J/FD/旧J bridge QA，原Angular执行saved-J primary与固定旧尺度共6LP，合格后才激活6forward有限剂量小验证。

## Relations

- continues [[Codex checkpoint - 2026-10-14T23-22-47Z - Realizable waveform coordinate phase one prepared]]
- relates_to [[native-axis-factor-pilot-20261013]]
- relates_to [[接收端局部构音响应坐标]]
- relates_to [[研究主线与证据地图]]
- relates_to [[开发跨 TFG 的自然音频增强头]]