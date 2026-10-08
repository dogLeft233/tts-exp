---
title: MFA-linear 机制小样本诊断 2026-09-13
type: experiment
permalink: tts-exp/experiments/mfa-linear-机制小样本诊断-2026-09-13
status: concluded
sample_count: 3
report: runs/mfa_linear_mechanism_pilot_20260913/report.md
tags:
- mfa-linear
- wav2lip
- syncnet
- replacement
- diagnostic
---

# MFA-linear 机制小样本诊断 2026-09-13

固定自然时钟筛查的前3条LRS3记录与既有3号静态正脸图，不按分数选择。此诊断补上上一轮未测的历史MFA-linear正对照，并分别检查一次输出重对齐后的局部时间修复，以及目标Wav2Lip mel直接输入与Griffin-Lim波形重建后再提mel之间的差异。

## 方法与结果
M为历史MFA-linear候选波形。M_REPAIR先对M以相同转写重新MFA；原定phone级映射因MFA异音标签不一致，三条均按预先固定规则回退到输出MFA word clock，生成一次确定性的分段线性波形坐标场，不做评分搜索。M_DIRECT_FIXED把M的Wav2Lip 80维mel直接输入同一冻结Wav2Lip，并与波形臂共用同一冻结人脸框。M_GL把同一mel经64轮Griffin-Lim逆变换、再由Wav2Lip重新提取mel。所有静态生成视频在共同主支持、lag±15上评分。

|比较|平均 ΔSync-C|正向样本|
|---|---:|---:|
|M原生 C(V_M,A_M) − C(V_N,A_N)|+0.106|2/3|
|M replacement C(V_M,A_N) − C(V_N,A_N)|−0.907|0/3|
|M_REPAIR原生 − M原生|−0.165|1/3|
|M_DIRECT_FIXED配M − 波形M配M|−0.036|0/3|
|M_DIRECT_FIXED配M − M_GL原生|+0.913|3/3|

## 结论与边界

历史MFA-linear在这个很小的静态条件下作为原生正对照成立（平均+0.106），但换回自然音频仍全部下降（平均−0.907）。所以“上一轮三个自然时钟方案无增益”不等于“静态Wav2Lip中完全不存在可驱动的MFA-linear效应”；更直接的解释是它们没有复现MFA-linear所携带的生成器输入结构。

一次以输出MFA word clock做的局部修复没有保住原生增益，且phone级对齐因标签不一致没有真正落实，因此不能将其解释为局部时间修复路线被否定。与波形M相比，固定框的direct-mel结果为−0.036（3/3不增），说明M进入Wav2Lip的常规波形→mel路径本身没有显著可回收的损失。相对Griffin-Lim重建后再提mel稳定+0.913，证明明显劣化的重建链确实会丢失对生成器有用的结构；它不证明现有MFA-linear vocoder链正是主要瓶颈。direct-mel是机制探针而非可播放音频方案。

n=3、单图、历史MFA-linear seen-fit、无主观评估或输出ASR/phone误差；不授权训练、泛化或声称真实自然视频兼容性改善。

## Observations

- [status] concluded
- [result] 历史MFA-linear原生平均ΔSync-C=+0.106（2/3正）；换回自然音频平均−0.907（0/3正）。
- [result] 固定框direct target mel相对波形M为−0.036（0/3正），相对Griffin-Lim重建再提mel为+0.913（3/3正）。
- [conclusion] 生成器可利用的MFA-linear输入结构存在，但当前natural-clock音频方案没有复现；强重建链可损失结构，而常规M波形→mel路径在此小样本没有显示可回收损失。
- [boundary] M_REPAIR三条均从phone映射回退至word映射，负结果不能否定phone级输出对齐修复。
- [report] runs/mfa_linear_mechanism_pilot_20260913/report.md；每条20cell矩阵在analysis.json，音频/视频在同一run。
- [validation] 固定框direct-mel补评分后共15个视频、3×20个矩阵单元；10项相关测试和Ruff通过，未复用历史分数。
## 结论与边界

历史MFA-linear在这个很小的静态条件下作为原生正对照成立（平均+0.106），但换回自然音频仍全部下降（平均−0.907）。所以“上一轮三个自然时钟方案无增益”不等于“静态Wav2Lip中完全不存在可驱动的MFA-linear效应”；更直接的解释是它们没有复现MFA-linear所携带的生成器输入结构。

一次以输出MFA word clock做的局部修复没有保住原生增益，且phone级对齐因标签不一致没有真正落实，因此不能将其解释为局部时间修复路线被否定。direct mel相对波形M仅+0.037，说明这条M音频在Wav2Lip重新提mel时的损失很小；但相对Griffin-Lim重建后再提mel稳定+0.986，证明“目标mel→波形→再提mel”可以显著损失对生成器有用的结构。direct-mel是机制探针而非可播放音频方案。

n=3、单图、历史MFA-linear seen-fit、无主观评估或输出ASR/phone误差；不授权训练、泛化或声称真实自然视频兼容性改善。

## Observations

- [status] concluded
- [result] 历史MFA-linear原生平均ΔSync-C=+0.106（2/3正）；换回自然音频平均−0.907（0/3正）。
- [result] direct target mel相对Griffin-Lim重建再提mel平均+0.986（3/3正）；相对波形M仅+0.037。
- [conclusion] 生成器可利用的MFA-linear输入结构存在，但当前natural-clock音频方案没有复现；波形重建链可成为明显信息损失点。
- [boundary] M_REPAIR三条均从phone映射回退至word映射，负结果不能否定phone级输出对齐修复。
- [report] runs/mfa_linear_mechanism_pilot_20260913/report.md；完整20cell/record矩阵在analysis.json，音频/视频在同一run。
- [validation] 10项相关测试和Ruff通过；音频、视频、矩阵全部由本run新生成，未复用历史分数。

## Relations

- follows [[自然时钟音频双目标小样本筛查 2026-09-13]]
- relates_to [[优质静态图低强度 Bridge replacement 2026-09-13]]
- relates_to [[31 - Replacement Salvage: Phone-Aligned Cross-Attention]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求完成少量样本三项机制实验 | September 13, 2026 | user |