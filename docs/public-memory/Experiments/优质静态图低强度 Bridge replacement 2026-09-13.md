---
title: 优质静态图低强度 Bridge replacement 2026-09-13
type: experiment
permalink: tts-exp/experiments/优质静态图低强度-bridge-replacement-2026-09-13
status: concluded
report: runs/static_image_bridge_lowalpha_20260913_v2/report.md
sample_count: 11
---

# 优质静态图低强度 Bridge replacement 2026-09-13

用户要求检验bridge 0.25和0.5是否在本地优质静态图上产生replacement增强，并在运行中要求只用前11条，同时比较不更换音轨的原生分数。

## 设计与样本

原计划固定data/data/image/{3,6,9}.png三张512×512近正脸图×22条音频；运行至11条时用户明确收束，最终每强度33对、11个source_group。之前已披露前6条部分分数，因此属探索性、运行中缩减，不称固定N确认性试验。第12条自然渲染与停队列存在竞态，保留一个未评分视频但排除。未按分数挑选个别图像或音频。

复用自然视频强度扫描已生成并核验的B025/B050波形。固定图像生成ROI与评分crop，评分前冻结完整真实frontend支持。主比较C(V_B,A_N)-C(V_N,A_N)，新增明确的原生比较C(V_B,A_B)-C(V_N,A_N)。三张图差值先按音频平均，对11组bootstrap10000次，seed20260913；CI未作多重比较或停止规则校正。RT=自然PCM identity，不单独渲染。

## 结果

自然基线V_N/A_N：C=5.595，D=7.170。

|比较|C|ΔC及95% CI|C改善对数|D|
|---|---:|---|---:|---:|
|B025原生，不换音轨|5.518|-0.078 [-0.145,-0.026]|8/33|7.266|
|B050原生，不换音轨|5.416|-0.179 [-0.290,-0.071]|7/33|7.255|
|B025换回自然音轨|5.488|-0.107 [-0.167,-0.042]|6/33|7.279|
|B050换回自然音轨|5.249|-0.347 [-0.446,-0.244]|1/33|7.532|

两强度三张图的平均ΔC均负，原生与replacement都未建立增强。原生比replacement分数更高，但仍低于自然基线，故不是仅更换音轨才出现下降。B025/B050 replacement C/D联合改善5/33、0/33；原生4/33、7/33。offset改变分别1/33、3/33。不能推出普遍无效或历史动态实验有bug。

## 核验

105个纳入视频、180个评分矩阵（含控制）；首条音频×三图repeat矩阵误差0，200ms延迟均lag+5，固定lag损伤5.690/6.459/4.877；LOCAL_SWAP两段双向偏好均通过。控制不是11条全量。

输入/音视频/模型/矩阵hash和配对绑定、无损转码逐帧、生成ROI外静态校验通过。独立重推支持并重算66个强度配对的原生/replacement差值，最大误差0。首条×三图15个cell官方前向最终误差0，非180个cell全部重跑。28项相关测试及Ruff通过。

独立复核最初出现图6/B050配N最大距离误差0.000011444，高于0.000010阈值；最小复现显示不同张量内存布局导致微小前向浮点差异。仅匹配worker的张量布局，不改输入/阈值/原分数后15个cell全为0。保留forward_failure_legacy.json及--layout legacy --forward-only --image-id 6复现命令。GPU任务已停止，无其他计算任务被终止。

## Observations

- [status] concluded
- [result] B025/B050原生ΔC=-0.078/-0.179；replacementΔC=-0.107/-0.347，均未建立增益。
- [conclusion] 自然视频上较少掉分不保证静态生成replacement增强；此次连不替换音轨的原生组合也低于自然基线。
- [verification] 独立66配对重算与15cell官方前向误差0；28测试通过；原分数未修改。
- [report] runs/static_image_bridge_lowalpha_20260913_v2/report.md，含原生视频与replacement三联屏链接。
- [boundary] 原计划22→用户要求前11；探索性seen-fit三身份，未校正多重比较/运行中停止；不授权训练/泛化。
- [boundary] alpha0.25在自然视频上基本未掉分；alpha0.5此前已有平均ΔC=-0.347，不能视作时间等效。

## Relations

- follows [[Bridge 自然视频 SyncNet 强度扫描 2026-09-13]]
- extends [[静态图片 Natural-to-TTS bridge 实验]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求冻结低强度静态图实验设计 | September 13, 2026 | user |
| 按用户要求缩减前11条并加入原生对比；完成两强度×三图，均未建立增强，保存视频与普通Markdown报告 | September 13, 2026 | user |
| 独立复核定位张量布局浮点差异，不改原分数/阈值，15cell前向与66配对重算误差0；更新concluded | September 13, 2026 | user |
