---
title: 静态图片 Natural-to-TTS bridge 实验
type: experiment
permalink: tts-exp/experiments/静态图片-natural-to-tts-bridge-实验
status: concluded
date: '2026-09-13'
hypothesis: 同一静态参考图下，冻结 BRIDGE_075 是否比 natural 与纯重建驱动产生更匹配原自然音频的视频？
report: openspec/changes/probe-static-image-natural-to-tts-bridge/specs/static-image-natural-to-tts-bridge/spec.md
tags:
- bridge
- static-image
- wav2lip
- replacement
- openspec
audit_status: requires_correction
audit_report: runs/static_image_bridge_20260913/audit_20260913/review.md
frontal_extension_status: planned
---

# 静态图片 Natural-to-TTS bridge 实验

## Context

用户要求先保存 bridge 历史结果，再设计并执行以静态图片输入的 bridge 实验。本实体同时记录 spec、实现和最终结果；规范在 OpenSpec，后续复核沿用同一实体更新，不另建重复笔记。

历史动态视频 bridge 探索轮 MAG_075 的 C 提升为 18/23（78.3%），平均 ΔC=+0.078；确认轮 BRIDGE_075 为 9/22（40.9%），平均 +0.031、95% CI 跨零。候选 mel movement 22/22 通过不能理解为 100% 同步提升。全项目的视觉输入并不相同：早期 Ditto 为肖像 PNG；本 bridge 为 LRS3 动态视频；后续已有静态 shift/参考诊断，但未替代本次冻结 B 波形的实验问题。

## Design

复用确认轮全部 22 条 seen-fit 记录和旧 N/B/S decoded PCM，alpha=0.75。每条只使用源视频第 0 帧 PNG，固定人脸框与评分 crop。五臂 N/N_REPEAT/RT/B/S；RT 为 alpha=0 的同契约 STFT 重建，避免把纯重建变化归为 TTS 增量。

Stage A 44 视频/66 cells，以独立 repeat 和同视频配 +200ms 音频检验测量。通过后 Stage B 66 视频/132 cells；合计最多 110 新视频/198 cells，最多 4 次有记录的失败工程重试。B 主比较一律用原自然音轨，并且同时比较 N 与 RT；B 自身音轨只作辅助。

统一固定评分几何、共同真实时间支持、官方 SyncNet frontend 和完整距离矩阵，保存 C/D/offset 及固定自然 lag 的 D_anchor。组 bootstrap 10000 次、seed=20260913；C 增益要求相对 N、RT 均 mean>0.05 且 95% CI 下界>0，并通过 D/D_anchor 与 offset 安全门槛。报告逐样本 C/D/联合改善比例。

LOCAL_SWAP 分别检查两个中间段内 N 画面偏 N、S 画面偏 S；不要求交换音频的生成质量与正常语音等效。bridge_gain 与 swap_transfer 独立输出，不能因后者未通过就宣称前者假设无效。

## Results

本次运行完成了 110 个视频和 198 个矩阵，但 September 13 用户要求复核后确认：**完整 spec 验收尚不成立**。原 validation 文件的 PASS 保留为历史运行输出，不再当作真正独立验收证据。实现未修复，原视频/矩阵/analysis/final 未覆盖。诊断报告：`runs/static_image_bridge_20260913/audit_20260913/review.md`。

- 原始结果：全部 22 条 / 22 groups，measurement=`PASS`；N_REPEAT 的 C/D/lag 与 N 完全一致。ND +200ms 在 22/22 条识别 +5 frames；原 D_anchor damage mean=6.449，CI [5.786,7.110]。
- 静态 bridge 原矩阵独立复算：B−N 与 B−RT 的 ΔC=−0.777，CI [−0.968,−0.597]；D 改善量（baseline D−B D）=−0.793，CI [−1.006,−0.601]；22/22 条 C/D 均下降。独立计算与原 ΔC 逐条差为 0，保留 `NO_STATIC_BRIDGE_GAIN_ESTABLISHED` 作为观测方向。
- 事后边界敏感性：主 W 两端各再剔除 5/10/15 个窗口，ΔC 分别 −0.766/−0.765/−0.779，均 0/22 正向、CI 上界仍小于零。已发现的边界问题不足以解释掉负增益。
- 对照纠偏诊断：按 MFCC/preemphasis/生成 mel-STFT 完整感受野排除 ND 填零和局部交换接缝；ND 每条去掉 6–7 个 A 窗口后仍 22/22 lag +5，anchor damage=6.405，CI [5.754,7.067]。LOCAL_SWAP 每段进一步去掉 2–5 窗后，四项均值 6.921/6.980/6.933/6.957，CI 下界全部 >6.10，22/22 四项正向；`SWAP_TRANSFER_OBSERVED` 的方向保留。此为事后诊断，不称修复后的预注册验收。
- 输入/绑定核对：22/22 N/B/S decoded PCM 等于父资产；S=A-C-B-D、ND、RT identity 正确；22 张 PNG 等于源第0帧；198 cells 的实际音频、视频、矩阵 hash/配对通过。前三条 N/B/S 共2334帧，生成 ROI 外逐像素等于静态图。
- 官方前向抽查：首条6个cell，独立官方 SyncNet forward + calc_pdist 对同crop/PCM/support的最大矩阵误差0.000006676，低于0.00001。不是全部198 cells的新前向。
- RT alpha=0 为22/22 decoded PCM identity，因此 B−RT 与 B−N 相同；不把 RT 当独立声学扰动。
- 辅助结果：同一 B 视频配 B 比配 N 的 C 平均高0.463，18/22正向；评分音轨改变，只能作描述性线索。
- 保留边界：training_authorized/generalization_established/mouth_leakage_proven/historical_gate_repaired 均false；人工审阅仍 NOT_HUMAN_REVIEWED。
- 原 focused tests 14 passed、Ruff/py_compile/OpenSpec strict通过，只代表原测试/文档检查通过，未覆盖此次发现的规范缺口。

已确认缺陷：
1. `validate.py` 调用与 runner 相同的分析函数并写回待验收结果，`independent=true` 不成立。
2. W 在读矩阵后按finite mask决定，未在评分前冻结真实frontend支持；ND补零窗口混入。
3. LOCAL_SWAP边界只按约5帧检查，未涵盖完整MFCC和生成mel/STFT感受野。
4. 两段辅助D0实际复制全局值，22/22两段相同；缺局部逐lag曲线与支持。
5. protocol缺完整代码/模型/spec hash冻结；Markdown播放路径缺playback/，逐条C显示n/a，缺完整联合改善比例等交付。

新旧差异同时包含静态输入、生成ROI、评分crop、编码及支持变化。旧首条生成使用整帧[0,224,0,224]，新用检测脸框xyxy[54,1,173,169]；不能唯一归因于静态图片或mouth leakage。下一步应先修复验收与交付，再用同处理链STATIC/DYNAMIC对照定位机制。

## 正脸参考替换扩展（planned）

September 13, 2026 用户要求用本地高质量静态人脸试跑。预先选择 data/data/image/{3,6,9}.png，均512×512、近正脸、嘴部无遮挡，图像选择不使用新分数。固定原22条N/B音频全交叉，共66对；首条音频在每张图上另做N_REPEAT、ND及S控制。沿用相同生成与评分裁剪规则，按完整frontend真实支持在评分前冻结窗口，用独立的小型分析脚本计算配对结果；按22个音频source_group统计，不将66组图像×音频当66个独立样本。RT旧PCM逐条等于N，校验后不重复渲染。此为探索性参考图敏感性扩展，不是旧spec完整修复或动态泄漏证明；不覆盖父运行。

## Observations

- [status] concluded
- [audit] requires_correction：110视频/198矩阵已完成，但完整spec验收未成立；原validation PASS并非独立分析实现。
- [result] 独立矩阵复算保留B−N/B−RT ΔC=−0.777，22/22 C/D下降；剔除额外边界后仍全部下降。
- [result] 排除完整frontend接缝支持后，ND仍22/22识别+5帧，LOCAL_SWAP仍22/22四项配对偏好为正；这些是事后敏感性诊断。
- [problem] W未预先冻结、ND补零窗口混入、LOCAL_SWAP感受野边界不完整、分段辅助D0误用全局值；需修复验证器和交付。
- [verification] 22条PCM/第0帧与198cell绑定核验通过；首条6cell官方SyncNet前向最大矩阵误差0.000006676。
- [insight] RT alpha=0在22条上为PCM identity，B−RT与B−N相同；不是独立声学扰动。
- [boundary] 静态与旧动态还同时改变生成ROI/评分crop/编码/支持，不能将差值当作静态输入因果效应或嘴型泄漏证明。
- [hypothesis] 新静态和人脸ROI条件可能更充分传递音频驱动，B改变谱幅度后使生成结果更难匹配N；与控制结果相容，尚未唯一定位机制。
- [limitation] 原测试通过未覆盖上述缺口；人类播放审阅未做；小规模前向抽查不代表全部媒体重新前向。
- [report] 复核证据与可重放诊断：runs/static_image_bridge_20260913/audit_20260913/review.md。
- [decision] 不授权训练/泛化/mouth leakage结论；保留观测方向，撤回完整独立验收保证。原实现和科学产物未覆盖，修复工作尚待执行。

## Relations

- follows [[LRS3 natural-to-TTS bridge confirmation result]]
- follows [[LOCAL_SWAP 最小重放诊断 2026-09-13]]
- relates_to [[Wav2Lip global shift response diagnostic 2026-09-08]]
- relates_to [[Wav2Lip reference conditioning interaction 2026-09-09]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求交付静态图片 bridge 实验 OpenSpec，并登记 planned；未运行实验 | September 13, 2026 | user |
| 实现并运行静态图片 bridge spec：A 门控 PASS，完成 110 视频/198 cells；静态 bridge gain 未建立，LOCAL_SWAP 分段 transfer 观察到；独立 validation、focused tests、Ruff、py_compile、OpenSpec strict validation 全通过 | September 13, 2026 | user |
| 按用户要求复核发现共享分析验证器、真实支持冻结/边界与局部D0及交付缺陷，撤回完整独立验收保证；独立矩阵复算、边界敏感性和首条官方前向抽查仍支持下降观测；登记requires_correction，原实现/科学产物未改写 | September 13, 2026 | user |
