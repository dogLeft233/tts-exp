---
title: TTS 音高起伏交换的 WORLD 与测量链修复 Spec
type: research
permalink: tts-exp/research/tts-音高起伏交换的-world-与测量链修复-spec
status: concluded
execution_status: complete
protocol: tts_f0_swap_repair_v2
date: '2026-09-17'
question: 修正F0干预与测量链后，固定pilot能否通过原数值QC，并区分检测分歧与重合成变化？
tags:
- tts
- f0
- wav2lip
- spec
- repair
result: tts-exp/experiments/tts-音高起伏交换-world-与测量链修复结果-2026-09-18
completed_at: '2026-09-18'
---

# TTS 音高起伏交换的 WORLD 与测量链修复 Spec

## 0. 下游任务与完成标准

实现并执行一次仅含音频的修复实验，回答两个问题：

1. 音高曲线是否按约定被修改，保存出来的 WAV 是否通过操作有效性检查？
2. 上轮有声区检查失败，更符合“两个音高检测器本来就不同意”，还是“重合成后新增了不一致”？允许两者并存或无法区分。

完成指交付正确代码、4 对固定 pilot 的完整诊断和明确的继续/停止结论。修复后仍失败也是合格结果。不能以获得通过或得到 TTS 阳性为完成标准。

本 spec 是 [[TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec]] 的方法修订附件；冲突处以本 spec 为准。协议命名为 `tts_f0_swap_repair_v2`。本次只运行 4 对 pilot，不生成 formal 音频、不调用 Wav2Lip/SyncNet、不训练、不新增 TTS、不更换声码器。复用现有 runner 和复算器，小幅修改即可，不新建实验框架。

## 1. 已知事实与本次纠正

上一轮 `runs/tts_f0_swap_f0spec_audit2/` 中，4 对只有 1 对双向通过数值 QC，正式阶段被阻断，没有视频和 SyncNet 分数。详见 [[TTS 音高起伏双向交换 Wav2Lip Pilot 结果 2026-09-17]]。

本次 spec 编写时检查代码和已存参数发现：

- `f0_interventions` 增加了按 w² 计算的均值修正，使 `Σ(w·Δs)/Σw≈0`；原科学目标是整句全部有声帧的普通平均半音不变，即 `mean(Δs[V])=0`。两者不等价。8 个旧接收方普通平均偏移约 −0.043627～+0.054398 半音，超过约定的 1e-8；旧报告约 3e-15 只证明加权量接近零。
- 旧剂量按 w 加权 RMS 计算，而原 spec 第 5.6 节要求在 w≥0.5 帧计算目标变化 RMS；本次明确使用后者。
- 旧输出 F0/QC 取自落盘前浮点数组；本次验收必须以实际 PCM16 WAV 重读值为准。
- 原流程缺少 DIO 对 RAW 的检测，不能仅靠 DIO(ID) 与 Harvest(RAW) 不一致，就断言 WORLD 损坏了有声区。
- 时间网格不同的兜底代码用 `np.interp` 插值 F0，可能跨零值制造有声帧。本次明确网格契约，禁止这样的自动修补。
- 缺失测量不得被当成通过；完整要求见第 5 节。

这些是公式/实现与诊断范围的纠正，不构成新的 TTS 机制结论。旧 run 保持只读，旧通过数保留为历史程序结果，不追溯改写为 v2 的结果。

## 2. 固定输入与预算

父 run：`runs/tts_f0_swap_f0spec_audit2/`。读取并验证其 `inputs.json`、`pilot_qc.json`、`parameters/*.npz` 及 WAV/原始输入文件 hash。新 run 在协议中记录父文件实际 hash 和所有改动代码/spec hash。

pilot 必须恰好为以下 paired_key，不重跑候选排序、不替换失败样本：

| paired_key（均带 aishell1_test_400__ 前缀） | sample_id | speaker |
|---|---:|---|
| BAC009S0765W0312 | 26 | S0765 |
| BAC009S0770W0414 | 89 | S0770 |
| BAC009S0901W0487 | 149 | S0901 |
| BAC009S0906W0401 | 188 | S0906 |

路径、完整 paired_key、sample_id、各自 MFA phone occurrence、真实 speaker 均以父 inputs 为准，表中值不符须报 INPUT_MISMATCH，不能猜。24 对 formal 清单只复制其绑定信息供后续使用，不处理其音频。

固定 pyworld 0.3.5（环境确实无法复现则报告 BACKEND_UNAVAILABLE）；16k mono、5ms、60–600Hz；合成分析 Harvest→StoneMask、cheaptrick、d4c；独立测量 DIO→StoneMask。记录其他默认参数和依赖版本，不扫参数。

每对 N/T 各 RAW、ID、LEVEL、CONTOUR，共 32 个最终 WAV。旧资产诊断可以只读重提取；新 v2 只生成一套修正后的音频。禁止更换检测器、填补无声 F0、改 ap/sp、压缩包络、缩小交换强度、放宽门槛或增加试验组合来提高通过率。

## 3. 第一步：先复算旧产物，再生成修正版

在改动前/通过独立公式审计旧参数，保存：

- 每个接收方的旧 `mean(Δs[V])`、`Σ(wΔs)/Σw`；不能把两者使用同一字段名。
- 旧剂量以及按本 spec 定义复算的新剂量；仅对照，不给旧 run 重新标 PASS。
- 旧 32 个 WAV 的实际采样点数、峰值、RMS，与 manifest 的对应关系。
- 旧成功数及原失败理由原样引用；发现新错误另列 audit_findings。

新协议在重新合成前冻结。后续发现确定的代码 bug 可以修，但须记录 bug 证据、失效受影响缓存，并对全部 4 对统一重跑；不得按每对结果设不同参数。

## 4. 必修公式：保持整句平均音高

复用原 spec 的 phone occurrence 映射、共同有声支持、段长度及 taper w；N/T 在各自时间轴独立构建。定义：

- V：接收方原始 Harvest→StoneMask F0>0 的全部帧。
- M：w>0；S：w≥0.5。所有有效权重都必须落在合法映射且有声的帧。
- s_R=12 log2(F0_R)，s_D 为同 phone occurrence 上映射来的供给半音值。
- μ_R=Σ_M(w s_R)/Σ_M w，μ_D=Σ_M(w s_D)/Σ_M w。
- d(t)=(s_D(t)−μ_D)−(s_R(t)−μ_R)。

唯一公式：

```text
Δ_LEVEL[M]   = w[M] * (μ_D - μ_R)
Δ_CONTOUR[M] = w[M] * d[M]
Δ_LEVEL[~M] = Δ_CONTOUR[~M] = 0
s_arm[V] = s_R[V] + Δ_arm[V]
F0_arm = F0_R.copy()
F0_arm[M] = 2 ** (s_arm[M] / 12)
# M 外直接保留原数组，避免 log/exp 往返破坏逐元素相等
```

删除旧 `identity_correction=Σ(w²d)/Σ(w²)` 及其减法，不再对 taper 后的变化重新加权去均值。因为 `Σ_M(w d)=0`，以上公式已经保证 `Σ_V Δ_CONTOUR=0`。

主恒等式从实际目标 F0 数组重新算 `abs(mean(12 log2(F0_CONTOUR[V]/F0_R[V])))≤1e-8`。辅助保存总和误差及 taper 加权均值；后者允许非零，不再作为失败条件。断言无声 mask 相同、M 外目标逐元素等于原值。目标越过 [60,600]Hz 则失败，不 clip。

两种剂量统一为 `sqrt(mean(Δ_arm[S]**2))`，S 为空为不可测并失败，不能填零。CONTOUR 剂量≥0.5 半音记可辨识；低剂量如实报告，不调整强度。sp/ap 每个接收方只从其原音频提取一次，ID/LEVEL/CONTOUR 完全复用。

## 5. 必修测量链：检查最终文件与完整分母

沿用原 spec 的长度修正上限 81 samples、一次全句 RMS 匹配及四臂共同安全缩放。保存 PCM16 后重读 WAV，重新计算长度/finite/峰值/RMS/局部包络，并在重读值上提取 F0。落盘前值可以保留为诊断，不能给最终 WAV 代签通过。

每个接收方均保存以下轨迹：

- H_source：用于生成的原始 Harvest→StoneMask 轨迹。
- H_RAW、H_ID、H_LEVEL、H_CONTOUR：对最终四个 WAV 重新运行 Harvest→StoneMask。
- D_RAW、D_ID、D_LEVEL、D_CONTOUR：对最终四个 WAV 重新运行 DIO→StoneMask。

H 是同方法复查，D 是使用另一提取算法的交叉检查；两者共享 WORLD/StoneMask，不能称完全独立的真值检测器。主要操作门仍用 D，H 不能投票覆盖 D 的失败。

时间契约：

1. 每条检测结果同时保存返回 time 和 F0；预期均从0开始、5ms间隔。
2. 与 H_source 网格长度相同且逐点时间误差≤1e-8 秒才直接逐帧比较。
3. 不一致即 TIME_GRID_MISMATCH；保存两网格与差异，不截短后算通过、不插值、不搜最优时间偏移。只允许修复由错误输入长度/参数导致的明确代码问题。
4. 无声为 F0=0；非法负值、NaN/Inf、缺失臂、形状错误、提取异常都失败。
5. 任何测量缺失都令该接收方及整个 pair 不通过。诊断流程仍继续处理其他 pair 并产生完整失败报告。

## 6. 诊断矩阵：把混在一起的变化拆开看

对每个接收方列出以下固定比较，全部来自真实输出重提取，不拿合成目标冒充测量：

| 比较 A→B | 用途 |
|---|---|
| H_source→H_RAW | 检查统一增益/PCM落盘和同提取器重测差异 |
| H_RAW→D_RAW | 未重合成时两检测方法已有多少分歧 |
| D_RAW→D_ID | 同一 DIO 测量在重合成前后怎么变化 |
| H_RAW→H_ID | 同一 Harvest 测量在重合成前后怎么变化 |
| H_source→D_ID | 原主 ID 门，保持不变 |
| D_ID→D_LEVEL / D_CONTOUR | 候选相对处理对照的有声 mask 与目标变化 |

每个比较保存：总帧数、A/B有声数、交集数、lost/gained有声数；mask 不一致占全句帧比例；coverage=交集/A有声数；交集内半音绝对误差 median/p90。分母为零则不可测，不输出0误差。候选还按第7节固定 S 算变化误差和覆盖。

额外用 H_RAW>0 固定参考集合列出四格计数：
D_RAW 和 D_ID 都有声、仅 RAW 有声、仅 ID 有声、两者都无声。这样区分重合成前已未检出和重合成后新增未检出；同时报告 H_source 与 H_RAW 的差异，不能把落盘影响藏掉。

每个接收方给一张图即可：上图 H_source/H_RAW/D_RAW/H_ID/D_ID；下图 ID/目标/实测 CONTOUR 和 w，分清自然/TTS。图只用于诊断，不据图删帧。

判读使用谨慎措辞：

- RAW 上已明显分歧：支持检测方法差异参与了失败，不能说 WORLD 已无损。
- RAW 上较一致，ID 后两种方法都新增失配：与重合成引入变化相符，不能仅凭自动 F0 断言听感损伤。
- 仅一种方法在 ID 后变化：检测器与重合成可能交互，尚未分清。
- 两者并存则写 mixed；无足够证据写 unresolved。不要强行给每对分配唯一物理原因。

## 7. 数值门与唯一继续规则

保持原数值门，纠正公式、剂量定义及实际落盘测量对象。逐接收方至少满足：

| 检查 | 通过条件 |
|---|---|
| 映射支持 | Σw/全部原有声帧数≥0.40；有效 phone≥5；原有声总时长≥1秒 |
| 参数与波形 | 有限、长度精确、无 clipping；sp/ap共用；原无声mask及M外目标不改 |
| CONTOUR平均 | 第4节整句普通平均误差≤1e-8 |
| ID mask | D_ID 与 H_source 全句 mask 不一致≤0.10 |
| ID coverage | D_ID 覆盖 H_source 原有声帧≥0.80 |
| ID F0 | 二者共同有声帧半音误差 median≤1、p90≤3 |
| 候选 mask | D_LEVEL/D_CONTOUR 各相对 D_ID 全句不一致≤0.05 |
| 候选 coverage | S∩D_ID有声∩D_arm有声 的帧数 / S帧数≥0.80 |
| 候选变化误差 | 在上行交集上，abs((D_arm半音−D_ID半音)−目标Δ_arm) 的 median≤1、p90≤3 |
| RMS | 每侧最终 ID/LEVEL/CONTOUR 相对 RAW 整体 RMS 差≤0.1dB；候选相对 ID 也≤0.1dB |
| 局部包络 | 20ms窗/10ms hop，以最终 ID 帧RMS高于其最大值−40dB的帧为固定支持；候选/ID绝对dB差 median≤1、p90≤3 |

对候选测量 coverage，分母必须是整个固定 S；不能先丢失检测失败帧再声称100% coverage。局部包络复用原数值定义；无有效帧必须失败。

pair 通过 = N和T两个接收方全部必需门通过；缺一臂/异常算失败。pilot 分母固定4，至少3对通过记 `READY_FOR_FORMAL_AUDIO`，否则 `MANIPULATION_NOT_VALIDATED`。3/4通过只代表可以开展正式音频验证，不能保证24对正式样本可用。

低剂量状态单列，不用它反复改公式；正式实验仍按原 spec 的75%双向可辨识剂量规则判断。没有人工试听仍记 QUALITY_NOT_ASSESSED；可以完成数值可行性，不声称音质/声调/可懂度已验证。

本 spec 到此结束。即使通过也不自动跑 GPU。后续使用新的正式 run，继承原24对冻结名单、v2修订、原评分矩阵和统计；正式音频重新QC，不借用旧pilot状态。如果本轮仍失败，输出“当前 WORLD + DIO 协议未验证可行”；换载体或改变测量协议需要新的方法 spec，不能继续在本轮试到通过。

## 8. 最小实现与交付

优先修改现有：
- `scripts/experiments/tts_f0_swap.py`
- `scripts/experiments/tts_f0_swap_recompute.py`
- `tests/experiments/test_tts_f0_swap.py`

增加明确的 repair/diagnose 入口即可，入口名由实现确定并在报告给出可复制命令。该入口必须仅能运行父run审计、4对音频、诊断、验证、报告；不要通过旧 `--stage all` 隐式进入 formal。旧协议产物与新协议缓存不得混用。

新产物根：`runs/tts_f0_swap_repair_<run_id>/`。最低包含：
- protocol.json：父run/输入hash、固定4对、算法版本、所有阈值、公式版本与代码/spec hash。
- audio/ 与 parameters/：32臂完整性账本、最终WAV hash、目标/实际测量F0和time、原映射/w/sp/ap、增益和长度元数据。失败臂明确缺失原因。
- diagnostics.csv、audio_qc.csv：每比较原始计数/分母/数值和每接收方失败理由，不能只保存布尔PASS。
- validation.json：代码与产物一致性、独立复算结论、固定4对完整性。
- report.md 与必要曲线图：旧/新普通均值、检测分歧、重合成前后变化、8接收方QC、4对gate、未解原因。
- 保留或生成盲名试听包映射；没有听检就不填评价。

不必为每阶段建独立服务/调度器/大量状态文件。缓存至少绑定音频hash、公式版本、测量配置及相关代码；改了这些必须重新测量/合成受影响产物。

## 9. 必须能抓住错误的自审

实现后先读 diff 对照本 spec，再运行有针对性的测试和真实pilot检查：

1. 手写非均匀 taper 的解析例：w=[0,0.5,1,0.5,0]，s_R=80，s_D−s_R=[0,0,2,0,0]。μ_D−μ_R=1，期望 LEVEL增量=[0,0.5,1,0.5,0]，CONTOUR增量=[0,−0.5,1,−0.5,0]。普通平均为0，加权平均为0.25；旧 w² 修正实现必须过不了该测试。
2. 恒定音高差仅LEVEL变；相同供给两臂均回到自身；无声/M外不改；复用原 phone occurrence/无声间隙映射测试。
3. 上例剂量在S内为sqrt(0.5)，不从被测函数输出构造expected。
4. 删除一个测量臂、造NaN、空S、错一帧时间网格都应失败；不允许 np.interp 跨有声/无声兜底。
5. 重读WAV是QC输入：测试让内存数组与文件内容不同，验证QC跟随文件；落盘长度/RMS/削波不合格能被发现。
6. 人工构造检测mask，独立核对coverage的分子分母、四格计数与缺失pair仍计入4的gate。
7. 真产物独立复算全部8个接收方的普通均值/剂量、所有mask计数与误差分位数、pair gate；复算器不import生产干预/QC/gate函数，可以用相同numpy基础运算。
8. 对读盘、哈希及缓存失效作必要检查；确认父run没被覆盖，视频和分数生成次数为0。

不要只验证测试数或程序退出码。validation PASS表示实现与产物正确，manipulation FAIL可以同时成立。

## 10. BM记录与最终回复

完成后创建一篇结果笔记到 Experiments，关联本修复spec和旧结果，记录新run、8接收方/4对分母、公式纠正证据、主失败原因、是否可继续。更新本spec的执行状态与结果链接；保留旧run的历史判定，明确它的均值恒等式检查对象有误。

用普通中文回答：音高交换做对了吗？检测失败原本就存在多少、重合成后新增多少？是否达到继续条件？如仍未通过，原因仍有什么不确定？本阶段始终不能回答“F0是否解释TTS在TFG上的增益”。

## 执行结果 2026-09-18

结果笔记：[[TTS 音高起伏交换 WORLD 与测量链修复结果 2026-09-18]]

- [status] concluded；协议 `tts_f0_swap_repair_v2` 已完整执行，run 为 `runs/tts_f0_swap_repair_v2_audit5/`
- [result] 固定 4 对、8 个接收方、32 个 WAV 均完成；新普通均值恒等式 8/8 通过，独立复算最大绝对误差为 1.31×10⁻¹⁴ 半音
- [result] 仅 2/8 接收方、1/4 pair 通过；继续门槛为 3/4，因此 gate 为 `MANIPULATION_NOT_VALIDATED`
- [diagnosis] S0765、S0770、S0901 在 RAW 阶段已存在明显 H/D 检测分歧；D_RAW→D_ID 仍显示重合成后新增变化。两类因素并存，无法唯一归因
- [validation] `validation.json=PASS` 且独立复算=`PASS`；没有生成视频或 SyncNet 分数
- [artifact] 8 个已完成接收方各有一张 SVG 轨迹图，路径和 hash 已纳入 `audio_manifest.json`
- [decision] 当前 WORLD + DIO 测量协议未验证可行；不进入正式音频、Wav2Lip 或 SyncNet 阶段。本结果不能回答 F0 是否解释 TTS 的 TFG 增益

## Observations

- [status] planned
- [question] WORLD重合成与F0检测链能否在固定4对pilot上通过原数值门，并区分已有检测分歧与重合成后新增变化？
- [finding] 旧CONTOUR实现保留taper加权均值，未保留原spec规定的整句普通平均；8个旧接收方普通平均偏移约−0.043627至+0.054398半音
- [decision] 恢复原普通均值公式，QC以最终PCM16读盘为准，补RAW的Harvest/DIO对照，固定样本及门槛
- [boundary] 本轮只修复及验证音频操作，不生成视频或SyncNet分数

## Relations

- amends [[TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec]]
- follows [[TTS 音高起伏双向交换 Wav2Lip Pilot 结果 2026-09-17]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求编写修复spec；核对旧代码/参数，明确普通均值公式、最终WAV测量、RAW检测对照及固定pilot退出条件 | September 17, 2026 | user request |
| 执行 v2 修复pilot并完成独立复算；更新执行状态、结果链接及 gate 结论 | September 18, 2026 | user request |
| 自审发现轨迹图和缺失pair分母校验缺口，补实现并统一重跑 audit5 | September 18, 2026 | user request |