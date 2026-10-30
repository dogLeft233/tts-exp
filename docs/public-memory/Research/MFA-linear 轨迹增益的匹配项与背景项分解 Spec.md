---
title: MFA-linear 轨迹增益的匹配项与背景项分解 Spec
type: spec
permalink: tts-exp/research/mfa-linear-轨迹增益的匹配项与背景项分解-spec
status: implemented
date: '2026-09-16'
protocol: mfa_linear_trajectory_decomposition_v1
source_run: runs/mfa_linear_trajectory_ablation_v2_support30_20260916
tags:
- mfa-linear
- trajectory
- syncnet
- decomposition
- spec
---

# MFA-linear 轨迹增益的匹配项与背景项分解 Spec

状态：✅ implemented；已按本 spec 完成 CPU-only 再分析、测试、独立审计和结果记录。
任务：实现并运行一个 CPU-only 的既有曲线再分析，交付可复算报告。一个脚本、一个测试文件即可，不训练、不生成新音视频、不调用 GPU/模型/API，不建设新实验框架。

## 1. 科学问题与边界

问题：音素内轨迹保留比例从 0 增加到 1 时，Sync-C 的变化来自最佳距离下降，还是曲线背景中位数上升？这种依赖是否在 TTS 来源比自然来源更强？

现有假设：
- H_match：保留轨迹主要改善最佳位置的音视频特征距离。
- H_background：保留轨迹主要提高曲线背景距离，使最优位置相对更突出。
- H_specific：上述响应在 TTS 来源比自然来源更强。
- H_coupling：分数优势依赖生成视频与评分音频的共同条件，未必能转移到自然音轨。

这四个假设不是互斥的。本轮直接定位的是分数的数值组成，不独立识别“真实嘴型更准确”“SyncNet 偏差”“发音更清晰”“声码器更适配”的因果贡献。

本轮是历史 S0765 单说话人 15 条、事后 support30 运行的探索性再分析。设计前已看过 C/D 剂量趋势和总体均值，因此不得称为预注册验证、独立复现或跨说话人结论。区间只描述该说话人历史 utterance 集的重采样不确定性，不能覆盖选样与事后设计的不确定性。

完整轨迹优于均值化也可能来自声码器对均值化输入不适配；两类来源都下降不等于定位到 TTS 特有机制。先完成本分析，后续独立数据或生成实验另立协议。

## 2. 冻结输入与最小校验

仓库根：[redacted-local-path]
唯一分数来源：runs/mfa_linear_trajectory_ablation_v2_support30_20260916/。
原始 v1、support30 源目录都只读。建议输出到 runs/mfa_linear_trajectory_decomposition_20260916/；解析真实路径后拒绝输出到任一父运行内部或覆盖父运行。

编写本 spec 时核验到的文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| inputs.json | 43af6d39d9bd4b59d74b63555e900a492a8cf7c695ed461c286a42b9c902404f |
| scores.csv | 98dabb93faa2b1ee6e89eccbf3bed64ed7ab8f0e867b9f5815fd513234da9b03 |
| curves.json | 5483bf2047c76e676bb9f223bd9f06da1b114f38f3530c0d574776780bfa1c80 |
| scores_manifest.json | cb6238fc6d5701498b065e3cc15038614e12b8d6816d1c7c34bf0dc114188b5a |
| reuse_manifest.json | b13e550cc4d29be33b881118df5834582f0f81d22b226a09a0d5be4c160d83b9 |

先校验上述 5 个文件，再计算；不匹配时列出文件、期望和实际 hash，标 incomplete，退出非零；不得自动更新期望 hash 或换另一运行。分析无需再加载原始音视频和大模型。

inputs.json 必须是 protocol=mfa_linear_trajectory_ablation_v2_support30、min_common_windows=30，15 条记录、sample_id=1…15、speaker_id=S0765，paired_key 唯一。沿用 inputs.records 顺序，所有关联按 (sample_id, paired_key, video_arm, audio_arm)，不得按 CSV 行号拼接。

scores_manifest.json 必须 complete、failures=[]、score_rows=curve_rows=expected_cells=225、sample_count=15、vshift=15、min_common_windows=30。

实际曲线格式：
- 顶层 curves.json = {schema_version, lags, curves:[...]}。
- 每行含 sample_id、paired_key、video_arm、audio_arm、lags、values、common_support_count、k_n。
- 顶层及每行 lags 必须严格等于 [-15,-14,…,15]；values 为 31 个有限实数。
- scores.csv 有 C、D、curve_median、k_star、d_zero、common_support_count、fixed_k_n、d_fixed_natural_lag。CSV 空字符串表示未提供，不是 0。
- 每个 cell 在 CSV 和曲线中恰好各出现一次，身份、支持数一致；非有限值、重复、缺失、额外 cell 均失败。
- 评分模型 hash 在 CSV 各行与 manifest 一致，同样本 score_box hash 一致。

八臂为 N_RAW、T_RAW、N_100、N_050、N_000、T_100、T_050、T_000。
六个重合成臂记为 R6。每条唯一 cell 集严格为：
1. (a,a)，a 遍历八臂；
2. (a,N_RAW)，a 遍历 R6；
3. (N_RAW,T_100)。

共 15×15=225 cell。每条除 (T_RAW,T_RAW) 外的 14 个 cell 支持数相等且 >=30。T_RAW 另有支持，只做原值展示和校验，不参与下文任何对比、bootstrap 或敏感性分析。

父运行保存的是每个 lag 已对时间窗口求均值的距离 d(k)，不是逐帧矩阵。不能从它伪造逐帧不确定性或恢复单位范数 embedding。

## 3. 逐 cell 重算指标

令 d 为当前 cell 的 31 点曲线。用 float64 从曲线重算：
- D=min(d)，越低越好。
- B=median(d)，即旧字段 curve_median；是全部 31 个 lag 的中位数，并非纯“错误配对”真值。
- C=B-D，越高越好。
- k_star 为取得最小值的第一个 lag，平局按 -15→15。
- d_zero=d(0)。

先验证重算值与 scores.csv：C/D/B/d_zero 用绝对误差 <=1e-6，k_star 和支持数严格相等。不允许四舍五入后校验。

lag 正号表示 audio[t+k] 使用更晚音频；不是官方 stdout offset。25 fps 下 1 frame=40 ms。不得用 k_star 的正负直接断言嘴型提前/滞后。

对同一条所有 14 个非 T_RAW cell，统一取自然基线 (N_RAW,N_RAW) 的 k_star 为 k_N。无论该 cell 自己的 k_n 是否为空，都从自然基线重新取得 k_N。
新增以下描述性指标：
- D_anchor=d(k_N)：固定坐标距离。
- C_anchor=B-D_anchor：固定自然 lag 的分数。
- search_bonus=D_anchor-D >=0，并核验 C=C_anchor+search_bonus。
- O=median{d(k): |k-k_N|>2}：排除自然基线最优位置 ±2 帧后的固定远侧背景。每条所有 14 cell 使用完全相同的 lag 集，不随候选最优位置移动。
- relative_margin=(O-D_anchor)/O。若 O<=1e-12，写 null 和原因，不加任意 epsilon、不要静默丢样本。正常完整统计中只要一个值为 null，该组此项 aggregate 写 null 并报告有效数；其他主指标仍可完成。
- lag_delta=k_star-k_N，单位 frame；同时保存毫秒值。

CSV 中 audio_arm=N_RAW 的 fixed_k_n/d_fixed_natural_lag 以及对应曲线 k_n 必须与重算一致；其他 cell 原本为空不算缺失。

C_anchor 不再给每个候选寻找最佳 lag，但 k_N 自身是自然基线数据选择的坐标，不是独立真值。relative_margin 只对整条距离曲线的正比例缩放不变，不等于单位范数 embedding 分析，也不能消除一般声学域差异。O 远侧位置未必全是错误音素，不能直接称“负样本准确率”。

## 4. 分解定义：先逐条对比，再聚合

对任意候选 cell a、基线 cell b，定义同方向的四个效应：
- delta_C = C_a-C_b。
- match_gain = D_b-D_a，正数表示最佳距离下降。
- background_gain = B_a-B_b，正数表示背景中位数上升。
- dominance = background_gain-match_gain，正数表示背景项数值贡献更大。

逐条核验 delta_C=match_gain+background_gain，绝对误差 <=1e-6。不得使用 D_a-D_b 却仍标成“改善”。

对于多 cell 的线性对比 L：
- delta_C=L(C)，match_gain=-L(D)，background_gain=L(B)。
- dominance=background_gain-match_gain，仍有相同恒等式。
- 固定 lag 辅助分解：match_anchor=-L(D_anchor)，search_change=L(search_bonus)；核验 delta_C=background_gain+match_anchor+search_change。
- 不计算任何“贡献百分比”。总增益可能接近零、两项也可能相互抵消。

原生视角 own_X(p) 指 (X_p,X_p)，固定自然音轨视角 natural_X(p) 指 (X_p,N_RAW)，X∈{N,T}、p∈{000,050,100}。
N 是自然来源，T 是历史 MFA-linear 条件特征来源；T_100 不是原始 TTS 波形。

## 5. 唯一主分析：三个原生轨迹对比

按下面顺序，对每条分别构造三个线性对比：

| ID | 公式 | 问题 |
|---|---|---|
| N_dose | own_N(100)-own_N(000) | 自然来源是否也依赖所消融轨迹 |
| T_dose | own_T(100)-own_T(000) | TTS 来源轨迹保留带来的数值变化 |
| source_interaction | T_dose-N_dose | TTS 是否比自然更依赖该操作 |

每个对比均报告 delta_C、match_gain、background_gain、dominance：共 12 个主统计量。禁止事后仅保留显著的来源或组件。

统计固定：
- utterance 配对分析，n=15；各样本等权，不按窗口数加权。
- numpy.random.default_rng(20260916)，integers(0,15,size=(20000,15))。
- 同一 index 矩阵同时用于所有条件、指标和主/辅助对比；先取每条对比，再对每次重采样的 15 条取均值。
- 报告 mean、median、positive_count（严格 >0）、n、普通 95% percentile CI。
- 12 个主统计量另报告 Bonferroni 区间：alpha_each=0.05/12；分位数 [0.05/24,1-0.05/24]，即覆盖率 99.583333…%。使用 np.quantile(...,method="linear")，保存未取整分位数。
- 主判读只用这 12 个校正区间；95% CI 仅辅助。无新增 p 值、speaker bootstrap 或帧级 bootstrap。
- source_interaction 先逐条 T_dose-N_dose，再 bootstrap；不能相减两个独立 bootstrap 区间。
- 来源是同一条的两种处理，不是两个独立样本组。15 条仍来自一个说话人。

## 6. 辅助分析：全部预先列出，不扩张矩阵

以下只提供普通 95% CI，并标 descriptive/unadjusted；不得与主统计混用作“显著机制”结论。

### 6.1 完整剂量与固定 lag

对 own 和 natural 两种视角、N/T 两种来源：
- 列出 keep=0、0.5、1 的 C、D、B、D_anchor、C_anchor、search_bonus、O、relative_margin 均值。
- 分解 050−000、100−050 两段变化；natural 视角另外给 100−000 和 T−N 的剂量交互。
- 列出每条 C 两段差、总体两段差，以及两段都严格 >0 的样本数。两段都 >0 才称该样本 C 单调上升；总体均值单调和逐条单调分开写。
- 对三个主对比报告 match_anchor、search_change、O 的差、relative_margin 的差；报告逐样本 lag_delta，以及各剂量臂 lag_delta!=0 的条数。
- 50% 几乎不变、仅 0% 损坏时，不写“清晰连续剂量响应”。保留原始两段数值供读者判断，不事后挑“几乎”的数值门槛。
- C 上升而 C_anchor 不升时，说明重新选择最优 lag 对增分有贡献；不能直接推断物理延迟的改善或恶化。

### 6.2 来源差与重合成参照

同时在 own/natural 视角分解：
- N_100−N_RAW：自然重合成参照。
- T_100−N_100：同声码器下的来源处理链差。
- T_100−N_RAW：历史 G_own 对应比较；natural 视角下为固定自然音轨总差。

N_RAW 在两种视角均为 (N_RAW,N_RAW)。核验第一项加第二项等于第三项的逐样本分解。
本轮 G_own 必须从曲线复算，预期均值约 +0.109；这仅为防错提示，不作为成功门槛。CI 若跨零，写“本轮阳性参照不确定”，不能据此推翻其他历史协议的 TTS 增益。

### 6.3 已有 2×2：区分条件响应

使用且仅使用现有四个 cell：
- Q00=(N_RAW,N_RAW)
- Q10=(T_100,N_RAW)
- Q01=(N_RAW,T_100)
- Q11=(T_100,T_100)

对 C、-D、B 分别应用：
- video_effect_at_N = Q10-Q00
- audio_effect_at_N = Q01-Q00
- av_interaction = Q11-Q10-Q01+Q00
- native_total = Q11-Q00
- audio_effect_at_T = Q11-Q10
- video_effect_at_T = Q11-Q01

核验 native_total=video_effect_at_N+audio_effect_at_N+av_interaction；每一线性对比也按第4节给出 C 的 match/background 分解。三种量共用同一抽样索引，不把它们视为重复独立证据。

这里 av_interaction 与第5节 source_interaction 不同，禁止都简称 I。
固定视频的 audio_effect 只证明评价端对音轨替换有响应，不能直接叫评分偏差。固定音轨的 video_effect 只证明视频改变影响该评分，不能直接叫真实同步改善。
Q 的 N_RAW 与 T_100 还混入编码/声码器与局部时序差异；此 2×2 不能识别纯 TTS 来源的训练因果，也不提供六个剂量臂完整的生成端/评价端分解。不得补造未评分的 cell。

### 6.4 支持阈值敏感性

阈值固定 30/35/40/45/50。按每条 14 个非 T_RAW cell 的共同支持筛样本，预期 n=15/13/12/10/8。
只重新聚合已有分数，不重评分、不截曲线、不改变 lag、不重切时间窗口。
表中报告每个阈值的样本 ID、n、三个主对比的四项均值及方向（positive/zero/negative）。
不增加 CI 或显著性检验，不按最有利阈值决定主结论。嵌套子集不是五次独立复现。

## 7. 预定判读规则

所有判读限定为当前探索性历史队列，先报告 G_own 不确定性，再讨论轨迹响应。

| 观察 | 允许表述 |
|---|---|
| T_dose 的 delta_C 校正区间下界 >0 | 当前条件下，保留轨迹提高原生 Sync-C |
| 上项成立，background_gain 与 dominance 校正下界均 >0 | 背景项贡献为正，且其数值贡献大于最佳距离项 |
| T_dose delta_C 与 match_gain 校正下界 >0，dominance 校正上界 <0 | 最佳距离项贡献为正，且其数值贡献大于背景项 |
| 两个组件均为正且校正区间都排除0，但 dominance 跨0 | 两项都有正向证据，不能区分谁更大 |
| 一个组件为负 | 明确说明抵消方向，不把总增益包装成两个机制都改善 |
| delta_C 不确定，组件却有变化 | 报告组件变化/抵消，不能说稳定总增益已建立 |
| source_interaction 的 delta_C 校正下界 >0 | TTS 来源对该消融操作更敏感；仍可能包含声码器域敏感度差 |
| N/T都改善，source_interaction 区间跨0 | 支持普遍轨迹作用的线索，尚未识别 TTS 特异性；不等于两来源等效 |
| 原生改善、固定自然音轨未改善 | 原生条件内增分尚未转移为自然音轨兼容性，机制仍开放 |

background_gain>0 不足以证明“错位配对更可区分”：结合固定 O、relative_margin 和 C_anchor 的方向描述。原始距离整体缩放也可提高 C；relative_margin 的结果仅作尺度诊断，不能代替独立评价器。
本轮没有人工同步、真实嘴型真值、音质因果对照；不能写“证明真实口型更准确”“证明 SyncNet 被欺骗”或“解释所有 TTS 优势”。
不能因为结果不支持假设而调 threshold、窗口、分组、主对比或消融强度。

## 8. 最小实现与输出

建议新增：
- scripts/experiments/mfa_linear_trajectory_decomposition.py
- tests/experiments/test_mfa_linear_trajectory_decomposition.py

现有 scripts/experiments/mfa_linear_trajectory_ablation.py 的 _curve_metrics、analyze_stage 和 _sensitivity_table 可供理解字段/方向，不修改旧脚本，不调用其生成或评分入口。可用标准库、numpy、matplotlib；不导入旧脚本导致 torch/模型加载。

需实现的单命令接口（当前尚不存在）：

    python scripts/experiments/mfa_linear_trajectory_decomposition.py --source-run runs/mfa_linear_trajectory_ablation_v2_support30_20260916 --output-dir runs/mfa_linear_trajectory_decomposition_20260916

只需要一个命令，不做多 stage、GPU 调度、数据库或大型独立 validator 框架。输出目录必须新建或为空；已有结果时明确失败，可换一个新的输出目录重跑，不自动清空。

交付文件：
1. inputs.json：spec 标识、source_run、5个输入文件的期望/实际SHA、15条身份、输出协议、bootstrap参数、脚本SHA、git commit（可取得时）、numpy版本。
2. cell_metrics.csv：225行重算 C/D/B/k_star/d_zero/support；14个机制cell有全部新增指标，T_RAW 新增机制指标写空并标 reference_only。
3. paired_effects.csv：每条、每个预定对比的分解，至少含 sample_id/paired_key/family/view/contrast/delta_C/match_gain/background_gain/dominance/match_anchor/search_change；辅助指标按适用情形保存，禁止用0代替缺项。
4. analysis.json：status、primary（三对比×四指标及双CI）、descriptive、dose、two_by_two、sensitivity、limitations。所有统计保留完整精度，禁用 JSON NaN/Infinity。
5. validation.json：每项校验名称、通过与否、期望/实际计数、最大恒等式误差；不是只写一个 valid=true。
6. report.md：中文，先给三个主对比的分解和不确定性，再给辅助结果、判读、局限。Sync-C 等分数展示3位小数；阈值、校正CI、样本数写清楚。
7. decomposition.png：上排 own、下排 natural，每排 C/D/B 三个 panel，N/T 各一条 keep=0/0.5/1 曲线；D 标“越低越好”。先算每条指标再画均值，不能先平均31点曲线再取min/median。

不要画因果贡献百分比饼图。图是 standalone PNG，无需交互页面。
出错时写 status=incomplete 与具体失败项，CLI 非零退出；不得输出成功报告。输入失败时不计算部分子集结果。正常完成时所有主值与CI均有限。
执行结束再计算一次5个源文件SHA，确认没有修改源运行。

## 9. 必须通过的少量科学测试

使用合成31点曲线与小型成对数据，不依赖 GPU：
1. 最佳位置固定为0：基线除中心5外均为10，C=5。
   - 仅中心降到4：delta_C=+1、match_gain=+1、background_gain=0。
   - 中心保持5，其余升到11：delta_C=+1、match_gain=0、background_gain=+1。
   - 全曲线加3：delta_C=0、match_gain=-3、background_gain=+3。
   - 全曲线乘2：C翻倍，relative_margin不变。验证“C升高不自动表示几何形状更尖锐”。
2. 平局时取第一个lag；k_N=+3时按lag值定位index=18，不能拿数组第3项或翻转符号。候选最小点移动时k_N/O的掩码保持不变，search_bonus及固定lag恒等式成立。
3. 样本身份乱序仍正确关联；重复/缺失cell、paired_key错配、lag乱序、NaN、共同支持不一致与输入SHA错配均明确失败；T_RAW支持与其余14cell不同不误报，但必须>=30。
4. 两条或三条人工样本验证source_interaction先配对再平均、四格分解恒等式；验证不同指标共用同一bootstrap索引及12项校正的精确分位数。避免只测试函数返回了自身计算值。
5. 构造两条最优lag不同的曲线，验证先逐条计算C/D/B再平均，而非对均值曲线取min/median。
6. 支持阈值只按完整14cell的样本支持筛选，T_RAW不进入；源文件前后SHA相同。

跑本脚本对应 pytest、py_compile、ruff；成功后不扩大到无关项目全测试。

## 10. 下游执行顺序与完成定义

1. 读本spec和仓库AGENTS；按Startup Router读取实验记忆指令。
2. 核验冻结输入，建立新输出目录。
3. 实现纯计算与必要测试；先用合成曲线验证符号、固定lag和配对。
4. 一次运行完整15条CPU分析；不需要GPU smoke。
5. 自我审阅公式、CSV关联、12项校正、图表方向和结论边界；修复实现错误后使用新输出目录重跑，科学参数保持不变。
6. 核验225输入cell、210机制cell、15 reference-only、45行主配对对比和12个主汇总；配对对比不因指标数重复计算样本数。
7. 按Startup Router更新本spec为implemented，并写入/更新对应实验结果笔记，含结果、结论与report链接。只在实际计算和验收完成后修改状态。
8. 最终交付报告路径、三个主对比结果、背景/匹配项判读、TTS特异性是否仍不确定、主要限制。结果为负或不确定同样属于有效完成。

## Observations

- [status] implemented
- [question] MFA-linear轨迹剂量响应来自最佳距离改善还是背景中位数变化，是否具有TTS来源特异性？
- [design] 复用support30运行的225条31点曲线；CPU-only，3个主对比×4指标，固定自然lag与已有2×2作辅助。
- [boundary] 历史单说话人探索性再分析；代数分解不等于真实同步或评价器偏差的因果归因。

## Relations

- extends [[MFA-linear 连续轨迹机制消融 support30 2026-09-16]]
- follows [[MFA-linear 连续轨迹机制消融实验 Spec]]
- relates_to [[LRS3 TTS 原生优势的分数分解与曲线诊断]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]

## Implementation

- [implementation] 新增 `scripts/experiments/mfa_linear_trajectory_decomposition.py` 与 `tests/experiments/test_mfa_linear_trajectory_decomposition.py`；冻结五个源文件 hash，按 identity key 读取 225 个 cell，支持可选 `k_n` 并从自然基线重算 `k_N`。
- [result] 完成最终运行 `runs/mfa_linear_trajectory_decomposition_20260916_v10`：N_dose `delta_C=+2.206`（Bonferroni CI `[+1.634,+2.838]`），T_dose `+2.072`（`[+1.585,+2.579]`）；两者背景项和 dominance 均为正且排除 0，source_interaction `-0.134`（`[-0.861,+0.517]`）跨 0。
- [validation] 225/210/15/45 cell 计数、固定 lag 恒等式、源 hash 前后不变、独立 bootstrap 重算均通过；专项与旧 ablation 联测共 26 个 pytest、ruff、py_compile 通过。报告路径为 `runs/mfa_linear_trajectory_decomposition_20260916_v10/report.md`。
- [boundary] 结论仍限于单说话人历史队列的 SyncNet 数值分解；没有 TTS 特异性因果证据，也不能把背景中位数上移解释为真实口型改善或评价器欺骗。

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据用户请求设计供下游实现的CPU曲线分解实验，冻结输入、计算、统计与判读；仅写spec | September 16, 2026 | user request / agent design |
| 按 spec 完成实现、v10 运行、独立审计并记录结果；状态更新为 implemented | September 16, 2026 | user request |
