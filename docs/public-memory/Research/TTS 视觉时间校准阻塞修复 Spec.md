---
title: TTS 视觉时间校准阻塞修复 Spec
type: note
permalink: tts-exp/research/tts-视觉时间校准阻塞修复-spec
status: implemented
protocol: tts_visual_timing_v2
question: 修复校准支持和实现错误后，能否进入独立视觉时间比较？
tags:
- tts
- visual-timing
- repair
- implementation-architect
- spec
---

# TTS 视觉时间校准阻塞修复 Spec

## 1. Objective

在既有 `tts_visual_timing_v1` 上做一次有边界的修复：扩大固定校准扰动的有效区域；修正静音导致的伪词层缺失、逐音素侵蚀时间支持、生成视频 LOW_MOTION 被错误剔除；补齐观测方向检验和独立门控复算。实现后执行新协议，校准通过才进入历史原生比较，后续仍遵守原实验的生成门控。

新协议名 `tts_visual_timing_v2`，正式目录 `runs/tts_visual_timing_v2/`，smoke 目录 `runs/tts_visual_timing_v2_smoke/`。这是根据 v1 已见结果修订的探索协议，不是未看数据的独立确认。本次交付已实现并完成 v2 验收；结果见文末实施与验收结果及 v2 结果笔记。

原 spec 未被本文明确修改的科学阈值继续有效。不得为了“跑通”降低门槛、换样本、选择最有利的窗口、把被跳过分支写成已测试。允许修复完成后仍得到合法的科学停止。

- [status] implemented
- [question] 修复测量支持和实现错误后，独立视觉时间代理能否完成校准并进入 TTS/N 比较？
- [finding] v1 的直接停止是中央区域事件不足；另有尚未进入主分析便可复现的代码错误，不能将所有阻塞统称科学阴性。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按 Implementation Architect 编写；只读复查 v1 原始事件、实际代码和 24 个 TextGrid；冻结一次修订方案 | September 20, 2026 | user request |
| 完成 v2 实现、smoke、正式运行与独立 checker；native 支持不足按原门停止 | September 20, 2026 | implementation |

## 2. Repository Model

`audit_inputs` → 固定 4 条校准与 12 条主样本 → `calibrate` → `compare_native` → 条件 `run_generation` → `report` → `check_run`。

事实来源：`runs/tts_visual_timing_v1/{manifest,calibration,validation}.json`、`events/calibration_*_REAL.json`、父 A 阶段 TextGrid、对应 Python 源码。旧 checker 的 PASS 表示它当时实现的检查通过，不表示所有科学门控都经过独立重算。

| 校准 ID 尾部 | 时长/s | 全片事件数 | v1 中央参考事件数 | 检测有效率 |
|:--|--:|--:|--:|--:|
| 6ul2TSvUDog_00007 | 18.08 | 7 | 2 | 1.00 |
| 6wk4dkYSrV0_00006 | 7.32 | 7 | 0 | 1.00 |
| 73jPh0eRPSY_00008 | 6.48 | 2 | 0 | 1.00 |
| 6qqqVwM6bMM_00007 | 13.32 | 3 | 1 | 1.00 |

四条 REAL 均为 MEASURABLE；v1 窗口是 `[0.35L+0.24,0.65L-0.24)`。因此直接证据是“中央支持不足”，不能归因于检测器完全没检测到动作，也不能保证扩大区域就通过校准。

另外已做只读函数诊断：

1. 当前 `audio_time_map(_parse_textgrid(N),_parse_textgrid(T))` 对 151–162 全部返回 `WORDS_TIER_REQUIRED`。TextGrid 实际存在 words tier；`_word_groups` 在排除静音前先检查词标签，句首静音被误报。
2. 仅在内存中去掉不可用 token 后，12 对均达到既有双侧 0.80 覆盖要求（N 最低约 0.910，T 最低约 0.924）。这是定位诊断，不是正式映射验收。
3. 此诊断下，10/12 条没有任何时长超过 480ms 的匹配 phone；156 有 5 个、160 有 1 个。当前逐 phone 两端各裁 240ms 会使大部分记录无支持。双侧连续的 phone 应组成支持块，但分段仿射映射仍保留各 phone 斜率。

## 3. Code Anchors

以下路径均相对仓库根。

| path | symbol | current role | required change |
|:--|:--|:--|:--|
| scripts/experiments/tts_visual_timing.py | main, audit_inputs | CLI、输入冻结 | 新增显式协议选择和 diagnose 阶段；v2 配置冻结；resume 验证身份 |
| 同上 | calibrate, _expected_event_time, _matched_recovery | 校准控制和判决 | 新控制区间、源时钟支持、观测方向、完整逐记录数值证据；v2 禁止最近帧冒充精确逆映射 |
| 同上 | _run_retime, _run_extract, _score_features | 缓存/worker 编排 | 绑定协议、参数、worker 与数据哈希；不得只凭文件存在或源视频哈希命中 |
| scripts/experiments/tts_visual_timing_metrics.py | local_warp_indices | 固定局部帧映射 | keyword-only protocol 参数，保留 v1 默认；增加 v2 固定折线 |
| 同上 | _word_groups, audio_time_map | 词序配对及 phone LCS | 先过滤静音/unknown，usable phone 才要求词归属；输出连续支持块，保留 phone segments |
| 同上 | aperture_events, _event_candidates | 原始事件提取 | 事件阈值不变；v2 修复完整邻域越界和平台最小值中点的时间表达 |
| 同上（新增符号） | continuous_support_blocks, evaluate_calibration_record | 无 | 小型纯函数：支持块与可测试判决；不新建框架 |
| scripts/experiments/tts_visual_timing.py | _aggregate_segment_distances, compare_native | 区间评分、统计 | 从连续块构造冻结支持；LOW_MOTION 与检测失败分开；保留额外/缺失事件惩罚 |
| 同上 | run_generation, _score_c_on_support | 条件生成/同窗评分 | 复用修复后的支持和有效性规则；修复本文列出的恒等 knots、共同窗和异常台账问题 |
| scripts/experiments/tts_visual_timing_worker.py | retime_video, main | FFV1 控制视频 | 显式传递 protocol，receipt 写控制配置和实际 mapping；Python 3.8 兼容 |
| scripts/experiments/check_tts_visual_timing.py | _independent_events, _distance, check_run | 目前主要重算事件和检查已有布尔值 | 独立重算 mapping、支持、校准全部条件、总门及下游门控；不能信任 passed/checks |
| scripts/experiments/tts_visual_timing_fixtures.py | run_synthetic_fixtures | 合成门控例子 | 覆盖真实失败结构，不只给 gate 函数手填阳性统计 |
| tests/experiments/test_tts_visual_timing.py | 现有测试 | 纯指标与部分门控测试 | 增加静音、短 phone 连续块、实际方向、LOW_MOTION、缓存和编排回归 |
| tests/experiments/test_check_tts_visual_timing.py | 现有测试 | 独立距离/静止验证 | 加入篡改证据必须失败的端到端小夹具 |

## 4. Reference Pattern

- `tts_visual_timing_metrics.py::event_distance`：继续使用既有有序编辑距离、空预测惩罚和确定性平局规则，不另建几何/DTW 指标。
- `tts_visual_timing.py::_append_cell`：沿用 stage/id/arm 去重更新，补充失败路径，不能少记未进入阶段。
- `tts_visual_timing.py::write_json`：继续原子写与拒绝 NaN。
- `tts_time_instance.py::parse_textgrid` 已正确输出“静音 word_index=None”；保留它，不修改历史共享 parser 去给静音捏造词。
- `tts_visual_timing_worker.py::retime_video` 的解码帧→索引映射→无音轨 FFV1→重新检测模式继续使用。不得用平移 landmarks 代替实际视频控制。
- checker 可以独立实现小型公式，但不调用 runner/metrics 的待审计判决函数。现有“读 checks 后 all()”不是可模仿的独立校准验证。

## 5. Invariants

1. 保留 v1 所有产物与原始结论，不覆盖旧 runs，不修改父 TextGrid/音视频。改代码前将此次涉及的旧源码复制到 v2 的 provenance/ 并保存原始路径和哈希；旧 run 的历史代码哈希不得改成新值。不能再用已修改源码的当前哈希宣称旧 run 完整复验。
2. 校准仍固定原 4 个 ID、不替补；主分析仍 151–162、固定分母 12。校准 3/4、每条至少 2 个参考事件、REPEAT 20ms、warp 恢复率 0.8/误差 40ms、单调误差及 FROZEN 门不降低。
3. aperture 阈值、200ms 邻域、LOW_MOTION=0.015、有效率 0.95、最长无效段 5 帧、编辑成本 240ms不调参。本文只修复边界/中点实现语义，必须标记版本变化。
4. 音频匹配仍按词序和词内确定性 LCS，双侧覆盖至少 0.80。不跳过错词、不跨重复词搜配、不插值未知音素/静音间隙。
5. 支持由真人 R、时钟来源和音频映射确定，不由 N/T 的视觉误差或 Sync-C 决定。生成没有动作应受漏检惩罚，检测失败应记不可测。
6. 不用视觉 DTW、互相关或 SyncNet 最优 lag 纠正待评估时间误差。保持既有模型、权重与官方预处理。
7. bootstrap 20,000、seed=20260920、原三项检验校正和全部 native/generation 阈值保持。不能把校准通过写成 TTS 有效。
8. 仅一个固定修订候选，不进行窗口/事件参数扫描。v2 失败即报告失败；不得在同次自动任务内继续改出 v3 直到阳性。
9. 不新增模型、训练、服务、调度框架或人工标注流程。单 GPU 串行；清理仅限本 run 临时文件。

## 6. Implementation Plan

### 6.1 先建只读诊断和版本边界

CLI 新增 `--protocol tts_visual_timing_v1|tts_visual_timing_v2`，默认 v1 保持兼容；本 spec 的所有命令明确传 v2。protocol 必须经参数/manifest 传递，禁止运行时修改模块全局常量。

新增 `--stage diagnose`：只执行输入审核、读取/重算 REAL 特征和 TextGrid 支持诊断，不生成扰动视频、不读取 N/T 视觉结果来选支持、不计算 BV/选择样本。写 `diagnosis.json`：4 条完整事件/旧窗口排除理由；12 条词层存在性、静音数、usable 无词归属数、匹配覆盖、phone 和连续块长度分布。旧事件文件只作为对照；v2 若修复了事件边界/中点语义，用原始 NPZ 重新提取事件。

写 `protocol.json`，包含全部阈值、折线节点、版本、固定 ID、父 manifest/本 spec/代码哈希；正式校准前冻结。resume 时重核对配置与所有消费输入，不匹配直接明确报 `CACHE_IDENTITY_MISMATCH`，不得将旧结果重新盖章为新协议。禁止 v2 指向旧 v1 run。

配置可以是现有模块内两个小字典，不新建插件/策略体系。v1 默认函数路径不变，v2 新逻辑显式选择；不能 monkeypatch 全局配置。

### 6.2 修复静音与词序映射，先用 24 个实际 TextGrid 验证

在 `_word_groups` 中先跳过非 usable token，再验证其余 token 的 word_index/word_normalized。正常静音没有词归属不构成错误；usable speech 没有词归属必须失败，不能为了通过丢掉它。

真实 words tier 要由 `_parse_textgrid` 包装器验证存在，并保存非静音/非unknown 的词序列表；复用原 parser 的区间读取，不能凭 token 中出现某个 word_index 就推断整词序完整。按非静音词的顺序配对，保留原 tier index 作证据；N/T 的静音数量不同不应要求 raw tier index 数字相同。合法词全部无可用 phone 时也不能悄悄把该词从词序中删除。

LCS 与 0.80 覆盖计算保持，未配到的 usable phone 仍进入覆盖率分母。无 words tier、usable phone 无词、词序不一致分别记录原因。诊断必须确认 12 对不再因为正常静音被误报；真正覆盖/词序不足可合法保留，不能硬编码全 12 条通过。

### 6.3 修复连续映射支持，避免逐音素扣掉整个区间

新增 `continuous_support_blocks(segments)`：验证每个 phone 区间正长度、按 N 和 T 均单调、不重叠。相邻两个 segment 仅当 N 端点相接且 T 端点相接（绝对容差 1e-6 秒）才放进同一个 block；不要求两个斜率相同。只用容差处理浮点误差，不跨越真实 gap；可吸附误差不超过容差的共享端点并记录。

block 保存 N/T 外边界、成员 segment 索引。映射仍逐 phone 仿射，不能用 block 首尾一条直线代替。保留原 segments 字段以兼容 build_audio_knots。

主支持按以下唯一规则生成：

1. 在 N 时钟的每个连续 block 中，选真人事件 r，其 `[r-0.240,r+0.240]` 完整位于 block 内，且真人所需帧邻域全部有效。不在每个 phone 上扣 240ms。
2. 将这些参考事件的上述邻域作区间并集，相接/重叠则合并；这是冻结评分支持。无合格 R 事件的 block 不参与，理由留档。边界统一左闭右开；精确落在共有边界的事件只归属一次。
3. 对同一支持，R 和 N 直接用 N 时钟；T 逐 phone 逆映射回 N 时钟。三者都按相同支持过滤，禁止 reference 用裁后区间而 prediction 用整个 block。
4. 支持内所有预测事件都保留，多余事件受惩罚；支持内无预测但有参考，E=240ms。逐支持区算成本后按总事件计数汇总，不跨缺口匹配、不平均区间均值。
5. 输出 `native_support.json`：冻结参考事件、支持区间、block/segment 来源、排除理由、配置/输入哈希。冻结动作发生在 N/T 视觉评分之前。generation 使用同一份支持，不根据新视频重选。

修正 `compare_native`：R 为 LOW_MOTION 或无足够参考支持时不可测；N/T 为 LOW_MOTION 时视为有效空预测参与惩罚；DETECTION_FAILURE/MULTIPLE_FACES 不可当空预测。generation 评分遵守同一规则。这一修改恢复原 spec 语义，不是新的筛选策略。

### 6.4 固定扩大的局部校准控制

v2 唯一控制折线为：

`(0,0),(0.10L,0),(0.20L,1),(0.80L,1),(0.90L,0),(L,0)`。

仍令 `s(t)=t+d*w(t)`，d∈{±0.08,±0.16}秒，输出 j 取 `floor(25*s(j/25)+0.5)`。帧数、fps、首尾保持；检查单调与合法索引，不用 clip。L<2 秒明确不足。正 d 读取未来、观测事件应提前。修改 `local_warp_indices` 和 worker 参数传递，不改变 v1 折线。

v2 参考源区间固定为 `S=[0.20L+0.400,0.80L-0.400)`，不是按事件密集处定位；400ms 来自最大 160ms 位移加 240ms 安全余量。这个修订受 v1 失败启发，须在报告承认已见数据。每条至少 2 个 R 事件，4 条全保留，预期稀疏第三条可能继续不足。

每个控制的可评分输出帧由 `mapping[j]/25 ∈ S` 决定。以“相同源时间支持”比较，不能把所有控制再次切同一个输出时钟窗口，以免平移后事件被人为裁掉。

期望时间通过实际离散 mapping 求逆。v2 支持内处于整数平移平台，若所需源帧在 mapping 中不存在则记 `CONTROL_MAP_UNSUPPORTED`；不要最近帧 fallback。重复命中取输出时间中点。半帧事件时间使用两相邻源帧各自逆映射时间的线性中点；保留真实分数时间，不用 Python round 强行取偶数帧。

REAL、REPEAT、四个 LOCAL、FROZEN 共 28 个 cell。除 REAL 可复用验证过的原始特征外，新的扰动视频必须实际生成并重新检测；REPEAT/FROZEN 可在全部参数与数据哈希一致时复用，不把 v1 LOCAL 当成 v2。

### 6.5 校准方向必须来自观测，科学状态必须可解释

事件时间细节：v2 的平台最小值时间是首末最小帧 PTS 的算术中点，另存 `event_position_frames`；`index` 可保留 nearest-half-up 供展示，但计算期望和匹配必须用分数位置。事件所需左右完整 5 帧邻域若超出片段，拒绝候选，不靠 max/min 截短后算通过。修改 runner 和 independent checker 各自实现，阈值保持。

新增纯函数 `evaluate_calibration_record`，输入原始 R 事件、各控制重检测事件、实际 mapping、有效性和冻结 S；输出数值、matches、五项 checks、passed 与原因。

恢复率和误差仍按“期望位置对重检测事件”做有序匹配，误差≤40ms 的匹配算恢复；恢复率分母为全部冻结参考事件。方向检验对这些恢复匹配，通过 reference_index 对应原 R 事件，计算 `observed_time-original_time` 的中位数；乘期望方向符号必须严格大于 0，无匹配失败。不能再只对 `expected_time-original_time` 检查符号。

REPEAT E≤20ms；四 warp 恢复率≥0.8、中位恢复误差≤40ms；各符号下 E160>E80>Erepeat；FROZEN 无有效动态且无事件。FROZEN 检测失败仍按原规则不产生可用正证据，报告必须注明其失败类型；完整原始特征缺失不能假装冻结控制成功。所有数值和每项布尔均落盘。

每条区分：
- `PASS`：数据齐且五项满足。
- `INSUFFICIENT_SUPPORT`：REAL 可测但参考不足，或固有可测域不足。
- `FAIL`：足够支持但指标没有通过已知扰动检验。
- `ERROR/DEPENDENCY_BLOCKED`：代码、输入、工具失败。

总门仍 3/4。前两种科学未通过与错误分开汇总；执行完整且校准不足可以 execution=COMPLETE、science=INSUFFICIENT_SUPPORT。smoke 永远 science=NOT_TESTED，不能触发正式生成。无需以 required_passed=99 代替显式 smoke gate。

### 6.6 使独立审计能够发现伪通过

checker 根据 manifest 协议选择规则；从 controls receipts/原视频帧数、原始 NPZ、PTS、事件文件重建 28 个精确的 id×arm 键。每种 arm 恰一条，不以“数量至少28”代替完整性；缺失证据必须有明确失败台账且不能支撑通过。

独立重算折线 mapping 与哈希、S 和输出支持、期望时间、匹配距离/恢复率、观测方向、单调顺序、FROZEN、逐记录及总门，比较落盘结果。不能导入主判决函数，不能只信任 checks、passed、science；从 worker receipt 或 arms 取 mapping，不能读取实际不存在的 summary.arm.mapping 后跳过核验。

对已执行 native，从 TextGrid、R/预测原始特征重算支持与 E/BV，验证固定12分母、可测交集、bootstrap索引和门控；对跳过阶段验证原因确实由重算门导致。校准 PASS 不代表 native PASS。

分别报告 `artifact_integrity`、`calibration_recompute`、`native_recompute`、`generation_recompute` 的 PASS/FAIL/NOT_RUN。没有执行的部分不能显示重算 PASS。坏数据/不一致返回非零；合法科学不足且证据自洽可返回0。

### 6.7 缓存与受控自动执行

缓存键至少包括输入字节哈希、协议、实际参数、执行该步骤的代码/worker哈希、模型哈希；NPZ/控制视频本身也核验输出哈希。旧视频哈希正确不代表旧控制配置正确。提取和事件评分分层：同源 landmarks 可验证后复用，但改评分协议必须重算事件。

`all` 顺序为 audit → diagnose → vsr-audit → calibrate → native → generation → report → validate。diagnose 不替代科学门。审计未通过不得继续评分；源数据/代码发生变化时不能静默 resume。修改工程问题后只在新配置身份下重算受影响步骤。

校准通过后 native 对全12条执行，支持不足也逐条报告。至少8个source和16个参考事件等原门槛保持。generation 只有校准、native BV和同交集ΔC都通过才执行；不因为用户说修复就强制放行。

### 6.8 下游进入前的最小一致性修复

这些问题已经由当前源码可见，随本次补丁处理，不等到得到阳性后再修：

- `run_generation` 当前 N_ID/T_ID 仅传两端 knots。应先 `build_audio_knots` 得到同一组实际 N/T 端点；N_ID 用 N→N、T_ID 用 T→T，数量与对应 T_NAT 相同。不改变 knots 合法性门。
- 原生 N/T 各用自己完整有效 SyncNet 窗；replacement 的 N/N_ID/T_NAT 用同一实际时间交集。不能仅取 min(row_count) 当时间对齐，更不能将 T_NAT 在一个长度算的 C 减去 N_ID 在另一个长度算的 C。优先复用 worker 保存的时间/裁剪来源；缺这些信息则补 receipt，不猜零起点。
- 视觉三/五臂必须逐一执行上面的状态分类；不能将 DETECTION_FAILURE 的空列表当正常漏事件。
- baseline统计各指标使用同一可测source交集；至少8组必须对全部参与baseline条件成立，不能仅以 Sync-C 可测数量代替。
- individual candidate/worker失败应逐cell记录、继续其他独立记录并生成报告；门控读缺失/None CI不得抛比较类型错误。任何不足按原门合法停止。
- 旧 VSR 本次只读沿用探索分支，不把 pair_rows 数值汇总升级成重新从 logp 完整复算。报告明确本次验证范围。

本节不改生成模型和科学判据。原 spec 其余尚未执行分支必须继续按原契约自审，不能用此前11个测试通过代替实际边界验证。

## 7. Expected Change Surface

### Must change

- `scripts/experiments/tts_visual_timing.py`
- `scripts/experiments/tts_visual_timing_metrics.py`
- `scripts/experiments/tts_visual_timing_worker.py`
- `scripts/experiments/check_tts_visual_timing.py`
- `scripts/experiments/tts_visual_timing_fixtures.py`
- 两个既有 `tests/experiments/test_*tts_visual_timing.py` 测试文件。
- 本 spec 的实施状态、新 v2 结果 BM 笔记、原 spec/核心问题节点的链接、HANDOFF.md 当前状态。

### May change

- `tts_visual_timing_syncnet_worker.py`：仅补共同时间窗所需来源/时间字段及对应验证，不修改模型前向。
- 新增一份小型固定 TextGrid 测试夹具；无需新增实验框架或第三个 runner。

### Should not change

- 历史 run/父清单/原音视频/TextGrid/旧报告；共享 `tts_time_instance.py` parser、第三方模型源码、任何模型权重。
- 原科学阈值、固定样本名单、语言划分、既有 VSR/F0/SyncNet 结论。
- 无关脏工作树、仓库结构、训练/部署环境。

## 8. Validation Plan

1. **静态与原回归**：
   `python -m py_compile scripts/experiments/tts_visual_timing.py scripts/experiments/tts_visual_timing_metrics.py scripts/experiments/check_tts_visual_timing.py`；
   `[redacted-local-path] -m py_compile scripts/experiments/tts_visual_timing_worker.py`；
   `pytest -q tests/experiments/test_tts_visual_timing.py tests/experiments/test_check_tts_visual_timing.py`。
   预期通过，v1 默认映射和既有距离数值不变。

2. **新增行为测试，全部必做**：
   - 首尾静音 word=None 不误判；真实 speech 无归属失败；words tier 真缺失失败；重复词及不同静音个数仍按词序配对。
   - 10个各100ms、双侧连续的 phone 可组成1秒支持块，中间事件不会因逐phone裁边消失；一个真实 gap 阻断跨gap支持；斜率变化不得被改成整块线性。
   - 同一冻结支持过滤R/N/T；额外事件受罚；正常LOW_MOTION预测得240ms；检测失败不可测；缺少参考不得给0误差。
   - v2 ±80/±160 实际索引分别在平台上为±2/±4帧，首尾固定且合法；事件靠支持边界时按源支持追踪，不被固定输出窗裁掉；半帧事件逆映射正确。
   - 完整邻域越界不收事件；最小平台中点可为半帧。
   - 错方向观测、始终无位移观测、缺失事件均不能通过；纯合成正确控制可通过，标记 SYNTHETIC_NOT_SCIENCE。
   - 篡改passed/checks/总门、一个mapping索引、一条事件时间，或删除/重复一个cell，checker必须拒绝相应不一致。
   - 校准失败禁止native/generation；校准通过native不足禁止generation；全门通过的小型编排fixture确实调用应调用步骤。合成结果不写成真实科学结果。
   - 同源但改protocol/delta/worker的缓存不命中；输入或输出哈希不符拒绝；跨v1/v2 resume明确失败。
   - generation identity knots逐点恒等且数量一致；非零crop起点的共同窗按实际时间对齐；None CI和单worker错误仍产出完整失败报告。

3. **24个实际TextGrid只读集成**：
   `python scripts/experiments/tts_visual_timing.py --protocol tts_visual_timing_v2 --run-dir runs/tts_visual_timing_v2 --stage diagnose`。
   预期确认正常静音错误已消除，落盘真实覆盖/连续块；此命令不要求科学通过，不运行N/T视觉比较。

4. **真实smoke**：
   `python scripts/experiments/tts_visual_timing.py --protocol tts_visual_timing_v2 --run-dir runs/tts_visual_timing_v2_smoke --stage all --smoke`；
   `python scripts/experiments/check_tts_visual_timing.py --run-dir runs/tts_visual_timing_v2_smoke`。
   使用原首校准ID和151，7控制真实I/O，checker完整验证，科学NOT_TESTED。smoke不能指导修改冻结参数。

5. **正式运行与独立验收**：
   `python scripts/experiments/tts_visual_timing.py --protocol tts_visual_timing_v2 --run-dir runs/tts_visual_timing_v2 --stage all --resume`；
   `python scripts/experiments/check_tts_visual_timing.py --run-dir runs/tts_visual_timing_v2`。
   预期28个校准cell齐全、总门独立复算一致；实际结果可以通过或不通过。通过才执行12条native及条件生成。重跑resume后台账不重复、结果一致。

6. **交付检查**：
   diagnosis/protocol/native_support（仅native进入后）/calibration/native/generation/analysis/validation/report与适用图存在。报告必须区分：原阻塞是否解除、软件错误是否修复、科学门是否通过、未测试部分是什么。BM写后读回，链接本spec、v1结果和原因总览。不得把“工程验收PASS”写成“TTS改善嘴型”。

## 9. Risks and Edge Cases

- 扩大平台可能仍不满足单调误差：周期性动作和离散匹配可能造成歧义。这是可接受的校准失败，不得降低E顺序门。
- 第三条只有边缘事件，预计仍难校准；保留它，不能复制其他片段事件或替换样本。
- v2 不允许“先试多种窗再选最好”。这是一次已见数据后的固定修订，所得结果仍探索性。
- 修复静音并不允许删去没有词标注的真实speech；必须保留错误。
- 连续支持块只表示映射连续，不表示整块相同语速；错误合并斜率会改写待测时间。
- 两端240ms要求与phone局部匹配严格性仍可能让主分析不足8组/16事件；不能借修复之名跨gap或降低门槛。
- 主脚本和checker复制同一错误不是真正独立；必须有手算小例、篡改测试和原始证据检查。
- 修改代码后旧run绑定的当前文件哈希会不同；保留历史快照与变更说明，不重写旧哈希制造一致。
- `SUPPORTED` 在calibration只指测量响应通过；不是机制成立，不是视觉真值，也不保证本地Wav2Lip效应复现。

## 10. Assumptions / Unknowns

- VERIFIED：v1四条REAL有效率1.0、事件总数7/7/2/3，中央事件2/0/0/1；只有一条校准通过。
- VERIFIED：12对实际TextGrid均因静音词归属检查返回伪WORDS_TIER_REQUIRED；仅排除不可用token的诊断可恢复双侧覆盖≥0.80。
- VERIFIED：当前支持侵蚀按phone执行，偏离原spec“连续有效映射段”；当前native将生成LOW_MOTION也拒绝。
- VERIFIED：现方向检查取期望位移，旧checker主要复用保存的checks，不能独立证明观测方向及全部校准门。
- VERIFIED：当前缓存对控制只查源视频哈希；协议改动需要新配置身份。
- VERIFIED：v2 更宽固定平台完成实际验证，3/4 校准记录通过；第 3 条仍因参考事件不足而保留为 INSUFFICIENT_SUPPORT。
- VERIFIED：v2 校准 3/4 通过；随后 native 仅 5/12 source、11 个参考事件可测，B_V 门未通过；ΔC 与 Wav2Lip 迁移未执行。
- UNKNOWN：generation 未执行，因此其完整真实运行是否还有未触发的问题仍未知；不能把未执行分支写成成功证明。

## 11. Handoff Contract

按第3节符号修改，先诊断/纯函数回归，再真实smoke，再冻结正式执行；保持第5节不变量，复用第4节模式，补丁限制在第7节。实现全部修复与失败路径，不做无关重构。遇到证据与计划矛盾，保存具体反例并报告，不能静默修改科学规则。

可验收交付包括：修复补丁、聚焦测试通过、真实28控制与独立重算、按真实门控完成的后续阶段、报告/BM/交接更新。允许合法科学停止，但不允许以“校准未过”为由漏实现静音/支持/审计修复。

若仅被校准阻断，结论写“测量仍未校准，TTS动作时间问题未测试”；若进入native但无支持，区分支持不足与效果不明确；只有真实满足后续门槛才能报告对应改善。

Relations：
- repairs [[TTS 嘴部动作时间校准与自然时间轴迁移实验 Spec]]
- follows [[TTS 嘴部动作时间校准与自然时间轴迁移实验结果 2026-09-20]]
- relates_to [[TTS 增强 TFG 的原因：证据、排除项与因果分叉]]


## 实施与验收结果（September 20, 2026）

- 状态已由 `planned` 更新为 `implemented`。实现文件：`scripts/experiments/tts_visual_timing.py`、`tts_visual_timing_metrics.py`、`tts_visual_timing_worker.py`、`check_tts_visual_timing.py`；对应实验测试已更新。
- 静态与回归验收：`pytest -q tests/experiments/test_tts_visual_timing.py tests/experiments/test_check_tts_visual_timing.py` 为 **15 passed**；相关脚本 `py_compile` 通过。
- v2 diagnose：12/12 主 TextGrid 对完成；N 覆盖率 0.910–1.000、T 覆盖率 0.924–1.000；正常静音不再触发伪 `WORDS_TIER_REQUIRED`，连续支持块按真实 gap 保存。
- v2 smoke：真实 I/O 和独立 checker 通过；按契约标记 `science=NOT_TESTED`，没有用 smoke 结果放行正式科学阶段。
- v2 正式运行：校准 3/4 PASS，1/4 为 `INSUFFICIENT_SUPPORT`，达到至少 3/4 的校准门；随后 native 对 12/12 执行，但仅 5/12 source、11 个参考事件可测，未达到至少 8 source/16 reference 的原门。B_V 可测组均值 65.1507 ms，98.333% CI [-0.0363, 158.8571] ms，native gate=`NO_CLEAR_SUPPORT`。
- generation 按原方案合法 `SKIPPED_BY_GATE`；没有把检测失败、空预测或未执行分支写成科学阴性。
- 独立 checker 最终 `runs/tts_visual_timing_v2/validation.json` 为 PASS，独立重算校准 28 个 cell、64 个正式特征/事件；native/generation 的未执行部分明确为 `NOT_RUN`。
- 结果笔记：[[TTS 视觉时间校准与自然时间轴迁移实验结果 v2 2026-09-20]]。本修复解决了 v1 的工程阻塞并完成校准，但没有解决 native 支持不足，因此不能据此判断 TTS 是否改善嘴型或是否带来生成因果收益。
