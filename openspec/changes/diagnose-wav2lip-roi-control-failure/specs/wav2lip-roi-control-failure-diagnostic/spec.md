## ADDED Requirements

### Requirement: Bind one immutable parent experiment

实验 SHALL 固定读取 `runs/wav2lip_face_roi_replacement_20260906_host_fix5/` 的以下文件，并分别验证文件 hash 与已有 JSON 自哈希（两者不是同一种 hash）：

| 相对父目录路径 | 文件 SHA-256 |
|---|---|
| `final.json` | `76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563` |
| `validation.json` | `328e562cc8dacfb960fb2319fb15f1cfa90eecf593f04434382c9a5e13bd2577` |
| `control.json` | `6944cc299f30f29a2c6f6a4f41db024014ec09aa583ca2bbc15c12133c8abce5` |
| `protocol.json` | `835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4` |
| `scores/control/manifest.json` | `60950f9a26abcb37d9113541e23acd038feb8f21b827f07fb3c2f07570aa3799` |

SHALL 沿父 manifest 核验本轮使用的 audio/video/media/score 绑定、文件 hashes 和 sidecar 配对。允许 fix5 的 symlink 指向 fix1，保存逻辑路径和 resolved 路径；不得把路径中 run-id 不同单独判为混用。逐条核对生成臂 G_N→N、G_NR→N_REPEAT、G_W→W；评分 cell 音轨由自身 audio_arm 决定，G_W/N 是合法的换轨 cell。

SHALL 核对 mux 的 decoded PCM 等于指定音频 PCM，同一 video arm 搭配 N/W 的视频流身份一致；N_REPEAT 与 N 的 PCM 相同。保留父 cohort 的顺序及 22 records / 22 source groups，ordered-ID hash=`5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。

SHALL 读取父 `final.spec_bindings` 中的 ROI spec 和 inherited timing spec 全文并验证 hash；两者是继承公式的依据。新 `protocol.json` 在计算诊断前冻结这些绑定、本 change 的 proposal/design/spec hashes、本轮源码 hashes、环境版本和全部参数。`tasks.md` 不参与锁定。

#### Scenario: Missing or modified parent assets

- **WHEN** 必需文件缺失、固定 hash/PCM 身份不符或 cell 缺失/重复
- **THEN** 输出 `BLOCKED` 和精确路径/cell/实际值，不换父 run、不补算分数、不仅分析剩余记录

### Requirement: Independently reconstruct the frozen gates

SHALL 读取全部 198 份有限值 `[window,31]` 距离矩阵。每条固定七个主 cell：`R/N, R/W, G_N/N, G_N/W, G_W/N, G_W/W, G_NR/N`；另有 `R/N, G_N/N` 两个 repeat=true cell。主 cell=154，重复 cell=44；全部按完整主键匹配。

SHALL 按继承 timing spec 独立重建 `s[n]=n+1920*sin(2*pi*n/(L-1))` 及其分段线性逆映射；s 的端点固定为 0 与 L−1，采样率 16000、每帧 640 samples、25fps。使用已核验帧数，Q 取 R/G_N/G_NR/G_W 帧数和 floor(L/640) 的最小值；各生成臂帧数必须相同。矩阵行数必须为 `min(F_video,floor(L/640))-5`。

共同候选行为 `15 <= r < Q-20`。以 `t=640*(r+2)` 得到 `d=(s(t)-t)/640`、`a=(t-s_inverse(t))/640`；PLUS 要求 d/a 均 ≥2.5，MINUS 要求均 ≤−2.5，各至少 5 行。重建的行集合必须和父 masks 相同，不能按分数选窗。行集合或派生预测不同应记录为审计差异；无法建立合法支持则 BLOCKED。

对共同候选行和 PLUS/MINUS 分别求矩阵的列均值曲线，SHALL 独立计算：`k=argmin(curve)`（并列按最小列索引）、`offset=15-k`、`D=min(curve)`、`C=median(curve)-D`。局部清晰峰要求最小/次小值间距严格 >0.010 且 offset 不为 ±15。

SHALL 复算父 ROI spec 的全部五组控制规则（评分重复性/基线、生成重复、A/B/C/O、own-audio、replacement-damage），不得只复用旧 passes。bootstrap 固定 22 个 source groups 排序、每次有放回抽 22 组、10000 draws、NumPy default_rng/PCG64 seed=20260905，每个指标重置 seed，线性 2.5/97.5 分位数。全精度判定，Sync-C 展示三位小数。

SHALL 对照父派生结果：行集合/offset/计数/布尔值必须精确一致，曲线/C/D/预测残差/均值/CI 的绝对误差容限为 1e−6；旧官方三位小数日志仅用 ≤0.001。记录超过容限的字段及双方数值，不用旧值覆盖复算值。派生不一致属于诊断发现，和源文件 hash 损坏分开处理。

#### Scenario: Stored pass flags disagree with raw matrices

- **WHEN** 原始输入完整但独立复算与历史派生值或 gate 不同
- **THEN** 保存字段级差异和最小可复现输入，诊断为 `AUDIT_MISMATCH`；不改父文件或自动重跑控制

### Requirement: Explain inherited C failure record by record

C SHALL 固定定义为两段 `actual=off(G_W/N)-off(G_N/N)`、`expected=−mean(d)`、`residual=actual-expected`。每条成功仍要求 G_N/N 基线可解释、双方两段峰均清晰、两段 |residual|≤1；计数要求 ≥18/22。

SHALL 输出全部 22 条的两段行数、双方 31 列曲线、offset、peak_gap、expected、actual、residual，以及以下可并存失败标志：`baseline_invalid`、`boundary_peak`、`unclear_peak`（gap≤0.010）、`offset_error`（|residual|>1）。峰不清晰时仍保存数值，但响应方向/幅度的解释标记为不可确定。

SHALL 分别统计每种标志涉及的记录数和完整失败 ID 列表，分母始终 22；多种标志可能重叠，不能相加当失败总数。A/B/O 同表保留作诊断背景，不替代 C。

#### Scenario: A peak is ambiguous but its offset is near prediction

- **WHEN** 某段 |residual|≤1 但 peak_gap≤0.010
- **THEN** 该条 C 仍失败，原因包含 unclear_peak；不能称为明确的模型响应不足，也不能删除该条

### Requirement: Decompose own-audio confidence without changing its endpoint

仅使用共同候选行的曲线。令 N 表示 G_N/N，W 表示 G_W/W，M 为曲线中位数，SHALL 逐条输出：

```text
own_C = C_W - C_N
own_D = D_N - D_W
median_change = M_W - M_N
own_C = median_change + own_D
```

SHALL 验证末行恒等式误差 ≤1e−9，保存两条完整曲线、M/C/D/offset、三项差值和各自均值/95% CI；门禁只采用原 own_C/own_D 的 CI 下界均 >−0.10，以及 offset 差≤1 帧的记录 ≥20/22。两个 CI 的端点不可直接相加；每项从逐条配对差值独立 bootstrap。

SHALL 报告各下界距 −0.10 的差、offset 合格计数、全部逐条差值。不得增设“CI 不能跨 0”规则，不做移除离群点、leave-one-out 挑子集、换 seed 或扩大样本来使门禁通过。

#### Scenario: Confidence crosses zero but meets noninferiority

- **WHEN** own_C CI=[−0.05,0.20]，own_D 与 offset 也达标
- **THEN** own-audio 门禁通过；若下界恰为 −0.10 则失败

#### Scenario: Minimum distance improves but confidence declines

- **WHEN** own_D>0 且 median_change 为更大的负数
- **THEN** 如实报告 own_C<0 及上述分解，只解释评分曲线变化，不据此断言声学劣化的因果来源

### Requirement: Deliver a bounded diagnosis and a concrete handoff

SHALL 使用新目录 `runs/wav2lip_roi_control_diagnostic_<run_id>/`，不覆盖已有目录。交付 `protocol.json`、`audit.json`、`diagnostics.json`（22 条记录、独立 gates、汇总和历史差异）、`result.md`、`final.json`、`validation.json`。大矩阵只引用并校验 hash，不复制或重新生成。

final SHALL 绑定本轮证据文件 hashes、父入口、预期/实际记录与矩阵数，使用以下唯一状态：

| 条件（按优先级） | status | diagnostic_decision | historical_scientific_decision |
|---|---|---|---|
| 缺资产、身份不符、无法计算或本轮验收失败 | blocked | BLOCKED | CONTROL_FAILED |
| 完整复算发现历史派生值不一致 | complete | AUDIT_MISMATCH | CONTROL_FAILED |
| 完整复算与历史一致 | complete | CONTROL_FAILURE_REPRODUCED | CONTROL_FAILED |

完整合法诊断且 validator valid 时退出 0（包括 AUDIT_MISMATCH）；阻塞/验收失败退出 2。发现历史偏差不等于本轮工程失败，源数据身份损坏则不能当作可复算的历史偏差。始终写 `new_generated_videos=0`、`new_score_cells=0`、`bridge_executed=false`、`training_authorized=false`、`cross_model_spec_eligible=false`、`generalization_established=false`。

离线 validator SHALL 从锁定输入独立复核 masks、C 失败标志、own-audio 分解/CI、全部 gate 与终态，不信任 producer 的 pass 字段。测试 SHALL 覆盖错 offset 符号、0.010/1 帧/−0.10 边界、跨零但非劣性通过、C/D 分解、错配/缺 cell、重复失败标志计数、伪造通过字段和 final 篡改。至少一个合成矩阵 fixture SHALL 有手工可验证的预期值。

报告 SHALL 用一张全记录表及简短说明回答三件事：原失败是否复现、C 失败由哪些可观测条件构成、own_C 与 own_D 的关系。给出一个最小下一步建议和支持证据：发现偏差则指出文件/字段与复现步骤，复现失败则提出要区分的假设及单一干预；证据不足明确写未确定，不自动执行建议。本轮不解锁 bridge，不把已见数据诊断称为 replacement 或泛化成功。

执行完成后 SHALL 按 BM Startup Router/实验指令，在 `tts-exp` 搜索后创建或更新同一份诊断笔记，绑定报告和真实状态，关联父实验。读后纠正父实验笔记中“跨零导致 own-audio 失败”的表述为实际非劣性阈值及 CI，保留父 CONTROL_FAILED 与历史事实。只写 spec 时不记录虚构实验结果。

#### Scenario: The historical failure is reproduced

- **WHEN** 审计、独立复算和 validator 全部完成，C=14/22，own_C 下界约 −0.170397，其他 gate 与父结果一致
- **THEN** 输出 CONTROL_FAILURE_REPRODUCED、失败分解及一个后续建议，完成 BM 记录后结束，不重新生成视频来追求 PASS
