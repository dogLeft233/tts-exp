## Purpose

为固定 LRS3 fit-only cohort 校准 Wav2Lip/SyncNet 对局部音视频时间错配的响应。通过先审计、再执行唯一控制分支，区分工程错误、自身配对失效和错配敏感性不足，给后续 bridge 实验提供可复核的依据。

## ADDED Requirements

### Requirement: Audit binds immutable history and a fixed cohort

实验 SHALL 以 `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/` 为只读历史输入，校验 `04_final/final.json` 的文件 SHA-256 为 `df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e`，并沿其 manifest hashes 验证上游文件。文件 SHA 与 JSON 内去除自哈希字段后计算的 artifact hash SHALL 分开校验。

实验 SHALL 原样复用历史 cohort 的顺序、22 records、22 source groups，ordered-ID hash 为 `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。这批数据已看过结果，SHALL 标记为 `fit_only_control_calibration`，不能声称新的独立确认。

新渲染前，审计 SHALL 覆盖全部 22 records 的 `N`、`N_REPEAT`、`LOCAL_SWAP` 音频/视频及四个控制评分 cell，逐项记录证据：

- 从 natural PCM 重建 `LOCAL_SWAP`，核对实际 PCM、样本数和三个边界。
- 追踪实际 Wav2Lip `--audio` 输入及其加载路径、face、geometry、checkpoint、临时目录，确认不同 arm 没有复用 natural 输入或渲染输出；`N_REPEAT` 是独立渲染。
- 从实际 mux 解码音频并与该 cell 指定音频比较 bytes；提取视频 elementary stream 并与该 cell 指定视频比较 bytes；同时核对 PTS 起点、25 fps 和音视频时长，不能仅检查 sidecar 的 true 标志。
- 核对 SyncNet 输入、独立 reference/data_dir、日志与 score 行的映射；reference 是缓存命名空间，实际音频来自传入媒体，不能用 reference 名字证明音频正确。
- 从原始日志复核 C/D/offset 及 damage 符号，记录源码/运行时与历史绑定是否一致。缺失 offset、非有限值、模糊的多 track 输出不能补成合法评分。

#### Scenario: Audit completes without an identified defect
- **WHEN** 所有审计项有可核实证据且未发现违反原控制契约的错误
- **THEN** 输出 `audit_decision=NO_DEFECT_FOUND`，允许锁定 `SMOOTH_WARP` 分支；此结论不声称已证明 LOCAL_SWAP 失败的声学原因

#### Scenario: Audit identifies a concrete defect
- **WHEN** 输入、缓存、mux、解析或公式存在可复现的契约违例
- **THEN** 输出 `audit_decision=DEFECT_FOUND`，记录最小复现、影响范围和证据路径，进入 `REPAIR_ONLY` 分支

#### Scenario: Audit cannot establish provenance
- **WHEN** 必需文件缺失、hash 不匹配、历史执行无法核实或审计未完成
- **THEN** 输出 `audit_decision=INCONCLUSIVE` 和终态 `BLOCKED`，不生成新控制媒体

### Requirement: One branch is frozen before new media execution

实验 SHALL 在新 run 的 protocol 中锁定审计 hash、分支、cohort、transform、评分参数、模型与执行源码 hashes、矩阵及统计门槛。历史结果读取是审计所需，SHALL 如实记录；协议锁定前 SHALL 不读取本次新评分。

- `REPAIR_ONLY` SHALL 最小修复已证明的工程错误，以原 `LOCAL_SWAP` 为控制臂，仅重跑原控制子矩阵，不改变其变换、geometry 策略、模型或门槛；缺陷回归测试通过后才执行媒体。
- `SMOOTH_WARP` SHALL 使用 `LOCAL_WARP_120` 为控制臂，不同时执行 LOCAL_SWAP。

两个分支 SHALL 互斥。出现科学失败后 SHALL 终止本次实验，不切换分支、不改强度或阈值重试。未能修复的工程错误 SHALL 产生 `BLOCKED`。

#### Scenario: A repair is verified
- **WHEN** 缺陷的回归测试已从失败变为通过且协议已锁定
- **THEN** 执行三个臂 `N, N_REPEAT, LOCAL_SWAP` 的新控制子矩阵

#### Scenario: An unsuccessful control tempts an adaptive retry
- **WHEN** 已锁定分支的自身有效性或敏感性未通过
- **THEN** 保留全部结果并输出 `CONTROL_FAILED`，不运行另一分支

### Requirement: The smooth control has one deterministic time map

两分支的 `N` 与 `N_REPEAT` SHALL 与 bound natural 的 decoded PCM bytes 完全相同。

`REPAIR_ONLY` 的 LOCAL_SWAP SHALL 使用原协议：`b1=floor(L/4), b2=floor(L/2), b3=floor(3L/4)`，输出 `x[:b1] || x[b2:b3] || x[b1:b2] || x[b3:]`。

`SMOOTH_WARP` SHALL 将 16-kHz mono PCM16 natural 的整数样本 `x[0:L]` 按以下固定映射构造 `LOCAL_WARP_120`：

```text
a = 1920 samples                         # 120 ms，所有 records 固定
n = 0, ..., L-1
s[n] = n + a * sin(2*pi*n/(L-1))         # 输出 n 读取输入 s[n]
s[0] = 0; s[L-1] = L-1                  # 消除浮点端点误差
z[n] = linear_interpolate(x, s[n])
y[n] = round_to_nearest_ties_to_even(z[n]) -> int16
```

计算 SHALL 使用 CPU float64；输出仍为原长 16-kHz mono PCM16。SHALL 验证 `L-1 > 2*pi*a`、`s` 严格递增且在 `[0,L-1]` 内。映射在前后半段包含相反方向位移，SHALL 不用单一 global shift 代替。输出 SHALL 不加 padding、截断、RMS 匹配、滤波或自动强度缩放。实验 SHALL 保存输入/输出 hashes、样本数、最大正负位移、最小/最大映射步长及 RMS/peak/clipping 描述性 QC。

#### Scenario: Smooth warp obeys the contract
- **WHEN** 输入格式正确且时间映射满足范围和单调性
- **THEN** 产出唯一固定控制，保持语序、首尾样本和总样本数，QC 在新评分前冻结

#### Scenario: A record is too short or malformed
- **WHEN** 任一 record 不满足输入或映射约束
- **THEN** 输出 `BLOCKED`；不跳过该 record、不降低振幅、不补样本

### Requirement: The calibration uses fresh renders and exactly four cells

令 `T` 为所选分支的控制臂。实验 SHALL 独立新渲染三个臂 `N, N_REPEAT, T`，每个 record 共用历史绑定的 face、geometry 策略、Wav2Lip checkpoint 和推理参数。不同 record/arm SHALL 使用独立工作目录。SHALL 不复用历史视频或历史评分，也不将 source face video 直接作为生成视频。

每个 record SHALL 仅评分以下矩阵，共 66 个新视频、88 个唯一新 cell：

| Cell | 用途 |
|---|---|
| `V_N/A_N` | 同次运行基线 |
| `V_N_REPEAT/A_N` | 独立重复性 |
| `V_T/A_T` | 控制自身有效性 |
| `V_T/A_N` | 换回自然音频的损伤 |

每个 mux SHALL copy video stream、使用 16-kHz mono PCM s16le，并验证解码 PCM、视频 elementary stream 和时间轴。SyncNet SHALL 保持原官方 V2 权重、预处理、`min_track=50` 和 global-offset 搜索范围；将所有有效参数含默认值记录进 protocol。每个 cell 的 reference/data_dir SHALL 唯一。SHALL 对缺失、重复、跨 record、绑定错误或不完整评分 fail closed，不使用零填补，不挑 track 或选择最好评分。

#### Scenario: A complete valid matrix is available
- **WHEN** 66 个新视频和 88 个 cell 的输入、输出及运行时绑定全部可验证
- **THEN** 才允许计算科学终态

#### Scenario: A media or score cell is invalid
- **WHEN** 任一必需 cell 缺失、歧义、被错误缓存、发生时间轴错配或不能解出有限 C/D 和整数 offset
- **THEN** 输出 `BLOCKED`，报告失败 cell，不能在剩余子集上判定控制通过

### Requirement: Calibration requires repeatability validity and sensitivity

实验 SHALL 对 22 个 source groups 作配对 cluster bootstrap：10,000 draws、seed `20260904`、沿用历史 NumPy `default_rng` 的 PCG64 随机序列、source-group labels 排序、每次有放回抽取 22 groups，计算 group mean 的均值，取线性插值的 2.5%/97.5% 分位数。每个 endpoint SHALL 从同一 seed 重新初始化，沿用历史配对抽样顺序。SHALL 使用未作展示舍入的数值判定，报告 Sync-C 三位小数。

记 `B=(V_N,A_N)`，`R=(V_N_REPEAT,A_N)`，`O=(V_T,A_T)`，`X=(V_T,A_N)`。所有差值定义为正值更符合预期：

| Gate | 逐 record 差值 | 通过条件（全部满足） |
|---|---|---|
| repeatability | `C(R)-C(B)`；`D(B)-D(R)` | 两个均值 CI 下界均 `> -0.10`；至少 20/22 的 offset 与 B 相差 ≤1 frame |
| own_audio_validity | `C(O)-C(B)`；`D(B)-D(O)` | 两个均值 CI 下界均 `> -0.10`；至少 20/22 的 offset 与 B 相差 ≤1 frame |
| replacement_sensitivity | `damage_C=C(O)-C(X)`；`damage_D=D(X)-D(O)` | 两个均值 CI 下界均 `> 0.10`；至少 18/22 的两个 damage 同时 `> 0` |

实验 SHALL 报告每个 gate，即使前面的 gate 已失败。上述 repeatability 为沿用原协议的非劣性/offset 检查，SHALL 不将其称为双侧等效性检验。SHALL 不把自然音频替换后变好、仅 offset 变化或波形明显移动当作敏感性通过。

#### Scenario: All registered gates pass
- **WHEN** 数据完整，且三个 gate 均通过
- **THEN** 输出 `engineering_decision=GO`、`scientific_decision=CONTROL_CALIBRATED`

#### Scenario: Any registered gate fails
- **WHEN** 工程有效但任一 gate 未通过（CI 下界等于其严格阈值不通过；计数达到 20/22 或 18/22 则满足对应计数条件）
- **THEN** 输出 `engineering_decision=GO`、`scientific_decision=CONTROL_FAILED`，列出所有失败 gate；自身有效性失败时不能单独解释敏感性为成功

### Requirement: Output is reproducible and does not promote historical bridge evidence

实验 SHALL 仅在新 `runs/lrs3_local_timing_control_calibration_<run_id>/` 写运行产物，历史 runs 保持只读。SHALL 提供 audit、冻结 protocol、audio/video/score manifests、逐 record 统计、final JSON、简短结果说明和不重新运行模型的独立 validator。终态 SHALL 是 `BLOCKED`、`CONTROL_FAILED` 或 `CONTROL_CALIBRATED`；`BLOCKED` 时工程决定为 `BLOCKED`，未执行 gate 为 null 并解释原因。

final SHALL 绑定所有上游产物 hashes，记录 cohort、分支、执行源码/命令、实际/期望矩阵计数、每个 gate 的均值/CI/计数和失败原因。SHALL 始终设置 `reference_conditioned_audio_head_spec_eligible=false`。独立 validator SHALL 从上游证据复算数量、hash、统计及判定，不能只信 final 中的 pass 标记。

完成后 SHALL 停在对照校准结论；不能重测 BRIDGE_075、重写历史 bridge 决定、自动进入训练/fine-tuning/部署/held-out evaluation、生成 TTS、重跑 MFA/DTW、按分数/QC/画面筛选样本或进行强度搜索。控制时间重采样仅限本 spec 明确规定的变换。SHALL 不自动 commit、push 或创建 PR。

#### Scenario: Calibration succeeds
- **WHEN** 最终结果为 `CONTROL_CALIBRATED` 且独立验证通过
- **THEN** 报告该固定 fit cohort/transform/endpoint 上的校准结果，建议单独设计 bridge 收益确认；历史 +0.032 不因此成为显著收益

#### Scenario: A resumed run differs from the frozen run
- **WHEN** 已有 cell 的输入、配置、代码绑定或输出 hash 发生变化，或已有不完整 cell 无法验证
- **THEN** 保留原产物并输出 `BLOCKED`，不覆盖或挑选重试结果；正常 resume 仅接受身份完全一致的完整 cell
