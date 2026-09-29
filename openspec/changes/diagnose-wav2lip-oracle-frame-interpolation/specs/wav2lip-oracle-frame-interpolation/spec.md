## Purpose

检验固定像素帧线性插值能否改善已知 retiming oracle 的 own-audio，并同时维持原控制有效性。

## ADDED Requirements

### Requirement: Freeze the parent cohort and the single intervention

实验 SHALL 以 design.md 的九个文件 hash 固定 oracle_v4，校验父 validation.valid=true、其 final 文件绑定及各 JSON 自哈希。SHALL 沿 manifests 校验实际使用的像素、N/W PCM、88 个历史评分 cell 及 embeddings/矩阵；父资产只读，缺失或不匹配为工程 BLOCKED，不换 run、不补算历史 cell 冒充缓存。

SHALL 使用父 protocol 全部 22 records / 22 source groups 和顺序，ordered-ID hash `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。SHALL 全文读取父 change `diagnose-wav2lip-roi-retiming-oracle/specs/wav2lip-roi-retiming-oracle/spec.md`，以及其引用的 timing / ROI / peak-recheck spec；继承映射、掩码和 SyncNet 定义。本 spec 只覆盖线性混帧、缓存复用、新 cell 计数和本轮终态。

任何新评分前 SHALL 冻结输入审计、全部帧索引/权重、掩码、44-cell 列表、阈值和 spec/源码/环境/权重身份。SHALL 从父矩阵重算历史结论，浮点统计与父 analysis 绝对差 ≤1e-6、全部离散判定相同，并确认仍为 ORACLE_OWN_AUDIO_UNRESOLVED；不一致为 BLOCKED。

#### Scenario: An old result is mistaken for a fresh inference

- **WHEN** 汇总父 V_ID/V_ORACLE 的四个 cell 和本轮 V_LINEAR 的两个 cell
- **THEN** 每条记录保留四个 `origin=parent_cached` 与两个 `origin=fresh`，总计 88 cached + 44 fresh；不得宣称 132 个新评分

### Requirement: Construct one deterministic pixel interpolation arm

SHALL 从父 V_ID 解码完整 F 帧 224×224 uint8 BGR24 的 X。音频原长为 L，采样率 16000 Hz，视频 25 fps。以 float64 重建父 s 数组与 u（s 的离散节点之间线性插值）：

```text
s[n] = n + 1920*sin(2*pi*n/(L-1)), n=0..L-1
s[0]=0; s[L-1]=L-1
t_i = 640*i, i=0..F-1
u_i = interp(t_i, arange(L), s)/640
j_i = floor(u_i); k_i = ceil(u_i); w_i = u_i-j_i
Y_i = (1-w_i)*float64(X[j_i]) + w_i*float64(X[k_i])
V_LINEAR[i] = uint8(floor(Y_i+0.5))
```

SHALL 验证 t_i 在 [0,L−1] 内，s 严格递增，u 非递减，j/k 均在 [0,F−1] 内；u 与父 `indices.q_float` 差 ≤1e-9。SHALL 对像素先转 float64 后运算，逐通道 nearest-half-up；不使用 np.rint、uint8 加权运算、gamma 校正、光流、锐化或其他平滑。u 为整数时 j=k，输出必须字节等于 X[j]；包含最后一帧的合法整数端点，不访问 j+1。越界为 BLOCKED，禁止 clip 索引或补尾。

SHALL 保存 u/j/k/w 与每帧源/输出像素 hash。使用父 FFV1/Matroska bgr0、pc/gbr 编码，完整解码后逐字节等于公式输出，F 帧、首 PTS=0、帧 PTS 与 i/25 偏差 ≤1 ms。父未变的其他颜色元数据亦保留。

SHALL 将父完整 N 和 W 分别 stream-copy 视频、PCM s16le mux 到同一 V_LINEAR 流；每个 mux 解码视频像素及 PCM 均须与指定来源一致。禁止截音频、重采样、AAC、-shortest、重新裁脸或按分数调 offset。本轮允许的混帧仅为以上显式公式。

#### Scenario: Half-frame interpolation and integer boundary

- **WHEN** u=2.5 且相邻通道值为 0/1，或 u=F−1
- **THEN** 前者输出 1；后者原样输出 X[F−1]，不进行越界取帧

### Requirement: Add exactly two fresh cells per record

SHALL 构造 22 个新视频流、44 个新 mux，并 fresh forward `V_LINEAR/N`、`V_LINEAR/W` 共 44 cell。SHALL 从新 mux 重新解码/提取特征；不能在父 embeddings 上插值来替代像素干预。

SHALL 沿用父 CPU SyncNet 契约：固定模型 SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`，batch=20、threads=4、vshift=15，官方预处理，float32 embeddings，距离 `sqrt(sum((visual-audio_or_zero+1e-6)^2))`，offset=15−column。保存完整 embeddings、[T,31] 矩阵、worker 结果及日志；分 cell 隔离工作目录。

SHALL 独立重建并核对父 `common_window_rows/plus_rows/minus_rows`、d/a 预测和所有窗口的真实支持，每条六个 cell 使用相同行；PLUS/MINUS 各至少 5 行。SHALL 同时报原生 global 和 common/PLUS/MINUS 指标，跨臂统计仅用 common。禁止按新峰、混帧比例或失败状态改掩码/删记录。

#### Scenario: A matrix or cell is missing

- **WHEN** 任一 cell 缺失、重复、非有限数、形状或掩码支持不合法
- **THEN** engineering_decision=BLOCKED、diagnostic_decision=null；不能在较小分母上给科学终态

### Requirement: Apply the original gates and a separate interpolation comparison

对任一掩码，SHALL 先逐列平均距离得到 31 点曲线 z，再取 `D=min(z)`、`C=median(z)-min(z)`、`off=15-argmin(z)`。峰清晰 SHALL 为最小/次小差严格 >0.010 且 |off|<15。基线 V_ID/N 的 PLUS/MINUS 均清晰、两段 offset 差 ≤1，至少 20/22。

SHALL 按以下两段检查判定，每条成功要求基线可解释、所涉峰均清晰且两段误差绝对值均 ≤1 帧；B/C_linear/O_linear 各至少 18/22：

| 检查 | 实测变化 | 沿用父掩码的预期 |
|---|---|---|
| B | off(V_ID/W)−off(V_ID/N) | mean(a(r)) |
| C_linear | off(V_LINEAR/N)−off(V_ID/N) | −mean(d(r)) |
| O_linear | off(V_LINEAR/W)−off(V_ID/N) | 0 |

SHALL 报告 common 指标的下列逐条差值及全 22 组均值/95% CI，D 一律定向为正数代表改善：

```text
own_C = C(V_LINEAR/W)-C(V_ID/N)
own_D = D(V_ID/N)-D(V_LINEAR/W)
damage_C = C(V_LINEAR/W)-C(V_LINEAR/N)
damage_D = D(V_LINEAR/N)-D(V_LINEAR/W)
gain_C = C(V_LINEAR/W)-C(V_ORACLE/W)
gain_D = D(V_ORACLE/W)-D(V_LINEAR/W)
```

own gate SHALL 为 own_C/own_D 两项 CI 下界均严格 >−0.10，且 common offset 与 V_ID/N 差 ≤1 的记录至少 20/22。damage gate SHALL 为 damage_C/damage_D 两项 CI 下界均严格 >0.10，且逐条二者同时 >0 至少 18/22。

SHALL 单独输出 `interpolation_improvement_supported`：仅当 gain_C/gain_D 的 CI 下界均严格 >0.10 才为 true；这是次要诊断，不能覆盖任一主 gate。SHALL 附报 common 曲线的 median/minimum 变化，帮助辨别 C 的变化来自背景距离还是最小距离；不增设事后门槛。

所有 CI SHALL 沿用排序 source groups、22 组有放回抽样、10,000 draws、NumPy default_rng/PCG64 seed=20260905，每项重置 seed、线性分位数 2.5/97.5；先算逐条配对差，再 bootstrap。SHALL 固定分母 22，完整精度判定，Sync-C 展示三位小数。

工程完整时 SHALL 依序给唯一科学终态，并报告所有可计算统计：

1. baseline 或 B/C_linear/O_linear 计数失败 → `LINEAR_TIMING_UNRESOLVED`。
2. own gate 失败 → `LINEAR_OWN_AUDIO_UNRESOLVED`。
3. damage gate 失败 → `LINEAR_DAMAGE_UNRESOLVED`。
4. 全部通过 → `LINEAR_ORACLE_CONTROL_SUPPORTED`。

#### Scenario: Scores improve but the original validity threshold still fails

- **WHEN** gain 两项 CI 下界 >0.10，但 own_C CI 下界 ≤−0.10，且 timing 通过
- **THEN** interpolation_improvement_supported=true、终态 LINEAR_OWN_AUDIO_UNRESOLVED；不得宣称控制修复

#### Scenario: Every linear-oracle gate passes

- **WHEN** 主 gates 全通过且独立 validator valid
- **THEN** 仅支持本 cohort 的线性插值 oracle；报告混帧/纹理与时间量化无法分离，不改写父 ORACLE_OWN_AUDIO_UNRESOLVED 或历史 CONTROL_FAILED

### Requirement: Validate independently and hand off the actual outcome

SHALL 写入新 `runs/wav2lip_oracle_frame_interpolation_<run_id>/`：protocol.json、input_audit.json、frames/manifest.json、media/manifest.json、scores/manifest.json、44 份 fresh 推理证据、analysis.json、final.json、result.md、validation.json。run-id 只接受字母/数字/下划线/连字符。SHALL 显式区分缓存引用与新产物，禁止覆盖父 runs 或已完成终态。

离线 validator SHALL 独立核对父固定 hashes/引用链、帧映射/输出像素/PCM/PTS，从全部使用的保存 embeddings 重算距离（max abs difference ≤0.0001），从矩阵重算掩码、曲线、bootstrap、计数、gain 和终态（统计容差 1e-6，离散判定完全一致）。不得调用 producer 的插值/统计/gate 函数或信任 pass 字段；允许共享纯 I/O/hash。不声称独立重复神经网络 forward。final 绑定证据，validation 绑定 final 文件 hash，避免循环哈希。

SHALL 测试整数/半整数坐标与 uint8 溢出防护、边界越界、像素错序/篡改、N/W 互换、缓存冒充 fresh、缺/重复 cell、矩阵/final 篡改、17/18 和 19/20 计数边界、gap=0.010、误差=1、CI 下界=−0.10/0.10、D 符号和终态优先级。合成测试只验证实现。

final SHALL 始终包含 `historical_scientific_decision=CONTROL_FAILED`、`parent_oracle_decision=ORACLE_OWN_AUDIO_UNRESOLVED`、`parent_own_audio_retested=false`、`new_generated_videos=0`、`bridge_executed=false`、`training_authorized=false`、`reference_conditioned_audio_head_spec_eligible=false`、`generalization_established=false`，并记录真实 fresh/cached 计数和 `oracle_own_audio_tested`。

下游 SHALL 先按 Startup Router/实验指令读取并更新 BM `Wav2Lip oracle frame interpolation 2026-09-07`，每次调用显式 project="tts-exp"。找不到时先用 2–3 个 text 变体搜索，再在 `Experiments` 创建 planned；执行时 running；实验完整且 validator valid 后 concluded，即使科学失败。工程阻塞时保留 running 并增加 `execution_status=blocked` 与具体 blocker，不能伪报 concluded。每次编辑前全文读取、保留并追加 changelog，frontmatter 和唯一 status observation 一致；写后读回检查文件目录和 permalink，无重复笔记。BM 不可用须报告未同步。

一次固定方案完成后 SHALL 停止。允许同协议/源码/输入 hash 完全一致的完整 cell resume；工程修复使用新正式 run。科学失败不改变幅度、插值核、阈值、音频、样本或 endpoint 重试。不运行其他 TFG/TTS、训练、bridge、MFA/DTW 或 sealed 数据。

#### Scenario: A complete negative experiment is delivered

- **WHEN** 44 fresh cell 与 88 cached 引用完整、独立验证通过、科学终态为 UNRESOLVED
- **THEN** tasks 的执行项可完成，BM 记 concluded 并保留负结论；最终给出 run 路径、门禁数字和 BM permalink
