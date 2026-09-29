## Purpose

用已知像素重定时构造，检验现有 ROI 生成域上的局部 timing 控制和 own-audio 有效性，为后续生成响应研究提供参照。

## ADDED Requirements

### Requirement: Freeze the full historical cohort and evidence

实验 SHALL 固定以下入口文件 SHA-256，并沿其 manifest 校验所有实际使用的 G_N 评分视频流、N/W 音频、矩阵和时间掩码；文件 hash 与 JSON 内部自哈希分别验证：

| 输入 | 文件 SHA-256 |
|---|---|
| `runs/wav2lip_face_roi_replacement_20260906_host_fix5/final.json` | `76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563` |
| 同目录 `protocol.json` | `835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4` |
| 同目录 `control.json` | `6944cc299f30f29a2c6f6a4f41db024014ec09aa583ca2bbc15c12133c8abce5` |
| 同目录 `media/control/manifest.json` | `977b2f48afe03380c710daa57515068c62d345245495bf37c6d45ada509061cf` |
| 同目录 `scores/control/manifest.json` | `60950f9a26abcb37d9113541e23acd038feb8f21b827f07fb3c2f07570aa3799` |
| `runs/wav2lip_roi_peak_recheck_20260906_review2/final.json` | `478873a531db61823159b499368cc31ef9ec9f72a6fc7b80b857e42d4b37e12c` |
| 同目录 `validation.json` | `883a7c6fe4596f422030dbdc996c95a7f5bb8a66c5f4b4827c68a4d9db1c86f0` |

SHALL 保持父 protocol.records 的全部 22 records / 22 source groups 与顺序，ordered-ID hash `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`，分类 `seen_fit_diagnostic`。不能只选上一轮 10 条或替换失败记录。

SHALL 全文读取并冻结以下继承 spec：`diagnose-lrs3-wav2lip-timing-transfer/specs/lrs3-wav2lip-timing-transfer-diagnostic/spec.md`（s、逆映射、共同窗口、PLUS/MINUS、峰定义）；`validate-wav2lip-face-roi-replacement/specs/wav2lip-face-roi-replacement/spec.md`（父媒体/控制）；`recheck-wav2lip-roi-local-peaks/specs/wav2lip-roi-local-peak-recheck/spec.md`（SyncNet 前向/距离）。以上路径均相对 `openspec/changes/`。本 spec 的显式覆盖优先，其余沿用。

任何新评分前 SHALL 冻结全部 22 条的输入审计、q 数组、掩码、阈值、源码/spec/环境/权重 hashes 和 88-cell 矩阵。缺失或身份不匹配输出 BLOCKED，不换父 run。

#### Scenario: Parent manifests point to an earlier immutable media run

- **WHEN** fix5 manifest 指向 fix1 的实际媒体且 hash 校验一致
- **THEN** 允许只读使用该文件并记录完整绑定，不复制成伪造的新生成资产

### Requirement: Construct an identity arm and a known video retiming arm

SHALL 从父 `media/control/manifest.json` 的 `rows[].streams.G_N` 解码 F 帧 224×224 BGR24 像素 X。使用音频原长 L（16 kHz mono PCM16），按继承公式以 float64 重建 `s[n]=n+1920*sin(2*pi*n/(L-1))`，端点强制 0/L−1，并验证 s 严格递增及 W 的原始 PCM 精确复建一致。W 复建仅审计，不替换历史音频。

SHALL 在任何评分前按下式构造逐帧索引，其中 i=0…F−1，s 在实数位置使用离散 s 的分段线性插值：

```text
t_i = 640*i
q_ID[i] = i
q_ORACLE[i] = floor(s(t_i)/640 + 0.5)
V_ID[i] = X[i]
V_ORACLE[i] = X[q_ORACLE[i]]
```

q SHALL 非递减、全部在 [0,F−1] 内、最大量化误差 `abs(q−s(t_i)/640) <= 0.5+1e-9`。不得 clip 越界索引；越界为 BLOCKED。由 q 产生的重复/跳帧是本轮明确允许的干预，不能额外循环/补尾/改长度。两臂 SHALL 保持 F 帧、25 fps、首 PTS=0、PTS 与 i/25 偏差 ≤1 ms。

两臂 SHALL 使用同一 FFV1/Matroska、bgr0、color_range=pc、color_space=gbr 无损编码配置，并保留父流其余颜色元数据；全部父流须先核对该格式，不能静默做颜色空间转换。解码像素逐帧与 X[q] 字节一致。保存源帧及输出帧 hashes、q 和量化误差；不重新裁脸/检测、不插帧、不混帧、不用 embeddings 重排代替像素重排。

SHALL 分别将完整 N/W PCM mux 到同一臂的视频流，视频 stream-copy、音频 PCM s16le，逐 cell 校验解码 PCM 与指定音频及视频流身份。禁止 AAC、`-shortest`、按分数对齐 offset 或改变原音频长度。

#### Scenario: An identity encode or oracle frame differs from its indexed source

- **WHEN** V_ID[i]≠X[i] 或 V_ORACLE[i]≠X[q[i]] 的解码像素存在任一字节差异
- **THEN** 工程 BLOCKED，记录 sample/frame，不进入科学解释

### Requirement: Score one complete crossed matrix with the validated endpoint

SHALL 新评分 `V_ID/N, V_ID/W, V_ORACLE/N, V_ORACLE/W`，每条四 cell，共 88 个；新视频流 44、新 mux 88、新 Wav2Lip 生成视频 0。每个 cell 从本轮 mux 重新解码并 forward，不把历史 embedding/矩阵作为新推理输入。

SHALL 使用现有 SyncNet Python 环境、CPU、batch=20、threads=4、官方模型结构/预处理、vshift=15、权重 SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。允许复用已复核 `SyncNetScorer`，保留 float32 visual/audio embeddings、完整 `[T,31]` 距离矩阵及执行绑定；epsilon 加在差值内部，offset=15−列号。评分进程/工作目录按 cell 隔离。

SHALL 从输入独立重算并核对父 masks 的 `common_window_rows/plus_rows/minus_rows`，全部 cell 使用同一组行、原音频 L 定义的 s 和原预期值；两段各至少 5 行，行的全部视频/音频窗口均有真实支持。不得随 q 缩小统计分母或按峰择窗。每个 cell 同时报告原生全局和共同窗口 C/D/offset，跨臂只比较共同窗口指标。

SHALL 用父非 repeat 的 G_N/N、G_N/W 对照本轮 V_ID/N、V_ID/W：44 对矩阵 max abs difference ≤0.001，global/common/两段 offset 和峰清晰判定一致。旧矩阵只用作这项复现检查及历史描述，保存新旧来源标志。

#### Scenario: New identity baseline does not reproduce the historical scores

- **WHEN** 媒体身份有效、88 cell 完整，但任一恒等臂对照不符合复现规则
- **THEN** 输出 BASELINE_NOT_REPRODUCED 并列出具体 cell/矩阵差/峰，不把 oracle 数值用于机制结论

### Requirement: Apply fixed local and paired validity rules

SHALL 按继承峰规则：最小/次小值差严格 >0.010、argmin 不在 ±15 才清晰；V_ID/N 两段清晰且彼此 offset 差 ≤1 为基线可解释，至少 20/22。对下表逐条判断：基线可解释、所涉两段峰清晰、两段实测与预期误差绝对值均 ≤1 帧才算成功；B/C/O 各至少 18/22。mean 只在对应父 PLUS/MINUS 行内计算。

| 检查 | 实测变化 | 预期变化 |
|---|---|---|
| B | off(V_ID/W)−off(V_ID/N) | mean(a(r)) |
| C_oracle | off(V_ORACLE/N)−off(V_ID/N) | −mean(d(r)) |
| O_oracle | off(V_ORACLE/W)−off(V_ID/N) | 0 |

SHALL 保留父连续映射的 d/a 预测；q 的离散误差仅描述，不替换 gate 预期、不扩大 1 帧容差。

SHALL 同时报告并判定以下共同窗口指标：

```text
own_C = C(V_ORACLE/W) − C(V_ID/N)
own_D = D(V_ID/N) − D(V_ORACLE/W)
damage_C = C(V_ORACLE/W) − C(V_ORACLE/N)
damage_D = D(V_ORACLE/N) − D(V_ORACLE/W)
```

own gate SHALL 为 own_C/own_D 的 95% CI 下界均严格 >−0.10，且 V_ORACLE/W 与 V_ID/N 的共同 offset 差 ≤1 的记录至少 20/22。damage gate SHALL 为 damage_C/damage_D 的 CI 下界均严格 >0.10，且二者同时 >0 的记录至少 18/22。

全部 CI SHALL 使用排序 source groups、22 组有放回抽样、10,000 draws、NumPy default_rng/PCG64 seed=20260905、每项重置 seed、线性分位数 2.5/97.5。保留完整精度判定，Sync-C 展示三位小数。分母固定 22，不清晰峰计失败，不删除记录。

科学终态 SHALL 依序唯一确定；即使前项失败，仍报告全部可计算统计：

1. 恒等臂复现失败 → `BASELINE_NOT_REPRODUCED`。
2. 基线 <20/22 或 B/C_oracle/O_oracle 任一 <18/22 → `ORACLE_TIMING_UNRESOLVED`，附逐项计数。
3. own gate 失败 → `ORACLE_OWN_AUDIO_UNRESOLVED`。
4. damage gate 失败 → `ORACLE_DAMAGE_UNRESOLVED`。
5. 全部通过 → `ORACLE_CONTROL_SUPPORTED`。

工程不完整 SHALL 为 `engineering_decision=BLOCKED, diagnostic_decision=null`，不能伪装成科学失败。工程完整为 GO；任何科学终态都不改写历史 `CONTROL_FAILED`。

#### Scenario: Oracle timing passes but own audio does not

- **WHEN** 基线复现、基线可解释性和 B/C/O 全通过，但 own_C CI 下界 ≤−0.10
- **THEN** 输出 ORACLE_OWN_AUDIO_UNRESOLVED；说明当前已知重定时构造的配对有效性仍未通过，不能用局部通过代替完整控制

#### Scenario: All oracle gates pass

- **WHEN** ORACLE_CONTROL_SUPPORTED 且独立 validator valid
- **THEN** 仅支持当前 22 条生成画面的已知重定时参照有效，建议后续研究实际生成响应路径；不得称父 G_W 控制已通过或 replacement 效应成立

### Requirement: Deliver a bounded experiment and maintain memory

SHALL 写入新 `runs/wav2lip_roi_retiming_oracle_<run_id>/`：`protocol.json`、`input_audit.json`、帧索引/像素证据、媒体和评分 manifests、88 份 embeddings/矩阵/日志、`analysis.json`、`final.json`、中文 `result.md`、`validation.json`。run-id 限定字母/数字/下划线/连字符；父 runs 只读。只允许输入/源码/协议/输出身份完全一致的完整 cell resume；工程修复后使用新正式 run，终态不覆盖，科学失败不调幅度、阈值、编码或选样重试。

离线 validator SHALL 独立从实际媒体重建 q/像素/PCM/PTS 绑定，从保存的 embeddings 独立复算距离（max abs difference ≤0.0001），从矩阵复算掩码、曲线、bootstrap、计数和终态，不调用 producer 的映射/统计/gate 函数或信任 pass 标志。允许共享纯 I/O/hash。final 绑定证据，validation 绑定 final 文件 hash，禁止循环哈希。validator 的独立范围是像素/embedding 后续链路，不声称独立复做神经网络前向。

SHALL 测试已知 ±3 帧索引与 offset 符号、nearest-half-up 舍入/单调/越界、像素错序、音轨互换、缺 cell、掩码/矩阵/final 篡改、17/18 与 19/20 边界、gap=0.010、误差=1、CI 下界=−0.10/0.10 和终态优先级。合成测试不得作为真实实验结果。

final SHALL 始终包含 `historical_scientific_decision=CONTROL_FAILED`、`parent_own_audio_retested=false`、`oracle_own_audio_tested`（按实际是否执行）、`bridge_executed=false`、`training_authorized=false`、`reference_conditioned_audio_head_spec_eligible=false`、`generalization_established=false`。不运行 TFG/TTS、训练、bridge、MFA/DTW、其他模型或 sealed 数据。

下游 SHALL 在执行前遵循 BM Startup Router/实验指令，显式 project="tts-exp"，用 2–3 个 text 查询搜索新实验名/slug，创建或更新同一 `Experiments/` 笔记为 planned；结束时全文读取后更新到真实 concluded/blocked，frontmatter 与唯一 status observation 一致，记录数字、局限、父关联、报告路径。写后读回核对目录大小写与无重复 permalink；BM 不可用时明确报告未同步，不声称已经保存。

#### Scenario: The scientific result is negative but all artifacts validate

- **WHEN** 88 cell 完整、validator valid 且终态为任一 UNRESOLVED
- **THEN** 实验记为 concluded，保留负面结果并结束；不为取得通过而进入新的参数搜索
