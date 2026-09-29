## Purpose

在固定 LRS3 真实音视频上保持自然音频逐样本不变，仅改变视频帧读取时间，检验冻结 SyncNet 的前向评分是否识别已知局部错位。该能力用于缩小历史控制失败的来源，结果仅为固定数据诊断，不是 TTS 收益确认。

## ADDED Requirements

### Requirement: Freeze the existing cohort and native timeline

系统 MUST 使用 `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json` 的原顺序 22 records / 22 source groups，文件 SHA-256 为 `b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b`。每条只读取其 `face_video` 原始真实视频与 `natural_audio`，校验记录内 hash 和同源时间关系；不使用生成视频充当真实视频。

系统 MUST 在新评分前冻结样本、媒体 hash、原始 PTS 到帧/样本的映射、裁脸轨迹、运行代码/模型/环境及本 spec 参数。输入要求为 25 fps 视频、16 kHz mono PCM16 音频。保留完整自然 PCM，视频和音频尾部长度差至多一帧；起点差至多 1 ms。不能通过裁音频、补音频、估计新全局 offset 或重新选样本满足契约。来源、原生配对或时间基准无法确认时写 `BLOCKED`。

#### Scenario: Missing or mismatched input

- **WHEN** 任一记录的源 hash、真实视频来源或时间映射无法验证
- **THEN** 系统报告该记录及原因，停止媒体评分，不替补记录或缩小 22 条分母

### Requirement: Change only the video sampling coordinates

系统 SHALL 产生 `REAL`、`REAL_REPEAT`、`VIDEO_WARP_120` 三臂，共 66 个媒体/评分 cell。REAL_REPEAT 从同一冻结输入独立执行编码与评分，不复制 REAL 的媒体或分数。

设真实视频有 F 帧、输出索引 n 从 0 到 F−1；REAL 与 REAL_REPEAT 读取 j[n]=n。扰动臂 MUST 使用 float64 计算 `s[n]=n+3*sin(2*pi*n/(F-1))`，强制两个端点分别为 0、F−1，再以 nearest-even 舍入为整数 j[n]。实际映射必须范围合法、单调不减、首尾固定、同时达到 +3 和 −3 帧位移。输出保留 F 帧及 25 fps PTS，直接选择原帧，允许重复/跳帧；不做像素混合、光流插帧或音频变形。

三臂 MUST 使用同一原视频预先计算的完整人脸 crop 序列，再按 j 映射 crop 帧；不在各臂重新检测或选择人脸。无法获得全长有效单人轨迹时 BLOCKED，禁止以全帧 box 代替。编码设置一致；每个 mux 解码后的 PCM 必须与输入自然 PCM 字节一致，视频帧数与时间轴必须吻合冻结映射。

#### Scenario: The video leads locally

- **WHEN** 一个五帧窗口的每帧均满足 j[n]−n=+3
- **THEN** 该窗口使用原视频晚 120 ms 的画面，音频保持原时间，保存实际五帧来源索引供独立检查

#### Scenario: Audio or geometry changes

- **WHEN** 任一输出 PCM 不同或某臂使用了不同的检测轨迹
- **THEN** 系统将实验标为 BLOCKED，不解释科学结果

### Requirement: Preserve global and local distance evidence

系统 MUST 冻结官方 SyncNet V2 权重（SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`）、官方预处理及 `vshift=15`，导出每 cell 的未舍入 `[window,31]` distance matrix、窗口绝对帧索引及官方全局 Sync-C、Sync-D、offset。列 k 对应官方 offset `15-k`；均值曲线的最小值为 D，median(curve)−D 为 C。矩阵重建全局结果与官方日志三位小数须在 0.001 内一致。缺列、非有限值、窗口错位或日志歧义均 BLOCKED。

系统 SHALL 额外计算两条局部曲线：PLUS 使用实际映射在整个五帧窗口内均为 +3 的窗口，MINUS 使用均为 −3 的窗口。掩码仅由帧映射确定，三臂使用完全相同的行；只保留所有 31 个 audio lag 窗口及 visual 窗口均在真实输入支持范围内的行，排除评分器的边缘 padding。每段至少 5 行，否则 BLOCKED，不改变分段条件。每段曲线为对应行的均值。

#### Scenario: Global offset search hides local changes

- **WHEN** 全局 C/D 变化很小，但 PLUS 和 MINUS 的曲线最小位置向相反方向移动
- **THEN** 系统保留局部证据，以后述局部恢复规则判定，不要求全局 C/D 同时下降

### Requirement: Use a fixed diagnostic decision rule

系统 MUST 按以下顺序判定，不用新结果调整参数：

1. 工程完整性：上述输入/媒体/曲线与 66 cells 全部有效，否则 `BLOCKED`，scientific_decision=null。
2. 重复性：REAL 与 REAL_REPEAT 的完整矩阵最大绝对差 ≤0.001，且全局及两个局部 offset 全部相同，22/22 满足才通过；否则 `REPEATABILITY_FAILED`。
3. 基线可解释性：局部曲线最小与次小值之差 >0.010 且最小位置不在 ±15 边界，称为清晰峰。REAL 的两个局部峰均清晰、彼此 offset 相差 ≤1 帧的记录至少 20/22，否则 `BASELINE_INCONCLUSIVE`。
4. 局部恢复：对每条记录计算 `delta_plus=offset(WARP,PLUS)-offset(REAL,PLUS)`，MINUS 同理。记录成功要求基线可解释、WARP 两个局部峰清晰、`abs(delta_plus+3)≤1` 且 `abs(delta_minus-3)≤1`。成功至少 18/22 为 `LOCAL_TIMING_DETECTED`，否则 `LOCAL_TIMING_NOT_ESTABLISHED`。基线不清晰的记录仍保留在分母，不能算成功。

所有有效完整运行 SHALL 报告逐记录 offset/峰间距/恢复误差、成功数、全局 `C(REAL)-C(WARP)` 与 `D(WARP)-D(REAL)` 的均值及 95% 配对 bootstrap CI。Bootstrap 按 source_group 排序，22 个组有放回抽取 22 个组，10,000 次，NumPy default_rng/PCG64 seed=20260905，百分位 2.5/97.5；全局分数与 CI 仅描述，不是额外 gate。所有统计保留未舍入值，显示 Sync-C 三位小数。

#### Scenario: Correct local recovery without global damage

- **WHEN** 前三个检查通过且 18/22 条满足两段恢复，但全局 C 的差值 CI 跨零
- **THEN** 系统输出 LOCAL_TIMING_DETECTED，同时明确全局 summary 未显示可靠损伤

#### Scenario: Baseline or repeatability is insufficient

- **WHEN** 重复性失败或可解释基线不足 20 条
- **THEN** 系统按上述优先级结束，不把后续局部恢复低计数解释为 SyncNet 缺乏时间信息

### Requirement: Provide a blinded review package without fabricating review

系统 SHALL 为 cohort 原顺序前 4 条生成 REAL/WARP 的匿名 A/B 对照包，随机顺序由独立的 seed=20260906 冻结，映射单独保存，不在展示文件名或画面泄露条件。使用同一音量与播放设置，表单包含“哪段更同步 / 无法区分”和“是否存在明显卡顿”。盲看仅辅助，不参与数值 gate 或样本选择。

#### Scenario: No reviewer is available

- **WHEN** 数值实验完成但无人提交盲看表单
- **THEN** 系统记录 `review_status=PENDING`，交付包并完成数值报告，不编造主观结果或阻塞数值终态

### Requirement: Deliver reproducible artifacts and bounded conclusions

系统 MUST 在独立 run 中交付冻结协议、媒体映射与 hash、66 cells 的原始曲线/日志、盲看包、`final.json`、`result.md` 及离线 `validation.json`。validator 从实际媒体、矩阵及协议独立重算上述 gate，不能直接复用 producer 的判定函数。恢复运行只复用输入/配置/代码/产物 hash 全部匹配的已完成 cell，失败或已完成终态不得被另一种终态覆盖。

LOCAL_TIMING_DETECTED SHALL 仅支持“当前真实视频与离散视频扰动下存在前向局部时间敏感性”；后续可另立实验区分历史音频扰动与 Wav2Lip 响应。其余科学终态须说明未建立何种证据。报告 MUST 保留帧重复/跳帧与全局视觉运动变化的混杂限制，以及 fit-only、非梯度、非替换收益的边界。

#### Scenario: The diagnostic succeeds

- **WHEN** 离线 validator 验证 LOCAL_TIMING_DETECTED
- **THEN** 系统交付诊断结论后结束，不自动重测 bridge、运行生成器、训练或扩大到 heldout/sealed 数据
