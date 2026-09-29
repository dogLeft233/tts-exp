## Purpose

通过固定三条历史样本、独立评分重放以及真实音视频同步交换对照，验证 LOCAL_SWAP 反向评分是否可复现，并区分历史数据问题、评估预处理问题与生成视频未充分跟随驱动的可能性，为下游提供可观看和复核的诊断证据。

## ADDED Requirements

### Requirement: Fixed inputs and read-only historical audit

实验 SHALL 只读审计原 bridge confirmation 的全部 22 条 LOCAL_SWAP 记录，新评分仅使用 design.md 固定的前三条。审计 MUST 重新核对实际 PCM 排列、mux 视频码流、生成音频绑定、原始日志与统计，而非仅信任清单标记。历史 run MUST 保持不变。

#### Scenario: History agrees with the summary

- **WHEN** 44 份日志及媒体身份均与对应 cell 相符
- **THEN** 报告确认历史记录可信，同时将尚未进行的新前向复现单独标注。

#### Scenario: Missing or inconsistent source

- **WHEN** 指定文件缺失、hash 不一致或音视频身份不明
- **THEN** 记录具体失败项，标记受影响分支阻断，不替换样本或静默修复输入。

### Requirement: Bounded replay matrix

实验 SHALL 按 design.md 对三条记录执行 A 两个历史端到端 cell、B 四个真实配对 cell、C 两个固定生成视频裁剪 cell，总计 24 个新评分 cell。实验 MUST 使用新工作目录和缓存，保存运行时与代码绑定，禁止按结果重试、生成 Wav2Lip/TTS 视频或重测 bridge。

#### Scenario: Replay produces the reverse preference

- **WHEN** A 的三条记录均为配 N 的 C 更高且 D 更低
- **THEN** 报告子集方向复现；按 C/D 差 ≤0.100 和 offset 差 ≤1 帧另外报告数值近似复现，不声称全量重新推理。

### Requirement: Known synchronous and asynchronous real pairs

实验 SHALL 使用固定真实视频裁剪和自然 PCM 的共同支持，构造帧与样本边界精确对应的 `R/N0`、`R/S0`、`Rs/N0`、`Rs/S0`。Rs 与 S0 MUST 按相同的 A-C-B-D 排列生成，保留逐帧与逐样本映射；N0/S0 MUST 与用于 A/C 的旧 N/S 清楚区分。轨迹选择 MUST 在评分前冻结且不依据分数。

#### Scenario: Both modalities are swapped

- **WHEN** 真实视频和音频使用相同边界一起交换
- **THEN** 段内部保留原有同步关系，报告其相对错配条件的距离和分数，而非预先假定模型一定通过。

#### Scenario: Timing support is invalid

- **WHEN** 起点、25 fps、共同支持、轨迹覆盖或最小窗口数无法满足 design.md
- **THEN** 输出对应输入或支持不足状态，不补齐、变速或事后对齐来制造通过结果。

### Requirement: Fixed scoring inputs and transparent local evidence

实验 SHALL 在 C 中固定同一组生成视频裁剪帧和绝对时间支持，分别配旧 N/S 的准确切片；B/C MUST 使用官方 SyncNet 前向并完成数值 parity。每个 cell SHALL 输出官方 C/D/offset、完整距离矩阵、时间/offset 轴、输入 hash 和窗口支持。

局部统计 MUST 使用相同窗口集合，剔除跨段感受野，按 design.md 分别计算两个中间段的 offset=0 距离偏好；不得将局部 D0 标成官方 Sync-D，或将不同端点的绝对分数混合汇总。

#### Scenario: Global search obscures a local response

- **WHEN** 全局最佳 offset 或聚合分数与局部固定时刻距离表现不同
- **THEN** 同时报告两种结果及有效支持，不只选择符合预期的一项。

### Requirement: Reviewable diagnosis with limited claims

交接 SHALL 包含 design.md 所列输入清单、历史审计、24 行评分表、局部证据、播放对照、最终机器可读状态和报告。判定 MUST 分开描述记录可信度、方向/数值复现、真实错配敏感性、生成视频音频偏好和原因定位；混合结果标为不确定。人工观看状态 MUST 如实记录。

#### Scenario: Known mismatch is detected but generated video favors natural

- **WHEN** B 在三条的两个中间段均检测到正确配对更好，C 仍偏向 natural
- **THEN** 报告评分能识别本对照、生成视频更匹配 natural，允许提出驱动转移不足的解释，但不声称已定位唯一生成器缺陷。

#### Scenario: Real-pair control fails

- **WHEN** 输入及 parity 验证通过、真实基线同步可信，但 B 未呈现预期偏好
- **THEN** 报告当前端点敏感性存疑或结果混合，不据此否定全部 SyncNet 结果或宣布 bridge 无效。
