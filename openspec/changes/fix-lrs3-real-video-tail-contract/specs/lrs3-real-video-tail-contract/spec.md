## Purpose

为固定 LRS3 真实视频局部时间诊断提供可复核的尾部时长契约，使完整原始音视频在有界尾差下参与实验，并保留来源、时间支持范围及旧阻塞证据，避免工程修订被误报为科学结论。

## ADDED Requirements

### Requirement: Apply a versioned bounded tail contract

系统 MUST 将本 spec 与父 change `diagnose-lrs3-real-video-local-timing` 的 `specs/lrs3-real-video-local-timing-diagnostic/spec.md` 组合使用。仅将父要求 `Freeze the existing cohort and native timeline` 中“尾部长度差至多一帧”替换为以下规则，其余要求继续有效。

设实际视频帧数为 F，自然 PCM 样本数为 N，25 fps / 16000 Hz 下每帧 640 samples，定义 `d=N-640*F`。新协议 `bounded_audio_tail_v2` MUST 接受且仅接受 `-640 <= d <= 1280`（音频最多短 40 ms、长 80 ms），使用整数比较。固定原 22 records / 22 source groups、原顺序和源 hash；不得按记录设置例外或从新分数选择阈值。超过旧上界的可接受记录 SHALL 标记 `extended_audio_tail`，原因标记 `unattributed`，不声称已证明 padding。

系统 MUST 验证同源裁切/音频提取的历史来源，音频样本 0 对应视频起点的差不超过 1 ms；仅 WAV 自带零起点不足以证明配对。逐帧 PTS 必须严格递增，满足 `abs((PTS[i]-PTS[0])-i/25)<=1 ms`；缺失时间或来源证据不得默认为通过。

#### Scenario: Accept the known tail mismatch without altering media

- **WHEN** 同源及时间轴审计有效，d 为 768 或 896 samples（1.2 或 1.4 帧）
- **THEN** 尾差检查通过并记录扩展尾差，保持全部输入 PCM 与视频帧，继续父实验其他检查

#### Scenario: Enforce exact limits and pairing

- **WHEN** d 为 -640 或 1280 且其他检查有效
- **THEN** 尾差检查通过；d 为 -641 或 1281、起点差超过 1 ms、PTS 不连续或来源不明时 MUST BLOCKED，scientific_decision=null

### Requirement: Audit the complete cohort before scoring

系统 MUST 在裁脸和评分前输出 `input_audit.json`，按历史顺序包含全部 22 条的源路径/文件 hash、F、N、d、尾差毫秒、原始 PTS 证据与来源证据引用、起点差、各项检查状态及错误。无法读取的字段为 null 并记原因；不能只报告首个失败。审计 MUST 记录 cohort hash、协议版本、自身 hash，以及共同支持区间 `[0,min(N,640*F))`（samples）；任一失败时仍交付完整审计和 BLOCKED 终态。

三臂及 scorer 实际提取的音频 MUST 与完整输入 PCM 字节一致；禁止裁剪、补齐、时间拉伸、重采样或全局对齐，也不得为匹配音频而延长视频。局部 PLUS/MINUS 仍使用父协议相同的内部行和边缘排除规则，全部 visual 窗口及 31 个 audio lag 窗口必须在共同有效支持范围内，每段至少 5 行。全局矩阵与官方全局 C/D 的重建范围和 parity 契约保持不变。

#### Scenario: Multiple invalid inputs

- **WHEN** 一条来源不明，另一条尾差超界
- **THEN** 审计保留两条失败和其余 20 条检查结果，停止裁脸和评分，保留 22 条分母

#### Scenario: Preserve audio and interior evidence

- **WHEN** 接受了音频长尾的记录并生成三臂
- **THEN** mux 与 scorer 音频都保留全部 N samples；局部行不得利用无视频支持的长尾或评分器边缘 padding，不满足支持范围时 BLOCKED

### Requirement: Validate revised runs without rewriting history

新 run MUST 冻结 `protocol_revision=bounded_audio_tail_v2`、上下界、父 spec 与本 spec 的 hash、父阻塞 run `lrs3_real_video_local_timing_20260905_v2` 的 final hash、input audit hash。final（包括 prepare 失败）MUST 绑定协议修订和审计 hash。原 v1/v2 终态及其历史输入只读；恢复仅限版本、代码和输入 hash 匹配的未终结 run，禁止复用旧版本 cell。

独立 validator MUST 从源媒体重算整数尾差、PTS、来源和支持范围，并与审计逐项比较；不得直接调用 producer 的通过判定函数或只信 final 的状态。被修改的审计、缺失的记录或引用 hash 不匹配必须验证失败。旧版 final 可按旧 schema 验证，但不得宣称满足新输入契约。

#### Scenario: Run and report under the amendment

- **WHEN** 修复及测试完成并开始一次新 run
- **THEN** 执行原 22×3 诊断和独立验证，沿用重复性、20/22 基线及 18/22 恢复判据；报告并更新同一 BM 实验笔记，区分输入契约通过、后续工程阻塞和科学终态，不按结果调参重试

#### Scenario: Reject tampered or incompatible artifacts

- **WHEN** 审计中的 N、PTS、源 hash 被改动，或尝试将旧协议产物用于新 run
- **THEN** 验证/恢复失败，原终态保持原样，不能仅凭 final 标记 valid
