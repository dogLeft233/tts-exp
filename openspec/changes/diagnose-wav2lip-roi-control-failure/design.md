## Context

父 change：`validate-wav2lip-face-roi-replacement`。正式入口为 `runs/wav2lip_face_roi_replacement_20260906_host_fix5/`，其 symlink/manifest 指向 fix1 的不可变 GPU 资产是合法来源。

历史 A/B/O、基线、重复性、replacement-damage 均通过，C 与 own-audio 未通过。A/B/O 通过不等于 C 必须通过；Sync-D 改善也不保证 Sync-C 改善。这两点正是本次要逐条解释的关系。

## Goals / Non-Goals

目标是一次完成「原结论能否独立复现、C 哪个条件失败、own-audio 的 C/D 为何不同向」的离线诊断。属于 `seen_fit_diagnostic`，不是新控制确认。

不新增音频/视频/评分、不调整 ROI/warp/阈值、不筛样本、不训练、不运行 bridge、不证明声学或模型机制的因果关系。没有发现实现偏差也是完整结果，不得为寻找通过条件反复扩展实验。

## Decisions

### 小实现与固定执行顺序

建议只用 `runner.py`（输入审计和编排）、`analysis.py`（独立复算与分解）、`validate.py`（验收）及 `__init__.py`，不建通用实验框架。

执行顺序：校验固定父输入 → 冻结本次协议 → 独立复算 → 对照历史 → 写报告 → 离线验证 → BM 记录。完整诊断即停止。

以下为待实现接口，不是已有命令：

```bash
/home/wjj/.venvs/wav2lip/bin/python -m scripts.experiments.wav2lip_roi_control_diagnostic.runner --run-id <id>
/home/wjj/.venvs/wav2lip/bin/python -m scripts.experiments.wav2lip_roi_control_diagnostic.validate --run-root runs/wav2lip_roi_control_diagnostic_<id>
```

仅 CPU；无需以 CUDA 不可用为由阻塞。后续若另立 GPU 实验，已知需要宿主命名空间执行，不能重装驱动解决普通 sandbox 的设备透传问题。

### 数据接口与复用边界

- 输入：父 `protocol.json` 的 `records/masks/predicted_frame_counts`，`scores/control/manifest.json` 的 `scores`，每项的 `matrix/matrix_sha256/media/audio` 及对应 sidecar。
- cell 主键固定为 `(sample_id, video_arm, audio_arm, repeat)`；禁止按列表位置 zip 不同 manifest。
- `control.json` 的 `per_record`、`gates` 和 score 行的 `local/common_global/reconstructed` 仅作为待对照值，不作为复算输入。
- 可复用历史 JSON、文件 hash、PCM 解码等 I/O helper。不能调用父 `analysis.analyze_control`、`cluster_bootstrap`、`scoring.local_evidence/reconstruct_global` 或 mask 构造函数来充当独立复算。
- validator 的核心公式、门禁与 bootstrap 不能调用本轮 producer 的统计/判定函数；允许共享 I/O。无需复制整个历史 GPU 流水线或验证未使用的 bridge 媒体。

### 只解释实际证据

C 的失败原因采用可并存的标志，不强行选一个“根因”：基线不满足、峰在边界、峰间距不足、响应误差 >1 帧。报告两段实际/预期 offset 和有符号残差；只有峰清晰时才解释响应方向和幅度。

own-audio 的分解是精确代数恒等式，不是新的统计门禁，也不是 acoustic mismatch 的因果证据。报告完整 22 条，不能移除负差值记录来证明非劣性。

## Risks / Trade-offs

本轮能确定失败发生在哪个判定环节，不能单凭矩阵区分生成器、插值声学变化和 SyncNet 表征的全部机制。报告若无法确定因果来源，应写“未确定”，并提出一个最小后续干预及其要区分的假设。

## Handoff

下游 agent 按 `tasks.md` 顺序实现，以 capability `spec.md` 为验收依据。发现缺资产或规范与历史值冲突时保留证据；不自行换 run、放宽阈值或执行修复实验。任务完成以诊断产物和验证为准，CLI 退出 0 不代表控制通过。
