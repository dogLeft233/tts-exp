## Context

动机见 [proposal.md](proposal.md)；所有样本、数值和判定以 [spec.md](specs/lrs3-local-timing-control-calibration/spec.md) 为唯一契约。

先读的历史材料：

- `tmp/handoff_lrs3_natural_to_tts_bridge.md`
- `openspec/changes/confirm-lrs3-natural-to-tts-bridge/design.md` 与其 delta spec
- `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/04_final/final.json`

现有代码入口在 `scripts/experiments/lrs3_natural_to_tts_bridge_confirmation/`。`render.py` 构造 Wav2Lip 命令和独立工作目录；`scoring.py` 提供 strict mux、cell 路由及评分；`stats.py` 提供 cluster bootstrap。当前 wrapper 使用 `constant_full_frame_fallback` geometry，多个 arm 并行渲染，评分也并行；这些是审计线索，本 proposal 未证明存在实现 bug。

## Goals / Non-Goals

**Goals:** 新建一个小实验入口，通过审计决定唯一分支；用已有基础函数完成媒体矩阵和可独立复算的结论。

**Non-Goals:** 建通用实验框架、复制整个历史包、重新选择 geometry/模型/评分器、在本轮完成 bridge 或训练路线。详细范围见 spec。

## Decisions

### 1. 审计先于控制选择

`00_audit/audit.json` 为每项检查保存 `sample_id/cell/check/status/evidence/expected/actual`，汇总为 spec 定义的三种审计状态。对旧分数的读取如实记为历史审计，不套用旧实验“零 score reads”断言。

从 manifest 指向的真实文件追踪到执行代码，重点检查模型实际加载音频的路径、渲染临时文件是否隔离、SyncNet 是否读取正确 mux。日志和 hash 可证明输入绑定，不能证明模型一定响应输入；natural source video 自带口型、全帧 box 可能削弱模型对驱动音频的响应，这些只作为机制假设记录，不能凭低分改 box。

`DEFECT_FOUND` 必须有违背原协议的具体证据及回归用例。修复后原控制子矩阵重跑即可；不同时引入新扰动，便于归因。`NO_DEFECT_FOUND` 才启用新控制；`INCONCLUSIVE` 报告缺失证据后结束。

### 2. 新控制只引入一次平滑时间变化

LOCAL_SWAP 交换了长语音片段，无法预先保证生成器在自身配对下仍有效。新控制使用 spec 的单周期正弦读取坐标，以固定幅度让局部提前和滞后，保留语序和长度。它只需线性插值，不引入声码器、对齐器或 TTS。

这是一个待检验的控制候选，120 ms 是本次设计预先选择的参数，没有通过数据搜索取得。global offset 无法精确抵消非恒定位移，但 SyncNet 仍可能不敏感；两件事不能混为一谈。

实作直接在 float64 的 PCM 整数幅值上插值，避免归一化/反归一化额外舍入；只进行 spec 规定的一次 nearest-even 舍入。恒零输入是合法的变换单元测试，不据此筛选真实 cohort。记录实际坐标 extrema/步长，禁止用 QC 决定重试或剔除。

### 3. 复用小函数，run root 显式传入

建议新目录仅按需要拆分 `audit.py`、`control.py`、`runner.py`、`validate.py` 和少量统计/清单辅助文件。复用历史的纯 hash/PCM/统计函数，以及经审计可用的 `render_driver`、`strict_mux`、`score_syncnet`。这些只是复用候选，调用前确认其副作用和配置依赖。

历史 `run_stage*`、`config.STAGES` 和 `write_blocked_terminal` 绑定旧 run root，不能直接作为新入口。新 runner 显式持有本次 paths 和 frozen config，不通过 monkeypatch 历史全局变量改路径。CLI 从干净目录执行 `--stage all` 必须按依赖顺序启动 audit，不能先尝试加载尚未生成的 manifests。

如需修复共用实现，最小修改并补原模块回归测试；记录旧/新代码 hash。只变更代码，不修改旧 run 或旧 OpenSpec。geometry、模型、SyncNet offset 搜索等科学设置保持原绑定，新的控制变换和命名空间由本 spec 明确覆盖。

### 4. 一套控制矩阵，三个科学 gate

两分支共用 spec 的四 cell 矩阵和 gate 公式，仅 `T` 的构造不同。全部新渲染/评分，减少跨 run 缓存与版本混杂。重复性通过也不能替代控制自身有效性。

每个 cell 的 ID 使用 `run_id + sample_id + video_arm + audio_arm` 的无歧义编码或 canonical hash；独立工作目录，记录完整命令与其 hash。审计时确认官方脚本实际读取的预处理路径，不根据 `--reference` 字符串推断输入。

对新 mux，验证帧数和视频时间轴保持 source render 不变（容器 time-base 量化容差 ≤1 ms），音频首样本从 0 开始，首个视频帧与音频的起点差 ≤1/25 s；禁止新增 global shift。视频相对音频的尾部差异按冻结 Wav2Lip 帧生成规则记录和验证，不用 `-shortest` 偷截 PCM。统计使用官方 scorer 正常 offset 搜索后的输出，不另调 offset 帮助 gate 通过。

### 5. 少量产物支持执行与复核

```text
runs/lrs3_local_timing_control_calibration_<run_id>/
  00_audit/audit.json
  01_protocol/protocol.json
  02_audio/audio_manifest.json
  03_videos/videos_manifest.json
  04_scores/scores_manifest.json
  05_final/final.json
  05_final/result.md
  05_final/validation.json
```

每个 manifest 引用其实际媒体/日志路径、hash 和上游 manifest hash；无需每个字段单独建文件。final 的自哈希复用现有 canonical 约定；validator 输出绑定 final 的文件 hash，final 不反向绑定 validation，避免循环哈希。

提供两个命令入口：

```bash
python -m scripts.experiments.lrs3_local_timing_control_calibration.runner --run-id <id> --stage all
python -m scripts.experiments.lrs3_local_timing_control_calibration.validate --run-root runs/lrs3_local_timing_control_calibration_<id>
```

runner 支持 `audit/protocol/audio/videos/scores/final/all` 阶段；`--run-id` 限定为字母、数字、下划线和连字符，路径只能在固定 runs 前缀下。纯测试使用临时目录并 stub 模型调用。validator 是单独入口，从原始日志和 manifests 复算矩阵、hash、统计与 gate，不调用模型；可复用纯 bootstrap，不能调用 producer 的终态判定函数来“验证自己”。

`BLOCKED` 也写 final，未运行统计用 null 和原因。`result.md` 是同一 run 的最终说明，不创建进度日志或额外知识库笔记。完整评分后的科学失败属于已完成实验，不能当作工程阻塞反复跑。

## Risks / Trade-offs

- 平滑时间重采样仍会改变局部音高和频谱 → 明确它不是纯 timing 因果隔离；自身有效性不过即控制失败，不扩大声学结论。
- 扰动可能太弱、落在静音处或模型保留 source 原口型 → 保留全部样本报告敏感性失败，不自动提高强度或更换 face。
- 复用已观察过的 22 records → 仅作控制校准，不称独立确认或泛化证据；不打开 sealed 数据。
- 旧函数暗含旧路径、默认参数或第一条日志匹配 → 用路径隔离、缺失 offset、歧义输出、错误缓存的回归测试锁住行为。

## Migration Plan

本 change 无部署或数据迁移。下游完成实现与测试后，按 audit 分支执行一次固定实验并独立验证，交付终态后结束。所有输出留在新 run；失败保留证据，重新设计需要另一个 change，不覆盖历史。
