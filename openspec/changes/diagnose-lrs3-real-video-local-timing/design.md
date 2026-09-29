## Context

动机见 [proposal.md](proposal.md)；参数、公式与终态以 [spec.md](specs/lrs3-real-video-local-timing-diagnostic/spec.md) 为唯一真相源。下游先读这两份文档，再执行 tasks。

历史依据：

- BM `tts-exp/experiments/lrs3-局部时间控制校准实验结果`：工程审计通过，控制自身有效性及替换敏感性失败。
- BM `tts-exp/experiments/lrs3-natural-to-tts-bridge-confirmation-result`：bridge 未建立 useful-effect gain。
- BM `tts-exp/tasks/局部-timing-与-acoustic-mismatch-分阶段诊断`：global shift 可恢复不等于 local timing 可恢复。
- 本地 `runs/lrs3_local_timing_control_calibration_20260905_v5/05_final/final.json` 与历史 cohort 提供输入来源，不复用其评分。

## Goals / Non-Goals

**Goals:** 一个固定三臂实验，证明或未能证明局部 offset 的已知方向响应；离线可复核。

**Non-Goals:** 通用实验框架、另训 critic、增加控制强度搜索，以及在本 change 内追查所有历史失败来源。

## Decisions

### 1. 在冻结 crop 后改变帧读取位置

原始 `face_video` 是真实 LRS3 视频，不是静态脸。对原视频运行一次官方检测/跟踪/crop，固定单个完整说话人轨迹及预处理参数，在评分前写入 protocol。要求轨迹覆盖完整视频；多条完整轨迹有歧义或覆盖不足就报告 BLOCKED，不按得分选轨迹、不增加全帧 fallback。

REAL、REPEAT、WARP 都从同一冻结 crop 帧序列构造，各自独立编码为 FFV1/PCM16 Matroska，保留无损解码帧和自然 PCM。帧位移在 crop 后应用，使不同臂没有独立跟踪/裁脸选择。保留原始全画面与同一 j 映射供辅助盲看；盲看包可使用 crop 媒体，明确其视野。

选择离散原帧读取是为了保留单帧像素；像素插值和光流会引入新的合成误差。代价是重复/跳帧，因此本实验不是无混杂的“纯 timing”因果证明。

### 2. 直接导出官方距离矩阵

现有 `third_party/syncnet_python/SyncNetInstance.py::evaluate` 已返回 `(offset, conf, dists_npy)`；矩阵形状为窗口数×31。新建轻量 wrapper 接收已裁脸媒体，直接调用该函数，不再调用会重新检测/裁脸的完整 pipeline。独立 tmp/reference 隔离每个 cell；校验 scorer 实际提取的 `audio.wav` PCM 也未被内部 `-async 1` 改变。

复用官方 MFCC、JPEG 帧提取和网络前向，不另造 proxy scorer。全局输出按官方全部行重建；PLUS/MINUS 使用同一矩阵的固定内部行，单独命名为 local diagnostic，不能冒充官方全局指标。对列顺序、offset 符号、边缘支持范围写合成测试，尤其验证画面前读三帧对应官方 offset 减三帧。

现有全局 log parser 本身不够用，因为只有 C/D/offset 会丢失局部信息。无需修改第三方网络；若 wrapper 无法取得完整矩阵，停止并报告，而不是从三项汇总反推。

### 3. 优先检查基线，再检验局部响应

先验证重复性和基线峰是否清晰，再解释 WARP。绝对 offset 可以包含原生偏移，因此用各段相对 REAL 的 offset 差，不移动音频去校正它。两段恢复分别要求相反方向，避免把一个全局 shift 当作成功。

spec 中峰间距及记录数门槛是本次预先选定的保守诊断标准，没有经过新结果搜索，也不是已有文献证明的通用阈值。基线不清晰得到 INCONCLUSIVE；完整实验的科学失败是有效终态。没有主观审阅者时数值部分仍可完成，盲看表单单独标为待填写。

### 4. 小型实现与交付

建议新包按职责分成 `runner.py`、`media.py`、`scoring.py`、`analysis.py`、`validate.py`；不强制文件数量。复用历史纯 hash/PCM/bootstrap 小函数，所有输出路径显式传入，不调用绑定旧 run 的 stage runner。

建议 CLI：

```bash
python -m scripts.experiments.lrs3_real_video_local_timing.runner --run-id <id> --stage all
python -m scripts.experiments.lrs3_real_video_local_timing.validate --run-root runs/lrs3_real_video_local_timing_<id>
```

阶段为 `prepare/media/score/report/all`；prepare 冻结输入与 crop/mapping，尚不读新模型分数。run-id 仅允许字母、数字、下划线、连字符。

```text
runs/lrs3_real_video_local_timing_<id>/
  protocol.json
  media/                 # 帧映射、crop 来源、三臂媒体及 manifest
  scores/                # 每 cell 矩阵、窗口坐标、日志及 manifest
  review/                # 匿名 A/B 媒体、空表单、独立映射文件
  final.json
  result.md
  validation.json
```

每个阶段 manifest 保存上游 hash、实际输入/输出 hash 与完整命令；保持单向依赖。validator 从媒体与矩阵复算，输出绑定 final hash。沿用原 run 只续做未完成且配置一致的 cell；修复导致代码变化时建新 run，保留旧证据。

## Risks / Trade-offs

- 原视频基线本身可能不清晰 → 预设 BASELINE_INCONCLUSIVE，保留全部记录。
- 帧重采样改变视觉连续性及头部运动 → 保存映射、报告限制；恢复正确符号比单纯分数下降更有解释力，但仍不证明纯因果隔离。
- 原视频裁脸协议不同于历史生成视频协议 → 阳性仅缩小失败来源，不能断言历史 scorer 正常或 Wav2Lip 故障。
- 官方搜索会吸收全局偏移 → 局部两段曲线作为主判据，全局 C/D 仅描述。

## Migration Plan

无需迁移或部署。实现并通过针对性测试后运行一次完整固定诊断，离线验证后交付结果与盲看包。历史文件只读，进一步的生成器诊断另立 change。
