## Context

动机与范围见 proposal；精确输入、公式、矩阵与终态以 `specs/lrs3-wav2lip-timing-transfer-diagnostic/spec.md` 为准。

规划时只读检查了已有 manifests 和时间映射，未执行新评分：22 条生成视频都是原视频前缀，G_N/G_W 帧数相等，生成视频的音频尾差为 1664–2176 samples；按本 spec 共同范围构造的 PLUS/MINUS 最少各有 20 行，满足 5 行下限。这是可行性检查，不是实验结果，执行者仍需检查实际媒体。

## Goals / Non-Goals

目标是用小型新模块适配既有媒体，新增部分主要为 manifest join、共同时间掩码与四项局部检查。保留原实验模块的默认契约；不复制整套渲染流水线，不增加配置搜索或新的模型。

## Decisions

### 1. 六个交叉 cell，全新统一评分

R/G_N/G_W × N/W 可以同时观察音频替换和生成响应。仅补 R/W 会缺少生成域敏感性参照；重跑生成器则扩大了本轮范围。两个基线重复评分使用同一 mux、独立 worker 进程与工作目录，核查本轮评分重复性。

新模块建议保持 `runner.py`、`protocol.py`、`analysis.py`、`validate.py` 和必要媒体适配即可。提供：

```bash
python -m scripts.experiments.lrs3_wav2lip_timing_transfer.runner --run-id <id> --stage all
python -m scripts.experiments.lrs3_wav2lip_timing_transfer.validate --run-root runs/lrs3_wav2lip_timing_transfer_<id>
```

runner 支持 `prepare|media|score|report|all`。prepare 完成全部资产审计、冻结 masks 与协议；all 执行完 report 后调用离线 validator，返回明确退出状态。

### 2. 复用原视频轨迹，不检测生成画面

从 tail_v2 `protocol.json` 的 `records[].crop.processed_track` 取轨迹。三个源视频按绝对帧 i 使用同一坐标，生成视频只取轨迹前缀。可复用 `lrs3_real_video_local_timing.media.crop_frames_from_track` 与父实验编码逻辑；无需重新调用 S3FD。每种视频编码一次，再分别 mux N/W；预期为 66 份裁后视频流、132 个 mux、176 个评分目录。

审计 calibration v5 `03_videos/videos_manifest.json` 的 `rows[].arms.N/LOCAL_WARP_120`；音频 manifest 的 `rows[].arms` 是列表，要按 arm 名查找。不要使用混同 WAV 容器和 PCM 的旧 format 字段：新审计同时保存 container_sha256 和真实 decoded_pcm_sha256。

父 final 及其 protocol/manifests 链必须核验；复用历史输入不等于信任路径字符串。生成视频历史 geometry 为 `constant_full_frame_fallback`，本轮只诊断这些既有输出，不静默改为另一套生成 geometry。

### 3. 文件保持原长，统计比较取共同时间

真实视频仍按 bounded_audio_tail_v2 验证；生成视频按固定 manifest 的精确帧数验证。不要把父模块的真实视频时长检查原样调用到生成视频上。所有 PCM 原长保存，不用 ffmpeg `-shortest`。

共同 Q 和候选行完全由输入长度决定，s 的周期始终是原始音频 L−1。距离矩阵仍按各媒体原长保存；局部曲线与跨视频 C/D 比较只聚合共同掩码。另保留官方原生全局指标供复核，并用不同字段名区分 `official_global`、`common_global`、`local`。

### 4. 数值预测保持显式，避免符号错误

用 float64 构建 s 数组，`interp(t, n, s)` 得正向映射、`interp(t, s, n)` 得逆映射；不再次变换 PCM。音频偏移与视频响应的预期方向相反，直接按 spec 表计算。

复用 `lrs3_real_video_local_timing.syncnet_worker` 导出官方距离矩阵（绑定实际文件 hash）；新的 score 适配层必须以 cell 指定的 N/W 检查 PCM，不能调用只接受 natural audio 的旧 `score_one`。无需修改第三方 SyncNet。可以复用通用 I/O、矩阵重建，但离线 validator 独立实现掩码、四项计数、bootstrap 与终态树。

### 5. 一次诊断，清晰收口

prepare 先冻结所有参数，缺文件等工程失败写 BLOCKED；完整科学失败按固定树交付。每次完整 cell 的输入/输出/代码身份用于安全 resume；不能覆盖终态。测试覆盖缺失/交换 cell、PCM 与 PTS 篡改、前后半段符号、逆映射、尾部共同支持、峰边界与 17/18、19/20 的计数边界。

结果写 BM 的单一实验笔记，按 Startup Router 先加载指令、搜索再更新。引用真实视频诊断时，以 Amendment result 的 tail_v2 完成结果为当前证据；其旧 Observations 中的 BLOCKED 是旧 run，不应被下游误当成当前状态。

## Risks / Trade-offs

- 音频线性插值同时改变时间和声学特征 → A 失败只能定位到音频控制/评分组合，不能宣称纯声学损伤。
- 生成 geometry、视觉质量和原视频头部运动仍可能影响 C/O → 保留 `UNRESOLVED` 分支，不声称架构因果或纯唇部响应。
- 共同裁脸不同于旧 Wav2Lip 官方流水线的逐视频检测 → 本轮结果只解释新统一诊断设置，历史 CONTROL_FAILED 不被改写。
- 22 条已经看过历史结果、局部 offset 是有限窗口近似 → 固定阈值和误差容差，不作独立确认或总体外推；不据此启动训练。

## Migration Plan

新增独立目录及 run，不迁移旧数据；完成后提交可复核产物路径与 BM 指针即可，无部署步骤。
