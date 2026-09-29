## Why

上一轮 `wav2lip_roi_peak_recheck_20260906_review2` 已完成 20 个新 SyncNet cell：8/8 历史失败、2/2 历史通过全部复现，终态 `PEAKS_REPRODUCED`，validator valid。没有发现评分实现差异。父 ROI 控制仍为 `CONTROL_FAILED`：局部响应 C=14/22；own-audio Sync-C 非劣性 CI 下界 −0.170397，未超过 −0.10。

历史证据链是：LOCAL_SWAP / 平滑音频 warp 控制失败 → 真实视频离散重定时可检测（22/22）→ 旧 full-frame 生成视频 C/O=0/22 → 正确 face ROI 后 C=14/22、O 通过 → 独立推理复现剩余偏差。跨实验存在其他设置差异，不能把 0→14 当作 ROI 的单因素效应。真实视频实验也没有回答“生成画面按同一音频映射重定时后，局部响应与 own-audio 能否同时通过”。

下一步做一次**已知视频重定时对照**：由现有 G_N 的像素构造与 W 共享时间映射的 V_ORACLE，绕过重新生成，并同时检验局部响应、自身配对和替换损伤。这给当前控制协议一个可验证的参照，帮助区分后续应研究评分/扰动组合，还是实际生成响应路径。它不是可部署生成头，也不证明单一因果机制。

## What Changes

- 固定父 22 records / 22 source groups，两个视频臂（恒等重编码 V_ID、已知重定时 V_ORACLE）× 两个音频臂（N、W），共 44 个新视频流、88 个新评分 cell。
- 从父评分用的 224×224 G_N 视频流取像素，冻结同一时间映射、掩码和原有容限；CPU 执行，无新 Wav2Lip 推理。
- 交付可独立验证的逐帧索引/像素绑定、评分矩阵、判定和 BM 实验记录。所有结果只用于诊断；历史控制失败继续保留。

## Capabilities

### New Capabilities

- `wav2lip-roi-retiming-oracle`: 在既有生成画面上构造与音频同映射的重定时对照，检验局部响应和配对有效性。

### Modified Capabilities

无。新实验不修改历史实验规则。

## Impact

新增 `scripts/experiments/wav2lip_roi_retiming_oracle/`、对应 `tests/experiments/` 和新 run。复用已复核 SyncNet worker，保持实现小型化。无新 TTS、Wav2Lip、MFA/DTW、模型下载、租卡或训练。本 change 仅设计；下游收到执行任务后按 tasks 完成一次实验。
