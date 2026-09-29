## Context

先读本 change 全部文件，再读 BM Startup Router → 实验指令，以及以下实验全文：

- `Wav2Lip ROI local peak recheck 2026-09-06`
- `Wav2Lip face-ROI replacement pilot 2026-09-06`
- `LRS3 真实视频局部时间敏感性诊断`（以 Amendment result 的 tail_v2 完成结果为准，其旧 Observations 仍有历史 blocked）
- `LRS3 Wav2Lip 局部时间传递诊断结果`

上一轮建议停止复评分线，本轮遵循这一结论，另立像素重定时干预。最终目标仍是“候选音频驱动视频后，换回 untouched natural audio 有同步收益”，未来才验证其他 TFG 和通用头；当前没有 replacement 收益确认。

## Experiment and interpretation

记 G_N 为自然音频驱动的现有 ROI 生成视频，W[n]=N(s[n]) 为既有局部 warp 音频。构造 V_ORACLE[i]=G_N[q(i)]，q 是 s 在视频时间戳处的最近帧索引。V_ID 使用 i→i，经过完全相同的无损编码。所有裁脸先在父实验完成，本轮重排整个裁后像素帧，不重新检测人脸，不用输出帧号重新裁源画面。

这个实验补齐“已知重定时生成画面”的参照。实际 G_W 是生成器对 W 的输出，V_ORACLE 是重排 G_N；两者可能同时存在唇部、头部运动、纹理和时序量化差异，所以即使 oracle 通过，也只能把后续重点指向实际生成响应路径，不能宣布 Wav2Lip 架构有 bug 或纯粹响应不足。

沿用父连续映射的近似 offset 预测和 1 帧门槛，正是为了检验原控制判据对已知变换是否适用。保存离散 q 和量化误差作为解释证据，不用它们事后改写预期值或让历史 C 变通过。own-audio 的新检验是 V_ORACLE/W 对 V_ID/N；父 G_W/W 的 own-audio 失败没有被重测或修复。

规划时已用父 manifest 的 L/F 做只读检查：22 条 q 全部单调、索引在界内，最大量化误差 0.499995 帧。尚未解码生成本轮媒体或评分；下游仍必须完成真实资产审计。

## Implementation

建议仅 `runner.py`（审计/构造/评分编排）、`analysis.py`、`validate.py` 加必要小型 common/config，不建通用框架。优先复用 `wav2lip_roi_peak_recheck.worker.SyncNetScorer`，其 `source_audio` 参数是当前 cell 的 N 或 W，报错文案中的 N 不表示只能传自然音频。不得继承旧 runner 的 10 条选样或固定 `AUDIO_ARM=N`。

固定 `/home/wjj/.venvs/syncnet/bin/python`、CPU、batch=20、torch threads=4，串行处理 cell。不安装环境，不需要 GPU 或宿主 GPU 提权。父 final 在 fix5，但媒体 manifest 可能合法指向 fix1；必须沿 hash 绑定解析实际路径，不能拼接假定所有文件都在 fix5。

媒体采用同一固定 ffmpeg、FFV1 / Matroska、bgr0 无损编码，沿用父流 color_range=pc、color_space=gbr 和其余颜色元数据，解码回 BGR24 做逐帧字节校验。规划时抽查首条父流符合该格式；正式审计必须检查全部 22 条。先从父 G_N 视频流直接解码原始像素，再构造 V_ID/V_ORACLE；不要先经 JPEG 再编码。JPEG 提取只在 SyncNet worker 内发生，沿用已复核契约。不做光流插帧、混帧或缩放。

预期入口（下游待实现）：

```bash
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_roi_retiming_oracle.runner --run-id <id>
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_roi_retiming_oracle.validate --run-root runs/wav2lip_roi_retiming_oracle_<id>
/home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_roi_retiming_oracle
ruff check scripts/experiments/wav2lip_roi_retiming_oracle tests/experiments/wav2lip_roi_retiming_oracle
openspec validate diagnose-wav2lip-roi-retiming-oracle --strict --no-interactive
```

## Decision and stopping

科学判定的精确优先级见 spec。原局部阈值、own-audio 和损伤门槛全部保持；矩阵完整但科学失败也是完成实验。

- 基线复现失败：定位本轮媒体/评分差异后停止。
- 已知重定时仍未过局部门槛：原 timing 判据在此生成画面/重定时构造下尚未校准，暂停用它归因实际生成器。
- 局部通过但 own-audio 或损伤失败：该控制组合仍不足以支持 bridge，下一设计应聚焦控制构造与评分有效性。
- 全部通过：已知重定时在当前生成画面上提供了有效参照；后续可另立实际生成响应与 oracle 的对照设计。

以上均不自动进入下一实验。报告同时列出历史 8 条 C 失败记录在本轮的表现，仅作描述，不把选中子组当独立验证，不据此筛样。一次完整诊断后结束，交付 BM permalink 与产物路径。
