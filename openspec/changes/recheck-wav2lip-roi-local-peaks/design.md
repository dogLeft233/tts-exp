## Context

依据 BM：`Wav2Lip face-ROI replacement pilot 2026-09-06`、`Wav2Lip ROI control failure diagnostic 2026-09-06`。此前从 0/22 到 ROI 后 14/22 的 C 变化提示应继续检查局部响应，但不能把不同实验间的变化当成单因素因果估计。

## Experiment

问题：固定媒体经过独立实现的相同 SyncNet endpoint，8 条失败的整数峰和 C 残差是否仍一致？

唯一变化为评分实现，模型、媒体、PCM、窗口、候选 offsets、阈值均固定。2 条通过记录用于检查独立实现是否普遍漂移。该结果是定向诊断，不从 10 条推断 22 条控制通过率。

使用官方 `SyncNetModel.S` 和权重，从现有 muxed media 新解码 JPEG/BGR 帧及 PCM，独立组窗、forward、距离计算。保存新提取的 visual/audio embeddings，使离线 validator 能在不重复神经网络推理的情况下独立重算距离。validator 验证的是 embedding→矩阵→结论链，不能独立证明神经网络推理无 bug，报告应说明这个边界。

优先使用已有 `/home/wjj/.venvs/syncnet/bin/python`，默认 CPU、batch=20、torch threads=4；不安装环境。允许在首次正式评分前选择本机 CUDA 并冻结设备信息；CUDA 必须通过宿主命名空间执行，普通 sandbox 的设备透传问题不需要重装驱动。中途改设备需新 run，不覆盖旧产物。CPU 无 CUDA 仍可完成。

## Implementation boundary

保持小实现：`runner.py` 编排/审计/汇总，`worker.py` 独立提取特征，`validate.py` 离线复核；允许小 common/config 文件，不创建通用框架。可复用上一诊断的 `load_parent()` 作为只读身份审计；不得用它的 diagnose/峰函数作为本轮独立结果。

独立 worker 可阅读 `third_party/syncnet_python/SyncNetInstance.py` 以实现同一契约，但不得 import/call `SyncNetInstance.evaluate`、`calc_pdist`、历史 syncnet_worker。不要通过 monkeypatch `.cuda()` 实现 CPU 支持；模型和张量显式 `.to(device)`。

预期接口（待实现）：

```bash
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_roi_peak_recheck.runner --run-id 20260906_luna --device cpu
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_roi_peak_recheck.validate --run-root runs/wav2lip_roi_peak_recheck_20260906_luna
```

已完成 cell 只有在协议、媒体、代码和输出 hashes 全部匹配时才可恢复使用；失败/部分输出保留，明确阻塞点，不按结果换样本、设备或容限。

## Handoff and stop

交给 `gpt-5.6-luna`、reasoning effort `max` 的 subagent，顺序完成 tasks。它负责代码、实际实验、验证和 BM planned→真实终态；主 agent 负责审核 spec 和验收。可修复本轮实现 bug，但必须记录受影响 run，源码变化后新建正式 run。一次完整复核即停止，不自行进入下一干预。

若复评分一致，只能说“独立实现复现了 SyncNet 所见的局部响应误差”；模型本身的时间分辨率、音频 warp 的声学变化、生成器响应仍不能被这个实验单独区分。若不一致，先报告具体 cell/列/偏差，后续针对预处理或距离实现设计单一修复。own-audio 门禁本轮未重测。
