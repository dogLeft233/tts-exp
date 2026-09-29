# LRS3 TTS 优势机制实验：下游执行入口

状态：**AUTOMATIC_COMPLETE，review_v15 已完成并通过独立 validator；人工感知 NOT_ASSESSED**。这是英文 LRS3 上已观察优势的机制诊断，主要节省方式是复用历史分数和生成视频。有效结果见 [review_v15 报告](../../../runs/lrs3_tts_gain_mechanism_review_v15/report.md)，下一步归因实验见 [新 spec](../disentangle-tts-native-gain/README.md)。

## 先读什么

1. [design.md](design.md)：问题、公式、样本、控制、统计与解释规则。
2. [input-bindings.json](input-bindings.json)：已核实的 50 条记录、45 个来源组、12 条曲线队列及输入 SHA-256。
3. [规范](specs/lrs3-tts-gain-mechanism/spec.md)：验收条件。
4. [tasks.md](tasks.md)：按依赖顺序实现。

## 最短执行路线

先实现 CPU 的 audit→decompose→report/validate，直接交付两个模型的历史增益分解；随后完成 LeapTalk 的 curves→analyze→perception-pack→report/validate。缺 GPU 或磁盘时保留 CPU 结果，准确报告曲线阶段待执行，不把整个实验写成失败。

已实现 CLI：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage audit
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage decompose
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage curves
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage analyze
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage perception-pack
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage perception-analyze
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.runner --run-id first --stage report
PYTHONPATH=. .venv/bin/python -m scripts.experiments.lrs3_tts_gain_mechanism.validate --root runs/lrs3_tts_gain_mechanism_first
```

支持 `--stage all` 顺序执行，控制台及时给出完成数；阶段结果和 stderr 保存到本 run。SyncNet 子进程使用 `/home/wjj/.venvs/syncnet/bin/python`，而非 qwen3 环境。默认不下载模型、不安装依赖、不调用云端。最新有效 run 为 `runs/lrs3_tts_gain_mechanism_review_v15`：200 个历史 cell、100 个模型内配对、45 个来源组，以及24主曲线/4控制均经独立validator核验。INTERIOR ΔC=+1.253，最佳距离改善+0.286且校正区间跨零；人工包已生成但尚无人评分。v13 的 RESOURCE_WAIT 是此前资源不足的历史状态，不能继续用作本实验当前状态。

## 执行前已经查明的事实

- LRS3 是原 multiset manifest 的 ID 151–200，不能误取前 50 条；两个模型各有 100 个历史评分 cell。
- `speaker_key` 全部为字符串 `lrs3`，不能用来分组。真实来源由视频路径的父目录读取，共 45 组。
- LeapTalk 这 50 对视频均在本机；12 条曲线队列已绑定 24 个 MP4 hash。Ditto 这里只绑定历史分数，未定位到完整原生成视频，因此本轮不要求 Ditto 新前向。
- 绝大部分旧目录只有 `syncnet.json`，不能假装有完整距离矩阵；零散缓存不能用于按可用性改选样本。
- 权重实际在 `third_party/syncnet_python/data/syncnet_v2.model`，不是已经缺失的 `data/syncnet_v2.model`。
- spec 编写时根盘仅约 447 MiB 可用；执行时重新检查。CPU 分解可先完成；曲线阶段空间不足时返回 RESOURCE_WAIT，不删除历史数据或启动占满磁盘的帧提取。

## 可直接复用的实现

- `third_party/syncnet_python/SyncNetInstance.py`：官方 frontend、`evaluate`、`calc_pdist`，返回 `[T,31]` 距离矩阵。
- `third_party/syncnet_python/run_pipeline.py`：官方检测、track 和评分 crop。
- `scripts/experiments/lrs3_real_video_local_timing/syncnet_worker.py`：独立环境 worker、PCM/矩阵落盘模式；它要求输入已是正确评分 crop，不能直接拿全脸 MP4 绕过裁剪。
- `scripts/experiments/wav2lip_replacement_reconciliation/analysis.py`：矩阵→曲线的参考。不要复制其固定 22 组约束；本实验点估计和 bootstrap 必须使用同一 group-weighted estimand。
- `scripts/experiments/lrs3_bridge_tts_quality/`：GPU lease 与文件绑定模式。只复用小模块，不启动其 TTS/渲染流水线。

## 交付标准

读者应能从 report.md 明白：英文优势的依据、C/D 的区别、两个加项的数值、完整曲线说明什么、哪些原因仍未识别。主结论不能只给一个 `GO/NO_GO` 标签。所有新增实验输出必须可追溯到固定输入与代码快照。
