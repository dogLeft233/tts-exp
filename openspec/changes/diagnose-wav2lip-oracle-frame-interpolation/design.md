## Context

下游阅读顺序：本 change 的 proposal → spec → design → tasks；然后 BM Startup Router → 实验指令 → `Wav2Lip ROI retiming oracle 2026-09-06` 与 `Wav2Lip ROI local peak recheck 2026-09-06` 全文。前者已完成，不能把其未通过门禁误当作尚未执行。

原计划最终要验证“候选音频驱动 TFG 后，换回未经修改的自然音频仍有同步收益”，再研究通用生成头。本轮只是其控制构造诊断；V_LINEAR 不是 TFG 对候选音频的实际输出。所有样本已见，分类为 `seen_fit_diagnostic`。

## Frozen evidence

父目录固定为 `runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4/`。以下均为文件字节 SHA-256，不能与 JSON 中 `artifact_sha256` 混用：

| 父文件 | 文件 SHA-256 |
|---|---|
| final.json | 919a5af95e361cbe218fc2d88003aa74d2e061d68d3543fe8fcbc7466cc0cff1 |
| validation.json | a8e4719af2aa1e15477871caadea16f2c7b6fc69b4404c71f19bca2d33546724 |
| protocol.json | 6bec09cb8abdc7d8de9b50dfb388f1395924b24c9b130b808a9c5387378fa079 |
| input_audit.json | f2b6603d4999db60ecd44b27a7396f25b6d582b43ff93abd2bac3b4ee6efc5c6 |
| analysis.json | 74ab3eb0b3fb0f8d3ced5be53117c2b9ea90fc2f7214b5aa1f8d7cec2d3d365e |
| frames/manifest.json | 3a713cf922dbe9c9849715d93ab0a7a219fe69502b9df5117ba50e4401a051e3 |
| audio/manifest.json | 4ea22a2fa2838891661d4cc5d18ccc3e3d4857d489afadd2ee42dcee5c39d3dd |
| media/manifest.json | 609cded610f1807f532fd80964c02e837521f7493c8214ed3c75eb6c2fe2a1da |
| scores/manifest.json | d25b08e26534c8a407eba594413ff5eb19cad9d4fe4160f0b9714a46e9148842 |

沿 manifest 解析实际路径。父 protocol 的逐条字段为 `sample_count`、`frame_count`、`indices`、`masks`；配置为 `frozen_config`。父 score manifest 的列表字段是 `scores`，帧 manifest 是 `rows`。不要假定所有 manifest 都叫 rows，也不要凭命名拼接 worker 结果路径。

规划时仅只读检查了 22 条 `indices.q_float` 均在 [0,F−1] 内；未生成 V_LINEAR、未评分。下游仍须独立重算映射、检查全部实际资产。

## Small implementation

建议新包只用 `runner.py`、`analysis.py`、`validate.py`，按需加小型 `config.py`/`media.py`。不要复制整套父 runner 或构建可配置实验框架。

- 像素解码、无损编码、PCM mux：复用 `wav2lip_roi_retiming_oracle.common` / `.media` 的纯工具。编码已适配 ffmpeg 8：命令 `-colorspace rgb`，ffprobe 的值仍须为 `gbr`。
- 新评分复用 `wav2lip_roi_peak_recheck.worker.SyncNetScorer`，每个 cell 的 `source_audio` 必须为该 cell 的 N/W，不能统一传 N。父 runner 可作为调用样例，不能继承其臂集合或计数。
- 统计可复用父 `peak`/`summarize` 等纯函数，但新分析须明确接受 88 个历史 cell 和 44 个新 cell；离线 validator 独立实现像素公式、距离和统计判定，不能调用 producer 的对应函数。
- 从 V_ID 的实际无损视频流解码 X，先核对逐帧 hash 与父 identity 证据。V_ORACLE 是比较臂，不是插值输入；对已经取整的 V_ORACLE 再插值无法回答本问题。

固定 `/home/wjj/.venvs/syncnet/bin/python`、CPU、batch=20、threads=4、串行 cell；沿用父 torch/numpy/OpenCV、ffmpeg 和权重。新 protocol 绑定实际使用的 scorer、依赖源码、环境、本 change 的 proposal/design/spec；可勾选的 tasks 不进入冻结 hash。评分相关版本/配置漂移则 BLOCKED，不能默默换 endpoint 再与缓存比。

## Interpretation and stopping

主问题为 V_LINEAR 是否同时通过 timing、own、damage；次问题为相对最近帧 oracle 的 own 改善是否达到固定阈值。两者分别报告，“有所改善”不等于有效控制。

若全部通过，只支持当前样本、当前评分器下线性插值 oracle 可用，后续另立实际 G_W 与 oracle 对照。线性混帧会同时改变帧量化、局部平滑和纹理，不能声称量化是唯一原因，也不能称实际生成器已修复。

若仍失败，结束本次固定插值诊断，记录哪个 gate 失败；不继续尝试光流、更多插值核、幅度或阈值。不足以由此宣判所有 oracle、Wav2Lip 或其他 TFG 不可行；下一次分支决策应综合成本与控制有效性另作设计。

## Handoff commands

下列新模块由下游实现，当前尚不存在：

```bash
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_oracle_frame_interpolation.runner --run-id <id>
/home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_oracle_frame_interpolation.validate --run-root runs/wav2lip_oracle_frame_interpolation_<id>
/home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_oracle_frame_interpolation
ruff check scripts/experiments/wav2lip_oracle_frame_interpolation tests/experiments/wav2lip_oracle_frame_interpolation
openspec validate diagnose-wav2lip-oracle-frame-interpolation --strict --no-interactive
```

交接给下游的任务：实现本 change，完成一次全 22 条实验与独立验证，并更新 BM `Wav2Lip oracle frame interpolation 2026-09-07`。科学失败也是合法完成；最终交付数字、判定、validator 结果、run 路径和 BM permalink。
