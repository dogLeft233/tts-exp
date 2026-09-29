## Context and decision

下游阅读 proposal → capability spec → design → tasks，并按 BM Startup Router 加载实验指令，读取 `Wav2Lip oracle frame interpolation 2026-09-07`、`Wav2Lip face-ROI replacement pilot 2026-09-06`、`LRS3 natural-to-TTS bridge confirmation result`。

保留最终目标 `Sync(mux(TFG(face,C),N)) > Sync(mux(TFG(face,N),N))`，其中 N 必须未改动。中文原始 TTS 自身配对优势不是该目标的成功证据；英文 LRS3 上也尚无稳定 replacement 收益。因此只值得一次明确止损的控制检查，不支持训练扩容。

本轮假设：在没有局部插值/变速的片段内部，已知共同平移应近似保留 SyncNet 距离曲线；实际生成链若也能跟随该平移，则可在新的明确适用域内重新设计候选收益实验。

五帧即 200 ms / 3200 samples，是 25 fps 视频、80 Hz Wav2Lip mel 和 100 Hz MFCC 时间网格的最小共同正平移单位。它不是对旧 120 ms 正弦幅度的搜索。新 P 在拼接点有跳变，整条音频不作为可部署候选；只检验内部平台。

## Frozen inputs

三个固定来源均只读，下面是文件字节 SHA-256。沿已验证 final/protocol 的证据链解析实际 manifests 与资产；不得把 JSON 自哈希误作文件 hash。

| 来源 / 文件 | SHA-256 |
|---|---|
| `runs/wav2lip_roi_retiming_oracle_20260907_oracle_v4/final.json` | `919a5af95e361cbe218fc2d88003aa74d2e061d68d3543fe8fcbc7466cc0cff1` |
| 同目录 `validation.json` | `a8e4719af2aa1e15477871caadea16f2c7b6fc69b4404c71f19bca2d33546724` |
| 同目录 `protocol.json` | `6bec09cb8abdc7d8de9b50dfb388f1395924b24c9b130b808a9c5387378fa079` |
| `runs/wav2lip_face_roi_replacement_20260906_host_fix5/final.json` | `76e005642ee125927e8f3f2a67a8cf9a0d19299ea02401c09c5989b85b643563` |
| 同目录 `validation.json` | `328e562cc8dacfb960fb2319fb15f1cfa90eecf593f04434382c9a5e13bd2577` |
| 同目录 `protocol.json` | `835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4` |
| `runs/wav2lip_oracle_frame_interpolation_20260907_linear_v1/final.json` | `3c34ed447326eb164cc5549cd902fc55f64f4d4b5ce8c3e8b5ce09489bd3a542` |
| 同目录 `validation.json` | `e5f82d26b3c7bd7f45bfeca9c0fe6589a5983ba6147348fc04e11aac46ea5820` |

oracle_v4 提供 V_ID 无损 224×224 像素、N PCM、V_ID/N 的 22 个 cached matrices/embeddings。ROI fix5 提供原 face、逐帧框、评分 crop 轨迹和冻结生成接口；引用 fix1 的实际资产合法。linear_v1 只提供历史结论，不复用其 44 个分数。

规划时仅检查 protocol 的帧数：22 条均可按固定公式获得平台窗口，最短 F=119，PLUS/MINUS 为 9/10 行。此检查未打开新 P 或读取新结果；下游仍须做完整窗口支持审计。

## Small implementation

新增 `scripts/experiments/wav2lip_integer_plateau_control/`，建议 `runner.py`、`analysis.py`、`validate.py`，按需拆 `config.py`/`media.py`。不复制整个父流水线，不增加可调参数框架。

- I/O：复用 `wav2lip_roi_retiming_oracle.common/media`。完整解码核验是最终依据；FFV1 编码颜色配置以父可运行适配为准（命令 rgb、probe gbr）。
- CPU scorer：`wav2lip_roi_peak_recheck.worker.SyncNetScorer`；每个 cell 传正确 source_audio，保存 fresh embeddings 和矩阵。
- GPU：调用 `wav2lip_face_roi_replacement/generation_worker.py` 的 JSON 音频映射接口，只传新臂 G_P，避免继承旧 runner 的控制臂和自动 bridge 分支。复用原 face/框与评分裁图轨迹，不把 V_P 当生成输入，不重跑检测。
- Stage A 结束独立 `validate --stage oracle`，生成仅绑定 A 的验收文件；Stage B 启动前检查该文件及其全部证据 hash。最终验收另存文件，避免覆盖阶段证据或循环哈希。

SyncNet 使用 `/home/wjj/.venvs/syncnet/bin/python`，CPU，batch=20、threads=4。GPU 使用 `/home/wjj/.venvs/wav2lip/bin/python`，V100 16GB，batch=4、串行视频；必须宿主命名空间执行，遵守项目 GPU 时段约束。sandbox 不见设备不是驱动故障。依赖/权重/配置漂移应 BLOCKED，不重新装驱动或换评分器。

## Interpretation and stopping

关键变化是**按源内容匹配的比较**：V_P/P 在输出 r 看的是原始 r+s 的内容，必须与 V_ID/N 的 r+s 行比较，不能拿 r 行内容差异充当损伤。旧 chronological own gap 仍完整报告，但它与本轮匹配比较是不同 estimand，不能互相冒充。所有新的掩码和规则均在评分前冻结。

Stage A 失败：暂停这个控制/评估分支；不再试位移大小、拼接位置或 oracle。Stage A 通过、B 失败：暂停当前 Wav2Lip 生成链控制分支；失败也可能含 face pose、回贴/裁图或时间网格因素，不等于单独证明生成网络无效。均通过：只建议新建固定候选 replacement 收益 spec，届时另行登记与本轮适用域一致的控制；不恢复旧 bridge runner 的门禁状态。

无论结果如何，本 spec 完成后停止。切换 TFG、扩数据或训练都需另作科学设计。本轮无法证明平滑变速失败的唯一原因，也不能证明拼接附近的自然度或全视频画质。

## Handoff commands

以下新模块由下游实现；`all` 自动执行 A → 独立验收 → 条件性 B → 最终验收，科学失败也是完整交付。

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_integer_plateau_control.runner --run-id <id> --stage all
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_integer_plateau_control.validate --run-root runs/wav2lip_integer_plateau_control_<id> --stage all
PYTHONPATH=. pytest -q tests/experiments/wav2lip_integer_plateau_control
ruff check scripts/experiments/wav2lip_integer_plateau_control tests/experiments/wav2lip_integer_plateau_control
openspec validate test-wav2lip-integer-plateau-control --strict --no-interactive
```
