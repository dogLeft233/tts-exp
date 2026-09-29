# Bridge 的 TTS 来源与质量比较：下游交接

状态：**implemented，流水线代码与独立验证器已实现；正式 GPU 实验尚未运行**。

先读 [proposal.md](proposal.md) 理解问题，再按 [design.md](design.md) 实现；[spec.md](specs/lrs3-bridge-tts-quality/spec.md) 是验收契约，[tasks.md](tasks.md) 是任务清单。数值、矩阵、缺失处理和终态以 design 与 spec 为准；两者矛盾必须先修正文档，不由实现者选择有利解释。

实验直觉：对同一个人说的同一句话，用本地、云端 TTS 分别提供一个声音目标；保留自然音频相位和时间长度，将其幅度谱向目标移动 75%；让两个 bridge 分别驱动同一个 Wav2Lip 视频；最后都换回原始自然音频，用 SyncNet 判断哪一种口型更匹配原话。

最重要的实现边界：

1. 历史 LRS3 bridge 的目标来自云端 Qwen。必须查生成 metadata 和 hash，不能按目录印象标注来源。
2. 两个 bridge 的**评分音轨都是同一份原始 N**。自身音轨评分只是诊断。
3. 两种 TTS 必须共用同一批记录、文本、自然参考、对齐算法、声码器、bridge 参数、视频条件及评分支持。
4. 本地与云端的原始音频不能直接拿去做 STFT 混合；先生成自然时间网格上的 `M_LOCAL/M_CLOUD`。
5. 主检验是两种 bridge 的配对差；“云端显著、本地不显著”不能代替差异检验。云端比本地好也不等于云端比 N 好。
6. 音质通过独立匿名听评衡量。未取得人工评分时，仍完成自动比较，明确标为 `QUALITY_NOT_ASSESSED`，不得伪造 MOS 或把 Sync-C 当音质。
7. 不沿用历史失败的生成式 LOCAL_SWAP 作为本实验通过前提。本实验单独验证渲染重复性、B0 重建和固定视频的错误音轨敏感性；这不代表已经解决局部时间传递问题。
8. 任一必需数据或对齐失败，保留失败台账并报告阻断；不能只保留双方恰好成功的高质量子集后宣称完成 22 条实验。

建议从 `python -m scripts.experiments.lrs3_bridge_tts_quality.runner --help` 开始验收 CLI。正式运行前先执行 `audit`，再按 `tts → targets → bridge → quality-pack → render → score → analyze` 顺序推进；GPU 阶段由串行 lease 和空闲检查保护，当前运行环境磁盘只剩约154 MiB，因此本次实现验证没有启动正式推理。
