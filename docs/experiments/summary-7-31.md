# TFG/THG 模型、HDTF 与多语言实验摘要

## 1. 已评分的 TFG/THG 模型对比

| 模型与协议 | 出处/年份 | Natural C -> TTS C | ΔC | Natural D -> TTS D | ΔD | 备注 |
|---|---|---:|---:|---:|---:|---|
| [Wav2Lip](https://arxiv.org/abs/2008.10010)<br>TFG/V2V 唇形同步 | ACM MM 2020 | 7.393 -> 8.922 | **+1.529** | 7.327 -> 5.941 | **-1.395** |  |
| [IMTalker](https://arxiv.org/abs/2511.22167)<br>TFG 人像生成 | arXiv 2025 | 6.235 -> 7.369 | **+1.134** | 8.523 -> 7.543 | **-0.980** |  |
| [V-Express](https://arxiv.org/abs/2406.02511)<br>TFG 人像生成 | arXiv 2024 | 5.975 -> 6.506 | +0.531 | 8.818 -> 8.236 | -0.582 |  |
| [MuseTalk 1.5](https://arxiv.org/abs/2410.10122)<br>TFG/V2V 视频配音 | 论文 arXiv 2024；1.5 checkpoint 2025 | 5.789 -> 6.326 | +0.537 | 8.081 -> 7.928 | -0.153 | 使用 1.5 checkpoint |
| [Ditto](https://arxiv.org/abs/2411.19509)<br>TFG/THG 基线 | ACM MM 2025（arXiv 2024） | 5.670 -> 6.819 | **+1.149** | 8.155 -> 7.550 | **-0.605** | 项目主基线 |
| [Hallo2](https://arxiv.org/abs/2410.07718)<br>THG/I2V 人像动画 | ICLR 2025（arXiv 2024） | 5.916 -> 5.846 | -0.070 | 8.830 -> 8.543 | -0.287 | 为公平比较强制关闭 `Kim_Vocal_2` 人声分离 |
| [JoyVASA](https://arxiv.org/abs/2411.09209)<br>TFG/THG 人像动画 | arXiv 2024 | 4.936 -> 5.591 | +0.655 | 9.499 -> 9.282 | -0.217 |  |
| [LatentSync](https://arxiv.org/abs/2412.09262)<br>TFG/V2V 潜空间唇形同步 | arXiv 2024 | 4.653 -> 4.581 | -0.072 | 8.894 -> 8.768 | -0.126 |  |
| [EchoMimic V2](https://arxiv.org/abs/2411.10061)<br>TFG/THG 人像动画 | CVPR 2025（arXiv 2024） | 2.980 -> 3.165 | +0.185 | 11.206 -> 11.054 | -0.152 | 使用合成静态姿态，不能代表真实驱动视频能力 |
| [SoulX-FlashHead](https://arxiv.org/abs/2602.07449)<br>THG/流式 I2V | arXiv 2026 | 6.034 -> 6.774 | **+0.739** | 8.939 -> 8.345 | **-0.594** | |
| [AVTR-1](https://github.com/avaturn-live/avtr-1)<br>THG/流式 I2V | 官方开放权重模型，2026 | 5.460 -> 5.788 | +0.327 | 8.830 -> 8.632 | -0.198 |  |
| [AvatarForcing](https://arxiv.org/abs/2603.14331)<br>THG/流式 I2V | arXiv 2026 | 2.687 -> 2.577 | -0.110 | 11.275 -> 11.773 | +0.498 | TTS 未提升；整体分数比较低 |
| [LeapTalk](https://arxiv.org/abs/2511.23199)<br>THG/流式 I2V | ViBT，arXiv 2025；代码/权重 2026 | 6.222 -> 7.221 | **+0.999** | 8.979 -> 8.111 | **-0.868** | 基于 SoulX-FlashHead |

## 2. HDTF 英文实验

| 协议 | Natural C -> TTS C | ΔC | Natural D -> TTS D | ΔD | 结论 |
|---|---:|---:|---:|---:|---|
| 完整音频 + 固定肖像图 | 6.451 -> 7.339 | **+0.888** | 7.694 -> 7.721 | +0.027 | C 有方向性提升 |

## 3. 多语言实验

多语言实验使用同一套 13 样本 `self_clone` 流程：每种语言分别用自然音频作为 Qwen3-TTS
参考音频和 Natural 条件，用对应文本生成 TTS，再由 Ditto 生成视频并用 SyncNet V2 评分。

| 语言与数据集 | 有效配对 | Natural C -> TTS C | ΔC | Natural D -> TTS D | ΔD | 结论 |
|---|---:|---:|---:|---:|---:|---|
| English<br>`Effect AI Scripted Speech 1.0 - English` | 13/13 | 5.547 -> 6.176 | **+0.629** | 8.533 -> 7.444 | **-1.090** | 有提升 |
| German<br>`Common Voice Spontaneous Speech 4.0 - German` | 13/13 | 5.647 -> 6.537 | **+0.890** | 7.707 -> 7.589 | -0.118 | 有提升 |
| Italian<br>`Italian TTS - female voice` | 13/13 | 6.963 -> 7.222 | +0.259 | 7.530 -> 7.343 | -0.188 | 有提升|
| Spanish<br>`Common Voice Spontaneous Speech 4.0 - Spanish` | 11/13 | 5.151 -> 5.188 | +0.036 | 7.761 -> 8.023 | +0.261 | 基本无提升，D 变差 |
| Korean<br>`Common Voice Scripted Speech 26.0 - Korean` | 13/13 | 5.949 -> 6.691 | **+0.742** | 8.012 -> 7.540 | **-0.472** | 有提升 |
| Japanese<br>`CSS10-Multilingual-LJSpeech` | 13/13 | 5.590 -> 5.541 | -0.049 | 8.210 -> 8.396 | +0.186 | 未提升；逐样本波动相互抵消 |
| Russian<br>`CSS10-Multilingual-LJSpeech` | 13/13 | 4.828 -> 5.900 | **+1.072** | 8.430 -> 8.101 | **-0.329** | 两个指标方向均改善 |