---
title: LRS3 纯 SyncNet WavLM-HiFi-GAN 微调
type: experiment
permalink: tts-exp/experiments/lrs3-纯-sync-net-wav-lm-hi-fi-gan-微调
note_type: experiment
tags:
- lrs3
- syncnet
- wavlm
- hifigan
- wav2lip
- audio-replacement
---

# LRS3 纯 SyncNet WavLM-HiFi-GAN 微调

## Context

本实验验证：使用自然视频作为视觉监督，是否可以仅靠纯 SyncNet loss 微调 WavLM 编码器和 HiFi-GAN 解码器，使输出音频驱动冻结 Wav2Lip 产生独立的视觉同步增益，并在把视频音轨替换回 untouched natural audio 后保留该增益。最终结果表明，固定坐标 SyncNet 训练 proxy 可以明显下降，但没有转化为自然视频兼容性或 Wav2Lip 下游视觉增益。

## Protocol

- 数据：LRS3 manifest 前约 50 条记录；按 source group 划分，train 33 条 / 27 groups，validation 10 条 / 6 groups，test 7 条 / 5 groups。
- 音频模型：WavLM-Large layer 6 + 配套 prematched HiFi-GAN；两者均参与微调，最终 checkpoint 只恢复 WavLM layer 6 和 HiFi-GAN。
- 训练目标：纯 fixed-coordinate cosine SyncNet V2 audio-embedding loss；没有 waveform reconstruction、mel、GAN、text 或 image loss。
- SyncNet 视觉 embedding 预计算并 detach；SyncNet 参数保持 `requires_grad=False` 和 `eval()`；候选 waveform 到 MFCC、SyncNet audio branch 的梯度保留。
- MFCC 严格匹配官方 SyncNet V2：16 kHz、25 FPS、20 个 MFCC frame 对 5 个视频 frame、4:1 映射、PCM scale `32768`、内部 float64 silence-floor parity。
- 训练使用固定音频坐标 `audio index = visual index + 2`，避免把不可微的 file-level 全局 offset 搜索放入 loss。
- 下游评估使用冻结 Wav2Lip 和官方 file-level SyncNet V2，包含 candidate-driver、strict untouched-natural-audio replacement，以及固定 natural video 的音频兼容性比较。
- replacement 使用 untouched natural WAV；视频流保留，mux 后 decoded PCM 必须与输入音频 byte-level 完全一致。

## Training optimization

最终训练 60 steps，loss 从 `0.693804` 降至 `0.469400`，绝对下降 `0.224404`，相对下降约 `32.3%`。WavLM layer 6 和 HiFi-GAN 都有非零梯度，SyncNet 参数没有梯度，因此固定坐标 SyncNet 训练目标确实被优化。

checkpoint：

```text
runs/lrs3_wavlm_hifigan_syncnet_only_fp64mfcc_20260826/checkpoint.pt
```

SHA256：`ab2fc1703eb80fca81a49204728fdd601be6d4d6e4beea02a8a4221accbbccf8`

## Train-split downstream result

训练集评分为 33 条样本、27 个 source groups、99 个 score cells；99/99 strict mux 通过 PCM exact-match。

| 条件 | Sync-C mean | Sync-D mean |
|---|---:|---:|
| natural video + natural audio | 6.945 | 7.449 |
| finetuned video + finetuned audio | 6.870 | 7.532 |
| finetuned video + untouched natural audio | 6.630 | 7.836 |

相对 natural baseline：

- candidate-driver：Sync-C delta `-0.075`，source-group bootstrap 95% CI `[-0.278, 0.126]`；Sync-D benefit `-0.083`，CI `[-0.242, 0.098]`；C/D/joint wins 为 `13/10/10`。
- strict natural-audio replacement：Sync-C delta `-0.314`，CI `[-0.412, -0.231]`；Sync-D benefit `-0.387`，CI `[-0.482, -0.301]`；C/D/joint wins 为 `1/2/0`。
- natural baseline 与 replacement 的 file-level AV offset 全部为 `-2`；candidate-driver 32 条为 `-2`、1 条为 `-3`。

训练集本身也没有出现 candidate-driver 的平均 SyncNet 提升；换回 untouched natural audio 后下降更明显。

训练集生成的微调音频位于：

```text
runs/lrs3_wavlm_hifigan_syncnet_only_fp64mfcc_eval_train_syncnetenv_20260826/audio/finetuned/
```

逐样本路径、SHA256、sample count 和 QC 信息记录在该目录下的 `summary.json` 的 `generated_audio` 字段中。

## Fixed-natural-video audio compatibility

为区分“音频本身是否适配自然视频”和“候选音频驱动后视频是否变化”，在同一批 33 条 train natural videos 上分别使用 natural audio、未微调 direct-resynthesis audio 和 finetuned audio 做官方 file-level SyncNet 评分。三种条件使用相同 natural video，且 33/33 mux 的 decoded PCM 均与各自源音频完全一致。

| 条件 | Sync-C mean | Sync-D mean |
|---|---:|---:|
| natural video + natural audio | 6.944758 | 7.448727 |
| natural video + 未微调 WavLM layer 6 + HiFi-GAN audio | 6.842182 | 7.517848 |
| natural video + finetuned WavLM + HiFi-GAN audio | 6.573636 | 7.790273 |

直接比较 finetuned − unfinetuned：

- Sync-C delta：`-0.268545`，source-group bootstrap 95% CI `[-0.396438, -0.143574]`。
- Sync-D benefit：`-0.272424`，CI `[-0.382032, -0.158676]`。
- finetuned 胜出：Sync-C `7/33`、Sync-D `6/33`、两项同时胜出 `6/33`。
- 未微调 audio 的 offset 全部为 `-2`；finetuned audio 为 `-2`（27/33）和 `-3`（6/33）。

因此，微调后的音频不仅没有改善自然视频上的兼容性，还显著劣于未微调的 direct-resynthesis audio。结果目录：

```text
runs/lrs3_wavlm_hifigan_syncnet_only_fp64mfcc_eval_train_natural_video_unfinetuned_audio_20260827/summary.json
```

## Held-out audit

同一 checkpoint 在 7 条 held-out 样本上完成 21 个 score cells，21/21 strict mux PCM exact-match，所有三格 AV offset 均为 `-2`。

- candidate-driver 相对 natural baseline：Sync-C delta `-0.367`，95% CI `[-0.749, -0.002]`；Sync-D benefit `-0.117`，CI `[-0.473, 0.239]`。
- strict natural-audio replacement：Sync-C delta `-0.194`，CI `[-0.341, -0.049]`；Sync-D benefit `-0.266`，CI `[-0.401, -0.128]`。

held-out 结果与训练集方向一致：candidate-driver 没有可靠增益，replacement 没有保留增益。

## Why loss decreased but downstream score did not improve

- `[insight]` 训练 loss 只约束“生成 waveform 经过 MFCC 和冻结 SyncNet audio branch 后的 embedding”与自然视频 visual embedding 在固定坐标上的 cosine 距离；它没有直接约束 Wav2Lip 生成的视频画面，也没有约束音频在换回 natural audio 后仍保持视觉收益。
- `[insight]` 训练使用固定 offset 和固定局部窗口，而官方 file-level scorer 会对整段音视频进行窗口搜索并估计 AV offset。微调音频在 6/33 条自然视频上从 offset `-2` 变成 `-3`，说明优化过程改变了时间结构；固定坐标 loss 下降不等价于官方整段评分提高。
- `[learning]` WavLM layer 6 与 HiFi-GAN 共同微调，可以通过改变局部频谱、能量包络、相位或短时节奏来降低冻结 SyncNet 的 audio-side proxy，而不必保留 natural waveform 与自然唇动之间的可替换关系。
- `[problem]` 视觉 embedding 虽然来自自然视频，但训练中没有经过 Wav2Lip 的 differentiable video path；因此 loss 无法知道某种 audio 改变会让冻结 Wav2Lip 生成怎样的画面，也无法惩罚 audio/video co-adaptation。
- `[learning]` 训练集上的 loss 下降证明的是 proxy optimization，不是 independent visual control。即使不考虑泛化，train split 的 candidate-driver 也没有提高；在固定 natural video 上，finetuned audio 还显著劣于未微调 audio。
- `[problem]` 所有 strict mux 的 decoded PCM exact-match 都通过，因此当前下降不能归因于音频封装、重新编码或 mux 改变了音频；主要问题是训练目标与科学下游终点不一致。

## Interpretation

- `[result]` 纯 SyncNet loss 从 `0.693804` 降至 `0.469400`，说明模型成功优化了训练 proxy。
- `[result]` 在固定 natural video 上，finetuned audio 相对未微调 direct-resynthesis audio 的 Sync-C delta 为 `-0.269`，Sync-D benefit 为 `-0.272`，两项 bootstrap CI 均不包含 0。
- `[insight]` proxy loss 的明显下降没有转化为 Wav2Lip 下游 SyncNet 增益；训练集 candidate-driver 平均仍略差于 natural baseline。
- `[problem]` strict natural-audio replacement 在训练集和 held-out 上都明显变差，未出现 replacement-safe visual gain。
- `[learning]` 仅用自然视频监督、冻结 SyncNet、共同微调 encoder/decoder，仍可能学习到与 candidate waveform 绑定的音频/视频 proxy，而不是脱离音轨的独立视觉控制。
- `[decision]` 不能把训练 loss 下降作为科学成功标准；后续判断必须继续以 candidate-driver、固定 natural-video audio compatibility 和 strict untouched-natural-audio replacement 三个下游终点为准。

## Relations

- relates_to [[LRS3 WavLM-HiFi-GAN direct resynthesis replacement]]
- relates_to [[Encoder-only SyncNet 微调与固定视频迁移性]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- depends_on [[tts-exp|Natural reference audio available at inference]]