# Model-aware routing v1

状态：FROZEN_PROTOTYPE（2026-10-13）

## 目的

把自然音频增强波形选择与 TFG renderer 解耦。增强 WAV 只作为视频生成输入；评分和最终成片换轨始终使用未修改的原自然 PCM 与原始采样时钟。

## 路由表

| model_id / alias | generator-control method | fallback |
|---|---|---|
| `musetalk_fixed_portrait_attempt16`, `musetalk`, `muse_talk` | `exact_k15_shape_env_pos025_neg025`（explicit11，+0.25 dB positive flux / −0.25 dB negative flux） | `rms_fixed` |
| `wav2lip_gan_bac9a81e`, `wav2lip` | `rms_fixed` | — |
| `ditto_hubert_fixed_portrait_attempt18`, `ditto` | `rms_fixed` | — |
| unknown | `rms_fixed` | — |

匹配顺序是 exact model ID → casefold alias → default。候选运行时缺少 PyTorch、source pin/shape digest 漂移、输出形状或有限性异常时 fail-closed 到 RMS。receipt 同时记录 `selected_method_id` 与实际 `applied_method_id`。

## 调用

```bash
python -m scripts.audio_enhancement.cli natural.wav generator.wav \
  --model-id musetalk --natural-reference-wav natural_16k.wav
# 将 generator.wav 交给 TFG renderer 后：
python -m scripts.audio_enhancement.cli natural.wav generator_again.wav \
  --model-id musetalk --natural-reference-wav natural_16k.wav \
  --generated-video generated.mp4 --mux-output final.mkv
```

第二次调用会丢弃生成视频自带音频，使用原自然 WAV 做 lossless PCM stream-copy mux；它验证解码 PCM、视频 elementary stream、帧时间戳和样本时钟。`--generated-video` 与 `--mux-output` 必须成对出现；CLI 不负责启动 renderer。

## 固定来源

- route config SHA256：`886f75c03977ab7d8ac7a5caecd568421c9a46cbf455d6b9cef87a9bf24fb00`
- explicit11 profile：`exact_k15_shape_env_pos025_neg025`，profile SHA256 `60f263846f7bfd0a5af9ad750d6629f12b248dca2725e896965c8b2b8541fa9f`
- main holdout plan SHA256 `007495aa3b4cf1fc9b20f506c33593da8bb95e54ea1e0914ac56f3d9a21cad3b`
- Ditto repair plan SHA256 `c71e47e7acc2eb2ec92062b6fc0d4c882c96309af2ff770ef40db7a5144213a5`
- source-conditioned head pins are embedded in `model_aware_routing.py` and checked before the candidate provider runs.

## 证据与边界

10 个未见 AISHELL 说话人、16 kHz mono、25 fps 的三模型 holdout 给出：Wav2Lip candidate−RMS full/guard20 `−0.008916/−0.005989`，MuseTalk `+0.014064/+0.025367`，Ditto `0/0`。因此当前交付是模型感知安全路由原型，不是单一候选跨所有 TFG 的统一胜出头。真实 torch CPU qualification 已执行；远端已有的生成→原 PCM 换轨→SyncNet smoke 是 engineering-only，不作为效果门。

覆盖边界：实测只有上述三个固定 model IDs、静态 portrait、25 fps、10 个未见说话人。未知模型只有 RMS fail-closed 契约；`shared_audio_head.pt` 仍为 `FIT_ONLY_COMPLETE` research-only，未接入生产路由。
