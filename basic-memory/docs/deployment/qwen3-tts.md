---
title: qwen3-tts
type: note
permalink: tts-exp/docs/deployment/qwen3-tts
tags:
- tts
- qwen3
- 部署
- deployment
- faster-qwen3
- v100
---

# Qwen3-TTS (faster_qwen3 0.6B) 本地部署记录

2026-10-23 已另行部署完整 [1.7B-Base](qwen3-tts-1p7b.md)，独立环境 `/home/wjj/.venvs/qwen3-tts-1p7b`；V100 FP32/eager 完整 self-ICL 推理通过。下文保留 0.6B 的既有部署记录；1.7B 不替换当前实验默认提供器。

本地 V100 (Tesla V100-SXM2-16GB) 上部署了 Qwen3-TTS 0.6B Base 自克隆 TTS 模型,用于项目流水线 02_tts 步骤与训练增强头数据生成。环境、权重与远程服务器 (RTX 4080, ditto conda env) 逐版本对齐,两个 safetensors 权重文件 md5 与服务器完全一致。

## 状态

- **状态**: ✅ 本地 V100 推理验证通过 (2026-08-11)
- **模型**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (0.6B ICL 自克隆 TTS)
- **远程参考**: 服务器 `/root/autodl-tmp/models/Qwen3-TTS-12Hz-0.6B-Base` (ditto conda env)

## 环境 (本地 venv, 与服务器版本对齐)

| 组件 | 版本 |
|------|------|
| venv | `~/.venvs/qwen3` (python 3.12) |
| torch | 2.6.0+cu124 (V100 必须 cu124, CUDA 13 不支持 sm_70) |
| faster-qwen3-tts | 0.3.0 |
| qwen-tts | 0.1.1 |
| transformers | 4.57.3 |
| librosa / soundfile | 0.11.0 / 0.14.0 |
| numpy / safetensors | 2.0.1 / 0.8.0 |

## 权重位置

- 本地: `/home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-0.6B-Base/`
- 主模型 `model.safetensors` 1.83GB + `speech_tokenizer/model.safetensors` 682MB, md5 与服务器一致
- 下载方式: curl 直连 hf-mirror resolve 端点 (huggingface_hub 客户端不可用)

## 使用 (与项目 scripts/common/tts/faster_qwen3.py 一致)

```python
from faster_qwen3_tts import FasterQwen3TTS
model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
audio_list, sr = model.generate_voice_clone(text=..., language="Chinese",
                                            ref_audio=ref_wav, ref_text=...)
```

- ICL 模式必须传 `ref_text`, 否则报 `ref_text is required when x_vector_only_mode=False`
- `model.synthesize()` 不存在, 接口是 `generate_voice_clone` / `generate`
- 本地 config override: `scripts/configs/tedlium3_local_v100.yaml` (model_id 指向本地路径)

## 性能 (V100 16GB)

- 加载: ~7s
- 生成 6.4s 中文语音: ~12s (含 CUDA graph 预热 3 次, 捕获成功)
- CUDA graph 捕获在 V100 上正常 (faster-qwen3-tts 0.3.0)
- 批量 TTS (TED-LIUM 3): ~690 条/小时 (~5s/条)

## 注意事项

- V100 不支持 bf16, 用 fp16 即可
- 本地无 conda, 用 venv; 系统需 `python3.12-venv` 包
- 网络: hf-mirror resolve 端点直连

## Observations

- [location] venv at ~/.venvs/qwen3 (python 3.12), torch 2.6.0+cu124 #deployment
- [location] weights at /home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-0.6B-Base/ #deployment
- [decision] V100 (sm_70) 必须用 cu124 torch 构建, CUDA 13 不支持 Volta 架构 #hardware
- [decision] 模型下载用 curl 直连 hf-mirror resolve 端点, huggingface_hub 客户端不可用 #network
- [interface] generate_voice_clone(text, language, ref_audio, ref_text) 是 ICL 入口, ref_text 必传 #tts
- [usage] ICL 自克隆模式 ref_text == text (同 02_tts.py 约定) #tts
- [performance] 批量生成速率 ~690 条/小时 (V100, ~5s/条) #benchmark
- [constraint] V100 不支持 bf16, 推理用 fp16 #hardware
- [verification] 权重 md5 与服务器一致 (3e597f6c... / ccf5bd95...) #reproducibility

## Relations

- part_of [[tts-exp]]
- used_by [[tedlium3-tts-augmentation]]
- deployed_with [[syncnet]]
- relates_to [[local-network-proxy-notes]]
