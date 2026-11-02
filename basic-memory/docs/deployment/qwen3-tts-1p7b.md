---
title: qwen3-tts-1p7b
type: note
permalink: tts-exp/docs/deployment/qwen3-tts-1p7b
status: active
verified: '2026-10-23'
tags:
- deployment
- qwen3
- v100
---

# Qwen3-TTS 1.7B Base：V100 完整模型部署

✅ 2026-10-23 完成下载、官方逐文件 SHA-256 核验、独立环境检查及一次完整声音克隆推理。模型为 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`，包含文字到语音编码生成、说话人条件与音频编解码能力。0.6B-Base 也完整；1.7B 不是云端 `qwen-audio-3.1-tts-plus` 的已证实同权重版本。

## 执行位置与版本

- 主机：`wjj@192.168.50.2`，Tesla V100-SXM2-16GB。
- 独立解释器：`/home/wjj/.venvs/qwen3-tts-1p7b/bin/python`。
- 权重：`/home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-1.7B-Base/`。
- 环境从既有 qwen3 复制独立包文件并修正 41 个文本入口；pip freeze 相等，`pip check` 无冲突；没有修改旧环境。Python 3.12.3、torch 2.6.0+cu124、qwen-tts 0.1.1、transformers 4.57.3。
- 实际通过的配置：生成模型及 codec 都显式 FP32、`attn_implementation='eager'`；单样本、无训练、无 FlashAttention 2。
- 权重来自 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)。下载文件按各自 revision 固定，完整大小与 SHA 均核对；逐文件清单见下方 run。主模型 revision `53a097f58d6aa0d688ae0985b4c5e34cfdad2626`。
- 主模型 SHA：`38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c`；codec SHA：`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`，与现有 0.6B codec 相同。旧 0.6B 主模型 SHA 仍为 `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6`。

## 实测与证据

Run：`runs/qwen3_tts_1p7b_deployment_20261023_attempt01/`，Ubuntu 与 E 盘均有完整小型验证产物；大模型权重仅部署在 Ubuntu，未声明已网盘备份。

以已有 BAC009S0002W0122 的 24kHz 参考和对应转写做完整 self-ICL：参考编码、说话人条件、文字生成及声音解码均实际执行。输出文字为“而对楼市成交抑制作用最大的限购”。seed=20261023，temperature=0.9、top_k=50、top_p=1、max_new_tokens=512、non_streaming_mode=False。

| 检查 | 实测 |
|---|---|
| 波形 | 24kHz、单声道、FLOAT WAV、3.920 秒；非零、全有限值；ffprobe 通过 |
| 模型加载 | 2.856 秒；不含进程导入和前置文件哈希 |
| 参考 prompt 提取与合成 | 13.034 秒，RTF=3.325；一条短句的单次结果 |
| PyTorch 峰值 allocated / reserved | 8,905,551,360 / 9,107,931,136 字节（约 8.29 / 8.48 GiB）；不等于整卡总占用 |
| 结束状态 | smoke 实际 exit 0；退出后整卡约 322MiB，无本次推理进程 |
| 校验脚本测试 | 两项通过：导入不启动部署/GPU；损坏的本地权重文件会被拒绝 |

`smoke_receipt.json` 记录参数、环境、参考绑定、源码及音频 SHA；`weights_manifest.json` 记录所有模型文件。输出 `smoke_full_icl_float.wav` 的 SHA 为 `806e682360e081037aea4500cebd06391910b30568a4aeb093a42c19dff8d25b`，E 盘副本读回一致。

主文件单连接下载偏慢后，仅中断下载进程，保留前缀并进行 32 路 Range 下载；最终对完整 3,857,413,744 字节核验官方 SHA。原连接日志、分段下载及环境迁移脚本均保留，未进行科学结果重抽。

**同日批量推理补充：** 同云端10人、每人3个固定seed的完整self-ICL已实际完成30/30、actual exit0，全部EOS，671.887秒；PyTorch峰值allocated 10,017,049,088字节（9.33GiB）、reserved 15,896,412,160字节（14.80GiB）。这比上方单条smoke占用更高，不能把8.48GiB当作批量上限。全部参考编码、seed、实际解码参数及保存EOS/code经独立CPU核对，与0.6B对应参考/设置一致；原始证据在 `runs/qwen17_level_comparison10_20261023_attempt01/generation/`，后续视频与评分在attempt02。该批次支持V10016GB运行此设置，不包含人工音质验证；评分结论见对应实验报告。

## 调用方式

使用上面的独立解释器执行以下 Python；参考文字必须与参考录音对应。与验证相同，显式将 codec 也加载为 FP32。

```python
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

path = '/home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-1.7B-Base'
model = Qwen3TTSModel.from_pretrained(
    path, local_files_only=True, device_map='cuda:0',
    dtype=torch.float32, attn_implementation='eager')
model.model.speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
    path + '/speech_tokenizer', local_files_only=True,
    device_map='cuda:0', dtype=torch.float32, attn_implementation='eager')
model.model.eval().requires_grad_(False)
model.model.speech_tokenizer.model.eval().requires_grad_(False)
with torch.inference_mode():
    waves, sr = model.generate_voice_clone(
        text='需要生成的文字', language='Chinese',
        ref_audio='/absolute/path/reference.wav', ref_text='参考录音的实际文字',
        x_vector_only_mode=False, non_streaming_mode=False, max_new_tokens=512)
sf.write('output.wav', waves[0], sr, subtype='FLOAT')
```

部署验证脚本为 `scripts/ops/deploy_qwen3_tts_1p7b_20261023.py`，其 `smoke` 子命令拒绝覆盖已存在的验证音频和 receipt。它用于部署核验，不自动纳入已有实验或替换默认0.6B provider。后续30条批量推理已如上单列；人工音质未验证，没有启动训练。

## Relations

- relates_to [[qwen3-tts]]
