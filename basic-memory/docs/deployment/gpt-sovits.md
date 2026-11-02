---
title: gpt-sovits
type: note
permalink: tts-exp/docs/deployment/gpt-sovits
tags:
- tts
- gpt-sovits
- 部署
- deployment
- v100
---

# GPT-SoVITS 本地 V100 部署记录

RVC-Boss 开源的少样本声音克隆 TTS (1 分钟语音即可克隆),v2 预训练模型零样本推理。V100 fp16/fp32 均可跑。

## 状态

- **状态**: ✅ 本地 V100 推理验证通过 (2026-08-14)
- **模型**: `lj1995/GPT-SoVITS` 预训练全家桶 (27 文件 5.4G)
- **官方仓库**: https://github.com/RVC-Boss/GPT-SoVITS
- **官方文档**: `docs/cn/README.md` (测试环境 py3.10 + torch 2.5.1 + cu124)

## 环境

| 组件 | 版本 |
|------|------|
| 位置 | `/home/wjj/tts-audio/GPT-SoVITS` (git clone) |
| conda env | `GPTSoVits` (python 3.10, `/home/wjj/miniconda3/envs/GPTSoVits`) |
| torch | 2.5.1+cu124 (**conda 装**, pip 装不了 — nvidia-cudnn-cu12==9.1.0.70 已从 PyPI 下架) |
| mkl | 必须 `mkl<2024` (默认 2024.x 导致 libtorch_cpu undefined symbol iJIT_NotifyEvent) |
| 依赖 | `pip install -r extra-req.txt --no-deps` + `pip install -r requirements.txt` |

## 权重位置

- 预训练: `GPT_SoVITS/pretrained_models/` — chinese-hubert-base, chinese-roberta-wwm-ext-large, gsv-v2final-pretrained (s1bert25hz ckpt + s2G2333k + s2D2333k), gsv-v4-pretrained, bigvgan, g2pw-chinese 等 5.4G
- G2PW: `GPT_SoVITS/text/G2PWModel/` — repo 自带骨架 (config.py),大文件从 pretrained_models/g2pw-chinese 逐文件软链进来 (`ln -sf ../../pretrained_models/g2pw-chinese/* .`)
- 下载: `snapshot_download('lj1995/GPT-SoVITS', local_dir='GPT_SoVITS/pretrained_models')` (走代理直连 HF, hf-mirror 会 LocalEntryNotFound)

## 使用 (CLI 推理)

```bash
cd /home/wjj/tts-audio/GPT-SoVITS
PYTHONPATH=. /home/wjj/miniconda3/envs/GPTSoVits/bin/python GPT_SoVITS/inference_cli.py \
  --gpt_model GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
  --sovits_model GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth \
  --ref_audio ref.wav --ref_text <(echo "参考音频文本") --ref_language 中文 \
  --target_text <(echo "目标文本") --target_language 中文 --output_path out_dir
```

- `is_half` 默认 True (V100 fp16 OK, fp32 也验证过)
- API 服务: `api_v2.py` (FastAPI); WebUI: `webui.py`
- ref 音频建议 3-10s; inference_cli 的 ref/target 是**文件路径**不是文本

## 性能 (V100 fp16)

- v2 zero-shot: 4s 音频 ~1.2s 生成 (RTF ~0.3), 模型加载 ~30s
- 输出 32000Hz

## 注意事项 (gotchas)

- **输出幅度跟随 ref**: AISHELL ref (-38dBFS) → 输出 -48dBFS, peak 0.02 看似静音实为正常语音 (onset 包络 1.2); 用轻音量 ref 时下游需响度归一化
- **torch 必须conda装**: pip cu124 wheel 依赖被下架的 cudnn 9.1.0.70
- **mkl<2024**: 否则 import torch 即崩 (iJIT_NotifyEvent)
- `install.sh` 不建 conda env, 需先激活; 且 socks:// 代理 scheme 会让其 httpx 崩
- 运行时 `unset ALL_PROXY` (socks scheme 炸 httpx), HTTP 代理用 http:// 形式
- PYTHONPATH 必须指向 repo 根 (tools/GPT_SoVITS 包内相对导入)

## Observations

- [location] repo at /home/wjj/tts-audio/GPT-SoVITS, env GPTSoVits (py3.10), weights 5.4G #deployment
- [decision] torch 2.5.1+cu124 conda 安装 + mkl<2024 pin (pip 路不可行) #tooling
- [decision] G2PW 用软链从 pretrained_models/g2pw-chinese 挂到 text/G2PWModel #tooling
- [insight] 输出响度跟随 ref 音量, AISHELL 轻录音 → 输出 -48dBFS 非静音 bug #tts
- [interface] inference_cli.py 参数是文件路径; PYTHONPATH 需指向 repo 根 #tts
- [verification] 2026-08-14 AISHELL sample1 ref 克隆生成 3.9s 中文, RTF~0.3 #verification
- [learning] install.sh 期望已激活 env; socks:// 代理 scheme 全链路踩坑 #gotcha

## Relations

- part_of [[tts-exp]]
- relates_to [[qwen3-tts]]
- relates_to [[index-tts-2]]