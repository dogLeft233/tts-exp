---
title: V100 本地部署 IndexTTS2 与 GPT-SoVITS
type: task
permalink: tts-exp/tasks/v100-本地部署-index-tts2-与-gpt-so-vits
assigned_to: claude
current_step: '2'
status: done
tags:
- task
- deployment
- tts
---

# V100 本地部署 IndexTTS2 与 GPT-SoVITS

## Observations
- [description] 在本地 V100-SXM2-16GB 工作站上按官方文档部署 IndexTTS2 与 GPT-SoVITS,完成 voice cloning 生成测试,部署文档写入 bm docs/deployment/
- [status] done
- [assigned_to] claude

## Steps
1. [x] 调研官方仓库与文档 (IndexTTS2=index-tts/index-tts, GPT-SoVITS=RVC-Boss/GPT-SoVITS)
2. [x] clone 两仓库到 /home/wjj/tts-audio/
3. [x] IndexTTS2: uv sync --extra webui (跳过 accel/deepspeed) + IndexTeam/IndexTTS-2 权重 + gpu_check + 推理验证 (4.7s 音频, fp32)
4. [x] GPT-SoVITS: conda env GPTSoVits (py3.10, torch 2.5.1+cu124, mkl<2024) + lj1995/GPT-SoVITS 5.4G + G2PW 软链 + inference_cli 验证 (RTF~0.3)
5. [x] 部署文档写入 bm: docs/deployment/index-tts-2.md, gpt-sovits.md, DEPLOY.md 加本地 TTS 表

## 结果

- IndexTTS2 ✅: /home/wjj/tts-audio/index-tts, V100 入口 `IndexTTS2(..., use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)`, 选 v2 (非 2.5, 用户决策: 2.5 克隆效果评价反而不如 2 且 V100 无 bf16)
- GPT-SoVITS ✅: 输出响度跟随 ref (AISHELL 轻录音 → -48dBFS 非静音), 下游用需响度归一化
- 文档: [[index-tts-2]] [[gpt-sovits]]
- 后续可选: 把两模型接入 tts-exp scripts/tts/ factory (TTSProvider ABC)
