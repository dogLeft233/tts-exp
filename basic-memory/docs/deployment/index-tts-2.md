---
title: index-tts-2
type: note
permalink: tts-exp/docs/deployment/index-tts-2
tags:
- tts
- indextts
- 部署
- deployment
- v100
---

# IndexTTS-2 (Bilibili) 本地 V100 部署记录

Bilibili 开源的工业级零样本 TTS,情感表现力强,支持时长控制。V100 (Volta sm_70) 上用 fp32 路径部署 (官方 API 显式支持)。

## 状态

- **状态**: ✅ 本地 V100 推理验证通过 (2026-08-14)
- **模型**: `IndexTeam/IndexTTS-2` (选 v2 而非 v2.5 — 社区反馈 v2.5 克隆效果反而不如 v2; 且 v2.5 主推 bf16, V100 不支持)
- **官方仓库**: https://github.com/index-tts/index-tts
- **协议**: bilibili 模型使用协议 (限研究/非商用)

## 环境

| 组件 | 版本 |
|------|------|
| 位置 | `/home/wjj/tts-audio/index-tts` (git clone, uv 管理) |
| venv | `index-tts/.venv` (python 3.11, `uv sync --extra webui`) |
| 安装 | 只装 webui extra; **跳过 accel (flash-attn) / deepspeed / torch_compile** — V100 全不支持 |
| uv | 0.12.3 (`~/.local/bin/uv`) |

## 权重位置

- 主权重: `index-tts/checkpoints_2/` (gpt.pth 3.5G + s2mel.pth 1.2G + qwen0.6bemo4-merge/ 1.2G + feat1/feat2.pt + bpe.model, 共 ~6GB)
- 首次运行自动追加小模型 → `checkpoints_2/hf_cache/` (semantic_codec 169M, bigvgan_v2_22khz, campplus, w2v-bert-2.0, 共 ~1.5G)
- 下载: `uv run hf download IndexTeam/IndexTTS-2 --local-dir checkpoints_2` (正确 repo id 是 **IndexTTS-2 带连字符**, 不是 IndexTTS2)

## 使用 (V100 关键参数)

```python
# 必须 cd /home/wjj/tts-audio/index-tts (editable 安装靠 cwd)
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2",
                use_fp16=False,        # V100 fp32 稳 (fp16 未验证)
                use_cuda_kernel=False, # 无编译内核
                use_deepspeed=False)
tts.infer(spk_audio_prompt="ref.wav", text="要合成的中文",
          output_path="out.wav", verbose=True)
```

- CLI: `uv run indextts2 -c checkpoints_2/config.yaml -m checkpoints_2 ref.wav "文本" out.wav`
- WebUI: `uv run webui.py` (默认 2.5, 切 v2 需在页面选)

## 性能 (V100 fp32)

- 加载+首次推理: ~13s (4.7s 音频)
- 输出: 22050Hz, 幅度跟随 ref 音量

## 注意事项 (V100/gotchas)

- **repo id 陷阱**: `IndexTeam/IndexTTS-2` (不是 IndexTTS2, 401/Invalid username 就是 id 错)
- ModelScope 对 `nvidia/bigvgan` 报 404 后自动 fallback hf-mirror — 非致命
- 首次运行联网下小模型 (~1.5G, hf-mirror ~1.2MB/s, 需几分钟); 预先 `export HF_ENDPOINT=https://hf-mirror.com`
- `uv run python /tmp/xx.py` 偶发找不到 indextts 模块 — 用 `.venv/bin/python` 且 cwd 在 repo 根
- 环境变量 `ALL_PROXY=socks://` scheme 会让 httpx 崩 — 用 `http://` 或 unset
- 批量大文件下载: `uv pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` (5x 提速)

## Observations

- [location] repo at /home/wjj/tts-audio/index-tts, weights at checkpoints_2/ (~7.5G 总) #deployment
- [decision] 选 v2 不选 v2.5 (克隆质量社区评价 + V100 无 bf16) #decision
- [decision] 跳过 flash-attn/deepspeed/torch_compile extras — V100 sm_70 不支持 #hardware
- [interface] IndexTTS2(cfg_path, model_dir, use_fp16=False, use_cuda_kernel=False, use_deepspeed=False) 是 V100 入口 #tts
- [verification] 2026-08-14 AISHELL ref 5.2s 克隆生成 4.7s 音频, onset 包络正常 #verification
- [learning] repo id 是 IndexTTS-2 带连字符; 小模型首跑自动下载到 checkpoints_2/hf_cache #gotcha

## Relations

- part_of [[tts-exp]]
- relates_to [[qwen3-tts]]
- relates_to [[gpt-sovits]]