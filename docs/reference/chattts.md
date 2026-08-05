# ChatTTS 使用说明

ChatTTS 是 2noise 团队开发的对话场景文本转语音模型，部署在项目评测服务器上用于批量 TTS 生成实验。

## 官方信息

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/2noise/ChatTTS |
| 官网 | https://chattts.com |
| PyPI | `pip install ChatTTS` (v0.2.5) |
| HuggingFace | https://huggingface.co/2Noise/ChatTTS |
| 模型大小 | ~1.1GB（safetensors 格式） |
| 训练数据 | 10 万+小时中英文，开源版 4 万小时 |
| 支持语言 | 中文、英文 |
| 许可证 | 代码 AGPLv3+，模型 CC BY-NC 4.0 |
| 最低 GPU | 4GB VRAM（30 秒音频） |
| 实时率 RTF | ~0.3（RTX 4090），~0.48（RTX 4080） |

## 部署信息

**服务器**: `connect.westd.seetacloud.com:20398` (RTX 4080 32GB, CUDA 13.2)  
**登录信息**: 见 `tmp/servers.sh` 中 `CHATTTS` 段  
**模型路径**: `/root/.cache/huggingface/models--2Noise--ChatTTS/snapshots/<hash>/`  
**Python**: `/root/miniconda3/bin/python3` (Python 3.12, torch 2.8.0+cu128)

### 已安装组件

| 组件 | 版本 | 备注 |
|------|------|------|
| PyTorch | 2.8.0+cu128 | CUDA 12.8 预装 |
| torchaudio | 2.8.0+cu128 | 必须匹配 PyTorch 版本 |
| ChatTTS | 0.2.5 | PyPI 最新稳定版 |
| soundfile | 0.14.0 | 音频输出（torchaudio.save 后端兼容问题） |
| modelscope | 1.38.1 | 模型下载备选（.pt 格式不兼容 v0.2.5） |

### 模型格式兼容性

ChatTTS v0.2.5 要求 **safetensors 格式**（HuggingFace 官方格式）：
- `asset/Decoder.safetensors`, `DVAE.safetensors`, `Embed.safetensors`, `Vocos.safetensors`
- `asset/gpt/model.safetensors`, `config.json`
- `asset/tokenizer/special_tokens_map.json`, `tokenizer_config.json`, `tokenizer.json`

ModelScope 的 `pzc163/chatTTS` 使用的是旧版 `.pt` 格式，**不兼容** v0.2.5。必须从 HuggingFace 下载。

### 网络与加速

| 站点 | 可访问 | 说明 |
|------|:------:|------|
| huggingface.co | ❌ | 直连不通 |
| hf-mirror.com | ✅ | HF 镜像，但 xet 传输协议会被 401 拒绝 |
| modelscope.cn | ✅ | 模型下载可用，但格式不兼容 |

**学术加速**（AutoDL 内置）：
```bash
source /etc/network_turbo   # 设置代理 http://10.37.1.23:12798
# 用于加速 HuggingFace 下载；下载完成后建议 unset http_proxy https_proxy
```

## 单句推理

```python
import ChatTTS
import soundfile as sf

MODEL_PATH = "/root/.cache/huggingface/models--2Noise--ChatTTS/snapshots/<hash>"

chat = ChatTTS.Chat()
chat.load(source="custom", custom_path=MODEL_PATH, compile=False)

wavs = chat.infer(["你好，世界！"])
sf.write("output.wav", wavs[0], 24000)
```

**重要**: 必须用 `source="custom"` + `custom_path` 指定模型路径。默认 `source="local"` 会触发 `download_all_assets()` 尝试从 GitHub 下载 `rvcmd` 工具，GitHub 不通会超时报错。

### 推理参数

```python
# 固定音色（随机采样 → 保存复用）
rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,       # 音色 embedding
    temperature=0.3,         # 采样温度
    top_P=0.7, top_K=20,    # top-P/K 解码
)

# 句子级韵律控制
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)
# oral_(0-9) 口语化, laugh_(0-2) 笑声, break_(0-7) 停顿

# 词级精确控制（跳过 refine_text）
text = 'What is [uv_break]your favorite food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, ...)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)
```

### 可用控制 token

| Token | 含义 |
|-------|------|
| `[laugh]` | 笑声 |
| `[uv_break]` | 短停顿 |
| `[lbreak]` | 长停顿 |
| `[oral_N]` | 口语化程度 (0-9) |
| `[laugh_N]` | 笑声强度 (0-2) |
| `[break_N]` | 停顿强度 (0-7) |

## 输出格式

- **采样率**: 24000 Hz
- **声道**: 单声道
- **格式**: float32 numpy array → WAV via soundfile
- 如需转为 16kHz（与 AISHELL-1 对齐）:
  ```bash
  ffmpeg -i in.wav -ar 16000 -ac 1 out.wav
  ```

## 批量生成脚本

测试脚本位于服务器 `/tmp/test_chattts.py`：

```python
# -*- coding: utf-8 -*-
import ChatTTS
import soundfile as sf

MODEL_PATH = "/root/.cache/huggingface/models--2Noise--ChatTTS/snapshots/1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"

chat = ChatTTS.Chat()
chat.load(source="custom", custom_path=MODEL_PATH, compile=False)

texts = ["你好，ChatTTS部署测试成功！"]
wavs = chat.infer(texts)
sf.write("/tmp/chattts_test.wav", wavs[0], 24000)
```

## 已知问题

1. **rvcmd 下载阻塞**: `chat.load()` 默认触发 GitHub rvcmd 下载（超时 10s），必须用 `source="custom"`
2. **torchaudio.save 后端不兼容**: torchaudio 2.8 的 WAV 保存可能报 `Couldn't find appropriate backend`，改用 `soundfile`
3. **hf-xet 与 HF 镜像冲突**: hf-xet 传输加速在 HF 镜像上返回 401，已卸载
4. **ModelScope 格式不兼容**: `.pt` 格式旧版模型无法通过 `check_all_assets` 验证
5. **非法字符警告**: 中文全角标点（如 `！`）会触发 `found invalid characters` 警告，不影响输出
6. **编译选项**: `compile=False` 可避免首次推理的 JIT 编译开销

## 模型恢复

如果 checkpoint 丢失：
```bash
source /etc/network_turbo
/root/miniconda3/bin/python3 -c "
from huggingface_hub import snapshot_download
model_path = snapshot_download('2Noise/ChatTTS', cache_dir='/root/.cache/huggingface')
print(model_path)
"
unset http_proxy https_proxy
```

## 交付 TFG 管道评测

1. 将生成的 wav 文件上传至主评测服务器 (`:24599`) 的 `runs/<run_id>/02_tts/`
2. 运行 `python scripts/03_ditto.py --run_id <run_id>`
3. 运行 `python scripts/04_eval.py --run_id <run_id>`
4. 运行 `python scripts/05_report.py --run_id <run_id>`
5. 汇总对比: `python scripts/06_cross_model.py`
