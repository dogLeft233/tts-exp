# IMTalker 部署记录

## 环境

- 容器: `connect.weste.seetacloud.com:43417`
- GPU: RTX 4080 SUPER 32GB
- Python 3.10, PyTorch 2.0.1+cu118
- Conda 环境: `imtalker`
- 权重: generator.ckpt 593MB + renderer.ckpt 2.0GB
- 源码: `/root/autodl-tmp/tfg_exp/IMTalker`

## 推理接口

```python
from app import AppConfig, InferenceAgent
cfg = AppConfig()
cfg.device = "cuda"
agent = InferenceAgent(cfg)
result = agent.run_audio_inference(
    img_pil,        # PIL Image
    aud_path,       # WAV file path
    crop=True,
    seed=42,
    nfe=10,         # number of function evaluations
    cfg_scale=2.0,
)
```

## 注意事项

- 需要 `HF_HUB_OFFLINE=1`（否则模型初始化时尝试下载 config.yaml）
- 权重文件在 `/root/autodl-tmp/model_weights/`，checkpoints 目录有 symlink
- Jupyter/TensorBoard 在 base 环境中运行，不影响推理

## 性能

- 显存: ~5.0 GB
- 速度: ~30s/样本
- 输出: 512×512 MP4
