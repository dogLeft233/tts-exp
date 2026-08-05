# EchoMimic V2 部署记录

## 环境

- 容器: `connect.weste.seetacloud.com:16461`
- GPU: RTX 4080 SUPER 32GB
- Python 3.10.20, PyTorch 2.5.1+cu124
- 源码: `antgroup/echomimic_v2` commit `38c86809`
- 权重: ~17GB（ModelScope 下载）
- Whisper tiny: SHA256 验证通过
- ffmpeg: imageio-ffmpeg 4.2.2 static

## 推理

```bash
cd /root/autodl-tmp/echomimic_v2
source ffmpeg-env.sh
python infer_acc.py \
  --config <config.yaml> \
  -W 512 -H 512 -L 120 \
  --steps 6 --cfg 1.0 \
  --device cuda
```

## 输入要求

- 参考图片 (PNG)
- 音频 (WAV, 16kHz mono)
- 姿态序列 (npy 目录，每帧一个 npy 文件)

## 已知问题

- **姿态是关键**：使用合成静态姿态导致 Sync-C 仅 ~3.0
- 需要真实驱动视频提取 DWpose 序列才能正常评分
- `infer_acc.py` 不支持 CLI 参数覆盖 test_cases（必须写 YAML 配置）
