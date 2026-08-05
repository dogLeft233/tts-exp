# JoyVASA 部署记录

## 环境

- 容器: `connect.weste.seetacloud.com:43417`
- GPU: RTX 4080 SUPER 32GB
- Python 3.10, PyTorch 2.2.2+cu121
- Conda 环境: `joyvasa`
- 权重: 5.2GB（motion_generator + motion_template + LivePortrait）
- 源码: `/root/autodl-tmp/tfg_exp/JoyVASA`

## 推理接口

```bash
cd /root/autodl-tmp/tfg_exp/JoyVASA
python inference.py \
  -r <reference_image> \
  -a <audio_file> \
  -o <output_dir> \
  --animation_mode human
```

## 输出

- 格式: MP4 (25fps, 含音频)
- 命名: `{ref_name}_{audio_name}.mp4`

## 性能

- 显存: ~1.4 GB（极低）
- 速度: ~23s/样本（最快）
- 输出: 512×512 MP4
