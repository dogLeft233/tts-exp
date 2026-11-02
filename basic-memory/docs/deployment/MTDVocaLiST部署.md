---
title: MTDVocaLiST部署
type: deployment
permalink: tts-exp/docs/deployment/mtdvoca-li-st-部署
---

# MTDVocaLiST部署

✅ Ubuntu真实媒体CUDA前向通过（2026-10-28）。本模型为独立蒸馏版，用户授权同时部署；不能冒充原VocaLiST。原版源码环境已准备，但官方Drive目录匿名访问为空、gdown列举[]，等待可信权重。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 初始部署与真实权重smoke | 2026-10-28 | 用户授权部署 |

## 固定来源与环境

- 机器wjj@192.168.50.2，根/home/wjj/av-sync-eval-20261028。
- 作者仓库https://github.com/xjchenGit/MTDVocaLiST ，commit5795416f9277094d3ae6215596a436a0a01f79ad。
- 源码MTDVocaLiST/；官方release https://github.com/xjchenGit/MTDVocaLiST/releases/download/v1.0/pure_MTDVocaLiST.pth 。本地weights/mtdvocalist/pure_MTDVocaLiST.pth，53278751字节。
- env /home/wjj/.venvs/mtdvocalist-20261028；Torch2.5.1+cu121、Python3.10，重型依赖只读.pth复用，不修改旧环境；独立Pillow10.4避免MFA Pillow12.3/libtiff ABI问题。
- 作者README类models.student_thin_200_all.SyncTransformer(d_model=200)，13280961参数，state_dict strict=True加载成功。

## 可运行入口与验证

```bash
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/av_sync_deployment_smoke_20261028.py --model mtd
```

已运行真实6秒自然音视频首5帧窗口。receipt /home/wjj/av-sync-eval-20261028/smoke/mtd_receipt.json，status PASS，raw logit3.359825849533081，模型输入video[1,15,48,96]与audio[1,1,80,16]，3.70秒，CUDA峰值9269863936字节（包含S3FD人脸检测）。tests/test_av_sync_deployment_smoke_20261028.py实际receipt integration通过；无pytest时runpy直接调用测试函数。

## 前处理与限制

遵作者MTD测试recipe：torchvision RGB uint8原0–255值，Resize96×96，取下半脸48×96；不要套用原VocaLiST的cv2 BGR/255。本项目适配器用现有Wav2Lip S3FD定位人脸，JPEG后torchvision读取；检测batch1。16k mel n_fft800、hop200、80bin、55–7600Hz、Slaney归一化，作者dB与[-4,4]缩放。帧f音频起点int(80*f/25)，不能套用SyncNet MFCC的4*f。

这只验证可运行：尚未实现正式全片聚合、校准时移控制和40人评分，不能由单窗口logit推出模型可靠性或TTS优势。源码许可证不在此作商用授权判断。

## Relations

- part_of [[Ubuntu同步评分器部署与EARS40云TTS 2026-10-28]]