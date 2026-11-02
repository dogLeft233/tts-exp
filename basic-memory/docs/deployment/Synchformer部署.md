---
title: Synchformer部署
type: note
permalink: tts-exp/docs/deployment/synchformer-部署
---

# Synchformer部署

✅ Ubuntu官方LRS3完整权重严格加载与真实媒体CUDA前向通过，2026-10-28。仅确认工程可运行，尚未完成正式时移控制或TTS偏好评分。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 官方LRS3部署与真实媒体smoke | 2026-10-28 | 用户授权部署 |

## 来源与环境

根/home/wjj/av-sync-eval-20261028，独立env /home/wjj/.venvs/synchformer-20261028，Python3.10、Torch2.5.1+cu121。重型依赖由.pth只读复用旧环境；新增timm0.6.12、ftfy6.1.1、av12.3.0、decord0.6.0、Pillow10.4.0，不修改旧环境。pip check仅存继承的sacrebleu/portalocker版本冲突，未影响本次推理。

官方https://github.com/v-iashin/Synchformer commit b66668a1521d7567cc760e5544b2b5b53179b687。推理源码放Synchformer-source/，source_metadata.json记录固定commit下124文件、2120671字节。早期Synchformer/完整clone被本次停止（带巨大训练CSV），该残留.git目录不是可用入口。

权重weights/synchformer/23-12-23T18-33-57.pt，1131113925字节，官方地址 https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.pt ，135分片DOWNLOAD_COMPLETE，state_dict strict=True加载237460245参数通过。cfg-23-12-23T18-33-57.yaml同目录。范围下载检查offset/长度；未做额外全文件SHA重读。

## 运行入口与结果

```bash
/home/wjj/.venvs/synchformer-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/av_sync_deployment_smoke_20261028.py --model synchformer
```

调用官方example.patch_config、get_model、test transforms、prepare_inputs、forward；patch_config关闭前端独立ckpt加载，完整官方checkpoint已含前端，避免额外AST/Motionformer下载。视频25fps、短边256保留宽高比，test分14段，每段16帧，输入video[1,14,16,3,224,224]与audio[1,14,1,128,66]。5秒真实自然音视频，21个时移logit有限，argmax offset0秒。GPU前向全流程7.80秒，CUDA峰值1928923136字节。官方cfg控制half precision，V100上FP16 autocast。receipt smoke/synchformer_receipt.json。

与MTD一起，两条实际receipt integration测试PASS，tests/test_av_sync_deployment_smoke_20261028.py，runpy直接执行（环境未装pytest）。代码在Windows .与Ubuntu项目同路径相对目录，未提交。

## 媒体兼容适配与边界

PyAV12.3不同于官方老版本8/9，尚不能宣称bit-exact复现。原PCM16 MOV被torchvision读成Short，官方get_video_and_audio在转float前mean报dtype错误；适配器改为pcm_f32le MOV，不改官方reader与变换代码。实际解码后80000采样点与原自然音频前5秒逐点最大差0，未归一化、未时移、未AAC重编码。证据smoke/synchformer_float_pcm_check.json。视频短边缩放、25fps、CRF18编码供smoke使用。

本次只验证部署，不由一个中文旧fixture的0秒预测推断中文/英文泛化或同步评价可靠性。正式cohort还须统一媒体前处理与时移控制；未自动生成40人TFG或开展偏好评分。

## Relations

- part_of [[Ubuntu同步评分器部署与EARS40云TTS 2026-10-28]]
- relates_to [[MTDVocaLiST部署]]