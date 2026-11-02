---
title: sparsesync
type: deployment
permalink: tts-exp/docs/deployment/sparsesync
status: active
date: 2026-10-28
tags:
- deployment
- SparseSync
---

# sparsesync

✅ Ubuntu部署、官方示例输出对照、20物理时移控制、28原生媒体评分和独立审计全部通过。原生P0严重饱和，细粒度排序仅探索使用，详见报告72。

| 日期 | 变更 | 授权 |
|---|---|---|
| 2026-10-28 | 首次部署与官方5秒原生子集评分 | 用户要求部署评分，评分前确认官方一致 |

## 位置与来源

- Ubuntu源码 /home/wjj/av-sync-eval-20261028/SparseSync-source。
- [官方仓库](https://github.com/v-iashin/SparseSync)，commit a5bee8a047c0ebeec66b18d4ddf4c5f0ef098a4f；source_receipt.json记录31个源码/配置blob及逐项校验。不是失败的SparseSync完整clone目录。
- venv /home/wjj/.venvs/sparsesync-20261028；独立site-packages的shared_runtime.pth只读引用synchformer-20261028、cogenav-sync-20261024及cross_tfg_muse_ditto_shared_torch251_cu121_attempt01的运行库。不得删除这些共享依赖后仍假设venv完整独立。
- Python3.10、torch2.5.1+cu121、torchvision0.20.1、torchaudio2.5.1、PyAV12.3、V10016GB。FP32按照官方example.py，无AMP。兼容环境，不宣称原2022锁版本逐位一致。
- 权重 /home/wjj/av-sync-eval-20261028/weights/sparsesync/22-07-13T22-25-49.pt，653037857字节，54348652参数，strict加载通过。
- 原配置同目录cfg-22-07-13T22-25-49.yaml；LRS3 NoFaceCrop 21类，−2至+2秒/200ms网格，官方5秒test crop。

官方下载根为https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/22-07-13T22-25-49/ 。小文件配置与.pt均在该目录。download_receipt.json记录HTTP范围/总长验证。下载较慢，最终由分块组装完成；chunks/、aux512k/和.part是机器本地下载缓存，不是第二份模型入口。未做额外全文件SHA读回。

## 调用

入口 scripts/experiments/sparsesync_official_cloud_rescore_20261028.py，阶段freeze、run、audit。正式run runs/sparsesync_official_cloud_rescore_20261028_attempt01已完成，**不要为恢复状态重新跑GPU**。只需复核时运行audit及对应pytest。

run阶段先完整加载模型：两个feature extractor仅跳过初始化预训练文件，随后full checkpoint strict=True覆盖全部参数。源码不改。官方main对照通过后，先物理±400ms控制再批量原生评分。main对照只拦截模型初始化以复用已完整加载模型、捕获输入/输出、跳过可视化重建；原main仍再次strict reload。Recorder输出与直接model forward一致。若直接裸调用官方CLI而不走该入口，会再次尝试下载两个初始化feature extractor权重；这些不是本次最终同步权重所缺失的参数。

视频/音频通过作者get_video_and_audio、get_transforms(cfg)['test']、prepare_inputs；复用此前float PCM MOV，官方读取结果与原WAV逐样本完全相等。所有test参数保持原配置。输入video[1,125,3,224,224]和audio[1,1,257,626]。原数据不足5秒时排除，不修改位置编码、不补帧、不拉伸。

主要连续分数P0是官方输出零时移类概率；它不是作者dataset accuracy。保留全部21类logits/概率和预测类别。源输入、原生评分源码与最终CPU分析源码分别有快照。

## 验证与限制

官方example.main对照第一条自然样本：audio/video/logits最大差均0，概率差5.55e−17。28媒体预处理检查通过；20/20物理±400ms控制P0下降、预测相对时移追踪正确。独立NumPy复算28组输出和36配对端点通过，Windows/Ubuntu各4项artifact测试通过。初次链因pytest缺失停在测试阶段，安装到专用venv后补测完成，没有重复GPU。

全部28条原生输入预测零时移、P0>99.95%，正式18对/6来源TTS胜率33.3%，差−0.009311个百分点，Sync-C冲突66.7%；主端点来源CI跨0。仅说明粗同步通过且P0饱和，不证明自然口型更好或SyncNet偏置。52短媒体未测。Sync-C沿用整段，SparseSync看中心5秒，窗口不同。

## Relations

- relates_to [[72-sparsesync-official-native-cloud-preference]]