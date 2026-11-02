---
title: infinitetalk
type: deployment
permalink: tts-exp/docs/deployment/infinitetalk
status: verified_but_not_adopted
updated: '2026-10-30'
official_commit: 50aa0a94184315407a991ae804d9b58d6d311ba8
tags:
- deployment
- '192'
- '3090'
- infinitetalk
---


# InfiniteTalk

✅ 2026-10-30 在 192 个人目录、单张 RTX3090 24GB 上跑通官方低显存单人示例，并完成服务器端音视频检查和本地读回。验证范围为一个短 clip；没有执行项目自然语音/TTS 队列或 SyncNet 评分。

❌ 项目采用状态：2026-10-30 用户判断 InfiniteTalk 和 MultiTalk 推理过慢，要求先停止，不进入当前实验。InfiniteTalk 的上述完成记录保留；不得据此继续启动批量或复跑。远端后台控制器及尚在运行的 MultiTalk 已停止，目标3090无计算进程、显存4MiB。

## 部署位置与官方版本

- 主机：10.108.16.192:22，账号 wangjiajun。
- 根目录：/home/data2/wangjiajun/TalkModels。
- 源码：InfiniteTalk/；[官方仓库](https://github.com/MeiGen-AI/InfiniteTalk)，固定提交 `50aa0a94184315407a991ae804d9b58d6d311ba8`。SOURCE_RECEIPT.json 保存逐文件 Git blob 校验；未修改模型源码。
- 独立环境：envs/infinitetalk；既有 EchoMimic 环境保持原样。
- 权重：models/Wan2.1-I2V-14B-480P、models/InfiniteTalk/single/infinitetalk.safetensors、models/chinese-wav2vec2-base。
- 目标 GPU：GPU-b937d7f2-4724-68fc-b815-cf2a58105122。不要依赖物理编号。

## 本次真实推理

官方输入 examples/single_example_image.json，包含录音棚人物图片与官方音频。参数：480P 桶、81帧、40steps、seed42、mode clip、motion_frame9、--num_persistent_param_in_dit 0，默认 offload_model=True。无量化、无LoRA，本次未开 TeaCache。480P 是官方面积桶，实际输出随输入宽高比为896×448。

| 指标 | 实测 |
|---|---:|
| 官方脚本计时区间：加载、生成、保存 | 5208.923秒（86分48.923秒） |
| NVIDIA 本进程驻留显存采样峰值 | 16384MiB = 16.000GiB |
| PyTorch max allocated | 13.785GiB |
| PyTorch max reserved | 15.695GiB |
| 进程 RSS 采样峰值 | 117.495GiB |
| 同期主机 available RAM 最小值 | 152.097GiB |
| 输出 | 896×448、25fps、81帧、3.24秒 |
| 输出文件 SHA256 | 92f212f487cd927bda84bf87f1864121ea8331721b1947f42ffdbfd426d89e0d |

显存/内存监控间隔0.5秒；NVIDIA 为采样峰值，PyTorch 是分配器累计峰值，口径不能混用。计时从包装器准备完 torch/flash 模块后执行官方脚本前开始，包含官方脚本导入、模型加载、推理和保存，不包含环境安装、权重下载及包装器自身前置检查。RSS 包含映射文件页，不等于纯匿名内存需求。

本次证明24GB3090可完成该配置；低显存模式依赖 CPU 内存卸载，速度较慢，不能从单个3.24秒样例外推所有长视频/分辨率的峰值。

## 证据与验收

远端 outputs/infinitetalk_3090_official_clip_attempt01/：

- result.mp4、execution.json、invocation.json。
- validation.json：EXIT_0，全部81帧解码、25fps、帧间平均像素变化1.445662、音频RMS0.075254、解码音频3.264秒，所有检查PASS。
- contact_sheet.jpg：首、中、尾等6帧。人工查看本地联系表，人物、构图连贯且嘴部姿态随帧变化；没有据此宣称人类感知或嘴型同步评分。
- 本地读回：runs/deployment_talkmodels_3090_20261030/early_infinitetalk/，local_video_readback.json 的 SHA 与远端一致。
- 根目录 environment-infinitetalk.freeze.txt、dependency_gate_infinitetalk_final.json、download_receipt.json、SOURCE_RECEIPT.json 为环境、依赖、权重、源码证据。

第一次本地验收辅助脚本把 numpy.bool_ 写入JSON时报序列化错误；已仅修正辅助脚本 moving 检查为原生 bool，并对同一成片重跑全部验收通过。没有因此重做或修改模型推理产物。

## 环境与已验证兼容点

Python3.10.22、torch2.4.1+cu121、torchvision0.19.1、torchaudio2.4.1、xformers0.0.28、flash-attn2.7.4.post1、diffusers0.35.2、transformers4.49.0、accelerate1.1.1、xfuser0.4.1、numpy1.26.4、opencv-python4.9.0.80。

diffusers0.31 缺少本版本官方导入的 no_init_weights/ContextManagers；升至0.35.2后官方CLI和实际推理通过。

flash-attn 使用官方 GitHub release 的 cu12torch2.4、cp310、cxx11abiFALSE wheel。SHA256 a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440。安装前在3090 BF16 forward 与 Torch SDPA 数值一致；源码编译尝试已停止，仅清理精确匹配的自身编译进程。

decord0.6 原始wheel的 WHEEL 错标cp36，因此原始 pip check 返回1（“not supported on this platform”）。不改元数据、不隐藏返回码；依赖门只豁免这一精确警告，保留证据，并实际解码官方375帧参考视频首、中、尾。其他依赖错误不能豁免。

chinese-wav2vec2-base 的 safetensors PR 下载过慢，使用官方 pytorch_model.bin 在 CPU 上转换，结果与官方PR规范文件逐字节相同：380204696bytes、SHA256 b86f5be7b752fc655c27387a75712f733315a30f976e5875491599615399e773。wav2vec_canonical_derivation.json 与 repairs/wav2vec_canonical_restore/repair_receipt.json 记录来源及恢复动作；原不完整传输文件保留。下载凭据明确标记本地恢复，不伪称下载。

## 安全复现

首次实际入口为 envs/infinitetalk/bin/python run_official.py InfiniteTalk GPU-b937d7f2-4724-68fc-b815-cf2a58105122，并设置相同 CUDA_VISIBLE_DEVICES。完整官方参数以 invocation.json 为准。包装器输出目录 attempt01 禁止覆盖，重跑必须创建新 attempt 名称并保留旧证据。

包装器限制4CPU线程/亲和性，torch allocator上限22GiB；执行前 GPU空闲、主机可用内存≥160GiB，运行时 available RAM<32GiB 停止自身。22GiB是本次保守进程上限，并非官方最低显存要求。只写个人data2，不改驱动、SSH或系统服务，不杀其他用户作业。

SSH有间歇性握手关闭；使用固定远端命令、间隔重试和主机密钥校验。Windows PTY密码输入用CR；密码仅在运行时输入，不写文件。避免自动重连的交互shell提示竞态。
