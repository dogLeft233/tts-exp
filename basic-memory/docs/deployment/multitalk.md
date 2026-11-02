---
title: multitalk
type: deployment
permalink: tts-exp/docs/deployment/multitalk
status: stopped_by_user_not_adopted
updated: '2026-10-30'
official_commit: c1ad84009b99ed36c97ab18e4517ca5f98692438
tags:
- deployment
- '192'
- '3090'
- multitalk
---

# MultiTalk

❌ 2026-10-30 用户判断 InfiniteTalk/MultiTalk 推理太慢，不适合当前实验，明确要求先停止。MultiTalk 已按要求中止，未生成完整验收视频；不得标成推理跑通或技术失败，也不得自动恢复。

## 已部署内容

192（10.108.16.192:22）账号 wangjiajun，个人根 /home/data2/wangjiajun/TalkModels。源码 MultiTalk/ 固定[官方仓库](https://github.com/MeiGen-AI/MultiTalk)提交 c1ad84009b99ed36c97ab18e4517ca5f98692438，逐文件 SOURCE_RECEIPT.json 校验。envs/multitalk 为独立环境。27个共享/独立权重共95.85GiB均已校验通过，下载凭据为根 download_receipt.json。

Python3.10.22、torch2.4.1+cu121、xformers0.0.28、flash-attn2.7.4.post1、diffusers0.35.2；环境细节及已验证 Decord 元数据警告见 [[infinitetalk]]，原始 pipcheck 警告保留，没有修改wheel元数据。两个官方CLI均已通过。

官方 low-VRAM 参数 --num_persistent_param_in_dit 0，mode clip、81帧、40steps、seed42、480P默认桶，官方双人输入 examples/multitalk_example_2.json，--use_teacache；无量化/LoRA。models/Wan2.1-I2V-14B-480P-MultiTalk 为按官方流程建立的基座文件视图，替换 MultiTalk index 并加入主权重；共享基座原文件未改。

## 中止记录

在同一张24GB RTX3090（GPU-b937d7f2-4724-68fc-b815-cf2a58105122）上真实进入去噪。日志最后完成24/40步，进度条生成阶段26分03秒；执行记录总计1773.957秒（29分33.957秒，含加载与中止处理），用户要求中止。

停止前本进程 NVIDIA 0.5秒采样峰值14686MiB（14.342GiB），PyTorch峰值allocated14801519616bytes、reserved15072231424bytes。这些只覆盖未完成运行，不是完整推理耗时/峰值，不可代替完成结果。

原 execution.json 保存 FAILED 和 Resource guard stopped this inference process：包装器将收到的SIGTERM统一转成这个异常。实际原因由 user_stop_receipt_20261030.json 明确证明是用户主动停止，不是主机内存阈值触发或OOM；不改写原始执行记录来掩盖退出。

先暂停精确匹配的自身控制器2235298，再SIGTERM自身推理2344543及其子进程，随后停止控制器和只读监控。所有目标核验账号、命令、PID和进程创建时间；未停止其他用户作业。停止后目标GPU无计算进程，显存4MiB、温度45°C、利用率0%，主机available RAM229.736GiB。源码、环境、权重、音频中间件及已完成的InfiniteTalk样片保留。

## 证据

- 远端 user_stop_receipt_20261030.json：明确中止原因、进程操作、GPU释放检查和两模型摘要。
- deployment_status.json：CANCELLED / STOPPED_BY_USER。
- outputs/multitalk_3090_official_clip_attempt01/execution.json 与 invocation.json；logs/inference_3090_multitalk.log。
- 已本地取回 runs/deployment_talkmodels_3090_20261030/user_stop/ 的停止凭据、执行JSON和完整MultiTalk日志。
- 不存在本模型完整成片验收；不启动项目自然语音/TTS队列或评分。
