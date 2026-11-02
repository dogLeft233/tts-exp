---
title: Codex checkpoint - 2026-10-30T09-42-00Z - TalkModels 3090 environments ready
  and final weight pending
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t09-42-00-z-talk-models-3090-environments-ready-and-final-weight-pending
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T09:42:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a18b7a-98d3-79d1-a715-721276a41216
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T09-42-00Z - TalkModels 3090 environments ready and final weight pending

## Summary
192 上两个独立环境及官方 CLI 已验证可用，27 个权重中已完成 26 个校验，完整模型推理尚未开始。

## Story
用户授权在 192 的单张 24GB RTX3090 上按官方低显存流程跑通 InfiniteTalk、MultiTalk，记录实际峰值显存与耗时，并要求保护共享服务器。SSH 间歇性失败后，原生 Windows OpenSSH 加固定命令和间隔重试恢复了连接；账号密码已在内存核对，密码不入文件。所有写入在个人 data2 目录，不动驱动、系统服务或其他用户任务。

采用原始官方源码、隔离环境、官方 --num_persistent_param_in_dit 0，依次执行 480P、81 帧、25fps、40 steps、seed42 的官方示例 clip，无量化/LoRA。控制器在权重全部校验后自动执行两次推理和音视频验证。此时不能回答实际显存与耗时，也不能把 FlashAttention 诊断当作完整推理成功。

## Working State
持久远端根目录 /home/data2/wangjiajun/TalkModels；控制器 resume_after_locked_manifest.py PID2235298，deployment_status.json 最新 RUNNING / WAIT_WEIGHTS。下载器 download_models_v6.py PID2253826（以 download_current.pid 为准），日志 logs/download_models_v6_after_canonical_restore.log。最新读取 26/27，86.59GiB；InfiniteTalk 主权重已通过，只剩 MeiGen-MultiTalk/multitalk.safetensors（9947889040 bytes）。最终 download_receipt.json 必须 PASS；partial receipt 不能代替最终验收。

两环境 envs/infinitetalk 和 envs/multitalk：Python3.10.22，torch2.4.1+cu121，xformers0.0.28，官方 flash-attn2.7.4.post1 wheel，diffusers0.35.2。升级 diffusers 解决官方 CLI 缺少 ContextManagers/no_init_weights；未改官方源码或既有 Echo 环境。官方源码 SOURCE_RECEIPT.json 固定 InfiniteTalk 50aa0a94184315407a991ae804d9b58d6d311ba8、MultiTalk c1ad84009b99ed36c97ab18e4517ca5f98692438。

优先 GPU UUID GPU-b937d7f2-4724-68fc-b815-cf2a58105122；替补 GPU-6f7ed39e-6268-524c-0c37-bce07db96bae。每次启动前确认无其他计算进程、VRAM<1GiB、host available RAM≥160GiB。只用一张3090，4CPU线程，torch allocator22GiB，运行时RAM低于32GiB终止自身。不要通过杀其他进程腾资源。

本机临时恢复入口 C:/Users/dogle/AppData/Local/Temp/tts192-talk-deploy/native_fixed_exec.ps1；使用固定命令 .sh 作为argv，密码仅在新 password: 提示时输入 CR。不要再用自动重连交互shell（提示竞态曾误吞一条命令作为密码）。远端命令文本须去掉CR。现有只读长监控工具 session90397 正运行，每50s打印状态；它不是部署控制器，断开监控不影响部署。本机 .codex/scratch/talkmodels_3090_deploy_20261030/deployment_state.json 明显过期，不可作为当前权威状态。

## Changed Files
主要是远端 run_official.py、validate.py、resume_after_locked_manifest.py、download_models_v6.py、verify_runtime_dependencies.py、publish_canonical_wav2vec.py、derive_canonical_wav2vec.py 及相关 JSON 证据。临时目录中保留脚本与 wheel。仓库存在大量本来就有的修改与未跟踪文件；本次未修改核心流水线代码，也没有提交/PR。Git 当前 SHA 仅本地核对，未验证远端提交可达。

## Verification
两个官方 --help 均通过。FlashAttention 在3090上 BF16 forward 与torch SDPA差0；这是模块诊断。原始 decord0.6 wheel 的 WHEEL 元数据错误标cp36，pipcheck 返回1；保留原始警告，依赖门只豁免该精确项，实际官方375帧参考视频首中尾解码成功。dependency_gate_*.json 记录 PASS_WITH_KNOWN_DECORD_METADATA_WARNING，不能宣称原始 pipcheck 全通过。

音频编码器 PR 下载太慢，CPU 从官方 pytorch_model.bin 派生 safetensors，得到与官方 PR 完全相同的380204696 bytes、SHA256 b86f5be7b752fc655c27387a75712f733315a30f976e5875491599615399e773，218 tensors。wav2vec_canonical_derivation.json、repairs/wav2vec_canonical_restore/repair_receipt.json 记录逐字节同一性，原未完成文件保留备查；下载 provenance 明确标注本地恢复，未伪称下载成功。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，本次已通过 GitHub read-only API 解析规范 URL。
- [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
- [MultiTalk](https://github.com/MeiGen-AI/MultiTalk)
- 远端 outputs/infinitetalk_3090_official_clip_attempt01、outputs/multitalk_3090_official_clip_attempt01 将保存 result.mp4、execution.json、invocation.json、validation.json、contact_sheet.jpg；此刻尚未确认产生。
- 根 validation.json 为两模型成片验收；检查81帧、音轨非静音、运动、音视频时长差<0.25s。
- 执行计时包含加载/推理/保存，NVIDIA进程显存0.5s采样峰值与torch allocation/reserved峰值分开记录。

## Observations
- [decision] 保护共享服务器：个人data2、独立环境、UUID限定单3090、串行运行，不改系统或既有Echo环境。
- [decision] 未完成下载/仅模块诊断不视为完整推理通过；实际视频与performance JSON验收后才报告测量值。
- [blocker] 最后一个约10GB MultiTalk主权重仍在下载；无完整模型性能结果。
- [next_step] 继续读取 session90397 或重新固定命令SSH，等待最后权重完成，跟踪两次实际推理与音视频验收，取回成片/JSON/环境清单并更新两模型部署文档。