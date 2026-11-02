---
title: Codex checkpoint - 2026-10-05T01-03-51Z - Verified native gradients and downloaded
  complete enhancement CLI outputs
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t01-03-51-z-verified-native-gradients-and-downloaded-complete-enhancement-cli-outputs
started: '2026-10-05T01:03:51.4551320+00:00'
capture: deliberate
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
status: open
agent: codex
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
git_sha: 1f40cae586727ffd662246a0e086e6e027908284
username: dogle
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T01-03-51Z - Verified native gradients and downloaded complete enhancement CLI outputs

## Summary

原生生成模型的九参数梯度和 RMS 投影后满秩已由实际产物独立复核，增强 CLI 的完整 CPU 运行与结果归档已执行，跨设备控制失败被保留并转入同 CPU 直接对照。

## Story

用户最初要求恢复 TTS 增强 TFG 的机制研究，随后明确目标是可适配多个 TFG 模型的自然音频增强头：增强 PCM 只驱动视频生成，播放与 SyncNet 评分恢复原始自然 PCM 和原时钟。用户要求持续推进至可用方法，授权重跑、下载及 Qwen ASR/TTS；最新授权允许用 Qwen ASR 复核内容。Root 维护理论、冻结协议与独立核验，GPT-6 Luna/max 子代理实现代码和实验。

主线同时推进两条路线：固定的六个局部声学候选在完整 dev20/10 speaker ×3 seeds×3 models 上评估；独立九参数局部过渡头先核验原生梯度，再扩到完整 fit40/20 speaker 的固定120步训练。只读开发结果不用于越过训练冻结流程；confirm80/40 unseen speakers 与 LatentSync 仍封存。最终方法、权重、跨模型确认和报告26尚未完成。

本轮实际 native05 CPU 运行生成了完整 31-lag 曲线、27个视觉嵌入、九方向有限差分和投影后 Jacobian。Root 下载十成员归档并独立重算27数组哈希、有限差分容差、Gram特征值和秩。九参数 SyncNet 奖励梯度 L2=0.184508245338，投影后波形 Jacobian rank=9，条件数176.27939165，优化器步骤为0。零参数输出与 canonical RMS 波形逐字节一致；但相对旧 GPU/batch4 缓存的曲线最大差5.4836273193359375e-5、视觉差7.82310962677002e-5，超过固定1e-5。完整结果仍是 COMPLETE_WITH_CONTROL_FAILURE/rc2。新 worker07 在保留旧失败的同时增加两次同 CPU、同模型和同批量的 RMS/zero 直接计算，不放宽阈值。

增强 CLI attempt02 修复元数据序列化后实际 CPU 运行 PASS_ENGINEERING_ONLY_V3_CLI_CPU：6.200435002秒、保守峰值 RSS897110016B，三条3.245/4.383/6.869875秒输入完成增强、元数据与禁止覆盖检查。三次冷进程约1.469/1.510/1.527秒，暖头中位数约9.397/10.220/14.919毫秒。它是零初始化工程可用性，尚不代表训练后的增益。Root 已实际下载七成员1044480B归档并核实全包哈希；逐源 RIFF/PCM、时钟、元数据及时间算术复核正在进行。

共享540行跨模型运行最新实际观测为500 PASS、0 FAILED，余40行仍运行；Root 没有读取部分效应或选择候选。实际终态与 GPU 释放后才启动其900行汇总、内容/视觉守卫、原生168行诊断及其他 GPU 工程任务。

## Working State

持久入口为 CONTEXT.md、当前 BM Task 与 Experiment；机器本地 scratch、远程 /dev/shm 和私有 PTY 是脆弱状态。服务器地址和 host pin 在项目参考记录中，凭据仅在私有 PTY50274 的 SecureString/环境中，文件及记忆不保存密码/API key。唯一实际远程派发仍经该私有会话。现有源码和请求以原始字节 SHA 固定，旧失败与未运行请求均保留。

Native05 原始归档在 `.codex/scratch/root_native05_readback_attempt01/`，独立核验在 `.codex/scratch/root_native05_gradient_readback_qc_attempt01.json`。归档681871B，SHA8dcf52b2633729546a7f803b20980a54cdd8666215b6b6fc6016b79bf2fbe847。CLI02 归档位于 `.codex/scratch/v3_enhancement_cli_cpu_result_readback_attempt02/downloads/v3_cli02_selected_outputs_attempt02.tar`，SHAe79e3280bb5e630775f51e142c7f8a57d99db92a0cc147aec95bcab84d0af362；报告65173B/SHA d6223fbbac16801003017a313fedc32c8150ac9ef5e3a2d2a5d689c15d9e59a7。

Native07 新 config/source ZIP/stage/readiness 包已由子代理准备，只有 worker07 的变化已审；其余完整审查与实际派发未执行。入口 `.codex/scratch/cross_tfg_audio_head_20261003/v3_v4_native_cpu_qualification_attempt05/stage_attempt03/request_index_stage_attempt03.json`。Config07 绑定新 attempt05 解包路径和独立 output06，复用已固定的旧 V3 CPU config 输入。

完整40源 WAV 已实际入服务器。缓存/原生 geometry builder 与120步 runtime仍在实现：不能复用旧8行缓存/固定8行工厂。所有40行须产生与正式 SyncNet 一致的 MFCC、1024维自然音频缓存、full512→run_pipeline→JPEG几何，并先做同 GPU 的全40 RMS/zero对照。

Resident资源启动代码02审查发现 Python入口 symlink、采样线程结尾、GPU两次稳定空闲等待和异常保留问题；子代理新03修复中。内存门槛应记录 allocated/reserved/NVML三列，采用其最大增量再加4GiB，并绑定同GPU/no-peer证据。源码完成不代表12行资源启动或48行冷/驻留RNG对照已运行。

## Changed Files

`scripts/audio_enhancement/enhance_v3_attempt02.py` 及对应 config 为已执行但未提交的 CLI 修复；`qualify_v3_v4_native_cpu_attempt07.py` 是新同 CPU 对照源码；native adapter03、多个请求/缓存构建草稿为未提交本地文件。CONTEXT.md 已更新到500行、梯度/CLI实际状态且94行；HANDOFF.md仍需刷新。既有 portable source axis、rebind 与 tests 的其他脏文件予以保留。当前 HEAD 是本地未发布 `1f40cae586727ffd662246a0e086e6e027908284`，无当前 PR；不将未发布 SHA 伪装成 GitHub 链接。

## Verification

Root 实际执行/复核了 native05 完整归档、27数组、九方向有限差分、投影后 rank9、零参数 PCM 字节一致及明确失败的旧 GPU 缓存容差。CLI02 源码差分、Python AST、所有请求的嵌入 Python AST/外层及内层 bash-n、上传闭包和14输入实际 preflight均完成；CLI实际完成后归档及完整传输哈希通过。旧 CLI01 serializer失败、native04环境失败、native05跨设备失败、未执行包装请求均保留，不计作科学阴性。没有运行 pytest 或单元测试。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/develop-cross-tfg-audio-head-20261003/` 与 `runs/cross_tfg_audio_head_20261003/` 是协议及实际产物的入口。

## Observations

- [result] 原生视频生成路径上九参数梯度均非零，RMS投影后 Jacobian 满秩9；此为训练可达性结果，不是跨模型效果结论。
- [result] 零初始化CLI在三种时长实际完成且禁止覆盖守卫通过，七成员完整归档已下载；独立三源字节和时间算术核验待完成。
- [decision] 保留原固定1e-5阈值和旧GPU缓存控制失败，新增同CPU独立baseline/zero控制；后续fit40必须先生成同GPU、同批量、同geometry的新baseline。
- [decision] 所有候选必须恢复原始自然PCM与时钟后评分，保留完整cohort失败分母；确认集仍封存。
- [blocker] 共享540行尚余40行，其他GPU任务等待其真实终态与释放；40行缓存/geometry及120步训练实现仍未实际执行。
- [next_step] 完成CLI七成员的原音/增强PCM、时钟、元数据和延迟独立复核，然后完整审查并实际派发native07同CPU对照，衔接40行缓存构建与固定120步训练。

## Relations

- continues [[Codex checkpoint - 2026-10-05T00-31-01Z - Preserved actual integration failures and relaunched native gradient qualification]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]