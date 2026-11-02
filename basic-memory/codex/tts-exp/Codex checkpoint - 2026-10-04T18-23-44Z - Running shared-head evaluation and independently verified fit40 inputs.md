---
title: Codex checkpoint - 2026-10-04T18-23-44Z - Running shared-head evaluation and
  independently verified fit40 inputs
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t18-23-44-z-running-shared-head-evaluation-and-independently-verified-fit40-inputs
trigger: compact
status: open
git_sha: 2aecd64b5131fac6c5b8ca52370ac156ba301062
username: dogle
capture: deliberate
project: tts-exp
model: gpt-6.1-sol
hostname: LAPTOP-E133GSLE
agent: codex
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
started: '2026-10-04T18:23:44.3901979+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T18-23-44Z - Running shared-head evaluation and independently verified fit40 inputs

## Summary

共享自然音频增强候选正在 Wav2Lip、MuseTalk、Ditto 上生成并换回原音；本轮独立核验了扩展拟合输入并保存可恢复的代码，下一步将源音频暂存到服务器，为局部声学干预与可训练方法提供固定输入。

## Story

用户最初要求从本地 Basic Memory、Git 和历史数据恢复 TTS 增强 TFG 的研究路线，随后明确要求完成可用于多种 TFG 的自然音频增强头。增强波形仅驱动生成，成片恢复自然音频及其原始时钟。用户要求持续迭代，通过失败尝试形成完整方法；根代理维护理论、协议、独立复核、BM 与私有运行入口，GPT-6 Luna/max 子代理负责实验与代码。最新授权允许必要时用 Qwen ASR 检查内容，API 密钥和服务器密码仍只存在于运行时安全变量中。

目前采用先拟合、再开发集原生视频、最后未见说话人和保留模型确认的路线。开发集 20 段、10 位男性说话人、3 个固定种子；确认集 80 段、40 位未见说话人与 LatentSync 继续封存。主终点为恢复自然音轨后的 SyncNet Cbest，按种子、音频、说话人、模型依次等权汇总，并保留 D0、Canchor、远偏移、内容和视觉护栏。最终方法必须超过无训练 RMS 基线，交付 CLI、配置、权重、延迟和可复现报告。当前共享 540 项是三个已固定谱形候选的开发集生成，不能用部分行选择参数。

截至 resource06 实际响应 2026-10-04T18:03:50Z，540 项中 79 项 PASS、0 FAILED、0 INCOMPLETE，生产者 RUNNING，全部已完成行都有生成、剥离音轨、原 PCM 换轨、追踪和官方评分的执行时间。按模型生成均时 Wav2Lip 6.619 秒、MuseTalk 34.663 秒、Ditto 44.970 秒；追踪均时 5.272 秒、官方评分均时 3.757 秒。该计时描述每次生成总时长，尚不能归因于权重加载。后续正在准备实际 MuseTalk/Ditto 持久后端，必须另外测量加载、设备转移、前向及种子/状态重置等价性，不能修改当前运行。

已完成的 V4 实际 540 行说明共享学习头对 identity 的模型等权改善为 +0.109302505，但对 RMS 为 -0.044299629；因此继续沿局部变化结构与接收模型特征响应方向推进。a(t) 是日志谱掩码的公共频率分量，它与原音 40ms log-RMS 的相关系数 -0.08214、与正 dB 谱通量 +0.43384，属于输入声学关联，并不是嘴部开合相关。共享候选还包含 r090 局部压缩，不能从本次共享读出声称压缩单独贡献。下一代准备 r090 对照、关闭压缩的无门控谱对照、正/负谱通量 × 强调/压平四项，总计六个候选；参数和名单尚未冻结。构造时保留原 STFT 系数相位，不等于 ISTFT 后重新分析相位严格不变。

## Working State

可恢复的 Git 状态：分支 codex/cross-tfg-audio-head-20261003，HEAD 2aecd64b5131fac6c5b8ca52370ac156ba301062 为本地未推送提交；50f83e296c0fd8e6da0af36d03493abb71806d40 保存原生前端测量实现，2aecd64 保存已复核的局部谱转换准备与 fit40 输入构建/数值核验。最近已验证的远端提交 c4a199890bd529928ec34b9be6da86a10a737c97；没有当前分支 PR。历史五个修改文件及大量旧未跟踪文件没有混入这些提交。CONTEXT.md 94 行。HANDOFF.md 为 ignored 本地快照，仍指向 67 行和 HEAD50f，下一次任务快照应更新它。

机器本地运行状态很脆弱，需要先核对，不得重启共享生产者：私有 PowerShell PTY 50274；EOF 完整性修复后的 SSH bridge 端口 10887。服务器 root@connect.weste.seetacloud.com:12349；SSH 主机指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/API key 不在笔记或命令参数里。远端项目 NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，临时根 SHM=/dev/shm/cross_tfg_audio_head_20261003。当前共享运行输出 SHM/shared_profile_shortlist_dev_attempt03_execution_attempt01/run，launcher PID549953/startticks770442756、runner550013/770442788、wrapper550016/770442789。最新 SHM 空闲 5648953344 字节、根盘空闲 2174087168 字节、GPU 空闲32230MiB；阶段切换会改变瞬时 GPU 数据。

源头状态与权威证据：

- 共享计划 SHA 43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df，rows SHA 64a41fb0c76943ac535d38afa9347d9ca0e91ad075e54bb514be262945a63abb，manifest SHA 060c5a71b745b04d1df1138b32f5f9fb6f8c608934a7888efa064bde4443c5f8。
- resource06 索引位于 runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/shared540_resource_status_attempt06/request_index_attempt06.json，索引 SHA c40f34c3fa264fe630bdfe54d9509c5786a99011e1a6b72379e9550af0836a78；79 项实际完整响应 SHA 4a4527b5a7df8b7b3f277fb93c0703447edaf070f9724913fcc71976ca87297b。再次轮询前保存原响应。
- fit40 固定 S0052–S0071 每位 j0/j15，共 40 段、20 位说话人；按 speaker 的固定 SHA 排序生成五折。输入在 runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/source_wav。manifest SHA 0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020。
- fit40_root_numeric_qc_attempt03 内根代理实跑 receipt SHA 96e46b78fe3eb739fe4500a91eb99b1ec0acfb8e83e49b642311905ae7846c41、NPZ SHA 544199f39058505711779112779c369b487f0bff62584c8c8afc4ca11ae957ed，40 段/320 数组 PASS，最大误差 1.9073486328125e-6。物理 40ms 平方能量和 dBFS 已分别命名。
- fit40_source_stage_attempt02 ZIP 5780988 字节、42 项，SHA 22ac5b6a83dab7ac2ed8e12e1d8f73b3b77d2b980c96e86fcbf9ba1ce1fe9a8e；根代理实际检查全部成员、CRC、原始 WAV、PCM16 样本和 JSON 内容一致。独立本地证据 .codex/scratch/root_fit40_source_package_qc_attempt02.json。只含源 WAV、manifest 和 registry，没有代码、模型或评分产物。
- 新 source stage packet03 索引 .codex/scratch/v5_fit40_source_stage_packet_attempt03/request_index_attempt03.json SHA 504dc3cd118633033aee6f93e2477278f3caf1774e393540e2f2b3cf3af5751d；worker22945 字节 SHA e4d7e2ee76237557c42c589f4c020a4bdaa99fed8df38af179183d8d07619aaa。子代理报告准备完成，根代理尚需完整读03差异和请求再实际依序 dispatch。02 完整读发现嵌套 r090 metadata 与 transport key 路径错误、control parent 未验证，所以02从未发送；03把 control 放入已存在 NS/job_control 并用实际 ZIP 重放 metadata validator。
- 900 行 join/视觉护栏 postrun04 源码已经实际暂存并核验，未执行分析。terminal discovery04 与 terminal-to-launch builder02 根代理完整阅读合格；builder34885 字节 SHA d35d6dd2fd5bba233e785214e4e86d3ac7a8b97d770e8958d7ab33e62315273e，仅在真实 TERMINAL_SEALED 响应后生成下一份实际 hash 绑定请求。不得用 RUNNING 响应启动 ASR/分析或制造终态 hash。
- native_encoder_probe_attempt04 三个真实前端 × 8 拟合片段 × 7 条件 =168 项，源码/计划/packer已实际暂存，前向未运行。Wav2Lip 是官方 NumPy/librosa mel，不可直接 autograd；MuseTalk Transformer可在镜像对齐后微分；Ditto部署ONNX需合格Torch镜像或停止梯度约束。即使 Ditto CPU，ORT 线程池也会竞争，因此168项等待共享生产者结束。

V5 NumPy scaffold V2 在 scripts/audio_enhancement/v5_local_transition_transform_scaffold_v2.py，SHA cf86b95b65b2c4a423082e0f717713983915cdc0674ee7ddb4fba2534e9906e0；独立原相位正增益 STFT 干预、原音正负相对线性谱通量门控、四符号项和无门控项共用 κ、精确原样本数及显式 RMS/peak。不提供默认参数，不会默默按候选单独投影。只审查、未执行转换/训练/模型。theory v3 SHA d6f2a6e63cc117a86eb6f87a2c6c9f644f58c102cea19cee54706b0e01adca0f 已完整复核修正相位/相关量/部署前端标签。PCEN、LEAF 只提供局部归一化设计动机，不是 TFG 增益证据。

未完成 source review：scripts/audio_enhancement/phase_clock_diagnostics_v1.py SHA 25ed6f204dea7e5c149f7c897dfccfebe972ad0ef30eda1a2b1dbdc59ae216c5 仅前240行根代理已读，需继续读余段。无支撑局部 xcorr 与零功率 centroid 有 NaN/validity mask；未来诊断写入不能用 V2 的全有限 NPZ writer 或静默填零。phase/centroid/xcorr 是声学变化量，不可称为音素边界位移。未来真正可训练头还需要 Torch 转换镜像、前端对齐与实际视频验证，不得把被冻结 baseline 视频的音频侧代理目标当作对生成器响应的梯度。

## Changed Files

两份本地新提交保存 native_encoder_probe_attempt02.py、pack_native_encoder_probe_attempt01.py、verify_native_encoder_probe_bundle_attempt01.py、native_encoder_probe_contract_v2.md、局部r090对照、V5 scaffold V2、fit40 producer/independent QC03、transfer manifest/runtime metadata/source package 构建器与 V5 contract V3；CONTEXT.md 记录实际 RUNNING。临时请求、响应与准备输出位于 ignored runs/ 和未跟踪 .codex/scratch/，必须以实际 SHA/receipt 恢复；不得把这些本地文件说成已推送产物。

## Verification

根代理完整复核所有被提交的新源码或其先前完整读后的明确差异；两次 git diff --cached --check PASS，无 pytest/单元测试新增或运行。fit40 数值 QC03 和 ZIP 独立全成员/PCM QC 实际 PASS。resource06 请求 SHA/bash 语法检查和实际执行 PASS；响应八字段完整、exit0、无 stderr/截断/EOF 错误。新 packet03 仅子代理 metadata replay/静态合格，尚没有根代理完整审查或远端执行。原始大数据 tar 先前已复核，不重复读取；存储清理仅已封存且独立恢复的旧输出，当前共享 RUNNING 目录没有删除或归档。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前本地未推送 HEAD `2aecd64b5131fac6c5b8ca52370ac156ba301062`；上一份本地提交 `50f83e296c0fd8e6da0af36d03493abb71806d40`。

## Observations

- [result] 共享540最近79项全部PASS，属于执行进度；尚未产生独立完整开发集候选结论。
- [result] 固定fit40已完成独立原始PCM/数值/ZIP核验，未执行增强变换或视频生成。
- [decision] 保留 RMS 强基线、共同支持集和全部失败行；确认集与保留TFG模型继续封存；共享候选结果完整读出后才冻结V5设置。
- [decision] 现在的谱门控和后续可训练头按原自然PCM及时钟匹配，源相位构造与再分析相位分开测量，输入公共掩码关联与嘴部运动关联分开描述。
- [blocker] 当前GPU流水线尚未终态；原生168测量、900行独立join、视觉和60段Qwen ASR需真实终态绑定后启动。
- [next_step] 根代理完整审查并实际依序执行 fit40 source stage packet03，核验源输入暂存receipt，同时保持现有共享540运行并继续完成phase/clock诊断审查。

## Relations

- continues [[Codex checkpoint - 2026-10-04T17-49-13Z - Shared-model execution and fit40 transition preparation]]
- advances [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]