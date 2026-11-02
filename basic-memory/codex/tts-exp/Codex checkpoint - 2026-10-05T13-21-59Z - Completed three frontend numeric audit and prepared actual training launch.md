---
title: Codex checkpoint - 2026-10-05T13-21-59Z - Completed three frontend numeric
  audit and prepared actual training launch
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t13-21-59-z-completed-three-frontend-numeric-audit-and-prepared-actual-training-launch
status: open
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
agent: codex
username: dogle
git_sha: 31d3111ccc3774ac3580c486845cec686587129b
capture: deliberate
project: tts-exp
repo_root: E:/Documents/tts-exp
started: '2026-10-05T13:21:59.2165362+00:00'
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T13-21-59Z - Completed three frontend numeric audit and prepared actual training launch

## Summary
三种 TFG 原生表征的 336 行配对分析已完成独立复核，Ditto 数据完整恢复；首轮可学习增强头的训练输入已绑定，正在完成实际存储回收以启动 120 步训练。

## Story
最初问题是 TTS 为什么改善中文 TFG 同步。用户已将主线推进为可用于多个模型的自然音频增强头：增强音频只驱动生成，交付视频换回原始自然 PCM 与原始时间轴。用户要求持续形成完整方法，代码和实验交给 GPT-6 Luna/max 子代理，根代理维护理论、协议、独立复核与 Basic Memory。Qwen ASR/TTS、服务器、历史网盘数据均已授权；密钥和密码只保留运行时，不能写入文件或知识笔记。

当前先验证固定 RMS 基线，再学习局部谱形头；最终评价来自真实视频和原音换轨评分。已有 Dev20 三模型 RMS 增益均值 +0.153602，固定局部频谱 profile 未达到共同增益门槛，所以推进九参数、固定 120 步的可学习头。Fit40/20 说话人可以学习，Dev20/10 说话人用于三个模型与三种子迁移；Confirm80/40 新说话人与 LatentSync 必须保持封存直至方法冻结。首轮尚无实际优化器更新、最终权重或新 Dev 视频，不得把已准备代码称为完成方法。

这次完成 Ditto 56 NPZ 的直接 SSH tar 流恢复，根代理对完整包和全部 58 个恢复文件重新校验字节与 SHA。随后三前端、8 个 fit 样本、7 个条件和两个锚的 336 配对行产生实际结果：1848 次 producer 数值复核零差异，最大误差 8.67e-14。根代理另行核对全部键与 42 组等样本均值，误差最大 3.33e-16。固定 RMS 对自然锚的中心化相对特征变化 Wav2Lip 19.759%、MuseTalk 5.758%、Ditto 30.318%；对固定 RMS 锚的 accent015 为 4.928%、16.232%、8.443%。这说明前端不同，对能量和局部谱形的响应次序也不同；这些数字是前端描述性证据，不能直接与另一 cohort 的生成视频增益作因果回归。

过往 transition norm ratio 将零/零记为 0，造成 Wav2Lip 自对照低于 1。新补充分析保留原表，明确有效区间与排除计数；代理报告实际完成 336 行，Wav2Lip 每锚排除 2/833 个区间，另两模型未排除，自对照有效区间恰为 1。根代理已完整审阅源代码，独立补充数值核对待执行。后续可区分局部帧能量与相对频带形状，不能把 log mask 去共同模式当成物理帧能量保持。

训练依赖的完整 40 样本时间几何与缓存已恢复并绑定实际哈希。启动器要求新鲜资源门槛；当前先将两个已完整复制的历史输出精确回收。25 个 precodec 文件的源和目的全部通过再次完整哈希及活动引用扫描；根代理实际执行 exact25 retirement03，返回成功，实际 receipt 及回收后资源复核正在取得。SadTalker 725066984 字节权重已完整复制并独立核对；其只读引用审核运行前 shell 检查退出 1、无 stdout/stderr，正在诊断实际 Python 路径，未执行源权重删除。不可跳过引用门槛或扩大删除范围。

## Working State
持久本地分支 `codex/cross-tfg-audio-head-20261003`，HEAD `31d3111ccc3774ac3580c486845cec686587129b` 为未推送提交。上次直接可解析推送提交为 c4a199890bd529928ec34b9be6da86a10a737c97；GitHub 当前分支查询无 PR。五个原有 dirty 源码/测试文件保持原样。大量 scratch、runs 和 HANDOFF 是本机易失或 ignored 状态，应以真实产物哈希和 BM 任务为恢复入口。

本机私有 PTY 50274 保存连接凭据，桥 port10887/PID252640；不得输出 secret。服务器非敏感入口 root@connect.weste.seetacloud.com:12349，主机公钥 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。远端工作 NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM=/dev/shm/cross_tfg_audio_head_20261003。正在 PTY 分派训练输出 collector05 源码上传/CPU help 预检；它必须在训练 run 创建前完成。不存在已运行训练的证据。

训练实际 binding 位于 job_control/fit40_transition_head_v3_attempt05/fit40_transition_head_v3_launch_binding_attempt05.json，38470 字节 SHA8c135265bc323651ed16f63a2d0f25f4f9c8273ea761a718830e06002847663e；准备启动索引 `.codex/scratch/fit40_transition_head_training_packet_attempt05_revision02/launch_wrapper_attempt08_actualbinding_run01/root_launch_index_attempt01.json`。存储至少 4GiB、GPU 至少 18432MiB 且无其他 GPU 进程、cgroup 至少 8GiB。启动后先通过 40 音频逐位零参数一致和完整 lag/视觉一致，再做 120 步；每步必须核实非零梯度和实际 theta 变化，终端结果另行全量重评分。

## Changed Files
- `scripts/experiments/cross_tfg_audio_head_20261003/build_v3_candidate_only_dev_plan_attempt03.py` 已提交，180 新候选视频复用 360 个严格限定对照，不改变原音时钟或评分支持集。
- `scripts/experiments/cross_tfg_audio_head_20261003/audit_native_encoder_condition_metrics_attempt04.py` 与 transition ratio supplement01 为本地新代码；实际 NumPy 分析已执行，无模型 forward。
- 输出收集 builder05 修复绝对远端路径判定和实际 attempt01 chunk 文件名，collector04/verifier04 核心保持已审阅版本，上传索引 stage_packet_attempt06 已审阅。

## Verification
Ditto 根复核 `runs/cross_tfg_audio_head_20261003/native_encoder_ditto_retry_readback_attempt04/root_actual_full_restore_qc_attempt01.json`：tar 27740160 字节 SHAacc813112e40378ba5f7d3066aac5ce554bca2188a7764468ae24697fdf7705a；全部 58 恢复文件 27688481 字节匹配。接收主记录 collection_receipt 14496 字节 SHA8019860d0fabf389ba77512eec4df196963efaceed72c72b69adae1fd2337b1b。

原生分析 `runs/cross_tfg_audio_head_20261003/native_encoder_condition_metrics_attempt04/numeric_audit.json` 677048 字节 SHA22ed1ba05e30cdf2cfd2aa6e4cb49944a5070371f9642b61b10c5b40ac9d43f4；根实际 QC02 才是完整 336 行/42 聚合通过。QC01 因 PowerShell 保留变量 `$error` 触发非致命错误，产生的 PASS 无效，明确由 failfast QC02 取代，原件保留。JSON 身份时间用 ConvertFrom-Json -DateKind String；此前默认转换时间造成的 EM receipt 身份错误已在 EN 更正，原始 UTC2026-10-03T22:36:56.234390+00:00，linked envelope SHA470c0d7a7183ef5c9b84ee31bc0c28217f0a849d8bb737b2a92ca9eafa8c7e91。

精确 25 文件 fresh readonly02 66183 字节 SHA52d7fe9d9f73223d63fee187dc995e99b6601a7e1c4fb267c506b67b99fc3ffe；retirement03 实际执行响应 SHAe6b06095274745bbb72ae69c1df60dcd3a0f01179bac14800a63f402750cc20fe，成功状态 RETIRED_EXACT25_PRECODEC_SOURCE_FILES_AFTER_VERIFIED_RECOVERY；完整实际 receipt/postcheck 尚待取得。Sad audit03 failed04 envelope SHA6871dd195bee936026acf01789227ae1c77da60b4833bd1cb2a199f0c4d5f6d3，退出1无输出，保留失败并使用新请求诊断，不覆盖重跑。不添加或执行泛用 pytest。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Task 当前 snapshot73 与 Experiment72 已直接读回，后续用实际新 milestones 更新，不能创建进度日志。

## Observations
- [result] 三前端168 NPZ、336配对行、42等样本聚合已实际分析且根独立核对通过；1848producer复核零差异。
- [decision] 保留固定RMS作为共同基线，优先完成已绑定九参数头真实训练/终端重评分/三模型Dev；局部能量与谱形消融作为后续可检验机制。
- [decision] Confirm80和LatentSync保持封存；评分增益需原音轨和完整失败分母，不将特征距离称为生成效果。
- [blocker] 首轮训练尚未启动；存储门槛需实际回收后复核，Sad引用审核的运行前退出待诊断，不能据此称为模型失败。
- [next_step] 完成collector05源码预检、精确回收后的新鲜资源复核，再立即分派已绑定120步训练并核实零控制、实际梯度和参数更新。

## Relations
- continues [[Codex checkpoint - 2026-10-05T12-48-48Z - Verified storage copies and preserved first learned training route]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]