---
title: Codex checkpoint - 2026-10-04T07-56-21Z - Completed native dev execution and
  content qualification
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t07-56-21-z-completed-native-dev-execution-and-content-qualification
project: tts-exp
hostname: LAPTOP-E133GSLE
status: open
started: '2026-10-04T07:56:21.9133067+00:00'
repo_root: E:/Documents/tts-exp
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
username: dogle
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T07-56-21Z - Completed native dev execution and content qualification

## Summary
原音增强头 v4 已完成三类 TFG 的 180/180 原生生成与原音评分，Qwen 内容保护独立复核通过；下一步必须从完整原始产物判断收益并补齐独立视觉验证。

## Story
用户从追查中文 TTS 改善 TFG 的原因，明确转向可迁移的自然音频增强头：只增强生成条件，成片播放及主评分均换回未改动的自然 PCM 和原始时钟。用户要求持续迭代到可用方法，并指定实验和模型代码由 gpt-6-luna/max 子代理承担，主代理维护理论、冻结方案、独立复核和 Basic Memory。阿里云 Qwen ASR/TTS、远程 GPU 和百度盘数据均已授权使用，秘密只保留在运行时。

报告25的已核验路线是局部谱结构影响编码器方向和生成口型，单纯响度不能解释完整效应；直接移植供体谱统计在既定中文支持集上存在受体兼容性问题。当前同时推进原生训练头 v4 与只由 fit 选择的共享局部谱候选。三类前端为 Wav2Lip mel、MuseTalk Whisper、Ditto HuBERT，开发集20条/10位说话人/3配对种子；确认集80条/40位说话人仍封存。SadTalker 和 LatentSync 已分别完成工程配对，但还没有增强头迁移效果结论。

本轮 Qwen 对20条 v4 音频真实调用全部成功，主代理从归档里的原始返回、官方转写及80份浮点/整数音频独立解码核对：297个标注字符0错误、20个独立请求ID、原音与增强音频散列及时间轴一致。内容检查要在最终生成 ledger 与暂存增强 PCM 精确连接后，才成为实际方法的保护证据。新鲜 status10 已确认 v4 180/180 PASS（每模型60），子进程退出、GPU空闲、540个阶段全部完成且无 stderr。这仍是执行结果，尚未从全部92/31式矩阵或视觉轨迹得出新头有效的结论。

存储恢复与研究并行。第一份旧 fit 输出已只读归档，1,279,971,156字节的档案已封印，源仍保留。慢速 SFTP 从已校验的228 MB前缀恢复到隐藏本地子进程，释放私有调度终端；最新已校验276 MB。必须本地全档散列、百度盘挂载副本散列、严格恢复与新鲜活动引用检查均通过后才能退休指定旧输出。旧 Muse/Ditto 输出还为独立视觉基线所引用，禁止提前退休。

## Working State
权威方案是 `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md` 与 `transfer-iteration.md`，当前任务和实验条目保持 active/implemented，正式报告26未写。Git HEAD `7088c1e1a563a9b627946f313cbc324b135736ad` 为本地未推送提交；已有历史 dirty 文件保留，不进行整体暂存，也未添加或执行 pytest。

机器本地、易失状态：私有 PowerShell PTY 50274 保存 SSH/API SecureStrings 和桥接会话。SFTP PID400448 于07:52:31 UTC启动，stdout/stderr异步读取，密码仅 stdin，无凭据文件或命令参数。服务地址和主机密钥在传输源及既有 server-access 指针中，凭据须从运行时安全恢复。服务器命名空间 `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`，输出 `/dev/shm/cross_tfg_audio_head_20261003`。当前 root 空闲2.293 GB、SHM1.111 GB，下一轮540实际生成仍未启动。

三子代理分别负责：mechanism_audit 的原始180证据包/视觉保护/内容连接；cross_tfg_execution 的540执行包和 SadTalker 配对证据；local_recovery 的可恢复存储及 LatentSync 原始92查询证据。必须保存完整失败分母和 fit/dev/confirm 边界，不按开发集改选已冻结 fit 排名。

## Changed Files
核心可追溯产物位于忽略的 `runs/cross_tfg_audio_head_20261003/v4_asr_content_evidence_attempt03_readback/`。主代理独立检查器 `.codex/scratch/root_v4_asr_content_numeric_qc_attempt02.py` 使用 RIFF 解码和 NumPy，避免依赖本机缺失的 SciPy。冻结的 SFTP02、严格恢复器和后台启动回执位于 `.codex/scratch/recoverable_output_archive_attempt02/`。任务 snapshot36/实验snapshot35 已写并直接读回；ADR011已创建并读回。

## Verification
内容证据 archive03 SHA `6bcf686ea20ce088d97267b36af2515f2a4e1ff7c83ff9c6e701eecd7028a52a`：129 tar项/128源项恢复散列通过。独立内容检查回执 `root_independent_content_qc_attempt02.json` SHA `29123ef69f6261b3a6c9da85f73947d93e8fb379ac853d6093c52a2c60049005`，20条/80 WAV全部通过；旧01环境失败保留。

180终态的直接服务器响应：`.codex/scratch/root_status_20261004T0754Z/v4_dev180_status_attempt10.json.response.json`；实际 result_manifest SHA `e3772dab9b020e390246ddcc41ac875598ce2f5f51cee88d41f68e08e74cf718`，plan raw `70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c`。Sad06 status13 为2/2 PASS_ENGINEERING_FIT_ONLY，不能当作增强效果。

共享540科学计划仍原封不动，raw SHA `43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`、payload `dfc9ced014839c2480228de0d9930a9690ee160cbc91af6784440e1c07bb3c58`。先前不一致来自 Windows 默认 GBK 解码，直接 UTF-8/raw bytes 复算通过，不需修正科学计划。执行 overlay 最新源码及新 packet03 仍需主代理逐项审查。

恢复档案 SHA `cd567ec15fade9d8f1d1ce6c0d1ae043e0c1d8042aa7178caf546c69b0f69039`、内部 manifest `f7c02fcb03aaf3cf9631fd4b1d031e789cf4e2578981353513b2e0cba852660a`；源含47,105文件。C盘主档位置 `C:/Users/dogle/.codex/tts-exp-artifacts/recoverable-output-archive-20261004-attempt01`，B镜像目标 `B:/tts-exp-archives/recoverable-output-archive-20261004-attempt01`；尚未全档完成/镜像/恢复/删除。

## References
[仓库](https://github.com/dogLeft233/tts-exp)，[已验证远端提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前分支无 PR，本地 HEAD 未发布。

## Observations
- [result] v4 三模型原生开发180全部完成且原音评分执行通过，效应大小待完整原始矩阵独立复算。
- [result] v4 的20条真实 Qwen 请求和80 WAV独立内容检查通过，297个字符0错误。
- [decision] 独立视觉基线必须覆盖三前端：既有300轨迹只属于Wav2Lip；新Muse/Ditto各120 identity/RMS正在准备，源输出保留。
- [decision] ADR011中 μ(f)=mean_t M(f,t)，是沿时间求均值的固定谱分量；δ=M−μ。四个分解单元共同使用同一 κ≤1 和相同压缩/RMS/时钟。前一CT检查点关于μ轴的简称有误，任务36已更正；不得独立clip两分量或假定视频效果可加。
- [blocker] 540实际执行受空闲空间门槛限制，必须完成可恢复归档后腾出空间；新头实际内容连接和三模型独立视觉检查仍待完成。
- [next_step] 取回180完整原始证据并核对生成输入与ASR暂存PCM，同时启动Muse/Ditto独立视觉基线；据冻结指标判定v4和共享候选，失败时继续迭代。

## Relations
- continues [[Codex checkpoint - 2026-10-04T07-07-47Z - Qualified Qwen content and portable method validation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]