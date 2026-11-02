---
title: Codex checkpoint - 2026-10-04T07-07-47Z - Qualified Qwen content and portable
  method validation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t07-07-47-z-qualified-qwen-content-and-portable-method-validation
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
project: tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
repo_root: E:/Documents/tts-exp
username: dogle
agent: codex
started: '2026-10-04T07:07:47.4323953+00:00'
model: gpt-6.1-sol
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T07-07-47Z - Qualified Qwen content and portable method validation

## Summary
新候选 v4 的20条独立Qwen ASR调用全部完成且转写通过；共享频谱候选的540行开发计划已冻结并独立核对，跨模型收益仍由正在运行的视频结果与后续视觉检查决定。

## Story
用户最初要求从本地代码、Git与历史数据恢复TTS改善TFG的研究路线，现在明确要求产出可用于多个TFG模型的自然音频增强头，并持续迭代到方法完备。实验约束保持为：增强PCM只送生成器，视频换回原始自然音频，主SyncNet评分也使用原始PCM和源时间轴。Root负责理论、协议、Basic Memory和独立证据核对，实验代码与运行由gpt-6-luna/max子代理承担。

本阶段同时推进学习式v4和共享局部谱形方法。共享fit的360行原始曲线已经由root独立复算，选择器只用8条历史fit音频冻结三个配置。新的540行开发计划通过root逐行检查，覆盖20条开发音频、10说话人、三个模型、三配对种子、三候选；控制臂为identity、rms_fixed、rms_localcompress。该计划仅为元数据，尚未执行540次生成。执行closure发现旧Ditto gate目录会被计划env覆盖，子代理正在制作仅修改操作路径的版本化overlay。

v4训练与20条CPU音频stage此前已经通过独立数值核对。本次20条新候选Qwen ASR调用真实完成，exit0、stderr0、20/20 PASS，gold CER与相对原音转写差异均0。复用的20条原音缓存要求原始F32与FFmpeg PCM16精确一致，gold文本从未送给provider。ASR只提供内容保真证据；视频改善需要原始距离矩阵、stage-to-render PCM joins与独立口型测量。v4视频评估最近实读为108/180 PASS、三模型各36、0失败，仍在运行。

局部声学解释已记录为ADR011：每帧频谱mask拆成频率平均μ与频率残差δ，所有消融臂用同一个有界κ，避免裁剪使μ/δ/full无法对应；时间压缩包络另作控制。谱mask每帧中心化只证明τmask=0，不能把视频效应直接归为线性中介。该消融尚未执行。

LatentSync的历史identity/RMS配对在原始源时钟完成工程评分，92个共同query，Cbest分别3.515与3.464；这不是新增强头的heldout收益。原始矩阵归档仍待独立核对。SadTalker在上游尺寸1平移量拼接处失败，新的窄scalar兼容shim保持官方几何表达与全局NumPy行为；请求09的JSON换行错误在运行前被root发现，fresh10修正，尚未发起新的06实验。

存储安排为C盘主归档/完整恢复，B盘百度CloudFS第二副本，E仓库只留小回执。四个旧输出只读审计已实际完成，合计约5.5GB；旧失败输出保持FAILED历史。第一个recovery04归档包已修复root审阅发现的SHA返回tuple误用，fresh02等待dispatch；未删除任何源输出。ASR小归档01实际失败：transform source audio_enhancement_heads.py位置缺失，原始20条ASR结果仍完好；子代理正在按qualified closure精确路径准备fresh02，不能放宽科学hash。

## Working State
持久研究任务为[[开发跨 TFG 的自然音频增强头]]，实验为[[跨 TFG 自然音频增强头的训练与原音换轨验证]]，机制报告25保持CONCLUDED。Task35/Experiment34先前读回成功，但其ASR0字段早于本次20调用完成；下一次快照应在原始ASR归档核对后更新。最终报告26与方法尚未完成。

机器本地/fragile：工作目录E:/Documents/tts-exp、分支codex/cross-tfg-audio-head-20261003。本地HEAD `7088c1e1a563a9b627946f313cbc324b135736ad`未推送；该提交只更新CONTEXT.md与transfer-iteration.md，HANDOFF.md为ignored本机快照。历史dirty核心脚本/测试与未跟踪实验文件保持原状，没有stage-all。远程server root@connect.weste.seetacloud.com:12349，hostpin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码/API只在运行时私有stdin/env，禁止记忆/文件/argv输出。私有PTY50274和bridge端口2133是fragile，不把存在当连接正常的证据。

## Changed Files
- CONTEXT.md与openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md已提交本地7088c1e；描述已实读状态。
- shared_profile_shortlist_dev_attempt03/plan.json与planned_rows.jsonl为冻结540元数据；root_independent_metadata_qc_attempt02.json为独立核对。
- .codex/scratch/cross_tfg_audio_head_20261003/stageb_v4_asr_content_attempt01/保存真实ASR安全launch05、状态读回与失败archive01原始回执。
- docs/adr/011-common-scale-mask-and-envelope-ablation已在tts-exp Basic Memory创建并读回，status open。

## Verification
共享fit原始archive SHA bf35f9377db4a2ce222369c0bc976dba8cc40a4c46e3d9b16e3ddc3326bb6223：367成员、360 NPZ、2520数组、独立utility误差0。540冻结plan raw SHA 43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df，rows64a41fb0c76943ac535d38afa9347d9ca0e91ad075e54bb514be262945a63abb；root QC02 SHA4932d5df61e4b972f4cd8307759dc2ab53e8d3f9b6160c0d0e323c1cea3b1bf8。

ASR真实status在requests_status_attempt01/00_read_v4_audio_asr_status_attempt01.json.response.json：COMPLETE/exit0/20评分全部PASS。results raw bff2b0324001a0f3e99fb3ded08b32718e6ebc0ce00af7767f3ed2e154f54b36，scores877569bc65c7947664f3e6baf82a284f2770fbe9e10e4e1ae2b5128a0ffb8ba4。archive01失败记录02_build_archive_attempt01.json.response.json，code1/source missing。此前ASR client04错误仅是preflight raw文件hash重建误差；client05用实读8514ceeb0c47bec660fe29bfd8b2b9bd912377f3ef341cb5141293fa83e65291成功，未把工程失败当科学阴性。

没有新增/运行pytest或unittest。运行的核对均为实际run产物读回、哈希、原始数值独立重算、元数据逐行连接与窄源码检查。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [已确认GitHub可解析的提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推送HEAD `7088c1e1a563a9b627946f313cbc324b135736ad`；该分支read-only GitHub查询无PR。
- openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md冻结科学标准；confirmation40说话人/80音频保持sealed，确认模型/种子family须先冻结。

## Observations
- [result] v4新候选20条独立Qwen ASR全部完成通过，CERgold0且与原音转写差异0；尚需stage-render PCM精确连接。
- [result] 共享fit360原始曲线与540开发元数据独立QC通过；540生成没有启动。
- [decision] 使用ADR011的commonκ分解，将每帧μ、频率残差δ、完整mask与时间包络分别比较，避免独立裁剪改变消融含义。
- [decision] C盘保留可恢复主归档与完整逐成员hash恢复，B盘保留验证副本；全部验证且再次检查活跃引用后才允许退休限定源输出。
- [blocker] ASR archive01实跑缺少一个transform source路径，fresh02修复中；视频视觉和全套开发收益尚未得到终态证据。
- [next_step] 先review/dispatch精确路径修复的ASR证据归档、Sad10与recovery04归档，再独立核对20内容证据并取得v4终态原始矩阵；持续540、视觉、heldout确认直至可用方法和报告26完成。

## Relations
- continues [[Codex checkpoint - 2026-10-04T06-32-29Z - Frozen shared candidates and staged content qualification]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]