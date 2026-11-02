---
title: Codex checkpoint - 2026-10-01T02-31-34Z - Completed null audit and active mechanism
  execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t02-31-34-z-completed-null-audit-and-active-mechanism-execution
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 2b78a03dfe32c4702dde15b573087563ee1e2cff
started: 2026-10-01 02:31:34+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-01T02-31-34Z - Completed null audit and active mechanism execution

## Summary

独立重建完成静态方向/范数的全域置换分析；随机剂量复跑已实际进入 GPU，原生 TTS 的完整音频域时间配对诊断及女性队列预检已启动，音量与 WORLD 机制进入经过静态复核的执行阶段。

## Story

用户要求恢复未同步的研究路线，并持续找出 TTS 增强 TFG 的原因，由 root 维护理论、推理、实验主线与本地 BM，实验和代码交给 gpt-6-luna/max subagents。最新补充允许用本地百度盘解决空间问题。本轮继续分开音频对生成器的作用与对 SyncNet 评分的作用：主实验的 RMS 生成路径已经有正效应；独立视觉测量也显示可见嘴部运动幅度和速度增加。接下来把音量沿 mel 坐标和实际编码器方向/范数拆开，另用 WORLD 在保持音素时钟与全局 RMS 的条件下做双向声源、谱形、局部能量移植。静态 TTS 均值方向实验只回答已冻结的静态轴，不概括全部音素或声源机制。

本轮完成并读回静态范数实验的全域补充：80 clips/40 speakers、320 scores、36,756 文件、80 latent factor checks，独立矩阵与五类统计重建零失败。200 个置换 excess CI 中，所有 C/anchor/search 区间跨零；10 个 FULL/VALID 的 B/D 次要距离端点区间不跨零。guard20 raw 方向 C excess −0.013649，99% CI [−0.034013,+0.007979]；FULL/VALID −0.000809 [−0.018762,+0.016733]。这进一步限定静态方向结论，不用次要距离结果改写预设 C/anchor 主判定。原始控制记录是有效 JSON；先前担心文字换行损坏的假设已被实际字节/解析核查排除。

随机对照早先 Float32 实际剂量误差超标，保留失败，不放宽 1e−6 门限。重建算法经过逐帧 Float32 实际剂量核查，再新命名空间重新跑 640 行。CPU materialization 已完成 80 contexts/640 row refs/880 arrays、零 integrity/dose/cap 失败。aux attempt03 已实际进入 GPU，最新已读状态为 5 samples/40 rows、零失败，实际 forward flags 符合 deterministic 合同。此次完成了可运行的排程和预检，尚未把启动或部分行写为科学结果。

## Working State

### 已封存证据

- 原生嘴部几何补充：`runs/recovered_20260930/independent_validation/native_geometry_visual_20261001_attempt02/analysis_v2/native_geometry_contrast_supplement.json`，SHA `b713a1f0bfc5d38ce8684d52b4dccd70f9153757283e37cb3c00ba48d7923c39`。80/40 下 Traw−Nraw aperture +0.010824 [0.008752,0.012922]、velocity/s +0.074732 [0.063392,0.086525]；Nfixed−Nraw aperture +0.002899 [0.001844,0.003940]、velocity +0.022588 [0.014555,0.030767]。这是运动响应；phone separation/AUC 的 16/17 speaker 合格队列未达到正判定，不能把几何结果写成音素准确度或感知同步。
- 静态范数全域补充：`runs/recovered_20260930/independent_validation/norm_all_domain_supplement_attempt02/independent_norm_reanalysis.json`，12,291,558 bytes，SHA `6659867e237adf065d87b6bfaaa7039518460960861458502ec44e751a9e97f4`；gzip SHA `38cea3d3c3a05698358a87568c4237968cf03ea351029527191bd954146668cf`。root 下载、解压校验和实际数值读取完成。旧/新 q00/q11 pixel/V/matrix 工程 bridge 仍待只读补充。
- 主实验 archive 301/301，1,587,388,518 bytes 全部 size/SHA 通过；本地验证 SHA `a0fce1e7e62d95d6ac15a70677a70c859d428b07e8f51046e5a6ae1a751172ce`。它们在 `runs/recovered_20260930/remote/evaluation_20260930_retry3_analysis_20261001/`；未全量解包且包中无 MP4/JPEG。不可误用旧 237 包 ledger。

### 当前机器上的活动执行与准备

服务器是 `root@connect.weste.seetacloud.com:12349`，以 weste 为准；连接密码和 API key 只保留在用户消息/进程运行时，不进入文件、笔记或参数输出。新任务根 `/root/tts-research-system-20261001/` 是私有空间；Python `/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python`。每个需 PSF 的子命令显式设置纯包 `deps/python_speech_features_only_attempt05/` 的 PYTHONPATH；不要混入 SyncNet 的 NumPy 2 site-packages。

- 随机 aux GPU：`source_axis/random_specificity_aux_20261001_attempt03/`，parent PID864415/child864417；code v2 SHA `792e0522572f5d02f476519d85139849af472348f723bd3c730a090a530a43e7`。LF-only launcher v6 SHA `976b65c4fcbe9f31cc0b8bd22b90fab7e9624038507c3d5e4ac174d8b89051ea`。status request `.codex/scratch/random_specificity_aux_attempt03_status_v2.json`。实际 contexts 在 `RUN/samples/*_latent_context.npz`，结果 `evaluation_results.jsonl`；不能按旧 glob 判 0 rows。完整 CPU QA SHA **`487bedd831e00012ebbf2524e5897454d57c098322868a35ac09c412948bae2b`**，是 64 字符；前一 immutable checkpoint 中多写了一个 0，仅笔记抄写错误，产物未变。
- 原生 cached-null CPU：`independent_validation/native_cached_null_20261001_attempt01/`，PID24950；source SHA `dc7a39903d234a27958d475f0800f73a1337fe26a339ca0a0ab7a91c1713a24a`，preflight 2560 plan/results/prior80π + source/spec/gzip/core seals PASS，launch record `aea6140ee09d076d0357775c31dd43a404eea9453fe50364e64487e05cf4d83ab`。主 retry3 没有 norm 格式 `all_files_sha256.json`；failed launch01 保留，launch02 仅删除错误的该文件存在要求并修复闭合引号，真实数据绑定不变。status request `.codex/scratch/native_cached_null_status_attempt01.json`。
- 女性 CPU preflight：`preflight/female_replication_attempt02/`，PID917832；status request `.codex/scratch/female_replication_attempt02_status.json`。input 40 clips/20 speakers；目标 640 rows/primary face。package SHA `8e8320c1f4c9bf9f57581155eda71d33eae3e705c939b4c8fe6d0611b87528a9`，stage v2 `cd561386b20e2f55a33d9ed04a4f5211beeb0c122501c337ba15029415820de9`。只有无模型前向预检启动，不是 female GPU 完成。
- Mel320：CPU CAL40/EVAL80 数值 partition 和独立重建完成。GPU attempt03 runner SHA `7d93a3656baebfec87f463bff0452a939ca2eda42e1c8cac18372f29ce52e738` 静态 PASS，修正 official N−5 窗口、每次 actual flags、latent→decode→crop→V 严格帧数、M0/M1有限及范围门限。正确未 dispatch bundle `.codex/scratch/mel_level_partition_20261001_attempt03_v2/` 绑定 mel spec `abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8`；v1 错绑 RMS spec，保存但禁止执行。assets/preflight CPU 可先做；GPU 等 aux 释放。必须保存 q00/q11 实际 z 与 hash/flags 给 true RMS，不能从旧静态轴取端点。
- True RMS：最新 no-forward CAL attempt06 20 clips/10 speakers PASS；source `adf42b4324b20097050465d1541dfa27ef390e904cc30c83d67047e516a7d808`，QA `f12ec636ea983a80855f458d04329099c97b40b6aebc2b2ec3f259bb30fe6833`，reviewer 静态 PASS。新增 EVAL 每帧 mixed radius≤1e−6 fail gate 与严格 decoder/V 时钟，q00/q11 exact endpoints。prepare bundle `.codex/scratch/rms_latent_factorization_20261001_attempt03/request_bundle_index.json` 未 root dispatch；需实际 mel03 run seals/encoder z 后才能做 EVAL factorization。CAL方向输入位移较大、范数平均约增加3%仅是数值校准描述，不是生成贡献结论。
- WORLD：物理 EVAL 1440 main+480 QA/fixed WAV 和对应 NPZ、80 pairs/160 recipients 完成；manifest `27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3`，plan `f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e`。最新 GPU runner `248031eeb86f69299557b351b4855f743a975279e50413cfa1c8750a6056dbcf` py_compile/diff-check + reviewer static PASS；完整 V×A cross/full-V zero，仅共同 V 前缀查询31lag且使用完整A；所有模块前向 flags 硬门、逐段帧数/音频窗口验证。尚无 CPU execution preflight/模型前向/效果结果。全设计1600 videos/5760 unique matrices/12160 labels；donor→recipient方向、RAW和within-WORLD family及 synthesis bridge 保持冻结。

机器状态不构成可复现结果：authenticated bridge/PTY、运行 PID、ignored HANDOFF 与 `.codex/scratch/` 都可能失效，恢复时先取当前状态。root only dispatch；subagents 为 local_recovery、remote_recovery、mechanism_audit，均 gpt-6-luna/max。

## Changed Files

`CONTEXT.md`（91行）和 WORLD alias annex 已在 local/unpushed `2b78a03dfe32c4702dde15b573087563ee1e2cff`。本轮更新 ignored `HANDOFF.md`、BM task/experiment/reference 多个笔记；若干 edit 后读回仍需补完。实验脚本与请求包多数 untracked，包含新 cached-null、norm 全域补充、mel03、RMS、auxv2、female、WORLD runner；旧测试文件既有 dirty 状态，不 stage-all，不运行 pytest。CodeGraph 无可用索引，已检查后用 rg。

## Verification

root 实际执行：norm supplement02 零失败报告下载/SHA/gzip重建/统计读取；native geometry 补充 SHA/实际数值读取；core301包校验；aux CPU materialization gate、launcher CR0/bindings、actual GPU flags/首40行；native cached-null 启动 preflight与 launch record；female package/source/纯PSF树、CPU启动；正确 mel03 bundle 两个 worker hash/CR0/spec 绑定。agents 实际 py_compile/static reviews 检查 RMS06/WORLD最新合同，通过但没有给出 GPU 结果。未增加或运行软件测试。

最新 repo/branch/root/SHA 已直接读取；GitHub read-only API 返回 canonical repo、pushed baseline可解析、此branch open PR=[]。本地 HEAD 不构造 GitHub commit链接。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/confirm-world-source-filter-20261001/evaluation-aliases.md`（51行，SHA d14260e…2f3f）
- `openspec/changes/confirm-mel-level-partition-20261001/spec.md`
- `openspec/changes/confirm-rms-latent-factorization-20261001/spec.md`
- `openspec/changes/diagnose-native-geometry-visual-20261001/spec.md`

## Observations

- [result] 静态方向/范数全域重建完成，200 excess CI 所有 C/anchor/search 跨零，10个FULL/VALID B/D次要距离区间非零；不改变预设主端点判定。
- [result] TTS 与 RMS 自然音量干预都有独立可见嘴部运动幅度/速度响应；音素区分与事件时钟尚无合格正判定，几何变化不能直接命名为感知同步改善。
- [decision] 继续按生成/评分四格与完整音频置换域分离机制，WORLD保持音素钟和全局RMS，真实RMS使用实际mel encoder endpoints；不把静态平均轴当全部TTS机制。
- [decision] 百度归档使用本地 `B:\tts-exp-project\archives` 网关，分块/hash/member/source-mapping验证再清理副本；当前 weste 无发现现成百度挂载。挂载写入不等于云端持久性确认，未删除科学证据。
- [blocker] aux640尚未完成独立消费；mel/RMS/female/WORLD尚需真实预检、GPU执行和独立统计消费，GPU保持单任务；native cached-null仍待完成报告。不能在启动或静态通过时结束研究。
- [next_step] 先查询 aux/native/female 当前状态并完成 mel03 assets与CPU preflight；aux封存后按单GPU队列执行mel→true RMS及female/WORLD，完成独立重建与机制综合，再写最终25报告。

## Relations

- continues [[Codex checkpoint - 2026-10-01T01-46-30Z - Visible loudness response and qualified execution gates]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[原生 TTS 优势的时间配对与可见口型诊断]]
- relates_to [[句级音量在编码器中的方向与范数响应]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]