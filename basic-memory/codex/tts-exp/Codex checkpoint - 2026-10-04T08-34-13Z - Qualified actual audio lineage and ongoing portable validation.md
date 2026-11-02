---
title: Codex checkpoint - 2026-10-04T08-34-13Z - Qualified actual audio lineage and
  ongoing portable validation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t08-34-13-z-qualified-actual-audio-lineage-and-ongoing-portable-validation
project: tts-exp
agent: codex
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
git_sha: 7088c1e1a563a9b627946f313cbc324b135736ad
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
started: '2026-10-04T08:34:13.5212111+00:00'
username: dogle
status: open
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T08-34-13Z - Qualified actual audio lineage and ongoing portable validation

## Summary
v4 的全部180次原生生成已完成，并与20条真实Qwen内容检查精确连接；三前端配对评分和独立唇部验证继续推进。

## Story
用户从探究中文TTS为何改善TFG，转向可用于较多TFG模型的自然音频增强头。生成条件用h(x)，成片播放和主评分始终使用未改动自然PCM及原时钟。用户要求失败后持续迭代，实验与模型代码交给gpt-6-luna/max子代理，主代理负责理论、冻结协议、独立证据复核、BM与调度。Qwen ASR/TTS、服务器及网盘数据均获授权，秘密只在运行时。

当前开发集20条/10位说话人/3配对种子，覆盖Wav2Lip mel、MuseTalk Whisper、Ditto HuBERT。v4实际终态180/180 PASS，每模型60；正式收益仍须从完整原始矩阵重算并与原音和固定RMS比较。主代理已完成真实生成输入连接：暂存增强PCM、实际生成PCM、20个Qwen请求、原音评分PCM及样本时钟全部精确匹配。297个标注字符0识别错误，内容有效实验单位是20条，不能当作180个独立内容样本。确认集80条/40位说话人保持封存。

独立视觉基线实际启动240条Muse/Ditto identity/RMS。CPU预检通过所有视频与时钟文件及冻结运行时；最新只读状态中Muse已完成，Ditto正在处理，尚无全批终态。v4的180条视觉包已准备但等待该批完成。原始score小包与9路公共查询支持集消费者由cross_tfg_execution实现，视觉公共有效帧和原音40ms logRMS关联由mechanism_audit实现。不能用Sync-C上涨代替独立嘴部运动或质量证据。

共享局部谱候选540科学计划保持不变，执行包07实际完成命名空间、两上传及29项安全解包；CPU执行预检和GPU启动尚未运行，因为SHM不足5 GiB。恢复第一份旧fit输出的SFTP继续在隐藏本地进程传输，尚无完整档案、B镜像或严格恢复，因此未删除源。另三份旧输出的只读SHA去重估算包已准备，尚未执行；Muse/Ditto旧源仍被当前视觉任务引用。

SadTalker实际attempt06已完成原音/RMS两臂工程验证，LatentSync历史fit两臂92×31评分已完成；二者都不是新头迁移收益。主代理审查发现Sad取回02错误指向旧attempt04，Latent取回06把不可变输入路径05误改成06，均未执行。此前Latent取回05实际仅因audio_sample_count字段名而失败，失败产物保留；代理正在修正新的不可变版本。

## Working State
权威方案为 `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`、`transfer-iteration.md`，任务snapshot37/实验snapshot36实际写入并精确读回，维持active/implemented。报告25是已有机制结论，报告26尚未形成。ADR011规定μ(f)=mean_t M(f,t)，δ=M−μ；分解单元使用共同κ及压缩/RMS/相位/时钟，不独立clip或假设视频效果可相加。

Git HEAD `7088c1e1a563a9b627946f313cbc324b135736ad` 为本地未发布，分支codex/cross-tfg-audio-head-20261003，无PR。历史dirty文件保留，未添加或运行pytest。临时取回包、响应、QC和runs均为机器本地/忽略产物，不能仅凭checkpoint假定仍在。

易失运行时：PowerShell PTY50274保存桥接及SSH/API SecureStrings；SFTP隐藏PID400448，stdout/stderr异步收集。服务器root@connect.weste.seetacloud.com:12349，主机密钥SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码不落文件。远端项目 `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`、输出 `/dev/shm/cross_tfg_audio_head_20261003`。新鲜视觉launcher PID531095/ticks767391522；不得提前退休旧Muse/Ditto源。远端/autodl-pub是公共NFS，禁止用作私有实验归档。

## Changed Files
主代理实际内容连接产物 `runs/cross_tfg_audio_head_20261003/v4_asr_render_content_join_attempt01/`，独立检查器 `.codex/scratch/root_v4_actual_render_content_qc_attempt02.py`。共享07执行包 `runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/packet_attempt07/`。视觉基线请求和真实响应 `.codex/scratch/cross_tfg_audio_head_20261003/visual_baseline_guard_attempt02/`。存储辅助源位于 `.codex/scratch/recoverable_output_archive_attempt02/`；各旧失败版本保留。

## Verification
实际v4终态manifest SHA `e3772dab9b020e390246ddcc41ac875598ce2f5f51cee88d41f68e08e74cf718`，results SHA `0f1f009207c85087ff7eab7378221ecdfe4d6ab3a01d83442b29e211871da5eb`；本地原样取回180行并核对。plan raw SHA `70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c`。

独立80-WAV/Qwen内容QC回执SHA `29123ef69f6261b3a6c9da85f73947d93e8fb379ac853d6093c52a2c60049005`；实际180生成连接独立QC回执SHA `8948fc0dd29e90ecdf40baa81b5beca87c7859fd8e2124805cb117d8c57bc14c`，完整20×3×3网格、原音/增强PCM/文件散列/时钟/源脸视频全部PASS。

视觉基线packet02 index SHA `a158e3be7bc7ccd6f4492175b5124e57ae4e2c128d2216fcddb177f37a846756`；主代理逐行核对120+120派生子集与原ledger完全一致。17_preflight真实PASS240；18_launch真实PID531095；continuation01实际alive/stderr0/progress3，Ditto RUNNING。全批终态未出现。

共享07请求SHA/尺寸及29zip项独立检查、所有外层exec命令经bash-n精确LF验证PASS；实际00–03成功，04/05未运行。B镜像助手SHA `5cc1e03d99368f4792124abb0fb7a9f3c879bddbaeaeb5ca9b01a074b467cc62` 与严格恢复器SHA `8f9574c70d7f6c11a77590f0cc16610091ad2f126a3828d1f7b869bd1e028f7a`已完整审查，未执行。恢复档案1,279,971,156字节SHA `cd567ec15fade9d8f1d1ce6c0d1ae043e0c1d8042aa7178caf546c69b0f69039`，47,105源文件仍保留。

## References
[仓库](https://github.com/dogLeft233/tts-exp)，[已验证远端提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前HEAD仅本地，未发布。

## Observations
- [result] v4原生180全部完成；20条真实Qwen内容检查精确绑定180实际生成输入，标注297字符0错误。
- [result] Muse/Ditto独立视觉基线CPU预检240通过，实际Muse已完成、Ditto在运行，全批封印待出。
- [decision] 工程源码修订必须保留既有输入路径，仅修改新输出命名空间；全局字符串替换可误指不可变输入，审查应逐项对照真实ledger。
- [decision] 归档完整散列、C主档/B副本和逐文件严格恢复以及新鲜活动引用检查全部通过后才退休指定旧源；当前未删除。
- [blocker] 共享540生成仍受SHM空间门槛限制；v4原始矩阵分析及独立视觉结论尚待完成。
- [next_step] 取得v4原始矩阵小包并按冻结FULL/GUARD20公共支持集重算三前端相对原音/RMS的效应，同时完成独立视觉验证，据此继续迭代增强头。

## Relations
- continues [[Codex checkpoint - 2026-10-04T07-56-21Z - Completed native dev execution and content qualification]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[011-common-scale-mask-and-envelope-ablation]]