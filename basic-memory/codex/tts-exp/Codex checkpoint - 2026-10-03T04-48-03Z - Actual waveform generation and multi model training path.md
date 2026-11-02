---
title: Codex checkpoint - 2026-10-03T04-48-03Z - Actual waveform generation and multi
  model training path
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t04-48-03-z-actual-waveform-generation-and-multi-model-training-path
started: '2026-10-03T04:48:03.4471894+00:00'
git_sha: e9e2dd1c82f4dfb28ec099f93facbb3eace51878
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
username: dogle
repo_root: E:/Documents/tts-exp
status: open
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
repository: dogLeft233/tts-exp
agent: codex
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T04-48-03Z - Actual waveform generation and multi model training path

## Summary

自然音增强头已有波形实现、独立评分消费者及实际Wav2Lip生成/原音换轨/新鲜SyncNet评分；下一步是在真实梯度更新后，用新说话人开发集选择可跨模型使用的方法。

## Story

用户最初要求恢复未同步的TTS增强TFG机制，报告25和可恢复交付已经完成。最新目标是产出适配较多TFG模型的自然音增强头：增强音只用于驱动视频，成片保留自然音，失败后继续迭代至完整可用方法。create_goal ACTIVE，无预算。ROOT维护理论、设计、BM与Git；代码和实验由gpt-6-luna/max子代理执行。旧机制提示增益与局部谱形影响生成器的声学表示，但可移植方法须实际生成视频并用原自然音评价。

方法采用等长16k mono float32波形，独立控制增益、局部包络和平滑有界STFT幅度残差。identity/RMS先验证全链，随后在历史fit训练、fresh dev选择、新说话人及未优化TFG确认。原音F32无损换轨、官方评分PCM16派生路径、可微浮点视频代理分别记录。训练不能只降低D0：本次实际RMS的Cbest提升来自背景距离B相对Dmin上升更多，而C@0略降，因此采用fit校准时延或平滑best-lag置信度目标。

## Working State

权威设计 `openspec/changes/develop-cross-tfg-audio-head-20261003/{spec.md,tasks.md,data-protocol.json}`。当前分支codex/cross-tfg-audio-head-20261003，local/unpushed HEAD e9e2dd1c82f4dfb28ec099f93facbb3eace51878；没有PR。ROOT scoped提交设计、历史fit边界、实际fresh roster及已实现阶段复选框。CONTEXT93行，既有legacy/tests修改保留。代理代码目录尚未提交，不能仅依赖Git恢复。

官方AISHELL/AISHELL-1 revision bbe295d530192a4cd41644b711c9aecd087df653。历史fit S0052–S0101，每sp16条；fresh男性S0002–S0051按固定hash分dev10/confirm40，2clip/sp。dev为S0005,S0022,S0024,S0037,S0038,S0039,S0041,S0044,S0049,S0050；确认40尚未下载/读取。data-protocol raw cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87。十个dev官方归档实际下载完成，`B:/tts-exp-audio-head-20261003/aishell_dev_speakers/development_pack_manifest.json` raw ead6a21a3a2f2ce0d4481eb5df48dd4194dbf23e10eeac7652920da24707b84d，COMPLETE，confirm_speaker_ids_read空。mechanism_audit独立核对发行revision、LFS大小/SHA及三个speaker roster互斥。local_recovery正在materialize_dev_rows.py复核实际归档bytes、按lex选择3.2–8sec有限16kmono，最新8/20 canonical WAV；完整行manifest尚未冻结。实际科学行/脸/模型角色/种子/方法/统计族须在新dev效应前冻结；主确认范围是中文男声。

`scripts/audio_enhancement/`有严格F32波形I/O、identity、RMS目标.0376838172675746、bounded谱形与局部包络、zero-init卷积STFT残差、CLI、PCM streamcopy mux、训练API与Wav2Lip可微适配器。总mask有界，post-head RMS匹配隔离增益，时钟与samplecount不变。实际synthetic I/O receipt `B:/tts-exp-audio-head-20261003/local_io_qc/audio_io_qc_receipt.json` raw ed64b4b4e1b74c2bb6b830a5741b0ad4ac313718fcaf515126a709e0febeecc2，ROOT读回PASS：48k stereo PCM16转8000 canonical sample、identity精确、RMS/其他baseline有限等长、silence精确。范围仅I/O。真实head一梯度/参数更新尚未运行；local已把API与source/context契约交cross执行方。Wav2Lip无.9 peak normalization；Torch前端constant padding/scipy float64预加重需要实际parity。训练可先用旧暴露小fit子集，不等全800条下载。

`scripts/audio_enhancement_eval/consumer.py`独立consumer已实现FULL共同query支持/guard20、31lag、等权speaker/model、共享bootstrap有限95%联合族、失败/缺失分母保留。主指标采用精确CPUtorch float32 mean/median/subtraction、2.5.1+cu124、1thread；float64仅诊断，bestlag k与官方AVoffset -k分开。dev-only无confirm可用，缺少声明seed/method保留INCOMPLETE；sealed result ledger仍一计划行一结果行精确有序。最新代理静态AST通过，consumer0bc7064b9ad7543e9fdda62b9dc9ad780bceaffb87fc84caa6d9ad97fca63693/schema627f4a4924b27516992b9ccf6fefa6730128560c6d950aca35836d8904c973b4；尚未消费新科学结果。

### 实际工程生成与重复性

`scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py`实际成功，远程 `/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/run/`，control/runner_exit_code=0，PASS_ENGINEERING_ONLY，三行0失败。identity/RMS/identity_repeat真实Wav2Lip GPU渲染，随后原F32 PCM streamcopy MKV，三个原音decodedPCM与92225 sample数全部精确。旧自然音pilot和face只用于工程QC，不能作为fresh方法泛化证据。完整ROOT提取manifest `.codex/scratch/cross_tfg_execution_attempt01/qc04_actual_manifest.json` 为本地重序列化，raw639e5c1667af396bb8b54b91dba768087d7df69b60e50de62f6008d2b96de970不是远程原文件hash。

三次fresh官方SyncNet每条138 query/31lag/同bestlag+2、offset-2。identity Cbest6.3035526276，RMS6.4521274567，repeat6.3208088875。RMS–identity +.1485748291；B+ .4193993、Dmin+ .2708244，C@0反而-.0223703。重复同seed实际仍不同：141帧512x512RGB meanabs .7371293/max27；raw矩阵RMS差.4861594，Cbest差+.0172563。readonly诊断实际raw43f4893f9e64d09a5cd906b2ec122c18be4e8ae0c243c09bd56a4a03e70a12b4。科学dev采用预声明20261003/04/05三配对生成种子，equal aggregate，不因非bitexact排除模型。不同bbox也须追踪；native与固定ROI端点分开。

额外实际发现：生成视频141帧，官方转换评分AVI143帧，crop PCM16取91520 samples。identity/repeat的official cropPCM16 raw91321866746580de045feb8fdcc83f7993aeff9e2ed6282385a64a7aa6be657f相同，但与直接原F32转PCM16前91520样本raw d95aa787d7757b01e2bebc26889a1c8f517429671cc4fc1db645720e0f04a1ba不相等。F32成片原音保持成功，不能把派生官方PCM16称为direct源slice精确；执行方正在查AVI/async/量化路径，训练代理标注不同支持。`26_read_compact_noise_track_model_status.json.response.json` 为完整实际只读来源。

失败01输入SHA漏首字节、02误resolve解释器失venv、03子cwd无temp导致实际forward后未写AVI均完整保留，04只修launcher/runtime不改旧模型源。旧Wav2Lip HEADbac9a81e63ecc153202353372e5724b83d9e6322、SyncNet907c0b579c2e2d83f0eae1b2ac9e720cde4e5623及权重sourcepins实际读回。

### 额外模型与易失状态

MuseTalk/Ditto source下载两轮均成功；实际model02新增pip bootstrap后工具安装成功，但两job都DOWNLOAD_FAILED No module named yaml，weightbytes0，不是模型科学失败。执行方要复用02 pinned source/overlay、补缺失下载依赖继续。MuseTalk0a89dec45a0192b824e3cf4daf96c239440c5ed8/Dittoc3e47eee2e626500017a0556b470d6d4182f85e8。权重目标 `/dev/shm/.../model_deploy_attempt02/`，借用已有torch而不升级旧env。后续LatentSync为候选heldout，不以MuseTalk aux SyncNet权重当LatentSync部署完成。

ROOT+/root/local_recovery+/root/mechanism_audit+/root/cross_tfg_execution 三Luna max运行，旧remote_recovery已中断。SSH入口root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码/API仅runtime。privatePTY50274/env桥接2133易失，ROOT完整读取request/source后dispatcher发起并保存actual.response。不得加入旧巨型授权框架阻塞已授权普通迭代。tmpfs约19.8GB/noexec，根与autodl-tmp紧张；BCloudDrive余1.266TB可承接新数据，无旧源删除/退休。

BM活动Task/Experiment mutable update03精确readback PASS，statusACTIVE/IMPLEMENTED partial，仍早于QC04成功与PCM16差异，下一实质里程碑同步；ADR006 OPEN。旧报告25/concluded、旧Task DONE、88-member便携交付保持。HANDOFF忽略本地文件尚需更新QC04/模型02结果。CodeGraph此前实际不可用，rg fallback不索引。

## Verification

ROOT实际读回synthetic I/O receipt、QC04三真实模型渲染/原F32换轨/新鲜官方评分、identity重复矩阵和RGB诊断、track clock/audio派生差异；dev10 archive manifest和官方roster有独立metadata复核。模型02实际失败缺yaml，训练实际更新和新dev科学收益未运行。未运行pytest/unit tests。本checkpoint前重新Router/实验指令/Coding Session读入、完整58同chat/0legacy搜索及direct BF身份核验、Git分支HEAD/canonicalGitHub/无PR/已推送baseline验证。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local/unpushed HEAD `e9e2dd1c82f4dfb28ec099f93facbb3eace51878`.
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md` and `data-protocol.json`.

## Observations

- [result] 原音F32无损换轨与真实Wav2Lip/SyncNet全链已成功；单旧pilot RMS Cbest+.149，仅工程起点，不是新方法跨模型确认。
- [decision] 用与最终Cbest相关的距离差目标训练，fit校准anchor或平滑bestlag；三配对dev种子量化实际生成波动，originalF32/nativePCM16/proxy分别记录。
- [blocker] 实际head梯度更新、20dev canonical rows冻结、额外TFG依赖与权重、跨模型科学结果待完成；空间可用tmpfs/B盘承接。
- [next_step] 执行真实oldpilot最小梯度与参数更新检查，同时完成fresh20dev行冻结和额外模型下载修复，启动预声明三种子的多模型开发比较并按结果迭代增强头。

## Relations

- continues [[Codex checkpoint - 2026-10-03T03-59-22Z - Fresh speaker protocol and first waveform generation packet]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[25-tts-tfg-mechanism]]