---
title: Codex checkpoint - 2026-10-03T03-59-22Z - Fresh speaker protocol and first
  waveform generation packet
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t03-59-22-z-fresh-speaker-protocol-and-first-waveform-generation-packet
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
agent: codex
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
capture: deliberate
repo_root: E:/Documents/tts-exp
trigger: compact
username: dogle
git_sha: 368c449548e24abfa803f4deca057424aeba03b0
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
project: tts-exp
started: '2026-10-03T03:59:22.5715934+00:00'
status: open
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T03-59-22Z - Fresh speaker protocol and first waveform generation packet

## Summary

跨TFG自然音增强头已形成可执行规格、明确的新说话人划分与波形实现；首次真实Wav2Lip生成/原音换轨/评分packet已提交服务器，尚无新跨模型效应结论。

## Story

原任务是恢复并解释TTS增强TFG机制，报告25与可恢复交付已完成。用户进一步要求产出适配较多TFG的自然音增强头，增强音仅驱动视频、成片换回原自然音，失败后持续迭代到完整可用方法。新create_goal ACTIVE，无预算上限。ROOT维护理论、规格、BM和Git；代码与实验由gpt-6-luna/max代理完成。旧RMS自然音固定评价正向结果提供起点，不能当作新方法的跨模型泛化证据。

路线是canonical16k mono float32等长波形，identity/RMS/局部包络/平滑有界STFT幅度残差，独立开关gain/envelope/frequency。先实际identity/RMS生成及原PCM换轨，随后fit/dev选择波形头，再新说话人与未优化TFG确认。优化目标必须服务于原自然音评分；D0下降不是Cbest增加，需观察31lag曲线并避免训练代理数值与真实成片混淆。

## Working State

权威规格 `openspec/changes/develop-cross-tfg-audio-head-20261003/{spec.md,tasks.md,data-protocol.json}`，新分支 `codex/cross-tfg-audio-head-20261003`。ROOT实际提交规格/入口e8a5d1a2aa3c090180ef655d6d75d8e93b103173、历史fit与PCM规则99b853ee01bac2b17a348c4c244582daf28fe97f、数据roster368c449548e24abfa803f4deca057424aeba03b0；全部local/unpushed/noPR。CONTEXT93行。BM活动Task/Experiment已实际更新IMPLEMENTED partial，receipt `.codex/scratch/cross_tfg_enhancement_bm_update_receipt_attempt02.json`；ADR006保持OPEN；最新roster写在Git协议，BM正文后续同步。

官方AISHELL/AISHELL-1 revision bbe295d530192a4cd41644b711c9aecd087df653实际REST库存100speaker packs，3.709GB。speaker.info实际核验fresh S0002–S0051全部男性。历史fit S0052–S0101每sp16条；fresh dev10sp/20clip为S0005,S0022,S0024,S0037,S0038,S0039,S0041,S0044,S0049,S0050；其余fresh40sp/80clip为confirm。按固定hash选择dev、lex选择3.2–8sec有限16kmono WAV，各2条；不按分数替换缺失。历史已暴露86sp从confirm排除。data-protocol raw cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87，ROSTER_FROZEN_ROWS_PENDING；实际PCM行、模型角色、方法/脸/seed与统计族仍待效应前冻结。主确认范围为中文男声，不能泛化到女声/其他语言。

`scripts/audio_enhancement/`实际wave I/O、identity/RMS、bounded谱形/局部包络、identity-init卷积STFT残差、CLI/PCM streamcopy mux、训练契约和Wav2Lip differentiable adapter已存在、仍编辑未提交。ROOT审查修正mask叠加±18dB误差为总±6dB约束、需要post-head RMS匹配独立增益；前端不做.9 peaknorm，STFT constant padding，scipy预加重float64精度需实际parity。adapter不得要求音频帧数等于Wav2Lip视频帧数（16mel尾窗使视频少约4帧），需保留全原音/官方min支持；可微浮点视频只是训练代理，不能宣称DIVX/JPEG实际视频数值同一。local_recovery负责这些修正及实际gradient/parameter-update QC、dev数据下载；本机Python无torch，代理报告synthetic I/O PASS但ROOT尚未收到独立receipt。

`scripts/audio_enhancement_eval/`独立consumer/schema实际存在，mechanism_audit修正union-speaker axis、reduced rational timestamps、dev-only空确认族、degenerate bootstrap。已回复采用float32 mean/median/subtraction主字段，float64仅诊断；best_lag=k、官方offset=-k。FULL共同query支持/guard20敏感性、31lag矩阵、等权speaker/model及有限95%联合族、失败/短片分母保留。尚未在新实际视频产物上运行。

远程source/inventory Stage0/0b/0c实际PASS：RTX4080SUPER/CUDA/两环境torch2.5.1cu124；只有Wav2Lip/SyncNet实际部署。根盘余1.397GB、autodl-tmp余.382GB，/dev/shm余19.835GB且noexec，可用于纯代码/权重/音视频，不放需要执行映射的.so。BCloudDrive本地余1.266TB。/autodl-pub为只读NFS。新代码namespace `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/`、新结果/权重 `/dev/shm/cross_tfg_audio_head_20261003/`，不改旧env/source；额外MuseTalk/Ditto/LatentSync部署尚未完成。

执行owner换为fresh `/root/cross_tfg_execution`（Luna max），旧remote_recovery已中断。新 `scripts/experiments/cross_tfg_audio_head_20261003/engineering_qc_oldpilot.py` SHA da47c2e6eea05ae04343b50b0bc4f386b8d94f1c3bb10a7e2330bc8c6abdc56f，三请求在 `.codex/scratch/cross_tfg_execution_attempt01/`。ROOT完整读取并通过privatePTY50274按顺序提交mkdir/upload/exec，首次实际运行结果待readback。旧natural_1和primary_face raw SHA固定，engineering_only，不参与候选选择；三次identity/RMS/identity-repeat实际GPU生成后原PCM_F32LE streamcopy mux核验，再official fresh SyncNet tracking/scoring/31lag NPZ。child明确同seed20261003；官方库mel positional→keyword兼容仅新wrapper。QC不能代替fresh科学确认。

### 持久与机器本地状态

旧报告 `basic-memory/docs/experiments/25-tts-tfg-mechanism.md` concluded与旧Task DONE、冻结88-member便携包保持。工作树既有legacy/tests修改及untracked agents文件保留，未stageall。BM/runs/scratch/HANDOFF为本机状态；privatePTY50274/SSHbridge2133/tokenenv易失，ROOT派发request并读取实际response，不以启动当完成。

服务器root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码/API仅runtime，不写文件/argv/BM。CodeGraph此前实际无可用索引，继续rg fallback。BM typed --content argv避免历史Windows stdin Unicode失败。

## Verification

实际Git/官方数据REST、speaker.info及三条scoped提交核验；BM mutable update02精确readback PASS。实际Stage0c CUDA PASS、Wav2Lip librosa positional失败已定位，fresh wrapper修复已提交执行。ROOT完整审查QC runner/三请求并发送；新生成/换轨/评分结果尚未返回，额外模型/训练尚未执行。未运行pytest/unit tests。写本checkpoint前Router/实验指令/Coding Session加载，完整57同chat/0legacy检索及direct前checkpoint身份核验，GitHubcanonical/无PR/pushedbaseline只读核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local/unpushed HEAD `368c449548e24abfa803f4deca057424aeba03b0`.
- `openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md` and `data-protocol.json`.

## Observations

- [result] 新目标ACTIVE；waveform实现和独立评估consumer已存在，首次真实Wav2Lip QC packet已提交，尚无新跨模型收益结果。
- [decision] 历史暴露数据可明确用于fit/dev；fresh S0002–S0051男性按10/40 speaker隔离dev/confirm，行哈希及完整统计族在科学效应前冻结。
- [decision] float32官方Score主值与float64诊断、lag k与官方offset -k、可微proxy与实际视频分别记录。
- [blocker] 新PCM输入行、额外TFG部署和实际训练/多模型收益待完成；磁盘紧张但可用19.8GB tmpfs及百度盘承接新产物。
- [next_step] 读取首个Wav2Lip实际生成/原PCM换轨/评分QC结果，修复任何真实失败，同时启动MuseTalk/Ditto部署与fresh dev音频下载，进入多模型identity/RMS实际基线。

## Relations

- continues [[Codex checkpoint - 2026-10-03T02-46-49Z - Cross TFG waveform head objective and implementation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[25-tts-tfg-mechanism]]