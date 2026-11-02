---
title: Codex checkpoint - 2026-10-07T20-35-29Z - Durable Fit120 candidate and mixed
  native qualification with Wav2Lip retry
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t20-35-29-z-durable-fit120-candidate-and-mixed-native-qualification-with-wav2-lip-retry
capture: deliberate
status: open
started: '2026-10-07T20:35:29.8555546+00:00'
repo_root: E:/Documents/tts-exp
git_sha: 6a9220efc74f89c7b87ed4235da125ab98504d4b
username: dogle
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
agent: codex
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T20-35-29Z - Durable Fit120 candidate and mixed native qualification with Wav2Lip retry

## Summary
冻结 Adam120 增强头与全部 Fit40 评分已持久备份并重新读取核验；MuseTalk/Ditto 原生严格重复通过，Wav2Lip 在修复独立工作目录后单独重跑，Dev180 元数据已恢复。

## Story
原问题是解释中文 TTS 为什么提高 TFG 的 Sync-C。用户将目标扩展为适配多个 TFG 的自然音频增强头，并授权持续实验和失败修复。增强 PCM 只驱动视频，最终交付与评分恢复完整原始自然 PCM 和源时钟。根 agent 管理理论、协议、Basic Memory、私有连接与独立复核；代码和实验由 GPT-6 Luna/max 子 agent 完成。

当前方法是有界频谱整形：32ms/8ms STFT、四频带静态项及正负谱通量门控相加，逐帧移除公共频谱偏移，限制3dB并保持全局 RMS。Fit40/20说话人的 Adam120/3遍已完成。根独立复算 rescore13 的40条原始评分：ΔCbest=+0.057816875（28/40条、15/20说话人正；Fit99%CI[0.019059472,0.101961554]），ΔCanchor=+0.148420191，Δfar=+0.141070795，ΔD0=-0.046865845（28改善、12变差）。这是冻结 Wav2Lip 神经前向及原生裁剪几何的可微代理端点，正式编码视频的官方评分和跨模型效果仍待 Dev。原生 producer05 归档4979成员、rawdecode04 的40完整PCM/40分析ROI/120NPZ/120AVI/80时钟已独立通过，未改归档或重跑 worker。

严格重复 strict09 实际终止：MuseTalk两条和Ditto两条均 PASS_EXACT_MATCH，Wav2Lip两条输出检查失败，虽然其进程返回0。根下载287148B原始最终manifest、两份qualification、四份Wav2Lip日志和固定推理源。实际原因是相对输出 temp/result.avi 对应的隔离工作目录缺少 temp，属于工程控制失败。新 runner11 仅创建 Wav2Lip childcwd/temp；保持bootstrap03、兼容shim、种子、GPU及确定性策略不变。strict10只重跑Wav2Lip两条，保留六条计划分母和strict09旧失败。根已完整审阅新源diff、9请求、pin和Bash；q0..6实际PASS，q7已detach，PID82709/start_ticks325982873，q8尚未执行。Muse/Ditto保留strict09原生资格，允许各模型拥有自己的精确最终manifest lineage；不能要求它们伪造相同runid。

Dev180 metadata03 已实际完成，20个源依赖闭包全部恢复。18,784,301B旧metadata归档超出bridge8MiB单文件限制，初次上传在传输前失败；根审阅并执行3个不超过6MiB分块、远端整文件重组SHA与后续q4..10，均PASS。冻结计划180条、20clip/10说话人×3模型×3种子，Confirm为空；无Dev音频模型运行或候选选择。计划rawSHA c37a87374f58b5bfa61a85a55027c591a099c50e3d529a90a7ed799c5d4bb00d，rowsSHA c61fc0f5a9e475672e67e22c3ef154343fdfb259fd56a60d1be1e94ef2231291。旧baseline只作metadata来源，正式对照必须同当前GPU新生成360条。

训练11+rescore13的小型持久快照已实际完成：39个payload、37,137,342B，根重新读取全部持久目标字节并核对SHA后PASS。第一次复制因不存在的第二RMS别名在创建目标前失败；新index02只移除该别名，保留真实训练RMS与全体证据，用已审阅原copier01实际执行。只读目标不可重跑覆盖。39文件快照不重复263MB已持久保存的producer归档或已有完整Fit源WAV。

## Working State
Task90 active、Exp89 implemented 仍是当前BM任务/实验快照，Task91/Exp90的新鲜prior已读但尚未更新。Dev冻结为20clip/10说话人×3种子×Wav2Lip/MuseTalk/Ditto=360 identity/RMS对照+180候选；验收各模型正增益、等模型额外+0.050 over RMS，并检查D0/Canchor/content/visual/GUARD20和完整失败分母。Confirm80/40未见说话人与LatentSync封存。Fit诊断不用于更换冻结step120权重。

当前服务器 root@connect.westd.seetacloud.com:18625，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea；密码/API key仅在根私有运行时。持久namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，易失bulk /dev/shm/cross_tfg_audio_head_20261003。私有PTY57588/bridge07/port12249是机器局部连接。最新预检overlay约7.1GB、SHM26.5GB、cgroup空闲58GB，正式Dev输出需容量与路径隔离审核；百度挂载尚未成为合格持久备份。

权威恢复指针：`.codex/scratch/root_repeat10_status_index_attempt01.json`；`.codex/scratch/root_strict09_terminal_readback_attempt01/`；`.codex/scratch/root_fit_durability_dispatch_attempt02/request_index_attempt01.json`；`.codex/scratch/root_metadata03_remaining_index_attempt01.json`。持久快照目标为 namespace 下 `durability_snapshots/fit40_attempt11_final120_rescore13_attempt01`，receipt SHA 1ce69e22c08120943ff0b1dd190447115333c5b77d8b107ec446c38936c06660，根全目标读回response SHA 4bb36be1792369ec9b9b98b5a40d7161ef76568f925ebb2f2a3544d2c364c661。

Muse/Ditto16原始startup/shim/adapter/NPZ下载请求在 `.codex/scratch/root_strict09_native_raw_qc_attempt01/requests_attempt01/request_index_attempt01.json`，已审阅未派发。新的只读verifier02修复原始venv路径与运行时resolved /usr/bin/python3.10误比较，不在Windows解析Linux路径；16来源pin不变，源SHA 550db2f3218d209362a30a1297fa67219f5c52cdc911a516d1c636510913bf22。需要先审阅tinydiff再运行NumPy独立QC。

binder02/template05未stage或run，三模型receipt refs仍为空；子agent准备standalone逐模型原生lineage验证器，根需完整审阅后独立准入。正式路线：各模型严格重复资格→binder02→builder09/controls08实际360→rebinder05/candidate04实际180→joiner04/rawmedia03/readback02。Fit-only4arm诊断worker02/spec已封存但未运行：actual、static_only动态门全部为0、正负门联合±64帧循环移位（512ms），每臂完整40条、同step120、无optimizer/Dev选择，保留边界诊断。GPU当前独占strict10。

## Changed Files
当前HEAD仍为本地未发布 `6a9220efc74f89c7b87ed4235da125ab98504d4b`，CONTEXT93行与被忽略HANDOFF20行尚未更新到本轮实际状态。五个既有portable_source_axis/overlay/tests修改未动。子agent新实验源未跟踪、runs/scratch忽略，属于机器局部而非已发布产物；Git短状态590行另存checkpoint_FI_git_status.txt。本轮新源主要为Wav2Lip runner11/launcher10、只读native verifier02、lineage verifier与Fit gate timing worker02；没有运行通用pytest。

## Verification
实际PASS：固定Adam120与40条原始代理评分/完整PCM；strict09原始manifest及Muse/Ditto两条精确重复；metadata03完整依赖、分块及整体归档SHA、冻结180计划；39文件持久快照及独立全目标字节SHA；strict10源/请求/Bash/资源预检及真实detach。尚未核验strict10终端、Muse/Ditto16本地原始数组、binder/Dev540和Fit gate诊断。已保留缺目录、旧依赖、旧RMS别名和8MiB传输拒绝原始失败，未当作科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
本次新HTTP200页面确认仓库、可解析已推commit与当前分支无匹配PR；当前HEAD为本地未发布。

## Observations
- [result] 固定Fit120代理ΔCbest=+0.058，已独立核验且全权重/数组/配置持久备份39文件37MB。
- [result] strict09 MuseTalk/Ditto原生重复精确一致；Wav2Lip缺temp导致输出失败，strict10单独修复后真实运行。
- [decision] 每模型引用自己精确的成功manifest；保留全部旧失败，共用固定输入、GPU、seed、策略与bootstrap。
- [decision] 冻结Dev540与Confirm封存、完整原PCM/源时钟、step120；Fit代理和门控诊断不能替代正式迁移验收。
- [blocker] 仍需Wav2Lip真实重复资格、各模型原始数组与逐行lineage闭包、容量合格的binder执行。
- [next_step] 读取strict10实际终端并下载Muse/Ditto16原始JSON/NPZ，独立复核后准入binder和正式Dev540。

## Relations
- continues [[Codex checkpoint - 2026-10-07T20-06-15Z - Verified fixed120 Fit proxies and running strict transfer qualification]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[016-source-clock-roi-and-full-delivery-pcm]]
- relates_to [[015-same-gpu-deterministic-controls]]