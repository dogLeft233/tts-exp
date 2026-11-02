---
title: Codex checkpoint - 2026-10-07T20-58-33Z - Completed gate interventions and
  verified native policy gap
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t20-58-33-z-completed-gate-interventions-and-verified-native-policy-gap
agent: codex
started: '2026-10-07T20:58:33.9816134+00:00'
branch: codex/cross-tfg-audio-head-20261003
capture: deliberate
repo_root: E:/Documents/tts-exp
git_sha: 7606d6565cf6b608d255e94423045f8123b2f813
status: open
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
username: dogle
repository: dogLeft233/tts-exp
trigger: compact
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T20-58-33Z - Completed gate interventions and verified native policy gap

## Summary
Fit40 四臂门控时序干预已实际完成160条；MuseTalk/Ditto原始数组独立复核通过，Wav2Lip重复差异已定位到MP4之前，并发现人脸检测器重新开启cuDNN benchmark的实际策略缺口。

## Story
原问题是解释中文TTS对TFG的Sync-C增强，用户把目标扩展为多模型自然音频增强头并授权持续实验、重跑和失败修复。增强音频只驱动视频，交付和评分恢复完整原始自然PCM和源时钟。根agent维护理论、协议、BM、私有SSH和独立核验，代码与实验由GPT-6 Luna/max子agent完成。

当前冻结方法是32ms/8ms STFT、四频带静态项和正负谱通量门控相加、逐帧移除公共谱偏移、3dB投影与全局RMS保持。Adam120/3遍已完成，rescore13独立复核Fit40代理ΔCbest=+0.057816875、ΔCanchor=+0.148420191、Δfar=+0.141070795、ΔD0=-0.046865845（12/40变差）；它是冻结神经前向及原生几何裁剪的可微代理，正式编码视频与跨模型效果尚待Dev。权重、配置、ledger和数组的39文件37,137,342B持久快照已逐文件从目标读回PASS；producer05归档4979成员与完整Fit源音频已持久保存，不重复备份或重跑。

为了检验局部时序是否有独立贡献，Fit-only diagnostic02冻结同一步120，对actual、dynamic全部归零的static_only、正负门联合循环移位±64帧（512ms）四臂各40条。根完整审查worker02、spec02、6辅助脚本、config与15请求；执行前修正source_stage_receipt→stage_receipt，并区分rescore13原始manifest SHA与canonical SHA。q0..12实际PASS、q13实际启动、q14实际终止：160/160完整、四worker返回0、stderr空，耗时78.041s，无optimizer、Dev、Confirm或方法选择。原始NPZ尚未下载/数值解释；四份都超过bridge8MiB，Fit agent正准备有全文件pin和offset的有界分块读回。

strict09 MuseTalk/Ditto四条已通过精确重复。根已下载16原始startup、adapter和NPZ并实际运行独立verifier03：全文件SHA、自校验、GPU/python/策略、source输入以及两模型各141×31成对距离矩阵PASS。verifier02曾把Ditto source_face误当face，03仅修正schema并增加路径/SHA验证，未放宽分数容差。独立receipt canonical 13ca6669c1d17403af9fd70d5d6dbc865658ae5d3cb7f451bf7c318a3a453dbe。

Wav2Lip strict10修好隔离cwd/temp后实际两次生成成功，但NONDETERMINISTIC_DIFFERENCE。根下载原始manifest（82763B，5003c9f754b6e7e4be2cc0d0bcb299bca8854c2b7f240b6bf6d64b130685cdee）和qualification（70811B，raw7bae8b1a00c4686316cf5033f01ff139ebe66f62002174feb05d4dd0a7727bb5；canonical1026ba2fae830531bff2f245b1d3005e7a3317427dfc00f96d9e8542004387dc）。每条最终换轨保持自身视频元素流、完整自然PCM与时钟。根实际CPU帧探针：两次中间AVI有15/141帧不同（39–47、102–107），MP4有131/141帧不同（10–140），四视频均141帧/25fps时钟精确。差异在MP4之前已存在，尚需pre-VideoWriter像素确定来自前向还是DIVX编码。

根从实际服务器固定、下载并完整读取face_detection/api.py（2266B，c050676db49d8eb5de19f13c171101892b60b992b2a730cfbc8534967e4faeaa）：CUDA FaceAlignment构造器将torch.backends.cudnn.benchmark改为True，覆盖bootstrap启动收据中的False。这是实测策略缺口；修复后控制重复才能确定其因果作用。dev agent正在准备W2L专用sitecustomize显式加载原bootstrap03后恢复构造后策略并记录前向/写帧证据；usercustomize路线因现有PYTHONNOUSERSITE=1无法加载而被根拒绝，不移除隔离。尚无修复packet或实际干预结果。Muse/Ditto已有资格应按模型分别引用其真实manifest，不强求伪造共同runid。

## Working State
当前Task91 active与Exp90 implemented已实际写入并读回，phase fit120_durable_metadata180_ready_w2l_native_nondeterminism_repair，但Fit四臂完成状态尚待更新。Dev180 metadata03实际完整冻结，20clip/10说话人×3模型×3固定seed；旧baseline仅metadata来源，同当前GPU新生成360 identity/RMS控制和180候选。验收各模型正增益、等模型额外+0.050 over RMS，并检验D0/Canchor/content/visual/GUARD20/全失败分母。Confirm80/40未见说话人与LatentSync封存。Fit诊断不改冻结候选。

当前服务器root@connect.westd.seetacloud.com:18625，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API只保留根私有运行时，不写文件。持久namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003；易失bulk /dev/shm/cross_tfg_audio_head_20261003。私有PTY57588/bridge07/port12249为机器局部连接，提示符可用，四臂已退出后GPU空闲。bridge单文件最多8,388,608B，分块<=6MiB；百度挂载尚不是合格持久备份。

权威指针：`.codex/scratch/root_gate_diag02_status_index_attempt01.json`（q14 response564c457796b9e80963c4a50a9dd1e96ad2ba721dadc46fe073dd1061163cc58fe）；远端job_control/fit_only_gate_timing_diagnostic_attempt02/terminal_receipt_attempt02.json（18638B，9356fa6bd8d418dcc3350b641d1a1a378b80c1b85b26829da62da7998419fef9）。全部四臂原始文件在SHM/fit_only_gate_timing_diagnostic_attempt02/{arm}/，不重跑。

原生核验在`.codex/scratch/root_strict09_native_raw_qc_attempt01/independent_readback_attempt03.json`；W2L原始readback及源码/frame探针在`runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_w2l_retry_attempt01/`。binder02/template05和lineage verifier01仅源已准备，未stage/run；binder根尚未完整阅读，三模型成功资格仍缺W2L。正式路线各模型原生资格→binder→360新控制→180冻结候选→joiner/rawmedia/readback。

## Changed Files
本地未发布HEAD `7606d6565cf6b608d255e94423045f8123b2f813`仅提交CONTEXT更新；CONTEXT93行和被忽略HANDOFF20行已记录持久候选/M-D复核/W2L非确定性，但仍写四臂待运行。现有portable_source_axis/overlay/tests用户修改未动。子agent新实验源未跟踪，scratch/runs忽略，属于机器局部状态。Git短状态另存checkpoint_FJ_git_status.txt，不把其视为发布状态。

## Verification
根实际执行且PASS：M/D16原始文件SHA与startup/adapter/完整矩阵独立QC；W2L10原始manifest/qualification读回；部署FaceAlignment源码固定下载与完整阅读；四视频CPU帧级探针及全部时钟；四臂完整源码/请求/Bash/输入/资源预检、真实启动、160条终端完成。先前Fit120原始代理与39文件持久目标全读回已PASS。未运行通用pytest，未宣称Dev540、W2L修复后重复或四臂数值机制完成。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
本次新HTTP200页面确认仓库、已推commit和当前分支无匹配PR；当前HEAD为本地未发布。

## Observations
- [result] Fit-only四臂160条实际完成，原始NPZ尚待有界分块读回与独立数值复核。
- [result] MuseTalk/Ditto严格原生重复的16原始文件与141×31矩阵独立复核通过。
- [result] Wav2Lip构造器实际重启cuDNN benchmark；中间AVI已有15帧重复差异，不能单凭最终MP4定位神经前向。
- [decision] 保留PYTHONNOUSERSITE=1，修复需实际构造后/前向前策略收据，并保留原成功模型manifest与所有旧失败。
- [decision] 冻结step120、完整原PCM与源时钟、Dev540及Confirm封存；Fit代理和机制诊断均不能替代正式跨模型验收。
- [blocker] W2L修复后实际资格和binder闭包仍待完成，正式Dev540尚未运行。
- [next_step] 对已完成四臂分块读取原始NPZ，核验与rescore13复现及actual/static/移位的完整说话人聚类端点，再继续W2L修复和Dev执行。

## Relations
- continues [[Codex checkpoint - 2026-10-07T20-35-29Z - Durable Fit120 candidate and mixed native qualification with Wav2Lip retry]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[016-source-clock-roi-and-full-delivery-pcm]]
- relates_to [[015-same-gpu-deterministic-controls]]