---
title: Codex checkpoint - 2026-10-07T19-06-14Z - Materialized fixed120 candidate and
  continuing cross model execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t19-06-14-z-materialized-fixed120-candidate-and-continuing-cross-model-execution
agent: codex
status: open
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
git_sha: cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9
cwd: E:/Documents/tts-exp
started: '2026-10-07T19:06:14.1980920+00:00'
username: dogle
model: gpt-6.1-sol
project: tts-exp
trigger: compact
repo_root: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T19-06-14Z - Materialized fixed120 candidate and continuing cross model execution

## Summary
Fit40 的固定 step120 权重已经生成可运行的增强音频；当前继续完成原音换轨核验与 Wav2Lip、MuseTalk、Ditto 的实际迁移评分。

## Story
本研究最初追查 TTS 为什么提高中文 TFG 的 Sync-C，用户随后要求把局部声学机制转成通用自然音频增强头：增强 PCM 只驱动视频，成片换回精确的原自然 PCM 与原时间轴。用户明确要求持续推进失败分析与方法迭代，允许重跑、下载和云 ASR/TTS；根 agent 负责理论与冻结协议，代码与实验由 GPT-6 Luna/max 子 agent 完成。

当前方法以 32ms/8ms STFT、四频带静态项与正负谱通量门控的 3×3 参数构造有界谱掩码，逐帧去公共谱偏移、整体 RMS 归一化，并保持采样时钟与样本数。实际 Fit40/20说话人 Adam120/3遍训练已结束，根 agent 逐步重算完整120步账本与终端权重。三个参数向量相加，不是互斥的三种方法；当前数值倾向在谱攻击/衰减处增强相对高频，尚需与 RMS 比较迁移收益。

固定末步候选的五个原始 metadata 文件已下载原始字节并独立核对 SHA；7模块 CLI 已部署。单条 Fit40 音频 BAC009S0052W0121 已实际用该候选运行并通过换轨策略、样本数、CPU/no CUDA 和 float WAV 读回核验；输出 WAV 250600B，SHA 9cf46db4c5caf7f058bda52b290ab5a79405fddf97ad541bfd9dc72624b6bfa0。这是候选可执行性结果，没有 Dev/Confirm 评分或方法选择。

固定120步重评分的09/10/11分别在 UUID 表示、工厂绝对路径、adapter赋值顺序处失败，均0/40评分行；训练本身未重做。修复只规范同一 GPU 的 UUID、用确切工厂源码身份替代迁移路径相等、把同一 adapter 对象绑定提前。子 agent 已准备新 attempt12 revision05，根 agent 尚未部署，禁止使用早期误改训练路径的 revisions02–04。

MuseTalk 9资产和6别名、Ditto12资产实际恢复并由根 agent 验证。Wav2Lip+Ditto 严格重复性 attempt05 已实际 detached 启动，PID54567/start_ticks325404561；前一个 attempt04 使用 resolve 后系统 Python 导致 numpy缺失，0行失败。新版本保留 venv 的 argv[0]，实际预检已核对 Python prefix/NumPy/Torch 和 GPU。Muse02 修复源码已部署，待 GPU 空闲后预检、启动，再合并全部6行重复性分母。

原生产归档全4979文件/287367187逻辑字节与 tar263629735B 已根 agent 全量核验通过；原音解码 rawdecode03 又确认全库存一致，却在第一条 selected-track MKV 的音频样本总数上失败。该行 full source62639样本，选中97帧理论切片62080样本；需要实际单文件 PCM 解码验证差异，不能凭理论放宽成片精确原音门槛。自然轨的全源PCM已通过独立 tiny probe，训练评分 provenance 使用 full natural waveform。原归档和失败记录均保留。

## Working State
持久状态以 [[开发跨 TFG 的自然音频增强头]] 和 [[跨 TFG 自然音频增强头的训练与原音换轨验证]] 的实际读回为准，当前 Task88 active/Exp87 implemented；还没有新的跨模型效应结论。冻结 Dev20/10说话人×3 seeds×3模型，360 identity/RMS控件与180候选；要求各模型正增益、等模型额外 +0.050 over RMS，并保留 D0/Canchor/content/visual/GUARD20/失败分母。Confirm80/40未见说话人与 LatentSync仍封存。

当前服务器是 root@connect.westd.seetacloud.com:18625，已记入 basic-memory/docs/reference/server-access.md；密码与 API key 只在根 agent 私有运行时变量，永不写入文件。GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。训练 namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，临时产物 /dev/shm/cross_tfg_audio_head_20261003，避免188MB剩余的 autodl-tmp。

机器局部易失状态：私有 PTY57588 的 bridge07/port12249 已可用；GPU优先给实际 repeat05，随后固定评分12或 Muse02。E盘约1GB空闲，只下小 metadata/NPZ，不复制263MB归档。忽略或未跟踪的 .codex/scratch 与 runs 保存原请求和响应；不能把它们当已发布成果。

## Changed Files
本轮 CONTEXT.md 为未提交修改（93行），HANDOFF.md 为被忽略的当前快照；二者尚需更新为11 adapter失败、repeat05实际运行与候选CLI通过。当前 HEAD cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9 为本地未发布提交。五个既有 portable_source_axis/overlay/tests 修改未动。

继续用 .codex/scratch/root_dispatch_reviewed_packet_attempt02.ps1 核对 pins/bash 并从私有 PTY 发请求。固定评分新包为 .codex/scratch/fit120_final_rescore_packet_attempt12_revision01/packet_revision05/；重复状态已准备 .codex/scratch/root_repeat05_status_index_attempt02.json；单文件 mux 诊断为 .codex/scratch/fit40_native_rms_geometry_attempt05_mux_probe_attempt01/00_readonly_selected_track_mux_diagnostic_attempt01_request.json。候选原始 metadata 在 .codex/scratch/root_candidate_original_metadata_readback_attempt01/，实际 CLI 响应在 runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/fit40_candidate_cli_smoke_attempt01/requests_attempt01/。

## Verification
实际通过：Fit40完整120更新链与终端theta、9项 Muse 权重、Ditto运行源码和header closure（历史安装deb已缺但非运行依赖）、候选原始metadata五文件全SHA、单样本候选CLI和原样本数读回。实际失败：固定评分09/10/11零行；repeat04 Python环境零行；rawdecode03 第一条 selected-track 精确PCM样本数。repeat05仍待实际状态读取，12与Muse02尚未启动。没有运行通用 pytest，没有 Dev/Confirm生成、评分、方法选择或报告26。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
当前分支 codex/cross-tfg-audio-head-20261003 无匹配 PR（新HTTP200页面查询）；当前本地HEAD未发布。

## Observations
- [result] 固定 step120 候选已在一条 Fit40 音频实际产生增强WAV并通过原始读回，尚未测得迁移收益。
- [decision] 保持冻结末步、原音PCM、原时间轴、RMS对照与完整失败分母；工程错误不转写为科学阴性。
- [decision] 子 agent 负责代码实验；根 agent 保持私有 SSH、理论、协议与独立核验。保留失败版本，只在新 attempt修复。
- [blocker] selected-track历史mux与full source长度不同，待实际单文件解码；固定评分adapter赋值修复待运行，三模型精确重复尚未齐备。
- [next_step] 读取 repeat05 实际状态，接着部署最小12修复和执行单文件mux诊断，进入完整Dev540生成与原音评分。

## Relations
- continues [[Codex checkpoint - 2026-10-07T18-29-08Z - Running fixed120 rescore with restored model assets]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]