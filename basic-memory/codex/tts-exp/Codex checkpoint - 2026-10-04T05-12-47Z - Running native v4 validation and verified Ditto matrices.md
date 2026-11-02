---
title: Codex checkpoint - 2026-10-04T05-12-47Z - Running native v4 validation and
  verified Ditto matrices
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t05-12-47-z-running-native-v4-validation-and-verified-ditto-matrices
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
username: dogle
project: tts-exp
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
started: '2026-10-04T05:12:47.2736284+00:00'
status: open
git_sha: 1679f680278a291d5e2a9091d2a2ceac2a41a70c
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T05-12-47Z - Running native v4 validation and verified Ditto matrices

## Summary
冻结 v4 已通过完整音频的公开 CLI/dispatcher 一致性检查，三 frontend 的180行开发验证已实际启动；Ditto原始矩阵的独立数值复算通过，共享360 fit正在执行修复后的独立consumer。

## Story
原任务是从本地记忆、代码、Git和历史数据恢复TTS增强TFG的路线并解释局部声学机制。用户进一步要求持续产出适配多个TFG的自然音频增强头，失败后继续迭代。报告25已完成机制阶段；当前工作是h(x)仅驱动生成，主要评分与播放始终使用原自然PCM及原时钟。root管理理论、冻结协议、BM、Git、私有传输与独立复核；实验实现和运行由gpt-6-luna/max子代理完成。Qwen ASR和数据下载已授权，密码与API key只保存在运行时，不能写入文件或笔记。

方法路线并行比较三frontend共享频谱profile与原生时序训练v4。profile按ADR009完整fit规则选至多三个；v4固定96步最终权重，不从dev选择epoch。开发集20音频/10位男说话人/3paired seeds固定。确认40说话人/80音频、独立嘴部运动guard、新候选ASR、部署CLI、延时及报告26仍未完成。候选有效后，用冻结mask的频带均值、时间包络与交互分解补充局部机制归因，现阶段仅准备协议，不额外渲染消融视频。

## Working State
HEAD `1679f680278a291d5e2a9091d2a2ceac2a41a70c`、分支 `codex/cross-tfg-audio-head-20261003`，本地未发布，无分支PR。新commit只提交CONTEXT.md，93行；HANDOFF.md是ignored本机快照。Task snapshot34与Experiment snapshot33已实际更新并直接读回，部分运行状态由本checkpoint更新。既有机制/tests脏文件和其他untracked源保留；未stage-all，未运行pytest/unittest。

v4最终权重317195B/SHA `3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4`；权威21文件位于 `runs/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_readback_attempt02/artifacts/`，root长度/hash及96步独立数值QC PASS。公开CLI smoke03完整62639样本PASS：CLI/core增强F32 SHA `dbbd64d87b38e64acd0e38f7509696a23057942a6a925686406ac17b535334f1` 完全相同，原自然F32 `f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c` 保持。root receipt `.codex/scratch/v4_dispatch_smoke_actual_receipt_attempt03.json`。旧CLI exit2与错误preflight全部保留，未改变科学权重。

v4 dev180 plan02 SHA `70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c`。完整18源closure/preflight实际PASS，packet03于04:50:36UTC启动launcher828859/starttick766123944、worker828863。远端TASK `NS/stageb_v4_dev_attempt02`、OUT `SHM/stageb_v4_dev_attempt02/run`。最新实际readback为6/180 PASS、三模型各2、0fail、stderr空，仍RUNNING；尚未读取效应。请求/回执在 `.codex/scratch/cross_tfg_audio_head_20261003/stageb_v4_dev180_attempt01/requests_attempt03/` 与 `requests_attempt04/`。必须dispatcher→core→adapter；仅candidate的180行不能直接给ASR guard，应按精确自然PCM/seed/模型/肖像/时钟与完成的baseline行合并单frontend。新候选Qwen API尚未执行，旧v3 CER0不得替代。

共享fit已COMPLETE360/360 PASS，ledger SHA `e67c753a2c29048757f629d18ac259b9f146f980cfaf94959d17048874fb0e0d`；空repeat对应冻结generation_repeat_qc=[]，不是重复QC PASS。consumer01实际失败于历史源PCM16而其只接受F32。consumer06是窄解码修复：mono16k PCM16按/32768转F32，验证文件hash/canonicalPCM/count与冻结subtype；所有换轨、原音、时钟、矩阵检查不变。source152356B/SHA `25aabe0e466ae5a89f8b1948f91bc84b83079acb2499a7ea9ff926f45e872ec9`，root完整diff审阅。fresh repair02五请求实际PASS，8源PCM16 preflight PASS，consumer PID833782/starttick766138731，latest约5分钟alive/nooutput。OUT `SHM/shared_frequency_fit_consumer_repair_attempt02/consumer_attempt01/output`。selector adapter与raw360导出已准备，等待终态后实际执行；fit-only NO_EVALUATION_ENDPOINTS/exit2须与所有行审计结果分别保留。

Ditto240开发行全部PASS，consumer03全部六个FULL endpoints COMPLETE；GUARD20仅固定短音频不足41queries而INCOMPLETE，不缩分母。FULL开发描述：RMS−identity +0.168718，local−RMS −0.028273，v3−RMS −0.170123且10/10speaker负。raw241 archive在 `runs/cross_tfg_audio_head_20261003/ditto_recovery_attempt05_raw_matrix_readback01/`，2587997B/SHA `1f250930fe709e36fa4660327e0fbf9d01bdbf5a672612743bf9be462aa26f16`。root新增独立QC实际PASS：241NPZ/1687arrays、60paired groups、源帧/有理时间戳/track映射、共同支持mask与FULL/GUARD端点。NumPy重算最大curve误差4.768372e-6、speaker2.861023e-6、endpoint3.814697e-7，小于3e-5。receipt为该目录 `root_independent_numeric_qc_attempt01.json`，不是确认集收益结论。

LatentSync官方identity03已真实原生渲染PASS，5本地媒体与原音FFmpeg解码PCM rootPASS；本机目录 `.codex/scratch/latentsync_fit_inference_attempt03/readback_attempt01/`。完整历史fit62639样本/99frames/25fps，无retime，播放原PCM。固定RMS转换已实际PASS：canonical生成PCM `f024d9fb22f5ace5bd950bb5b514648d2ab896273cb260973bac0f53335ce172`，targetRMS0.0376838172675746，峰限未激活。RMS04失败于VAE receipt文件名，发生在CUDA/生成之前，日志保留。fresh05仅namespace及receipt文件名窄修复；packet `.codex/scratch/latentsync_fit_rms_inference_attempt05/packet_attempt04/packet_index_attempt04.json` 全9实际PASS，launcher90493/starttick766190787、worker90495，initial RUNNING。packet05仅printf真实换行修订，未执行，不能重复launch。工程pair scorer已准备，须绑定实际RMS05报告后执行；未打开heldout/candidate选择。

SadTalker05实际应用pinned np.float=builtin float兼容修复，两臂landmark检测均成功，但推理失败于preprocess.py中 `np.array([w0,h0,s,t[0],t[1]])` 的size-one数组/标量混合。实际stderr已读回，两臂无视频评分。agent正在准备仅此构造的size-one scalar兼容修复，要求原source/function hash固定、等价数值、独立namespace；禁止全局np.array patch。工程失败不是科学阴性。

本机fragile transport为私有PTY50274/bridge2133及内存凭据；服务器 `root@connect.weste.seetacloud.com:12349`，公钥pin已有实际回执。NS=`/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`；SHM=`/dev/shm/cross_tfg_audio_head_20261003`。单文件传输上限8MiB，bulk分片≤6MiB；GPU两次5秒稳定空闲准入，不终止同行。需新空间读数，旧sealed产物可导出/网盘归档后再谨慎腾空间，不能删除active权重或原研究数据。

## Verification
已实际运行的root检查包括v4权重/96步数值、CLI完整波形字节等价、Ditto原始241矩阵独立重算、Latent5媒体/hash/原PCM解码；BM Task34/Experiment33写入与全文readback PASS。所有运行判断以response/terminal产物为证据，prepared和launch不写为成功。新candidate ASR/完整v4 dev/shared selector/跨模型确认仍待完成。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified remote baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 真实协议路径：`openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md`、`transfer-iteration.md`。

## Observations
- [result] 冻结v4公开CLI与实际dispatcher在完整62639样本上F32完全一致，开发180行实际运行并已6PASS。
- [result] Ditto原始矩阵与全部FULL开发端点经root独立复算通过，v3未获得跨frontend的可用收益。
- [decision] 保持原自然PCM、源时钟、固定支持集/分母、失败行与确认封存；共享候选须同时过跨模型效果、内容和运动guard。
- [blocker] shared360 consumer终态/selector、v4完整dev/new ASR、Sad标量兼容、LatentRMS/scorer、确认和交付仍需实际完成。
- [next_step] 读取sharedconsumer06与v4dev最新compact状态，完成shared360原矩阵QC和冻结selector，同时接上候选ASR及Sad/Latent工程pair。

## Relations
- continues [[Codex checkpoint - 2026-10-04T04-32-01Z - Native model pass and complete shared fitting]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]