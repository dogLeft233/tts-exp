---
title: Codex checkpoint - 2026-10-08T09-27-50Z - Committed usable candidate and running
  cross-model Dev
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t09-27-50-z-committed-usable-candidate-and-running-cross-model-dev
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
project: tts-exp
trigger: compact
model: gpt-6.1-sol
started: '2026-10-08T09:27:50.6454699+00:00'
branch: codex/cross-tfg-audio-head-20261003
capture: deliberate
git_sha: 52d146bd925a7773238065ed11bf7afb85e5b76f
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
username: dogle
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T09-27-50Z - Committed usable candidate and running cross-model Dev

## Summary
冻结增强头已交付可复现离线包并提交实际证明；三模型 Dev 对照仍在运行，最新只读快照为 129/360 全部 PASS。

## Story
用户最初要求恢复 TTS 增强 TFG 的历史研究路线，随后将目标扩展为适配多个 TFG 模型的自然音频增强头，并要求持续推进到完整方法。增强后的 PCM 驱动视频，最终恢复完整原自然 F32 PCM 与原始时钟。ROOT 负责机制推理、冻结协议、独立核验、Git 与 BM；实现和实验由 GPT-6 Luna/max 子代理负责。服务器迁移、重跑、Qwen API、下载和网盘均已授权，凭据仅走私有运行时。

当前候选保持 Fit40/20 说话人、Adam120/3 passes 冻结。频谱与校准解释主要 Fit Cbest 收益，正确时序门控另有固定时钟 D0 收益；这些 Fit 结果是机制线索。新 Dev 采用 20 句/10 说话人、三 seeds 和 Wav2Lip/MuseTalk/Ditto，先完成 360 identity/RMS 对照，再生成 180 候选。预设要求三模型方向为正、等权模型平均相对 RMS 至少 +0.050，并检查 D0/Canchor、实际生成输入的内容、原音换轨、视觉与 GUARD20。Confirm80/40 未见说话人及 LatentSync 保持封存。

本轮完成先前实际实验的本地逐项复核：离线包隔离 CPU CLI 输出的 RIFF、全 F32 PCM、长度、参数与来源由 ROOT 和独立子代理确认；旧失败 controls17 的紧凑快照全部 1,968 物理文件读回与来源索引、aliases、receipt 一一对应。已将 17 份实际证明/包、README、CONTEXT 与 .gitattributes 共 20 文件提交到本地 HEAD52d。可用 CPU 工具已成立，候选的跨模型选择资格仍待完整 Dev。

修复后 controls18/attempt10 的六个模型×方法单元均产生成功结果。ROOT 独立核验快照04的 89 行和 observer 首次 112 行；随后读取 observer 第二次快照，2026-10-08 09:12:10 UTC 为 129 行全 PASS、PID 身份匹配、完整 ledger 计数、无 result manifest。第二次快照尚未另做 ROOT 深层 raw/self 复核，不据局部行数宣布增强效应。只读 observer03 每 900 秒检查一次并在终态停止等待 ROOT 消费。

候选 180 物化计划缺少当前服务器绑定引用，ROOT 完整审阅了仅处理元数据的 linker worker 和九项请求。worker 建立带来源链的 reference-only 输入，再交给冻结 rebinder12；不打开音频或触碰 GPU。ROOT 在执行前发现 q00 的 shell 字面反斜杠引号错误，已让 Dev 保留未执行 attempt01，并制作新 request/index02。旧失败实验完整媒体 complement 归档 packet02 由子代理制作，加入 tar/chunk 核验后全 26,328 源文件重哈希和全 stage 树复查，尚未 ROOT 全文审阅或执行。ASR 子代理正在准备小型静态 ZIP 与安全 stdin 客户端；拒绝未收到的虚构 terminal schema，最终必须读取真实候选终态。

## Working State
仓库 E:/Documents/tts-exp；分支 codex/cross-tfg-audio-head-20261003；当前 HEAD `52d146bd925a7773238065ed11bf7afb85e5b76f` 为本地未推送提交。Git index 无待提交内容。五个不相关已跟踪改动与大量旧 untracked/scratch 保留。Task108 和 Experiment107 已写回并直接读回，execution phase 为 dev_controls360_repair_attempt10_running。

服务器 root@connect.westd.seetacloud.com:18625；主机密钥 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码与 API key 不入文件或 BM。远端根 /root/tts-research-system-20261001；私有 PTY57588、bridge07 port12249、Windows observer PID87568 为易失状态。恢复时先检查真实身份和已有回包，禁止重放有响应的请求。

GPU run_id cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10；PID344830/start_ticks329924069；binding09 raw SHA dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5；plan09 SHA 8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb。实际物理目录 /dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360，候选目录应为同一父目录下的 candidate180。新 receipt.output_root 指 controls 物理叶，归档 consumer 不可套用旧版本父目录语义。旧 preflight 已过期，后续候选真实启动前需重新测容量。

本地 E 盘仅约 653 MB 空闲，完整媒体必须用 F 盘暂存再 create-only 复制到 B 百度 CloudFS，并做整文件 SHA 读回；挂载写入不等于网盘服务端同步确认。旧紧凑快照位于远端 durability_snapshots/controls360_attempt17_failedterminal_compact_attempt02；视频/JPEG complement 尚未归档，不能删除 SHM。

## Changed Files
- openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/README_fit_step120_candidate.md 与 v3_step120_fit_candidate_offline_bundle_attempt02.zip：已提交实际 CPU 使用说明和 44,326 B 离线包，SHA 32145ea20d18602d2fda2b66ca6fcc7256c215ad9fd70ca23da5a91534a33700，NOT_SELECTED。
- 同目录 v3_step120_actual_cli_output_root_qc_attempt01.json、独立 QC、CLI metadata/response 和 dev360_failed_controls17_compact_*：已提交实际输出与快照复核证明。
- .codex/scratch/dev360_status_observer_attempt03/：冻结只读 observer 源和所有逐次请求/回包，机器本地证据。最新 polls/20261008T091210Z_00044a1a/response.json raw SHA 123f0bc8998b025858fd34c2c053189b783542ec713afa19eaa4216405022efb。
- .codex/scratch/dev360_candidate180_cpu_prepare_attempt01/ 与 dev360_excluded_media_archive_attempt02/：仅准备的不可变 packet，未执行；下一请求版本由代理交付。

## Verification
隔离 CLI 的实际输出 250,600 B、62,639 samples/16k/mono/F32，全 PCM SHA ef72ce37b21878436a7f38888bff04a71446d71a4d2e7ebf85bb84fec6cd2f6c；所有样本有限、长度与配置/来源匹配，约 2.389 秒，整 WAV 与既有实现一致。ROOT 和独立子代理均实际核验。紧凑快照全 1,968 文件实际 raw SHA 与来源映射匹配，1,966 数据对象及完整旧 360 失败分母保留；这不替代未重新计算的 NPZ 数值 QC。提交前 17 工件/129,980 B Git blob 与工作区字节一致；20 提交文件非 ZIP 的凭据模式扫描通过。ROOT 已实际复核新运行的 CPU09、启动 receipt、快照04/首次 observer，第二次 129 计数刚读取。没有新 pytest、环境安装、push 或 PR。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮直接只读确认 canonical URL；当前分支 PR 查询无匹配项。
- 当前 HEAD `52d146bd925a7773238065ed11bf7afb85e5b76f` 为本地未推送；不构造远端提交链接。
- openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md：跨模型失败路线与冻结资格协议。

## Observations
- [result] 冻结 Fit 增强头离线包已通过实际隔离 CPU CLI 和全 PCM 独立复核，并连同真实运行证明提交到本地 HEAD52d。
- [result] 修复后的三模型对照实际运行至只读快照 129/360 全 PASS；完整跨模型增强效应尚待全部实验与资格检查。
- [decision] 维持冻结参数和三模型+RMS资格协议，完整保留失败分母；增强只驱动生成，交付恢复原完整 F32 PCM 与时钟。
- [decision] 仅元数据 candidate linker 可与 controls GPU 并行，角色必须标记 reference-only/nonqualification；候选运行需真实终态和新容量检查。
- [blocker] 360 对照、180 候选、实际 generator PCM 内容守卫、原音换轨、540 视觉/GUARD20及完整媒体归档仍需完成。
- [next_step] 审阅新 q00 修复 packet 并执行候选 CPU 计划准备，同时推进旧媒体归档/ASR 静态部署；完成 controls360 后顺序执行 candidate180 与所有内容、时钟、视觉和完整分母统计资格。

## Relations
- continues [[Codex checkpoint - 2026-10-08T08-35-52Z - Sealed failed-run evidence and actual offline CLI]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]