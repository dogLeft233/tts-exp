---
title: Codex checkpoint - 2026-10-08T10-02-35Z - Actual candidate inputs and running
  three-model Dev
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t10-02-35-z-actual-candidate-inputs-and-running-three-model-dev
status: open
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repository: dogLeft233/tts-exp
started: '2026-10-08T10:02:35.9072855+00:00'
repo_root: E:/Documents/tts-exp
project: tts-exp
cwd: E:/Documents/tts-exp
username: dogle
trigger: compact
agent: codex
git_sha: 52d146bd925a7773238065ed11bf7afb85e5b76f
capture: deliberate
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T10-02-35Z - Actual candidate inputs and running three-model Dev

## Summary
冻结候选的180行当前服务器计划和32项ASR静态资料已实际准备并双重复核；三模型对照继续运行，最新只读观察178/360全PASS，旧媒体归档容量问题已有明确修复路线。

## Story
用户最初要求恢复TTS增强TFG的历史研究，随后将目标扩展为可用于多个TFG模型的自然音频增强头，并授权持续实验、重跑、下载和Qwen API。增强PCM驱动生成，成片恢复完整原自然F32 PCM和原时钟。ROOT维护机制、冻结协议、独立复核、Git及BM；实现与实验由GPT-6 Luna/max子代理承担。

冻结候选仍为Fit40/20人、Adam120/3 passes。既有Fit机制报告显示主要Cbest收益来自频谱整形，正确局部门控相对错位额外改善固定时钟D0约0.010。新Dev采用20句/10人×3 seeds×Wav2Lip/MuseTalk/Ditto，先360 identity/RMS对照，再180候选。三模型正向、等模型平均over RMS至少+0.050及D0/Canchor、实际生成输入内容、全原音时钟换轨、视觉和GUARD20验收不变；Confirm80/40未见说话人及LatentSync封存。

本轮实际执行只处理元数据的candidate linker/rebinder九项请求，生成180条当前服务器绑定候选和360个不同对照引用，保留冻结权重/config及三前端语义；ROOT和Dev独立核对全部来源、顺序、20句/10人/3种子、每模型60行、Wav2Lip帧映射，均通过。该准备不打开音频，不读取对照终态，不生成媒体，不构成候选运行资格。已执行index03保持不可变。

另实际执行ASR静态部署六项请求，远端新static_project包含32个已核对ZIP成员及20个自然音转写。ROOT逐成员全SHA与实际summary核对，Signal独立17项检查通过。尚无API调用；私有客户端正在按真实candidate结果、manifest、execution、run_status设计，拒绝虚构终态schema，必须严格180行PASS、子进程rc0、完整确定性证明才能启动。

旧失败controls17完整媒体archive03真实启动后在容量预检退出：持久磁盘6,893,547,520 B可用，需7,114,081,320 B，差220,533,800 B；没有tar、chunks或成功receipt。原始launch/failure/state通过base64独立读回与全SHA核验并保留。新的archive04方案在持久盘仅存一份完整tar、在独立SHM目录生成可重建传输分块，分别保留5GiB余量，不删除原数据或影响当前GPU。实现正在准备，尚未ROOT全文审阅或执行。

## Working State
仓库E:/Documents/tts-exp；分支codex/cross-tfg-audio-head-20261003；本地未推送HEAD `52d146bd925a7773238065ed11bf7afb85e5b76f`，Git index尚无新暂存。离线候选包及前17份实际证明已提交；本轮新8份实际proof共49,313 B已create-only复制到reproducibility目录，尚未提交，见parallel_metadata_proofs_persisted_manifest_attempt01.json。五个不相关已跟踪改动和大量旧untracked保留。BM Task109/Experiment108已更新并直接读回，仍active/implemented。

当前服务器root@connect.westd.seetacloud.com:18625；主机密钥SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea；远端根/root/tts-research-system-20261001。密码/API key仅运行时，永不入文件或记忆。私有PTY57588、bridge07 port12249、Windows observer PID87568为机器本地易失状态。

controls360 attempt10 PID344830/start_ticks329924069；binding09 raw dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5，plan09 raw 8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb。物理叶/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360；candidate180应为同父目录兄弟。新launch receipt.output_root为物理controls叶，binding.output_root为父namespace，后续归档必须分别处理。observer03每900秒只读；09:42:11的161行已ROOT深层raw/self/PID/fullcounts复核，09:57:12的178行快照刚直接读取，下一步复核并消费后续快照。尚无结果manifest/新跨模型效应。

E盘低空间，不存媒体；F盘暂存→B百度CloudFS create-only复制和全SHA读回。挂载写入不等于云端同步。旧compact1968文件已持久保存并双重复核，完整媒体尚待archive04。源SHM不可删除。候选13 CPU metadata preflight packet02已准备未派发，GPU launch须真实对照终态、候选预检和新容量证据。

## Changed Files
- openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/candidate180_actual_*：五份候选绑定/ASR实际summary与独立QC，以及ROOT合并QC，工作区未跟踪。
- 同目录dev360_actual_observer_poll04_root_qc_attempt01.json及两份media_archive_attempt03 launch/failure receipt，未跟踪。
- .codex/scratch/dev360_candidate180_cpu_prepare_attempt01/downloads_attempt02/：六个真实下载，rebound plan raw 5a01e0ab1d6d4545ba4f6de332aced0ee1920f2001e51f5f552684d8b9926038，机器本地证据。
- .codex/scratch/dev360_step120_asr_private_launch_attempt01/：32成员静态包、真实回包/summary和独立QC；private client仅准备。
- .codex/scratch/dev360_excluded_media_archive_attempt03/：真实失败归档来源、请求和回包，保留不重放。

## Verification
candidate实际九请求与六下载的bytes/raw/self、180行顺序、360对照引用、三frontend/帧绑定及冻结来源均通过ROOT与Dev复核。ASR六请求实际完成，32成员全SHA与远端summary一致，ROOT及Signal复核通过，无API调用。归档三份base64读回实际解码并匹配本地全SHA，failure receipt明确容量失败且未产生tar。observer129/147/161 raw/self/PID/完整分组计数已ROOT实际通过，最新178仅刚读取待独立proof。新8工件create-only复制全字节核验通过，尚未Git暂存/blob复核。没有泛用pytest、环境安装、push或PR。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮直接只读HTML确认canonical URL与当前分支无匹配PR。
- 当前SHA `52d146bd925a7773238065ed11bf7afb85e5b76f`是本地未推送提交。
- openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md及reproducibility/README_fit_step120_candidate.md为协议与可用候选入口。

## Observations
- [result] 180候选的当前服务器计划与32成员ASR静态部署已实际完成并独立复核，尚未候选GPU生成或API调用。
- [result] 三模型对照最新观察178/360全PASS；161行已完成ROOT独立raw/self/PID/fullcounts核验。
- [decision] 冻结候选及完整失败分母、三模型over RMS资格与全原音PCM/时钟交付保持不变。
- [decision] 旧媒体仅保存一份持久完整tar，传输分块放独立SHM并明确可重建/易失语义，各文件系统保留5GiB容量余量。
- [blocker] 完整controls360/candidate180及内容、时钟、540视觉、GUARD20和统计资格仍需完成；archive03容量失败须用新04修复。
- [next_step] 核验后续controls快照并完成180候选启动的真实终态准入，同时审阅执行archive04；完成全Dev后做实际generator PCM内容、原音换轨、视觉/GUARD20和完整分母统计，再依结果推进方法。

## Relations
- continues [[Codex checkpoint - 2026-10-08T09-27-50Z - Committed usable candidate and running cross-model Dev]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]