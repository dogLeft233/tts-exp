---
title: Codex checkpoint - 2026-10-30T11-40-32Z - TalkModels stopped by user and GPU
  released
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t11-40-32-z-talk-models-stopped-by-user-and-gpu-released
status: open
task_status: stopped_by_user
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T11:40:32Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a18b7a-98d3-79d1-a715-721276a41216
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T11-40-32Z - TalkModels stopped by user and GPU released

## Summary
用户因推理过慢明确要求停止，MultiTalk、控制器及监控均已停止，目标3090释放；InfiniteTalk一个官方样片已完整验收。

## Story
原目标是192单张24GB3090按官方低显存流程跑通两模型、测实际显存与耗时，同时保护共享服务器。已完成独立环境、源码和27个权重校验，使用官方参数 num_persistent_param_in_dit0、480P桶、81帧、40steps、seed42，无量化/LoRA。InfiniteTalk完成后顺序执行MultiTalk（开官方TeaCache）。

最新用户指令为“算了吧，这两个推理太慢了，不适合我们做实验，先停了吧”，取代此前继续跑通的要求。已先暂停精确匹配的自身控制器，再终止自身MultiTalk推理及子进程，停止控制器、远端只读监控及本机两个监控SSH重试进程。不再启动GPU作业。源码、权重、环境及成片保留，未删数据或更改系统/他人任务。

## Working State
远端 /home/data2/wangjiajun/TalkModels/deployment_status.json 为 CANCELLED / STOPPED_BY_USER。user_stop_receipt_20261030.json 保存操作和资源释放证据：目标GPU-b937d7f2-4724-68fc-b815-cf2a58105122 无计算进程，显存4MiB、45°C、利用率0%，主机available RAM229.736GiB。控制器2235298、推理2344543均已终止；本机监控重试555804、451444及各自SSH子进程已精确停止。

本地 runs/deployment_talkmodels_3090_20261030/early_infinitetalk/ 保存视频、execution.json、validation.json、本地解码凭据和联系表；user_stop/ 保存停止凭据、最终状态、MultiTalk执行JSON及完整日志。.codex/scratch/talkmodels_3090_deploy_20261030/deployment_state.json 已更新当前权威取消状态，历史字段仅供追溯。临时脚本路径 C:/Users/dogle/AppData/Local/Temp/tts192-talk-deploy 属于脆弱本机缓存。

## Verification
InfiniteTalk EXIT_0，5208.923秒（86分49秒），NVIDIA0.5秒采样峰值16.000GiB；Torch maxallocated13.785GiB、maxreserved15.695GiB。输出896×448、81帧、25fps、3.24秒，带非静音音轨。服务器逐帧/音频/时长/运动检查全部PASS，本地81帧读回、SHA与远端一致，联系表已人工查看。视频SHA256 92f212f487cd927bda84bf87f1864121ea8331721b1947f42ffdbfd426d89e0d。单个官方示例不能外推长视频，也没有项目批量或评分。

MultiTalk最后日志24/40步（生成阶段26分03秒），用户停止时总计1773.957秒（29分34秒），中止前采样峰值14.342GiB，未完成视频验收。这不是完整推理benchmark。原execution.json为FAILED及Resource guard stopped…：包装器统一将SIGTERM转成该异常；停止receipt明确是用户中止，非OOM/内存阈值，保留原记录不改写。

辅助validate.py曾因numpy.bool_ JSON序列化失败，仅将moving检查转原生bool后同一InfiniteTalk成片重新验收PASS。官方源码未改。环境已验证情况及Decord精确元数据警告见模型部署文档，不能称原始pipcheck全通过。

## Changed Files
新增 basic-memory/docs/deployment/infinitetalk.md、multitalk.md，更新DEPLOY.md：区分单样例验收成功、未完成中止和用户当前不采用两模型的决定。更新scratch恢复状态、保存本地证据；未修改核心流水线、未提交或创建PR。仓库当前有1357个修改/未跟踪条目，大量为原有工作，不能清理或归入本次。Git SHA仅本地核对，未证明远端提交可达。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，之前已read-only API确认规范URL；当前分支无查询到匹配PR。
- [InfiniteTalk官方](https://github.com/MeiGen-AI/InfiniteTalk)，固定50aa0a94184315407a991ae804d9b58d6d311ba8。
- [MultiTalk官方](https://github.com/MeiGen-AI/MultiTalk)，固定c1ad84009b99ed36c97ab18e4517ca5f98692438。
- 本地 user_stop/user_stop_receipt_20261030.json 是取消原因及GPU释放权威证据。

## Observations
- [result] InfiniteTalk完整官方低显存短片通过，24GB3090足够此配置，但3.24秒成片耗时86分49秒。
- [decision] 用户认为两模型太慢，不适合当前实验，要求先停；不得继续此前“两个都跑通”的目标或自动复跑。
- [result] MultiTalk在24/40步后按用户要求中止，未获得完整推理耗时/峰值或验收成片。
- [next_step] 保持GPU任务停止并保留证据，只有用户明确提出新指令后才重新评估或启动。

## Relations
- continues [[Codex checkpoint - 2026-10-30T09-42-00Z - TalkModels 3090 environments ready and final weight pending]]
- relates_to [[infinitetalk]]
- relates_to [[multitalk]]
