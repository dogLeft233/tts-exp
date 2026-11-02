---
title: Codex checkpoint - 2026-10-29T03-46-00Z - Controlled SyncNet training design
  authorized for Chinese LiPS pilot
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t03-46-00-z-controlled-sync-net-training-design-authorized-for-chinese-li-ps-pilot
status: open
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T03:46:00Z'
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-29T03-46-00Z - Controlled SyncNet training design authorized for Chinese LiPS pilot

## Summary
受控评分器训练方案已具体化，用户现已授权准备20小时Chinese-LiPS并运行两个负例权重条件的小规模训练；执行前发现Ubuntu磁盘仅剩6.2GB且有其他评分作业。

## Story
目标是区分TTS真实时间对应改善与训练负例塑造的评分偏好。共同窗口报告73已确认原始零时移仍有TTS优势，不能只归因搜索；历史时移、音素池、独立null及视觉校准已做过，不能重包装为新实验。权威方案为 [[受控评分器训练与同步专家交叉干预实验路线 2026-10-29]]，本轮先完成方案而未下载或训练。随后用户明确要求执行其第一步，下一阶段应实现而非重复提案。

方案推荐Chinese-LiPS真实同录中文AV，官方训练中确定性选约20h、官方验证约2h，保留说话人隔离。采用公开SyncNet V2 S架构（5帧、20MFCC步、1024D原始欧氏距离），双编码器和BN共同随机初始化。N0与N1使用完全相同正负配对、次序、batch和训练预算；N0等权，N1在有符号lag/VAD/能量层内按全窗口近似音素序列相似性重加权，相似/不相似目标50/50，层内权重和保持且总负权重均值1。候选负lag建议±240–1000ms，40ms步；这是自定受控配方，不是官方V2训练复现。首先一个配对seed验证功能，再扩至至少三个seed。无声学增强；不同声学因素后续独立试验。

先在独立真实AV已知时移/错配上检验同步学习和保留能力，再评旧20来源云TTS自然原生视频；旧队列仅探索，跨网络比较胜率/固定错误工作点，不能直接比较欧氏单位差。若基线不学会同步或不重现官方偏好，不能直接解释官方V2。

## Working State
权威计划已追加“第一步具体化：数据、初始化与负例重加权（2026-10-29）”，状态SPEC_ONLY且读回10152字符、旧正文保留验证通过。此刻执行尚未实现。Windows当前根E:/Documents/tts-exp；Ubuntu根/home/wjj/tts-audio/tts-exp，通过BatchMode SSH wjj@192.168.50.2可达。Ubuntu V100 16GB，目前MTDVocaLiST英语评分PID132875约1006MiB，非本任务，禁止终止。Ubuntu仅6.2GB空闲；WindowsC约64GB空闲、E约7.8GB。先解决有界数据存储与下载策略，不能整包占满磁盘。GPU使用是易变快照，启动前重查。

本地dirty工作树21修改、1320未跟踪为继承状态，未清理或覆盖。本地SHA未验证远端，作为本地提交记录；GitHub连接器确认仓库身份、当前分支无open PR。

## Changed Files
- basic-memory/Experiments/受控评分器训练与同步专家交叉干预实验路线 2026-10-29.md：方案及第一步规格，尚无训练代码改动。

## Verification
- BM方案追加前后精确核验通过，旧正文保留，新增段落一致。
- 官方Chinese-LiPS论文/作者HF页面已核验：约100.84h、207说话人，官方speaker-disjoint train/val/test；未下载，人工对齐声明不构成硬件毫秒真值。
- SyncNet官方S架构源码核验；V2完整历史训练配方未知。
- SSH、磁盘、GPU进程只读检查成功；gh未安装，改用GitHub连接器完成身份及PR查询。
- 没有运行代码测试、新GPU评分或训练，无声称收敛结果。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Chinese-LiPS author data](https://huggingface.co/datasets/BAAI/Chinese-LiPS)
- [Chinese-LiPS paper](https://arxiv.org/html/2504.15066v1)
- Local commit: `2b71639c3b555293375d248350999377168d515d`.

## Observations
- [decision] 用户已经授权20h数据准备、两组小规模训练与后续同步能力/TTS偏好验证，不再询问是否开工。
- [decision] 第一轮只改变相同负例的loss权重；不混入声学增强、架构更换、推理池更换或TTS伪同步正例。
- [blocker] Ubuntu系统盘仅剩6.2GB；需采用可控数据大小或另一个可用缓存位置，不能直接下载93GB全库。
- [next_step] 审核Chinese-LiPS文件组织与可选择下载方式，确定20h存储布局并实现数据/训练配对与能力门。

## Relations
- continues [[Codex checkpoint - 2026-10-29T02-44-31Z - Scorer mechanism audit complete and no retraining validation design]]
- relates_to [[受控评分器训练与同步专家交叉干预实验路线 2026-10-29]]
