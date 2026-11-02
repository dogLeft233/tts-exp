---
title: Codex checkpoint - 2026-10-29T04-10-52Z - Chinese LiPS data pipeline running
  and paired SyncNet training queued
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t04-10-52-z-chinese-li-ps-data-pipeline-running-and-paired-sync-net-training-queued
status: open
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T04:10:52Z'
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

# Codex checkpoint - 2026-10-29T04-10-52Z - Chinese LiPS data pipeline running and paired SyncNet training queued

## Summary
20h Chinese-LiPS双条件SyncNet先导已实现、通过44项Ubuntu CPU测试；真实数据准备正在运行，正式GPU训练等待20h训练／2h验证数据READY。

## Story
用户要求进一步区分SyncNet零时移TTS优势来自真实对应改善还是训练负例塑形，已明确授权准备20小时Chinese-LiPS并按两组方案实际训练。本轮完成执行代码和预处理真实链，权威设计/执行入口为 [[受控评分器训练与同步专家交叉干预实验路线 2026-10-29]]，现为IMPLEMENTED／DATA_PREPARING，不能再把它称仅提案，也不能称已训练成功。

两个新网络采用官方V2 S架构、双塔及BN随机初始化。同seed1729、同正负配对/顺序/batch/预算，仅负loss权重不同。各5000steps/batch16/Adam1e-4/margin10，无声学增强；N1在signedlag/VAD/能量层按整窗近似phone相似性平衡权重，缺共同支持的层两组共剔。训练前强制pool覆盖门，验证用真实独立AV和已知PCM时移，best优先满足同步门，再AUC/MAE。两组均过门后才评旧20来源80媒体原生零时移TTS偏好。一个seed为功能先导，不构成稳定机制结论；不以TTS分数选模型。

原HF训练分卷缺最终中央目录、processed包仅96px嘴ROI，因此实现严格Range单成员索引/抽取原始FACE，按可用分盘确定性前缀候选后speaker-hash选实际20h，不假称完整85h均匀采样。官方val另选2h，test锁定。新请求固定HF revision db96948538811029011eee44602438a26710ecd9，早期main来源保留说明。Ubuntu盘仅余6.1GB，数据使用专属/dev/shm易失cache，C盘原件及processed持久备份；完整备份/时长/speaker检查后才原子发READY。

## Working State
主run：`runs/chinese_lips_syncnet20h_20261029_attempt01/`，Ubuntu同相对路径位于`/home/wjj/tts-audio/tts-exp/`。当前controller PID165780，状态`controller_status.json`=WAIT_DATA，启动04:05:58Z；四脚本及官方S源码已启动前冻结并绑定`frozen_sources.json`。其链为DATA→MFA→pool门→其他CUDA任务空闲60s→N0/N1顺序训练→能力门→偏好。失败停链、有界超时，只清理自己独立进程组，不终止其他英语评分。两次WAIT_DATA阶段修复的旧状态/log/source及receipts均保留；从未启动训练后重写历史。

Windows数据持久根`C:/Users/dogle/chinese_lips20h_20261029/`。当前data PID477088，日志`pipeline_attempt03_pinned.stdout.log`/stderr，index PID474196/`scan_pinned.*`；04:10Z状态已持久processed122val、train0，仍未READY。原始候选索引持续增长，4CPU预处理在运行；准备估4–6小时不是保证。最终ready位于`/dev/shm/chinese_lips20h_20261029/data_ready.json`，train_manifest/val_manifest供controller。恢复先读状态及PID实际命令，不凭旧PID操作；Ubuntu重启后需从C持久副本恢复cache。

本地git分支codex/cross-tfg-audio-head-20261003，SHA为本地`2b71639c3b555293375d248350999377168d515d`，21修改/1330未跟踪，大量继承工作未动，新增10个实验代码/测试未提交。GitHub连接器确认仓库及当前分支无open PR。无推送或生成器重训。

## Changed Files
- `scripts/experiments/chinese_lips_{data,phones,syncnet_train,preference,controller}_20261029.py`及五个对应`tests/test_chinese_lips_*_20261029.py`。
- BM既有实验方案追加执行协议，更新开头与metadata为IMPLEMENTED，旧正文/变更记录保留并读回核验。
- 主run本地保存CPU测试receipt、真实前后向sanity、WAV审计、两clip phone、冻结source哈希；Ubuntu保存运行控制、源码和全部未来产物。

## Verification
Ubuntu syncnet环境完整5文件测试实跑44 passed in8.61s。官方S共13,631,616参数、14BN fresh、同seed hash一致不同seed不同；真实224BGR/13MFCC输入CPU前后向finite且grad全部finite。独立复核修正MFCC20步实际215ms/3440samples完整支持，spn视unknown剔除。评估从checkpoint绑定的冻结源码重载，双塔forward逐位parity，篡改SHA拒绝。own-process-group反例含leader已退出、孙进程忽略TERM，清理通过。

已实际完成下载→scp→4CPU稀疏S3FD裁脸/PTS核查→C备份，最早完整36val批通过，随后计数已增长。原WAV与FACE内音轨5clips/3speakers：最佳波形lag全部0samples，corr .99062–.99649，首中尾lag0；不构成硬件AV同步真值。旧本地MFA字典与模型inventory错配导致spn的pilot已隔离；下载官方China v3字典后80627pronunciations全部匹配，2真实clip分别113/131phones、spn0。未正式GPU训练，未评新模型TTS偏好。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Chinese-LiPS official dataset](https://huggingface.co/datasets/BAAI/Chinese-LiPS)
- `runs/chinese_lips_syncnet20h_20261029_attempt01/cpu_test_receipt.json`
- `C:/Users/dogle/chinese_lips20h_20261029/cohort_build/pipeline_status.json`

## Observations
- [decision] 两条件只改负例loss权重，所有元数据排除/覆盖门在TTS输出前冻结，首轮仅一个配对seed。
- [decision] 稀疏S3FD2s插值、前缀数据采样和近似MFA phone均明确为自定控制方案，不冒充官方V2训练复现或独立真值。
- [result] 已实现并启动真实有界数据与等待训练链；44 CPU tests PASS，尚无训练或偏好结果。
- [blocker] 完整20h/2h数据尚在准备；GPU另有英语评分作业，不杀或挤占。
- [next_step] 检查Windows data pipeline与Ubuntu controller状态，数据READY后确认pool覆盖门和两组实际训练收敛；两组同步门通过才解读TTS偏好。

## Relations
- continues [[Codex checkpoint - 2026-10-29T03-46-00Z - Controlled SyncNet training design authorized for Chinese LiPS pilot]]
- relates_to [[受控评分器训练与同步专家交叉干预实验路线 2026-10-29]]
