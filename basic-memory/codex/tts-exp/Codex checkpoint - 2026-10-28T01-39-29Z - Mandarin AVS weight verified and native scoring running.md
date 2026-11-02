---
title: Codex checkpoint - 2026-10-28T01-39-29Z - Mandarin AVS weight verified and
  native scoring running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t01-39-29-z-mandarin-avs-weight-verified-and-native-scoring-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T01:39:29Z'
username: dogle
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

# Codex checkpoint - 2026-10-28T01-39-29Z - Mandarin AVS weight verified and native scoring running

## Summary
AVS_u中文先导已完整下载并严格加载官方权重，修复视觉工具模块导入后通过32/40时移控制，现正运行60条云TTS原生评分。

## Story
原问题是区分TTS的Sync-C优势与真实同步改善。此前CoGenAV、PEAVS与SyncReward原生比较和官方调用审计已完成，见上一checkpoint及其权威实验笔记。AVS论文核对表明u无需真实参考，m/v需要同音频真实视频，不适合现有静态图生成数据。用户授权先中文后英文，现问权重是否完成和能否启动；已实际修复并重启，而非仅提出方案。
80条输入（20自然、60云TTS）沿用现有数据，无新API或TFG生成。主分逐帧cos均值，次要展平cos，FP32官方AV-HuBERT最终层、缺模态特征置零，AutoAVSR口部适配。精确论文checkpoint与归约未确认，因此称论文定义复现而非作者官方评分包；英文训练模型的中文适用性仍需验证。

## Working State
权威run为Ubuntu /home/wjj/tts-audio/tts-exp/runs/avs_u_zh_pilot_20261027_attempt01，deploy /home/wjj/avs-u-20261027。下载收据1928060161 bytes，SHA256 51857a1438c090cca1e3901b99b8b8cf74092763d58344874180ec56f5131261；禁止重复下载。最初后台自动评分10月27日15:13因from avhubert import utils取得fairseq.utils而Compose缺失失败，日志与status/exit保留.initial_import_failure。修复为显式importlib.import_module('avhubert.utils')，不改科学选择；execution_repair.json保留原协议hash并绑定最新脚本154be804fee5eac61d1992746d50f0be49e842588a46c187bf98583dd4c93721，前修复收据另存。10月28日重启PID13998，最近status RUNNING scored26（20自然+6云），不要重复启动。直接后台子进程无新launcher退出监护，须读score.log/status并检查PID，不能依赖陈旧score_exit。
40项真实PCM±400ms固定视频控制primary_lower=32，冻结门槛32，PASS，已自动继续60云评分。通过只是粗时移控制，不证明中文感知真值、无TTS偏好或优于Sync-C。结束后按20来源等权、strict12分层，三draw不独立。英文后补尚未冻结队列。
Windows唯一入口E:/Documents/tts-exp；只同步小元数据，大权重/embedding在Ubuntu，尚未完整归档新run。保留E>1GiB。用户暂停生产及取消交叉音轨不恢复，不关闭外部ComfyUI。本次GPU开始322MiB/0%，评分后1693MiB。大量继承dirty/untracked，未清理提交或推送。

## Changed Files
未跟踪scripts/experiments/avs_u_zh_pilot_20261027.py及tests/test_avs_u_zh_pilot_20261027.py；运行脚本本次只修显式模块导入，测试增加可选真实权重前向产物重放。代码与小产物仍依赖本地未提交工作树。已更新现有Task，不建进度日志。

## Verification
前序160输入SHA、音频frontend对上游max误差0、配置兼容随机API探测均通过。现在真实官方权重strict_encoder加载成功，首样本及±400ms真实前向成功（.1014214→.0787519/.0724761）。父侧独立重算截至21条embedding主分max误差0，parent_score_replay.json。更新的三测试函数本次直接执行PASS（含真实权重产物重放），未pytest。两次独立裸导入探针因上游路径和重复model注册失败，不能称通过；实际主入口前向成功才是调用修复证据。
本次实际git root/branch/SHA与dirty状态已读；GitHub REST确认canonical仓库，当前分支open PR为空。本地SHA未核实远端可解析，不构造commit链接。checkpoint用户级配置不存在，采用项目.codex/basic-memory.json。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- basic-memory/docs/reference/avs-metrics-paper-review-20261027.md
- runs/avs_u_zh_pilot_20261027_attempt01/protocol.json
- runs/avs_u_zh_pilot_20261027_attempt01/controls.json

## Observations
- [result] 官方权重下载完整并实际严格加载，中文时移控制32/40下降，达到预设门槛。
- [decision] 保留原冻结协议，工程兼容修复以单独收据记录，不按分数改模型、归约或32/40门。
- [decision] 中文先导与英文后续分开解释，不用指标投票证明口型真值。
- [blocker] 原文精确checkpoint、作者独立实现及中文感知效度未确认；当前原生比较尚未完成。
- [next_step] 读取PID13998及status/score.log，完成全部80原生分数后独立重放并报告来源等权差值、TTS胜率与Sync-C冲突，保留strict12分层。

## Relations
- continues [[Codex checkpoint - 2026-10-27T06-46-00Z - Native scoring official audit complete and AVS suitability review]]
- relates_to [[AVS_u中文域同步控制与自然TTS原生比较 2026-10-27]]
