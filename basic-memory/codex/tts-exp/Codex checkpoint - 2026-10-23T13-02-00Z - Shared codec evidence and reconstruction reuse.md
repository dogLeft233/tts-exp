---
title: Codex checkpoint - 2026-10-23T13-02-00Z - Shared codec evidence and reconstruction
  reuse
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t13-02-00-z-shared-codec-evidence-and-reconstruction-reuse
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T13:02:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T13-02-00Z - Shared codec evidence and reconstruction reuse

## Summary
旧Qwen自然音频全码本重建实验仍有效，但不足以解释1.7B优势；共享codec意味着不必因模型规模改变而重复同一重建。

## Story
研究目标是解释归一化后TTS同步分数优势。用户最新问0.6B无增益是否使旧自编码自解码实验失效、1.7B是否需重做。本轮读BM正式报告并区分独立问题：当前0.6B未确认优于自然，不撤销纯codec控制；但旧N/R/T缺少稳定TTS正优势，不能证明文字生成任务造成优势。同10人norm自然7.108、0.6B6.861、1.7B7.517。1.7B对0.6B+0.656较稳健，相对自然+0.409严格9人区间跨零。旧10人codec自配R−N+0.001，95%[-0.204,+0.177]；固定自然评分−0.168。多数自动音素中心接近，但unknown与异常未人工裁决，不证明时间等价。

固定粗码run确认两模型同SHA codec。自然音频encode/decode不经过0.6/1.7B文字生成主干，所以无需仅换模型目录重做R。建议核对既有40人R是否覆盖当前10人，并核对音频、前处理和官方评价绑定，再复用N/R/T06/T17比较。旧codec自定义源PCM与新official链均值不能直接相减。T17>R即便成立，也只是完整生成相对重建的联合差异，非纯文字LM或粗/细码因果。本轮仅建议，没有执行新的资产核验或计算。

## Working State
此前固定粗码技术验证已完成并HOLD：精确自身重放通过，交叉补全内容及局部时间门未通过。权威run runs/qwen_fixed_coarse_20261023_attempt01/report.md，有效时间结果checks/mfa_lexical_repair02/summary.json；旧checks/summary.json为无效OOV首轮。既有4测试、216文件SHA读回通过，详见前checkpoint。本轮未新增代码或实验，无训练/TFG/SyncNet。工作区大量继承dirty/untracked，未清理或提交。Git分支/HEAD本轮直接核验；GitHub查询确认repo及当前分支无PR，HEAD远端可解析性未确认，不构造commit链接。

## Verification
本轮读取Router、地图、实验指令、codec与1.7B及固定粗码正式笔记、本地固定粗码report；确认前checkpoint精确agent/session身份。未重跑测试。BM写入脚本首次语法错误发生在任何工具调用之前，没有产生笔记或科学结果。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/qwen_codec_roundtrip_20261022_attempt01/report.md
- runs/qwen17_level_comparison10_20261023_attempt02/report.md
- runs/qwen_fixed_coarse_20261023_attempt01/report.md

## Observations
- [decision] 保留旧codec控制；不把相同codec重复运行视为1.7B新机制实验，不启动训练。
- [blocker] 尚未逐项核实当前10人R与T17的样本及官方评分绑定，不能报告T17−R数值。
- [next_step] 若用户授权继续，先核对现有R能否纳入同10人同协议N/R/T06/T17比较，只补缺失单元。

## Relations
- continues [[Codex checkpoint - 2026-10-23T12-45-19Z - Fixed coarse conditional completion probe held]]
- relates_to [[Qwen 自身语音编解码的时间保持与生成评分拆分 2026-10-22]]
- relates_to [[同一10人上的Qwen1.7B与音量归一化对照2026-10-23]]
