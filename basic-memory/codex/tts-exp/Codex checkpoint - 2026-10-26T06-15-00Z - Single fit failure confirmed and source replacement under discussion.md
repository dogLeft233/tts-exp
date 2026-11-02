---
title: Codex checkpoint - 2026-10-26T06-15-00Z - Single fit failure confirmed and
  source replacement under discussion
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t06-15-00-z-single-fit-failure-confirmed-and-source-replacement-under-discussion
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T06:15:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T06-15-00Z - Single fit failure confirmed and source replacement under discussion

## Summary
完整拟合集仍只有S0168W0257重建失败；用户开始讨论替换这一条语音，尚未执行任何替换。

## Story
原目标为逐音素静态、动态与联合差方向迁移的11臂验证，先完成拟合集技术校准。上一轮独立Whisper与Paraformer支持S0168参考覆盖不全，固定镜像包成员SHA与库存相同；分阶段线性探针支持编码表示与声码器均有声学还原不足的线索，但不能量化不可逆损失或责任比例。有界L6频谱校正按说话人五折完整复验100人300条，仍为299条通过、S0168一条失败。
最新用户先询问是否只有一条失败，继而问“我们换一条语音呢？”。这是替换方案讨论；本检查点没有把原先不剔除样本约束默默改写成已执行替换。应明确按内容完整性而非重建分数制定替换规则，保存旧失败及原队列凭据，版本化新拟合集并全量复验。开发集和确认集继续不动。

## Working State
权威入口 runs/phoneme_direction_20261026_repair03/report.md 与 technical_status.json、full_postfilter_quality.json。当前状态HOLD_RECONSTRUCTION_AND_INCOMPLETE_REFERENCE，STOI中位0.900296250348263，S0168=0.749654298575322，原单条门0.750000。不能四舍五入当通过。
完整参考逐字内容仍非人工确认。新小型Ridge头已经拟合，冻结WavLM/HiFiGAN/TTS未训练；候选未准入或冻结，尚未运行新TFG或SyncNet。远端大产物在 /home/wjj/tts-audio/tts-exp/runs/phoneme_direction_20261026_repair03/；本地40份交付此前SHA通过。当前没有本聊天后台任务。
工作树有大量其他聊天dirty/untracked，本聊天源码亦未提交。Git根/分支/HEAD本次直接核对；GitHub只读核实仓库和分支无PR。HEAD未核实远端可解析。

## Changed Files
本次状态问答没有改实验代码、媒体或冻结清单。前轮实现见 scripts/experiments/phoneme_direction_20261026/ 中 separate_encoder_decoder.py、probe_l6_postfilter.py、calibrate_l6_postfilter.py、audit_l6_postfilter.py、Whisper核验脚本及 tests/test_phoneme_spectral_probe_20261026.py；三份canonical方案/实验/地图已更新，具体清单由上一检查点和run源码快照保存。

## Verification
本次重新读取本地technical_status.json、报告和canonical实验笔记，失败列表只有BAC009S0168W0257_N_0。前轮远端20测试通过、本地14通过及2依赖模块跳过；本次未重新跑模型或测试。原门、300条样本及冻结转写仍未改变。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`，未验证GitHub解析。
- runs/phoneme_direction_20261026_repair03/report.md

## Observations
- [result] 100人300条自然语音最新重建复验仅S0168W0257失败，其余299条通过。
- [blocker] S0168重建低于原门且参考覆盖不全，技术校准仍未通过。
- [decision] 没有执行替换或更改阈值，开发与确认保持不动；用户现在讨论换一条。
- [next_step] 向用户说明内容缺陷驱动的固定规则替换与按分数换样的区别，明确新版本全拟合集复验及原失败留档。

## Relations
- continues [[Codex checkpoint - 2026-10-26T06-07-51Z - S0168 source mismatch verified and full fit spectral repair still held]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
