---
title: Codex checkpoint - 2026-10-18T11-09-00Z - Heldout common gain direction experiment
  completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t11-09-00-z-heldout-common-gain-direction-experiment-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 11:09:00+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a139e9-67ce-7b41-90f2-5b7019492a69
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T11-09-00Z - Heldout common gain direction experiment completed

## Summary
20人拟合的固定增益共同方向在10位留出说话人上带来正向Sync-C响应，但未完整复现实际+3dB，且共同分量与残差的直接差异不明确。

## Story
本会话从proposal核心贡献推进到幅度作用机制。此前几何补算显示增益的编码变化具有跨人共性，TTS方向共性则很大程度依赖原电平。用户本次要求真正移植增益共同方向，与剩余方向对照，用未参与计算的说话人验证，并留意GPU占用。

已完成Wav2Lip冻结权重内部干预：报告46的20人、男女各10人，用逐帧单位编码的+3dB减RAW改变量先句平均、再说话人等权，冻结g。另选CAL的S0052–S0061十位男性、各首个字典序句子，未参与拟合。COMMON使用r0*unit(u0+g)，RESIDUAL使用r0*unit(u1-g)，保持原编码长度；残差是相对固定均值的余量，非正交投影，也需要该测试句真实增益编码。另有BASE、BASE_REPEAT、实际GAIN_P3、FULL_DIRECTION及两个非负敏感性对照，共80视频。全程固定原自然PCM评价、同脸/框/时钟，保存完整31时滞距离。

主guard20结果：GAIN+0.122，FULL_DIRECTION+0.100，COMMON+0.055（95%[+0.008,+0.104]），RESIDUAL+0.044（[-0.015,+0.106]）；COMMON_NONNEG+0.069，RESIDUAL_NONNEG+0.027。COMMON-GAIN=-0.067[-0.120,-0.009]，COMMON-RESIDUAL=+0.011[-0.040,+0.069]。预先固定±0.05等效门未通过，不能把单臂显著性差异当作共同方向优于残差。非负共同对照主要增加错位背景距离，不代表同一种评分变化机制。上述为未校正探索性95%区间；旧CAL仅拟合留出，不是全新未见确认集，不推感知质量或增益解释比例。

## Working State
任务已完成，没有本次GPU进程或科学阻塞。Ubuntu V100正在其他会话的SyncTalk训练且根盘/SHM紧张，因此只读复制已有代码权重，没有占用其GPU。实际在本机RTX4060的隔离环境执行，batch4，峰值allocated约299MiB/reserved386MiB，已释放。

权威报告与小型结果在runs/gain_shared_direction_20261018/：report.md、protocol.json、analysis.json、independent_qa.json、run_receipt.json、execution_source_seal.json、artifact_manifest.json。完整203文件/50,042,172字节位于机器本地C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/run_attempt03/；env/Scripts/python.exe为独立GPU环境。不能将这些未提交/机器本地资源声称为远程完整备份。

前两次执行原件保留：run因RTX TF32导致编码bridge不符，在生成前失败；禁用TF32后run_attempt02通过bridge，生成11条后Windows原子替换receipt遭瞬时PermissionError。加入有界写重试后run_attempt03完整重跑80条，科学协议和阈值未改。旧失败不是科学阴性。C盘当前约2.23GB，未来工作先查空间。

## Changed Files
- 新增未跟踪scripts/experiments/gain_shared_direction_20261018/{core,prepare,run,analyze}.py及tests/experiments/test_gain_shared_direction_20261018.py。
- 已写Experiments/增益共同方向与残差的留出说话人生成验证.md并更新Research/研究主线与证据地图.md；未改proposal。
- analyze.py默认路径仍attempt02；任何重分析必须显式设置GAIN_RUN_DIR为上述run_attempt03，避免误读旧失败。
- 共享仓库有大量其他会话修改与未跟踪文件，未清理、未提交。

## Verification
实际完成：4个单元测试、pip check、40拟合输入SHA与10测试PCM来源核验、说话人无重合；四个跨硬件编码bridge在禁用TF32后maxabs≤2.444e-6。十句native-forward/direct-decoder一致、十句BASE重复像素/特征/距离完全相同。独立NumPy从80组V/A重建距离，最大误差3.8147e-6；标量和条件重建误差0，编码长度最大相对误差1.407e-8。没有运行感知评价或训练。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，只读GitHub API已确认。
- 当前SHA `2b71639c3b555293375d248350999377168d515d` 仅本地；GitHub解析返回422，不提供远程commit链接。当前分支PR查询为空。

## Observations
- [result] 固定共同方向在拟合留出说话人上足以诱发部分评分响应，尚不够替代真实加大音量。
- [decision] 固定均值余量与正交分解区分；保留原角度剂量，不宣称相同剂量的方向优劣。
- [decision] 不占Ubuntu既有训练GPU；保留失败执行及全部非负敏感性结果，不扩样追逐阳性。
- [next_step] 向用户交付本次结果、直接对照区间与报告链接；后续若继续研究再另定匹配角度剂量或独立质量端点，本次不自动扩实验。

## Relations
- continues [[Codex checkpoint - 2026-10-18T10-40-00Z - TTS direction commonality analysis inputs located]]
- relates_to [[增益共同方向与残差的留出说话人生成验证]]
- relates_to [[TTS 编码方向共性与电平混淆的配对音素补算]]
- relates_to [[研究主线与证据地图]]