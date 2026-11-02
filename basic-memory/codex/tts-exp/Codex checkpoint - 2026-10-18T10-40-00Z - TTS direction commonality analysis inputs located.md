---
title: Codex checkpoint - 2026-10-18T10-40-00Z - TTS direction commonality analysis
  inputs located
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t10-40-00-z-tts-direction-commonality-analysis-inputs-located
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 10:40:00+00:00
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

# Codex checkpoint - 2026-10-18T10-40-00Z - TTS direction commonality analysis inputs located

## Summary
已有20人纯增益方向补算完成；当前用户要求继续实际检验TTS编码方向变化及其与增益方向的共性，已定位20句10人完整配对缓存。

## Story
本会话围绕proposal核心贡献，从幅度响应机制延伸到共享方向。此前用报告46的20人、2155编码帧，先逐帧归一化编码再相减：+3dB跨说话人平均变化余弦0.812，留一人0.896，逐帧留一轴0.394；说明句级共享分量，不能称所有帧同向或通用增强。结果和独立数值复核已保存。

最新用户要求检验TTS。当前已读项目记忆和源码，定位2026-09-30新队列CAL20/10男性的natural/tts RAW与FIXED编码、自然FIXED±3dB和speech masks。尚未执行新的TTS几何分析。两种发声不能逐帧直接相减；拟先做逐帧单位编码的语音段均值对比，再寻找配对MFA音素作敏感性分析。固定RMS对照区分整体电平，但不能排除局部能量和发声结构。旧static source axis无增强收益，几何共性不得改写成因果增强。

## Working State
当前分支有大量其他会话修改和未跟踪文件，未清理、未提交。默认python缺numpy；使用C:/Users/dogle/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe。Basic Memory MCP未暴露，bm CLI可用，PYTHONIOENCODING=utf-8。GitHub只读API确认仓库；当前分支PR查询空，当前SHA未核验远端可见性。

权威输入：runs/recovered_20260930/remote/calibration_20260930/calibration_features/*.npz，calibration_summary.json（clips为字典），calibration_axes.npz。natural_z_plus3db以natural_z_fixed为基线，不能相对RAW。固定RMS=0.0376838172675746。现有20个natural TextGrid可读，尚未找到CAL的TTS TextGrid；v2_cal_mfa_producer_manifest在remote/world_phone_transfer_20261001_overlay/execution_view_inputs，可能有解析后的音素记录需检查。源calibrate.py已由CodeGraph读取。

## Changed Files
上一轮：runs/encoder_direction_commonality_20261018/{analyze.py,result.json,qa.json}（本机产物）；basic-memory/Experiments/纯增益编码方向的跨说话人共性补算.md及Research/研究主线与证据地图.md。当前TTS分析尚未写脚本或结果。

## Verification
先前100个增益NPY全部匹配manifest哈希；独立公式余弦最大误差1.11e-16。当前只检查CAL NPZ键/形状、summary和源码，未声称新实验完成。PowerShell读取完整summary曾有gbk解码错误，后续显式UTF-8。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local checkout SHA: `2b71639c3b555293375d248350999377168d515d` (remote availability not checked).

## Observations
- [decision] 对比单位编码的变化方向，按说话人等权，RAW/FIXED分开；不得用原始编码余弦高代替变化方向共性。
- [decision] 不启动新训练、不干扰推进实验或SyncTalk训练session；先复用CPU缓存。
- [next_step] 编写并运行TTS/FIXED/纯增益同队列几何分析，检查能否取得匹配音素，再独立复核及保存结论。

## Relations
- continues [[Codex checkpoint - 2026-10-16T10-52-00Z - Training amplitude design and Ubuntu history verified]]
- relates_to [[纯增益编码方向的跨说话人共性补算]]
- relates_to [[TTS 来源方向与声学机制确认]]