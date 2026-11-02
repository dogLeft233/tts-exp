---
title: Codex checkpoint - 2026-10-14T21-11-24Z - Native screen closed and output ownership
  gap identified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t21-11-24-z-native-screen-closed-and-output-ownership-gap-identified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 21:11:24.497540+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-14T21-11-24Z - Native screen closed and output ownership gap identified

## Summary
18条构音正负干预screen已关闭；随后源边界和实际mask定位到Muse输入j/测量j+4的直接覆盖0/14帧，当前转为output-owned坐标与无损原件的小方案准备。

## Story
用户原目标是解释TTS驱动优势并实现跨TFG自然音频增强头，最新明确要固定RMS/内容/时间、隔离native三因素并用独立嘴部指标验证，同时降低审计和弱候选规模。报告39原生/物理筛查与报告40旧144轨迹功能坐标已关闭。本轮仅两句×三模型×同路线zero/正/负18短视频，原GPT-6 Luna/max执行，ROOT一次实际科学QA。

18视频、原自然PCM/clock、评分/口部轨迹全部完成并实际检查。冻结p下zero六格raw/z Cp/Hp都正，完整双向门仅1/6格、0模型双人。SyncNet增益与冻结构音对比可以分离。原数组和门完整保留在报告41，未事后改phase/dose/符号。

后续只CPU：同phone自然局部支持诊断发现Muse plus全部6token距离缩小但功能指标不稳，不支持统一off-manifold解释；token分量中Ditto6/6 signed odd raw/z正、Wav/Muse语境混合。ROOT继续沿实际forward理解坐标，而未扩大候选。

关键新发现是Muse真实2D UNet逐prompt row输出同index帧。原operator修改audio MFA core j，固定功能端点却读j+4，两句各7个Cp帧与所有修改prompt行交集0；Hp p-4反而直接暴露。IMAGE逐帧MediaPipe无VIDEO trackerstate，预编码PNG已删除、现存H264可能传播编码差异。因此报告41开头显式收窄decoder解释：冻结screen失败不直接说明Muse缺少构音轴，应先修输入所有者/输出事件映射。Wav按16列mel chunk对端点7/7直接覆盖；Ditto同索引1/7但80-frame/10overlap状态使同索引数不能代表因果范围。

## Working State
权威入口是研究地图、Task `native-axis-factor-pilot-20261013` current_step18、`HANDOFF.md`、报告41与专题[[接收端局部构音响应坐标]]。最新Task/实际receipt优先于本不可变快照。

新 `output_owned_musetalk_protocol_attempt01.json`已冻结SOURCE/CPU preparation only：原audio core I→prompt rows I+固定p4，在owned行真实baseline m上作同natural9FIT axis/同dose旋转；clock/PCM/RMS/portrait/latent/seed不改，越界/重叠拒绝。未来上限仅6Muse短row（两句zero/正/负，newzero用于缺失预编码原件），仍未GPU准入。primary IMAGE几何来自保留precodec RGB/PNG/无损视频，H264仅标准SyncNet/codec对照；原S17/p/17lags/raw+z门保留。

原Muse Luna准备CPU物化source；原angular Luna已提交SOURCE_ONLY precodec runner02与接口manifest，尚未ROOT审阅/执行。不可把SOURCE_ONLY当准入。两代理复用原Luna/max，无模型替换。

服务器wjj@192.168.50.2:22保持运行，用户取消关机；当前无GPU模型任务。scratch、远端SHM和未提交BM仍machine-local/fragile，非完整灾备。旧训练/中心/Dev/Confirm/归档/cloud donor/visual teacher停放。秘密不进文件/argv/BM/log。

## Changed Files
- 新正式报告 `basic-memory/docs/experiments/41-native-articulatory-response.md`与三组PNG/SVG、三CSV；已加入0/14直接暴露解释修正。
- `basic-memory/Research/研究主线与证据地图.md`、Task18、三个专题、README、HANDOFF与CONTEXT99行已维护；不写独立进度日志。
- probe scratch保留完整readout、科学QA、局部距离/组件JSON、源边界备忘、新output-owned协议及未执行源码。Git9条tracked dirty中本轮只改CONTEXT，其余继承；1003条untracked不归为本轮，也不删除/重置。

## Verification
- 唯一readout05 worker2542617/49386476，UTC20:17:39–20:20:38Z rc0/18PASS，完整152文件903740B读回；launcher/supervisor/worker已退，终态GPU0%/5MiB/空owners。不能重复launch；未来启动前用fresh状态。
- ROOT媒体QA3d4b46d7…已对18实际PTS及全解码自然PCM核对；ROOT响应QA72ef059b…：36矩阵/18geometry/7146几何比较maxerror0，独立f64列均值对f32曲线舍入差2.746e-6，已验曲线精确重建B/Dmin/D0/C。
- Local-support CPU PID101560 rc0；ROOT603数值比较max6.66e-16（QA1a6139ed…）。Token JSON-only CPU PID103472 rc0；ROOT144公式比较maxerror0（8d668bec…），18token+3显式缺失。无新模型/编码/MFA。
- SOURCE-only boundary memo6f69b668…与实际mask快照2d2b4500…共同定位Muse0/14直接端点覆盖；没有新forward或导数。joint扰动除以单token剂量不识别partial Jacobian。
- 原renderer/readout前置错误、一次额外CLI选项0分析错误及图表schema/ABI失败保留，局部修复后成功，不扩大环境审计。source03图面QA接受，最新plot receipt dd6011da…，科学CSV不变。
- 未运行实现测试/pytest/compileall、安装依赖、API或新训练。Git直接只读确认当前branch/HEAD/dirty；GitHub repositoryAPI200、exact-head PR空。HEAD此前commitAPI422，local/unpushed，不构造远端commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- local/unpushed HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`，无PR。

## Observations
- [result] 找到具体输入→输出坐标缺口：Muse的直接条件作用域与固定Cp帧完全错开，不能把这个screen解读成已隔离decoder轴失败。
- [decision] 数值/冻结门不改，解释在报告和地图显式修正；下一只修output ownership并保留无损原件，不选新phase或扩大旧dose/模型矩阵。
- [decision] 先一次CPU候选QA和最小source审阅，再决定最多6Muse短row，当前GPU尚未准入；长期head目标active，尚无统一head胜RMS。
- [next_step] 接收output-owned CPUsource与precodec renderer源码，核对I+p映射和原件接口，准许一次CPU物化并核验直接覆盖/norm/Gram/non-owned/clock后再决定六行执行。

## Relations
- continues [[Codex checkpoint - 2026-10-14T20-26-22Z - Native response videos qualified and measurement readback]]
- tracks [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- supported_by [[41-native-articulatory-response]]
- relates_to [[接收端局部构音响应坐标]]

## Resume Artifact Pins

- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/output_owned_musetalk_protocol_attempt01.json` SHA256 `561e81a747b62c02af7a2465eac9457436013ffba45286be15d8c6cf40c9123e`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/temporal_ownership_boundary_attempt01.md` SHA256 `6f69b668e1ab47ded5f378424ff77588fd5e748c21f66615279094873ebf5062`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/root_musetalk_direct_exposure_snapshot.json` SHA256 `2d2b4500d23ba289d2c60629ed9e4c09cf8fca8e4a7877173376d6f7f252b62e`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/native_height_musetalk_runner_attempt02_precodec.py` SHA256 `484f08bac841513a754eb94910b91b8fbc863bc9ca080d74746f3b221e80eeb0`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/musetalk_precodec_output_interface_attempt01.json` SHA256 `7386f6e4959a4b67c0ce7df019fbc80a0de91c01dc01927f1a14a6bfb109ca2c`.