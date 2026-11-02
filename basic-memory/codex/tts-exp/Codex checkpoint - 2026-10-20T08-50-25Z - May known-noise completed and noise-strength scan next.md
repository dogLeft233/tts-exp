---
title: Codex checkpoint - 2026-10-20T08-50-25Z - May known-noise completed and noise-strength
  scan next
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t08-50-25-z-may-known-noise-completed-and-noise-strength-scan-next
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T08:50:25.824246+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a157fe-7431-7040-8199-ea310042a054
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-20T08-50-25Z - May known-noise completed and noise-strength scan next

## Summary
May冻结编码器的已知噪声单强度实验已封存；用户要求继续扫描噪声强度，分辨低音量选择性压平与数值下限。

## Story
最初问题是低音量训练Sync-C提升是否因弱噪声在缩放时被删除。实际May历史浮点波形没有新增静音段；AVE有mel下限，wav2vec2首层GN epsilon改变低方差通道响应，但不能把通道自动叫噪声。报告59覆盖全May逐层弱响应核查。报告61加入已知白/粉红噪声、3种子、完整train/val/test、14块449次冻结前向：整体内容和噪声都乘0.5后，末层单位方向扰动相对变化−0.116534%到+0.082363%；只减已知噪声扰动下降15.6254%–29.9480%，低内容尺度补回噪声则增加15.9424%–33.7570%。没有训练、TFG、SyncNet，本结果不能证明训练收益机制。

最新用户要求不同强度噪声，查找低到某程度是否被低音量压平。已向用户说明保持整体0.5，扫描名义整split SNR0–100dB每10dB，延伸极弱精度诊断；先train/test各首320080采样20.005秒，复用报告61已封存噪声和3种子，30dB点与前轮逐值对齐。当前尚未创建或运行新的强度扫描代码；下一执行需要新根，绝不修改已封存报告61根。

## Working State
权威已完成产物F:/Datasets/tts-exp/may_known_noise_20261020_v1，final_receipt passed、1619成员2330198714bytes，manifestSHA d771ed6fd0af1403e0782b5e465a2f2920a1d8037dedc3183873960be1b0303d。报告basic-memory/docs/experiments/61-may-known-noise-encoder-pilot.md。原May在F:/Datasets/tts-exp/synctalk_gain_training_20261017/arms/{split}/gain_0/aud.wav，整split float32标准化后才切块。前轮噪声noise/{split}_{white|pink}_{seed}.npy已用protocol绑定SHA。权重F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/weights/wav2vec2_base，facebook/wav2vec2-base revision0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8。

本机Python C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe，transformers依赖F:/Datasets/tts-exp/encoder_survey_20261018/deps，plotdeps同级。RTX4060 8GB；F约26GB空闲。FP32、TF32off、eager、eval/inference_mode。首GN epsilon默认1e-5，输入减半+epsilon2.5e-6在前轮所有七层严格逐值恢复。不要把同精度重复相同误称为数值分辨率。

## Changed Files
本聊天新增scripts/experiments/wav2vec2_weak_signal_20261020、may_known_noise_20261020及对应tests，报告59/61与图、BM研究地图和实验索引。仓库当前1084项dirty/untracked，含其他会话改动，不全量clean/stage/commit。分支codex/cross-tfg-audio-head-20261003，本地HEAD `2b71639c3b555293375d248350999377168d515d`。Basic Memory工具未暴露，使用C:/Users/dogle/AppData/Roaming/uv/tools/basic-memory/Scripts/python.exe通过basic_memory.mcp.tools API；发布例子.codex/scratch/publish_may_known_noise_20261020.py。

## Verification
前轮独立QA重算449输入、2352层配对、75264样本配对、18噪声文件，误差0；4个CPU测试通过。所有84噪声chunk epsilon补偿七层严格相同，clean重复严格相同。当前强度扫描未运行，不能把计划写成结果。CodeGraph本轮已查但新增代码未索引且自动同步因磁盘满冻结，按规则回退源文件。Basic Memory已读路由/地图/实验指令；用户默认不训练。

## References
- Canonical repository verified: https://github.com/dogLeft233/tts-exp
- Branch PR read-only query: 0 results
- GitHub /commits/2b71639c3b555293375d248350999377168d515d lookup unavailable HTTP422

## Observations
- [result] 已知30dB噪声下，整体音量0.5基本保留相对编码扰动；noise-only对照确有剂量效应。
- [decision] 新扫描用同一噪声实例做0–100及更弱SNR，始终以相应音量无添加噪声特征为参照；不重新标准化混合物。
- [decision] 极弱扰动用float64单位向量直接相减，避免2−2cos消减；加入FP64编码诊断区分FP32数值下限。前轮source run SHA f7a6d930c34656de02853c9710b2adc6d3ccd3a082ee842910062d0f5b17e575。
- [next_step] 实现并运行新may_noise_strength_20261020冻结剂量扫描，核验30dB桥接与epsilon控制、精度边界，再独立分析并发布下一空闲编号报告（当前最大61，发布时重查）。
