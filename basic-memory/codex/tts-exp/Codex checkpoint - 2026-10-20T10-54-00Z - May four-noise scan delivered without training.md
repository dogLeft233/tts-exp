---
title: Codex checkpoint - 2026-10-20T10-54-00Z - May four-noise scan delivered without
  training
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t10-54-00-z-may-four-noise-scan-delivered-without-training
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T10:54:00.899105+00:00'
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

# Codex checkpoint - 2026-10-20T10-54-00Z - May four-noise scan delivered without training

## Summary
May四类已知干扰的强度扫描完成并独立核验、发布报告65；低音量编码减敏依赖片段与干扰类型，不能据此确认训练收益来自噪声减少。

## Story
用户最初追问低音量训练Sync-C提高是否因为弱噪声变成零。此前浮点波形核查不支持新增静音段，wav2vec2逐层实验则指向首GN epsilon造成尺度依赖。已知白/粉红噪声和极弱剂量结果进一步说明，整体内容与噪声共同缩小不等于降低输入SNR。最新授权是在之前May实际音频上，扫描单录音背景讲话、四人babble、带起伏语音谱形噪声及稀疏点击，先不训练。用户最后“现在够了吗？”是对主机可用内存的询问；释放内存后任务继续且全部完成。

本批固定A=s+n、B=0.5(s+n)，分别相对E(s)、E(0.5s)比较末层单位方向扰动，也记录七层原值样本/逐帧矩和相对L2。两May历史标准化前缀各20.005秒、每类三固定实例、FP32名义整split SNR0–100每10dB及120/140dB；首实例80/90/100/120dB另做FP64。746个科学条件、34次恢复桥接，总780次验收前向，0训练/TFG/SyncNet。

结果不是统一去噪：名义100dB三实例中位数变化，背景讲话train+6.3%/test−74.0%，babble+29.5%/−73.6%，起伏谱形+7.4%/−62.4%，点击+21.1%/−2.5%。最一致的≥20%减敏是test起伏谱形90/100/120dB三实例均满足；train没有任何类型在任一档三实例均满足，也无两前缀六实例共同满足的类型/档位。FP64首实例复核保留主要片段差异；不是硬噪声门限或原May本底噪声被消除的证据。极弱FP32分辨标记必须保留。

## Working State
权威报告 [[65-may-four-noise-scale-scan]] 已发布，研究地图与实验索引已读回确认。持久结果根 `F:/Datasets/tts-exp/may_four_noise_20261020_v2`，final_receipt passed，2438成员、5621067256逻辑bytes，manifest SHA `75542f8dccb92ed125d3e800e7a930c38f69a4cd27c45988651ebf60b77a8345`；完整清单每成员已fresh SHA读回。这里硬链接继承部分不变数组，逻辑bytes不是新增物理占用。最终图已目视核对，正文不作训练收益结论。

原v1主机125MiB首GN复制失败，370次已提交结果获准继承；22次完整但未提交输出保持排除、原根不改写。v2采用256行流式CPU捕获，数学和精度协议不变，19进程均先逐层完整SHA桥接报告63。第一次流式恢复job1又因主机提交余量不足失败，0新增完整结果；随后冻结资源调度器，用户释放内存后完成。原失败根所有成员最终复核未变。全部已完成模型前向含排除22次为802；另2次中断尝试不计成功。

机器本地副作用：为缓解C盘空间，将本项目公开Wav2Lip/SyncNet assets缓存30文件490490059bytes转存F:/Datasets/tts-exp/host_cache_relocation_20261020_v1/gain_common_assets，原C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/assets为junction，复制和兼容路径SHA均通过后删除旧物理副本。没有移动Python环境或科学输出；receipt在该F根，并收入本批sources。此缓存不是本轮wav2vec2权重变更。

所有路径是本机持久盘，未声称云归档。当前仓库1089项dirty/untracked包含其他会话工作，未全量stage/commit、未清理。实验计算无活动进程。环境Python C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe，transformers/plotdeps在F:/Datasets/tts-exp/encoder_survey_20261018。Basic Memory经C:/Users/dogle/AppData/Roaming/uv/tools/basic-memory/Scripts/python.exe的basic_memory.mcp.tools调用；项目映射.codex/basic-memory.json，用户级配置不存在。

## Changed Files
科学脚本scripts/experiments/may_four_noise_20261020/{run,recover_streaming,resume_driver,analyze,seal}.py及README.md；测试tests/experiments/test_may_four_noise_20261020.py；报告basic-memory/docs/experiments/65-may-four-noise-scale-scan.md及三图assets；BM研究地图/实验README更新。发布器.codex/scratch/publish_may_four_noise_20261020.py、当前checkpoint脚本是本机辅助入口。冻结run/recover/resumer源码hash受科学/恢复协议绑定，不可后改伪装原执行。

## Verification
6项CPU unittest实际通过，包含FP32/64非连续数组流式矩/样本/hash与整数组精确等值。独立QA通过24噪声逐值重建、6源freshSHA、两整split标准化桥接、746输入hash+34额外参照hash、308224七层raw样本对、688000完整末层帧对、70参照层桥接、48epsilon配对、344波形配对。不同求和实现矩最大差5.4569682106375694e-12，在预设容差内。B−0.5s严格等于0.5(A−s)，没有低音量独有删除；48个epsilon2.5e-6控制七层完整hash恢复A。全部19worker receipts passed。原协议不变且最终完整SHA封存通过。CodeGraph按规则先查但数据库/磁盘问题导致自动同步冻结，新脚本未索引，回退直接源码。没有训练/TFG/SyncNet检查，因为本轮明确不训练。

## References
- Read-only GitHub lookup  unavailable: HTTP Error 403: rate limit exceeded. Local SHA remains `2b71639c3b555293375d248350999377168d515d`; do not infer a remote commit link.
- Read-only GitHub branch query found no open PR (0 total historical matches).
- Read-only GitHub lookup /commits/2b71639c3b555293375d248350999377168d515d unavailable: HTTP Error 422: Unprocessable Entity. Local SHA remains `2b71639c3b555293375d248350999377168d515d`; do not infer a remote commit link.
- Formal report: `E:\Documents\tts-exp\basic-memory\docs\experiments\65-may-four-noise-scale-scan.md`
- Evidence receipts: `F:/Datasets/tts-exp/may_four_noise_20261020_v2/independent_qa.json`, `final_receipt.json`, `delivery_manifest.json`, `failed_attempt_snapshot.json`, `recovery_protocol.json`, `worker_receipts/`.

## Observations
- [result] 四类干扰强度扫描已完成、报告65已发布并完整封存；746科学条件+34恢复桥接、0训练/TFG/SyncNet。
- [result] test弱起伏谱形减敏最一致，但train对应趋势相反，点击声不稳定；没有跨两May前缀的一致≥20%减敏档位。
- [decision] 编码响应只能作冻结编码器筛查；不把编码减敏叫噪声删除、训练损害或训练收益。
- [decision] 内存故障保留失败根和排除输出，通过不改变数值的流式捕获及逐层hash桥接恢复；不改写冻结历史。
- [next_step] 向用户交付报告65与完整曲线，解释片段依赖；任何训练实验需后续明确授权并单独设计，当前无需追加计算。

## Relations
- relates_to [[65-may-four-noise-scale-scan]]
- continues [[Codex checkpoint - 2026-10-20T08-50-25Z - May known-noise completed and noise-strength scan next]]
