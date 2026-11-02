---
title: Codex checkpoint - 2026-10-04T12-57-03Z - Qualified spectral evidence and recoverable
  cross-model continuation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t12-57-03-z-qualified-spectral-evidence-and-recoverable-cross-model-continuation
branch: codex/cross-tfg-audio-head-20261003
git_sha: 0597b3dfbfd6bd5df93c96d7cb70be12ce476e59
model: gpt-6.1-sol
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
started: '2026-10-04T12:57:03.3057683+00:00'
agent: codex
trigger: compact
repo_root: E:/Documents/tts-exp
username: dogle
status: open
repository: dogLeft233/tts-exp
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T12-57-03Z - Qualified spectral evidence and recoverable cross-model continuation

## Summary

跨 TFG 自然音频增强头继续研发：V4 的完整三前端开发结果和局部谱结构已独立核验，旧输出按可恢复流程回收，新一轮共享候选尚待实际运行。

## Story

用户最初要求恢复 TTS 增强 TFG 的研究路线，后来要求做出适配多种 TFG 的自然音频增强头，并持续推进到方法完备。增强音频只用于驱动生成，评分视频恢复原自然 PCM 与原时钟。用户已授权服务器、百度网盘、数据下载及阿里云 Qwen TTS/ASR，最新消息再次授权 ASR 质量不足时使用 Qwen ASR。密码与 API key 只走运行时；root 维护理论、协议、Basic Memory 和独立核验，实验与模型代码由 GPT-6-LUNA/MAX 子代理执行。

V4 在20 clip/10 speaker ×3 seed、Wav2Lip/MuseTalk/Ditto 的540行九路配对 FULL 支持上，相对 identity 的 Cbest 等权 +0.109，相对 RMS −0.044，三个模型相对 RMS 均为负。完整结果决定了下一轮采用共享前端证据，而不能继续把 Wav2Lip fit-only 收益当作迁移收益。历史 fit360 已选出的三个共享候选及 dev540 计划保持冻结；04CPU/05GPU 尚未运行，确认80 clip/40 speaker与保留模型 LatentSync 仍封存。

本次完成并提交了局部谱结构及资源说明。root 直接核验196个掩膜数组/180字段，并从28段原音独立重算340个关联字段。时间变化 logmask 的共同项 a 占29.658%、谱重分配 r 占70.342%；a 与局部 logRMS 相关 −0.082，与正向 logmag flux 相关 +0.434。最终第四版图的数值与前版一致、视觉核验合格。这些是描述性证据，下一轮需用原音换轨视频与匹配物理帧功率的消融检验局部谱变化门控。

存储遵循 ADR012，C/F双包哈希与F严格全恢复之后才退休具体旧 run。Muse 9644文件已全恢复，旧 run和重复远端tar实际精确退休，最新8个保护路径复核通过。failed-fit03实际完成1,291,222,198字节归档、源输出完好，四连接SFTP正在传回C；F镜像与严格恢复代码已准备但未运行。其后继续处理旧 Ditto 输出，达到 shared540 原定5GiB SHM入场条件。ADR013已创建并读回，固定新增 fit20 speaker的原始320槽位与j=0/15共40端点、5×4 speaker folds及最多6个新候选预算；目前只是元数据准备，未声称40音频已取得，也没有打开确认集。

## Working State

Durable anchors: `openspec/changes/develop-cross-tfg-audio-head-20261003/`、BM Task41/Experiment40、ADR007/009/011/012/013及完整报告25。报告26和最终可用方法尚未完成。当前本地未发布HEAD `0597b3dfbfd6bd5df93c96d7cb70be12ce476e59`，无PR；历史dirty文件保留。

Machine-local state: ignored HANDOFF、scratch、runs与C/F归档需保留。私有PTY50274持有运行时凭据，bridge06端口10887/PID252640提供完整stdout EOF读取；旧bridge05空闲。服务器 root@connect.weste.seetacloud.com:12349，凭据不写文件。实际四连接传输PID388956尚未退出、stderr为空；不能仅凭partial文件大小宣布整包合格。

Ditto已有实际ONNX图元数据审计，输入Sub→Div之后进入Conv/LayerNorm，但输入常量数值尚未解码；正在准备只读常量采集和CPU增益/特征探针。前端笔记v2的Transformers行引用需纠正，900行评分join attempt03已准备待root完整审核；均未作为新的科学结果。

## Changed Files

- `CONTEXT.md`及`openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`：已在当前HEAD提交实际V4结果、局部谱结构、独立核验和工程资源预算。
- `scripts/audio_enhancement_eval/decompose_v4_delta_common_mode_attempt04.py`：子代理最终描述图，数值相等与视觉核验均实际通过。
- `basic-memory/docs/adr/013-canonical-endpoint-subset-for-expanded-historical-fit.md`：固定新增历史fit的规范端点和预算，实际BM创建并读回。
- `.codex/scratch/recoverable_output_failedfit_transfer_attempt03/`：实际四连接SFTP归档恢复；index archive.id有缺少profiles的标签错字，实际路径、源码常量、包哈希和pack收据ID正确，已执行包不修改。

## Verification

V4 joint archive SHA349c5b62f09c1a775013039d2abb3bd72d7c6b6233531fb7f87e6398fff981d4；root恢复1721成员/1700资产/540原矩阵，最大数值差5.72e−6。20个实际Qwen ASR、297字0错误、原音PCM/时钟与视觉支持均合格。

Mask NPZ 46,753,644 B/SHA69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be，原数组最大差2.13e−14、原音关联最大差1.33e−15。最终图/summary/source manifest/rows/receipt五个文件pins核验及root视觉核验通过，见 `.codex/scratch/v4_mask_final04_root_readback_receipt.json`。

Muse run退休收据 SHA087b7278b43caafe81c0ac164d1dac4b7bc64cdabf2ce513f1181b1dc64923fe；tar退休收据 SHAdafc818359b4cc281d441d3b4d704840bd53a52e18aca84903bd14390ca62cc8，实际回读和8个保护路径通过。failed-fit03归档 SHA234006a3cc5efef762f1a943b79529fe4443751fad7f56a76cf718579e888638，pack完成与源保留实际通过，C下载终验/F恢复待执行。

SadTalker archive06 SHA3a2d9f79c072757818f263b135511da4ff81f7bc8c88e3aa5c3698746d83bb83；root260成员/14数组/35帧25fps/自然PCM及10个FULL字段核验，最大差1.91e−6，n1 RMS−identity Cbest −0.044仅为工程资格。不曾运行pytest或新增单元测试。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)

## Observations

- [result] V4完整开发集没有超过RMS控制；可用多模型增强头目标继续，通过共享频谱候选、局部谱变化门控及多前端fit改进推进。
- [decision] 原音换轨后的Cbest保持primary；D0/Canchor/B/offset和独立视觉为控制，不能把logmask范数称为物理功率或将相关描述写成因果。
- [decision] ADR013固定规范端点与speaker folds；候选参数仍待共享dev结果，不接触封存确认PCM。
- [blocker] shared540已stage00–03，04CPU/05GPU待运行；旧输出可恢复回收未完成，SHM仍低于5GiB入场门槛。
- [next_step] 完成failed-fit实际C整包与F严格恢复核验，审计后精确回收旧输出，再启动冻结shared540。

## Relations

- continues [[Codex checkpoint - 2026-10-04T12-00-59Z - Qualified local spectral evidence and continuing portable enhancement]]