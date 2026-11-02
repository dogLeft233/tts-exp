---
title: Codex checkpoint - 2026-10-18T12-17-20Z - Mainstream encoder commonality completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t12-17-20-z-mainstream-encoder-commonality-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 12:17:20.821941+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T12-17-20Z - Mainstream encoder commonality completed

## Summary
ROOT直接完成11个主流语音编码器的同队列共性扩展，统一RMS后TTS平均方向仍在多数预训练表示中保留，电平耦合和局部响应具有明确编码器边界。

## Story
项目从TTS原生高分转向编码输入如何影响生成。旧报告46的幅度验证、Wav2Lip方向补算与留出生成已经完成；本轮用户要求将主流编码器都下载测一遍，明确不用子代理，空间可自行处理。因此ROOT编写并运行固定CAL20/10男性每人两句的几何分析，保留官方原生前处理，分别测RAW TTS、双方FIXED TTS、自然FIXED±3dB。整个扩展没有新TFG/SyncNet或训练，也没有恢复暂停的通用增强头。

11个新checkpoint覆盖HuBERT英base/large/中文base、wav2vec2 base/XLS-R53、WavLM base-plus/large、Whisper base/small、data2vec audio base、W2v-BERT2；每个121正式前向，合计1331。另复用120个已验证Wav2Lip latent，0新forward。共同663对音素、中央半段516、20句全保留。所有层native支持统计保存，跨模型共同支持表按预定0/1/2/middle/last层；均值方向、LOSO与局部音素对轴一致性分开。

末层统一RMS的TTS跨人cos在11预训练编码器0.165–0.506，Wav2Lip0.015（RAW0.329）。纯+3dB在Wav2Lip/WavLM base-plus/Whisper base/small变化RMS约0.0721/0.0350/0.0275/0.0311，跨人cos0.884/0.479/0.613/0.577；若干归一化链响应只有1e-6–5e-5量级。预训练表示的平均LOSO0.327–0.698，而局部音素对共同轴仅0.068–0.268。原始TTS轴对增益轴cos在Wav2Lip0.933，另外三条强增益链仅0.105–0.177。结论是模型/前处理依赖的共同平均分量，不是所有音素同一方向、跨模型同坐标或生成收益。

## Working State
本轮已完成；Task《主流语音编码器的方向共性扩展》done，正式报告52及地图/索引更新。所有本机科学worker已退出，GPU回到约2701MiB原桌面占用；Ubuntu其他会话训练未被干预。旧head/P2仍暂停或独立待办，旧goal工具状态不改。

持久实际根F:/Datasets/tts-exp/encoder_survey_20261018；轻量镜像runs/encoder_survey_20261018。results SHA a66111a83714b9279f0696d86ce0502f5eac4f1aa9a6ba254a8d6027834f4a6e；artifact_manifest SHA b27737489d0c917dd774c514acfbd6f8b183eb927b0b446e97714c2ce5ea9839。全部240音素NPZ、首句末层原数组、input/protocol、每模型receipt/官方revision/配置/SHA和三版producer正文保存。本机F盘持久文件不是云备份；新结果尚未云同步。公共权重body已回收，可以按固定revision重新下载。

W2v-BERT原生attention mask的padding位置为零向量，首次在首句重复检查失败；依据确切调用路径有2个已计算前向，没有完成clip NPZ。原receipt/log在failures/w2vbert2_attempt01，原batch_status rc1保持。仅修正原生无效mask支持后单跑该模型121forward，其他10不重跑，科学阈值/clock/PCM未改；最终看recovery_status和新receipt。失败2次与正式1331次分开记录，不能抹去失败或重launch已完成任务。

空间处理仅回收副本/公共缓存：May本地九传输parts与完整打包副本回收，删除前重新核对全部643展开科学成员/6,049,784,599 bytes body/bytes/SHA及整包SHA，均通过；展开成员和此前独立云读回九分片均保留。当前恢复位置见《编码器扩展的执行环境与资产存储》。NTFS透明压缩不改逻辑字节；pip下载缓存清理不改安装环境。F结束空闲约5.50GB、C约2.14GB。没有改CloudDrive/sudoers或旧拒绝删除对象。

## Changed Files
- 新增未跟踪scripts/experiments/encoder_survey_20261018/{prepare,run,cache_reference,analyze,verify,plot,report}.py。
- 新正式报告basic-memory/docs/experiments/52-encoder-direction-commonality.md、两张图及SVG/PDF资产；新BM运行/存储参考和完成Task。
- 维护Research/研究主线与证据地图、实验README、CONTEXT99行、HANDOFF36行。完成任务不留未完成看板。
- 当前HEAD仍为本地实查2b71639c3b555293375d248350999377168d515d，本轮未提交/push；九项继承tracked dirty及其他untracked保持。无本轮PR。

## Verification
240个实际NPZ和448组共同支持跨人/LOSO/局部音素统计由不导入producer的NumPy消费者重建；方向范数最大误差2.895e-8、公式最大误差4.441e-16。11首句真实RAW重复控制通过，官方模型无missing/mismatched encoder参数，权重body实际SHA/可用官方LFS SHA匹配。静态两图已目视，无裁切或数值替换。前四producer后引入low_cpu_mem_usage仅减小加载内存，第三版修复native mask，实际各版本source均保留。未运行pytest/实现测试、感知评价、新视频或训练。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，本轮结束只读GitHub查询当前head无PR；本地HEAD未构造未经核验的远端commit链接。
- basic-memory/docs/experiments/52-encoder-direction-commonality.md
- runs/encoder_survey_20261018/{results.json,table.csv,qa.json,artifact_manifest.json}

## Observations
- [decision] 用户明确本轮不用子代理，ROOT直接执行；规模固定11模型/同一旧CAL20，表示分析完成后不自动扩生成。
- [result] 统一RMS后TTS共同分量具有编码器边界，Wav2Lip的强电平混淆不能外推为所有语音表示的规律。
- [limitation] 10位已暴露男性、单Qwen、无独立自然重录；方向几何不能识别合成特有原因或感知同步收益。
- [next_step] 从报告52讨论接收端坐标与小型生成因果验证的选型，等待用户新的研究范围；不重跑本轮或恢复暂停head。

## Relations
- continues [[Codex checkpoint - 2026-10-18T11-49-00Z - Completed amplitude evidence and authorized broad encoder survey]]
- relates_to [[52-encoder-direction-commonality]]
- relates_to [[主流语音编码器的方向共性扩展]]
- relates_to [[编码器扩展的执行环境与资产存储]]
- relates_to [[研究主线与证据地图]]