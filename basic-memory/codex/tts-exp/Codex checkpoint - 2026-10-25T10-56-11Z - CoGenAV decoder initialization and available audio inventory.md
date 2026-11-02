---
title: Codex checkpoint - 2026-10-25T10-56-11Z - CoGenAV decoder initialization and
  available audio inventory
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t10-56-11-z-co-gen-av-decoder-initialization-and-available-audio-inventory
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T10:56:11Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-25T10-56-11Z - CoGenAV decoder initialization and available audio inventory

## Summary
冻结CoGenAV特征接预训练波形解码器的路线已调研；Windows与Ubuntu现有数据足够先做5小时试验，尚未启动训练或下载。

## Story
用户希望通过对齐TTS特征到自然时间轴并还原语音，验证TTS原生Sync-C优势能否转移。前序视频重定时未超过自然基线，详见上一交接。现阶段改为研究特征到波形解码器：冻结CoGenAV，用同一音频的原始特征与波形训练重建，不能把TTS特征直接监督成自然波形。推荐以bshall/knn-vc普通重建g_02500000.pt初始化HiFi-GAN，CoGenAV 25fps/768D经可训练适配器转成50fps/1024D。先冻结声码器训练适配器，再低学习率联合微调；WavLM L6可作训练期教师。普通checkpoint优于prematch仅为工程假设，未比较。官方推理hub会移除weight norm，训练应加载原训练结构，重新初始化优化器。精确无损重建不能保证；先检查未变形重建，再做DTW特征变形与下游五臂对照，不能用Sync-C优化解码器。

用户明确将Ubuntu数据也纳入盘点。已核验历史fit-only AISHELL可用候选40人12383句16.351544479小时；80人24622句31.795329271小时包含前者，不能相加，也不能不查暴露直接用于训练。现有TTS400仅约0.463378241小时，域覆盖偏薄，且是否可训练须查split。最新用户问AISHELL-3五小时空间，官方44.1kHz/16bit/mono对应1.5876GB；转16k PCM16为0.576GB，两份共2.1636GB。官方OpenSLR提供19G整包，不能把五小时最终占用误作整包下载所需空间。

## Working State
权威盘点文件：runs/audio_data_inventory200_20261024_attempt01/inventory.json与source_manifest.json；600条200人新验收数据仅1.03458小时，和大池不同，fit/development/confirmation必须保持冻结。远端/home/wjj/tts-audio/tts-exp，SSH wjj@192.168.50.2。80个归档位于data/native_mechanism_holdout_20260926/archives与data/fresh_aishell_disjoint_ref_target_20260929/archives。
最近读数Ubuntu约83G可用；Windows E仅1021386752bytes、C约47GB。Windows data/aishell3/data_aishell3.tgz文件16885083534bytes已存在，但未校验完整性，不计为已可用。TED-LIUM32个parquet总15915260170bytes，本轮未解码计算时长。双机镜像不重复计数。另一个chat的audio200 production/input24是24k FLOAT自然参考输入，不是TTS；不可将wave库不支持FLOAT误判为损坏，也不干预其工作。

## Changed Files
本轮研究与盘点未改代码、未提交推送、未运行训练。本次唯一新增为此不可变交接。工作树存在大量继承修改和未跟踪文件；本轮git status再次确认dirty，切勿恢复其他chat改动。

## Verification
前一盘点实际核对80/80归档大小与既有清单一致，并逐归档读取首条WAV，80/80 mono16k有效；没有重新全量SHA或解码全体语句。时长合计来自已有逐句manifest。实际检查自然400与TTS400文件头均无错误，Windows镜像不重复计数。remote input24用soundfile成功读，stdlib wave初始FLOAT失败是格式兼容性。无模型训练测试结果。
当前重新读取Git根、branch、SHA与status；GitHub REST确认canonical仓库且当前分支open PR为0。SHA只核实本地，不构造远程commit链接。上一checkpoint全文元数据确认agent/session_id匹配；同名异chat不接续。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [kNN-VC training and checkpoints](https://github.com/bshall/knn-vc)
- [AISHELL-3 official download](https://www.openslr.org/93/)
- [AISHELL-3 official audio specification](https://aishell-3.oss-cn-beijing.aliyuncs.com/AISHELL-3%20ReadMe.pdf)

## Observations
- [result] 已有历史fit候选16.35小时，足够先选5小时验证预训练解码器微调；数据量足够不等于CoGenAV特征保留全部声学细节。
- [decision] 优先复用现有数据；新训练前锁定说话人与暴露边界，先证明重建可用，避免自然/TTS声码器效应混淆。
- [decision] 下载空间与最终抽样音频空间分开计算；AISHELL-3本机已有未验证归档，可先核验而非重复下载。
- [next_step] 回答五小时AISHELL-3容量问题；如用户要求实施，先校验已有归档与划定不重叠训练/验证集，再准备解码器先导。

## Relations
- continues [[Codex checkpoint - 2026-10-25T10-26-26Z - CoGenAV video retiming concluded and decoder scope clarified]]
