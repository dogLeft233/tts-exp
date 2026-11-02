---
title: CoGenAV 特征波形解码器五小时微调先导
type: task
permalink: tts-exp/tasks/co-gen-av-特征波形解码器五小时微调先导
status: completed
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
tags:
- cogenav
- decoder
- training
result: LOW_FIDELITY_CONTENT_RETAINED
---

# CoGenAV 特征波形解码器五小时微调先导

## 目标与授权

用户明确要求用已准备的五小时AISHELL-3按推荐方案微调波形解码器，检查CoGenAV音频特征重建质量。完成范围为自然音频同源重建先导：冻结CoGenAV、适配器+普通预训练HiFi-GAN微调、验证音频和独立质量评估；没有授权自动扩展DTW/TTS/视频实验。

## 固定执行

Ubuntu run `/home/wjj/tts-audio/tts-exp/runs/cogenav_decoder_pilot_20261025_attempt01`。训练4202句/40人18101.731秒，验证429句/4人1810.6秒。源数据见关联指南，来自AISHELL-3官方test重新自定义划分，不能再冒充官方test独立泛化。

官方CoGenAV Base raw25Hz768D frozen -> 训练集统计的可逆逐维标准化 ->512通道3个残差时序卷积 ->2倍插值 ->1024D ->官方bshall/knn-vc regular HiFi-GAN。G权重g_02500000.pt SHA f98b760e0e5fd0019cffd3a9d22bab5c4c2fe38491532d7680a8ad07eaf3e8dd，官方releasev0.1 asset98519080，严格state加载通过。WavLM-Large L6仅作同源教师；CoGenAV/WavLM不训练。不用SyncNet，不加本轮GAN/featurematching微调。

有效协议protocol_revision03.json SHA d50671c24c7da3020ae279052d4dfa1c25b18be1be00d4ab1141589913a76c75，schedule_revision03.json冻结2000适配器热身+4000联合更新，batch4/accum2，7040样点loss中心+每侧16个40ms上下文。热身G冻结，teacherMSE主损失并每4步小权重waveformloss；联合adapterLR1e-4/G1e-5，0.1teacherMSE+10logMel+MRSTFT。原protocol、smoke、schedule保留，不覆盖。训练真实执行source_snapshots/train；主要PID290480/wrapper290473，21,600秒硬上限。7.5GiB分配上限，与另一chat音频200生成共享GPU，不控制其他进程。

切片上下文12实际支持不足，正式训练前修正16；FP64 actualAdapter+G full-vs-crop误差~1e-14；FP32约2.7e-5绝对/3.7e-6相对舍入。全长推理先左右复制16个原始CoGen特征，再在16*640处裁回原始采样点数，从而不排除首尾语音loss中心。直接WavLM对照仍native50Hz、不改时间轨迹，缺尾≤3帧复制并从teacherMSE掩掉，不叫完美oracle。

## 评估与复核

16条预先固定、每验证人4条的monitor只按logMel选best；最终生成全部429条。父侧固定另40条每人10条、不含monitor，比较original/untrained/WavLMdirect/CoGenfinal，PESQ-WB/STOI/ESTOI/谱差/RMS及离线Paraformer CER，无人评结论。parent_quality_selection.json SHA526d348f35b1eb642e371653510c3dfa2d86edeb284e4d146ee190831dc2e787。parent_quality_driver.py已在本任务运行，等infer_exit0后自动CPU评估，状态parent_quality_driver_status.json；root owns最终报告/归档，worker不封存。

父侧4631缓存逐SHA、形状、原生时钟及教师尾mask通过；454535训练帧统计均值/方差精确重算一致，未混入验证；6条官方encoder重新前向与缓存逐值完全一致。identity/0dBnoise校准PESQ/STOI/谱差通过。原始ASR在两种交错顺序出现9与7编辑，已保留旧输出、在最终CoGen结果出现前v2按utterance固定随机状态，单句交错重复校准通过，未改模型或ASR前端配置。最终只报v2同条件结果。

## 完成要求与下一步
## 完成结果

✅ 已完成，主训练6000更新和全部429验证重建真实exit0；父侧固定40质量评估与全量独立核验通过。CER原音1.53%、CoGen3.07%，但CoGen PESQ1.116/STOI0.665、RMS−6.525dB，结论为内容保留较好而波形低保真。最佳joint4000/global6000，模型SHA1127aa8ed9f33ce25562df2cfc8d599cd49eea726d57b9e6ed6eae8094f9ac65。报告runs/cogenav_decoder_pilot_20261025_attempt01/report.md；完整结果见 [[CoGenAV 五小时解码器微调的重建质量 2026-10-25]]。

独立teacher-free CLI重放逐点exact；C本机6.514GB归档5313payload逐SHA读回通过，归档SHA824e606263ac8c6d52451e08f6d21c3098e968d8756a60d0f47826f0103f2761。本次所有计算已结束，未追加优化、TTS/DTW/视频实验。后续只在用户新要求下开展保真改进或诊断。

## Observations

- [decision] 先做没有对抗微调的预训练G重建先导，结果不等同于CoGenAV特征可逆性上限。
- [next_step] 本次已完成，无在途计算；如用户继续，先诊断并改进未变形重建保真，再考虑TTS或时间对齐。

## Relations

- uses [[AISHELL-3 解码器微调语料与逐文件恢复核验]]
