---
title: AISHELL-3 解码器微调语料与逐文件恢复核验
type: note
permalink: tts-exp/docs/reference/aishell-3-解码器微调语料与逐文件恢复核验
status: ready
date: '2026-10-25'
tags:
- aishell3
- decoder
- dataset
- verified
---

# AISHELL-3 解码器微调语料与逐文件恢复核验

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

用户授权检查已有AISHELL-3压缩包并抽取5小时16k音频，供CoGenAV特征到波形解码器微调。已完成数据准备和双机读回，未启动训练、编码器前向、TTS或视频生成。

## 数据入口

权威报告 `runs/aishell3_decoder_data_20261025_attempt01/report.md`，同目录保存summary、package_receipt、local_audio_verification、remote_audio_verification、delivery_receipt。Ubuntu数据 `/home/wjj/tts-audio/tts-exp/data/aishell3_decoder_5h_20261025_attempt01/`；Windows完整本机副本 `${USER_HOME}/.codex/tts-exp-artifacts/aishell3-decoder5h-20261025/dataset_attempt01/`，不是百度网盘。入口dataset_config.json、train.jsonl、validation.jsonl；逐文件清单file_manifest.jsonl。

训练40人4202句18101.7313125秒（5小时1分41.7秒），验证另4人429句1810.6秒。全体4631 WAV共637555820bytes，16k单声道PCM16，Soxr precision28；无响度归一化、静音裁剪或额外dither。训练8男32女，验证1男3女，每人取3–15秒完整语句至至少450秒，不截断末句。种子aishell3-decoder-20261025-v1；先划验证，训练/验证说话人ID、规范化中文文本与输出PCM SHA交集均0，不按SyncNet分数选择。训练有192个端点PCM样点，验证0；未做听感/ASR质量审计。

## 原包损坏与逐文件恢复

原始 `./data/aishell3/data_aishell3.tgz` 16885083534bytes，完整读取SHA8a9d7647eb6bfbdbeaa812140829d768f9af77414ca8ef59996503c9f8f9ffad，gzip EOF及tar结构校验FAIL，原件未改。先尝试补齐至镜像19057141777bytes的整包，但下载缓慢，随后改为逐文件独立核验，不能称整包已修复。未完成修复副本17132547470bytes已回收，abandoned_repair_receipt.json保留。

所有4631条原始完整WAV的字节数和SHA-256均与独立镜像shenyunhang/AISHELL-3 LFS一致，固定revision e1af0fa23933585f1844b3c07b71006f86589bab；44目录原始API响应、逐源pin保存在source_mirror_pins/。再次从原包提取时验证完整文件SHA后才重采样。损坏归档整体FAIL与已恢复选定文件PASS必须区分。

**恢复来源是官方test分区**，用于本次独立重建任务并重新按说话人冻结划分；使用后AISHELL-3官方test不再是独立评测集。未加入AISHELL-1下游评测音频，但跨语料真实说话人身份及CoGenAV预训练暴露未核验。

## 核验与实现

新增scripts/prepare_aishell3_decoder_data.py、repair_aishell3_archive.py、build_aishell3_decoder_subset.py、verify_aishell3_decoder_subset.py及对应测试。12测试通过；真实Soxr抗混叠检查通过。两机逐条读取全部4631 WAV，文件/PCM SHA、格式、采样点时钟（误差≤1个16k采样点）、来源pin、完整集合及分集隔离通过。首次独立复核因浮点累加顺序出现时长差异，改为累计整数采样点后通过，未修改数据。

交付tar660776960bytes，SHA c9d3801b4b3332ad5cd1e9cc65757447f87718f25f107541f527a9b7a24dee07；清单SHA6f26ded832f13a206b0df63057dd190dfd71ea18e0a16411687267a6aa66b8f8；Ubuntu4694payload逐SHA通过。封存dataset README早于最终运输，最终状态以run report及delivery_receipt为准。不改已封存数据或清单。

## Observations

- [result] 五小时训练加半小时独立说话人验证数据已可用，双机验证通过。
- [decision] 完整归档仍标记FAIL；仅以逐原始WAV镜像SHA-256一致作为恢复文件准入，不将完整header等同于完整归档。
- [decision] AISHELL-3官方test已用于解码器语料，后续不可冒充独立benchmark；现有自定义验证集保持隔离。
- [next_step] 如用户要求训练，先缓存冻结CoGenAV原始特征，以同源波形作重建目标，先检验未变形重建质量。
