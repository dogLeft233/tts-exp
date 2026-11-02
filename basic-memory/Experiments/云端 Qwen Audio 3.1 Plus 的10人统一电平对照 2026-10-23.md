---
title: 云端 Qwen Audio 3.1 Plus 的10人统一电平对照 2026-10-23
type: note
permalink: tts-exp/experiments/qwen-audio31-cloud-subset10-2026-10-23
status: concluded
scientific_status: POSITIVE_CLOUD_PROVIDER_PILOT_AT_FIXED_RMS
execution_status: COMPLETE_30_NEW_50_REUSED_SCORES_INDEPENDENT_AUDIT_AND_LOCAL_ARCHIVE_READBACK_PASS
hypothesis: 相同来源、RMS、脸、Wav2Lip与官方评分下，云端Qwen是否有不同于本地0.6B的分差
result: cloudT−N +0.562，97.5%[+0.171,+0.903]；cloudT−localT +0.809，[+0.351,+1.281]
conclusion: 固定当前下游时云端提供器仍有正评分差；不能将本地未确认推广为所有TTS无优势或归一化必然消除优势
inputs: existing40固定10位来源、同自然参考与文本、50个SHA绑定官方基线评分
outputs: runs/qwen_audio31_cloud_subset10_20261023_attempt01
report: runs/qwen_audio31_cloud_subset10_20261023_attempt01/report.md
model: 用户指定qwen-audio-3.1-tts-plus自克隆；Wav2Lip GAN；SyncNet V2当前官方完整链
branch: codex/cross-tfg-audio-head-20261003
base_commit: 2b71639c3b555293375d248350999377168d515d
code_paths:
- scripts/experiments/qwen_audio31_cloud_subset10_20261023.py
- scripts/experiments/nrt_two_arm_current_pipeline_20261023/downstream.py
- scripts/experiments/nrt_two_arm_analysis_20261023.py
- scripts/experiments/nrt_two_arm_audit_20261023.py
tags:
- tts
- cloud-qwen
- amplitude
- official-pipeline
- no-training
---

# 云端 Qwen Audio 3.1 Plus 的10人统一电平对照

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

**同一批10位来源、同一固定RMS/参考脸/Wav2Lip与官方评价下，云端Qwen的Sync-C高于自然组，也高于本地0.6B组。** 当前下游并不必然使TTS优势消失；尚未隔离模型大小、提示策略、参考处理或解码中的具体原因。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-10-23
- Verification Status: ANALYZED；实际合成/生成/评分完成，保存距离与主要统计独立CPU回算PASS；没有另一轮随机TTS再生成复现。
- Version Label: qwen_audio31_cloud_subset10_20261023_attempt01

Basic Memory MCP不可用；按Startup Router和实验规范本地写入读回，不声明图谱索引已更新。代码为工作区新改动，以run源码快照/SHA为执行证据，base_commit不代表已包含这些改动。

## 问题与冻结设计

原40人本地0.6B扩样及官方复评分均未确认稳定T−N优势。用户要求当前40条取小子集换云端Qwen，同时另做历史16人当前流程实验；本笔记只记录已完成的云端组。

在云端结果出现前，从参考≥5秒的15人中按原排序用`round(k*14/9)`固定10人：S0002/S0007/S0011/S0015/S0016/S0023/S0025/S0030/S0031/S0039。每人用同一自然录音复刻自己的音色，同一固定文本合成3次；没有系统音色替代、试听或按分数挑选。该子集含旧pilot且本地结果已知，属于探索性比较，不是未见样本确认。

旧VC模型首合成404 “Model not exist”，0输出；注册一次成功的工程尝试完整保留。用户随后明确指定`qwen-audio-3.1-tts-plus`，新attempt采用`voice-enrollment/create_voice`及HTTP `SpeechSynthesizer`，关闭参考预处理/音量规整，请求24kHz WAV，无风格/语速指令。10次复刻+30次合成均HTTP200，99.253秒，无重试。key只走内存/SSH stdin，未写代码、命令或文件，结束清除内存变量。

云端输出按原流程转16k FLOAT、以float32增益归一到RMS0.0376838172675746、保存mono PCM16，无时间编辑。固定`data/data/image/3.png`、同Wav2Lip GAN权重、batch4/nosmooth/25fps。50个N/R/localT官方评分逐SHA复用，只新生成/评价30云端格。

官方链保留当前run_pipeline→run_syncnet/min_track25、内部AVI/MP3及offset，第一打印track为主，所有track保留。MKV入口画面/PTS/PCM全exact验证；规范PCM入口不等于内部无损。generic官方protocol.order仍沿用旧N/R/T文字，实际roles/input_manifest/执行顺序均C1/C2/C3，说明见implementation_notes.json；不改冻结代码或凭据。

## 结果与含义

每人先平均3调用，再按10位来源配对bootstrap10000次，PCG64 seed20261023；两项主C比较使用97.5%区间。全部draw，包括重复云端文件与本地无EOS长尾，保留主分析。

| 比较 | ΔSync-C | 97.5%区间 | 提高人数 |
|---|---:|---|---:|
| cloudT−N（主） | +0.562 | [+0.171,+0.903] | 8/10 |
| cloudT−localT（主） | +0.809 | [+0.351,+1.281] | 9/10 |
| cloudT−R（次） | +0.773 | [+0.425,+1.114] | 9/10 |
| localT−N（次） | −0.247 | [−0.936,+0.336] | 6/10 |

C均值N7.108、R6.898、本地T6.861、云端T7.670。次要D的云端−自然−0.260，95%[−0.446,−0.057]；云端−本地−0.101，[−0.351,+0.186]未确认，不能称两个指标均改善。

30调用只有15份不同provider字节；各人不同文件数1/1/2/2/1/2/2/1/2/1，重复完整保留，不追加重抽、不当30名独立说话人。时长4.16–7.76秒、相对自然比0.693–1.046，未作删样。可能是托管解码或缓存，未验证原因。

沿用既有EOS=true、时长比[0.5,2]资格，28/30本地draw合格。处理后描述性补充：云端−合格本地+0.571，95%[+0.308,+0.817]；合格本地−自然−0.009，[−0.477,+0.388]。云端正差不只由两条本地停止异常决定；该敏感性不替代全draw主结果，也没有额外因果资格。

**不能再概括为“归一化或这张脸使所有TTS无法涨分”。** 同样设置下，TTS来源切换有明确差异；提供器与脸/音量的交互仍可能存在。托管权重、大小、训练集、参考提示和内部解码未受控，不能称纯规模或纯LM任务效应，也不能计算历史优势的原因占比。没有人类同步盲评、独立嘴部准确度或训练收益证据。11/11统计谬误核查与逐人结果见完整报告。

## 工程核验与保存

render实际exit0（300.777秒）、official实际exit0（381.943秒），总682.738秒，GPU释放后才启动历史组，无训练/重叠。30新评分+50复用共80格齐全。独立脚本核验80组媒体/PCM、全部parity/PTS/逐帧hash、阶段日志退出码、保存距离/打印轨道与15项主要统计，PASS，最大主要统计误差1.78e−15。26个源码/模型pin、30个参考输入绑定PASS；22项源码测试通过；统计图实际目视PASS。

- 报告：`./runs/qwen_audio31_cloud_subset10_20261023_attempt01/report.md`；E仅1545元数据/日志/图文件，12839421字节。
- Ubuntu完整run同名目录；封存manifest SHA `0b8c0b4dda0b4890b614ad0137a5315f86f9c0142d07e1d02e8e0dbd92092aec`。
- C完整tar：`${USER_HOME}/.codex/tts-exp-artifacts/nrt-two-arm-pilot-20261023/qwen_audio31_cloud_subset10_20261023_attempt01_full.tar.gz`，335048651字节，SHA `824c09315796ca551fb9e9f0e59ea062db98322b2e6c96a7a6e830c3f52f1ec0`。
- 10444完整成员实际读出逐SHA，743 JSON/3 JSONL解析PASS；E1545文件SHA通过，不声称E展开全部媒体。
- 外置读回凭据：同C目录`qwen_audio31_cloud_subset10_20261023_attempt01_local_readback_receipt.json`，UTC2026-10-23T04:22:47。portable_path_map保留原执行路径→归档副本映射，不改原凭据；未声明网盘同步。
- analysis协议SHA `87861cac14ee5723de6c5f4902cc2888516afd955454d7fedd3a7e8e5cc63507`；执行协议SHA `094a47362c3e719180c6aaa34ebbdfdda314917e030bbfd591f457b9f85fce50`。

## Observations

- [status] CONCLUDED；云端合成、30新视频/评分、80格独立主回算和完整本机归档读回完成，无训练。
- [conclusion] 固定当前RMS/脸/评价链时云端提供器仍有正Sync-C差；本地未确认结果不能外推所有TTS。
- [limitation] 10人已分析来源的探索性子集，模型/参考/解码联合变化，没有纯任务或人类准确度证明。
- [decision] 保留全部draw、云端重复和本地异常；资格分析仅补充，不自动扩样或启动训练。

## Relations

- relates_to [[历史 AISHELL16人采用当前流程的复核 2026-10-23]]
- follows [[完整 Qwen TTS 的40人官方流程复评分 2026-10-23]]
- follows [[完整 Qwen TTS 的40人扩样与生成可靠性复核 2026-10-22]]
- relates_to [[历史TTS优势与40人NRT结果的协议差异核查 2026-10-22]]
- updates [[研究交接总览]]

## Changelog

- 2026-10-23：按用户指定模型完成云端10人自克隆对照，保存前驱接口失败、全draw结果与完整本机归档。
