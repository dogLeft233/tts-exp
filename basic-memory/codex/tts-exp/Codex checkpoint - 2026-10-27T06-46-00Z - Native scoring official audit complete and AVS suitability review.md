---
title: Codex checkpoint - 2026-10-27T06-46-00Z - Native scoring official audit complete
  and AVS suitability review
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-27-t06-46-00-z-native-scoring-official-audit-complete-and-avs-suitability-review
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-27T06:46:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-27T06-46-00Z - Native scoring official audit complete and AVS suitability review

## Summary
云端60条原生评分和官方推理审计完成；SyncReward FP32没有平均TTS优势，但BF16只复测时移控制，当前正在调研AVS_u/m/v是否更适合本项目。

## Story
用户持续要求区分TTS的Sync-C优势与真实口型优势，已完成云端Qwen Audio3.1Plus同20来源三次调用、60原生视频四指标比较，并按用户要求停止交叉音轨任务。权威原生run为 runs/cloud_av_preference20_20261027_attempt05/，官方审计run为 runs/av_metric_official_audit_20261027_attempt01/。前一排队checkpoint已过时，不应恢复其执行器。
原生Sync-C ΔT−N=+0.745160、48/60 T更高；CoGenAV Eq1 −0.0008358、31/60；SyncReward FP32 −0.0195223、19/60；PEAVS原batch1 −0.403537、6/60。20来源为独立单位，60draw仅21唯一provider字节；12严格单句参考与8扩展克隆参考分层解释。SyncReward描述区间跨0，不能据此证明自然更好或否认感知优势。
审计确认媒体时间偏移正确、原代码和权重匹配。PEAVS上游dataset把视频长度误用音频长度，造成padding和伴随批次依赖；官方默认batch64对本次cloud偏好方向无翻转，更新Δ为−0.4080848。SyncReward历史FP32与官方FP32入口三个anchor精确一致；V100虽然没有原生BF16硬件却可执行Torch BF16模拟，官方BF16实测完成。BF16±400ms仅14/40下降、7平、19上升，PE官方64为25/40下降，均未达32/40门，仍CONTROL_FAILED。不能把两者当同步真值。
用户最新要求仔细读AVS_u/m/v论文判断更合适性。已读取原文2405.04327v1与AV-HuBERT官方仓库，research skill显式要求后台调研，因此派发 avs_paper_review，写reference研究文件；尚未下载权重、部署或评分。

## Working State
Windows唯一入口 E:/Documents/tts-exp；Ubuntu有效repo /home/wjj/tts-audio/tts-exp，ssh wjj@192.168.50.2。工作树继承大量dirty/untracked，本轮未提交推送。用户暂停的生产PID320414保持暂停；交叉runner1638959已按要求终止，不恢复。ComfyUI后来外部重启，勿套用先前关闭授权。当前GPU状态未重查。
完整审计本机C备份 C:/Users/dogle/.codex/tts-exp-artifacts/av-metric-official-audit-20261027/audit_evidence.tar.gz，36148970bytes，SHA79715c0c65db09b7dc9787bd8a871c799f4b7fd90ffa02973b12edf0b0e1e96a，431成员逐个读回核验。C不是百度网盘。E仅小报告/元数据，保留>1GiB。
原生完整60 cloud尚未用BF16重评，31.7%与−0.019522是FP32结果，不可标为官方BF16确认。

## Changed Files
未跟踪 scripts/experiments/av_metric_official_media_audit_20261027.py、syncreward_official_parity_20261027.py、peavs_official_parity_20261027.py、peavs_batch_dependency_20261027.py、syncreward_bf16_controls_20261027.py、av_metric_official_audit_report_20261027.py；版本化入口score_peavs_official64_20261027.py与score_syncreward_official_20261027.py，旧入口未覆盖；tests/test_av_metric_official_audit_20261027.py。已同步Ubuntu，新入口真实smoke通过。官方模型源码未改。PEAVS与SyncReward部署笔记及云端结果记忆加入审计更正。

## Verification
前序已实际完成40媒体逐PCM/帧/PTS、AAC后时移、官方source/weights哈希、PE最小2样本批次反例、FP32原入口精确对照、BF16全63探针和两个新入口GPUsmoke；5实际产物集成函数直接PASS及py_compile，未pytest。完整审计备份430payload加manifest读回通过。
本次实际读git root/branch/SHA与dirty状态，gh不可用，公开REST确认canonical仓库及当前分支open PR为空；本地SHA远端未验证，不构造commit链接。新调研尚未跑科学实验。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 审计主报告 runs/av_metric_official_audit_20261027_attempt01/official_inference_audit_report_20261027.md
- [AVS原论文](https://arxiv.org/html/2405.04327v1)
- [AV-HuBERT官方权重入口](https://github.com/facebookresearch/av_hubert)

## Observations
- [result] 官方设置修正没有使PEAVS或SyncReward通过本地已知时移控制。
- [decision] FP32云端原生比较与BF16控制分开解释；不得宣称已完成BF16全原生重评。
- [decision] 用户取消交叉音轨，冻结旧评分历史，使用版本化新入口。
- [blocker] AVS原文checkpoint精确版本、作者独立评分实现及中文域可靠性尚待确认。
- [next_step] 完成AVS论文及官方实现核对，说明AVS_u可无参考评估，AVS_m/v的真实视频参考约束，并给出本项目适用性与验证顺序。

## Relations
- continues [[Codex checkpoint - 2026-10-27T03-14-44Z - Cloud Qwen20 corpus complete and AV scoring queued]]
- relates_to [[PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27]]
- relates_to [[云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27]]
