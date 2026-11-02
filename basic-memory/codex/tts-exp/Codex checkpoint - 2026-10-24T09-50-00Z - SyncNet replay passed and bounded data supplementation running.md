---
title: Codex checkpoint - 2026-10-24T09-50-00Z - SyncNet replay passed and bounded
  data supplementation running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-24-t09-50-00-z-sync-net-replay-passed-and-bounded-data-supplementation-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-24T09:50:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-24T09-50-00Z - SyncNet replay passed and bounded data supplementation running

## Summary
旧20人的140条官方特征重放和窗口算术已通过独立复核；按用户条件授权，60条缺失1.7B媒体与160位新说话人的自然录音正在补齐，全部训练保持关闭。

## Story
研究要解释自然音频、0.6B与1.7B在相同电平下的SyncNet偏好，而非把高分直接说成人类口型更好。用户从WIMHF音频-only设计推进到执行：先让子代理验证现有20人的官方流程、窗口和特征，再盘点200人语料与存储，通过后补数据，暂不训练。执行采用实际官方crop、整数PCM/MFCC、最终Linear前512维和完整31lag曲线重放；父代理独立计算避免由同一runner自证。完整RF和FA中心映射保留，未先整句avg pool/PCA。

140条的官方evaluate-forward重放完成，所有31lag float32曲线与旧缓存完全一致。这里复用旧官方人脸轨迹，未重新从原视频运行检测。父侧独立NumPy评分、512到最终Linear和全部窗口索引通过。时间戳算术通过不证明FA语义真值；共同支持19/20来源，一条S0006 L06d2仅75%覆盖；两个旧无EOS长尾末字保留异常标志。

物理库存98人不等于可纳入98人。旧项目封存/历史使用者必须排除，扩样改为原40人加160新人；200人每人3句共600自然来源，规划100fit/40dev/60confirm。新160都是公开train分区、不在项目历史身份名单。全4200媒体仍受语料最终资格和约57GB活动盘预算限制；目前只放行老40缺的60条L17媒体，以及600条自然录音准备。不会因为已有局部通过就放行训练或全量视频生成。

## Working State
权威结果在 runs/syncnet_audio_replay20_20261024_attempt01/report.md、final_status.json、final_manifest.json；final manifest SHA f42bcf61367ee1f7dd840ebded29c4b5a7b35783acba55aa949cf77d8d14a987，597文件194280020字节。小凭据在本机，完整缓存仍远端 /home/wjj/tts-audio/tts-exp。SSH wjj@192.168.50.2；本机唯一入口 E:/Documents/tts-exp，旧本机路径退役，远端同名旧路径仍合法。

运行中且不可重复启动：
- /root/audio_data_supplement_prepare 是唯一GPU持有者。runs/audio_data_supplement_20261024_attempt04，远端executor PID631862（其tool session68785）；截至最近子代理回报21/60已官方评分、0技术失败。protocol SHA37f5a5fc14273cb5f0ad6240131f946871aa469afcb4149d5bfc365ef37e05ca，source SHA745526171fd398d914fd64c21674cb55bfcb4733d6f738b05f49ee892c666465。parent_release.json仅授权missing60_L17_only。保留原FP32/eager/self-ICL、seed、文本、参考、归一化和Wav2Lip官方评价；每阶段磁盘至少12GiB+122779452B。前attempt02被系统桌面GPU进程挡住、未产媒体，旧失败保留；仅允许精确路径 /usr/libexec/gnome-remote-desktop-daemon 后另立attempt04，未杀进程或放开其他GPU作业。
- /root/audio_data_inventory200 CPU收集：runs/audio_data_inventory200_20261024_attempt04，Windows PID168820，contract SHAdd9fa68b322c2f51c8309393626ebc8757458a538b0abc309951d9a102dad6d9，collector SHA11408aa581b650ba1b5dcac21b3df35c91f8418691bcb8bdf7bc050e9ba6d85c。旧40人120原WAV和pilot2人6WAV已选定、源SHA/SSH raw与PCM SHA通过，剩158包开始S0088。逐包最多54.7MB F自有scratch；完整包SHA后选3、SSH读回后仅清自己新临时包，保留选定WAV；E>=1GiB、F>=1GiB+包、远端同12GiB+123MB门。160包总6.239GB；禁止TTS和评分。公开镜像 ModelEvolution/AISHELL_1固定81d91b4a613e945238889b0235e0c5ca5c46161e；100重合包与原发布者逐SHA相同，但不称其余300已官方独立验证。两pilot的各1WAV与第二独立上传者逐字节相同。
- /root/syncnet_preference_probe 当前已结束CPU归档预检，可followup继续40人cache消费。runs/syncnet_audio_dataset40_20261024_attempt01准备280行：140旧特征复用、80旧媒体待前向、60新媒体待producer；CPU FA80已完成，累计220FA可复用。source03支持producer最终consumer_manifest及 --fa-index。等待60producer完成并释放GPU，再新attempt刷新实际manifest，核对140新feature/60新FA名单并由父侧放行；不能现在抢GPU。

candidate200_speaker_plan.json在inventory attempt02，SHA03bb9bdb2abbcdae1380142c082cd227b16f3a1852c4c7e8a0196a4c482ad41c，父侧核验200唯一、100/40/60、160新人、包字节数。split尚未最终可用冻结，必须查600源、原文近重复>=.90、PCM重复、自克隆参考泄漏和每人3句资格；低资格按预声明无标签规则替换，禁止看模型分数换人。

小归档 runs/audio_data_archive_preflight_20261024_attempt01 已完成：501760B TAR、3自然WAV，Ubuntu CloudFS写入、Windows B全SHA与反向恢复全通过。archiveSHA32c3dbb9b96d315a38c75a4f3c3cc79953ba35b215fc17828ca8949084357778；manifest SHA6432bbf584cc773f3ea246293bc0fe75df0b12c3f6868fec2e664cfec4f05f1c。Cloud路径 B:/tts-exp/tts-exp-archives/audio-preference-data-20261024/storage-preflight-attempt01。provider durable flush、真实spool位置/独立backend cache仍未证明；不可据此删除媒体或放行4200。建议后续首轮<=512MiB分片、<=280逻辑媒体、双host恢复验收，仍需解决持久化凭据。

本机Git有1204状态条目，大量继承修改/未跟踪，未清理或提交。分支和SHA本轮只读确认。GitHub仓库身份确认、当前head PR查询为空；SHA只作本地标识（此前远端解析422），未构造commit链接。进程PID和远端可用性为易失状态，恢复先检查，不重复启动。

## Changed Files
- scripts/experiments/syncnet_audio_replay20_parent_audit_20261024.py 与 syncnet_window20_parent_audit_20261024.py：父侧独立CPU审计；对应 tests/test_syncnet_audio_replay20_parent_audit_20261024.py。
- 子代理 scripts/experiments/audio_data_supplement_20261024.py、syncnet_audio_dataset_cache_20261024.py 及对应测试；采集器与冻结源在各run中。父侧不得覆盖子代理在写的源。
- BM规范方案第12–14节已加入200人计划、实际前端/窗口gate、执行授权但training_started=false；新技术核验实验笔记和地图已保存。技术报告后续应补充“原98仅物理库存、实际40+160”的资格修订和最终实际补数，避免写进度日志。

## Verification
父侧 parent_independent_replay_audit_attempt02.json PASS：140输入/原来源/track/score SHA、PCM→MFCC exact、512→官方Linear最大误差7.152557e-7，独立NumPy31lag最大误差5.722046e-6，np canonical C exact；torch/np reduction差7.629395e-6保留而不改标签。parent_independent_window_audit.json PASS_INDEX_ARITHMETIC_ONLY：16612窗口、2247锚点、6741偏移变体。实际远端unittest 2项PASS，包含全cache集成与source-vs-bound manifest回归。

首次父审计误把协议input_manifest_sha当成增强manifest；源码freeze确实绑定旧probe源。旧source SHA f372f3efc0f5bc28893946c162fe867f1ec574cc10142073f41610f9f6fdd753，新bound SHA c9e2398a9ccaad22c683aeaa08b0c9a7f4b2ccdcad60c66ec132690d4fb5e701。140继承行一致；保留attempt01失败、用identity_overlay_attempt02澄清，未重写科学协议或降低容差。final status仅技术重放与索引通过，不是FA真值验证。

子代理报告：producer9项准备测试PASS；自然采集5库测试+4真实CPU合同测试PASS，父侧未重跑全部这些测试。CPU FA80来源资源名单及prepared SHA由父侧核验后放行，80成功；尾字2.75–2.85s长区间保留待审。小归档跨host结果为子代理执行回报，文件本机已同步。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/syncnet_audio_replay20_20261024_attempt01/report.md
- runs/audio_data_supplement_20261024_attempt04/parent_release.json
- runs/audio_data_inventory200_20261024_attempt02/candidate200_speaker_plan.json
- runs/audio_data_inventory200_20261024_attempt04/collection_contract.json
- runs/audio_data_archive_preflight_20261024_attempt01/report.md

## Observations
- [result] 140条官方track阶段特征重放完全复现31lag曲线，窗口算术独立通过；19/20来源具有预定共同支持。
- [decision] 无任何新训练；特征发现未来只用音频，视频仅保留官方评分与验证参照；11格点约615ms仍为待验证解释patch。
- [decision] 原40之外历史58人和封存dev/confirm身份不因物理可读而获纳入；扩样为40+160，最终600源资格冻结前不放行4200媒体。
- [decision] 保留技术失败、无EOS/长尾、低对齐覆盖；不按分数换seed或删异常，不删除历史资产。
- [blocker] 全量媒体活动盘预算约57.2GB而当前仅约14.65GBfree；网盘小包双host字节一致尚非provider持久化证明。
- [next_step] 继续监控现有60媒体与158自然包作业，验收完成凭据后释放GPU给40人140条增量特征重放，随后汇总真实完成量与全量存储未决门。

## Relations
- continues [[Codex checkpoint - 2026-10-24T08-47-00Z - WIMHF design and audio-only explanation scope]]
- summarizes [[SyncNet音频特征官方重放与200人语料存储核验 2026-10-24]]
- relates_to [[WIMHF式局部视听差异与SyncNet偏好解释方案 2026-10-24]]
- relates_to [[研究主线与证据地图]]
