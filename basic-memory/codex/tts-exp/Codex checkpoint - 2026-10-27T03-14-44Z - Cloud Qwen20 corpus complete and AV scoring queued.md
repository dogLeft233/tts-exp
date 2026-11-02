---
title: Codex checkpoint - 2026-10-27T03-14-44Z - Cloud Qwen20 corpus complete and
  AV scoring queued
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-27-t03-14-44-z-cloud-qwen20-corpus-complete-and-av-scoring-queued
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-27T03:14:44Z'
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

# Codex checkpoint - 2026-10-27T03-14-44Z - Cloud Qwen20 corpus complete and AV scoring queued

## Summary
云端Qwen Audio3.1Plus的20来源60条语音已生成并独立核验、完整备份；四指标原生视频评分已排队，尚无新GPU分数。

## Story
本聊天在检验TTS原生Sync-C优势是否反映口型改善或评价器偏好。此前DTW音频/视频重定时及STFT幅度移植未超过自然音频基线，不重新包装成新路径。前一检查点的PEAVS/SyncReward评测已推进：138原生项及40时移控制对两个模型均完成，472交叉项尚未全完成。原生118配对中，SyncReward与Sync-C冲突58/118(49.2%)，PEAVS66/118(55.9%)；Sync-C认为TTS更高73/118，CoGenAV Eq1为45/118，SyncReward49/118，PEAVS51/118。20来源等权T-N分别Sync-C+0.120369、Eq1-0.004831、SyncReward-0.020355、PEAVS-0.170340，各描述性95%区间均跨0。尤其±400ms使得分下降仅SyncReward16/40、PEAVS26/40，低于预设32/40，科学状态CONTROL_FAILED；不能把新指标当同步真值或据冲突认定SyncNet有偏好。权威原生审计笔记见Relations。

用户随后询问历史云端TTS是否更高。已全文核对两份20261023实验和本地报告：同10来源统一RMS，N7.108、local0.6B6.861、local1.7B7.517、cloud3.1Plus7.670；cloud-N+0.562(8/10)、cloud-local0.6+0.809(9/10)区间不跨0，cloud-local1.7+0.154区间跨0。raw均值依次6.737/6.604/7.266/7.522。只能说明具体模型和协议，不能归因于云端托管。刚才118配对只包含本地模型，不含cloud。

最新用户明确授权用qwen-audio-3.1-tts-plus生成刚才同队列TTS并重评，提供了运行时凭证。凭证未写入文件、代码或记忆；云端子进程已结束、key环境移除，后续评分不需要密钥。已冻结20个既有来源、每来源3次、不按结果选样的60条语音。12个原参考>=5秒，严格沿用相同单句参考；8个目标<5秒，使用该目标和字典序第一条同说话人不同句的拼接作为克隆参考，目标文本和生成音频时钟不修改。这8个与local参考存在混杂，单列扩展参考层，不混作严格12来源主比较。全部使用旧固定RMS0.0376838172675746、16k单声道PCM16，无时间拉伸。60调用输出仅21种唯一字节，全部保留；重复draw不增加独立来源数。

## Working State
权威新run：runs/cloud_av_preference20_20261027_attempt05/，Ubuntu /home/wjj/tts-audio/tts-exp，SSH wjj@192.168.50.2。Windows唯一入口E:/Documents/tts-exp。protocol SHA d181952455967c6f769f023429732b4dd7ee22135992d017e528896cee46bb39，audio_manifest SHA8b04534cebe8c2f1bef6955db731204ecbd8dd602137c64ab36c146d36762da0。cloud_completion为COMPLETE60/60，独立synthesis_independent_audit PASS。E上只有协议、清单、审计、状态快照和synthesis_report.md，不称全媒体本地展开。

完整合成备份C:/Users/dogle/.codex/tts-exp-artifacts/cloud-av-preference20-20261027/synthesis_bundle.tar.gz，43836450bytes，SHA1f26a3728bc3537f23740dac35bfeaef0f6adb94feb543d7bcd8561af3124ae8。462成员含461数据文件+manifest，C侧逐文件读回核验通过；涵盖60provider及canonical音频、20实际参考及原参考组成、源代码/收据/审计。C为本机备份，不是百度网盘；未来GPU产物尚不在该归档。部分音频路径指向attempt04首3条，8个参考指向attempt03，归档按项目相对路径收全依赖，不可仅搬attempt05目录就删除先前目录。

当前机器易失状态：后台pipeline PID1722701仍活跃，cloud子进程1722711已经正常退出。status=WAITING_GPU，next_stage=render_sync_BAC009S0002W0122_C1，显存630MiB/util0，ComfyUI PID1719508仍有308MiB计算上下文；并非GPU满载。等待无外来计算任务、显存<1024MiB/util<15连续3次30秒，以及前一650格评分run终态COMPLETE，避免两个等待者同时抢GPU。前一run av_metric_preference_20261026_attempt02/PID1638959仍WAITING_GPU cross011，SyncReward442/650、PEAVS178/650，原生与控制均完成。现有生产controller PID320414为用户要求的Ts暂停，2040/2400，不恢复、不杀。不要终止外来ComfyUI。没有创建定时监控或通知。

后续执行链已部署：60cloud各自驱动相同官方Wav2Lip GAN/图3/25fps/nosmooth/batch4；SyncNet沿用自然基线精确MKV PCM16 copyts/avoid_negative_ts disabled，保留官方pipeline/run_syncnet、float32 lag-major聚合C/D/B/offset，核验PCM、像素与PTS；PEAVS/SyncReward使用与旧评测相同ALAC MP4及官方已部署wrapper；CoGenAV用既有Base/Eq1提取器和独立标注custom bestlag，保留真实波形时移控制。计划报告严格12、扩展8、全20的来源等权分差/偏好/与Sync-C冲突，描述性bootstrap10000 seed20261026。SyncReward V100 FP32/center4.75s和PEAVS全片支持差异不隐去。不能称新评分完成。

## Changed Files
本轮新建、均未提交：scripts/experiments/cloud_av_preference_20261027.py、cloud_av_preference_synthesis_20261027.py、cloud_av_preference_queue_20261027.py、cloud_av_preference_synthesis_audit_20261027.py；tests/test_cloud_av_preference_20261027.py、test_cloud_av_preference_synthesis_audit_20261027.py。前序av_metric_preference_20261026.py及对应test也仍未跟踪。大量其他继承dirty/untracked本次未归属或清理。当前branch codex/cross-tfg-audio-head-20261003、local SHA2b71639c3b555293375d248350999377168d515d；未commit/push。

attempt01准备中缺input24路径、无协议或API；attempt02首次enrollment120s读超时，0合成，服务端后来成功创建，Windows只读list_voice确认后复用；attempt03只准备参考无API；attempt04首3合成成功，第二音色上传15s写超时，只读list确认第二音色不存在。attempt05通过SSH反向loopback代理修复上传路径，验证复用首3音频和首音色，只生成其余57，完成60。隧道Windows ssh PID145540已在合成完成后关闭，无后续API依赖。保留失败收据，不伪造首次成功或重复生成。

attempt05协议冻结后、任何GPU评分前，以execution_repair.json附加说明将SyncNet入口绑定到与N完全同构的MKV/PCM与PTS检查。原orchestrator SHA57c49b8c4c83e6f9ba77a0e9d65ab937148d18729b5f662378135fb90c33c32b，修复后7f87bd858f9cbe52796ce99bfda38591030cd18c4e385ca5cdc170c511acd3db；execute显式校验原SHA与repair新SHA，其余输入/权重/脚本pins不放宽。冻结协议不重写，source_snapshot保留两版。

## Verification
独立CPU审计不导入producer，逐60文件重做provider->16k float转换和固定RMS/量化，provider/canonical/参考SHA、文本/身份、采样数、未改目标时钟均通过；最大重建差1 PCM16 LSB，最大RMS目标误差2.2526239782105772e-6。60记录恰为20x3、21个unique provider byte sequences，重复未丢弃。

用旧20自然activesd在syncnet环境重算canonical C，max误差4.76837158203125e-6。w2l_syncnet_v100_attempt01环境numpy1.24无法读numpy2 pickle，实际orchestrator明确用/home/wjj/.venvs/syncnet/bin/python；cloud CPU沿用w2l环境；CoGenAV解释器是/home/wjj/.venvs/cogenav-sync-20261024/bin/python，非不存在的cogenav。所有实际解释器存在。

4个测试函数直接执行PASS：SyncNet lag-major官方布局、指标解析保留精度/忽略raw中间文件、停止controller不阻塞而运行阻塞、缺draw清单拒绝。未运行pytest，不称pytest通过。py_compile运行成功。真实147帧自然样本构建MKV PCM16和ALAC MP4的CPU IO控制，解码PCM、逐帧像素和PTS全部保持。没有新GPU前向测试；要等资源验证完整链。归档C逐SHA462成员全部通过。GitHub只读确认canonical dogLeft233/tts-exp，当前分支open PR查询为空，本地SHA未验证远端可解析，不造commit链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Cloud reference requirements](https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide)
- [Voice enrollment HTTP API](https://help.aliyun.com/en/model-studio/voice-clone-design-http-api)
- 历史比较本地报告：runs/qwen_audio31_cloud_subset10_20261023_attempt01/report.md；runs/qwen17_level_comparison10_20261023_attempt02/report.md。
- 新数据交付：runs/cloud_av_preference20_20261027_attempt05/synthesis_report.md、synthesis_independent_audit.json、synthesis_bundle_receipt.json。
- 旧原生审计：runs/av_metric_preference_20261026_attempt02/native_conflict_report_20261027.md、native_mean_scores_20261027.json。

## Observations
- [result] 新cloud60/60音频完成，20来源，独立音频审计与完整C备份读回通过；GPU评分尚未开始。
- [decision] 12严格同参考为主层，8扩展参考单列；生成目标不重定时，固定RMS，全部重复draw保留。
- [decision] 保留PEAVS/SyncReward时移控制失败边界，不因模型投票称Sync-C偏好或感知口型更好。
- [decision] 不恢复用户暂停的生产controller，不终止外来ComfyUI；新任务等旧650格评分结束再使用GPU。
- [blocker] ComfyUI常驻GPU及旧评分未完成使新评分等待；当前仅630MiB使用仍不满足无外来计算进程门。
- [next_step] 读取attempt05/status及pipeline.log、旧评测status确认资源进展；资源允许后由现有pipeline完成60视频/四指标评分，再独立核验12/8/20层统计与冲突并报告。失败时按保存收据修复断点继续，不重新调用已完成TTS。

## Relations
- continues [[Codex checkpoint - 2026-10-26T08-29-54Z - PEAVS SyncReward preference evaluation queued]]
- relates_to [[PEAVS与SyncReward原生自然TTS评分分歧及同步控制失败 2026-10-27]]
