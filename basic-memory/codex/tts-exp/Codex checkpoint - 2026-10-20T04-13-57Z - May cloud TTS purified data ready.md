---
title: Codex checkpoint - 2026-10-20T04-13-57Z - May cloud TTS purified data ready
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t04-13-57-z-may-cloud-tts-purified-data-ready
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T04:13:57.675923+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-20T04-13-57Z - May cloud TTS purified data ready

## Summary
用户最新授权的May云端TTS净化训练数据已准备、内容修复和独立验收，12/12段Sync-C严格>5；0新生成器训练，后续训练前AVE缓存与loader资格尚待执行。

## Story
原chat研究TTS原生同步评分能否迁移到自然驱动/训练收益，随后聚焦音量与归一化。上一交付报告56已完成，用户把图改为只看Sync-C：F:/Datasets/tts-exp/may_postnorm_sync_c_only_20261020_v1/may-sync-c-only.png及PDF、plot-source.py、figure-receipt.json完成、实际view_image读图通过；封存科学根may_postnorm_scale_20261020未变。该简图已作为前一最终回复展示。

最新用户要求先找“最开始TTS净化后作训练数据”的方法，使用May、无音频归一化，与历史第一次成功baseline尽量只变数据，并提供阿里runtime key允许Qwen ASR/TTS。随后明确“别用本地的tts”，指定qwen-audio-3.1-tts-flash和qwen-audio-3.1-asr-flash，最后要求净化后Sync-C至少高于5。历史BM/Ubuntu实际来源确认净化是NAS SyncDrift滑窗搜索、中值/PCHIP、单调帧重排，不是降噪；旧10人推理对齐不能称已有SyncTalk重训成功。

这次回到报告50自然0dB/原AVE/seed42/60000成功FP32更新/4096rays/maxsteps16/原loss AdamW LR EMA，而不是报告54–56新encoder短程小试。原initial tensorSHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1、initial文件与frame order按SHA绑定。自然val/test音轨与JSON不改。只准备数据，不启动生成器/optimizer/旧队列。

## Working State
权威新根F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1，入口current_preparation.json、final_data_delivery_receipt.json、prepared_dataset_v1/readiness_receipt.json与DATASET_README.md。416成员delivery_manifest fresh全SHA通过；final receipt passed/DATA_READY_NO_TRAINING。prepared_dataset_v1/tts_purified/train含DOUBLE WAV与新transforms；三图目录junction至原F源资产，不得删除。source_binding明确img_id原source所有权、aud_id新TTS时钟；自然val/test精确副本、baseline初始与协议副本均在同数据包。raw API失败、首次漏句与修复、embeddings/距离/索引/窗口记录全保留。API key仅进程stdin/内存，不进文件/argv/BM，211个新文本文件扫literal无key；本checkpoint也不存secret。

原train支持[0,4200)共168秒，按固定声学低能量规则分12段。云端3.1 ASR→声音克隆→原生24k→resample_poly16kDOUBLE，无客户端音量归一化/拉伸/clip/limiter；voice-enrollment显式enable_preprocess=false与enable_volume_normalization="false"。NAS参数0.5秒窗、0.1秒stride（25fps round后2帧）、±1000ms、median3，逐段先uniform视频时钟匹配TTS原生时长再净化，source不跨段/heldout，图像与全部几何同source id。最终TTS156.56秒/3914音频帧，guard后3849监督行，baseline4150行。原frame_order的4150域不能直接索引3849新行，后续训练必须另冻结成对采样映射，明确该时长/重复/跳帧是数据处理差异，不冒称完全同帧序。

最终12段guard20 C全部严格>5：最低6.12065315246582，等权均值7.257146279017131。prepared_dataset_v1/sync_c.csv保存逐段全精度。独立119474距离float64重建maxerror5.3045098020731984e-6，全部独立C>5。NAS与QC同一SyncNet，技术门不是感知或未来训练效果证明。FP32/TF32off、source-owned ROI、音频float64×32768原PCM单位。0新模型训练、0本地TTS；AVE新特征cache/真实provider loader资格未执行，不能把自然cache当TTS或现在直接训练。

机器局部SHM /dev/shm/may-tts-purify-data-20261020-v1保留运行副本，权威F读回已闭合；资格pid12905/12906与修订13930均自然退出，最新ps已无本任务GPU。桌面2391及既有ComfyUI7002未动。没有新增自动化或发其他chat消息。原Patrick coarse7保存暂停、fine14/Chris/Cathy和24full仍暂停，不归档chat，不恢复旧controller。

BM新实验tts-exp/experiments/may-云端-tts-净化训练数据对照为planned/phaseDATA_READY_NO_TRAINING；Task tts-exp/tasks/准备-may-的云端-tts-净化训练数据done/current_step3，仅数据任务完成。地图对应当前段已发布、全部读回验证。行政QA与BM publication receipt放F:/Datasets/tts-exp/may_tts_data_admin_20261020_v1，不追加到已封存416成员科学根；最终5测试源也保存该admin目录，科学执行源镜像保留原执行时版本。

## Changed Files
新增未提交scripts/experiments/may_tts_purified_data_20261020/：prepare_audio.py、finish_asr.py、bind_baseline.py、cloud31_audio.py、purify_remote.py、repair_missing_sentence.py、repair_reference.py、finalize_dataset.py，以及tests/experiments/test_may_tts_data_preparation.py。scratch内history/BM读取、cloud接口probe、SHM stage/repair/readback/seal/publication/checkpoint源为机器局部恢复指针；F execution_sources保存实际源。无改官方网络、loss、共享环境、CloudDrive或旧科学根，不批量stage/clean/commit/push。

## Verification
原音chunk重组与原PCM逐样本一致；29400训练侧源及4200源RGB freshSHA；val/test WAV与JSON原件精确；baselineinitial/frameorder文件SHA通过。3.1新ASR/TTS API实际200，12最终回转文字WER0。源所有权/全部3849记录camera与geometry/aud_id映射、无heldout交叉、云端模型/归一化标记及逐段C门共5项实际数据检查直接调用通过；环境缺pytest，首次pytest模块启动失败未称通过，未跑整库。F首次153成员、修订21成员Linux读回SHA通过，最终416成员二次fresh核验与BM三笔读回通过。全部distance矩阵独立重建passed；无额外视觉盲评或新训练，未实际view_image新数据（仅上轮C-only图已读图）。

## Failure Boundaries
历史qwen3-tts-vc-2026-01-22注册成功但合成404；Flash系统音色克隆400。不能据此判新3.1不支持；新接口voice-enrollment/SpeechSynthesizer实际200，ASR需input_audio、format、sample_rate并读取新嵌套output。3.1首ASR用旧参数format empty400，修正后200。ASR probe脚本一处括号准备语法错误已修，0实际调用；原旧BM读取json datetime错误default=str修复，不是实验故障。

001第一版漏前一句，3.1与旧Qwen3 ASR均确认、WER0.4、净化C2.479，初版DATA_QC_FAILED永久保留。原参考分句修复前句仅0.88秒，仍漏句、WER0.4，未采用；最终用已内容通过的同May训练000参考voice、同3.1模型/seed42，完整文本WER0、C6.583。只修内容错误，不扫描Sync-C/净化参数、不删段，其余11段不重合成/不重净化。readback修订tar因Windows pipe尾部padding背压阻塞，仅终止两个本任务CPU读回进程，增加stdout drain后同文件精确复读21成员passed；没有GPU/科学重放。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1/final_data_delivery_receipt.json
- F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1/DATASET_README.md
- [Qwen-Audio3.1 ASR调用](https://help.aliyun.com/en/model-studio/fun-asr-flash-recorded-speech-recognition-http-api)
- [Qwen-Audio TTS调用](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-http-api)

## Observations
- [result] 云端Qwen-Audio3.1净化数据已交付，12/12 C>5，最低6.121/均值7.257，0本地TTS/0生成器训练。
- [decision] 用户禁止本地TTS，选择云端3.1 ASR/TTS；原May/AVE无归一化baseline及自然heldout保留，只准备数据。
- [decision] 缺句必须内容修复并保留失败，不能静默删段或靠Sync-C调参；同SyncNet净化/技术QC不外推感知与训练收益。
- [next_step] 等待用户是否要求启动训练；若授权，先冻结3849行成对采样映射、生成新AVE缓存并通过真实loader资格，再从报告50同initial按原60000成功FP32预算执行，不直接恢复旧实验。

只读GitHub核验：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}；HEAD `2b71639c3b555293375d248350999377168d515d`为local/unconfirmed remote，不构造commit链接。

## Relations
- continues [[Codex checkpoint - 2026-10-20T03-19-11Z - May postnorm chart delivery]]
- relates_to [[May 云端 TTS 净化训练数据对照]]
- relates_to [[准备 May 的云端 TTS 净化训练数据]]
- relates_to [[50-synctalk-may-gain-training-pilot]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1070条；大量其他任务改动，不批量stage/clean/commit。
