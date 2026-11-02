---
title: Codex checkpoint - 2026-10-20T06-30-27Z - May purified TTS AVE qualified and
  training started
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t06-30-27-z-may-purified-tts-ave-qualified-and-training-started
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T06:30:27.518459+00:00'
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

# Codex checkpoint - 2026-10-20T06-30-27Z - May purified TTS AVE qualified and training started

## Summary
用户授权的新May云端TTS净化训练已完成AVE与真实loader资格，从报告50原完整初始化正式运行；第2000步完整断点已在F独立SHA及CPU状态审核通过。

## Story
本chat从TTS相对自然语音的同步评分与迁移边界，转向音量/输入归一化。报告54–56短程编码器实验与Sync-C-only图已经交付，Patrick新RMS复跑按用户要求在coarse7保存暂停，旧fine14/Chris/Cathy/24full保持暂停。随后用户要求回到“最初TTS净化后训练数据”的路线，使用May、不做音频归一化，与首次成功baseline尽量只改训练数据，明确禁止本地TTS并选择云端qwen-audio-3.1-asr-flash/tts-flash。上一checkpoint封存了12段内容完整、Sync-C门通过的数据准备，不等训练收益。

最新用户明确要求“先完成AVE缓存和加载检查，通过后开始训练，参数尽量和第一次成功May baseline保持一致，只换净化后的TTS训练数据”。本轮因此执行完整准入后启动一个0dB训练臂，目标60000成功FP32更新。调用报告50的原train函数，保留原AVE512、seed42完整initial、4096rays/maxsteps16、原L1及正则/AdamW/指数LR/EMA；自然val/test原件不变。3849新监督行不同于原4150行，采用同np.default_rng(42) repeated permutation算法另冻结60000索引，不能冒称同一帧序列；这是数据处理差异，不调优化算法。

## Working State
权威训练根F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1。formal_admission.json SHAebe56efc058de612d35695f77cd4691ce9199db76361b18f24e3eb137851773b绑定最终protocol、执行源、官方源、权重、initial、order、feature/qualification receipt及全部使用的train/val WAV/JSON与cache。protocol SHA87e5ccc0018acd962e6e37c04c12512005cd5229d8992211b857ae75319983ed，pre-admission旧协议另保留。formal_sources_v1保存冻结源，execution_sources保存早期部署原件，二者不能混淆。模型/utils/renderer源码SHA与报告50相同，provider为原baseline三处已批准cache/heldout适配的逐字节相同副本，不改loss或网络。

baseline50 F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017，official commit43079fb3f349675f6cb33031bb07262f7566cdd5。原initial文件SHA5a9efaad3111acd10a08d59527fa5c7a1b036cda6c0579a4ffe163c890000659，tensorSHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1。新orderSHA7c9e27b78945c88a1ed47f40149b07c42ca5f46d462cec3cf772dbbeb37febd3；initialization.json原文件保留旧orderSHA，新的frame_order_receipt才是3849域authority，不偷改原initial JSON。正式模型重新加载原initial，2步资格checkpoint不warmstart。

易失Ubuntu：wjj@192.168.50.2，SSH必须-b192.168.50.1，R=/dev/shm/may-tts-purified-training-20261020-v1；T=/dev/shm/synctalk-may-gain-inference-20261016为隔离CUDA/GCC扩展。lexical Python /home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python（torch251cu121）；共享venv与CloudDrive未改。R/data指向data_restore/dataset，source May24204个图像/landmark/torso及bs/bc传输SHA通过；新train0替换TTS WAV/JSON，自然原train0保存在natural_gain_0_original，heldout其他gain保持。原AVE权重SHA1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4。

正式launcher PID30101/flock→job30102→trainer30112，2026-10-20T06:25:44Z启动，PIDs需以实际job/ps为准。commandSHAd401f2acfb775db2059865f04f24cda2010e76a98b0b62d1697eaec38c88b52b。startup_review_v1.json独立审计2400/2400success/0skip、loss全有限、每行order/source_img_id/source_aud_id/LR及原initial精确；这是启动前缀，不是60000完成。formal_trace.jsonl按每次实际更新记账；原train_curve/每2000步验证/EMA1000和checkpoint不变。run_training只加有限性拒绝、记录及fullcp RNG/counters，不自动restart/replay或改精度。正式失败封存terminal_state.pt/formal_failure.json，job wrapper持久退出码。

Windows只读备份observer：observer_launcher父296568，controller/receipt实际host_pid98152，均会退出或轮换以receipt为准。每30秒核完整checkpoint/val JSON，F remote_artifacts下全成员SHA通过；live是可变状态/trace/stdout快照，不冒称冻结文件。所有新fullcp保留，无旋转删除；没有评分/视频/自动化启动。第2000步F fullcp SHA749a3cfa6d4c8055fd2a46efa2e46fa03f4629d6767499b6bd9204e00f337e05，CPU真实load检查201浮点tensor/5938022元素全有限，model/optimizer/scheduler/EMA(2次更新)/metadata/RNG/counters齐全，2000success/0skip，0GPUforward/optimizer重放。first_checkpoint_independent_cpu_review.json是此独立证据。此时backup transport_errors0，不声明cloud备份。

Task tts-exp/tasks/训练-may-的云端-tts-净化数据对照已新建in-progress/current_step2；实验tts-exp/experiments/may-云端-tts-净化训练数据对照更新running/phaseTRAINING；研究地图当前段和Task Board读回通过，memory_publication_receipt.json保存。旧准备Task仍done，只代表数据阶段完成，不把整个训练研究done，不逐epoch文档日志。

## Changed Files
新增未提交scripts/experiments/may_tts_purified_training_20261020/：features.py、common.py、qualify.py（原错误review保留）、qualify_v2.py、wait_build_qualify.py、run_training.py、job.py、host_backup.py；tests/experiments/test_may_tts_training_v1.py。R/baseline_train_original.py是历史train.py精确副本。scratch bootstrap/stage/deploy/repair/order/readback/admit/startup-review/publication脚本为机器局部恢复指针，正式实际源已冻结F。继承大量无关dirty，不批量stage/clean/commit/push。CodeGraph先试但因41文件新鲜度拒绝，随后直接源码读审。

## Verification
新AVE训练full forward两次逐位相同：padded[3915,512]、aligned[3849,1,512] SHA49bd6a5b674a528e006920f3735d287dd47a2b2335763aaab663034f9103cc1d；原DOUBLE输入被librosa读取等于逐样本float32 cast，无波形归一化。自然val两次重算逐位相同，且与历史cache逐位一致，paddedSHAf343f3750263bcbeecfa421496ef170e917d9f623643b1419f968cb9d734049e。train3849/val550/test1100真实NeRFDataset加载，通过36处原source像素、provider相机变换、全音频特征索引检查。2次真实FP32入口更新loss/grad/model/optimizer有限，model/optimizer/scheduler/EMA/RNG保存/CPU读回通过，正式未加载这份资格state。

qualification_readback_manifest逐成员fresh F SHA及独立NumPy验证padded/repeat/aligned/source aud_id、natural val parity和新60000order全精确。5项实际CPU检查直接调用通过：优化参数同baseline、order域/算法、真实cache/loader receipts、原官方network/utils/renderer/providerSHA、备份路径越界/Windows驱动拒绝；pytest环境缺失未运行pytest或整库。前序数据416科学成员manifest95cabf40c2bd50e62cf6763d9c3f7a9ba6b8a656d16db1fc39a50b5a772c0491本轮重新fresh精确，原件不动；12段C最低6.121/均值7.257是同SyncNet净化技术门，不是生成器结果。

准备故障原件保留，均0正式训练更新：CUDA缺cusparse头，复用已安装107头symlink修复隔离T/include；首次修复exec命名空间变量碰撞没有编译；CPU编译超过waiter300秒，资格未进入；CRLF脚本启动在Python前失败，run_env_v2以LF修复；qualify.py把val的原HWC全图误当训练flatten，模型构造前assert拒绝，qualify_v2按原布局检查后36项通过。没有重放正式optimizer或更改provider/math。资格前compile并非训练数值故障。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1/formal_admission.json
- F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1/controller/receipt.json
- F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1/first_checkpoint_independent_cpu_review.json
- 旧Patrick暂停authority user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090；不得恢复旧controller。

## Observations
- [result] 新TTS AVE/cache和36真实loader检查及2success资格通过，正式训练真实启动，2400/2400/0启动前缀核验；第2000步fullcp F SHA与CPU全state有限通过。
- [decision] 同报告50原AVE/seed42完整initial/60000 FP32预算和原优化参数；仅新TTS净化训练数据，无归一化、无本地TTS，自然heldout保持。
- [decision] 新3849行按同随机排列算法另冻索引，不能复用4150域或称同帧序；资格权重不用于formal，所有新fullcp保留F，无自动故障重放。
- [next_step] 继续只读跟踪实际job/trace与F备份到60000步；首次故障先封存并停，若完成则独立核验全部30断点/最终state/验证曲线后准备与原baseline匹配的后续评分，不能把当前启动或数据C门当最终结果。

只读GitHub证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}；HEAD `2b71639c3b555293375d248350999377168d515d` local/unconfirmed remote，不构造无效commit链接。

## Relations
- continues [[Codex checkpoint - 2026-10-20T04-13-57Z - May cloud TTS purified data ready]]
- relates_to [[May 云端 TTS 净化训练数据对照]]
- relates_to [[训练 May 的云端 TTS 净化数据对照]]
- relates_to [[50-synctalk-may-gain-training-pilot]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1072条；大量其他任务改动，不批量stage/clean/commit。
