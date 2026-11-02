---
title: PC-Talk Chinese-LiPS全量104配对九评分器2026-10-31
type: task
permalink: tts-exp/codex/tts-exp/pc-talk-chinese-li-ps-全量-104-配对九评分器-2026-10-31
status: active
priority: high
current_step: 4
---

# PC-Talk Chinese-LiPS全量104配对九评分器2026-10-31

- [status] active
- [current_step] 3
- [priority] high

## 用户授权与冻结范围
用户在报告84的21对试跑后要求“跑下全部的数据”。本轮105自然+104已有云Plus TTS=209媒体，21说话人；缺024_29_F_TY_139的TTS（明确Engine411历史失败），自然侧也生成评分，但不造补项、不入配对统计。配对分析104对；不新调云ASR/TTS，不恢复HDTF TalkLikeYou用户暂停。

同RMS−24、mono16kPCM16、每说话人固定HDTF性别标签匹配人像、person192/lip1/无EMC/seed42/crop2.3/原生pasteback25fps、相同九评分器/权重/原生支持和端点。209条预检查全能达到−24且峰值≤−1dB，最大.831465。完整参考区间和训练重合未知，非无泄露/人类嘴型质量证据。

## 统计与复用预定（新数据评分前）
多句先在说话人内平均，bootstrap/signflip按说话人（最多21），不把104句当104独立人。主要严格质量84对/17说话人（自然源零饱和且无provider触顶标记），补充全104对/21人；另列新增83对的同人未评句敏感性（严格新增67对/17人），不称独立说话人复现。每端点每分层要求该说话人预定句对全部N/T可测；缺测不补值、不降低门槛。每分层10k speaker bootstrap/100k signflip seed20261031，Holm9；统计是探索性，全量包含已看过的pilot21。

报告84的42媒体和八类逐片端点（除PEAVS）输入/图片/模型/参数/原始数组SHA核对后复用；新增167TFG/SyncNet及其余逐片评分。CoGenAV仅在科学协议字段完全相同、唯样本清单/数量不同且原缓存SHA/输入SHA核验时允许旧协议缓存；禁止重写旧NPZ。PEAVS因batch依赖全量重新评分：208配对媒体字典序batch64+64+64+16，另缺TTS自然单独batch1，仅描述，不混入配对组。所有原件/失败/复用映射保留，旧pilot不改。

## Steps
- [x] 完整清单/输入与父结果复用SHA、模型及环境核对；冻结209媒体、104对、统计计划。
- [x] nohup单GPU/4线程/affinity0-3 顺序完成剩余167生成与Sync-C；原42不重跑。209推理/SyncNet全完成，checkpoint PASS。
- [x] 九评分器全量覆盖，原生控制/支持、PEAVS全量批次及缺测单列。九项各209媒体/104完整句对，独立raw/stat及本机自动验收PASS，图与媒体agent QA/正式报告待补。
- [ ] 原数组/来源与说话人聚合独立复算、完整下载与SHA/图/报告验收。

## 执行入口
源码 scripts/experiments/pctalk_lips_full104_20261031；新run /home/data2/wangjiajun/av-evaluation/jobs/pctalk_lips_full104_rms24_nine_20261031_attempt01。**已启动，不是已完成**：服务器orchestrator PID572620（nohup bash run.sh）顺序生成→SyncNet→checkpoint/主脸矩阵核验→八项→原数组/说话人统计独立复算→环境复核→归档。准备PASS：209媒体/104对/84严格对/17严格说话人，2537复制文件SHA通过，100源码/12权重/18 SyncNet pins和PC部署环境匹配；CoGenAV84旧NPZ与科学协议兼容验证、绑定源码编译通过。数学/缺测/协议复用4 pytest、语法PASS。首次读取时已有61/209成片（42复用+19新增），新输入实际GPU前向及PCM/帧数/动态验收正常。generation_status.completed是已遍历清单项，真实可用数以generation/*/receipt.json计数，不误报旧42重复生成。

本机结果目录 results/pctalk/lips_full104_rms24_nine_20261031_attempt01/。local watcher PID69706（/tmp/tts192-client-env/bin/python -u scripts/experiments/pctalk_lips_full104_20261031/watch_local.py，Popen start_new_session=True 独立session）已启动：每45s只读服务状态，服务端失败不重试科学；终局自动核验本人进程退出、下载小凭据、SFTP断点续传完整归档并查SHA、逐媒体原件/原始端点校验、独立说话人聚合/统计复算、制作CSV/PNG/PDF/可筛选104对+单侧N浏览页。脚本只运行时复用已有host-pin凭据读取器，未存密码或API key；本机休眠/重启不影响服务器任务，监听/传输恢复状态见local_delivery_status.json、server_status.json及local_watcher_detached.log。它不自动发布BM正式报告或宣称人类视觉验收；READY_FOR_REVIEW后仍需agent核看图、写正式报告/更新Task，不把先验84结论充作全量结果。

控制另报208配对媒体的416个±400ms诊断/每类（为主要控制门），含单侧自然共209媒体418个诊断单列，不借单侧样本改变配对控制资格。PEAVS208配对重评+单侧自然独立batch1。

已读本次最新192手册；个人data2、UUID727单卡、4线程/affinity0-3、生成Torch45%/评分60%，不改共享base/驱动/他人进程。按pilot时长全量需数小时；当前模型仍运行中，未形成全量显著性结论。原pilot84已归档done且不改；HDTF TLY继续用户暂停。

## 本机监听启动验收
最初直接nohup的本机PID69378未在命令结束后存活，日志为空；未影响服务器科学任务。已改为Popen start_new_session=True、stdinDEVNULL独立session启动69706，跨工具调用ps确认Ss存活并写出WAITING_SERVER；首个后台快照为68/209成片（42复用+26新）。local_watcher_launch.json保留前/后启动方式。后续监控和日志以69706/local_watcher_detached.log为准，不能重启服务器全量任务来修监听。

## Relations
- follows [[84-pctalk-chinese-lips21-nine-scorers]]
- relates_to [[Chinese-LiPS 官方测试集 21人×5句]]

## 2026-10-31全量检查点与后续串行任务
209推理/SyncNet已全部完成；syncnet_checkpoint PASS，按说话人：all104句对/21人N7.597/T7.959/Δ+0.361，CI[+0.213,+0.511]，19/21正；strict84/17 N7.626/T8.000/Δ+0.374，CI[+0.198,+0.555]。去历史pilot的新83句/21人Δ+0.314。只是全量Sync-C检查点，Holm9/最终九项原始数组及统计验收尚待；不能把这些p/效应当已完成九评分器报告。最新服务器阶段vocalist；cogenav/avs_u/mtd已结束，PC任务继续运行。

用户追加“等pc talk跑完后跑下dystream”；[[DyStream Chinese-LiPS全量104配对九评分器2026-10-31]]的服务器队列PID980230已经独立session挂起，只有本PC run complete+独立验收PASS+terminal检查通过后才启动DyStream，当前0新DyStream前向。未改PC脚本/源码冻结、未中止或重启PC。DyStream不是恢复已暂停HDTF TLY。

## 2026-11-01已完成计算与本机自动验收
服务器COMPLETE，209/209推理、九项各209媒体/104完整句对；独立raw与36分层speaker统计审计PASS、环境/原复用数据保全及terminal PASS；本机完整归档SHA a440d0572a8024d3b04025fd8c803969b5d8d5100fa6d2dde3dccc608fa7523b，final_delivery_receipt PASS，local_delivery_status READY_FOR_REVIEW。全部媒体、逐句CSV/说话人图/104配对HTML已在results/pctalk/lips_full104_rms24_nine_20261031_attempt01。本机原件/端点及独立说话人统计重算PASS，212链接存在；人工图/媒体QA和正式编号报告尚待，不把Task提前done。

最终Sync-C全104对/21人N7.597/T7.959/Δ+0.361，CI[+0.213,+0.511]，Holm9=.0017099829；严格84对/17人N7.626/T8.000/Δ+.374，q=.0086399136；去历史pilot83句/21人Δ+.314，q=.0110398896。全部九项实际支持209/104，无缺测；六类±400ms控制PASS，SyncNet/PEAVS/SyncReward该cohort未新增相同控制，保留适用边界。

DyStream在PC terminal通过后自动接续，但首worker在torch导入之前受本地queue.py遮蔽标准库queue影响失败（0完成/0评分）；2026-11-01已保留失败证据，改等待入口名后独立resume，同参数输入未动。DyStream的恢复进度见其Task，不再是等待PC尚未完成。