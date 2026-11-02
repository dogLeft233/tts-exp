---
title: PC-Talk Chinese-LiPS21配对九评分器2026-10-31
type: task
permalink: tts-exp/codex/tts-exp/pc-talk-chinese-li-ps21-配对九评分器-2026-10-31
status: done
current_step: 4
priority: high
completed: '2026-10-31'
---

# PC-Talk Chinese-LiPS21配对九评分器2026-10-31

- [status] done
- [current_step] 4
- [priority] high

## 最终交付
[[84-pctalk-chinese-lips21-nine-scorers]]为正式报告，研究地图/实验索引/SYNC-C-SCORES/部署记录/数据入口已更新并读回。用户要求用新数据跑一些PC-Talk；复用另一会话已有104/105成功云Plus TTS，每人成功ID字典序首条，21说话人各一对，42唯一视频，九评分器均21完整来源。不是105目标或104成功对全量；未新云调用或训练。

主要无源饱和/TTS触顶标记17人仅Sync-C通过Holm9：N7.458/T8.027/Δ+0.569，95%[+0.320,+0.795]，q=.00675，14/17正。补充全21 N7.489/T8.036/Δ+0.547，q=.00081，18/21正；另PEAVS−.410676、q=.01464负差通过。Synchformer全21虽15人TTS较高，但均差−.714个百分点、q=.76919，未确认平均优势。其它未确认不等于等效；非人类嘴型质量或完整无泄露结论。

## Steps
- [x] 定位另一会话已有104条TTS、清单及21张HDTF性别匹配人像；读取当次192手册。
- [x] 输入/模型/运行时SHA资格核验，42唯一生成，Sync-C及原始矩阵/stdout/主脸轨迹核验。
- [x] 其余八原生评分器及六类504条±400ms控制；九项42媒体/21来源，无缺测或插补。
- [x] 本机完整传输SHA、九端点配对重构、18分层统计复算、图/HTML验收、正式报告与终局凭据。

## 固定条件及边界
Chinese-LiPS官方test音频；21人各独立ID参考，单成功TTS draw；每位音频说话人固定同一张F/M标签匹配HDTF异人人像，N/T同图。RMS−24 mono16kPCM16；person192/lip1/无EMC/seed42/crop2.3，按原生left-right顺序选最大bbox（21图实际均index0）。官方pasteback454–904²、25fps、153–309帧，完整PCM不移/不补，视觉尾部缺约0.193–0.237s。四个质量标记来源024/027/074/141保留全21，主要17预先排除这些标记。

HDTF人像作者训练归属、完整组件训练集和参考原录制绝对时段不重叠未核。旧83为固定3.png/三逻辑draw及不同音频/参考/片长/支持集，不据显著性差异归因模型升级或效应增强。HDTF20 TalkLikeYou用户暂停始终有效，未恢复。

## 验收凭据
本机 results/pctalk/lips21_rms24_nine_20261031_attempt01/ 下 final_delivery_receipt.json PASS；index.html为最终21对视频浏览，paired_scores.csv、summary.json及九面板PNG/PDF齐全。42 MP4位于results/pctalk/{natural_raw,tts_raw}/各来源目录，已核SHA；未声明网盘备份。

证据包531226799字节，SHA256 5f461c6d9e58f0e7a897fc0b0f22c23ff277f6041a173c0f148fd3b9196571d8，2925内容+manifest共2926成员全部读回。local_raw_audit九端点/42原件归一化/媒体绑定PASS；local_statistics_audit18分层均值/10k bootstrap/100k signflip/Holm9/胜数独立误差0。服务器两分类器完整CPU FP32 softmax精确重放+独立FP64（max1.737391e−7）通过；Sync-C FP64max误差2.251327e−6、stdout3dp全匹配，42单track155–311帧，主脸bbox最小IoU .907610，42offset全−2、内部MP3保留无补偿。

AVS83/84、MTD68/84（达门需68）、Voca77/84、两P0各84/84降分；CoG峰移84/84且median excess正。SyncNet/PEAVS/SyncReward本队列该类控制未重测，PEAVS批次依赖和旧不足边界保留。MTD旧65/82未过与本批结果分开，不回写。2 pytest、语法、42成片各首中尾三帧与九面板图视觉核看、45HTML链接核验PASS；非连续视听盲评。

## 已解决问题与原件保留
1. 首N官方454²有效输出被旧512²验收拒绝；只修私有几何校验，复用原worker成片，raw/scoring MP4 SHA不变，无重复该TFG。
2. 27生成后下一条131_29_F_LY_009_T在模型加载前资源断言失败；原无process/worker_result。generate_resource_wait03仅把原16GiB空余/util≤85断言改45秒等待，沿同GPU/参数续缺项。原执行33源码及27完成项388科学文件SHA不变。
3. 模板20人/80条/chinese20/旧父输入/PEAVS64+16和strict_reference描述显式更正；inherited_metadata_original保留原protocol/inputs及SHA，567媒体文件未变。选样、主要17规则、音频/图/视频、模型/评分端点/种子未改，42次TFG和SyncNet不重跑。

## 当前执行状态
服务器run /home/data2/wangjiajun/av-evaluation/jobs/pctalk_lips21_rms24_nine_20261031_attempt01；源码 scripts/experiments/pctalk_lips21_20261031，执行原件/修复和freeze在证据包；本机验证源码另存local_source。原模型100 tracked源码/12权重及18 SyncNet pins、26共享评分源码/五环境前后匹配；PC环境也匹配部署freeze。

terminal_receipt PASS：47记录PID均退出，无绑定本run的本人命令；最新orchestrator362904及旧334371均结束。全局GPU当时1279MiB/15%可能为其他任务，不称整卡空闲。旧RUNNING42、校验/资源失败快照都是历史，不重新launch。无待执行单元。无Git提交/PR；继承dirty未清理。

## Relations
- follows [[83-pctalk-chinese20-nine-scorers]]
- relates_to [[84-pctalk-chinese-lips21-nine-scorers]]
- relates_to [[Chinese-LiPS 官方测试集 21人×5句]]
