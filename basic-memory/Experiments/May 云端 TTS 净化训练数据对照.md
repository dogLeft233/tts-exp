---
title: May 云端 TTS 净化训练数据对照
type: experiment
permalink: tts-exp/experiments/may-云端-tts-净化训练数据对照
status: concluded
phase: CONCLUDED
model: Cloud Qwen-Audio3.1 ASR/TTS; future original AVE
outputs: F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1
evaluation_outputs: F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1
report: docs/experiments/64-may-purified-tts-training.md
data_quality_followup: timewarp degeneracy found; repair not started
---






# May 云端 TTS 净化训练数据对照

## 当前交付

评测已完成并独立验收，科学状态CONCLUDED。一个TTS/NAS训练0dB模型，原AVE/同initial/60000 FP32/0skip，固定最终EMA60；原自然test三档驱动、原PCM统一评分，与报告50自然训练0dB模型匹配比较。-6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206 / +6dB：新C5.911、自然baseline 9.036、Δ-3.125。权威F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1；新三档3300 test帧、1650 val帧、12矩阵91140距离、96个最终val浮点帧/曲线、原PCM/全像素/GT ROI和250帧重复通过。旧baseline9帧float/RGB完全复现，四原音频SyncNet特征重算逐位相同，F完整成员SHA及第二次独立矩阵重建通过，真实图/代表帧检查完成。未声明云备份，所有新媒体/训练断点保留，无训练或评分重放；原source WAV路径准备失败在remote部署/GPU前修正，保留。新数据3849监督行与自然4150行不同，NAS同SyncNet数据QC不等效果，单May/seed42不能推广总体/感知结论，旧Patrick等继续暂停。正式报告[[64-may-purified-tts-training]]。

## 当前训练验收

训练阶段已完成并独立验收，phase=TRAINING_COMPLETE_PENDING_EVAL。2026-10-20T07:13:04Z（北京时间15:13）job rc0自然退出，60000attempts/60000success/0skip，耗时约47.3分钟，无本任务活动训练进程。F根may_tts_purified_training_20261020_v1的completion_review_v1/review.json为独立验收入口：128成员fresh SHA、60000完整trace有限loss/逐行source图像与audio绑定/冻结order/LR精确、60000 ray hash记录通过；全部30个fullcp CPU真实load，model/optimizer/EMA有限，scheduler/metadata/RNG/counters齐全，最终EMA60。最终step060000.pt SHA32e4be04483a290da463a20d98a1c28c42c104959ab0e2aeebcb0f1004ce3ba2。93条验证摘要/2976帧检查有限；三自然val gain后段MSE变化−0.877%/−1.193%/−0.986%，均通过预定±2%平台筛查，此筛查不等统计收敛或收益。正式冻结源、initial、cache/WAV/JSON fresh remote SHA均精确。只读备份host已结束，全部新checkpoint保持F，无cloud备份声明。

新训练仍为报告50原AVE、同initial/seed42、FP32/4096rays/maxsteps16和原优化合同，无输入归一化；新训练数据为云端TTS/NAS净化3849行，自然val/test保持。0新视频/0新SyncNet，不能把净化数据C最低6.121或训练loss当作新生成器Sync-C，尚未判断是否优于自然baseline。此处只发布训练验收，不自动新增GPU评测或重训，旧Patrick/fine14/Chris/Cathy/24full持续暂停。

## 问题与范围

用户希望回到最初TTS替换后净化再作训练数据的路线，使用May、不做音频归一化，与首次成功baseline尽量只改变数据。数据准备已完成；最新用户授权通过AVE缓存和真实loader检查后启动训练。后续明确禁止本地TTS，指定云端qwen-audio-3.1-tts-flash与qwen-audio-3.1-asr-flash，并要求净化后Sync-C高于5。

## 启动准入记录（历史）

用户已授权通过AVE缓存和真实loader检查后训练；新run F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1已启动一个0dB无归一化模型。沿用报告50原AVE、seed42原完整初始化tensorSHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1、60000成功FP32更新、4096rays/maxsteps16、原L1及正则/AdamW/LR/EMA。自然val/test全部不改，正式执行调用原baseline trainer，未加载资格权重或旧训练权重。

新TTS AVE全量两次逐位重复，train padded[3915,512]→aligned[3849,1,512]；自然val重算与历史cache逐位一致。真实train3849/val550/test1100加载、36像素/pose/audio索引检查、独立2success/0skip FP32入口和model/optimizer/scheduler/EMA/RNG保存读回通过。F已逐成员fresh SHA且独立NumPy复核；5项CPU真实产物/参数/路径拒绝检查直接调用通过（未安装pytest，不称pytest通过）。共同initial文件与官方network/utils/renderer/provider SHA同baseline。3849新样本域按同seed42 repeated permutation生成60000索引，SHA7c9e27b78945c88a1ed47f40149b07c42ca5f46d462cec3cf772dbbeb37febd3；不直接用旧4150域，训练数据时长/采样序列差异明确。

权威formal_admission.json、formal_launcher.json、startup_review_v1.json及controller/receipt.json。远程/dev/shm/may-tts-purified-training-20261020-v1是易失执行副本，SSH需-b192.168.50.1；只读host observer把每2000步全部fullcp/验证/trace持久F并fresh SHA，保留所有新checkpoint、不自动重启、无评分/视频启动。科学失败必须封存再停，0新cloud备份声明；旧Patrick/fine14/Chris/Cathy及24full继续暂停。

准备失败均保留：CUDA缺cusparse头由隔离symlink恢复；修复helper命名空间碰撞发生在编译前；等待300秒超时未运行qualify；CRLF启动失败未进Python；首loader reviewer错误地把val全图HWC与flatten比较，在模型构造前拒绝，修订qualify_v2按原provider布局通过，provider不改。全部发生于formal之前、0正式optimizer重放。当前已记录正式启动前缀由startup_review_v1.json固定核验，不能称60000训练完成。

## 数据准备权威产物

F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1/final_data_delivery_receipt.json及prepared_dataset_v1/readiness_receipt.json为数据入口。416成员manifest已逐成员fresh SHA读回；详细数据说明DATASET_README.md，12段分数在prepared_dataset_v1/sync_c.csv。该准备交付不等于训练实验完成，不新编正式报告编号。

报告50自然0dB/原AVE/seed42/60000成功FP32更新/4096rays/maxsteps16/原loss AdamW LR EMA为未来对照，initial tensorSHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1；initial文件和frame order精确复制。自然val/test WAV及JSON原件不改，train source[0,4200)共168秒，29400训练侧源文件freshSHA。与54–56新encoder短预算实验分开。

## 数据处理与实际验收

云端3.1 ASR→同May原声克隆→原生24kWAV→resample_poly16kDOUBLE；voice-enrollment关闭enable_preprocess及enable_volume_normalization；无RMS/z-score/loudnorm/clip/limiter、本地TTS或音频时长拉伸。按各TTS原生时间对自然图像均匀映射，然后复用历史NAS SyncDrift搜索、中值/PCHIP与单调帧索引，0.5秒窗/0.1秒步长/±1000ms/median3；25fps下stride实际round为2帧。图像、pose、bs/眼部、landmark、mask、torso同source id，不跨段/val/test。

最终12段guard20 Sync-C严格>5：最低6.121，等权均值7.257。独立从原始embeddings以float64重建119474个距离格，最大误差5.305e-6，全部12段独立C亦>5；FP32 SyncNet/TF32off、source-owned ROI，原PCM单位float64×32768。5项实际数据检查直接调用通过（该环境无pytest，pytest启动失败保留；不是声称pytest运行成功）。所有本任务GPU已退出，不干预既有ComfyUI和桌面进程。

第001段第一版漏句，经3.1与旧Qwen3两个ASR确认，原WER0.4/C2.479保留，初版DATA_QC_FAILED不升级。原参考分句修复仍漏句，保留；最终改同May训练000段已通过内容检查的云音色，模型/seed不变，完整文本WER0/C6.583。其余11段不重合成/不重净化，最终12段回转文字WER0。内容修复非Sync-C调参或静默删段。历史qwen3-tts-vc旧模型404与3.1首次旧参数format empty均保留；新3.1正确接口实际200。API key只stdin/内存，没有写入代码、文件、argv或BM。

## 数据交付时的边界（历史，当前资格见上）

原生TTS156.56秒/3914音频帧，guard后3849监督行（自然baseline4150）。时长/重复/跳帧属于数据改变，不能称逐帧顺序完全相同；旧frame_order的4150域不能直接索引3849新行，未来需另冻结成对采样映射。仅数据准备，不运行AVE新缓存或真实loader；后续训练前仍需资格，不能复用自然特征。NAS本身优化SyncNet，同模型C门只属技术QC，不证明感知或未来训练收益。旧Patrick coarse7与fine/Chris/Cathy/24full继续暂停，无自动化。


## 后续数据质量诊断（2026-10-20）
用户要求检查后，[[66-may-tts-alignment-loader-audit]]定位原净化映射的长冻结/跳帧问题：3849记录仅2905不同原图，24.532%相邻重复，最长1.8秒有音频/AVE变化却固定同图。累计最大值压平改变15.943%映射帧，无最大冻结/局部质量拒绝门。AVE全量逐位重算、3849特征/pose/eye和108真实collate通过；原中间offset旧值未被真实加载使用，非整体72帧错位。原C>5只属整段同SyncNet验收，不足以认定训练监督质量合格。报告64新旧C差仍为真实观察，但不能据此判TTS路线优劣；0重训，待修订数据后对照确认。

## Observations
- [status] concluded
- [result] 三档新模型Sync-C评测与独立QA及F交付完成；-6dB：新C5.296、自然baseline 8.669、Δ-3.373 / +0dB：新C5.703、自然baseline 8.909、Δ-3.206 / +6dB：新C5.911、自然baseline 9.036、Δ-3.125。
- [verification] 原AVE/initial/优化预算/heldout同报告50；9帧baseline replay、250帧重复、12矩阵/像素/PCM/ROI和F全成员passed。
- [limit] 单May/seed42，TTS时长与图像映射改变；不推感知或总体因果，不恢复旧实验。
- [close_date] October 20, 2026

## Relations
- compares_to [[50-synctalk-may-gain-training-pilot]]
- relates_to [[MFA-linear 与原 TTS 视频对自然音轨对齐小样本试验 2026-09-24]]

- executed_by [[训练 May 的云端 TTS 净化数据对照]]

- reported_in [[64-may-purified-tts-training]]
