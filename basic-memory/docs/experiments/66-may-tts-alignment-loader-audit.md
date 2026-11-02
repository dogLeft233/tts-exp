---
title: 66-may-tts-alignment-loader-audit
type: experiment
permalink: tts-exp/docs/experiments/66-may-tts-alignment-loader-audit
status: concluded
phase: DIAGNOSIS_COMPLETE
result: AVE/cache/loader passed; unconstrained NAS timewarp degeneracy found
outputs: F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1
---

# 66-may-tts-alignment-loader-audit

## 结论
用户看到报告64新TTS训练test0 C5.703低于自然baseline8.909后，要求先检查净化音画对齐、AVE及训练加载。本轮发现明确的训练数据时序退化：不是读错文件或AVE缓存错配，而是原NAS短窗映射允许大量冻结/跳帧，验收只检查整段C>5，没有约束映射连续速度和局部残余偏移。此结果不能量化该缺陷解释了C下降多少，也不能据此判定TTS训练总体有害；本轮0生成器构造/0optimizer/0重训。

## 音画映射与数据内容
封存416成员全部fresh SHA通过；最终12段DOUBLE音频逐样本重拼与实际训练WAV完全一致。最终audio id连续40..3888，共3849行；保存的全3914帧mapping按audio id查询与每条训练img_id完全一致。3849条相机/几何JSON除重写aud_id外逐字段同原source，未跨heldout。原中间chunk receipt在修复001段长度后，部分global_audio_start_frame元数据仍旧；最终拼接与JSON是从各段长度重新累计，真实loader不读旧offset，确认不是72帧整体错位，原元数据不偷改。

监督仅覆盖2905个不同源图，是自然baseline4150张的70.0%。相邻重复24.532%，跳号>1占12.136%；最长同一图45帧/1.8秒，最大相邻源帧号差34帧/1.36秒（不是声称物理时间偏移1.36秒）。图像重复/跳变是数据处理实际产物，未被此前全局Sync-C门拒绝。

## 定位到的机制与实际例子
从冻结NAS源码和每帧correction_ms独立重建：round(ms/40)加到目标帧、clip边界，然后对倒退索引用累计最大值压平。12段全部映射逐项精确复现；624/3914帧（15.943%）被该单调压平改变，没有最大冻结长度/最大跳帧拒绝门。

最严重样例在第008段、全局TTS音频帧[2641,2686)，对应105.64–107.44秒：训练标签始终为source2944，持续1.8秒。该段原音频RMS0.068597（全段实际幅度，非静音标记或人为增益）；AVE跨时间变动RMS0.514054。完全处于冻结段内部的41个5帧SyncNet窗口，视觉embedding只有1种，音频embedding有41种；说明音频条件变化时监督视觉被固定，不能把这个例子解释为音画文件读取正常就等于监督合理。原source2944实际图与原始波形/AVE变化图已view_image检查；3.4秒原音轨片段hold_case_preview.mp4仅供观看，H264/AAC未用于评分，没有声称agent听过或盲评通过。

保存after距离上另做12帧/2帧stride的局部诊断：1621短窗中67.613%最佳lag为0，14.682%绝对lag>3帧；而12整段的最佳lag全部0。短窗SyncNet会有估计噪声，不能当独立物理对齐真值，但进一步说明整段C>5不能代替局部对齐验收。未根据该诊断改图、改分数、调搜索参数或删段。

## AVE复核
实际训练WAV用原预训练AVE权重、原前端/80Hz mel→25fps/两端各重复2帧重新全量前向3915×512；和训练时cache逐位相同，F读回数组再次逐位核对。float64 WAV→float32读取逐样本相同，无额外RMS/zscore/clip；原weight SHA1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4。取样mel窗口按独立时间公式int(80*(aud_id−2)/25)、16mel帧验证一致，未发现25fps/80Hz、加2帧或旧自然cache混用错误。

TTS被监督AVE RMS0.864613，映射到同source自然AVE RMS0.827853；ReLU零比例54.949%与54.838%。这些只证明幅度/统计分布不同且没有全零/非有限，不证明TTS语义表示与自然等价。

## 真实加载复核
用本次实际冻结provider/common重新构造train3849，不实例化生成器。全部3849中心特征、全部8帧上下文、所有pose/eye/图像路径匹配；audio id连续，未因删行压缩内部时钟，只有监督两端沿用baseline零填充。对108个真实collate（含所有拼接边界和严重冻结两端）独立核对原图实际ray采样像素、pose、bs/eye、landmark face框、torso alpha与背景、音频上下文，全部通过。旧资格证明“加载的数据与准备文件一致”，没有验证准备后的时间映射足够合理，这一边界需要纠正。

## 验证与建议边界
3项实际产物测试直接执行通过：新全量AVE与原训练artifact相同、真实视觉冻结与音频变化、旧中间offset不冒充最终时钟；无pytest运行声明。运行源/训练源fresh SHA保持，原数据/断点/报告64所有分数不动，旧Patrick等保持暂停。预览首次使用系统Python缺依赖，在输出前失败；改用既有lexical环境，无安装/无GPU训练重放。

先处理不受限的单调压平和局部残余偏移，再验收数据；修正后若做等预算对照训练，才能估计此问题对−3.206分差的贡献。当前检查不能排除TTS发声/韵律与自然测试分布差异，也不能把所有差异归因于这一个机制。没有自动生成修订数据或启动训练。

权威根F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1：cpu_data_audit.json、mapping_mechanism_audit.json、loader_ave_receipt.json、recomputed_ave.npy、local_residual_sync.csv、hold_case_diagnostic.png、hold_case_preview.mp4和test_receipt.json。

## Observations
- [status] concluded
- [result] 净化映射存在明确时序退化：24.532%重复、最长1.8秒非静音能量段固定同一图，累计最大值压平机制逐项复现；原C>5质量门不充分。
- [result] AVE全量重算逐位同；3849中心/上下文/pose/eye与108真实collate通过，未发现缓存或loader索引错误。
- [limit] 尚无修订数据重训，不能量化时序退化对Sync-C分差的因果份额；局部SyncNet不是物理/感知真值。

## Relations
- follows [[64-may-purified-tts-training]]
- diagnoses [[May 云端 TTS 净化训练数据对照]]
