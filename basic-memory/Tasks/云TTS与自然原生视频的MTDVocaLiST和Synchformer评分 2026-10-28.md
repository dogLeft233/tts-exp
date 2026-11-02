---
title: 云TTS与自然原生视频的MTDVocaLiST和Synchformer评分 2026-10-28
type: note
permalink: tts-exp/tasks/云-tts-与自然原生视频的-mtdvoca-li-st-和-synchformer-评分-2026-10-28
status: complete
---

# 云TTS与自然原生视频的MTDVocaLiST和Synchformer评分 2026-10-28

用户授权两新模型对之前中文云TTS原生视频评分，统计T胜比例、T-N平均分差与Sync-C冲突比例。不是刚下载的EARS，也不重发云API。

## 冻结设计与入口

Ubuntu /home/wjj/tts-audio/tts-exp/runs/cloud_mtd_synchformer_20261028_attempt01；脚本scripts/experiments/cloud_mtd_synchformer_20261028.py与对应test，两机均有。输入复用avs_u_zh_pilot_20261027_attempt01/inputs.json的80媒体；Sync-C复用cloud_av_preference20_20261027_attempt05/paired_scores.json的60对，20独立来源各3TTS。52媒体不足5秒，仅28可用官方5秒；全组主口径共同原生中央2秒，视频帧格取整，无padding/stretch/norm/最佳lag。

MTD官方权重strict README student_thin_200_all，test_stu RGB0–255 JPEG Resize96下半48与mel hop200，S3FD检测256缩放框回原分辨率；中央50帧作者lastframe=min_length−5约定45窗口；主分mean raw logit、次分mean sigmoid，为项目连续分数归约，不冒充官方retrieval accuracy。
Synchformer官方LRS3权重、官方test变换；2秒改crop_len_sec2/n_segments5，主分softmax零时移P0，次分负期望绝对时移；这也是项目标量adapter，长度适配而非原5秒benchmark。另原5秒14segments合格配对子集单报。两者控制20N真实全PCM±6400samples固定长零填充，同视频同crop；至少32/40下降为PASS，失败照实CONTROL_FAILED，不调端点。置信区间按20来源聚类bootstrap10000 seed20261028，60条并非独立样本，strict12另报。旧Sync-C全片与2秒新支持不完全一致，解读须明确。

## 执行

三条测试（原生支持无拉伸、真实PCM非循环平移、冲突方向/平局/来源聚类）runpy实际PASS。freeze成功80，eligible5s28。MTD当前exec session58829，先缓存80×50帧S3FD人脸，再载模型评分；两阶段不共享显存。Ubuntu外来进程占约8GiB，无kill。另有Synchformer独立venv，之后顺序跑，不改旧环境。尚未完成数值汇总。

## Relations

- follows [[Ubuntu同步评分器部署与EARS40云TTS 2026-10-28]]
- uses [[MTDVocaLiST部署]]
- uses [[Synchformer部署]]

## 执行修正（冻结产物与receipt保留）

MTD中央2s80/80已完成，但主logit时移控制14/40下降，CONTROL_FAILED，不能当同步真值。后续补同一公式的全原生时间支持，非另择端点，原2秒失败保留；amendment在MTD full分数产生前、偏好摘要查看前写入mtd_full_support_amendment.json。MTD full当前session10861，复用中央50帧缓存，只裁额外帧。
Synchformer2秒尝试实际失败：官方可学习位置编码固定198tokens，输入72tokens不能相加；2秒UNMEASURABLE，不插值权重、不凑5秒。另曾遗漏官方源码cwd导致motionformer_src import失败，补os.chdir，execution_repair.json记录。native5s_amendment.json在任何Synch分数前冻结：仅原生>=5秒28媒体，官方14segments，已28/28成功；完整T-N18对/6来源，其余42对不能官方原生评分。累计10自然的20个错位控制，不能套全组32/40阈值，应按冻结80%比例16/20。后续analysis会保留UNMEASURABLE视图，最终仅报告可测5秒子集。两机脚本与三条单元检查已更新并PASS，无云API/TFG重跑。

## 最终交付

MTD full80/80，Synch原生5秒28/28完成，168组原始输出独立标量重算PASS，小JSON/CSV已复制./runs/cloud_mtd_synchformer_20261028_attempt01。正式报告[[68-cloud-mtd-synchformer-native-preference]]。MTD full logit：T胜0/60，平均−1.152499，冲突48/60=80%，control17/40失败；辅sigmoid T胜3/60，差−.080795，冲突51/60=85%，控制29/40亦失败。Synch P0：T胜15/18，差+.012936698，冲突3/18=16.7%，控制20/20通过；18对6来源不可外推全组。共同18对MTD0胜、Synch15胜，分母不能解释全部分歧，但MTD无合格同步控制不能当裁判。没有再做云调用、TFG重生成、旧暂停任务或交叉音轨。

## 用户追加MTD官方调用审计（2026-10-28）

按diagnosing-bugs做可复现对照。原全片17/40控制失败已回放assert复现：BAC009S0002W0122_N_0延后6400samples原logit−4.151115、延后−3.677114。审计脚本scripts/experiments/mtd_official_audit_20261028.py，对应test2项PASS；权威run runs/mtd_official_audit_20261028_attempt01。AST提取作者test_stu.py原封不动的Dataset与calc_pdist，原裁剪框重导出JPEG并直接运行官方Dataset，不仅手工仿写。

前三个样本pilots（字典序自然首3，不按偏好选择）：官方Dataset图像张量逐点等于旧缓存，torchaudio与soundfile波形逐点相同，Mel误差<1e-5；README_all与test_stu普通类strict加载同权重，logit max差0，batch1/31差约4e-6–8e-6。官方31shift retrieval在3条上原生峰−3帧，实际音频−400ms为+7帧、+400ms为−13帧，全部相对变化10帧准确。旧SyncNet这3条av_offset均−2帧，因此原生不是零偏移参考。

进一步full512 S3FD（关cudnn benchmark避免9GB搜索workspace，不改模型）单帧框[137,97,358,372]与旧256框[136,112,358,374]不同。正在3条全512裁剪对照，session59153；第一条结果均值仍不单调，但retrieval峰仍−3/+7/−13。原pilot session8500完成0，结果results.json；full512对照results_full512.json。尚待扩20N的真实PCM±400控制、汇总并明确区分未校准raw均值与官方K5/时移恢复。旧report68与分数不覆盖。

## MTD审计最终完成

正式报告[[69-mtd-official-parity-and-shift-control-audit]]，audit_summary.json已同步两机。20/20给定同框的官方Dataset图像/波形逐点一致，Mel最大差0，复现logit最大差1.40667e−5；README/test两类首3来源logit差0。官方calc_pdist真实PCM±400ms40/40相对时移变化恢复正确，误差0帧，所有N模型原生峰−3，前移+7、后移−13（interior guard20）。按先导固定lag+3诊断，剩余17来源34/34控制降分，总40/40；不逐条校正、不重算TTS偏好。边界剔除的原零lag raw仍12/40，旧全片17/40不覆盖。512框完整3来源对照仍只1/6 raw降分但6/6时移恢复，非主要修复。

纠正：旧均值分数非官方K5，核心call/frontend无错误不等于整套官方协议；人脸检测256适配与评分归约是已知差异。官方原生K5均值4.60%（零附近±1判据与−3峰不同），不能谎称改用K5后零偏移控制已通过。通过的是已知偏移变化恢复。旧T胜0/60不能作MTD官方判自然口型更准确/SyncNet偏置的证据。本轮无TTS全组新端点重评、无API/训练/TFG。session8500、59153、75071均结束0。