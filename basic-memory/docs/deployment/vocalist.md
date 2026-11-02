---
title: VocaLiST原版部署
type: deployment
permalink: tts-exp/docs/deployment/voca-li-st-原版部署
status: active
deployment: PASS
---

# VocaLiST原版部署

✅ 原版作者LRS2 5帧权重已在Ubuntu部署并通过真实媒体GPU推理。与蒸馏版MTDVocaLiST分别记录。

## 来源与运行环境

- 作者源码：[vskadandale/vocalist](https://github.com/vskadandale/vocalist)，commit c265b59bcf7cd8559a265aaf454a9333ec24cb0f。
- [官方Release](https://github.com/vskadandale/vocalist/releases/tag/interspeech2022)中002.zip是独立压缩包，882753219字节，唯一成员VocaLiST_Weights/vocalist_5f_lrs2.pth，解压961635703字节；001仅含唱歌任务权重及TensorBoard，不需要合并。
- Ubuntu部署根 /home/wjj/av-sync-eval-20261028，源码vocalist/，权重weights/vocalist/vocalist_5f_lrs2.pth。
- 独立环境 /home/wjj/.venvs/vocalist-20261028；torch2.5.1+cu121，V100 16GB，FP32推理。复用旧环境依赖是只读路径，不修改旧环境。并非声称逐位复现作者2022环境。
- 模型models.model.SyncTransformer()，80106561参数；checkpoint含state_dict、optimizer、global_step、global_epoch，权重strict加载。

## 官方方法与易错处

直接执行test_lrs2.py原文Dataset、calc_pdist、eval_model函数体。原版cv2.imread为BGR，cv2.resize96后除255，取下半48×96；视频窗口5帧，[1,15,48,96]。不可套用MTD torchvision RGB0–255张量。
音频16k单声道，作者sf.read→FloatTensor、80bin Mel、FFT800/hop200、Slaney55–7600Hz、作者dB及[-4,4]归一化，16列Mel窗口。图像和音频样本不做额外增益、时间拉伸或最佳时移校正。
官方K5为31个候选偏移（±15帧）中预测落在零点±1帧的窗口比例；K7–K15对邻近窗口得分平均，容差仍±1帧。保留作者边界-1000掩码及边界平均规则，不能当作平均logit分数。全512 S3FD JPEG人脸预处理沿用同组媒体已有正式裁剪，但必须经原版Dataset重新读取。

## 验证与入口

scripts/experiments/vocalist_official_smoke_20261028.py已修正AST环境os变量；smoke/vocalist_receipt.json PASS，strict_load及author_dataset_verbatim=true，真实GPUlogit3.50795126，峰值342389760字节，2.19秒。对应tests/test_vocalist_official_smoke_20261028.py实际PASS。
组评分入口scripts/experiments/vocalist_official_cloud_rescore_20261028.py，freeze/score/analyze阶段；run runs/vocalist_official_cloud_rescore_20261028_attempt01。与Sync-C配对使用既有同组60对，未改音轨或重算基准。CPU端点两个测试实际PASS，完整结果需看summary.json及independent_audit.json，不能把部署通过等同于科学控制通过。

## 下载故障处理

旧Google Drive空不是原版权重不可得，作者Release自2022即提供备用下载。官方重定向临时URL会jwt:expired，必须刷新地址；高并发在当前网络下可能SSL失败，最终32并发持久连接完成。支持分片续传，稀疏.partial名义大小不能认完成，以DOWNLOAD_COMPLETE、最终ZIP及成功解压判断。不额外重复大文件SHA，不下载无关RVC同名模型。

## Relations

- relates_to [[VocaLiST原版官方Release权重补齐 2026-10-28]]


## 全组评分完成
官方80原生媒体、60配对已完成，正式报告[[71-vocalist-official-native-cloud-preference]]。K5 TTS胜30%、均差−0.937个百分点、与Sync-C冲突60%；全部矩阵独立重算PASS。首3自然±400ms共6控制均准确追踪±10帧。guard20全80原生峰均−3帧、K5低值和边界贡献明显，不能把这些数当感知真值。完整部署与评价任务已结束，不需再启动下载或评分。
