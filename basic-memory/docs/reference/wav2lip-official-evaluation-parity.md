---
title: Wav2Lip官方评分路径与当前项目流程对比
type: reference
permalink: tts-exp/docs/reference/wav2lip-official-evaluation-parity
date: 2026-10-31
status: source_audit_complete
---

# Wav2Lip官方评分路径与当前项目流程对比

✅ 已完成源码和封存运行凭据审查。未运行新评分、训练或生成；未恢复用户暂停的HDTF实验。

## 结论与范围

当前链可以表述为“采用官方SyncNet V2完整人脸处理及LSE公式，附项目输入音轨规范和min_track25设置”。不能表述为“与Wav2Lip论文全部评分条件完全一致”。

比较对象为近期报告75的EARS40及报告76/77/79的中文九评分器SyncNet链，并检查10月31日HDTF控制器。根目录scripts/04_eval.py仍是另一入口，默认min_track100，不应代表所有近期实验。本文只比较生成视频之后的LSE-C/LSE-D流程；Wav2Lip论文FID是视觉分布指标，不属于这里的同步评分，也不等于目前九评分器整套协议。

依据：Wav2Lip官方仓库commit bac9a81e63ecc153202353372e5724b83d9e6322；其引用的joonson/syncnet_python当前commit 907c0b579c2e2d83f0eae1b2ac9e720cde4e5623；另比对2020最后版本6efbb1c。完整源码与逐文件SHA在runs/wav2lip_official_eval_audit_20261031/official_sources及official_source_manifest.json（63文件）。本次核验audit.json保存项目封存入口SHA和短视频计数。

## 官方实际上有两条路径

[官方evaluation说明](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/README.md)明确区分：

1. LRW/LRS2/LRS3测试：calculate_scores_LRS.py → SyncNetInstance_calc_scores.evaluate。对输入视频解码、逐帧resize224×224，**本评分脚本不重新做人脸检测或追踪**。这不能推断数据或生成前从未裁脸。
2. ReSyncED或自己的生成视频：calculate_scores_real_videos.sh → run_pipeline.py → calculate_scores_real_videos.py → 同一评分类。先25fps转换、镜头检测、S3FD脸检、追踪和裁剪，再评分。我们近期链属于这一类。

[官方real shell](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/calculate_scores_real_videos.sh)第6–7行、[LRS入口](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/calculate_scores_LRS.py)第49–60行可直接核对。

论文4.2.1使用的是公开预训练SyncNet，不是Wav2Lip训练用的expert discriminator；当前syncnet_v2.model属于正确指标家族。[论文](https://arxiv.org/html/2008.10010v1#S4.SS2.SSS1)

## 逐环对比

| 环节 | 官方real-video路径 | 近期项目 | 判定及可能影响 |
|---|---|---|---|
| 评分输入音轨 | 直接读取提交视频内的音轨 | 保留视频流，替换为对应规范PCM16/16k/mono WAV，MKV封装 | 明确项目协议差异；若生成原MP4为AAC，解码PCM与延迟可能不同 |
| 25fps转换 | ffmpeg -qscale:v 2 -async 1 -r 25 → video.avi | 调用相同代码 | 一致；不是只修改fps标签，可能实际丢/复制帧 |
| 音频提取 | 从上述video.avi取mono PCM16/16k | 相同 | 一致；video.avi创建时未指定音频编码，不能假定全程无损 |
| 帧提取/压缩 | AVI→JPEG(qscale2)，裁剪后XVID AVI，评分时再JPEG解码 | 相同 | 一致；重复编码是官方链的一部分，会改变像素 |
| 检测器 | S3FD、facedet_scale .25、conf .9、NMS .1 | 相同源码/默认值 | 一致；不是直接复用生成模型的脸框 |
| 分镜 | ContentDetector划分镜头 | 当前官方2026实现 | 当前版本一致；与2020的SceneDetect API及依赖不同，边界不保证逐帧相同 |
| 追踪 | IoU>.5，允许检测帧间距≤25，框线性插值，平均脸大小最大维>100像素 | 相同 | 一致；检测漏帧、镜头边界仍决定实际支持 |
| 最短轨迹 | 默认100，实际条件len(detections)>100 | 25，实际>25 | 明确差异，影响可评分集合/短轨纳入 |
| 框平滑 | 中心x/y及半尺寸s分别13点median | 相同 | 一致；生成器的nosmooth不关闭本步骤 |
| 裁剪 | crop_scale .4，偏向下方扩展，灰值110补边，resize224×224 | 相同 | 一致；裁的是扩展人脸，不是单独嘴部或关键点仿射对齐 |
| 音频裁切 | 按track首末帧/25裁切到毫秒，crop AVI音频copy | 相同 | 一致；评分支持由轨迹决定，并不保证完整句 |
| 网络视频输入 | 224×224 BGR、5帧；无额外灰度、除255或嘴部裁剪 | 相同 | 一致；W2L评分类另有一次resize224，已224时通常恒等 |
| 网络音频输入 | scipy读int16幅值→python_speech_features默认MFCC；每视频步取4k:4k+20 | 相同 | 一致；不是Wav2Lip生成器mel，不做额外RMS/LUFS标准化 |
| 共同长度 | min(视频帧数,floor(PCM样点/640))，窗口起点range(min_length−5) | 相同 | 一致；保留官方尾端少一窗的行为，不改为−4 |
| 偏移搜索 | ±15帧共31候选、边缘音频embedding零填充、欧氏距离 | 相同 | 一致；不是只评零偏移，不剔除边缘lag |
| 句级标量 | 先对各lag跨窗口平均；D=min，C=median−min，offset=15−argmin | FP32距离矩阵按同式重算 | 一致；不是逐窗C平均或逐窗最小值平均 |
| 轨迹汇总 | real脚本每条0*.avi输出一行D/C，不自动选主讲者 | 当前控制器assert只有一个track | 单轨适配；多轨时当前链失败，不隐式择最高分 |
| 输出精度 | W2L脚本返回numpy标量，upstream日志显示3位 | 近期保留矩阵全精度，展示3位；04_eval解析3位日志 | 近期无3位量化；04_eval存在最多约.0005的单标量舍入 |
| 数据集统计 | LRS逐视频等权均值，real shell仅收集track行 | 多draw先按来源聚合，再配对及bootstrap/Holm | 属实验设计差异，不能与论文表格直接等同 |

预处理参数依据：[当前官方run_pipeline](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/run_pipeline.py)，第24–33、68–105、112–178、187–241、267–310行；距离与输入依据：[Wav2Lip评分类](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/SyncNetInstance_calc_scores.py)，第22–32、53–157行。

## 实际运行证据

从results/talklikeyou/chinese20_rms24_20261030_attempt01/evidence.tar.gz读取实际封存code_snapshot，而不仅查看设计脚本：

- 036_SyncNetInstance.py：bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6
- 037_run_pipeline.py：3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf
- 024_run_syncnet.py：c78fe9f75665aa9e7a2bb523e4e7e3734a857137ff70b29c88a7b4893c422ded

三文件与本次下载的官方2026源码逐字节相同。项目10月29日source快照中SyncNetModel.py也与当前官方同SHA（41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22）。这不等于已逐字节验证2020所有依赖及权重正文。

实际stage receipt表明以--min_track 25 --overwrite运行，returncode0。scoring_inputs.json的80逻辑视频最短57、最长193帧，其中28条≤100帧，即使每帧检测成功也无法通过官方默认>100检测门。这里28包含同音频/视频别名，不能当28个独立样本。门槛改动的主要影响是纳入集合；对于同一条超过100检测且其它条件相同的轨迹，25本身不会改变其框或网络公式。

当前控制器：scripts/experiments/talklikeyou_chinese20_20261030/controller.py第58–79行；HDTF同名控制器同段；EARS在scripts/experiments/ears40_w2l_nine_scorers_20261029.py第192–218行。根04_eval.py第63–112行走完整官方两步，默认100，但parse_syncnet_output仅匹配首条track，offset正则不接受负号，cache也不绑定媒体/源码哈希；这些是通用入口的潜在限制，不适用于近期assert单track/矩阵归约链。本文未修改代码。

## 音频与时钟需保留的边界

项目当前先remux规范PCM并核对decode PCM及视频PTS；这确保输入官方管线时的配对，不意味着后续官方链不改变PCM。官方video.avi步骤未写-c:a copy或显式codec；实际编码依赖ffmpeg构建和默认值。应检查真实video.avi的codec、解码波形、起始PTS及相对输入的延迟，不能仅凭后续-c:a copy认定无损。

2020到2026的run_pipeline保持核心检测阈值、跟踪、median13、裁剪与ffmpeg实参，但SceneDetect换API、无分镜回退结束时间也换写法；加上OpenCV/FFmpeg/PyTorch/CUDA差异，不能宣称论文时期bitwise复现。近期部署记录明确是torch2.7/A6000，旧基线torch2.5/V100；FP32张量也不自动意味着cuDNN禁用TF32。现有gpu_guard不改全局TF32。

部署下载凭据记录S3FD来自作者Oxford sfd_face.pth，SHA d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491；另有face_alignment的s3fd权重供其它流程，两者不能混用。本文未重新下载大权重、逐tensor比较或实时核验远端软链，因此不将文件名/下载凭据升级为论文时期权重完整同一性的证明。

## 如何解释当前分数与后续最小核验

C是搜索最佳lag后的对比度，D也是该最佳lag处的平均距离；偏移视频仍可能高C。因此“C高但肉眼不同步”本身不证明程序走错：应同时看offset和零偏移距离。中位数9的framewise confidence平滑不参与最终C；13点框平滑则直接改变网络图像。

建议当前方法描述明确写“官方SyncNet V2 real-video pipeline，min_track25，规范PCM替换音轨，FP32矩阵归约”。若要补论文默认敏感性，固定现有视频分别比较100与25的覆盖，严格单独报告共同合格支持；不要把短片失败计作低分、补帧或为通过门延长视频。另固定视频比较原MP4内嵌音轨与规范PCM输入，测官方中间音频实际变化；检查bbox轨迹、首末帧、MFCC和31lag曲线，而非只看两个最终数。LRS直接resize路径只能作为另一个前处理对照，不应当成任意全画面视频的默认修复。

以上是核查建议，未执行新GPU评分；HDTF用户暂停状态不变。

## Relations

- relates_to [[syncnet]]
- relates_to [[79-talklikeyou-chinese20-nine-scorers]]
- relates_to [[75-ears40-english-wav2lip-nine-scorers]]