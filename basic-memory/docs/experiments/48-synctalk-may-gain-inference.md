---
title: 48-synctalk-may-gain-inference
type: experiment
permalink: tts-exp/docs/experiments/48-synctalk-may-gain-inference
status: concluded
tags:
- synctalk
- may
- gain
- inference
- syncnet
---

# 48 — May预训练checkpoint的纯幅度推理对照

日期：2026-10-16。✅ 完成；仅推理，0训练。结论限定为一个May身份、官方验证段前12秒。

| 修订 | 日期 | 内容 |
|---|---|---|
| Initial | 2026-10-16 | 三档增益、零档独立重复、原始PCM评分、原始距离矩阵和像素独立复核 |

## 结果

**现成May模型对纯音量增益有实际视频响应；这条片段上+6dB的SyncNet分数小幅提高，−6dB降低。** 四组评分音频都是完全相同的原始PCM。两次0dB的渲染浮点帧、无损视频RGB和评分矩阵均逐位相同，因此处理效应高于本次重复误差。结果不证明普遍最佳音量、训练电平收益或感知质量改善。

预先固定的主分析：同一基线人脸裁剪、31个lag、前后各去20个特征窗，剩余255窗，SyncNet V2 FP32。Sync-D为最佳lag距离，越低越好；D0为原时钟lag0距离。

| 驱动电平 | Sync-C↑ | 相对0dB ΔC | Sync-D↓ | D0↓ | 最佳AV offset |
|---|---:|---:|---:|---:|---:|
| −6dB | 8.906 | −0.356 | 5.927 | 14.280 | −3帧 |
| 0dB | 9.262 | 0.000 | 5.791 | 14.212 | −3帧 |
| 0dB独立重复 | 9.262 | 0.000 | 5.791 | 14.212 | −3帧 |
| +6dB | 9.373 | +0.111 | 5.752 | 14.060 | −3帧 |

全窗口敏感性分析包含官方式边缘零补的295窗，不替换主分析：−6/0/+6dB的Sync-C分别9.020/9.254/9.443，ΔC为−0.234/+0.189；Sync-D为5.868/5.705/5.634。

+6dB主分析的ΔC=+0.110874由背景median距离增加+0.071775及最佳距离下降0.039099共同构成；D0下降0.151650。−6dB背景下降0.220879、最佳距离增加0.135292，D0增加0.068399。最佳offset三档始终−3帧，未消除既有偏移。不能只凭C上升称时序更准确；D0虽改善，也不是人工感知或嘴型标签。

## 视频是否真的变化

使用无损FFV1+PCM正式文件，逐帧解码后与渲染前RGB哈希匹配。相对0dB：

| 电平 | RGB平均绝对差（0–255） | 有变化的像素比例 | 原真实帧嘴部区域平均绝对差 |
|---|---:|---:|---:|
| −6dB | 0.085383 | 6.533% | 1.740656 |
| 0dB独立重复 | 0 | 0% | 0 |
| +6dB | 0.084435 | 6.654% | 1.668910 |

最大通道差为−6dB的120和+6dB的144。嘴部区域由每个原真实帧的68点landmark第48:68点边界加12像素固定，不由增益后输出选择。这里衡量的是口部区域像素变化，未做生成后嘴部运动几何、盲评或准确性测量。

带标签的三屏播放版为 `previews/may_gain_comparison.mp4`，左−6/中0/右+6dB；音轨统一来源于原始PCM，为播放转AAC。**播放版H264/AAC不参与评分**。正式评分与像素端点来自各arm的 `original_pcm.mkv`，其视频FFV1、音频PCM16无损。

## 冻结设计和来源

- 官方SyncTalk commit：`43079fb3f349675f6cb33031bb07262f7566cdd5`；资源来源及完整下载见 [[47-synctalk-amplitude-frontend]]。
- May checkpoint：官方 `trial_may/checkpoints/ngp.pth`，SHA256 `79e5c0056702357e176e9895b237bfcab1a517b85f4e468586282954e7379e79`。全部学习参数加载；只有官方best保存时明确删除的训练用density_grid未载入，推理用density_bitfield实际载入且非零。没有随机缺失学习参数。
- 官方May validation的前300帧，img_id 5520–5819，25fps、512×512、12秒；不是按分数选片。保留官方val JSON，通过data_range=[0,300]选前缀；同pose、eye、背景、torso、个体码、checkpoint、渲染参数和codec。
- 完整原243秒双声道16k PCM16已核验两声道逐位相同。驱动采用全段单声道float64乘10^(dB/20)，写DOUBLE WAV，官方AVE从WAV重新提取，维持全段上下文及原aud_id映射；不乘已有缓存、不加limiter、不做RMS/峰值归一化。
- 全段驱动峰值：−6dB=0.238372895、0dB=0.475616455、+6dB=0.948979589，全部<1。每个DOUBLE文件读回与纯标量公式逐位相等。
- 评分PCM严格取原文件第一声道samples[3532800:3724800]，恰好192000样本/12秒，无时移、伸缩、淡入或增益。PCM内容SHA256 `68ffa2de3aa08083933670a54ca6471a11d293d90de4c64a702d73143903ada9`；四个正式mux逐样本核验相同。
- arm顺序固定−6/0/0独立重复/+6；每次重建模型和数据集、同seed0。使用官方-O对应FP16/exp_eye、cuda_ray、max_steps16、dt_gamma1/256，非portrait、原帧torso背景，perturb=False。每组300帧正式渲染约21秒，不包含首次编译/音频编码准备时间。
- 已先运行一个0dB首帧smoke确认模型与图像有效，之后完整四组。没有调电平、挑帧、扫描offset或为获得正结果修改前端。

## SyncNet合同与核验

评分代码在 `scripts/experiments/synctalk_may_gain_inference_20261016/score.py`；冻结合同 `score_protocol.json` 在新分数出现前写入。

1. 在0dB无损视频上运行现有S3FD，默认scale0.25、conf0.9、取最大人脸；预留的scale1 fallback与缺失插值都未触发，300/300检测成功。按官方几何，s/x/y分别medfilt13、crop_scale0.4，resize224。同一套逐帧track用于所有arm，没有各臂独立裁剪造成的混杂。
2. 正式输入直接来自FFV1解码BGR并裁剪，不经过历史pipeline的JPEG/XVID。这是共享裁剪、无损图像合同，**不把绝对分数与历史默认压缩链排名混为一谈**。
3. 现有SyncNet V2权重strict加载、FP32 eval、禁用autocast/TF32。原始PCM生成一次MFCC和音频编码，所有arm复用同一真实音频编码；不同arm各自实际运行视觉编码。
4. 遵循官方min_length−5：300帧产生295个5帧视频/20步MFCC窗口。vshift15，保存[295,31]原始距离矩阵；主guard20保留255窗，所有lag均有真实支持。C=median(平均距离曲线)−min，D=min，offset=15−argmin；另保存D0、median背景及全部曲线。
5. 独立验证读取原始May PCM、四个实际driver和四个实际无损mux，重建纯标量关系、核验原始PCM与1200帧RGB；用保存的音/视编码以NumPy float64重新计算36580个距离单元，最大误差3.9753e−6；重新计算全部full/guard20评分，最大误差1.4818e−6。误差来自FP32/64计算路径，均小于预设容差。
6. 0dB重复的300个浮点帧哈希、RGB帧、视觉编码及距离矩阵均逐位同，评分矩阵最大差0。增益臂像素和评分变化均超过此控制。

3项合同测试实际通过：拒绝将加增益PCM用于评分、拒绝原音频循环移位、保证边缘分数不能替换guard20主分数。完整结果见 `independent_validation.json`，不是只检查文件存在。

## 环境与空间处理

Ubuntu根盘起止约355/354MB空闲，V100初始空闲。所有新增CUDA编译器、GCC、wheel、数据和生成结果均放独立SHM目录，没有改已有模型环境，没有清理历史实验，也没有再次向根盘写缓存的百度盘上传。

- NVIDIA官方channel CUDA12.1.105 nvcc、cudart开发头、CCCL12.1.109、driver开发头，下载来源与SHA在resources清单。
- 系统GCC13不兼容旧CUDA前端；从Ubuntu官方仓库下载可解压的GCC12.4包，逐包对照apt元数据SHA512，解压到SHM，不执行系统安装或sudo。
- 新依赖仅在任务隔离目录：trimesh4.5.3、tensorboardX2.6.2.2、PyMCubes0.1.6、torch-ema0.3、lpips0.1.4、ninja1.11.1.3。只用于导入和编译，没有实例化额外LPIPS或landmark下载模型。
- 渲染采用原已资格torch2.5.1+cu121/librosa0.11.0；评分采用已有syncnet环境torch2.5.1+cu124。源码、权重、依赖版本和编译so哈希在 `environment_snapshot.json`。
- 官方代码工作副本仅4个backend.py的C++14→C++17及编译器兼容flag改动；网络、渲染算子、音频前处理、provider均未改。完整diff在 `compatibility.patch`/environment_snapshot；编译实际使用隔离GCC12和V100 sm70。没有更改数值算子以追逐分数。
- 编译早期失败依次为wheel解压后ninja路径、CUDA头文件可见性和GCC13兼容；补齐只读头文件并隔离GCC12后全部通过。评分attempt01因S3FD相对权重路径未切仓库目录停止，未产生分数；修复cwd后的attempt02为canonical，未改模型或科学合同。失败日志保留。

## 产物与复现

**持久本机目录**：`E:/Documents/tts-exp/runs/synctalk_may_gain_inference_20261016/`。其中保留正式无损视频、共享裁剪、音/视特征、距离矩阵、逐帧像素统计、协议、资源、验证和日志。

**Ubuntu完整运行目录（易失）**：`/dev/shm/synctalk-may-gain-inference-20261016/`，canonical为 `render_attempt01/`、`score_attempt02/`；播放版在 `previews/`。根盘另保存小型报告/协议/收据入口于 `/home/wjj/tts-audio/synctalk-may-gain-inference-20261016/`。SHM不能当永久备份。

完整有效结果压缩包已传回本机，Ubuntu与本机SHA256均为 `be3b866d8f3dea438d61b7bf3265e35ac9f4b2b8f4fef321c00d1448630161e8`。包保留四组正式FFV1+PCM视频、全部score数组和metadata；排除重复的无音轨FFV1及可由已存源PCM精确重建的全段driver/data软链接，避免无意义重复占用空间。实际driver在Ubuntu独立验证后保留于SHM；原May全包和源WAV在本机此前持久资源中。

```bash
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
# 只读重建核验，不运行生成器或评分网络
$A/env/bin/python $R/validate.py --root "$R" --output "$R/independent_validation_reproduction.json"
# 新推理必须使用新输出目录，保持原产物不覆盖；环境见run_render.sh
```

首次重建完整环境与执行步骤见同名脚本目录README。后续若研究跨片段/身份或训练电平，应另冻结设计和预算；本次没有启动训练，也没有恢复暂停的其他研究路线。

## Observations

- [status] CONCLUDED：四组视频和SyncNet评分完成，0训练。
- [result] May单片段上−6dB ΔC−0.356、+6dB ΔC+0.111；原自然PCM不变，重复误差0。
- [conclusion] 纯幅度在该已训练SyncTalk AVE生成器上有真实视频及评价响应；不证明普遍质量提升或训练收益。
- [report] `runs/synctalk_may_gain_inference_20261016/report.md`，raw scores SHA256 `93cab5e5d884aeedf66b05328f296ba7d4733c94244e518b4a5710fce6856ca8`。

## Relations

- follows [[47-synctalk-amplitude-frontend]]
- relates_to [[46-amplitude-gain-independent-validation]]
- relates_to [[研究主线与证据地图]]