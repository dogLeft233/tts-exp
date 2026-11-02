---
title: server192-nine-scorers
type: guide
permalink: tts-exp/docs/deployment/server192-nine-scorers
tags:
- deployment
- server192
- scorers
---

# 192服务器九评分器与数据部署

✅ 2026-10-30：SoulX-FlashHead Pro、九评分器和既有中英文规范音频均在个人数据区完成部署。九项均在新生成p002视频上实际前向；该阶段仅部署验收。随后完整中文20实验已完成，见下节与报告76。

## 完整实验已完成（2026-10-30）

中文20来源80媒体与九评分器、归约审计和本地SHA/统计复算已完成，见 [[76-flashhead-chinese20-nine-scorers]]。主要同句参考12人未确认Holm9后效应；全20人Sync-C+0.212，PEAVS偏自然（Holm9 p=.04131）。两个5秒分类器实际完整配对7人，其他七项20人。输入统一RMS−24dBFS，为用户本轮新条件，未拼接旧低增益N分数。SyncReward官方支持短片内部片段补齐，4.75秒不是其loader最低时长；下文早期部署指南的该门槛由本条更正。80视频/九评分器结果在jobs/flashhead_chinese20_rms24_20261030_attempt01；本人任务已结束，不重复部署或运行。

## 当前执行范围：仅中文20人（2026-10-30用户确认）

用户对先前异步问题的答复明确为“只准备中文20人”，覆盖此前等待后按中英双队列准备的假设。后续新TFG任务只使用中文20来源的20N+60既有TTS draw，共80音频；英文EARS40从后续运行清单排除，既有迁移凭据保留。远端 /home/data2/wangjiajun/av-evaluation/active_cohort.json 和本地交付同名文件已建立，服务器逐条SHA回查80/80 PASS、唯一来源20。README已同步范围。

用户同时确认原评估机已开机；实际 BatchMode SSH wjj@192.168.50.2 成功，hostname=wjj-Default-string、whoami=wjj、pwd=/home/wjj。本次仅只读核验，不重新迁移已通过的环境，也未启动完整实验。

## 使用入口
TFG根为 /home/data2/wangjiajun/SoulX-FlashHead-Pro，评分根为 /home/data2/wangjiajun/av-evaluation。先读取server-192 skill及飞书手册、实时查看GPU占用，再设置GPU_UUID并以nohup/tmux调用 run_scorer.sh。包装器限CPU4线程、单UUID和torch显存60%，用互斥锁避免本评分入口并发。所有环境、权重、cache、临时文件和输出均在个人目录，共享conda只用于创建个人环境，未改共享base或系统驱动。

可执行命令、各环境冻结清单与细节在仓库 results/soulx_flashhead_pro/server192_nine_scorers_20261030/README.md；远端README.md同内容。机器可读总凭据 deployment_validation.json；本地evidence/是远端证据归档回读，SHA256为78d6ab354ed8ccd0f34a55bc46c1882814aa3f74b4b6b2be3d92fc95626e2298。该归档约8MB，含视频、原始矩阵/embeddings/logits、日志、代码及资产下载receipt。源码scorers/source_manifest.json验证476blob，PEAVS六权重另有Git blob清单。

## 数据与验证
deployment_input_manifest.json保留原路径与新路径映射；data_validation.json为160条音频/271文件全部大小和SHA PASS、missing0。英语40人80条N/T；中文20源20N+60已有TTS draw，三draw不作为独立来源。固定face为3.png。中文原规范自然音频通过本机原归档按原流程恢复，最终全20文件SHA和旧清单相同；全部60TTS也相同，没有新合成或替换样本。旧清单和冻结科学代码不回写。

新片为512×512、25fps、262帧，驱动音频10.44325秒。评分ALAC MP4保持原视频流，167092个解码PCM样点与原16k PCM16 WAV逐点相同。

| 评分器 | 沿用口径 | 部署单片结果 |
|---|---|---:|
| SyncNet V2 | 官方完整FP32 pipeline Sync-C | 7.914 |
| CoGenAV | 全片Eq1 mean ReLU cosine | 0.5179972053 |
| PEAVS | 官方batch64配置、本验收一条 | 2.9043863 |
| SyncReward | 官方BF16、encode1、中心4.75秒 | 1.6484375 |
| AVS_u | 项目论文定义复现、全片frame cosine | 0.0809920499 |
| MTDVocaLiST | 作者K5、31候选±1帧恢复 | 0.67578125 |
| VocaLiST | 同上 | 0.6328125 |
| Synchformer | 官方LRS3、整数中心5秒21类P0 | 0.0002663291 |
| SparseSync | 同上 | 0.0000040338 |

MTD/VocaLiST的原始31候选矩阵经独立CPU归约一致；真实完整音频平移±400ms后内部偏移从1到11/−9帧。两分类器原片argmax+0.2秒，真实时移后+0.6/−0.2秒，完整logits保留。AVS_u真时移后frame均值0.054983/0.037334。CoGenAV仍仅extraction/forward通过，不把EXTRACTED_NOT_YET_QUALIFIED当作总体qualification。各分数不共尺度、不投票等同感知真值。

## 兼容边界
基础环境用FlashHead完整conda clone，冲突包以venv隔离；原.pth复用造成FlashHead NumPy卸载，已恢复1.26.4并通过完整新片生成验证。旧不完整环境eval-base_incomplete_pth禁止使用。torchcodec进程预加载系统libffi7，PEAVS使用系统CUDA12.1 NVRTC，FFmpeg为个人FFmpeg7；没有系统修改。PEAVS VGGish的源码固定cache通过个人symlink连接已校验权重；S3FD按SyncNet/Wav2Lip分别绑定正确格式。

MTD/VocaLiST采用视频/PCM共同261完整帧前缀，记录排除末1帧，原片保持不变，不补静音不校正音轨。分类器中心68帧/43520样点，125帧/80000样点；先平移完整音频再切窗。旧AVHu的argv==1触发notebook绝对导入，适配器确保正常package分支。当前torch2.7/A6000与旧torch2.5/V100未验证逐位一致；matmul TF32=False、cudnn TF32=True已记录。

全部本人推理/下载进程已结束，GPU释放。该部署验收结束时，完整队列尚未运行；随后中文20完整实验已完成并另存报告76。实际批量须固定生成设置与失败规则；PEAVS80条按64+16固定批次，不能用逐片部署示例替换批量口径。

## Observations
- [result] 九评分器全部实际前向PASS，160条规范音频同SHA就绪；功能验收不等于总体效果结论。
- [decision] 保留既有科学定义和原始证据，新部署适配单独记录；不修改历史执行路径来冒充原执行。
- [next_step] 完整中文实验已交付，后续从报告76及不可变证据包复核；英文未纳入，不自动重跑。
- [changed_file] 全局server-192 skill新增references/personal-deployment.md并通过quick_validate，记录隔离环境、数据SHA、私人FFmpeg和进程级兼容路径。

## Relations
- uses [[soulx-flashhead]]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Verify nine scorer deployment and 160 canonical audio files on 192 | October 30, 2026 | user |
| Apply delayed user choice: Chinese20 only; verify original evaluator host online | October 30, 2026 | user |

| Complete Chinese20 audible-gain full experiment; link report76 and clarify official short-clip support | October 30, 2026 | user |
