---
title: CoGenAV特征重定时TTS视频并换回自然音频的六来源先导 2026-10-25
type: note
permalink: tts-exp/experiments/co-gen-av-特征重定时-tts-视频并换回自然音频的六来源先导-2026-10-25
status: concluded
scientific_status: NATIVE_TTS_GAIN_NOT_TRANSFERRED_BY_FIXED_VIDEO_RETIMING
training: false
tags:
- cogenav
- dtw
- video-retiming
- syncnet
- replacement
---

# CoGenAV特征重定时TTS视频并换回自然音频的六来源先导 2026-10-25

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

| Notes | Modification Date | Approved By |
|---|---|---|
| Initial completed selected-six video retiming experiment | 2026-10-25 | 用户明确要求增强音频驱动视频，再与自然音频对齐比较原生Sync-C |

## 问题与方法
用户检验原生TTS高分能否在视频重定时并换回自然音轨后超过自然视频原生。复用上一轮fresh官方Wav2Lip GAN同头像/设置N/T视频，六来源S0003/0005/0006/0007/0008/0014，来源按历史T>N筛选，非随机样本。Tnative平均7.448，Nnative6.649。

原论文2.1/2.2的对齐是同步训练对的表示学习，没有异语速视频重定时推理。主方法官方CoGenAV Base严格加载，FP32独立audio(N)/video(T)特征、upsampler之前768维、25fps；代价1−ReLUcos，固定DTW为本项目适配。对照CoGenAV audioN/audioT与MFCC13CMVN audioN/audioT DTW(1−cos)、整段等比例。共同闭端点归一化band.25，非对角+.05，diag优先；N索引对应T路径索引中位数，首尾锚定，floor(x+.5)选帧，无额外lag校正/Sync-C调参。

所有条件仅重定时视频、完整自然PCM不变。输出取原N实际帧数187/85/81/124/116/147，特征截至各自视频长度；最近帧重复/跳帧，FFV1 bgr0无损，解码BGR精确等于源选帧。N另走同流程codec对照。无新TTS/TFG/训练/API；18次CoGen前向，30次官方SyncNet pipeline→score。min_track25/batch20/vshift15；全单轨起0，每来源五条件窗口数同184/82/78/121/113/144。官方前端轨迹实际0..n+1，比输入多2尾帧，五条件共同保留，非剔除补帧RF评分；实际评分PCM五条件逐样本相同。原自然mux PCM也逐采样保持。

协议SHA e2169d3a870a75978148e646251d828247bd2b7cddbaaff55789bdd465080cfd，源码/输入/权重/官方前端在运行前冻结。权重仍90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61。

## 结果
官方连续[31,windows]Torch CPU float32、来源等权，所有重定时格子用自然音频评分。
|条件|平均C|Δ原N|Δ同流程N|超任一N来源数|
|---|---:|---:|---:|---:|
|原N|6.649|0.000|—|—|
|N codec|6.660|+0.011|0.000|—|
|CoGenAV AV|5.814|−0.834|−0.846|0/6|
|CoGenAV AA|5.360|−1.289|−1.300|0/6|
|MFCC AA|5.146|−1.502|−1.514|0/6|
|线性|2.183|−4.465|−4.477|0/6|

精确均值N6.648656447728475/Ncodec6.660017490386963/Tnative7.44825291633606/AV5.814390579859416/AA5.359908739725749/MFCC5.146422704060872/linear2.1833949089050293。AV相对原N均值−.8342658678690592、codec−.8456269105275472；六差值全负，最近原N是S0014−.239，最差S0006−1.819。AV在5/6高于MFCC，平均也高于AA及线性，但无一超过自然基线。Tnative使用自己T音轨，仅背景，不能混为本次同音轨格子。

Ncodec−原N均值+.011，单来源−.031至+.067；解码源BGR相同仍存在封装/色彩处理/官方重编码差异，未确认单一原因，两基线结论相同。AV相对codec ΔD+.709、ΔB−.136，C=B−D下降.846；最佳offset AV六条0、Ncodec−2，为评价器lag而非物理真值。

## 核验与产物
根侧不导入执行器，无新增神经前向，独立重算18成本矩阵、DP最优、路径方向/投影、MFCC，矩阵差0、代价最大7.1054e−14；30逐帧源索引/PCM/25fps/PTS、原始官方tracks、activesd逐数组、canonicalC/D/B/offset/日志与共同支持均PASS。父侧3、执行器2契约函数直接PASS，非pytest；首末裁剪montage目视嘴部居中，不作感知同步判断。首次shell因相对cwd找不到脚本，在实际审计前退出；绝对路径首次实际审计PASS，追加直接tracks检查再审亦PASS。无科学失败或重跑。

新scripts/experiments/cogenav_video_retime_20261025.py、cogenav_video_retime_parent_audit_20261025.py及对应tests。首选中文报告runs/cogenav_video_retime_20261025_attempt01/parent_report.md；parent_independent_audit.json最终复核。

完整远端/home/wjj/tts-audio/tts-exp/runs/cogenav_video_retime_20261025_attempt01；${USER_HOME}/.codex/tts-exp-artifacts/cogenav-video-retime-20261025完整本机备份(非网盘)，E仅小元数据。主8201文件773929653bytes，父/子分别全SHA核验PASS，manifest44547bd7e5ead16c34ff21b88a85a25749d1b3f5cc547e1bbf60861d24b27fbb；archive1464369bd8a8115da956b227f2d0c3fdee762a32e2af88db7ec8b0213f244378。父侧7成员及其manifest共8文件单独addendum，SHA0b5073e6ea1c34bcef1167f2b1e3374acdc602203ab4ecfce67d32a4258054c7，C8复制及远端7成员SHA核对，不改主清单。打包一次copytree遇官方weights递归symlink，284KB失败副本保留同级package_partial_failed_attempt01，改源代码专用复制后完成，凭据保留。未提交推送，GPU终态322MiB/0%。

## Observations
- [result] 当前固定CoGenAV跨模态DTW能明显缓解等比例重定时的错配，但六来源均未将TTS原生高分迁移到自然音频评分。
- [limitation] 真实音素路径无真值，帧重复/跳帧可能影响运动；有条件六来源先导，不能推断理想对齐必不可能或证明/否定广义TTS评分偏好，没有人评结论。
- [decision] 所有四方法完整报告，不以Sync-C反调对齐或只挑最高来源；自然波形固定、官方完整支持含共同尾补帧。
- [next_step] 本次授权实验已完成；没有自动启动训练或扩大搜索。

## Relations
- follows [[自然音频已知时间变形的原生驱动SyncNet对照 2026-10-25]]
- follows [[CoGenAV与MFCC及Mel的已知时间变形对齐先导 2026-10-25]]
- relates_to [[让两个音频共享 SyncNet 时间坐标]]
