---
title: CoGenAV与MFCC及Mel的已知时间变形对齐先导 2026-10-25
type: note
permalink: tts-exp/experiments/co-gen-av-与-mfcc-及-mel-的已知时间变形对齐先导-2026-10-25
status: concluded
scientific_status: KNOWN_SYNTHETIC_CLOCK_NO_COGENAV_ADVANTAGE_OVER_MFCC
training: false
tags:
- cogenav
- dtw
- alignment
- known-warp
---

# CoGenAV与MFCC及Mel的已知时间变形对齐先导

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

| Notes | Modification Date | Approved By |
|---|---|---|
| Initial bounded experiment and independent audit | 2026-10-25 | user explicitly requested known-stretch/pause comparison on TTS Sync-C>N examples |

## 问题与冻结设计
此前论文解释区分了表示对齐与时间路径搜索。用户明确授权新小实验：在TTS比自然Sync-C高的已有样本上，比较CoGenAV音频特征+DTW与MFCC/Mel+相同DTW找回已知时间变化的精度。没有新TTS/TFG/SyncNet或训练。

20既有来源中12满足至少一合格TTS draw1 C>N，按来源编号取前6，优先L06否则L17。实际S0003、0005、0006、0007、0008、0014，自然与所选TTS共12原始音频，历史ΔC分别+0.676/+0.626/+0.248/+1.316/+0.273/+1.663。按结果条件选择的先导，不代表随机总体，也不按分差最大挑例。

各音频identity/stretch/pause/combined共48信号、48真实CoGenAV官方Base audio-only pre-upsampler FP32前向，严格原权重。三个区段phase-vocoder速率0.8/1.25/1，nfft512hop128；第一切点选原音频约三分之一±150ms最低40ms能量点，第二点三分之二；新增320ms静音。各段实际样本长度提供仿射目标时钟，不是独立音素事件真值。统一40ms网格，原始RMS>.01max能量支持，排除内部切点±80ms；插入静音无原始对应。固定共同DTW：闭合端点、归一化index带宽.25、diag/h/v、非对角+.05、平局diag优先，无连续步数上限。CoGen原生特征cosine，MFCC13/Mel80逐音频逐维CMVN后cosine，比较固定特征流程，不是纯编码器权重消融。

协议SHA de16fb137798e032daea8124ea22968476fa0bd5ad01cde930a8d0690b98e233，执行源码SHA16fae2a5661f2744af94949965c684eaa7d6fc9fe15b24d03ee59ecdcb191c38。无事后调参。

## 结果
来源等权forward能量支持MAE，先平均各来源两臂及条件：
|变形|CoGenAV ms|MFCC ms|Mel ms|整段等比例 ms|
|---|---:|---:|---:|---:|
|stretch|10.161|9.566|9.007|128.285|
|pause|1.359|0.000|5.041|87.500|
|combined|10.779|9.972|14.563|178.289|
|三条件平均|7.433|6.513|9.537|131.358|

CoGen比MFCC平均多0.920ms，只1/6来源更小；三条件单元平均>80ms比例CoGen0.339%/MFCC0.047%/Mel0.957%。父侧逆向路径诊断平均CoGen6.929/MFCC6.657/Mel7.666ms。identity全方法浮点零。nearest40ms target-frame oracle_grid不是数学误差下界：matched-frame中位数可落20ms半网格。320ms恰好8格，是容易的整格停顿控制，不外推任意停顿。

6对真实N/T三种路径已保存，CoGen与MFCC平均路径分歧按来源9.9–28.7ms；无真实时间标注，不以此判断谁正确，没有真实波形重定时、replacement或新Sync-C收益。

## 核验与变更
ARS实验工作流code_runner子代理执行。新增scripts/experiments/cogenav_dtw_pilot_20261025.py、tests/test_cogenav_dtw_pilot_20261025.py；父侧scripts/experiments/cogenav_dtw_parent_audit_20261025.py及对应tests。父侧未导入执行器，独立重建全部时钟/支持/144路径预测/误差/来源聚合，差0；独立动态规划最优代价与保存路径差max1.421e-14，另18真实路径首尾/单调/投影核验通过。3执行器+2父契约函数直接断言PASS，未pytest；两PNG实际目视。

父审计初版等价浮点运算顺序在末端静音帧/恰好40ms阈值断言失败，调整父审计算术顺序后通过；原科学协议、源码、数组、指标不变。首次tar运行内目录元数据变化返回1，保留记录，最终在/tmp打包exit0，无科学重跑。

## 资产与边界
权威中文报告./runs/cogenav_dtw_pilot_20261025_attempt01/parent_report.md；完整${USER_HOME}/.codex/tts-exp-artifacts/cogenav-dtw-pilot-20261025与远端/home/wjj/tts-audio/tts-exp/runs/cogenav_dtw_pilot_20261025_attempt01。E仅小文件，余量1.208GB。C是本机备份，不是网盘。主清单215成员33031600bytes，SHA568b039f6e168b32f1550fb87e7791ab32be05b6155ddf3ad417d5598f707c76，父侧再次全部SHA通过；完整tar SHA38d82f71ade593636e99ae23bbd8f0277391d1e747775fc1c1d9a699c5aa5c56。主清单冻结；父审计/中文报告/父源码/父测试4成员另addendum，C/远端一致核验。GPU完成回322MiB；无commit/push，不动继承dirty或其他chat任务。

## Observations
- [result] 已知人工局部时间变化中，三种DTW远优于整段线性映射；CoGenAV未优于MFCC，平均差仅0.920ms，六来源不足以作普遍优越性结论。
- [limitation] 相同声音的人工变形比真实自然/TTS跨音色和发音匹配容易；通过该测试不证明真实音素匹配正确。
- [decision] 不将时间路径差异、目标时钟误差或Sync-C变化直接等同感知唇形同步收益；不自动启动重定时/TFG/训练。

## Relations
- follows [[CoGenAV与SyncNet评分的20来源扩展对照 2026-10-24]]
- follows [[固定视频与实际评分PCM的留来源谱形干预 2026-10-25]]
- relates_to [[让两个音频共享 SyncNet 时间坐标]]
- relates_to [[Mask重建的精细对齐与TTS mel推理]]
