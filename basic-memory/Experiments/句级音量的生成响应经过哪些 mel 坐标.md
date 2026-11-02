---
title: 句级音量的生成响应经过哪些 mel 坐标
type: experiment
permalink: tts-exp/experiments/句级音量的生成响应经过哪些-mel-坐标
status: concluded
project: tts-exp
started: '2026-10-01'
repository: dogLeft233/tts-exp
---

# 句级音量的生成响应经过哪些 mel 坐标

## 问题与设计

Fresh男性自然80/40的固定RMS生成端G已为+0.192，99%[+0.111,+0.276]，二脸和unit也为正；下一步解释音量怎样进入生成器。实际Wav2Lip前端使用80维mel幅度、20log10、min-100dB/ref20dB、对称[-4,4]裁剪。非边界上乘PCM gain g对应1.6log10(g)的归一mel常量变化；已经裁剪的mel丢失低于边界的信息，不能拿统一post-mel shift当原始PCM gain。

恢复9/27已有精确U/S partition代码规则；旧统计缺失，fresh独立重跑。Root在已读RMS/来源/sham及corrected supplement结果后作出此informed follow-up，非原来源预注册主门。权威协议openspec/changes/confirm-mel-level-partition-20261001/spec.md，rawSHA abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8，本地未push commit326bb7bad35f198855766b57b46e2f0e90b03ee0。

## 冻结操作

使用原N80/40×2、primary face、原canonical PCM和TARGET_RMS .0376838172675746；原float64 gain→float32 waveform无clipping。M0=official RAW mel、M1=official FIXED mel，保存F32，U=(-4<M0<4)&(-4<M1<4)，S=~U，精确端点判断不拟合epsilon。四格M0 / where(U,M1,M0) / where(U,M0,M1) / M1，每bin只是复制实际合法端点，closure逐元素精确。U不称纯speech，S不称纯silence；amplitude floor与归一化lower clamp分开报告。

CAL20 CPU QA在forward前报告mask/lock-unlock/one-ULP/gain/差值/closure，保存完整数组；全EVAL输入mask/rows/seals先封存，不按mask或gain筛选。320行四格fresh encoder/decoder/SyncNet false afterdetection，固定同RAW N A/clock/k3，实际V/A共同支持，raw/unit、full/valid/guard20/guard15全存。q00/q11与原RAW/FIXED是新执行桥接，旧flags不追认。

## 统计与实施
interior q10−q00、boundary q01−q00、joint q11−q00、I=q11−q10−q01+q00、q10−q01；clip→两clip/speaker→40等权，20k PCG64 seed20260930 99CI。C/anchor/B/D/正确search D0−D及100 shared full-A permutations完整报告。Joint C/anchor阳性复核既有level generator路径；partition是预声明次要机制，不写独占winner或中介百分比。

CPU输入与神经响应均已完成并独立复核；完整结果见下方“独立 mel 分区结果”。producer scripts/experiments/tts_mel_level_partition_20261001.py rawSHA02d3f858c623c6930b95b366729c511856491b16f61ca685c88c69f2f52e80a4。Remote pinned前端16k/80mel/FFT800/hop200，NumPy1.26.4、SciPy1.13.1、librosa0.10.2.post1、soundfile0.14.0，audio71d5e927…249c4c/hparams4a4c31dc…c71cf。实际兼容sr keyword shim已绑定。

CAL20 pairs包括20N+20T波形，40/40rows、160/160cells、exit0。Exactq端点复制、float64 closure、finite/range和U/S互补全PASS，无一ULP边界值。自然interior理论1.6log10gain误差max8.85e-6、T9.31e-6。gain dB p50自然+5.264/T−2.365，是CAL描述不是响应。calibration_attempt01 QA SHA47f2bbf7ceff904395389d2e16b43af77af36e9ff4765b7ab7f46d765a4d9318；manifestafc6e6c72a729d62944916644cacdf89e405a73f1e0f678026b090df9b032c4d。

EVAL80自然输入、320q-cells完成，164ledger files精确校验、80mask/partitions独立重算、finite/range/identity全PASS，model_forward_performed=false/effects_read=false。根/root/tts-research-system-20261001/mel_level_partition_20261001/evaluation_attempt01/；plan ab6d713246ed6a9f11ecfcea08be2aae791a23e240ee4ab5791e31745be17f66，rows208c4bff640c6d7e0db21a0f58695e8946690ea8a0b6f4030fe9ef467e88fca9，manifest4f4b9bfbb1b51cb0e75585126ba3bde1c87d40bb444ea0581dc9512e0568a3f9，ledgerd9d52ced138ce13df0602b1a14ecf7d9646df41db80af1d86e80e1530a970172，独立数组audit e71b71be4ec8b6e34bb4c8f2da8f7390725c1e1836225dfd78a12da980047f41。Root接收代理数值QA，最终consumer直接核完整输入/矩阵。

local_recovery实现320GPU runner，remote_recovery唯一GPU调度，mechanism_audit最终独立消费。q00/q11必须保存exact encoder z文件及array hash与actualforward flags，衔接真实RMS方向/范数实验。已有native独立视觉全80/40显示N fixed增加aperture/velocity而T fixed减少；phone shape可用16/17speaker未达到改善门槛，两个端点分别解释。
## Observations
- [status] implemented
- [decision] PCM gain拆成双方interior坐标与触边界坐标，用实际RAW/FIXED数组精确复制，U/S不等同speech/silence。
- [decision] 全80/40同RAW N评分时钟，全部四臂新执行；q00/q11 exact z接真实RMS latent factorization。
- [verification] CAL40waveforms/160cells PASS；EVAL80/320输入封存、164ledger files和80mask独立复算PASS，尚无模型forward或effect读取。
- [next_step] 实现并运行320神经响应，独立核partition、fresh runtime、距离矩阵与100置换统计。
## Relations

- explains [[TTS 来源方向与声学机制确认]]
- relates_to [[编码器方向与范数对生成响应的分离]]
- relates_to [[原生 TTS 优势的时间配对与可见口型诊断]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]

## 已执行的 mel GPU 分区

实际run为 `/root/tts-research-system-20261001/mel_level_partition_20261001_attempt03/gpu_attempt05_artifacts/`。源码05 SHA849386cffa18efefe022f1b4079339e2f4342fdaa8f4b33e69bbeb74117d33a1，assets05 SHA4f4d20b8b8c4eb90ffff6216d83166daae0bc6d865ee5c77c2bb6ec4c34a6654。CPU preflight06实际80/320 PASS、no-forward/effectsFalse。GPU worker parent220940/child220965实际exit0，80 complete/320 generated_scored、2000artifact entries、零失败。run_manifest SHAcd5c1d4cc21dd32eccdd7111db18b88de57d0aec3b6079214602a7279f3996b6。

全actual forward满足deterministic=True/cuDNNdet=True/benchmark=False/bothTF32=False；A461/lip1816/audioencoder724/decoder724/faceblock7，N72–235、V=N−5。严格U/S分区与四格冻结不变。source04曾有EVAL manifest 63字符字面值错误，source05只修这一字符，失败preflight保留。

科学状态仍IMPLEMENTED：完成生成与评分后，local agent独立重建fullV×fullA、raw/unit四支持集、100π/20k99，并核验joint是否复制已封存RMS生成效应；当前不从producer分数直接宣布分区贡献。

## 独立 mel 分区结果

独立consumer07实际80clips/40speaker、320 cells、640 cell-geometry records完整PASS，逐文件/数组、CAL40和EVAL80原PCM→gain→mel与13-key精确分区重算、实际5flags事件与所有V×A矩阵/31 lag核验。root实际下载7,167,348B、SHA112ff916b3c6b1790f3cf3db1b75721f08f84486aca0e16d828ec432bf4bc6fb并读完整JSON。权威 `runs/recovered_20260930/independent_validation/mel_partition_consumer_20261001_attempt07/independent_mel_report.json`；source7251a7873c38c9218a0cd2e7d0c5e8715a578303b1f60e7c8d47c40c8a5edecd。20k PCG64/40-speaker等权/99%CI，100 shared full-Aπ/四supports/raw-unit、固定anchor全部报告。

primary guard20、同RAW-N评分的结果：

| 操作 | raw ΔC与99%CI | raw 时间excess与99%CI | unit ΔC与99%CI |
|---|---|---|---|
| interior U q10−q00 | +0.171 [+0.101,+0.245] | +0.167 [+0.099,+0.239] | +0.025 [+0.016,+0.035] |
| boundary S q01−q00 | −0.009 [−0.037,+0.019] | −0.011 [−0.038,+0.016] | −0.001 [−0.005,+0.004] |
| joint q11−q00 | +0.190 [+0.117,+0.274] | +0.185 [+0.114,+0.266] | +0.028 [+0.018,+0.038] |
| U−S q10−q01 | +0.180 [+0.109,+0.253] | +0.177 [+0.110,+0.248] | +0.026 [+0.015,+0.037] |
| interaction | +0.027 [−0.019,+0.077] | +0.029 [−0.015,+0.077] | +0.003 [−0.003,+0.010] |

在raw/unit × 四support × C/C_anchor的16组合中，interior、joint和interior−boundary的observed及time-excess下界均正；boundary与interaction没有组合下界正。音量正路径因此明确定位到两端都未触及裁剪边界的mel坐标，完整操作复现正向生成效应，并且U单独优于S单独。U/S按真实数值边界定义，不称speech/silence；不计算独占贡献百分比，不把CI跨0写成已证明零效果。

consumer05的ledger Path-vs-SHA错误、06的_score_one_row返回值当作partition错误均保留；07改成独立CPU partition map按ID连接，严格比较U/S count、mask-arraySHA和NPZ SHA，评分/null/bootstrap公式未变。

## 跨 run 端点的实际桥接

独立CPU endpoint bridge01实际320行/80clip SHA4fcb2aae718a65cf7bb6c808c6c48b4fe8ad2d6e08312ece3b0e83c6d7386aea，3,558,076B；root下载完整SHA与JSON读回。产物 `runs/recovered_20260930/independent_validation/endpoint_bridge_20261001_attempt01/endpoint_bridge.json`。MAIN natural q00→MELq00、MAIN natural q10→MELq11各80clip，latent maxabs .000893/.000867，0 bitexact；视频帧数80/80匹配，但pixels、V、A并非bitexact，pixel meanabs约.722/.732。MAIN axis baseline→normq00、orth_plus→normq11的latent80/80 bitexact，fresh decodedpixels/V/A仍非bitexact。桥接分别保留stored padding，并在matched query和full-A/legacy两种边界下重建曲线；不把这些描述性差异归因于单个runtime设置，也不冒称精确旧run重放。新的独立joint正向复制作为科学结果解释。

- [result] interior与joint在完整16组合observed及time-excess均正，U−S配对优势也为正，确认未裁剪mel坐标的生成路径。
- [verification] 独立consumer07完整80/320和跨run端点桥接320实际完成，root下载SHA/read PASS。
- [decision] 新实现以独立joint复制支持旧RMS作用，保留非bitexact跨run差异；真实RMS方向/半径继续使用本MEL密封z0/z1。