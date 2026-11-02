---
title: FIXED 背景音素内 MFCC 谱形相位反转 2026-09-27
type: experiment
permalink: tts-exp/experiments/fixed-背景音素内-mfcc-谱形相位反转-2026-09-27
status: concluded
---

# FIXED 背景音素内 MFCC 谱形相位反转

## Observations

- [status] concluded
- [result] 主 raw/guard20/71/15 的 T−N C 响应 +0.196098，99%CI [−0.020126,+0.382330]，未确认变化；原残余 +0.254472→+0.450570。N C −1.110499、T C −0.914400，均99%CI为负。
- [decomposition] 主 B 响应差 +0.012185 CI跨0，D响应差 −0.183913 [−0.320463,−0.051297]。事件1411 raw margin gap响应 +0.133610 CI跨0；来源interaction margin下降 −0.716028 [−0.951257,−0.507997]，主要来自正距interaction改变，负距interaction响应未确认。
- [conclusion] 实际评价编码器对phone内谱形次序有反应，两臂受损且来源匹配interaction减弱；未消除T−N残余。主差距未确认变化，9个预定附录gap为正；不以附录替换主，不称零效应、纯相位或真实音频/生成因果。
- [verification] 全200 baseline/inverse A对旧FIXED exact；独立200前端/索引、2019092距离→1764曲线、1411event、412统计/4332逐clip效果PASS。receipt dfc9e61d…；无新视频/CloudDrive，GPU释放。
- [report] runs/tts_fixed_mfcc_phase_reversal_20260927/report.md，SHA6958d367e974730019aeceb480caf68bd16c4cf14716f6b66105d0dfe99546d2。评分前独立批准lossless gzip，完整曲线/事件/新A保留。
- [question] 固定原生时钟/能量C0与FIXED视频时，实际评价前端的phone内谱形时间次序是否贡献T−N Sync-C差距？
- [design] 原26cal/74eval共200音轨、主旧71/15；先完整MFCC序列，对完整原样本支持落同一可用phone的帧反转C1…C12，C0/边界混合/静音不动，再按原4步长切20帧。原V/A缓存引用，新A真实forward；k3及±15固定，处理后lag不作为失败/重选门。
- [controls] baseline与两次反转A对旧FIXED A全200严格exact；先cal模型控制通过再eval，无cal效应汇总。所有原样本保留，不按覆盖/效果删样。
- [resources] 新A NPY块上界88.9140625MiB，metadata/代码/统计/收尾≤10MiB，合计98.9140625<100MiB；新阶段总192MiB、本地底4.5GiB，无新视频。
- [boundary] 改的是模型MFCC输入，原波形虽保持不变，不能称实际音频增强或生成因果；C0与重排谱形未必对应可实现音频，保留分布外损伤及共发音混杂。

## Relations

- follows [[共同音素窗口内部与上下文定位可行性 2026-09-27]]
- follows [[校准固定绝对电平生成评价四格 2026-09-27]]

## Changelog

- September 27, 2026：root原则授权，准备完整协议/源码及输入seal；独立先验PASS前不新forward/评分。

- September 27, 2026：新 forward 前 v2 绑定 joint 双 hash 与 NPY 接口；v3 按独立审阅将 event unit 对齐旧 float32 算术（官方71不变），旧版本无损保留。v3 protocol cffdb6b7…，独立先验 receipt 0aeadfb6… PASS；全200 CPU输入/原frontend/反转恢复 exact，开始52cal实际编码器控制，无新评分。

- September 27, 2026：全200输入/模型exact门通过，联合field协议先于score封存；完成原74/主71及1411event统一分析，独立复算PASS。预评分纯lossless gzip适配保持科学字节与原预算，报告/BM concluded；不扩剂量、改阈值或删支持。