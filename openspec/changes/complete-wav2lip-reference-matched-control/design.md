# B：补完正确控制下的参考图交互

## Context

先读[公共入口](../../parallel-next-experiments-20260910.md)。旧F46只有控制结果，F46_C未生成。E0的16/16来自P/F0，不能直接放行本分支；本轮必须独立重建F46控制及其历史绑定。

## Goals / Non-Goals

回答相同固定CORRECT残差相对N的响应是否随参考条件变化。保持F0/F46、原16条/8组和原主统计。无需新残差、训练、参考帧搜索或额外候选。

## Decisions

### 1. 固定2×2

输入快照分支B。F0_N用P/control_scores原N，F0_C用Q/candidate_scores的CORRECT；按ID关联，driver逐值对应P/drivers。F46_N/F46_C在本run新生成，不能将F46_N相对F0_N的主效应当音频增量。

F46来自P/static_faces绑定源视频的零基解码帧46及box文件第46项xyxy，源25fps。按F0相同方式无padding crop、INTER_LINEAR resize224×224；锁定像素、PTS、box、源hash。源帧/box缺失或不合法为BLOCKED；任一F46与F0像素相同为REFERENCE_INPUT_DEGENERATE，全体停止。每cell静态重复93帧。新F46须与旧reference_manifest绑定像素一致，否则调查工程原因，不重新选帧。

### 2. 先CPU控制，再fresh复验

先从旧F46控制manifest的embeddings重建两个域，保存legacy和matched结果；这一步没有候选或新评分。距离与U按公共公式，matched delay使用q=r+j-10、物理offset=10-j，自然q=r+j-15、offset=15-j。缺真实支持不得padding。

在F46中，offset_delay−offset_N∈[-6,-4]至少14/16；未补偿损伤取legacy delay曲线在**F46_N自身**argmin(z)处减N，8组等权、PCG64(20260909)、10000次、95%CI下界>0且至少7/8组正。不预设必须16/16。计算正确但门槛不满足→CONTROL_FAILED，0新模型；旧日志13/16仅是复现项。

历史控制通过后执行公共F0控制，再新生成16 F46_N，前两条分别额外forward一次F46_N_REPEAT；重复像素bit-exact、矩阵≤1e-4、端点≤1e-6、offset相同。每个新F46_N加一个A_DELAY评分（+3200 samples、前补0、尾裁到原长度），重验上述F46门槛。delay只作评分控制，不当driver。

全部控制独立验收后才生成16 F46_C；不用旧runner的固定15-j helper处理matched delay。可复用纯SyncNetScorer并从embedding重建所需lag域。新/旧F46_N也核对解码像素及原评分矩阵，冲突BLOCKED，不偷偷替换历史基线。

预算：F0 replay2+F46_N16+F46_REPEAT2+F46_C16=36视频；上述评分36+parity2+delay16=54。F0_N/C共32个科学cell复用；所有失败重试另记。完整重验比复杂的局部缓存恢复稍贵，但让当前运行时和新候选可直接比较。

### 3. 唯一主问题

四cell统一使用U及 `k0=argmin(z_F0_N)`：

```text
g0 = z_F0_N[k0] - z_F0_C[k0]
g1 = z_F46_N[k0] - z_F46_C[k0]
I = g1 - g0
```

按公共B统计计算I。99%CI严格排除0、abs(mean I)>0.05、至少7/8组I与总体同号，三条件齐备→REFERENCE_DEPENDENT_RESPONSE；否则NO_REFERENCE_INTERACTION_ESTABLISHED。正I可能同时g1<0，因此无replacement阳性标签。

报告g0/g1/I、四cell C/D/A、每个参考在自身N anchor下的附录结果。附录不得改主判定或选择更好的参考。

### 4. 实现与验收

实现包为 `wav2lip_reference_conditioning_interaction`，对应命令中的 `<package>`。参考旧reference的media和gpu_worker接口；逐项重写有缺陷的控制/validator，公共纯worker只读复用。validator独立重建四cell矩阵、F46 matched索引、两套控制、共同k0和组bootstrap；旧PASS不是验收替代品。

反例：+5帧延迟的两域offset分别15-j/10-j；P/F0通过但F46失败不能生成C；第46帧不是46秒；box次序错误；参考主效应抵消的手算差中差；I>0且g1<0不能replacement；缺cell/改数字/错replay像素拒绝。

BM唯一实体：`Wav2Lip reference matched-control completion 2026-09-10`。报告实际F46通过数、控制状态、I是否可计算、预算和原问题是否终于得到回答。

## Risks / Trade-offs

- 参考变化同时含嘴形/姿态/照明/crop→仅称参考条件交互，不能隔离嘴部先验。
- 已见样本与同一个固定残差→交互阳性需多参考和新source-group确认，不续调残差。
- 旧F46正确域可能仍失败→有效停止也是完成，保持旧父结论，不降低14/16门槛。

## Migration Plan

新run承载全部复验；旧B与E0只读，历史缺陷通过结果指针说明，不覆盖旧final。
