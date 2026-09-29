# B：参考图像 × 固定音频干预

## 入口与范围

完整读取 `../../parallel-replacement-probes-20260909.md`。本轮不重做语义机制或调content-residual；只把原失败构造当作固定探针，问其响应是否依赖参考图。新参考可能同时改变口型、姿态、照明及crop，故结果只归因于“整个参考条件”，不能声称隔离了嘴部先验。

## 2×2设计

全体16条/8组；C直接使用P中既存CORRECT driver，N也使用P原driver，逐值/hash一致。不得重训或重新挑seed。

| cell | 图像 | mel | 来源 |
|---|---|---|---|
| F0_N | P第0帧静态face | N | P缓存 |
| F0_C | 同上 | CORRECT | Q缓存 |
| F46_N | 原源视频第46帧静态face | N | 新生成 |
| F46_C | 同上 | CORRECT | 新生成 |

F46是**零基decode frame index=46**（在25fps源上约1.84s），不是按评分/嘴张合选帧。使用P/static_faces manifest绑定的source_face和box文件；读取box列表第46项xyxy，按与F0相同方式无padding crop、INTER_LINEAR resize到224×224。冻结源frame PTS、像素hash、box和裁后图hash；源必须25fps且46帧真实存在、box合法，不重新检测/换帧/补齐。缓存box未覆盖46是BLOCKED。每cell都重复其单张静态face93帧，禁止输入动态源视频。P/F0必须按第0帧规则重建一致。

先完成prepare并锁定全部F46，再接触新评分。若任一F46裁后像素与F0完全相同，标记 `REFERENCE_INPUT_DEGENERATE` 并停止全体科学交互，不替换记录。不使用landmark检测来择图，因此不能预先保证两张图的嘴部差异。

## 控制与预算

公共F0控制：2个parity评分、前2条N replay生成/评分。F46先生成16个N，前2条分别独立再forward一次N_REPEAT；像素bit-exact、评分矩阵≤1e-4、offset相同，否则BLOCKED。

F46/N的16条各加A_DELAY评分，不生成新视频。延迟3200 PCM samples、长度不变；严格沿公共父配对搜索规则（N -15..15、delay -10..20，physical offset分别15-j与10-j）。offset差-5±1至少14/16。损伤使用F46/N自己的自然anchor和**legacy未补偿delay曲线**，8组等权、父seed20260909/10000/95%下界>0且至少7/8组正。这里只检验F46评分敏感性，不把F46自己的anchor替代科学交互的公共k0。

F0/F46控制独立验收全部通过才生成16个F46_C。合计fresh视频=F0 replay2+F46 N16+repeat2+C16=36；评分=上述36+parity2+delay16=54。复用F0_N/C共32评分cell。所有新视频mux原N；A_DELAY仅是明确标记的控制例外。

## 主交互：不混入参考图本身的分数差

每条固定 `k0=argmin(z_F0_N)`，四cell在同一U、31列和k0计算：

```text
g0 = z_F0_N[k0]  - z_F0_C[k0]
g1 = z_F46_N[k0] - z_F46_C[k0]
I  = g1 - g0
```

唯一主统计量为I，按公共99%组bootstrap区间。`REFERENCE_DEPENDENT_RESPONSE`要求区间严格排除0、abs(mean I)>0.05、至少7/8组I与总体均值同号；正号表示第二参考下相对N的响应更好，负号表示更差。其余为 `NO_REFERENCE_INTERACTION_ESTABLISHED`，不是参考图等效性结论。工程、退化、控制失败先于科学判定。

完整报告g0/g1/I及四cell C/D/A；另列每个参考以自己N anchor计算的within-reference收益作为敏感性附录，不能用它替换主I或挑更好anchor。**F46_N分数高于F0_N不构成音频增量证据；I正也可能只是g0负得更多、g1仍为负。**因此所有分支无replacement阳性标签。

若交互明确，下步仅建议用多个预固定参考验证稳健性/研究reference-conditioned控制；若不明确，停止本探针，不继续扫参考帧。两者均不修改原CORRECT整体阴性，不授权头训练。

## 交付

包 `wav2lip_reference_conditioning_interaction`，接口/产物按公共契约；额外 `reference_manifest.json` 与四cell配对表。聚焦测试：frame46不是第46秒、xyxy转换、源box身份、同一静态帧重复、共享k0、差中差消除单纯reference主效应、正I但负g1不判replacement、F46控制缺失不生成C。

BM唯一实体 `Wav2Lip reference conditioning interaction 2026-09-09`。记录参考变化不能隔离嘴部原因、g0/g1/I、控制状态和实际预算。
