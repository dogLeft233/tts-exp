# E4：重建基底与内容增量交互

先读公共 `../../parallel-replacement-next-20260909.md`，再读P原design四臂定义与Q续跑design。唯一新问题：masked模型的内容增量是否只在接近其重建输出的基底上有下游作用？并非通过加大原CORRECT幅度挽救Q。

## 资产与五臂

D/04_reconstruction/reconstruction.json字节SHA=`79f79c0e1a5a439324dce1d35c0e2861ce428ece7f87f6c78eb195871f757df6`。
用D三个seed=20260901/02/03的同一个hard-negative模型对应PAIRED_TTS、NAT_ONLY、SAME_PHONE_WRONG_INSTANCE预测，沿D/drivers与reconstruction hash验证。每seed checkpoint三条件相同，三个checkpoint具体SHA见P原design。只读数组/hash，不加载模型训练。

M=P/N；Z=mean_seed(D/NAT_ONLY driver)，先float64平均，不把独立NAT_ONLY模型冒充同模型zero辅助。
重新按P原公式算C原始残差=mean(PAIRED_TTS-NAT_ONLY)、W等范数wrong残差和三臂共用a，重现P/CORRECT及P/WRONG逐值一致；K外残差0。记c=a*C、w=a*W。
新基底残差 `R_base=Z-M`（K外严格0）；
`b=min(0.25,0.5/max(abs(R_base)))`；
`B=clip(M+b*R_base,-4,4)`，此中间B保留float64。
固定五个cell：

| cell | mel | 来源 |
|---|---|---|
| N | M | P缓存 |
| N_CONTENT | float32(clip(M+c,-4,4)) | P driver / Q CORRECT评分 |
| BASE | float32(B) | 新 |
| BASE_CONTENT | float32(clip(B+c,-4,4)) | 新 |
| BASE_WRONG | float32(clip(B+w,-4,4)) | 新 |

关键：BASE_CONTENT使用未提前float32量化的B，独立validator按同序复算；N_CONTENT/CORRECT原幅度a不变。BASE与加增量臂均由同一个B形成。任一record R_base范数<=1e-12、BASE恒等或内容实际增量0为INPUT_DEGENERATE；不调b、不删样。自然完整旁路仍保留，不把BASE误称完整N基线。

保存实际postclip增量 `BASE_CONTENT-BASE` 与 `N_CONTENT-N` 范数比，以及BASE_CONTENT/BASE_WRONG相对BASE增量范数比。每record两比均在[0.95,1.05]且非零才interaction_norm_valid。裁剪不合格不重标a/b；可报绝对候选收益但不声称匹配增量/内容特异机制。b取上限仅是一次固定假设，不扫描到masked原Z找更高分。

## 主假设与交互

所有五cell评分用同一P/F0静态脸、原完整N音轨、U、P/N k0。复用N与N_CONTENT前独立从embeddings重算Q/CORRECT/N三指标与组统计，不能仅信Q/analysis。公共控制验收后只新生成BASE/BASE_CONTENT/BASE_WRONG共48视频，加2 replay/2 parity，总50视频/52评分。

唯一replacement主假设 `gain(BASE_CONTENT,N)`，阈值按公共99%约定。
固定anchor交互：
`g_N=z_N[k0]-z_N_CONTENT[k0]`；
`g_B=z_BASE[k0]-z_BASE_CONTENT[k0]`；
`I=g_B-g_N`。
完整报告BASE/N、BASE_CONTENT/N、BASE_WRONG/N、BASE_CONTENT/BASE、BASE_CONTENT/BASE_WRONG的C/D/A及g_N/g_B/I，禁止仅展示退化基底上的相对修复。

诊断 `base_interaction_signal` 要求interaction_norm_valid、I的99%CI下界>0、mean I>.05、至少7/8组I>0；否则false。它可以与BASE_CONTENT仍低于N同时成立，只解释基底依赖。
主假设失败→NO_BASE_CONTENT_GAIN_ESTABLISHED，不论I多大都不升级replacement。主通过且范数有效、base_interaction_signal=true、BASE_CONTENT对BASE_WRONG的C/D/A三个99%CI下界全>0且7/8组联合正→BASE_DEPENDENT_CONTENT_SIGNAL_TO_CONFIRM；主通过但机制合取不满足→NATURAL_GAIN_MECHANISM_UNRESOLVED。
所有授权false，不把此基底交互称为文本语义因果或waveform可达。

## 交付和测试

包 `wav2lip_reconstruction_base_interaction`，资源/阶段/独立验收按公共契约。测试seed先融合、同模型zero条件身份、K外不变、固定c对两基底相同、clip影响范数、正I但仍差于N、wrong胜负不替代主假设、漏cell拒绝、float64 B中间不提前量化。独立validator从D预测及P/M重建五臂，再由embeddings重算所有端点/统计。
BM唯一实体 `Wav2Lip reconstruction base interaction 2026-09-09`，报告绝对收益与基底交互两个答案、范数有效性、实际预算，停止该固定构造或仅建议独立确认。
