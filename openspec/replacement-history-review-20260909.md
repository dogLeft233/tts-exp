# Replacement历史复核与下一步（2026-09-09）

本次为BM、spec与源码只读复核；没有重新跑所有旧实验。证据分“旧报告可读数字”“已发现实现缺陷”“尚待独立重算”。实验目标是同一自然音轨N下的replacement，不是候选自己的音轨分数。

## 旧实验有哪些问题

1. 早期Ditto声学干预曾漏掉重复生成噪声。BM `tts_tfg_mechanism_report` 记录identity重跑ΔC约−1.161，足以解释原约−1.0至−1.2的干预下降；扣除identity后LUFS/tilt/compression/expansion未显示明确独立效应。旧基线“约12dB响度解释”与该机制报告不一致，不能沿用作已确认机制。
2. 早期中文特征可分性使用pinyin均匀切片，低估自然语音；BM `english_wav2sem_fp_fs_summary` 的MFA修订撤回了“中文TTS提高可分性”。该笔记仍有“Fs饱和、完全经Fp作用”的过强机制措辞：未显著不等于等效，表征差异也不证明因果中介。
3. 早期正向观察混合了diagonal、自适应offset、不同时间支持和处理链。它们不是虚构数字，但不足以证明固定N时钟的口型收益。全局shift被自由offset吸收不代表控制失效；局部swap也可能破坏生成器自身适用性，不能由此宣布所有candidate无效。
4. 不同run同时改变候选PCM、生成框、ROI、编码/时钟、评分裁剪。历史对账已证明不能将差异单独归因于某个评分器或窗口。Compatibility/non-inferiority也不等于有用增益。
5. 最新并行实现存在明确spec偏离：B/design“控制与预算”已要求delay lag[-10,20]；`scripts/experiments/wav2lip_reference_conditioning_interaction/score.py:13` 固定offset=15-index，`runner.py:131` 直接用legacy U offset计数，未按该配对域重建。13/16是旧域统计，不能当作本spec正确控制的科学失败。
6. 最新A/B/C validator的PASS覆盖不足：A/validate.py复用producer temporal_contrast且不重算analysis；B/validate.py只验shape/计数/少数字段，不重算delay和主统计；C/validate.py复用row_supports且不重算局部beta/CI。B的F0 replay_pass只检查U列表。C producer还用两个各≥7组阳性替代“同一≥7组联合阳性”，零曝光检查只覆盖整条无曝光，漏掉部分记录里的零曝光行。这些需要E0逐项量化影响，不能预先说所有数值错误。
7. A/B的analyze会经过生成路径，final先写complete/PENDING而不绑定最终validation，不符合只读分析/单向完成证据约定。现有runs仍保留；下游审计写旁路结果。若已有代码未绑定当前历史spec，只能报告来源不完整，不补写伪历史hash。
8. BM最近三份笔记同时保留planned和concluded，OpenSpec若干旧任务复选框亦滞后。不能只按checkbox判断跑过与否，更不能仅按PASS布尔判断科学可靠。

上一轮对“9单测、3 strict、3 validator全部通过”的程序状态描述可以成立，但把它解释为“3路按spec完成独立数值验收”不成立；本次明确纠正。没有证据把所有历史阴性一并推翻。

## 目前有哪些结论

| 证据 | 结果 | 合理解释 |
|---|---|---|
| 中文Ditto历史diagonal | ΔC=+1.246，原n=9 | 特定生成音轨/评估音轨组合优势；不是replacement |
| 中文Ditto G×E | G_T/E_N相对G_N/E_N ΔC=−3.247；交互+7.290 | 换回自然音轨后不能直接保留diagonal优势；共适配是一种解释，非唯一机制证明 |
| 英文Fs/Fp历史 | 英文优势未稳定复现；MFA后自然可分性总体更好 | 不支持“语义更多/更可分即可提升同步”的简单链条 |
| phone-aligned TTS mel | 30条ΔC=−0.102，95%CI[−0.136,−0.066] | 该构造有效工程条件下有害；优于shuffle只是少伤害 |
| natural→MFA-linear谱桥 | discovery +0.078；22条确认约+0.032，CI跨0且控制失败 | 有过探索信号，没有确认收益 |
| 新谱迁移控制通过轮 | MAG ΔC=−0.889；ENV=−0.042 | 停止该固定谱迁移，不否定所有头 |
| 历史200ms shift同媒体复核 | FULL自由ΔC=+0.298可复现；新整帧同J自由ΔC=+0.341，固定anchor收益−0.099、CI跨0 | 自由分数现象保留；固定自然时钟改善未建立；评分入口差异也未确认 |
| masked语义/轨迹辅助 | hard-negative paired胜NAT_ONLY及错配；未证明胜完整natural | 修补缺失上下文有价值，不等于增强已有完整输入 |
| 完整natural内容残差续跑Q | ΔC=+0.008，95%CI[−0.029,+0.040]；ΔA=+0.019、CI跨0 | 固定增量没有建立replacement收益 |
| 新时间平滑/增强A | 报告ΔC=+0.012/−0.015，99%CI跨0 | 报告阴性；独立验收待E0补齐 |
| 参考交互B | 未生成F46_C；legacy offset仅13/16 | 实现遗漏已知配对域，交互科学问题尚未回答 |
| 局部响应C | 报告beta=+0.064、local_gain=+0.030，99%CI跨0 | 没有已报告正信号；独立验收待E0补齐 |
| 视觉教师missingness | spec已写，BM仍planned | 是尚可并行执行的一条，不能当已完成阴性 |

BM主要入口：`tts_tfg_mechanism_report`、`english_wav2sem_fp_fs_summary`、
`LRS3 Wav2Lip phone-aligned mel causal NO-GO`、`LRS3 natural-to-TTS bridge confirmation result`、
`Wav2Lip spectral structure replacement 2026-09-08`、`Wav2Lip replacement endpoint reconciliation 2026-09-08`、
`Wav2Lip historical shift rescore 2026-09-08`、`LRS3 masked TTS trajectory-specificity diagnosis`、
`Wav2Lip natural content residual continuation 2026-09-09`及三份并行实验笔记。

## 下一步决策

有限推进，换成可证伪的小问题：先核实最新证据，再分开测试波形前端的幅度响应、保留音素边界的内部稳定化、重建基底与内容增量的交互；同时完成已有视觉教师审计。具体五路在 [并行交接](parallel-replacement-next-20260909.md)。

E2最接近模型无关波形操作；E3仅用已存音素时间标签，检验发音事件结构而非句义；E4检验masked正结果为何不能迁移到完整N。E3/E4为mel研究，阳性也还差波形可达性。旧natural-only policy已NO_GO，新固定算子不是旧policy续训。
没有正信号则关闭各固定构造，停止同批小步调参；有正信号才做新source-group确认和第二TFG，之后再讨论泛用生成头。
