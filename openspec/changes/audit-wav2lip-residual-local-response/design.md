# C：固定残差的时间支持与局部响应

## 入口

完整读取 `../../parallel-replacement-probes-20260909.md`。本轮是retrospective diagnostic，已知全U整体阴性，不把新局部指标当作原主指标的替代。只用P/Q/D已有数组和JSON，零视频解码、零模型、零fresh scorer、零landmark、零GPU。

## 曝光比例：先定义支持，不看分数选窗

K读取P/drivers/manifest的每行 `construct.K`，并从D/05_drivers/drivers.json的used_masks `[global_start_frame,global_end_frame)` 独立重建其有序并集，坐标是80Hz mel列。所有seed/条件K必须相同，CORRECT与N在K外逐值一致。K是预先允许干预的位置，不按实际收益或非零幅度重新裁K。

从官方chunk规则独立重建第t个视频帧输入的mel列集合 `T_t`：正常start=int(t*80/25)，16列；遇到start+16>308时使用最后16列并停止。必须93个chunk；不能用简单t×3代替。对SyncNet行r，其视觉embedding使用视频帧r..r+4，故：

```text
B_r = union(T_t for t in range(r,r+5))
e_r = len(B_r intersect K) / len(B_r)
```

以唯一mel列集合计数，重叠chunk列不能重复加权。仅在冻结U=30..57做新分析，不读FULL别的行来选更强窗口。保存每行chunk起止、B_r/K交集、e_r和全部索引；先写入hash-bound support产物，再读取候选矩阵做新统计。

每条 `k0=argmin(mean_U D_N)`，行级收益：`d_r = D_N[r,k0] - D_CORRECT[r,k0]`。绝不能每行自行argmin，也不计算“逐行Sync-C均值”代替全U Sync-C。均值mean_U d必须复现父ΔA（总体mean=0.019221，精确值取已绑定analysis，容差1e-6）。

## 两个合取诊断量

先对每记录i计算U内均值、去均值协方差和方差，再在组g的两条记录汇总：

```text
v_i = mean_U((e_i - mean_U(e_i))**2)
c_i = mean_U((e_i-mean_U(e_i))*(d_i-mean_U(d_i)))
beta_g = sum(c_i for i in g) / sum(v_i for i in g)
local_gain_g = sum(e_ir*d_ir for i in g for r in U) / sum(e_ir for i in g for r in U)
```

beta是组内汇总的**记录内**曝光—响应斜率，不是因果效应；先逐记录去均值，不能把记录间均值差当斜率。local_gain是该组曝光加权的行级收益，不是全组完整视频平均收益。无曝光记录仍在输入/全U复验/零曝光控制中；在这个明确的条件估计量中自然贡献零权重，不把其“未观测局部收益”填成0。

设计期只读K/chunk几何预检发现 `lrs3_7VRzn8hc5mc_00016`、`lrs3_7c5t6FkvUG0_00001` 在U完全无曝光，其余14条有曝光变化，所有8组都有非零记录内方差。因此预先采用上述组内汇总，不按新的候选收益筛样；此时未计算新beta/local_gain。下游重验并报告每条e范围/v_i，不把上述预检当正式验收。

若任一组sum(v_i)≤1e-12或sum(e_ir)≤1e-12，完整保留16条/8组，终态 `LOCAL_SUPPORT_NOT_IDENTIFIABLE`，不丢组或改U。全部8组可辨识时直接对8个beta_g/local_gain_g等权、按公共99%组bootstrap。唯一主问题是合取：beta与local_gain的CI下界均>0，至少7/8组两者同正，才给 `LOCAL_RESPONSE_ASSOCIATION_TO_CONFIRM`；否则 `NO_LOCAL_POSITIVE_ASSOCIATION_ESTABLISHED`。不规定beta任意大才算，且不把窗口数当样本数。

另外按同定义描述WRONG/SHUFFLE相对N的行级d、beta/local_gain（区间可报但无新的判定标签），全部报告、不挑最差对照做机制证据。CORRECT胜SHUFFLE不代表胜N；正beta但local_gain仍负也不能称局部收益。

## 独立工程诊断

对e=0的行，5个视觉帧的所有输入chunks均未触及K；静态face和模型配置相同，应无可检测响应。核对CORRECT/N对应视觉embedding与完整31列距离行最大差≤1e-4，违反为BLOCKED（支持映射或缓存身份问题），不是“远程语义效应”。没有e=0行记check_not_applicable，不人为制造负对照。

从缓存embeddings独立重建N/CORRECT/WRONG/SHUFFLE的完整矩阵和父三个全U contrast，确认复现父analysis；不要把父rounded报告数当真值。控制证据按公共契约CPU复现，不运行当前GPU链。缺文件/hash错/支持错误是BLOCKED，不是LOCAL_SUPPORT_NOT_IDENTIFIABLE。

## 结论边界和下一步

即使局部关联通过，也只建议在新source-group上预固定支持、研究是否有可泛化的稀疏干预；**本轮不应用局部gate生成新视频、不重加权原主指标、不宣称全局平均确实掩盖了可用replacement**。窗口高度相关、e还与音素/能量位置混杂，不能证明语义因果。

若没有局部关联，停止“只是整体平均稀释了收益”的补救解释；不是证明所有局部语义路线无效。不可辨识则说明现有U/K不足回答，不自动移动窗口。父NO_INCREMENT_ESTABLISHED在所有终态均不变。

## 交付

包 `wav2lip_residual_local_response`；support_indices.npz、exposure.json、reused_manifest.json、逐记录/逐组表与公共产物。测试手算重叠chunk并集、末chunk、5帧支持、全U冻结anchor、零曝光行恒等、beta正但local_gain负、记录间均值差不能冒充记录内斜率、零曝光记录保留/整组不可辨识、禁止GPU/模型入口。

BM唯一实体 `Wav2Lip residual local response audit 2026-09-09`。记录曝光范围、不可辨识数、两个诊断量、父整体阴性仍成立及零模型预算。
