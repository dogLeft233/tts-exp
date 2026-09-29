# C：视觉教师的动态轨迹是否有特异证据

## Context

先读[公共入口](../../parallel-next-experiments-20260910.md)。父missingness结果使用canonical mouth的逐帧距离中位数；它没有分离静态嘴形和轨迹。本轮是新指标的回顾性诊断，父115/133、21/23 observed覆盖和完整分母保持不变。

## Goals / Non-Goals

以同一时间支持检验候选MFA-linear视频是否更贴近真实嘴部运动，并检查指标对时间顺序是否敏感。只读JSON/NPZ，不解码或hash媒体、不加载landmarker、SyncNet或TFG。任何结果不修复旧Stage01、不授权Stage02。

## Decisions

### 1. 输入与共同支持

快照分支C。H=`runs/lrs3_visual_teacher_missingness_20260909_v1`；读取H/records.json里的399个feature路径及其已绑定hash，H/analysis.json里的133条/23组、observed与共同G/N/M索引，H/support_indices.npz为交叉核验。原父P=`runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4`。

NPZ键canonical_mouth[T,31,2]、valid[T]、timestamps_s[T]；allow_pickle=False，有限值、无效帧mouth=0、时间递增。坏缓存是BLOCKED，不按科学缺失处理。

从原数组重建G/N和G/M单调一对一配对：容差为两序列中位帧间隔的较小值/2；长度<=1时G间隔0.02秒、另一臂沿用G。双指针差delta=t_X−t_G，abs(delta)<=tol同时前进并配对，delta<−tol只X前进，否则只G前进。各pair只留双侧valid；以共同G索引取两pair交集J，映射出三臂索引。核对H的索引逐值一致。

逐条复核旧exact_time距离（逐帧RMSE的中位数）、coverage分母max(T_G,T_X)、valid fraction门槛.90、两pair coverage门槛.85及父有限性检查；observed=父eligible且共同coverage>=.85且J非空。复现115/133、21/23、原missingness L/U（1e-10容差）。新分析不重新选eligible或删除不利样本。单帧/静止的合法序列保留，其动态为0，不记作新缺失。

### 2. 预指定分解和反转

对observed记录，在J上分别取G/N/M，float64；对X∈{N,M}定义：

```text
mu_G = mean(G, axis=time); mu_X = mean(X, axis=time)
Gc = G-mu_G; Xc = X-mu_X
e_total_X = mean((G-X)^2)                 # 所有时间×31×2元素
e_static_X = mean((mu_G-mu_X)^2)          # 31×2元素
e_dynamic_X = mean((Gc-Xc)^2)
e_reverse_X = mean((Gc-Xc[::-1])^2)       # 沿J顺序反转，不重新配对
b_dynamic = (e_dynamic_N-e_dynamic_M)/(e_dynamic_N+e_dynamic_M+1e-12)
q_N = (e_reverse_N-e_dynamic_N)/(e_reverse_N+e_dynamic_N+1e-12)
q_M = (e_reverse_M-e_dynamic_M)/(e_reverse_M+e_dynamic_M+1e-12)
```

必须满足e_total=e_static+e_dynamic，容差`1e-10*max(1,e_total)`。报告raw差及bounded量；两误差全0时bounded量0。反转保留静态均值和序列取值，破坏顺序；固定一次，不搜索shift/DTW/最优反转区间。J不连续时只沿观察序列反转，报告最大原始时间间隙；本量不称速度误差，也不假设重新采样等间隔。

MSE用于精确分解；它与父RMSE中位数是不同端点，不据此改写父结果。主问题只有b_dynamic；q_N/q_M是时间敏感性对照，e_static为解释性输出，不独立搜阳性。

### 3. 明确两个分母

**全队列**：用原133条和23组，缺失bounded b/q均取[-1,1]。每组界限`[(sum(observed)-missing)/n_g,(sum(observed)+missing)/n_g]`，再23组等权。对b_dynamic、q_N、q_M分别报告L/U；这是有限队列缺失界限，不是CI。

**observed条件估计**：115条先组内均值，再21个有observed的组等权。固定PCG64(20260910)、20000×21组bootstrap，三量共用索引，双侧99%CI、linear quantile。标签必须为observed_subset，缺2组和18条显式列出；不能推断全部133条或新数据。

单个量的observed筛选条件：99%CI下界>0、均值>0.02、至少17/21组>0。0.02是本轮提前固定的实用筛选阈值，不是已验证的感知阈值。按以下顺序给唯一diagnostic_decision：

1. 输入/数值验证失败：BLOCKED，science=not_available。
2. q_N或q_M未过observed筛选：TIMING_SPECIFICITY_NOT_ESTABLISHED；不将b_dynamic单独正值认作动态教师证据。
3. b_dynamic未过observed筛选：NO_OBSERVED_DYNAMIC_ADVANTAGE_ESTABLISHED。
4. 三项observed都过，且三项全队列L均>1e-12：CACHED_DYNAMIC_SIGNAL_ROBUST_TO_MISSINGNESS。
5. 三项observed都过但全队列界限不满足：OBSERVED_DYNAMIC_SIGNAL_REQUIRES_NEW_COHORT。

完整披露所有数字，不因全队列界限跨0隐藏observed结果，也不靠observed结果放行父教师。第4/5种只建议新完整source-group队列确认；第2/3种停止该固定教师上的投入。反转对照未过不是证明教师不存在。

### 4. 实现与验收

实现包为 `lrs3_visual_teacher_missingness`，对应命令中的 `<package>`；只需runner、analysis、validate。可借鉴本包 `analysis.py` 的load_feature/exact_matches格式；新validator独立实现配对、分解、bounds、组统计和判定。array支持文件存G/N/M索引，JSON存每记录e/b/q、每组分母和界限；不用复制媒体。

反例测试：纯常量嘴形偏差全部进入static；相同变化轨迹dynamic=0；同均值反向轨迹使dynamic增大；静止轨迹q=0不会被移除；两pair不同支持必须取交集；全缺组保留[-1,1]；不等组大小等权；observed阳性但全体跨0保持限制；篡改分析并重签仍拒绝；mock模型入口被调用即测试失败。

BM唯一实体：`LRS3 visual dynamic specificity 2026-09-10`。只新增本回顾性诊断，原视觉教师及missingness实体不改。

## Risks / Trade-offs

- 去均值会忽略真实口型的平均开合差→同时报告static/total，动态阳性不等于整体质量更好。
- landmark抖动与时序采样仍影响动态误差→要求固定反转对照并保留测量误差限制。
- observed选择偏差→21组条件估计与23组缺失界限并列，禁止合并为一个“教师已有效”结论。

## Migration Plan

新建CPU分析包和run；399份features及父review只读，0新模型/视频/评分。无需环境部署或数据迁移。
