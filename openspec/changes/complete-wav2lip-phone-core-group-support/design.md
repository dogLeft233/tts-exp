# A：按原组级支持补完 phone-core

## Context

先读[公共入口](../../parallel-next-experiments-20260910.md)。旧design“每组必须有曝光”与 `wav2lip_phone_core_shrinkage/construct.py:construct_candidates` 的逐记录异常不一致；旧input_audit中两条均为 `no public U exposure`。本轮保留历史INPUT_DEGENERATE记录，只新增纠错续验。当前尚未重算8组覆盖。

## Goals / Non-Goals

目标是回答原PHONE_CORE是否优于完整N、是否优于同支持同范数平滑。仅修改曝光判定的聚合单位；不延长视频、不移动U、不增加音素、不更换样本、不做新的幅度扫描。

## Decisions

### 1. 输入与纯构造

输入快照分支A，包含P/Q/D与旧phone-core结果。以D/05_drivers/drivers.json逐record的12个driver（3seed×4条件）中的used_masks，经mask_sha256关联D/03_data/mask_manifest.json。12份去重后的mask身份必须相同；global_start_frame/end_frame须等于natural_core_start_frame/end_frame。非空label，排除sil/sp/spn/<eps>，只用L>=5的core；区间[0,308)内、不重叠。空集合为INPUT_DEGENERATE。禁止用padding mask范围冒充core。

M为float64的原natural mel。每core [s,e)、j=0..L-1：

```text
w[j] = min(1, j/2, (L-1-j)/2)
mu = mean(M[:,s:e], axis=time)
Rp[:,s+j] = w[j] * (mu - M[:,s+j])
S = reflect-pad2后沿时间用[1,4,6,4,1]/16卷积M
Rg[:,s+j] = w[j] * (S[:,s+j] - M[:,s+j])
Rg *= norm(Rp) / norm(Rg)  # record全矩阵Frobenius范数
a = min(0.25, 0.5 / max(max(abs(Rp)),max(abs(Rg))))
PHONE_CORE = float32(clip(M+a*Rp,-4,4))
GENERIC_CORE = float32(clip(M+a*Rg,-4,4))
```

core外残差0、两端w=0；clip后core外/两端须与原float32 M逐值相同。任一preclip范数<=1e-12、postclip范数为0、postclip norm(GENERIC−N)/norm(PHONE−N)不在[.95,1.05]均停止全队列INPUT_DEGENERATE。输入坏shape/非有限/重叠/hash冲突为BLOCKED。复算旧14条成功driver，应逐值相同；旧两条只重建缺少的公式结果，不改变规则。

### 2. 曝光在组级判定

独立构造93个chunk的列索引T[t]，规则见公共入口。SyncNet第r行读取视频帧r..r+4，支持 `S[r]=union(T[r:r+5])`。令W为该record所有w>0的mel列：

```text
exposed_rows_i = [r for r in U if intersection(S[r],W_i)非空]
record_exposed_i = len(exposed_rows_i)>0
group_exposed_g = any(record_exposed_i for i属于g)
```

逐record输出core数量、W、exposed_rows、候选实际changed_columns以及两种候选在S[r]上的扰动范数，后者是诊断不新增门槛。任何组group_exposed=false→全队列INPUT_DEGENERATE、0模型。每组通过则保留全部16条进入原实验；零曝光record仍在分母，真实生成评分，不填0或只报告14条。旧规则会拒绝的record集合应与旧input_audit一致，否则先解释来源不一致并BLOCKED。

采用组级规则的理由是原设计的推断单位就是source_group。删记录或把U改到干预最强处都会改变问题，故本轮不采用。

### 3. 运行和判定

通过prepare的支持验收后执行公共controls；两者都经独立validator PASS才生成32个候选。加2个N replay，总34视频；加2个parity，总36评分。使用公共P/F0、原N音轨、U/k0与A的bootstrap规则。

唯一主比较PHONE_CORE/N用公共gain。未过→NO_PHONE_CORE_GAIN_ESTABLISHED。主通过且PHONE_CORE/GENERIC_CORE的C/D/A三个99%CI下界全>0、至少7/8组三项同正→PHONE_CORE_SPECIFIC_SIGNAL_TO_CONFIRM；主通过但该机制条件不满足→NATURAL_GAIN_MECHANISM_UNRESOLVED。机制比较不另加mean ΔC>.05。完整报告GENERIC_CORE/N但不改主假设。

### 4. 最小实现与验收

实现包为 `wav2lip_phone_core_shrinkage`，对应命令中的 `<package>`。producer可借鉴旧construct的公式，移除其逐记录曝光异常，在聚合后判定；不直接调用会提前抛错的旧construct。validator自行重建core/权重/driver/曝光及统计，可共享文件读取。候选数值容差1e-6（旧driver逐值比较）、矩阵1e-4、统计1e-6；索引/计数/终态精确一致。

关键反例：同组一条无曝光另一条有曝光应保留两条；整组无曝光停止；只改变窗口外候选不得删除record；短core、L=5权重、重叠、错mask、clip范数失败；自由C变好而anchor变差不能阳性；伪造analysis并重签hash仍验收失败。

BM唯一实体：`Wav2Lip phone core group-support completion 2026-09-10`。输出注明旧逐记录停止、实际8组支持、每条零曝光情况及科学结论。

## Risks / Trade-offs

- 已见16条反复使用→仅探索，阳性另做新source-group确认。
- 零曝光记录可能稀释全U效果→保留原问题和分母，不把局部收益冒充完整replacement。
- mel候选未证明波形可达→即使阳性也不直接训练波形头。

## Migration Plan

新增隔离包及run；父包/产物不改。任一绑定冲突停止本分支即可回退，无主流水线迁移。
