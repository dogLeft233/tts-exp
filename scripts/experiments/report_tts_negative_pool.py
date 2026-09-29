"""Build the frozen negative-pool report and source provenance receipts."""
import collections
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.experiments.tts_pcm_residual import cluster

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"runs/tts_negative_pool_20260926"
SOURCE=ROOT/"runs/tts_prepost_mel_20260926"


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p,scores,support,analysis=(read(BASE/f) for f in ("protocol.json","scores.json","support.json","analysis.json"))
    supplementary={}
    byid={}
    offsets={a:collections.Counter() for a in "NT"}
    for c in scores:
        receipt=read(SOURCE/"scores"/c["image"]/(c["id"]+".json"))
        for a in "NT":
            offsets[a][receipt["cells"][a]["20"]["offset"]]+=1
        if not c["eligible"]:
            continue
        key=(c["id"],c["speaker"])
        d=byid.setdefault(key,collections.defaultdict(list))
        for m in ("C","D","B"):
            d["eligible_full_"+m].append(c["arms"]["T"]["official"]["full"][m]-c["arms"]["N"]["official"]["full"][m])
        for arm in "NT":
            for modality in ("audio","visual"):
                d[f"{arm}_{modality}_norm"].append(c["arms"][arm][modality+"_norm"])
            for pool in ("native","matched"):
                d[f"{arm}_{pool}_time"].append(np.mean([q["pools"][pool]["time_abs"] for q in c["arms"][arm]["queries"]]))
    for name in next(iter(byid.values())):
        supplementary[name]=cluster([np.mean(v[name]) for v in byid.values()],[k[1] for k in byid])
    diagnostics={"offset_histogram_full114":{a:dict(offsets[a]) for a in "NT"},"supplementary":supplementary,"support":{}}
    for path in (BASE,BASE/"lag_-1",BASE/"lag_+1"):
        sp=read(path/"support.json")["rows"]
        good=[r for r in sp if r["eligible"]]
        errors=[abs(n["errors"][a]) for r in good for n in r["nodes"] for a in "NT"]
        duplicated={a:0 for a in "NT"}
        pairs=0
        durations={a:[] for a in "NT"}
        for r in good:
            for q in r["queries"]:
                pairs+=len(q["donors"])
                for a in "NT":
                    js=[r["nodes"][dn]["j"][a] for dn in q["donors"]]
                    duplicated[a]+=len(js)-len(set(js))
                    durations[a].extend(abs(j-r["nodes"][q["node"]]["j"][a])/25 for j in js)
        info={"n":len(good),"speakers":len(set(r["speaker"] for r in good)),"queries":sum(len(r["queries"]) for r in good),"donor_phase_pairs":pairs,
              "duplicate_donor_grid_pairs":duplicated,"nearest_time_error_quantiles_s":np.quantile(errors,[0,.5,.95,1]).tolist(),
              "donor_abs_lag_quantiles_s":{a:np.quantile(durations[a],[0,.25,.5,.75,1]).tolist() for a in "NT"},
              "excluded":[{"id":r["id"],"queries":len(r["queries"]),"occurrences":r["n_occurrences"]} for r in sp if not r["eligible"]]}
        diagnostics["support"][path.name]=info
    (BASE/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    lines=["# 原生 TTS 的共同音素负样本池干预（2026-09-26）","",
           "主协议固定 lag 下未确认共同负池 margin/rank 优势；独立校准 lag 在静态脸上存在转移偏差。预先冻结的 k=2 敏感性中，优势在共同 event/phase 负池及单位范数表示下仍然存在。负池效应本身依赖 lag；这不足以对原生 Sync-C 增益作因果分解。","",
           "## 数据与冻结协议","",
           "复用前序 38 条 evaluation 音频、13 说话人、3 静态源图的原生 N/T 视频与 exact PCM 音频 embedding；无重新生成、无 GPU、无 embedding 插值。26 条不同 ID、同说话人 calibration 旧动态脸 N/T 的 guard20 offset 中位数 −1，故固定 audio_index−visual_index=k=1。预先附加 k=0、k=2 敏感性，未挑选最优。", "",
           "MFA speech occurrence 顺序严格相等；相邻 speech 的中间 gap 对应同一 occurrence 边界。两臂 interval≥80ms；phase=.25/.5/.75，最近40ms音频网格，音频窗中心j*.04+.1075s，视觉窗中心i*.04+.08s；≤20ms误差且中心仍在该标签。正对V[j−k],A[j]。speech query 两臂 guard20；共同 donor 在两臂均距 query 3..15格（120–600ms），每 query≥5 donors，每条≥8 queries且≥5 speech occurrences。所有支持门由 annotation/index 决定。","",
           "每个 query 原始池为 A[j+d], d=±3..15；共同池是两臂完全相同 occurrence/phase donor IDs。每 donor phase 等权，query 等权，3图条内均值，再条目在speaker内均值，speaker等权。最近格点重复保留并公开；无分数筛选。主端点 M=mean negative distance−fixed positive distance，不是官方 Sync-C。95/99%CI 为20000次speaker bootstrap(seed20260926)。","",
           "## 支持与校准转移","",
           "| k | 条目/说话人 | queries | donor phase pairs |", "|---|---:|---:|---:|"]
    for folder,k in [(BASE/"lag_-1",0),(BASE,1),(BASE/"lag_+1",2)]:
        d=diagnostics["support"][folder.name]
        lines.append(f"| {k} | {d['n']}/{d['speakers']} | {d['queries']} | {d['donor_phase_pairs']} |")
    lines.extend(["",f"主协议排除：{diagnostics['support'][BASE.name]['excluded']}。32条仍覆盖13speaker。时差、重复及各门排除计数见 support.json / diagnostics.json。", "",
                  f"静态脸 full guard20 最佳 official offset 分布（各114 cells）：N={dict(offsets['N'])}，T={dict(offsets['T'])}。旧动态脸校准−1不能当作新静态脸正确物理事件 lag；k=1主结果具有明确校准转移限制。", "", "## 官方分数到新端点的桥接", "",
                  "下表 T−N；同query官方曲线仍先逐lag平均，再median/min。全支持→入组32条全行→共同event queries分别公开。固定k替换min，和mean替换median是不同算术变化。原始池是正对中心±3..15，在k=1时对应官方相对视觉lag −14..−2及4..16，多出+16列；另列官方31列中 |lag−k|≥3 的mean版本，防止隐含改变混入。", "",
                  "| 端点 | T−N | 95%CI | 99%CI |", "|---|---:|---|---|"])
    def row(label,r):
        return f"| {label} | {r['speaker_mean']:+.6f} | [{r['speaker_ci95'][0]:+.6f}, {r['speaker_ci95'][1]:+.6f}] | [{r['speaker_ci99'][0]:+.6f}, {r['speaker_ci99'][1]:+.6f}] |"
    for label,r in [("官方C，38条全行",analysis["official_full_C"]),("官方C，32条全行",supplementary["eligible_full_C"]),("官方C，同query",analysis["official_queries_C"]),("同query median−fixed D",analysis["official_queries_fixed_median_margin"]),("同query official negative mean−fixed D",analysis["official_queries_fixed_mean_official_margin"]),("原始池 M",analysis["raw_native_margin"]),("共同池 M",analysis["raw_matched_margin"]),("负池干预差中差",analysis["raw_did_margin"])]:
        lines.append(row(label,r))
    lines.extend(["","不能将 +0.520 到 +0.114 的变化全部归给负池。主协议中优势先在 min→固定k 步骤失去确认；共同池干预差中差本身跨0。这首先是旧动态cal到新静态eval的测量校准不匹配，不能据此解释为原生优势来自lag选择或优化。另行采用固定26cal在同静态协议修复。","", "## 固定 lag 敏感性", "", "| k / 端点 | T−N | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for folder,k in [(BASE/"lag_-1",0),(BASE,1),(BASE/"lag_+1",2)]:
        a=read(folder/"analysis.json")
        for name in ("raw_native_margin","raw_matched_margin","raw_did_margin","raw_matched_rank","unit_matched_margin","unit_matched_rank"):
            lines.append(row(f"k={k} {name}",a[name]))
    lines.extend(["","k=2 是事前固定敏感性而非主检验；这一条件下，负池构成相同后 raw margin、条件rank、unit margin/rank 均保留99%正区间，且差中差不确定。它弱化“只有负池组成”或“只有embedding范数”解释，但主k=1结果不确定、k=0负池干预甚至显著负向，故不能宣称跨lag稳健。", "", "## 各臂正负距离与组成", "",
                  "| arm/pool | positive | negative | M | rank |", "|---|---:|---:|---:|---:|"])
    for a in "NT":
        for pool in ("native","matched"):
            vals=[analysis[f"level_{a}_raw_{pool}_{m}"]["speaker_mean"] for m in ("positive","negative","margin","rank")]
            lines.append(f"| {a}/{pool} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines.extend(["","| arm/pool | same occurrence | same phone other | different phone | gap | mean |lag| (s) |", "|---|---:|---:|---:|---:|---:|"])
    for a in "NT":
        for pool in ("native","matched"):
            vals=[analysis[f"level_{a}_composition_{pool}_{cat}"]["speaker_mean"] for cat in ("same_occurrence","same_phone_other","different_phone","silence_gap")]
            vals.append(supplementary[f"{a}_{pool}_time"]["speaker_mean"])
            lines.append(f"| {a}/{pool} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines.extend(["","共同池统一的是 donor event/phase ID与类别权重；两臂实际 |lag| 分布仍不同（T较短），且相比原始池有时间窗选择变化。因此识别的是联合参照池干预，不能专称纯phone组成效应。","", "## 表示自分离与分层", "", "| 端点，k=1 | T−N | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for name in ("raw_matched_AA","raw_matched_VV","unit_matched_AA","unit_matched_VV"):
        lines.append(row(name,analysis[name]))
    lines.extend(["","A/A与V/V分离度在raw/unit下都提高，但这是表示几何诊断，不能独自解释AV同步或嘴型真值。分层需每query≥3该类donors、每条≥5该类queries；主k=1 different-phone有32条/13speaker，同phone其他实例15条/10speaker，gap7条/6speaker，同occurrence无足够支持。完整分层统计保存在analysis.json；缺失未放宽门槛。","", "## 验证与溯源", "",
                  "重放228个原生距离矩阵，float64独立距离与原float32 eps=1e−6距离最大误差见replay.json。全部资产SHA、三种lag的逐query距离/rank/支持、speaker bootstrap复算，以及6个历史known-delay控制见各validation.json；测试4 passed。源码快照、协议/产物SHA见provenance.json。", "",
                  "代码：scripts/experiments/tts_negative_pool.py、check_tts_negative_pool.py、report_tts_negative_pool.py；测试：tests/experiments/tts_negative_pool/test_protocol.py。", "",
                  "## 解释边界", "",
                  "MFA跨臂标签一致不等于物理发音事件真值；phone不是viseme，200ms输入窗跨多phone，静音gap也不是经人工确认的静音。校准/eval为ID分离而非speaker分离；38条为历史42条eval中phone-compatible子集，支持后仅32条。单位归一化改变距离几何，不能按raw与unit之差计贡献。三张脸是固定历史图，不能当独立统计样本。结论针对冻结Wav2Lip/SyncNet及当前数据，没有人眼评分/感知同步结论。", "",
                  "与旧2026-09-17 A_rank的新增点：原生优势已在当前静态几何确认，统一PCM，三脸，且在同query上实际干预event/phase负参照池并给出官方统计量桥接；不是把旧rank指标机械复跑。"])
    (BASE/"report.md").write_text("\n".join(lines)+"\n")
    provenance={}
    paths=[Path(__file__),ROOT/"scripts/experiments/tts_negative_pool.py",ROOT/"scripts/experiments/check_tts_negative_pool.py",ROOT/"tests/experiments/tts_negative_pool/test_protocol.py"]
    snapshot=BASE/"source"
    snapshot.mkdir(exist_ok=True)
    for path in paths:
        shutil.copy2(path,snapshot/path.name)
        provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in BASE.rglob("*.json"):
        if path.name!="provenance.json":
            provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance[str((BASE/"report.md").relative_to(ROOT))]=hashlib.sha256((BASE/"report.md").read_bytes()).hexdigest()
    (BASE/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(BASE/"report.md")


if __name__=="__main__":
    main()
