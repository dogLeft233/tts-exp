"""Standalone native midpoint intervention report and fixed summary figures."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"runs/tts_native_midpoint_20260926"
NAMES={"dynamic_cloud":"动态云", "static_cloud":"静态云", "static_local":"静态本地Q"}


def read(name):
    return json.loads((OUT/name).read_text())


def fmt(x, precision=3):
    a,b=x["ci99"]
    return f"{x['mean']:+.{precision}f} [{a:+.{precision}f}, {b:+.{precision}f}]"


def main():
    p,s,control,verify=read("protocol.json"),read("summary.json"),read("controls.json"),read("independent_validation.json")
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax,group,title in zip(axes,NAMES,("Dynamic cloud","Static cloud","Static local")):
        r=s[f"{group}/raw/eval"]
        xx=[r["effects"][q]["C"]["residual_TminusN"] for q in ("T2N/base","T2N/M","N2T/M")]
        means=np.array([x["mean"] for x in xx]);ci=np.array([x["ci99"] for x in xx])
        ax.errorbar(range(3),means,yerr=np.stack((means-ci[:,0],ci[:,1]-means)),fmt="o",capsize=4)
        ax.axhline(0,color="gray",lw=.8)
        ax.set_xticks(range(3),["Native baseline","Scale T midpoint\nto N","Scale N midpoint\nto T"],rotation=15,ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("T - N confidence, raw geometry (99% speaker CI)")
    fig.tight_layout();fig.savefig(OUT/"native_midpoint_residuals.png",dpi=180);fig.savefig(OUT/"native_midpoint_residuals.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7))
    for ax,geometry in zip(axes,("raw","unit")):
        r=s[f"dynamic_cloud/{geometry}/eval"]
        for arm,label in (("N","Natural"),("T","TTS")):
            xx=[r["dose"][arm+"/"+d]["C"] for d in ("dose075","base","dose125")]
            means=np.array([x["mean"] for x in xx]);ci=np.array([x["ci99"] for x in xx])
            ax.errorbar([.75,1.,1.25],means,yerr=np.stack((means-ci[:,0],ci[:,1]-means)),marker="o",capsize=3,label=label)
        ax.axhline(0,color="gray",lw=.8);ax.set_xlabel("Midpoint dynamics multiplier alpha");ax.set_title(geometry+" geometry");ax.set_ylabel("C change from alpha=1 (99% CI)")
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/"midpoint_fixed_doses.png",dpi=180);fig.savefig(OUT/"midpoint_fixed_doses.pdf");plt.close(fig)
    with (OUT/"effects.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["group","geometry","support","direction","condition","metric","contrast","n","speakers","mean","ci99low","ci99high"])
        for key,r in s.items():
            for effect,rr in r["effects"].items():
                for metric in ("C","B","D","D_anchor"):
                    for contrast,x in rr[metric].items():
                        w.writerow([*key.split("/"),*effect.split("/"),metric,contrast,x["n"],x["speakers"],x["mean"],*x["ci99"]])
    lines=["# 原配视听中点动态与配对差向量：可控表示干预", "",
           "## 结论", "",
           "动态云的独立校准 ID 外 48 条评估中，在保持每个 anchor 配对差向量不变时，把 TTS 中点动态幅度匹配到自然臂，会明确降低 TTS 的原配 C；反向放大自然臂也明确缩小 T−N 分差。两方向处理后的正残余仍存在，不能视为全解释。unit 几何中的方向一致。静态两组的 M-only 处理效应在 99% CI 下尚未确认，不能把动态结论推广到所有源图/生成条件。", "",
           "这是原配计分函数上的表征干预：原始音视频不变，虚拟 embedding 变换不等于实际嘴型改善。M 只称视听中点轨迹，R 只称配对差向量；不把它们当作共享生理因子与真实口型误差。", "",
           "## 分母、独立校准与原生基线", "",
           "source PCM、官方联合长度及缓存固定前端；guard20 的原 query 行和原 ±15 lag 均保持。用旧26cal的有效 raw guard20最佳lag整数中位数选共同k0，半整数向0截断。unit沿用此k0。所有臂中位数差≤1、|k0|≤5的门通过。共同k0只是校准的测量代理，不是逐条物理事件真值。", "",
           "|组|cal有效cell数|臂lag中位数|k0|主eval n/speaker|N原生C|T原生C|原生raw ΔC（99% CI）|", "|---|---|---|---|---|---|---|---|"]
    for group,g in p["groups"].items():
        r=s[f"{group}/raw/eval"]["effects"]["T2N/base"]["C"]["residual_TminusN"]
        lines.append(f"|{NAMES[group]}|{g['calibration_counts']}|{g['calibration_arm_medians']}|{g['k0']}|{r['n']}/{r['speakers']}|{fmt(s[f"{group}/raw/eval"]["scores"]["N_base"]["C"])}|{fmt(s[f"{group}/raw/eval"]["scores"]["T_base"]["C"])}|{fmt(r)}|")
    lines += ["", "动态完整74中排除26cal后48条为主；同speaker可能同时有cal与eval，不是未见speaker确认。静态仍从原74eval取组内共同guard20支持：云排除a1_013/a1_053/a1_100，本地排除a1_013/a1_029。没有ASR/MFA内容筛选，实际长/重复输出仍保留。静态本地cal仅a1_025/Q2联合L38不足guard20，其他同ID臂仍参与pooled校准；无按score删样。", "",
              "## 主 M-only 干预", "",
              "T→N：只将T中点动态幅度乘sigmaN/sigmaT，N保持原状；N→T：只将N中点动态乘其倒数，T保持原状。处理效应是被处理臂的C变化；残余始终按T减N。Q1/Q2逐个配对、逐个计分，然后条内均值。speaker等权，20k PCG64(20260926) bootstrap，99%CI。", "",
              "|组/几何|T→N 的T处理效应|T→N 后T−N残余|N→T 的N处理效应|N→T 后T−N残余|", "|---|---|---|---|---|"]
    for geometry in ("raw","unit"):
        for group in NAMES:
            r=s[f"{group}/{geometry}/eval"]
            t=r["effects"]["T2N/M"]["C"];n=r["effects"]["N2T/M"]["C"]
            lines.append(f"|{NAMES[group]}/{geometry}|{fmt(t['treatment'])}|{fmt(t['residual_TminusN'])}|{fmt(n['treatment'])}|{fmt(n['residual_TminusN'])}|")
    lines += ["", "raw与unit不可跨量纲相减或算贡献比例。unit从逐行L2归一化的原始embedding重新估计全部M/R尺度，变换后不再归一化；它是另一距离几何，不冒充官方Sync-C。", "",
              "### 动态主队列的 B/D 与 anchor", "",
              "|方向/raw M-only|ΔC|ΔB|ΔD(min)|ΔD固定k0|bestlag改变比例|", "|---|---|---|---|---|---|"]
    r=s["dynamic_cloud/raw/eval"]
    for direction in ("T2N","N2T"):
        x=r["effects"][direction+"/M"]
        anchor=x["D_anchor"]["treatment"]
        lines.append(f"|{direction}|{fmt(x['C']['treatment'])}|{fmt(x['B']['treatment'])}|{fmt(x['D']['treatment'])}|{anchor['mean']:.3g}|{fmt(x['lag_changed_fraction'])}|")
    lines += ["", "anchor逐向量差保持不变，但C里的D仍跨31lag取最小值；其他lag变化可令最佳lag迁移，所以ΔD(min)不必为0。T→N 的C下降由B下降和min D变化共同形成，不能把C变化称为真实同步误差变化。", "",
              "## 对照四格与处理后残余", "",
              "R-only只乘rhoN/rhoT（反向用倒数）；both同时改变两尺度。下表全部预设，未按效果挑选。", "",
              "|组/几何|方向/条件|被处理臂ΔC|处理后T−N|", "|---|---|---|---|"]
    for geometry in ("raw","unit"):
        for group in NAMES:
            r=s[f"{group}/{geometry}/eval"]
            for direction in ("T2N","N2T"):
                for condition in ("base","M","R","both"):
                    x=r["effects"][direction+"/"+condition]["C"]
                    lines.append(f"|{NAMES[group]}/{geometry}|{direction}/{condition}|{fmt(x['treatment'])}|{fmt(x['residual_TminusN'])}|")
    lines += ["", "raw动态R-only处理效应尚不明确；unit下R-only有更明确变化，说明几何选择会改变两种操作的读数。两项并非独立因子或可相加的生理成分；both的结果也不授权百分比分解。", "",
              "## 固定剂量", "",
              "两臂分别使用alpha=.75/1/1.25、beta=1，不选最优点。alpha=1为原生基线、处理效应0。", "",
              "|组/几何|臂|alpha .75 的ΔC|alpha 1.25 的ΔC|", "|---|---|---|---|"]
    for geometry in ("raw","unit"):
        for group in NAMES:
            r=s[f"{group}/{geometry}/eval"]
            for arm in ("N","T"):
                lines.append(f"|{NAMES[group]}/{geometry}|{arm}|{fmt(r['dose'][arm+'/dose075']['C'])}|{fmt(r['dose'][arm+'/dose125']['C'])}|")
    lines += ["", "剂量结果显示本计分函数对中点动态幅度很敏感；它自身不能证明自然/TTS差异只由该幅度产生。", "",
              "## 系数与操纵检查", "",
              "|组/几何|alpha T→N：speaker均值99CI|alpha配对中位数[IQR]|beta T→N：speaker均值99CI|beta配对中位数[IQR]|", "|---|---|---|---|---|"]
    for geometry in ("raw","unit"):
        for group in NAMES:
            r=s[f"{group}/{geometry}/eval"]["coefficients"]
            a,b=r["alpha_T2N"],r["beta_T2N"]
            aq,bq=a["pair_quantiles_0_5_25_50_75_95_100"],b["pair_quantiles_0_5_25_50_75_95_100"]
            lines.append(f"|{NAMES[group]}/{geometry}|{fmt(a['statistics'])}|{aq[3]:.3f} [{aq[2]:.3f},{aq[4]:.3f}]|{fmt(b['statistics'])}|{bq[3]:.3f} [{bq[2]:.3f},{bq[4]:.3f}]|")
    lines += ["", "完整min/p5/p25/p50/p75/p95/max与逐对系数已保存。sigma/rho目标按定义达到，muM不变；所有被访问有效坐标均变换。summary中的manipulation保存各臂各条件变换前后sigma/rho的speaker摘要。", "",
              "## 原始公式与支持证明", "",
              "在共同校准k0下，A'_t=A_(t+k0)，M_t=(A'_t+V_t)/2，R_t=(A'_t−V_t)/2。I=20..L−21，muM=mean_I(M)，sigmaM²=mean_I||M−muM||²，rhoR²=mean_I||R||²。M'=muM+alpha(M−muM)，R'=betaR；V'=M'−R'，Anew_(t+k0)=M'+R'。", "",
              "坐标域J=max(0,−k0)..min(L−1,L−1−k0)。评分仍比较V'_i与Anew_(i+s)，i∈I、s∈[−15,15]。因为|k0|≤5，i和q=i+s−k0均在J，没有额外丢query、插值、端点补值或改变原lag。", "",
              "每个距离的向量差 = alpha(M_i−M_q)−beta(R_i+R_q)。anchor s=k0时q=i，中点项严格消失，alpha不会改变配对差。其他lag包含中点的时间变化。M加同一时间常量也在所有被评分距离中抵消。", "",
              "独立复算的平方距离使用alpha²||ΔM||²+beta²||ΣR||²−2alpha beta〈ΔM,ΣR〉，并加入pairwise_distance的2epsilon·sum(alphaΔM−betaΣR)+1024epsilon²。主计算直接重建向量后按float32/eps1e−6评分，没有用公式近似替代原端点。", "",
              "M/R构造与尺度计算用float64；评分前转float32。以下将精确代数不变量与实际float32舍入误差分开列出。", "",
              "|控制|最大误差|", "|---|---:|"]
    for k,v in control.items():
        lines.append(f"|{k}|{v:.6g}|")
    lines += ["", f"独立复核 {verify['pair_geometries']} 个pair×geometry、{verify['direct_distance_cells']:,} 个直接距离及等量平方公式距离。最大误差：直接 {verify['max_errors']['direct_distance']:.3g}、平方式 {verify['max_errors']['quadratic_distance']:.3g}、统计 {verify['max_errors']['statistics']:.3g}。4项单元测试通过，原native guard20矩阵重放误差0。全程CPU，没有GPU/新TFG。", "",
              "## 支持桥接", "",
              "动态full74包含校准ID，仅探索桥接；静态previous70沿用上一阶段四臂共同支持。主结论不从桥接中挑分母。", "",
              "|桥接/几何|原生ΔC|T→N M-only处理效应|处理后残余|", "|---|---|---|---|"]
    for key,r in s.items():
        if key.endswith("/eval"):
            continue
        base=r["effects"]["T2N/base"]["C"]["residual_TminusN"];m=r["effects"]["T2N/M"]["C"]
        lines.append(f"|{key}|{fmt(base)}|{fmt(m['treatment'])}|{fmt(m['residual_TminusN'])}|")
    lines += ["", "## 边界与下一步", "",
              "本轮识别了动态原配分差对中点轨迹幅度的可控敏感性，并保留明确残余。它没有识别幅度变化来自TTS声学、Wav2Lip生成、SyncNet表征哪条路径，也不把归一化得到的变化称为生理原因。静态效应不确定、独立新speaker尚未验证，不能据此声称所有TTS更同步。是否需要新的声学/生成路径确认交由理论端决定；本实验不自动下载或生成新批。", "",
              "## 可复现产物", "",
              "- protocol.json：独立cal结果、解释gate、完整输入/分母与冻结参数。",
              "- baseline_before_intervention.json：查看新干预效应前列出的主支持原生C。",
              "- pairs/：每个T实例、双几何的12矩阵、四格/剂量分数、操纵和控制。",
              "- summary.json / effects.csv / coefficients.json：speaker CI、双方向完整对比与支持桥接。",
              "- controls.json / independent_validation.json / artifact_hashes.json：控制、独立复算与来源。",
              "- native_midpoint_residuals.pdf/png、midpoint_fixed_doses.pdf/png：独立可导出图。",
              "- scripts/experiments/tts_native_midpoint.py、check_tts_native_midpoint.py、report_tts_native_midpoint.py；tests/test_tts_native_midpoint.py。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    paths=[x for x in OUT.rglob("*") if x.is_file() and x.name!="artifact_hashes.json"]
    paths += [ROOT/"scripts/experiments"/x for x in ("tts_native_midpoint.py","check_tts_native_midpoint.py","report_tts_native_midpoint.py","tts_native_boundary_audit.py")]
    paths += [ROOT/"tests/test_tts_native_midpoint.py"]
    h={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    (OUT/"artifact_hashes.json").write_text(json.dumps(h,indent=2)+"\n")
    print("report saved",len(h),"hashes")


if __name__=="__main__":
    main()
