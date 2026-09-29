"""Write tables and standalone figures from the frozen CPU analysis."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tts_native_boundary_audit_20260926"


def read(name):
    return json.loads((OUT/name).read_text())


def fmt(x):
    value = f"{x['mean']:+.3f}"
    return value if x["ci99"] is None else value+f" [{x['ci99'][0]:+.3f}, {x['ci99'][1]:+.3f}]"


def main():
    s, regions = read("summary.json"), read("region_summary.json")
    support, audit = read("supports.json"), read("audit.json")
    changes = read("intervention_summary.json")
    panels = [("dynamic", "T-N", "动态云 TTS", "Dynamic cloud"),
              ("static", "C-N", "静态云 TTS", "Static cloud"),
              ("static", "Q-N", "静态本地 Q1/Q2", "Static local")]
    policies = ["guard0", "valid", "guard15", "guard20"]
    labels = ["Full padded", "Valid pairs", "Guard 15", "Guard 20"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (panel, comp, _, english) in zip(axes, panels):
        xx = [s["native"][panel][f"source/{comp}/joint/{pol}/common"]["C"]["delta"] for pol in policies]
        means = np.array([x["mean"] for x in xx]); ci = np.array([x["ci99"] for x in xx])
        ax.errorbar(np.arange(4), means, yerr=np.stack((means-ci[:, 0], ci[:, 1]-means)), fmt="o", capsize=4)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_title(english)
    axes[0].set_ylabel("Native TTS - N confidence (99% speaker CI)")
    fig.tight_layout()
    fig.savefig(OUT/"native_scoring_policies.png", dpi=180)
    fig.savefig(OUT/"native_scoring_policies.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for col, (panel, comp, _, english) in enumerate(panels):
        t_arms = ["Q1", "Q2"] if comp == "Q-N" else [comp[0]]
        n = regions[f"{panel}/source/N/joint"]
        ts = [regions[f"{panel}/source/{a}/joint"] for a in t_arms]
        for ri, region in enumerate(("I", "E", "P")):
            ax = axes[ri, col]
            nz = np.array(n["regions"][region]["curve"]["mean"])
            tz = np.mean([t["regions"][region]["curve"]["mean"] for t in ts], axis=0)
            if region == "P":
                nz[15] = tz[15] = np.nan
            ax.plot(np.arange(-15, 16)*40, nz, label="N")
            ax.plot(np.arange(-15, 16)*40, tz, label="TTS")
            ax.set_title(f"{english}: {region}")
            if ri == 2:
                ax.set_xlabel("Audio lag (ms)")
            if col == 0:
                ax.set_ylabel("Mean distance")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT/"region_curves.png", dpi=180)
    fig.savefig(OUT/"region_curves.pdf")
    plt.close(fig)
    with (OUT/"native_tables.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "channel", "comparison", "length", "policy", "support", "metric", "arm", "n", "speakers", "mean", "ci99_low", "ci99_high"])
        for panel, results in s["native"].items():
            for key, result in results.items():
                for metric in ("C", "B", "D", "best_lag"):
                    for arm, x in result[metric].items():
                        writer.writerow([panel, *key.split("/"), metric, arm, x["n"], x["speakers"], x["mean"], *(x["ci99"] or [None, None])])
    lines = ["# 原配 TTS Sync-C：有效配对与边界计分干预", "",
             "## 核心结果", "",
             "动态云 74 条/13 speaker 的原配增益在删除虚构 embedding padding 后仍明确为正，而且增益增大。静态云的全有效轨迹也为正；静态本地的全有效轨迹尚不明确，其 guard20 阳性依赖更靠内部的 query 支持。结论不能跨生成前端/源图条件泛化。", "",
             "主分析采用**官方计分公式与联合长度的固定前端评估**。以下均同支持：动态 74，静态四臂共同 70/15 speaker。TTS 实例先分别计分后条内平均，speaker 等权，20,000 次 bootstrap、99% CI、PCG64(20260926)。", "",
             "|队列|guard0|仅有效 pair|guard15|guard20|", "|---|---|---|---|---|"]
    for panel, comp, title, _ in panels:
        xs = [s["native"][panel][f"source/{comp}/joint/{pol}/common"]["C"]["delta"] for pol in policies]
        lines.append("|"+title+"|"+"|".join(fmt(x) for x in xs)+"|")
    lines += ["", "### 对计分策略的配对干预", "", "|队列|去 padding 后 ΔC 的变化|valid→guard20 的 ΔC 变化|有效 head/interior/tail 等权 ΔC|", "|---|---|---|---|"]
    for panel, comp, title, _ in panels:
        a = changes[f"{panel}/source/{comp}/joint/valid-guard0"]["C"]
        b = changes[f"{panel}/source/{comp}/joint/guard20-valid"]["C"]
        eq = s["native"][panel][f"source/{comp}/joint/equal_valid_regions/common"]["C"]["delta"]
        lines.append(f"|{title}|{fmt(a)}|{fmt(b)}|{fmt(eq)}|")
    lines += ["", "去除 padding 的操作是逐 lag 删除无效 audio 索引后重新归一化，query 集合因此依 lag 而变。guard15/20 则要求所有 lag 共用同一内部 query 集合。valid→guard20 同时改变真实边缘上下文与 query 权重，不能把它称为纯静音或音素组成效应。", "",
              "## 精确算术分解", "",
              "对每个 lag，把全曲线分成内部 I、真实有效边缘 E、虚构 padding P。交换 N/T 的整组区内曲线和整组区权重，共四格，并取两个交换顺序的平均。表中曲线项+权重项=总原配 ΔC；这是非线性 median/min 函数的精确算术归因，不是生成机制的因果贡献。", "",
              "|队列/策略|总 ΔC|区内曲线项|区权重项|", "|---|---|---|---|"]
    for panel, comp, title, _ in panels:
        for policy in ("guard0", "valid"):
            x = s["shapley"][f"{panel}/source/{comp}/joint/{policy}"]["C"]
            lines.append(f"|{title}/{policy}|{fmt(x['total'])}|{fmt(x['curve'])}|{fmt(x['weight'])}|")
    lines += ["", "动态 valid 的区权重项接近零且 CI 跨零，区内曲线项仍为正。padding 与长度/区权重没有制造这个动态原配增益；在原 full padded 策略下，区权重项反而抑制增益。该结论尚未区分固定毫秒错位的音素/时间尺度组成与有效音视频表示的真实分离度。", "",
              "## B/D 与最佳 lag", "",
              "B=31 个 lag 平均距离的中位数，D=最小平均距离，C=B−D。B 上升不是自动的测量偏差。以下是分别计算 B/D 后的配对差，最优 lag 在每条各策略内重新取最小值。完整最佳 lag、N/T 各自 B/D/C、每条曲线见 cells 和 native_tables.csv。", "",
              "|队列/策略|ΔB|ΔD|ΔC|", "|---|---|---|---|"]
    for panel, comp, title, _ in panels:
        for policy in policies:
            x = s["native"][panel][f"source/{comp}/joint/{policy}/common"]
            lines.append(f"|{title}/{policy}|{fmt(x['B']['delta'])}|{fmt(x['D']['delta'])}|{fmt(x['C']['delta'])}|")
    lines += ["", "## 联合长度与历史桥接", "",
              "令 F 为缓存视觉前端实际解码帧数，S 为 16 kHz PCM 样本数。官方联合循环 L=min(F,floor(S/640))−5；本批 track_start 均为 0。旧独立窗口提取先得到 F−4 个视觉行、floor(S/640)−5 个音频行，再取较短长度，导致视频限制的静态缓存多 1 行。", "",
              "- 动态 296 cells（N/T×source/legacy）联合长度全部等于旧长；新动态主表精确重放历史 sourcePCM guard0 +0.741、guard15 +0.807、guard20 +0.935。",
              "- 静态审计 520 cells（100 主脸四臂+15 条额外两脸），全部修正 −1 行。新主评分只用原 74 eval/image3；26 cal 不混入。",
              "- Ditto source 94 cells 长度不变；legacy 94 cells 比旧共同截断扩展 1–3 行。这是 source/legacy 各自联合长度桥接，不把声码差与长度差混同。",
              "- 末端 audio padding 在截短两模态后重建。单纯裁掉矩阵最后一行不能复现新的边界。", "",
              "|静态对比/策略|旧长度 ΔC|联合长度 ΔC|配对长度修正差|", "|---|---|---|---|"]
    for comp in ("C-N", "Q-N"):
        for policy in ("guard0", "guard20"):
            old = s["native"]["static"][f"source/{comp}/old/{policy}/common"]["C"]["delta"]
            new = s["native"]["static"][f"source/{comp}/joint/{policy}/common"]["C"]["delta"]
            diff = changes[f"static/source/{comp}/joint-old/{policy}"]["C"]
            lines.append(f"|{comp}/{policy}|{fmt(old)}|{fmt(new)}|{fmt(diff)}|")
    lines += ["", "### 支持台账", "",
              "主表对同一队列所有四个策略、两种长度与对比使用固定共同记录。动态 74/74、Ditto 47/47；静态四臂共同 70/74（15 speaker），长度修正未新增排除。原 guard0 表也受旧 guard20 可评分门约束，不是全部短片段的 guard0。", "",
              "|静态排除 ID|N 旧→联合 L|C|Q1|Q2|", "|---|---|---|---|---|"]
    for e in support["static"]["excluded"]:
        lines.append("|"+e["id"]+"|"+"|".join(f"{x['old_L']}→{x['joint_L']}" for x in e["cells"])+"|")
    lines += ["", "最大支持补充不要求所有策略均可算：guard0/valid 覆盖全部原 74 eval；guard15/20 按各自 L 门和具体对比变化，详见 supports.json 与下表。无 ASR/CER 过滤，已知长重复 Q1 也保留，因此本地结果含实际生成质量分布。", "",
              "|静态对比/策略|最大支持 n/speaker|ΔC|", "|---|---|---|"]
    for comp in ("C-N", "Q-N", "Q-C"):
        for policy in policies:
            x = s["native"]["static"][f"source/{comp}/joint/{policy}/maximal"]["C"]["delta"]
            lines.append(f"|{comp}/{policy}|{x['n']}/{x['speakers']}|{fmt(x)}|")
    lines += ["", "## Ditto 描述与边界", "",
              "47 条均来自 S0765，一个音频 speaker，以下只给该历史队列均值，不给推断 CI。source PCM 的 full/valid 接近零，不影响动态云多 speaker 的阳性证据；legacy 包含历史编码链差异。", "",
              "|音频链/长度|guard0|valid|guard15|guard20|", "|---|---|---|---|---|"]
    for channel in ("source", "legacy"):
        for ver in ("old", "joint"):
            vals = [s["native"]["ditto"][f"{channel}/T-N/{ver}/{p}/common"]["C"]["delta"] for p in policies]
            lines.append(f"|{channel}/{ver}|"+"|".join(fmt(x) for x in vals)+"|")
    lines += ["", "## 公式、解释限制与复核", "",
              "- 每行 i、lag s∈[−15,15]（40 ms/格）比较 V_i 与 A_(i+s)。越界 A 是全零 embedding。距离用 float32 `pairwise_distance(eps=1e−6)`；curve 用 float64 均值以匹配既有报告。offset=−argmin_lag。官方 Torch float32 归约差另作数值核验，不改变端点命名。",
              "- guard0 每 lag L 行；valid 每 lag L−|s| 行；guard15 共同 L−30 行；guard20 共同 L−40 行。L>40 时 I/E/P 每 lag 分别 L−40、40−|s|、|s| 行。P 在 s=0 无行，权重0、曲线占位0，不解释为真实距离0。",
              "- 实际 head/tail 是最初/最后20个 query 起点（各800 ms）。5帧视觉和20帧 MFCC 窗口跨多个音素，真实端点可含静音、发音和生成上下文；本分析没有按语音活动划分，也不把 guard20 称为纯语音。",
              "- 只去除了 lag 搜索的虚构零 embedding，不消除音频特征、视频生成和解码前端固有的边缘上下文。相同旧缓存前端保留，不能称整个官方检测/裁剪/JPEG流水线完全一致。",
              "- 区内 `positive/background` 仅指该臂该策略的最小/中位曲线 lag 上的距离，不代表物理同步或错误音素真值。曲线图是 speaker 等权的距离描述；不会对平均曲线算 C 来代替逐条 C。",
              "- 曲线×权重 4 格的所有组合与 B/D/C 存在 pair_shapley.json；Q1/Q2分别与N计算后条内均值，未先混合两个实例的曲线再取 median/min。无贡献百分比。", ""]
    validation = read("independent_validation.json")
    scoring = read("scoring_validation.json")
    lines += [f"复核：{scoring['cells']} cells、{validation['distance_cells']:,} 个距离；旧矩阵重放最大误差 {scoring['archived_matrix_max_error']:.3g}；NumPy独立距离最大误差 {validation['max_errors']['distance']:.3g}；独立统计最大误差 {validation['max_errors']['statistics']:.3g}。",
              f"I/E/P 曲线闭合误差 {s['max_curve_closure_error']:.3g}，Shapley B/D/C 闭合误差 {s['max_shapley_closure_error']:.3g}；官方 float32 与保留 float64 归约最大差 {scoring['official_float32_vs_float64_reduction_max_error']:.3g}。4 个单元测试通过。全部 CPU，无 GPU 模型推理。",
              "", "## 产物", "",
              "- audit.json / source_hashes.json：来源、旧/联合长度、输入哈希和前端 metadata。",
              "- protocol.json：新效应计算前冻结；proposal.md：评分接口与可复现命令。",
              "- cells/：每 cell 两种矩阵、每 lag 有效数、各区曲线、B/D/C、最佳 lag。",
              "- summary.json / native_tables.csv / intervention_summary.json：同支持、最大支持、配对策略与长度差。",
              "- pair_shapley.json / region_summary.json：四格逐条分解、各臂区曲线/计数/距离摘要。",
              "- native_scoring_policies.pdf/png / region_curves.pdf/png：独立可导出图。",
              "- independent_validation.json / scoring_validation.json / artifact_hashes.json：复算与产物溯源。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    proposal = """# 冻结接口与复现

理论端在新解释效应计算前指定：动态云74 sourcePCM为主，静态74eval/image3为复核，Ditto47单speaker仅描述。保留旧前端，联合循环 L=min(F,floor(S/640))−5 为新主长度；旧长度仅桥接。未生成新音视频，未调用GPU。

对31个lag，C(z)=median(z)−min(z)，B=median(z)，D=min(z)。所有策略先按query行求lag曲线，再执行该非线性公式。valid只删除越界音频索引；guard15为全lag共同有效；guard20进一步去掉各端5个真实query。并列报告完整支持与每lag有效计数。

L>40时：I=[20,L−20)，E=有效且不在I，P=无效；数量L−40、40−|s|、|s|。z=Σ_r w_r z_r 精确闭合。valid将P权重设0，再按L−|s|归一化。独立次要反事实为真实head20/interior/tail20的三曲线等权，不搜索最佳权重。

对N/T两臂各自的区曲线Z与权重W，f_ab=C(Σ_r Z_(a,r) W_(b,r))。曲线项=.5[(f10−f00)+(f11−f01)]；权重项=.5[(f01−f00)+(f11−f10)]。对B/D同样计算。两项和=原native差，完全是算法算术。Q1/Q2逐个作4格，然后条内均值。bootstrap以speaker为独立单位，条内/同speaker先均值，20k PCG64 seed20260926；主99CI，保留95CI。

```bash
.venv/bin/python scripts/experiments/tts_native_boundary_audit.py audit
.venv/bin/python scripts/experiments/tts_native_boundary_audit.py freeze
.venv/bin/python scripts/experiments/tts_native_boundary_audit.py score
.venv/bin/python scripts/experiments/tts_native_boundary_audit.py analyze
.venv/bin/python scripts/experiments/check_tts_native_boundary_audit.py
.venv/bin/python -m pytest -q tests/test_tts_native_boundary_audit.py
.venv/bin/python scripts/experiments/report_tts_native_boundary_audit.py
```

已有protocol时freeze会拒绝覆盖；score逐cell可恢复。所有源哈希在评分前和独立复核末尾校验。
"""
    (OUT/"proposal.md").write_text(proposal)
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"]
    paths += [ROOT/"scripts/experiments"/f for f in ("tts_native_boundary_audit.py", "check_tts_native_boundary_audit.py", "report_tts_native_boundary_audit.py")]
    paths += [ROOT/"tests/test_tts_native_boundary_audit.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/"artifact_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print("report and figures saved", len(hashes), "hashes")


if __name__ == "__main__":
    main()
