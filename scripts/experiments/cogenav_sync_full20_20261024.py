"""Source-clustered descriptive expansion of the frozen CoGenAV pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from cogenav_sync_audit_20261024 import dump, sha
from cogenav_sync_compare_20261024 import compare

FIELDS = ["eq1", "zero_common", "max_score", "zero_excess", "peak_excess"]


def common_qualified_pairs(pairs, planned):
    """The prior protocol removes a failed draw from BOTH TTS arms."""
    bad = {(r['sample_id'],r['draw']) for r in planned if r['arm']!='N' and not r['qualified']}
    return [p for p in pairs if p['qualified'] and (p['source'],p['draw']) not in bad]


def interval(x):
    """One row per source. Marginal descriptive source bootstrap, not a test."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    rng = np.random.default_rng(20261024)
    means = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "sources": len(x)}


def summarize(pairs):
    sources = sorted({p["source"] for p in pairs})
    families = sorted({p["family"] for p in pairs})
    out = {"sources": len(sources), "pairs": len(pairs), "fields": {}, "families": {}}
    for field in FIELDS:
        rates = []
        for source in sources:
            # Equal family weight within each source, including qualification sensitivity.
            rates.append(np.mean([np.mean([p["score_float32"]*p[field] > 0 for p in pairs
                                             if p["source"]==source and p["family"]==family]) for family in families]))
        sc, co = [], []
        for source in sources:
            for family in families:
                ps = [p for p in pairs if p["source"]==source and p["family"]==family]
                sc.append(np.mean([p["score_float32"] for p in ps]))
                co.append(np.mean([p[field] for p in ps]))
        conflicts = sum(p["score_float32"]*p[field] < 0 for p in pairs)
        retained = [p for p in pairs if abs(p["score_float32"])>=.1 and abs(p[field])>=.005]
        out["fields"][field] = {
            "conflicts": conflicts, "agreements": sum(p["score_float32"]*p[field]>0 for p in pairs),
            "source_balanced_agreement": interval(rates),
            "source_family_spearman_descriptive": float(spearmanr(sc, co).statistic),
            "source_family_conflicts": int(np.sum(np.array(sc)*np.array(co) < 0)),
            "source_family_groups": len(sc),
            "near_tie_sensitivity": {"pairs": len(retained), "conflicts":sum(p["score_float32"]*p[field]<0 for p in retained)},
        }
    for family in families:
        ps = [p for p in pairs if p["family"]==family]
        family_out = {"pairs": len(ps), "sources": len(sources)}
        for field in ["score_float32"]+FIELDS:
            source_values = [np.mean([p[field] for p in ps if p["source"]==source]) for source in sources]
            family_out[field] = interval(source_values)
        out["families"][family] = family_out
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.run).resolve()
    analysis_protocol = {
        "version": 1, "design": "exploratory expansion, no confirmatory unseen-cohort claim",
        "planned_clips": 140, "planned_sources": 20, "pilot_sources": 4, "additional_sources": 16,
        "endpoints": FIELDS, "comparisons": ["L06-N", "L17-N", "L17-L06"],
        "views": ["full20", "additional16", "common_qualified_full20"],
        "statistics": "source-balanced family/draw means; source bootstrap10000 seed20261024 marginal descriptive95CI; no p values",
        "near_tie": "pilot-chosen |deltaC|>=0.1 and |deltaCo|>=0.005; sensitivity, not a significance threshold",
        "control": "raw waveform +/-160ms peak movement within40ms; >=80% overall; per-view rates also reported",
        "search": "zero_common vs max_score use exactly same video centers; Eq1 true-length is separate",
        "forbidden_claims": ["human truth", "pure evaluator preference", "official CoGenAV benchmark reproduction"],
    }
    path = root / "analysis_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == analysis_protocol
    else:
        dump(path, analysis_protocol)
    if args.freeze_only:
        print("Frozen", sha(path))
        return
    complete = json.loads((root / "extraction_complete.json").read_text())
    rows = json.loads((root / "scores.json").read_text())
    unsupported = json.loads((root/"unmeasurable.json").read_text())
    assert complete["planned_clips"] == len(rows)+len(unsupported) == 140
    assert {x["id"] for x in unsupported} == {"BAC009S0011W0121_L06_3","BAC009S0016W0121_L06_3"}
    assert all(x["reason"] == "EXCEEDS_NATIVE_30SEC_INPUT" and not x["prior_qualified"] for x in unsupported)
    assert complete["clips"] == len(rows) == 138 and len(complete["sources"]) == 20
    assert len({r["id"] for r in rows}) == 138
    assert {arm:sum(r["arm"]==arm for r in rows) for arm in ["N","L06","L17"]} == {"N":20,"L06":58,"L17":60}
    protocol = json.loads((root / "protocol.json").read_text())
    allowed = {sha(root / "protocol.json"), protocol["reuse"]["protocol_sha256"]}
    max_error = 0.
    controls = []
    for row in rows:
        f = root / "embeddings" / (row["id"]+".npz")
        assert sha(f) == row["embedding_sha256"]
        data = np.load(f)
        assert str(data["protocol_sha256"]) in allowed
        a, v = data["audio"].astype(float), data["video"].astype(float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # Scalar implementation independent of the producer's lag_curve function.
        def curve(aa):
            return np.array([np.mean([max(0.,float(np.dot(aa[t+k],v[t])))
                                     for t in range(15,len(v)-15)]) for k in range(-15,16)])
        eq = np.mean([max(0.,float(np.dot(aa,vv))) for aa,vv in zip(a,v)])
        replay = curve(a)
        max_error = max(max_error, abs(eq-row["eq1"]), float(np.max(np.abs(replay-row["curve"]))))
        shifted = np.load(root / "embeddings" / (row["id"]+"_rawshifts.npz"))
        for control in row["raw_shift_controls"]:
            aa = shifted[str(control["shift_frames"])].astype(float)
            aa /= np.linalg.norm(aa, axis=1, keepdims=True)
            sc = curve(aa)
            max_error = max(max_error,float(np.max(np.abs(sc-control["curve"]))))
            peak = int(sc.argmax()-15)
            err = peak-int(replay.argmax()-15)-control["shift_frames"]
            assert peak == control["peak_lag_frames"] and err == control["peak_movement_error_frames"]
            assert bool(abs(err)<=1) == control["pass_within_one_frame"]
            controls.append(dict(source=row["sample_id"], **control))
    assert max_error < 1e-6
    pilot_root = Path(protocol["reuse"]["root"])
    pilot_ids = {r["sample_id"] for r in json.loads((pilot_root/"selected_inputs.json").read_text())}
    pairs, groups, _ = compare(rows, allow_missing=True)
    planned = json.loads((root/'selected_inputs.json').read_text())
    views = {"full20": pairs, "additional16": [p for p in pairs if p["source"] not in pilot_ids],
             "common_qualified_full20": common_qualified_pairs(pairs, planned)}
    assert len(views['common_qualified_full20']) == 174
    summary = {name:summarize(ps) for name,ps in views.items()}
    passed = sum(c["pass_within_one_frame"] for c in controls)
    summary["validation"] = {
        "controls_passed": passed, "controls_total": len(controls), "controls_rate": passed/len(controls),
        "independent_scalar_float64_max_error": max_error,
        "pilot_cache_import_files": len(json.loads((root/"cache_import.json").read_text())["files"]),
        "new_forwards_last_attempt": complete["new_forwards"],
        "unique_new_forwards_total": (len(rows)-28)*4,
        "planned_clips": 140, "scored_clips": len(rows), "unmeasurable": unsupported,
        "peak_histogram": {str(k):sum(r["peak_lag_frames"]==k for r in rows) for k in sorted({r["peak_lag_frames"] for r in rows})},
        "analysis_protocol_sha256": sha(path),
        "status": "EXPLORATORY_KNOWN_SHIFT_PASS" if passed/len(controls)>=.8 else "CONTROL_FAILED_UNVALIDATED_METRIC",
    }
    summary["arm_means"] = {arm:{f:float(np.mean([r[f] for r in rows if r["arm"]==arm]))
                                  for f in ["score_float32"]+FIELDS} for arm in ["N","L06","L17"]}
    dump(root/"comparison_full.json",summary)
    dump(root/"pairs.json",pairs)
    dump(root/"source_family_means.json",groups)
    lines = ["# CoGenAV 与 Sync-C：20来源扩展对照", "", "官方Base模型与论文Eq1实现，外加派生时移诊断；非官方同步benchmark或感知真值。沿用先导模型、权重、配置、预处理和PCM。旧4来源28条缓存按SHA复用；全20来源计划140条，实际138条成功，2条0.6B d3各40.88秒超过当前30秒输入设置，明确UNMEASURABLE、未截断或改模型上下文，原先qualified均为false。新增16来源实际110条评分。所有计划输入和未测原因保留。", "",
             f"状态：{summary['validation']['status']}。真实波形±160ms控制{passed}/{len(controls)}通过；独立float64逐点重放最大误差{max_error:.3g}。", ""]
    for name,view in summary.items():
        if name in {"validation","arm_means"}:
            continue
        lines += [f"## {name}", "", f"{view['sources']}独立语音来源，{view['pairs']}pair；重复draw与N复用不增独立样本。", "",
                  "|CoGenAV端点|同向|反向|来源等权同向率及描述性95%区间|来源×family均值Spearman|",
                  "|---|---:|---:|---|---:|"]
        for field,stats in view["fields"].items():
            agreement=stats["source_balanced_agreement"]
            lines.append(f"|{field}|{stats['agreements']}|{stats['conflicts']}|{agreement['mean']:.1%} [{agreement['ci95'][0]:.1%},{agreement['ci95'][1]:.1%}]|{stats['source_family_spearman_descriptive']:.3f}|")
        lines += ["", "|比较|ΔSync-C|ΔEq1|Δzero_common|Δmax|", "|---|---:|---:|---:|---:|"]
        for name2,fs in view["families"].items():
            lines.append(f"|{name2}|{fs['score_float32']['mean']:+.3f}|{fs['eq1']['mean']:+.6f}|{fs['zero_common']['mean']:+.6f}|{fs['max_score']['mean']:+.6f}|")
    lines += ["", "## 解释与限制", "", "Eq1按真实全长；zero_common/max使用同一clip内部共同视频中心，搜索±600ms。因此搜索效应只能用zero_common与max比较。max只取最佳相似度，不等于Sync-C的时移参照对比。zero/peak_excess减外围lag均值，属于本项目派生诊断，单列避免把定义差异混成结论。", "",
              "来源bootstrap同时保留每来源所有family/draw，区间是边际描述性，不作多终点确认性检验。20来源都来自已分析的既有队列，新增16仅表示CoGenAV此前未评分，不能称全新盲法确认队列。固定头像、中文域未人类校准、原生发音时钟与支持跨臂不同；不能从两模型分歧单独判定SyncNet偏好或谁更符合感知。", "",
              "完整输入/协议/模型与代码版本见protocol.json、analysis_protocol.json、eligibility_protocol.json、model_load.json与source快照；不重新生成视频或重算SyncNet。首次执行在第一条超30秒输入前停止，已算缓存保留；冻结可测资格扩展后原协议恢复，未重算成功缓存。资格敏感性只使用前次冻结qualified标签，不按CoGenAV结果删样。所有均值/CI与近零敏感性详细数值在comparison_full.json。"]
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"validation":summary["validation"],**{name:{"sources":v["sources"],"pairs":v["pairs"],"fields":v["fields"]} for name,v in summary.items() if name in views}},indent=2))


if __name__ == "__main__":
    main()
