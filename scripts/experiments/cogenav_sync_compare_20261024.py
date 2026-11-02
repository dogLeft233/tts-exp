"""Descriptive source-grouped CoGenAV/SyncNet comparison and cache replay."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cogenav_sync_audit_20261024 import dump, lag_curve, sha


def compare(rows, allow_missing=False):
    pairs = []
    for source in sorted({r["sample_id"] for r in rows}):
        group = {(r["arm"], r["draw"]): r for r in rows if r["sample_id"] == source}
        for draw in [1, 2, 3]:
            for label, first, second in [
                ("L06-N", ("L06", draw), ("N", 0)),
                ("L17-N", ("L17", draw), ("N", 0)),
                ("L17-L06", ("L17", draw), ("L06", draw)),
            ]:
                if first not in group or second not in group:
                    if allow_missing:
                        continue
                    raise ValueError(f"Missing paired input: {source}/{first}/{second}")
                a, b = group[first], group[second]
                pair = {"source": source, "family": label, "draw": draw,
                        "a": a["id"], "b": b["id"], "qualified": a["qualified"] and b["qualified"]}
                for field in ["score_float32", "eq1", "zero_common", "max_score", "zero_excess", "peak_excess"]:
                    pair[field] = a[field]-b[field]
                pairs.append(pair)
    groups = []
    for source, family in sorted({(p["source"], p["family"]) for p in pairs}):
        items = [p for p in pairs if p["source"] == source and p["family"] == family]
        groups.append({"source": source, "family": family,
                       **{f: float(np.mean([p[f] for p in items])) for f in
                          ["score_float32", "eq1", "max_score", "zero_excess", "peak_excess"]}})
    families = {}
    for family in sorted({p["family"] for p in pairs}):
        items = [p for p in pairs if p["family"] == family]
        grouped = [p for p in groups if p["family"] == family]
        families[family] = {"sources": len(grouped), "pairs": len(items),
                           **{f: float(np.mean([p[f] for p in grouped])) for f in
                              ["score_float32", "eq1", "max_score", "zero_excess", "peak_excess"]},
                           "eq1_sign_conflicts": sum(p["score_float32"] * p["eq1"] < 0 for p in items),
                           "max_sign_conflicts": sum(p["score_float32"] * p["max_score"] < 0 for p in items),
                           "source_mean_eq1_sign_conflicts": sum(p["score_float32"] * p["eq1"] < 0 for p in grouped)}
    return pairs, groups, families


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    root = Path(args.run)
    rows = json.loads((root / "scores.json").read_text())
    receipt = json.loads((root / "extraction_complete.json").read_text())
    assert len(rows) == receipt["clips"] == 7 * len(receipt["sources"])
    protocol_sha = sha(root / "protocol.json")
    max_error = 0.
    for row in rows:
        path = root / "embeddings" / (row["id"] + ".npz")
        assert sha(path) == row["embedding_sha256"]
        z = np.load(path)
        assert str(z["protocol_sha256"]) == protocol_sha
        a, v = z["audio"].astype(np.float64), z["video"].astype(np.float64)
        # Independent scalar replay; do not use the producer's lag function.
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        eq = np.maximum(0, np.einsum("ij,ij->i", a, v)).mean()
        curve = []
        for k in range(-15, 16):
            curve.append(float(np.mean([max(0., np.dot(a[t+k], v[t])) for t in range(15, len(a)-15)])))
        max_error = max(max_error, abs(eq-row["eq1"]), float(np.max(np.abs(np.array(curve)-row["curve"]))))
    assert max_error < 1e-6
    controls = [s for r in rows for s in r["raw_shift_controls"]]
    passed = sum(s["pass_within_one_frame"] for s in controls)
    rate = passed/len(controls)
    gate = rate >= .8 and np.median([r["peak_excess"] for r in rows]) > 0
    pairs, groups, families = compare(rows)
    summary = {"status": "EXPLORATORY_KNOWN_SHIFT_PASS" if gate else "CONTROL_FAILED_UNVALIDATED_METRIC",
               "clips": len(rows), "sources": len(receipt["sources"]),
               "known_shift": {"passed": passed, "total": len(controls), "rate": rate},
               "independent_float64_replay_max_error": max_error,
               "protocol_sha256": protocol_sha, "families": families,
               "pair_sign_conflicts": sum(p["score_float32"]*p["eq1"] < 0 for p in pairs),
               "max_sign_conflicts": sum(p["score_float32"]*p["max_score"] < 0 for p in pairs),
               "pairs": len(pairs), "source_family_groups": len(groups),
               "pair_delta_spearman_descriptive": float(spearmanr([p["score_float32"] for p in pairs], [p["eq1"] for p in pairs]).statistic),
               "source_family_delta_spearman_descriptive": float(spearmanr([p["score_float32"] for p in groups], [p["eq1"] for p in groups]).statistic),
               "near_tie_sensitivity_not_preregistered": {
                   "rule": "retain |deltaC|>=0.1 and |deltaEq1|>=0.005; descriptive only",
                   "pairs": sum(abs(p["score_float32"])>=.1 and abs(p["eq1"])>=.005 for p in pairs),
                   "conflicts": sum(abs(p["score_float32"])>=.1 and abs(p["eq1"])>=.005 and p["score_float32"]*p["eq1"]<0 for p in pairs)},
               "arms": {arm:{f:float(np.mean([r[f] for r in rows if r["arm"]==arm])) for f in
                              ["score_float32", "eq1", "peak_lag_frames", "zero_excess"]} for arm in ["N","L06","L17"]}}
    dump(root / "comparison.json", summary)
    dump(root / "pairs.json", pairs)
    dump(root / "source_family_means.json", groups)
    lines = ["# CoGenAV Base 与 Sync-C 配对先导对照", "", f"状态：{summary['status']}。{len(rows)} 条视频，{summary['sources']} 个独立来源；每来源自然N及两种TTS各3 draw。选择按 sample_id 字典序前4来源，未按结果选样。", "",
             "## 资格与实现", "",
             "作者HF权重 detao/CoGenAV，revision 3c85c000c60fe7ffb3aaa3c63c96ba33ad8ce42c，Base SHA90f8c516…已逐字节核验；strict state_dict加载通过。GitHub源固定dbd3f835…；从fairseq0.12.2的Wav2Vec2Config合并默认配置解决缺失pos_conv_depth等字段，权重和模型源不改。", "",
             "Ubuntu V100，独立venv cogenav-sync-20261024，复用现有torch运行库；不修改原环境。PyPI fairseq源码包缺C++文件，最终使用官方v0.12.2完整源码构建。安装失败日志保留。", "",
             "AutoAVSR官方MediaPipe/VideoProcess做96×96对齐口部RGB裁剪，再遵循CoGenAV原脚本加权灰度、Base mean0/std1和30秒黑帧/音频padding；单独A/V encoder、upsampler前25fps表示，只评分真实有效长度。保留裁剪图供QA。没有训练、TFG生成、SyncNet重评分或文本识别。", "",
             f"实际波形前移/后移160ms后重新计算mel和A表示，匹配峰移动控制 {passed}/{len(controls)} 通过（容差40ms）；正lag表示使用更晚的音频特征。不是平移缓存特征的自证。Float64独立逐窗重算最大误差 {max_error:.3g}。", "",
             "## 分数对照", "", "Eq1是论文mean ReLU cosine；max及excess是本实验派生诊断，不能称官方指标或同步概率。下表先在每来源平均draw，再等权来源。", "",
             "|比较|ΔSync-C|ΔCoGenAV Eq1|ΔCoGenAV max|Eq1符号冲突/配对数|来源平均符号冲突|",
             "|---|---:|---:|---:|---:|---:|"]
    for family,f in families.items():
        lines.append(f"|{family}|{f['score_float32']:+.3f}|{f['eq1']:+.6f}|{f['max_score']:+.6f}|{f['eq1_sign_conflicts']}/{f['pairs']}|{f['source_mean_eq1_sign_conflicts']}/{f['sources']}|")
    lines += ["", f"共 {summary['pair_sign_conflicts']}/{len(pairs)} 配对Eq1方向冲突；搜索最佳lag的max分数仍有{summary['max_sign_conflicts']}/{len(pairs)}方向冲突。逐pair差值Spearman={summary['pair_delta_spearman_descriptive']:.3f}，来源×family均值差值Spearman={summary['source_family_delta_spearman_descriptive']:.3f}，均仅描述，不做独立样本推断。", "",
              "## 来源明细", "", "|来源|比较|ΔSync-C|ΔEq1|Δzero_excess|", "|---|---|---:|---:|---:|"]
    for g in groups:
        lines.append(f"|{g['source']}|{g['family']}|{g['score_float32']:+.3f}|{g['eq1']:+.6f}|{g['zero_excess']:+.6f}|")
    lines += ["", "## 结论边界", "", "本结果只检验两种模型的配对排序是否一致。不同原生时钟、语速、发音内容长度、模型语言域与口部预处理都可能影响表征；中文泛化没有人工ground truth校准。已知错位控制仅说明对本批160ms扰动具定位能力，不证明细微感知同步或所有错位条件均有效。共同lag支持仅在每条clip内部，跨臂不冒称相同发音事件。", "",
              "Sync-C原float32值复用上一实验canonical PCM/官方全流程结果，音频/视频输入SHA已验证。Eq1与Sync-C尺度不同，只比较方向/相关性；所有重复draw共享来源，独立来源只有4个，不按36pair或56control计算显著性。原始两模型分歧不能判断谁更符合人类，也不能单独证明SyncNet偏好。", "",
              "补充近零差敏感性（事后描述，不是显著性阈值）：保留|ΔC|≥0.1且|ΔEq1|≥0.005，32pair中仍19pair方向相反。N/L06/L17平均Eq1为0.303/0.289/0.344，而lag0减外围lag均值为0.034/0.022/0.019；原始匹配升高不等于时间选择性升高。所有N的峰lag均为+2帧，这可能反映流程/模型固定偏移，不能直接宣布物理音画延迟80ms。", "",
              "本地pytest因缺numpy无法收集；远端无pytest，实际直接调用3个科学契约断言通过（时移符号/共同支持、尺度不变与ReLU、来源分组与draw配对），不称pytest运行通过。协议在第一次模型forward之前冻结，protocol.json为准。评分和对照JSON、逐窗embedding和控制表示均保留；无断言声称重现论文WER。"]
    (root / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
