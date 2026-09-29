"""User-requested first-11 finalization without modifying the frozen 22-record protocol."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import lowalpha_probe as l
from .common import file_sha256, read_json, write_json_atomic

ROOT = l.c.REPO / "runs/static_image_bridge_lowalpha_20260913_v2"


def summarize(rows: list[dict], mode: str) -> dict:
    target = "B" if mode == "replacement" else "B_own"
    groups = sorted({r["source_group"] for r in rows})
    values = {"C": [], "D": [], "anchor": []}
    pairs = []
    for r in rows:
        n, b = r["N"], r[target]
        pairs.append({"sample_id": r["sample_id"], "source_group": r["source_group"], "image_id": r["image_id"],
                      "C": b["C"] - n["C"], "D": n["D"] - b["D"],
                      "anchor": n["D"] - b["curve"][n["lag"] + 15]})
    for g in groups:
        local = [r for r in pairs if r["source_group"] == g]
        assert len(local) == 3
        for k, series in values.items():
            series.append(float(np.mean([r[k] for r in local])))
    return {"C": float(np.mean([r[target]["C"] for r in rows])),
            "D": float(np.mean([r[target]["D"] for r in rows])),
            "delta_C": l.f.summary(values["C"]), "D_improvement": l.f.summary(values["D"]),
            "anchor_improvement": l.f.summary(values["anchor"]),
            "C_positive_pairs": sum(r["C"] > 0 for r in pairs),
            "joint_positive_pairs": sum(r["C"] > 0 and r["D"] > 0 for r in pairs),
            "offset_changed": sum(r[target]["lag"] != r["N"]["lag"] for r in rows),
            "by_image": {im: l.f.summary([r["C"] for r in pairs if r["image_id"] == im]) for im in ("3", "6", "9")}}


def main() -> None:
    p = l.prepare(ROOT)
    original_count = len(p["records"])
    p["records"] = p["records"][:11]
    scope = {"original_count": original_count, "analyzed_count": 11,
             "sample_ids": [r["sample_id"] for r in p["records"]],
             "reason": "User requested stopping at the first 11 during execution; early six-record results had been disclosed. Not a fixed-N confirmatory test.",
             "excluded_artifact": "videos/3/lrs3_6VnKV1sr5VQ_00008/N.ffv1.mkv: record12 unscored render raced with stop; preserved, excluded",
             "protocol_sha256": file_sha256(ROOT / "protocol.json"),
             "analysis_code_sha256": file_sha256(Path(__file__))}
    if (ROOT / "analysis_scope.json").exists():
        assert read_json(ROOT / "analysis_scope.json")["sample_ids"] == scope["sample_ids"]
    write_json_atomic(ROOT / "analysis_scope.json", scope)
    result = l.analyze(ROOT, p)
    result["analysis_scope"] = scope
    result["comparisons"] = {arm: {mode: summarize([r for r in result["pairs"] if r["arm"] == arm], mode)
                                  for mode in ("replacement", "native")} for arm in l.BRIDGES}
    baseline = [r["N"] for r in result["pairs"] if r["arm"] == "B025"]
    result["baseline"] = {k: float(np.mean([r[k] for r in baseline])) for k in ("C", "D")}
    write_json_atomic(ROOT / "analysis.json", result)
    with (ROOT / "per_pair.csv").open("w", newline="") as handle:
        keys = ["sample_id", "image_id", "arm", "replacement_delta_C", "native_delta_C", "replacement_D_improvement", "native_D_improvement"]
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        for r in result["pairs"]:
            writer.writerow({"sample_id": r["sample_id"], "image_id": r["image_id"], "arm": r["arm"],
                             "replacement_delta_C": r["delta_C"], "native_delta_C": r["own_vs_N_diagonal_C"],
                             "replacement_D_improvement": r["D_improvement"], "native_D_improvement": r["N"]["D"] - r["B_own"]["D"]})
    lines = ["# 静态正脸 Bridge 0.25 / 0.50：原生与 replacement 对照", "",
             "## 实验与计算流程", "",
             "使用本地 data/data/image/{3,6,9}.png 三张512×512近正脸图。固定原队列前11条音频，全部与三图交叉，每强度33对、11个独立音频source_group。",
             "原计划22条，运行到11条时用户要求收束；此前已披露前6条部分结果。这是探索性、运行中缩减的样本集，不是预先固定N的确认性试验。第12条在停止瞬间留下的未评分自然视频不纳入分析。",
             "B025/B050直接复用上一轮强度扫描的波形并核验hash：保留natural相位，向MFA-linear的log幅度谱插值0.25/0.50，并按原契约RMS缩放。alpha不是时间偏移或对齐百分比。",
             "每张图固定生成ROI、评分crop和25fps，以同一Wav2Lip GAN分别生成N、B025、B050视频。所有评分前冻结完整MFCC/preemphasis与生成mel/STFT真实支持，±15帧lag均有效。没有按新分数筛选图像或个别音频。",
             "Sync-C=距离曲线中位数−最小值（高为好），Sync-D=最小值（低为好）。曲线先在固定共同窗口上平均。C有offset搜索，不能用高C单独证明零偏移。与历史含边界padding的全局C不直接比较。",
             "同一音频在三张图上的差值先平均，再对11个音频组配对bootstrap10000次，seed=20260913；95% CI未做多重比较或运行中停止校正。", "",
             "|比较口径|Bridge视频评分音轨|基线|回答的问题|", "|---|---|---|---|",
             "|原生，不做replacement|生成时使用的同一bridge音频|N驱动视频配N音轨|bridge音视频组合本身是否更同步？|",
             "|replacement|将音轨换回原自然音频N|同一N基线|bridge是否生成更适合原自然音频的口型？|", "",
             "## 结果", "", "|组合|平均Sync-C|相对基线ΔC（95% CI）|C改善|平均Sync-D|D改善量|", "|---|---:|---:|---:|---:|---:|",
             f"|自然基线 V_N/A_N|{result['baseline']['C']:.3f}|—|—|{result['baseline']['D']:.3f}|—|"]
    for mode, label in (("native", "原生"), ("replacement", "换回N")):
        for arm in l.BRIDGES:
            s = result["comparisons"][arm][mode]
            a = s["delta_C"]
            lines.append(f"|{arm} {label}|{s['C']:.3f}|{a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]|{s['C_positive_pairs']}/33|{s['D']:.3f}|{s['D_improvement']['mean']:+.3f}|")
    lines += ["", "D改善量=基线D−候选D，正数为好；原生比较同时改变视频和评分音轨，不能当作replacement收益。", "",
              "## 分图与稳定性", "", "|强度/口径|图3 ΔC|图6 ΔC|图9 ΔC|C/D联合改善|offset变化|", "|---|---:|---:|---:|---:|---:|"]
    for arm in l.BRIDGES:
        for mode in ("native", "replacement"):
            s = result["comparisons"][arm][mode]
            means = "|".join(f"{s['by_image'][im]['mean']:+.3f}" for im in ("3", "6", "9"))
            lines.append(f"|{arm}/{mode}|{means}|{s['joint_positive_pairs']}/33|{s['offset_changed']}/33|")
    lines += ["", "## 小规模控制", "", "首条固定音频×三图，非11条全量控制。"]
    for r in result["controls"]:
        lines.append(f"- 图{r['image_id']}：repeat矩阵误差{r['repeat_error']:.8f}；延迟200ms的lag变化{r['delay_lag_change']}帧、固定自然lag损伤{r['delay_anchor_damage']:.3f}；测量控制={r['measurement_pass']}，LOCAL_SWAP双向偏好={r['swap_pass']}。")
    lines += ["", "## 结论边界", "",
              "本11条队列中，两种强度在原生和replacement口径下的平均ΔC均为负，四个未校正bootstrap区间上界均小于零：没有观察到相对自然基线的增强。原生分数比换回N更高，但仍未超过基线。",
              "以主表配对差值及区间判断：正向但CI跨零只能写未建立增益；负向且CI上界小于零支持此样本集上的下降。三张图不是跨身份泛化验证，不能授权训练，也不证明历史动态实验存在bug。",
              "原生高于replacement表示生成结果更匹配其驱动音轨，不自动意味着高于自然基线。自然视频上较对齐也不能保证静态生成的replacement增强。", "",
              "## 产物与观看", "",
              "- analysis.json：完整分数、区间、控制、全部66个强度配对结果。",
              "- per_pair.csv：逐音频×图像×强度的原生/replacement差值。",
              "- protocol.json保留最初22条计划；analysis_scope.json记录用户缩减至前11条的原因与名单。",
              "- independent_verification.json：独立矩阵重算与首条音频×三图15个cell官方前向抽查（生成后读取）。",
              "- videos/：无音轨无损评分资产；playback/：下面的可播放示例。", "",
              "下列对照固定为第一条音频，未按得分挑选。三联屏从左至右N/B025/B050，统一播放N，观察replacement；原生单视频播放各自bridge音轨。", ""]
    for im in p["images"]:
        lines.append(f"- 图{im['id']}：[replacement三联屏](playback/{im['id']}.mp4) · [B025原生](playback/{im['id']}_B025_native.mp4) · [B050原生](playback/{im['id']}_B050_native.mp4)")
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print({"baseline": result["baseline"], "comparisons": result["comparisons"]}, flush=True)
    l.playback(ROOT, p)
    r = p["records"][0]
    for im in p["images"]:
        for arm in l.BRIDGES:
            target = ROOT / "playback" / f"{im['id']}_{arm}_native.mp4"
            video = ROOT / "videos" / im["id"] / r["sample_id"] / f"{arm}.mkv"
            l.f.job(ROOT, [str(l.c.FFMPEG), "-y", "-v", "error", "-i", str(video), "-i", r["audio"][arm]["path"],
                           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                           "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(target)], target.with_suffix(".log"))


if __name__ == "__main__":
    main()
