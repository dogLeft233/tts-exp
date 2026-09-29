"""Frozen three-portrait replacement test at bridge alpha .25 and .50."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from . import config as c
from . import frontal_probe as f
from .common import (
    file_sha256,
    read_json,
    read_pcm16,
    verify_self_hashed_json,
    write_json_atomic,
    write_self_hashed_json,
)

DEFAULT_ROOT = c.REPO / "runs/static_image_bridge_lowalpha_20260913"
SWEEP = c.REPO / "runs/natural_video_bridge_sweep_20260913_v2"
BRIDGES = ("B025", "B050")


def prepare(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "protocol.json"
    if target.exists():
        p = verify_self_hashed_json(target)
        for path, sha in p["bindings"].items():
            if file_sha256(path) != sha:
                raise ValueError(f"frozen input changed: {path}")
        return p
    if (root / "videos").exists():
        raise ValueError("cannot freeze after rendering")
    parent_path = f.DEFAULT_ROOT / "protocol.json"
    p = copy.deepcopy(verify_self_hashed_json(parent_path))
    for path, sha in p["bindings"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"parent binding changed: {path}")
    p["bindings"][str(parent_path)] = file_sha256(parent_path)
    p["bindings"][str(Path(__file__).resolve())] = file_sha256(Path(__file__))
    for r in p["records"]:
        sid = r["sample_id"]
        result_path = SWEEP / sid / "result.json"
        old = read_json(result_path)
        p["bindings"][str(result_path)] = file_sha256(result_path)
        natural, _ = read_pcm16(Path(r["audio"]["N"]["path"]))
        sweep_n, _ = read_pcm16(SWEEP / sid / "N.wav")
        if not np.array_equal(natural, sweep_n):
            raise ValueError("natural audio differs between experiments")
        for arm in BRIDGES:
            path = SWEEP / sid / f"{arm}.wav"
            values, meta = read_pcm16(path)
            if len(values) != len(natural) or meta["container_sha256"] != old["audio_hashes"][arm]:
                raise ValueError("bridge audio hash or length mismatch")
            r["audio"][arm] = {"path": str(path), **meta}
            p["bindings"][str(path)] = meta["container_sha256"]
        r["support"] = f.frozen_support(len(natural))
    for im in p["images"]:
        write_json_atomic(root / f"score_box_{im['id']}.json", im["score_box"])
    p.update(experiment="frontal_lowalpha_replacement", alpha=[.25, .5],
             primary="C(V_B025/B050,A_N)-C(V_N,A_N)",
             baseline="new natural renders; RT PCM identity checked by parent",
             scope="exploratory seen-fit 22 audios x 3 fixed portraits; two unadjusted comparisons; no training authorization",
             gain_rule="mean delta C > .05 and CI lower > 0; D and N-lag distance improvements CI lower >= 0; measurement controls pass")
    write_self_hashed_json(target, p)
    return verify_self_hashed_json(target)


def pair_metrics(nn: np.ndarray, bn: np.ndarray, bb: np.ndarray, w: list[int]) -> dict:
    n, b, own = (f.metrics(m, w) for m in (nn, bn, bb))
    return {"N": n, "B": b, "B_own": own, "delta_C": b["C"] - n["C"],
            "D_improvement": n["D"] - b["D"],
            "anchor_improvement": n["D"] - b["curve"][n["lag"] + 15],
            "offset_change": b["lag"] - n["lag"],
            "own_vs_N_diagonal_C": own["C"] - n["C"],
            "own_preference_C": own["C"] - b["C"]}


def aggregate(rows: list[dict]) -> dict:
    keys = ("delta_C", "D_improvement", "anchor_improvement", "own_vs_N_diagonal_C")
    groups = sorted({r["source_group"] for r in rows})
    grouped = []
    for group in groups:
        local = [r for r in rows if r["source_group"] == group]
        if len(local) != 3 or len({r["image_id"] for r in local}) != 3:
            raise ValueError("each source group must have all three portraits")
        grouped.append({"source_group": group, **{k: float(np.mean([r[k] for r in local])) for k in keys}})
    return {"grouped": {k: f.summary([r[k] for r in grouped]) for k in keys},
            "by_image": {im: {k: f.summary([r[k] for r in rows if r["image_id"] == im]) for k in keys}
                         for im in sorted({r["image_id"] for r in rows})},
            "C_improved": sum(r["delta_C"] > 0 for r in rows),
            "joint_improved": sum(r["delta_C"] > 0 and r["D_improvement"] > 0 for r in rows),
            "offset_changed": sum(r["offset_change"] != 0 for r in rows),
            "group_rows": grouped}


def controls(root: Path, p: dict) -> list[dict]:
    r = p["records"][0]
    rows = []
    for im in p["images"]:
        nn, rep, nd, ns, sn, ss = [f.load_matrix(root, im, r, v, a) for v, a in
                                  (("N", "N"), ("N_REPEAT", "N"), ("N", "ND"),
                                   ("N", "S"), ("S", "N"), ("S", "S"))]
        w = r["support"]["primary"]
        error = float(np.max(np.abs(nn[w] - rep[w])))
        n, d = (f.metrics(m, r["support"]["delay"]) for m in (nn, nd))
        damage = d["curve"][n["lag"] + 15] - n["D"]
        local = {}
        lag = f.metrics(nn, w)["lag"]
        for label, by_lag in r["support"]["local_by_lag"].items():
            lw = by_lag[str(lag)]
            if len(lw) < 5:
                raise ValueError("insufficient local swap support")
            col = lag + 15
            local[label] = {"N_prefers_N": float((ns[lw, col] - nn[lw, col]).mean()),
                            "S_prefers_S": float((sn[lw, col] - ss[lw, col]).mean())}
        rows.append({"image_id": im["id"], "repeat_error": error,
                     "delay_lag_change": d["lag"] - n["lag"], "delay_anchor_damage": damage,
                     "measurement_pass": error < 1e-5 and d["lag"] - n["lag"] == 5 and damage > .1,
                     "local_swap": local, "swap_pass": all(v > .1 for x in local.values() for v in x.values())})
    return rows


def analyze(root: Path, p: dict) -> dict:
    rows = []
    for im in p["images"]:
        for r in p["records"]:
            nn = f.load_matrix(root, im, r, "N", "N")
            for arm in BRIDGES:
                bn = f.load_matrix(root, im, r, arm, "N")
                bb = f.load_matrix(root, im, r, arm, arm)
                rows.append({"image_id": im["id"], "sample_id": r["sample_id"], "source_group": r["source_group"],
                             "arm": arm, **pair_metrics(nn, bn, bb, r["support"]["primary"])})
    checks = controls(root, p)
    summaries = {arm: aggregate([r for r in rows if r["arm"] == arm]) for arm in BRIDGES}
    for arm, result in summaries.items():
        s = result["grouped"]
        result["gain_established"] = bool(all(r["measurement_pass"] for r in checks)
            and s["delta_C"]["mean"] > .05 and s["delta_C"]["ci95"][0] > 0
            and s["D_improvement"]["ci95"][0] >= 0 and s["anchor_improvement"]["ci95"][0] >= 0)
    result = {"status": "complete", "pairs": rows, "summaries": summaries, "controls": checks,
              "protocol_sha256": file_sha256(root / "protocol.json"), "training_authorized": False}
    write_json_atomic(root / "analysis.json", result)
    return result


def report(root: Path, p: dict, result: dict) -> None:
    lines = ["# 优质静态图 Bridge 0.25 / 0.50 replacement 实验", "", "## 计算流程", "",
             "固定本地3、6、9号512×512近正脸人像，与全部22条seen-fit音频交叉。没有按新分数筛图或筛音频。",
             "N=自然音频；B025/B050=上一轮自然视频扫描的同一波形（hash核验），自然相位+向MFA-linear的log幅度插值，alpha分别0.25/0.50；不是时间插值比例。",
             "同图、同生成ROI、同评分crop，以Wav2Lip GAN分别生成N/B025/B050视频。主比较均配原自然音轨：ΔC=C(V_B,A_N)−C(V_N,A_N)。正数表示replacement增强。",
             "共同窗口在评分前按完整MFCC/preemphasis和生成mel-STFT感受野冻结；C=±15帧距离曲线中位数−最小值。D改善=自然基线最小距离−bridge最小距离；anchor改善固定N最佳lag。均以正数为好。",
             "同音频三张图先平均差值，对22个独立source_group配对bootstrap10000次，seed=20260913。每强度66对不是66个独立音频样本。两强度CI未作多重比较校正。",
             "RT与N逐采样相同，不单独重渲染。桥接自身音轨评分仅辅助，不替代自然音轨replacement指标。", "", "## 主结果", "",
             "|强度|平均ΔSync-C（95% CI）|C改善对数|C/D联合改善|D改善|固定N lag距离改善|", "|---|---:|---:|---:|---:|---:|"]
    for arm, s in result["summaries"].items():
        a = s["grouped"]["delta_C"]
        lines.append(f"|{arm}|{a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]|{s['C_improved']}/66|{s['joint_improved']}/66|{s['grouped']['D_improvement']['mean']:+.3f}|{s['grouped']['anchor_improvement']['mean']:+.3f}|")
    lines += ["", "## 分参考图", "", "|图像|强度|平均ΔC|95% CI|C改善|", "|---|---|---:|---:|---:|"]
    for arm, s in result["summaries"].items():
        for im, stats in s["by_image"].items():
            a = stats["delta_C"]
            lines.append(f"|{im}|{arm}|{a['mean']:+.3f}|[{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]|{a['positive']}/22|")
    lines += ["", "## 辅助结果与结论", ""]
    for arm, s in result["summaries"].items():
        a = s["grouped"]["own_vs_N_diagonal_C"]
        lines.append(f"- {arm}：增强判据通过={s['gain_established']}；offset改变{s['offset_changed']}/66；各配自身音轨的对角ΔC={a['mean']:+.3f}（更换评分音轨，不是主replacement效应）。")
    lines += ["", "## 小规模控制", "", "仅第一条固定音频×3图，不是全量控制。"]
    for r in result["controls"]:
        lines.append(f"- 图{r['image_id']}：repeat矩阵误差{r['repeat_error']:.8f}；延迟200ms lag变化{r['delay_lag_change']}帧，固定lag损伤{r['delay_anchor_damage']:.3f}；measurement={r['measurement_pass']}，LOCAL_SWAP双向偏好={r['swap_pass']}。")
    lines += ["", "## 边界与视频", "", "探索性、已使用队列、3个固定身份；不授权训练或泛化。自然视频上较少掉分不能保证生成视频replacement增强，也不能单凭SyncNet确认局部音素时间完全对齐。",
              "所有生成视频在videos/<图号>/<样本>/<臂>.mkv（无音轨、无损评分资产）；playback为首条冻结音频在三图的N/B025/B050左右对照，共同播放N音频。未按结果挑选。", ""]
    for im in p["images"]:
        lines.append(f"- [图{im['id']}视频对照](playback/{im['id']}.mp4)")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def playback(root: Path, p: dict) -> None:
    r = p["records"][0]
    for im in p["images"]:
        target = root / "playback" / f"{im['id']}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        base = root / "videos" / im["id"] / r["sample_id"]
        cmd = [str(c.FFMPEG), "-y", "-v", "error"]
        for arm in ("N", *BRIDGES):
            cmd += ["-i", str(base / f"{arm}.mkv")]
        filters = ";".join(f"[{i}:v]drawtext=text='{arm}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black[v{i}]" for i, arm in enumerate(("N", *BRIDGES)))
        filters += ";[v0][v1][v2]hstack=inputs=3[v]"
        cmd += ["-i", r["audio"]["N"]["path"], "-filter_complex", filters, "-map", "[v]", "-map", "3:a",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-threads", "2", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(target)]
        f.job(root, cmd, target.with_suffix(".log"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze", "playback"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.run_root.resolve()
    p = prepare(root)
    if args.command == "prepare":
        print("FROZEN 3 images x 22 audio x 3 primary arms", flush=True)
        return
    if args.command == "run":
        for i, r in enumerate(p["records"]):
            for im in p["images"]:
                arms = ("N", "N_REPEAT", "S", *BRIDGES) if i == 0 else ("N", *BRIDGES)
                for arm in arms:
                    video = f.render(root, im, r, arm)
                    audios = (["N", "ND", "S"] if arm == "N" and i == 0 else ["N"] if arm in ("N", "N_REPEAT") else ["N", arm])
                    f.score(root, im, r, arm, video, audios)
                    print(f"DONE audio={i+1}/22 image={im['id']} arm={arm}", flush=True)
        result = analyze(root, p)
        report(root, p, result)
        playback(root, p)
    elif args.command == "analyze":
        report(root, p, analyze(root, p))
    else:
        playback(root, p)


if __name__ == "__main__":
    main()
