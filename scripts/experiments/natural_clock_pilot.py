"""Three-source, one-portrait exploratory dual-endpoint screening."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.audio import (
    write_pcm16,
)
from scripts.experiments.natural_clock_pilot_audio import FP_ROOT, MODEL_ROOT, equalize
from scripts.experiments.static_image_bridge import config as c
from scripts.experiments.static_image_bridge import frontal_probe as f
from scripts.experiments.static_image_bridge.common import (
    file_sha256,
    read_json,
    read_pcm16,
    verify_self_hashed_json,
    write_json_atomic,
    write_self_hashed_json,
)

ROOT = c.REPO / "runs/natural_clock_pilot_20260913"
PARENT = c.REPO / "runs/static_image_bridge_lowalpha_20260913_v2/protocol.json"
ARMS = ("N", "EQ", "DAC", "FASTPITCH")


def prepare(root):
    if (root / "protocol.json").exists():
        p = verify_self_hashed_json(root / "protocol.json")
        for path, sha in p["bindings"].items():
            if file_sha256(path) != sha:
                raise ValueError(f"frozen binding changed: {path}")
        return p
    if root.exists():
        raise ValueError("new experiment requires fresh directory")
    old = verify_self_hashed_json(PARENT)
    records = copy.deepcopy(old["records"][:3])
    source = {
        r["sample_id"]: r
        for r in read_json(c.REPO / "runs/static_image_bridge_20260913/inputs.json")[
            "records"
        ]
    }
    files = [
        PARENT,
        Path(__file__),
        Path(__file__).with_name("natural_clock_pilot_audio.py"),
        c.WAV2LIP_CHECKPOINT,
        c.SYNCNET_MODEL,
        MODEL_ROOT / "nvidia_fastpitch_210824.pt",
        MODEL_ROOT / "hifigan_gen_checkpoint_10000_ft.pt",
        Path("/home/wjj/.cache/descript/16khz/0.0.5/dac/weights.pth"),
    ]
    files.extend(Path(f.__file__).parent.glob("*.py"))
    files.extend(FP_ROOT.rglob("*.py"))
    for r in records:
        sid = r["sample_id"]
        r["source_video"] = source[sid]["face_video"]["path"]
        r["transcript"] = source[sid]["transcript"]
        r["textgrid"] = str(
            c.REPO
            / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/01_mfa3_screen_retry1/natural_textgrids"
            / f"{sid}.TextGrid"
        )
        r["audio"] = {
            k: v for k, v in r["audio"].items() if k in ("N", "N_REPEAT", "ND", "S")
        }
        for value in r["audio"].values():
            if file_sha256(value["path"]) != value["container_sha256"]:
                raise ValueError("parent audio hash mismatch")
            files.append(Path(value["path"]))
        if (
            file_sha256(r["source_video"])
            != source[sid]["face_video"]["container_sha256"]
        ):
            raise ValueError("parent video hash mismatch")
        files.extend([Path(r["source_video"]), Path(r["textgrid"])])
    image = copy.deepcopy(old["images"][0])
    files.append(Path(image["path"]))
    p = {
        "records": records,
        "images": [image],
        "arms": ARMS,
        "selection": "first 3 records and first portrait of existing protocol, frozen before scoring",
        "real_noninferiority_margin_C": 0.1,
        "native_useful_gain_C": 0.05,
        "scope": "exploratory seen-fit n=3, one image; no multiple-testing correction; no generalization",
        "methods": {
            "EQ": "fixed +2dB broad high shelf at 1500Hz, natural STFT phase, global RMS restored",
            "DAC": "16k full-codebook identity reconstruction, crop only, no loudness normalization",
            "FASTPITCH": "natural MFA durations quantized to 256/22050 grid, approximate IPA->CMU, natural phone-mean F0 and energy; fixed LJSpeech voice; no warp",
        },
        "controls": "first source: N rerender, 200ms delay scoring, LOCAL_SWAP bidirectional response",
        "bindings": {str(path.resolve()): file_sha256(path) for path in files},
    }
    root.mkdir(parents=True)
    write_self_hashed_json(root / "protocol.json", p)
    write_json_atomic(root / f"score_box_{image['id']}.json", image["score_box"])
    return verify_self_hashed_json(root / "protocol.json")


def audio(root, p):
    for r in p["records"]:
        pcm, _ = read_pcm16(Path(r["audio"]["N"]["path"]))
        eq, meta = equalize(pcm)
        rt, _ = equalize(pcm, 0)
        if not np.array_equal(rt, pcm):
            raise ValueError("STFT identity is not exact PCM")
        out = root / "audio" / r["sample_id"] / "EQ.wav"
        write_pcm16(out, eq)
        write_json_atomic(
            out.with_suffix(".json"),
            {**meta, "sha256": file_sha256(out), "roundtrip_pcm_exact": True},
        )
    for method in ("dac", "fastpitch"):
        f.job(
            root,
            [
                str(c.REPO / ".venv/bin/python"),
                "-m",
                "scripts.experiments.natural_clock_pilot_audio",
                method,
                "--root",
                str(root),
            ],
            root / "logs" / f"audio_{method}.log",
            gpu=True,
        )
    for r in p["records"]:
        n, _ = read_pcm16(Path(r["audio"]["N"]["path"]))
        for arm in ARMS[1:]:
            path = root / "audio" / r["sample_id"] / f"{arm}.wav"
            vals, meta = read_pcm16(path)
            if len(vals) != len(n) or not np.isfinite(vals).all():
                raise ValueError("candidate audio contract failed")
            r["audio"][arm] = {"path": str(path), **meta}
    write_self_hashed_json(
        root / "audio_manifest.json",
        {
            "records": p["records"],
            "protocol_sha256": file_sha256(root / "protocol.json"),
        },
    )


def load_records(root, p):
    a = verify_self_hashed_json(root / "audio_manifest.json")
    if a["protocol_sha256"] != file_sha256(root / "protocol.json"):
        raise ValueError("audio protocol mismatch")
    for r in a["records"]:
        for meta in r["audio"].values():
            if file_sha256(meta["path"]) != meta["container_sha256"]:
                raise ValueError("audio changed")
    p["records"] = a["records"]
    return p


def real_score(root, p):
    import torch

    from scripts.experiments.local_swap_minimal_replay.media import (
        lock_source_crop,
        validate_zero_based_pts,
    )
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        make_face_detector,
    )
    from scripts.experiments.static_image_bridge.score_worker import (
        audio_embedding,
        visual_embedding,
    )

    sys.path.insert(0, str(c.SYNCNET_ROOT))
    from SyncNetInstance import SyncNetInstance

    torch.set_num_threads(2)
    detector = make_face_detector()
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(str(c.SYNCNET_MODEL))
    scorer.eval()
    rows = []
    for i, r in enumerate(p["records"]):
        source = Path(r["source_video"])
        pts = validate_zero_based_pts(source)
        _, crops, cropmeta = lock_source_crop(source, detector)
        v, _ = visual_embedding(np.stack(crops), scorer, "cuda", torch)
        w = r["support"]["primary"]
        scores = {}
        folder = root / "real" / r["sample_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for arm in (*ARMS, "ND", "S") if i == 0 else ARMS:
            a, _, _ = audio_embedding(
                Path(r["audio"][arm]["path"]), scorer, "cuda", torch
            )
            mat = np.full((min(len(v), len(a)), 31), np.nan)
            for lag in range(-15, 16):
                ts = np.array(w)
                if (
                    ts.max() >= len(v)
                    or (ts + lag).max() >= len(a)
                    or (ts + lag).min() < 0
                ):
                    raise ValueError("frozen support outside embeddings")
                mat[ts, lag + 15] = torch.nn.functional.pairwise_distance(
                    v[ts].float(), a[ts + lag].float()
                ).numpy()
            path = folder / f"{arm}.npy"
            np.save(path, mat, allow_pickle=False)
            scores[arm] = {
                **f.metrics(mat, w),
                "matrix": str(path),
                "matrix_sha256": file_sha256(path),
            }
        row = {
            "sample_id": r["sample_id"],
            "scores": scores,
            "crop": cropmeta,
            "pts": pts,
            "source_sha256": file_sha256(source),
            "windows": w,
        }
        rows.append(row)
        write_json_atomic(folder / "result.json", row)
        print("REAL_DONE", i + 1, r["sample_id"], flush=True)
    write_json_atomic(root / "real_results.json", {"rows": rows})


def generate(root, p):
    im = p["images"][0]
    for i, r in enumerate(p["records"]):
        for arm in (*ARMS, "N_REPEAT", "S") if i == 0 else ARMS:
            video = f.render(root, im, r, arm)
            audios = [arm, "N"] if arm not in ("N", "N_REPEAT") else ["N"]
            if arm == "N" and i == 0:
                audios = ["N", "ND", "S"]
            f.score(root, im, r, arm, video, list(dict.fromkeys(audios)))
            print("GENERATED", i + 1, arm, flush=True)


def comparison(n, a):
    return {
        "delta_C": a["C"] - n["C"],
        "D_improvement": n["D"] - a["D"],
        "anchor_improvement": n["D"] - a["curve"][n["lag"] + 15],
        "lag_change": a["lag"] - n["lag"],
    }


def analyze(root, p):
    from scripts.experiments.static_image_bridge.lowalpha_probe import controls

    im = p["images"][0]
    real = {
        r["sample_id"]: r["scores"]
        for r in read_json(root / "real_results.json")["rows"]
    }
    checks = controls(root, p)
    rows = []
    for r in p["records"]:
        nn = f.metrics(f.load_matrix(root, im, r, "N", "N"), r["support"]["primary"])
        for arm in ARMS[1:]:
            own = f.metrics(
                f.load_matrix(root, im, r, arm, arm), r["support"]["primary"]
            )
            crossed = f.metrics(
                f.load_matrix(root, im, r, arm, "N"), r["support"]["primary"]
            )
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "source_group": r["source_group"],
                    "arm": arm,
                    "real": comparison(
                        real[r["sample_id"]]["N"], real[r["sample_id"]][arm]
                    ),
                    "native": comparison(nn, own),
                    "replacement": comparison(nn, crossed),
                    "baseline": nn,
                    "candidate": own,
                }
            )
    summaries = {}
    for arm in ARMS[1:]:
        subset = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {
            endpoint: {
                metric: f.summary([r[endpoint][metric] for r in subset])
                for metric in ("delta_C", "D_improvement", "anchor_improvement")
            }
            for endpoint in ("real", "native", "replacement")
        }
        s = summaries[arm]
        s["dual_positive_samples"] = sum(
            r["real"]["delta_C"] >= -0.1 and r["native"]["delta_C"] > 0.05
            for r in subset
        )
        s["exploratory_dual_gate"] = bool(
            all(c["measurement_pass"] and c["swap_pass"] for c in checks)
            and s["real"]["delta_C"]["ci95"][0] >= -0.1
            and s["native"]["delta_C"]["mean"] > 0.05
            and s["native"]["delta_C"]["ci95"][0] > 0
            and s["native"]["D_improvement"]["ci95"][0] >= 0
            and s["native"]["anchor_improvement"]["ci95"][0] >= 0
        )
    result = {
        "rows": rows,
        "summaries": summaries,
        "controls": checks,
        "n": 3,
        "images": 1,
        "interpretation": "exploratory only; n=3 bootstrap intervals unstable, unadjusted; diagonal C is not pure video improvement",
        "protocol_sha256": file_sha256(root / "protocol.json"),
    }
    write_json_atomic(root / "analysis.json", result)
    lines = [
        "# 自然时钟音频双目标小样本筛查",
        "",
        "## 设计与计算流程",
        "",
        "固定既有队列前3条音频、正脸图3号；三条音频分别来自三个source group。不是新样本验证，不按分数筛选。",
        "每条音频均进行EQ、DAC与FastPitch处理。真实自然视频固定，只改变评分音轨；另用静态图Wav2Lip生成各臂视频，并用驱动时的同一PCM评分（原生，不替换）。",
        "EQ：固定1500Hz宽高架+2dB，保留自然STFT相位，恢复整句RMS，不改变时间轴。",
        "DAC：16k全码本identity重合成，补齐模型hop后只右裁回自然长度，不改音量。",
        "FastPitch：自然MFA音素时长映射到256/22050秒网格，输入自然音素平均F0和能量，直接合成mel再用官方HiFi-GAN合成波形。无事后时间拉伸。IPA→CMU发音映射近似且重音不精确，使用LJSpeech单女声，不是自克隆。MFA输入边界锁定不代表输出发音边界已验证。",
        "所有视频/音频共用冻结的完整前端真实支持和±15帧搜索。Sync-C=平均距离曲线中位数−最小值；D改善=自然基线D−候选D；固定lag改善在自然基线最佳lag计算。C高不等于零延迟。",
        "先冻结3条再评分；n=3配对bootstrap10000次，区间不稳定且未校正多重比较。预设真实视频非劣性界−0.100；原生参考增益+0.050且CI下界>0，并要求D及固定lag不退化与控制通过。纯筛查，不作泛化结论。",
        "",
        "## 汇总结果",
        "",
        "|方案|真实视频ΔC [95% CI]|静态生成原生ΔC [95% CI]|原生D改善|双条件样本数|探索门槛|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm, s in summaries.items():
        a, b = s["real"]["delta_C"], s["native"]["delta_C"]
        lines.append(
            f"|{arm}|{a['mean']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}]|{b['mean']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]|{s['native']['D_improvement']['mean']:+.3f}|{s['dual_positive_samples']}/3|{s['exploratory_dual_gate']}|"
        )
    lines += [
        "",
        "## 逐条结果",
        "",
        "|样本|方案|真实ΔC|原生ΔC|原生D改善|原生lag变化|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['sample_id']}|{r['arm']}|{r['real']['delta_C']:+.3f}|{r['native']['delta_C']:+.3f}|{r['native']['D_improvement']:+.3f}|{r['native']['lag_change']}|"
        )
    lines += [
        "",
        "## 控制与限制",
        "",
        str(checks),
        "使用独立重复渲染、200ms延迟和LOCAL_SWAP双向偏好控制。复用渲染器验证无损逐帧一致及ROI外静态；输入/模型/矩阵hash绑定。",
        "本轮未完成主观盲评、ASR/PER或输出重新forced alignment，不能宣称听感/可懂度/真实同步提高。FastPitch包含说话人、发音映射与声码器差异，不能单独归因于时长控制。",
        "原生比较同时改变画面和评分音频，分数增益不等于纯口型质量改善；replacement仅作辅助记录。",
        "评分使用无损画面和独立原始PCM。playback中的MP4/AAC只用于观看，不回流评分。",
        "",
        "## 产物",
        "",
        "逐条音频：audio/<sample_id>/<arm>.wav；原生预览：playback/<sample_id>/<arm>.mp4；原始距离矩阵：matrices/与real/；analysis.json含完整距离曲线和辅助replacement。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("SUMMARY", summaries, flush=True)


def playback(root, p):
    im = p["images"][0]
    for r in p["records"]:
        for arm in ARMS:
            out = root / "playback" / r["sample_id"] / f"{arm}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            f.job(
                root,
                [
                    str(c.FFMPEG),
                    "-n",
                    "-v",
                    "error",
                    "-i",
                    str(root / "videos" / im["id"] / r["sample_id"] / f"{arm}.mkv"),
                    "-i",
                    r["audio"][arm]["path"],
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "veryfast",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(out),
                ],
                out.with_suffix(".log"),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["prepare", "audio", "real", "generate", "analyze", "playback", "all"],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    p = prepare(root)
    if args.stage == "prepare":
        return
    if args.stage in ("audio", "all"):
        audio(root, p)
    if args.stage == "audio":
        return
    p = load_records(root, p)
    if args.stage == "all":
        f.job(
            root,
            [
                str(c.SYNCNET_PYTHON),
                "-m",
                "scripts.experiments.natural_clock_pilot",
                "real",
                "--root",
                str(root),
            ],
            root / "logs/real.log",
            gpu=True,
        )
        generate(root, p)
        analyze(root, p)
        playback(root, p)
    else:
        {
            "real": real_score,
            "generate": generate,
            "analyze": analyze,
            "playback": playback,
        }[args.stage](root, p)


if __name__ == "__main__":
    main()
