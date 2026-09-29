"""Exploratory frozen-portrait replay; does not change the original spec/run.

Uses the audited rendering/forward workers, but not their analysis or validator.
All time windows and image choices are frozen before any new scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as c
from .common import (
    bytes_sha256, file_sha256, read_json, read_pcm16, run_logged,
    verify_self_hashed_json, write_json_atomic, write_self_hashed_json,
)
from .images import generation_box, score_box, select_detection

PARENT = c.REPO / "runs/static_image_bridge_20260913"
DEFAULT_ROOT = c.REPO / "runs/static_image_bridge_frontal_20260913"
IMAGE_IDS = (3, 6, 9)


def generator_support(t: int) -> tuple[int, int]:
    return int(t * 3.2) * 200 - 401, (int((t + 4) * 3.2) + 15) * 200 + 400


def frozen_support(length: int) -> dict:
    """Real MFCC (including preemphasis) and 5-frame generator receptive fields."""
    def valid(t, lo, hi, lag):
        g0, g1 = generator_support(t)
        return g0 >= lo and g1 <= hi and (t + lag) * 640 - 1 >= lo and (t + lag) * 640 + 3440 <= hi

    candidates = range(length // 640)
    primary = [t for t in candidates if all(valid(t, 0, length, k) for k in range(-15, 16))]
    delay = [t for t in primary if (t - 15) * 640 - 1 >= 3200]
    b1, b2, b3 = length // 4, length // 2, 3 * length // 4
    swapped_join = b1 + b3 - b2
    local = {}
    for label, lo, hi in (("2", b1, min(b2, swapped_join)), ("3", max(b2, swapped_join), b3)):
        local[label] = {str(k): [t for t in candidates if valid(t, lo, hi, k)] for k in range(-15, 16)}
    if len(primary) < 20 or len(delay) < 20:
        raise ValueError("insufficient full-frontend real support")
    return {"primary": primary, "delay": delay, "local_by_lag": local,
            "generator": "[floor(3.2t)*200-401, (floor(3.2(t+4))+15)*200+400)",
            "mfcc": "[(t+lag)*640-1, (t+lag)*640+3440)"}


def gpu_ready(root: Path) -> None:
    """Wait without touching other users' work; recheck before each GPU job."""
    while True:
        state = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [int(x.strip()) for x in state.splitlines()[0].split(",")]
        stamp = {"time": time.time(), "gpu": state}
        with (root / "gpu_checks.jsonl").open("a") as f:
            f.write(json.dumps(stamp) + "\n")
        if fields[2] >= 5000 and fields[3] <= 20:
            return
        print(f"GPU_WAIT {state}", flush=True)
        time.sleep(15)


def job(root: Path, cmd: list[str], log: Path, *, gpu: bool = False) -> None:
    if shutil.disk_usage(root).free < 1500 * 1024**2:
        raise RuntimeError("less than 1.5 GiB free disk; stop without deleting existing artifacts")
    if gpu:
        gpu_ready(root)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    run_logged(cmd, c.REPO, log, env=env)


def prepare(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "protocol.json"
    if frozen.exists():
        p = verify_self_hashed_json(frozen)
        for path, sha in p["bindings"].items():
            if file_sha256(path) != sha:
                raise ValueError(f"frozen file changed: {path}")
        return p
    if (root / "matrices").exists():
        raise ValueError("cannot freeze protocol after scoring has started")
    source = verify_self_hashed_json(PARENT / "audio_manifest.json")
    records, bindings = [], {}
    for row in source["rows"]:
        arms = {**row["arms"], **row["controls"]}
        arrays = {}
        for arm, meta in arms.items():
            values, actual = read_pcm16(Path(meta["path"]))
            assert actual["pcm_sha256"] == meta["pcm_sha256"]
            assert actual["container_sha256"] == meta["container_sha256"]
            bindings[meta["path"]] = actual["container_sha256"]
            arrays[arm] = values
        n = arrays["N"]
        assert np.array_equal(n, arrays["RT"]) and np.array_equal(n, arrays["N_REPEAT"])
        assert len(n) == len(arrays["B"])
        b1, b2, b3 = len(n) // 4, len(n) // 2, 3 * len(n) // 4
        assert np.array_equal(arrays["S"], np.concatenate([n[:b1], n[b2:b3], n[b1:b2], n[b3:]]))
        assert np.array_equal(arrays["ND"], np.concatenate([np.zeros(3200, np.int16), n[:-3200]]))
        records.append({"sample_id": row["sample_id"], "source_group": row["source_group"],
                        "audio": arms, "length": len(n), "support": frozen_support(len(n))})
    assert len(records) == 22 and len({r["source_group"] for r in records}) == 22
    requests = []
    for identity in IMAGE_IDS:
        path = c.REPO / f"data/data/image/{identity}.png"
        frame = cv2.imread(str(path))
        assert frame is not None and frame.shape == (512, 512, 3)
        bindings[str(path)] = file_sha256(path)
        requests.append({"sample_id": str(identity), "image": str(path)})
    request_path = root / "detection_requests.json"
    write_json_atomic(request_path, {"rows": requests})
    # Detector CLI takes a list; serialize it from the frozen image selection.
    request_path.write_text(json.dumps(requests, indent=2) + "\n")
    detection = root / "detections.json"
    job(root, [str(c.WAV2LIP_PYTHON), str(Path(__file__).with_name("detect_worker.py")),
               "--requests", str(request_path), "--output", str(detection)], root / "logs/detect.log", gpu=True)
    images = []
    for row in read_json(detection)["records"]:
        selected = select_detection(row["detections"])
        image = {"id": row["sample_id"], "path": row["image"], "rgb_sha256": row["image_sha256"],
                 "width": row["width"], "height": row["height"], "detection": selected,
                 "generation_box": generation_box(selected, row["width"], row["height"]),
                 "score_box": score_box(selected)}
        write_json_atomic(root / f"score_box_{image['id']}.json", image["score_box"])
        images.append(image)
    for path in [PARENT / "audio_manifest.json", Path(__file__), Path(__file__).with_name("images.py"),
                 Path(__file__).with_name("common.py"), Path(__file__).with_name("config.py"),
                 Path(__file__).with_name("render_worker.py"), Path(__file__).with_name("score_worker.py"),
                 Path(__file__).with_name("detect_worker.py"), c.WAV2LIP_ROOT / "audio.py",
                 c.WAV2LIP_ROOT / "hparams.py", c.SYNCNET_ROOT / "SyncNetInstance.py",
                 c.SYNCNET_ROOT / "SyncNetModel.py", c.WAV2LIP_CHECKPOINT, c.SYNCNET_MODEL, c.FFMPEG]:
        bindings[str(path.resolve())] = file_sha256(path)
    for path in sorted((c.WAV2LIP_ROOT / "models").glob("*.py")):
        bindings[str(path.resolve())] = file_sha256(path)
    p = {"experiment": "frontal_portrait_bridge_sensitivity", "status": "frozen_before_scoring",
         "created_unix": time.time(), "images": images, "records": records, "bindings": bindings,
         "primary": "C(V_B,A_N)-C(V_N,A_N), same image, full real support, lags -15..15",
         "controls": "first frozen audio on each image: independent N_REPEAT, ND +200ms, S",
         "sampling_unit": "22 source groups; average three fixed-image deltas within group",
         "bootstrap": {"draws": 10000, "seed": 20260913}, "alpha": 0.75,
         "image_selection": "3,6,9 selected visually for frontal pose and unobscured mouth before scoring; no claim of professional high resolution",
         "RT": "all22 decoded PCM identical to N, no duplicate rendering",
         "format": "FFV1 worker -> libx264rgb CRF0; every decoded frame and outside-ROI pixels verified",
         "scope": "exploratory, seen-fit audio, fixed 3 identities; not old spec acceptance or dynamic leakage proof"}
    write_self_hashed_json(frozen, p)
    return verify_self_hashed_json(frozen)


def verify_transcode(source: Path, compact: Path, image: dict) -> dict:
    cap1, cap2 = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(compact))
    reference = cv2.imread(image["path"])
    x1, y1, x2, y2 = image["generation_box"]
    mask = np.ones(reference.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = False
    import hashlib
    digest = hashlib.sha256()
    count = 0
    try:
        while True:
            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if ok1 != ok2:
                raise ValueError("lossless transcode frame count mismatch")
            if not ok1:
                break
            if not np.array_equal(f1, f2) or not np.array_equal(f1[mask], reference[mask]):
                raise ValueError("decoded pixels changed, or original motion outside ROI")
            digest.update(f1.tobytes())
            count += 1
    finally:
        cap1.release()
        cap2.release()
    if count < 25:
        raise ValueError("too few frames")
    return {"frame_count": count, "raw_bgr_sha256": digest.hexdigest(),
            "all_pixels_equal": True, "all_outside_roi_static": True}


def render(root: Path, image: dict, record: dict, arm: str) -> Path:
    base = root / "videos" / image["id"] / record["sample_id"]
    base.mkdir(parents=True, exist_ok=True)
    output, receipt = base / f"{arm}.mkv", base / f"{arm}.json"
    if receipt.exists():
        old = read_json(receipt)
        assert file_sha256(output) == old["compact_sha256"]
        return output
    intermediate = base / f"{arm}.ffv1.mkv"
    worker = base / f"{arm}.worker.json"
    job(root, [str(c.WAV2LIP_PYTHON), str(Path(__file__).with_name("render_worker.py")),
               "--image", image["path"], "--image-rgb-sha256", image["rgb_sha256"],
               "--audio", record["audio"][arm]["path"], "--box", *map(str, image["generation_box"]),
               "--checkpoint", str(c.WAV2LIP_CHECKPOINT), "--ffmpeg", str(c.FFMPEG),
               "--outfile", str(intermediate), "--result", str(worker), "--batch-size", "4"],
        root / "logs" / f"render_{image['id']}_{record['sample_id']}_{arm}.log", gpu=True)
    job(root, [str(c.FFMPEG), "-y", "-v", "error", "-i", str(intermediate), "-an", "-c:v", "libx264rgb",
               "-crf", "0", "-preset", "veryfast", "-threads", "2", "-pix_fmt", "bgr24", str(output)],
        base / f"{arm}.transcode.log")
    parity = verify_transcode(intermediate, output, image)
    wr = read_json(worker)
    assert parity["raw_bgr_sha256"] == wr["encode"]["raw_frame_sha256"]
    assert wr["audio_sha256"] == record["audio"][arm]["container_sha256"]
    write_json_atomic(receipt, {"compact_sha256": file_sha256(output), "output": str(output),
                               "parity": parity, "worker": wr,
                               "removed_redundant_intermediate": str(intermediate)})
    # Only this run's newly-created duplicate, after exact decoded verification.
    intermediate.unlink()
    return output


def score(root: Path, image: dict, record: dict, arm: str, video: Path, audios: list[str]) -> None:
    output = root / "matrices" / image["id"] / record["sample_id"]
    if all((output / f"{arm}__{a}__worker.json").exists() for a in audios):
        return
    request = root / "requests" / f"{image['id']}_{record['sample_id']}_{arm}.json"
    write_json_atomic(request, {a: record["audio"][a]["path"] for a in audios})
    job(root, [str(c.SYNCNET_PYTHON), str(Path(__file__).with_name("score_worker.py")),
               "--video", str(video), "--video-arm", arm, "--sample-id", record["sample_id"],
               "--score-box", str(root / f"score_box_{image['id']}.json"), "--audio-json", str(request),
               "--model", str(c.SYNCNET_MODEL), "--output-dir", str(output), "--batch-size", "20"],
        root / "logs" / f"score_{image['id']}_{record['sample_id']}_{arm}.log", gpu=True)


def metrics(matrix: np.ndarray, windows: list[int]) -> dict:
    if len(windows) < 5 or max(windows) >= len(matrix):
        raise ValueError("insufficient frozen windows or incomplete generated frames")
    selected = matrix[windows]
    if not np.isfinite(selected).all():
        raise ValueError("frozen support includes non-finite distances; cannot adapt W")
    curve = selected.mean(axis=0)
    col = int(curve.argmin())
    return {"C": float(np.median(curve) - curve[col]), "D": float(curve[col]),
            "lag": col - 15, "curve": curve.tolist()}


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("invalid summary")
    draws = np.random.default_rng(20260913).integers(0, len(a), (10000, len(a)))
    return {"mean": float(a.mean()), "ci95": np.percentile(a[draws].mean(1), [2.5, 97.5]).tolist(),
            "positive": int((a > 0).sum()), "n": len(a)}


def load_matrix(root: Path, image: dict, record: dict, v: str, a: str) -> np.ndarray:
    base = root / "matrices" / image["id"] / record["sample_id"]
    meta = read_json(base / f"{v}__{a}__worker.json")
    expected_video = root / "videos" / image["id"] / record["sample_id"] / f"{v}.mkv"
    assert Path(meta["video"]) == expected_video
    assert meta["sample_id"] == record["sample_id"] and meta["video_arm"] == v and meta["audio_arm"] == a
    assert meta["audio"] == record["audio"][a]["path"]
    assert meta["audio_sha256"] == file_sha256(meta["audio"])
    assert meta["video_sha256"] == file_sha256(expected_video)
    assert meta["matrix_sha256"] == file_sha256(meta["matrix"])
    assert meta["model_sha256"] == c.SYNCNET_MODEL_SHA256
    assert meta["video_meta"]["crop_box"] == image["score_box"]
    return np.load(meta["matrix"], allow_pickle=False)


def analyze(root: Path, p: dict) -> dict:
    pairs, controls = [], []
    for image in p["images"]:
        for i, record in enumerate(p["records"]):
            nn = load_matrix(root, image, record, "N", "N")
            bn = load_matrix(root, image, record, "B", "N")
            bb = load_matrix(root, image, record, "B", "B")
            w = record["support"]["primary"]
            mn, mb, own = metrics(nn, w), metrics(bn, w), metrics(bb, w)
            pairs.append({"image_id": image["id"], "sample_id": record["sample_id"], "source_group": record["source_group"],
                          "window_count": len(w), "N": mn, "B": mb, "B_own": own,
                          "delta_C": mb["C"] - mn["C"], "D_improvement": mn["D"] - mb["D"],
                          "anchor_improvement": mn["D"] - mb["curve"][mn["lag"] + 15],
                          "B_own_C_preference": own["C"] - mb["C"], "offset_change": mb["lag"] - mn["lag"]})
            if i == 0:
                rep = load_matrix(root, image, record, "N_REPEAT", "N")
                nd = load_matrix(root, image, record, "N", "ND")
                ns = load_matrix(root, image, record, "N", "S")
                sn = load_matrix(root, image, record, "S", "N")
                ss = load_matrix(root, image, record, "S", "S")
                repeat_err = float(np.max(np.abs(rep[w] - nn[w])))
                delay_w = record["support"]["delay"]
                dn, dd = metrics(nn, delay_w), metrics(nd, delay_w)
                damage = dd["curve"][dn["lag"] + 15] - dn["D"]
                local = {}
                for label, by_lag in record["support"]["local_by_lag"].items():
                    lw = by_lag[str(mn["lag"])]
                    if len(lw) < 5:
                        raise ValueError("local control lacks pre-frozen support at natural lag")
                    col = mn["lag"] + 15
                    local[label] = {"windows": lw, "N_prefers_N": float((ns[lw, col] - nn[lw, col]).mean()),
                                    "S_prefers_S": float((sn[lw, col] - ss[lw, col]).mean())}
                controls.append({"image_id": image["id"], "sample_id": record["sample_id"], "repeat_matrix_error": repeat_err,
                                 "delay_lag_change": dd["lag"] - dn["lag"], "delay_anchor_damage": damage,
                                 "measurement_pass": repeat_err < 1e-5 and abs(dd["lag"] - dn["lag"] - 5) <= 1 and damage > 0.1,
                                 "local": local, "swap_transfer_observed": all(x[k] > 0.1 for x in local.values() for k in ("N_prefers_N", "S_prefers_S"))})
    by_image = {im["id"]: {key: summary([x[key] for x in pairs if x["image_id"] == im["id"]])
                            for key in ("delta_C", "D_improvement", "anchor_improvement", "B_own_C_preference")} for im in p["images"]}
    grouped = []
    for record in p["records"]:
        rows = [r for r in pairs if r["source_group"] == record["source_group"]]
        assert len(rows) == 3
        grouped.append({"source_group": record["source_group"], **{key: float(np.mean([r[key] for r in rows]))
                        for key in ("delta_C", "D_improvement", "anchor_improvement")}})
    result = {"status": "complete", "protocol_sha256": file_sha256(root / "protocol.json"), "pair_count": len(pairs),
              "source_groups": len(grouped), "by_image": by_image,
              "grouped": {key: summary([r[key] for r in grouped]) for key in ("delta_C", "D_improvement", "anchor_improvement")},
              "positive_pairs_C": sum(r["delta_C"] > 0 for r in pairs),
              "positive_pairs_joint": sum(r["delta_C"] > 0 and r["D_improvement"] > 0 for r in pairs),
              "controls": controls, "pairs": pairs, "group_rows": grouped,
              "claim_boundary": p["scope"], "training_authorized": False,
              "validation": "hash/pairing/pixel/support checks plus new independent matrix analysis; not full original spec acceptance"}
    write_json_atomic(root / "analysis.json", result)
    with (root / "per_pair.csv").open("w", newline="") as f:
        keys = ["image_id", "sample_id", "source_group", "delta_C", "D_improvement", "anchor_improvement", "offset_change"]
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in pairs)
    return result


def playback(root: Path, p: dict) -> None:
    # All pairs, fixed N audio for side-by-side; clearly labelled supplementary B-own playback.
    for image in p["images"]:
        for record in p["records"]:
            target = root / "playback" / image["id"] / f"{record['sample_id']}__N_vs_B.mp4"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            base = root / "videos" / image["id"] / record["sample_id"]
            filters = "[0:v]drawtext=text='N - natural driver':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black[n];[1:v]drawtext=text='B - bridge 0.75':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black[b];[n][b]hstack=inputs=2[v]"
            job(root, [str(c.FFMPEG), "-y", "-v", "error", "-i", str(base / "N.mkv"), "-i", str(base / "B.mkv"),
                       "-i", record["audio"]["N"]["path"], "-filter_complex", filters, "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-threads", "2", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(target)],
                target.with_suffix(".log"))


def report(root: Path, p: dict, result: dict) -> None:
    lines = ["# 本地正脸静态图 bridge 配对试验", "", "本试验是参考图敏感性探索，不是旧 spec 的完整修复验收。",
             "", "## 计算流程", "", "固定本地3、6、9号512×512正脸图，与原全部22条seen-fit自然音频全交叉，共66对。图像在评分前凭正脸/无遮挡选择，未按分数筛图。",
             "N=原自然音频；B=冻结alpha=0.75 bridge；N/B都使用同一张图、固定检测脸框和评分crop。RT的PCM逐条等于N，未重复渲染。",
             "主比较两段生成视频都配原自然音频：ΔC=C(V_B,A_N)−C(V_N,A_N)；D改善=D(V_N,A_N)−D(V_B,A_N)。正数才是改善。",
             "评分前按MFCC/preemphasis和生成mel/STFT完整感受野冻结共同真实支持；每个窗口遍历±15帧lag，先平均距离曲线，C=曲线中位数−最小距离。",
             "同一音频的三张图先平均差值，再对22个source_group bootstrap 10000次(seed=20260913)。66对不是66个独立音频样本。",
             "GPU串行batch4生成、batch20评分；无损视频逐帧转码校验，所有帧ROI外与参考图一致；父运行未覆盖。", "", "## 结果", "",
             "| 参考图 | ΔSync-C均值（95% CI） | C改善 | D改善均值 |", "|---|---:|---:|---:|"]
    for identity, row in result["by_image"].items():
        a, d = row["delta_C"], row["D_improvement"]
        lines.append(f"| [图{identity}](../../data/data/image/{identity}.png) | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {a['positive']}/{a['n']} | {d['mean']:+.3f} |")
    a = result["grouped"]["delta_C"]
    lines += ["", f"按音频分组平均：ΔC={a['mean']:+.3f}，95% CI [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]；C改善{result['positive_pairs_C']}/66对；C与D联合改善{result['positive_pairs_joint']}/66对。",
              "", "## 小规模测量对照", "", "仅首条冻结音频×三张图；不能当22条全量对照。"]
    for row in result["controls"]:
        lines += [f"- 图{row['image_id']}：repeat矩阵最大差{row['repeat_matrix_error']:.8f}；+200ms音频lag改变{row['delay_lag_change']}帧，固定自然lag距离损伤{row['delay_anchor_damage']:.3f}；LOCAL_SWAP两段双向偏好通过={row['swap_transfer_observed']}。"]
    lines += ["", "## 解释边界", "", "原图是512像素近正脸人像，不等于专业高清素材；参考身份和脸部几何也改变，因此这不是只操纵清晰度/朝向的单变量实验。",
              "B自身音轨评分只作辅助，不能与N原音轨主结果混算。三张图有限、音频seen-fit，不证明一般无效、跨身份泛化，也不证明动态嘴型泄漏。人工审阅仍待用户完成。",
              "", "## 视频对照", "", "左右分别为N驱动/B驱动，播放同一条原自然音轨。以下包含全部66对，顺序固定，无挑选。", ""]
    for record in p["records"]:
        links = " · ".join(f"[图{im['id']}](playback/{im['id']}/{record['sample_id']}__N_vs_B.mp4)" for im in p["images"])
        lines.append(f"- {record['sample_id']}：{links}")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze", "playback"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.run_root.resolve()
    p = prepare(root)
    if args.command == "prepare":
        print(f"FROZEN {len(p['images'])} images x {len(p['records'])} audio records", flush=True)
        return 0
    if args.command == "run":
        # Controls first on each identity; no score-directed change of cohort or images.
        ordered = [(im, r, i) for i, r in enumerate(p["records"]) for im in p["images"]]
        for number, (image, record, i) in enumerate(ordered, 1):
            arms = ["N", "N_REPEAT", "S", "B"] if i == 0 else ["N", "B"]
            for arm in arms:
                video = render(root, image, record, arm)
                audios = {"N": ["N", "ND", "S"] if i == 0 else ["N"], "B": ["N", "B"], "N_REPEAT": ["N"], "S": ["N", "S"]}[arm]
                score(root, image, record, arm, video, audios)
            print(f"PAIR_DONE {number}/66 image={image['id']} audio={record['sample_id']}", flush=True)
        result = analyze(root, p)
        report(root, p, result)
        playback(root, p)
    elif args.command == "analyze":
        result = analyze(root, p)
        report(root, p, result)
    else:
        playback(root, p)
    print(f"COMPLETE {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
