"""Fixed-real-video, natural-phase bridge dose response (exploratory)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.audio import (
    float_to_pcm16,
    read_pcm16,
    write_pcm16,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.audio import (
    phase_preserving_blend,
)
from scripts.experiments.static_image_bridge.audio import (
    delayed_audio,
    quarter_swap,
    stft_roundtrip_alpha0,
)
from scripts.experiments.static_image_bridge.score_worker import (
    audio_embedding,
    sha256_file,
    visual_embedding,
)

REPO = Path(__file__).resolve().parents[2]
ARMS = ("N", "RT", "B025", "B050", "B075", "B100", "MFA", "SWAP", "DELAY200")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def distance_matrix(visual, audio, count):
    """Identical real support for all 31 lags; no padded embeddings."""
    rows = np.arange(15, count - 15)
    if len(rows) < 25:
        raise ValueError("Insufficient common support")
    return np.stack(
        [
            np.linalg.norm(visual[rows] - audio[rows + lag] + np.float32(1e-6), axis=1)
            for lag in range(-15, 16)
        ],
        axis=1,
    )


def metrics(matrix, natural_index):
    curve = np.mean(matrix.astype(np.float64), axis=0)
    index = int(np.argmin(curve))
    return {
        "C": float(np.median(curve) - curve[index]),
        "D": float(curve[index]),
        "offset": 15 - index,
        "D0": float(curve[15]),
        "DN": float(curve[natural_index]),
        "curve": curve.tolist(),
    }


def summarize(rows):
    rng = np.random.default_rng(20260913)
    summary = {}
    for arm in ARMS:
        values = [r["scores"][arm] for r in rows]
        delta = np.array([r["scores"][arm]["C"] - r["scores"]["N"]["C"] for r in rows])
        boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        summary[arm] = {
            **{
                k: float(np.mean([v[k] for v in values]))
                for k in ("C", "D", "D0", "DN")
            },
            "delta_C": float(delta.mean()),
            "delta_C_CI95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "improved": int((delta > 0).sum()),
            "offset_changed": sum(
                r["scores"][arm]["offset"] != r["scores"]["N"]["offset"] for r in rows
            ),
            "offsets": [v["offset"] for v in values],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    # A fresh directory prevents silently mixing stale inputs or code versions.
    out.mkdir(parents=True, exist_ok=False)
    inputs_path = REPO / "runs/static_image_bridge_20260913/inputs.json"
    cohort_path = (
        REPO
        / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
    )
    inputs = json.loads(inputs_path.read_text())["records"]
    cohort = {r["sample_id"]: r for r in json.loads(cohort_path.read_text())["records"]}
    from scripts.experiments.static_image_bridge.config import SYNCNET_MODEL

    model = SYNCNET_MODEL
    dump(
        out / "protocol.json",
        {
            "arms": ARMS,
            "intended_n": len(inputs),
            "inputs_sha256": sha256_file(inputs_path),
            "cohort_sha256": sha256_file(cohort_path),
            "code_sha256": sha256_file(Path(__file__)),
            "model_sha256": sha256_file(model),
            "support": "full-track real video; t=15..count-16; count=min(video,audio_samples//640)-5",
            "note": "Common-support SyncNet V2, not padded official global C; exploratory",
        },
    )
    import torch

    from scripts.experiments.local_swap_minimal_replay.media import lock_source_crop
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        make_face_detector,
    )

    sys.path.insert(0, str(REPO / "third_party/syncnet_python"))
    from SyncNetInstance import SyncNetInstance

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    detector = make_face_detector()
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(str(model))
    scorer.eval()
    rows, failures = [], []
    for record in inputs:
        sid = record["sample_id"]
        work = out / sid
        work.mkdir()
        try:
            source = Path(record["face_video"]["path"])
            if sha256_file(source) != record["face_video"]["container_sha256"]:
                raise ValueError("Source video hash mismatch")
            npath = Path(record["audio"]["N"]["path"])
            mpath = Path(cohort[sid]["mfa_linear_audio"]["path"])
            if sha256_file(npath) != record["audio"]["N"]["container_sha256"]:
                raise ValueError("Natural audio hash mismatch")
            if sha256_file(mpath) != cohort[sid]["mfa_linear_audio"]["sha256"]:
                raise ValueError("MFA audio hash mismatch")
            natural, _ = read_pcm16(npath)
            mfa, _ = read_pcm16(mpath)
            if len(natural) != len(mfa):
                raise ValueError("Length mismatch")
            audios = {
                "N": natural,
                "RT": stft_roundtrip_alpha0(natural)[0],
                "MFA": mfa,
                "SWAP": quarter_swap(natural)[0],
                "DELAY200": delayed_audio(natural)[0],
            }
            construction = {}
            for arm, alpha in zip(ARMS[2:6], (0.25, 0.5, 0.75, 1.0), strict=True):
                waveform, meta = phase_preserving_blend(natural, mfa, alpha)
                audios[arm] = float_to_pcm16(waveform)
                construction[arm] = meta
            historical, _ = read_pcm16(Path(record["audio"]["B"]["path"]))
            if not np.array_equal(audios["B075"], historical):
                raise ValueError("B075 does not reproduce historical bridge")
            _, crops, crop_meta = lock_source_crop(source, detector)
            visual, _ = visual_embedding(np.stack(crops), scorer, "cuda", torch)
            visual = visual.numpy().astype(np.float32)
            count = min(len(crops), len(natural) // 640) - 5
            matrices, hashes = {}, {}
            for arm in ARMS:
                path = work / f"{arm}.wav"
                hashes[arm] = write_pcm16(path, audios[arm])
                emb, _, _ = audio_embedding(path, scorer, "cuda", torch)
                matrices[arm] = distance_matrix(
                    visual, emb.numpy().astype(np.float32), count
                )
                np.save(work / f"{arm}_distance.npy", matrices[arm], allow_pickle=False)
            ni = int(np.argmin(matrices["N"].astype(np.float64).mean(axis=0)))
            row = {
                "sample_id": sid,
                "source_group": record["source_group"],
                "source_video": str(source),
                "source_sha256": sha256_file(source),
                "audio_hashes": hashes,
                "construction": construction,
                "crop": crop_meta,
                "support_count": count - 30,
                "historical_B075_exact": True,
                "roundtrip_max_pcm_error": int(
                    np.max(np.abs(audios["RT"].astype(int) - natural.astype(int)))
                ),
                "scores": {arm: metrics(matrices[arm], ni) for arm in ARMS},
            }
            dump(work / "result.json", row)
            rows.append(row)
            print(
                f"PASS {len(rows)}/{len(inputs)} {sid} N={row['scores']['N']['C']:.3f} B075={row['scores']['B075']['C']:.3f}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            dump(work / "failure.json", failures[-1])
            print(f"FAIL {sid}: {exc}", flush=True)
            if isinstance(exc, RuntimeError):
                raise
        dump(
            out / "results.json",
            {
                "completed": len(rows),
                "intended": len(inputs),
                "failures": failures,
                "rows": rows,
            },
        )
    if not rows:
        raise RuntimeError("No scoreable samples")
    summary = summarize(rows)
    dump(
        out / "summary.json",
        {
            "completed": len(rows),
            "intended": len(inputs),
            "failures": failures,
            "arms": summary,
        },
    )
    lines = [
        "# 自然视频上的 Bridge 强度扫描",
        "",
        f"完成 {len(rows)}/{len(inputs)}。固定真实视频，不经过生成模型。",
        "",
        "全臂共享完整人脸轨迹及31个lag均有效的时间支持。C为曲线中位数减最小值，D为最小距离；与含边界padding的历史全局分数不直接比较。",
        "",
        "|音频|C|ΔC vs N（95% bootstrap CI）|D|D@0|D@N offset|offset变化数|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for arm, s in summary.items():
        lo, hi = s["delta_C_CI95"]
        lines.append(
            f"|{arm}|{s['C']:.3f}|{s['delta_C']:+.3f} [{lo:+.3f}, {hi:+.3f}]|{s['D']:.3f}|{s['D0']:.3f}|{s['DN']:.3f}|{s['offset_changed']}/{len(rows)}|"
        )
    lines += [
        "",
        "B025/B050/B075/B100为自然相位+不同log幅度插值强度；B100不是MFA。RT为alpha0重建。SWAP为中间两季度互换。DELAY200为前补零、尾截断的200ms延迟。",
        "",
        "置信区间按样本配对bootstrap（每条来自独立source group），探索性未校正多重比较。SyncNet变化不能单独区分声学表征变化与局部音素节奏变化，也不授权训练。",
        "",
        "逐样本分数、完整距离矩阵、构造参数、输入hash及crop支持见相邻JSON/NPY/WAV。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
