#!/usr/bin/env python
"""Video-quality metric suite for TFG output comparison.

Purpose
-------
Sync-C measures lip-sync only.  This script scores the *same* generated videos
with the standard talking-head / video-generation metric family so that a
Sync-C gain can be checked against visual quality:

  * no-reference IQA   : NIQE, BRISQUE, CLIP-IQA, MUSIQ     (pyiqa)
  * full-reference     : LPIPS / SSIM / PSNR vs the clip's own first frame
  * temporal           : LPIPS and L1 between consecutive sampled frames
  * sharpness / motion : Laplacian variance, frame-difference energy
  * distribution       : FID (InceptionV3 pool3), KID, FVD (r3d_18 features)

Everything is computed per video, then aggregated per group, then compared
pairwise.  Distribution metrics need only unpaired frame/clip samples, so they
are unaffected by the duration mismatch between natural and TTS audio.

Notes on comparability
----------------------
* FID uses torchvision InceptionV3 (IMAGENET1K_V1) pool3 features.  Absolute
  values are NOT comparable to published FID baselines (which use the
  `pt_inception-2015-12-05` port); only relative comparisons inside one run of
  this script are meaningful.
* FVD uses torchvision r3d_18 (Kinetics-400) pooled features as a 3-D CNN
  proxy for the I3D feature bank of the original FVD.  Same caveat.
* LPIPS/SSIM/PSNR are computed at 256x256 so that all groups share one scale.
* NIQE/BRISQUE honour their native resolution dependence; frames are resized
  with aspect preserved and centre-cropped to 512x512 first.

Usage
-----
    python scripts/41_video_quality_metrics.py --manifest m.json --out outdir
    python scripts/41_video_quality_metrics.py --manifest m.json --out outdir \
        --shard 0/4 --no-distribution

Manifest schema
---------------
{
  "compare": [["group_a", "group_b"], ...],       # optional
  "entries": [
     {"group": "leaptalk/natural", "item": "1", "video": "path/to/1.mp4"},
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------


def sample_indices(total: int, want: int) -> np.ndarray:
    """Uniformly spaced frame indices, always including first and last."""
    if total <= 0:
        return np.zeros(0, dtype=np.int64)
    if want <= 0 or total <= want:
        return np.arange(total, dtype=np.int64)
    idx = np.linspace(0, total - 1, want)
    return np.unique(np.round(idx).astype(np.int64))


def read_video_frames(path: Path, want: int) -> tuple[list[np.ndarray], float, int]:
    """Decode a video, returning uniformly sampled RGB frames (HxWx3 uint8)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    idx = sample_indices(total, want)
    frames: list[np.ndarray] = []
    if len(idx) == 0:  # frame count metadata missing: read sequentially
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if want > 0 and len(frames) >= want:
                break
        cap.release()
        return frames, fps, len(frames)
    wanted = set(int(i) for i in idx)
    last = int(idx[-1])
    pos = 0
    while pos <= last:
        ok = cap.grab()
        if not ok:
            break
        if pos in wanted:
            ok2, frame = cap.retrieve()
            if ok2:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pos += 1
    cap.release()
    return frames, fps, total


def resize_short_center(img: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize (short side -> size) + centre crop."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    top = (nh - size) // 2
    left = (nw - size) // 2
    return out[top:top + size, left:left + size]


def to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Plain resize to size x size (used for fixed-aspect generated frames)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# distribution metrics
# ----------------------------------------------------------------------------


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two feature banks (n, d)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False).astype(np.float64)
    s2 = np.cov(f2, rowvar=False).astype(np.float64)
    d = mu1.shape[0]
    s1 = s1.reshape(d, d) + np.eye(d) * eps
    s2 = s2.reshape(d, d) + np.eye(d) * eps
    diff = mu1.astype(np.float64) - mu2.astype(np.float64)
    # symmetric PSD square root of s1@s2 via eigendecomposition
    covmean = np.eye(d)
    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(s1 @ s2)
        if not np.all(np.isfinite(covmean)):
            raise ValueError("non-finite sqrtm")
        covmean = np.real(covmean)
    except Exception:
        w, v = np.linalg.eigh(s1 @ s2)
        w = np.clip(np.real(w), 0.0, None)
        covmean = np.real(v @ np.diag(np.sqrt(w)) @ np.linalg.inv(v))
    fid = float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    return max(fid, 0.0)


def polynomial_mmd2(x: np.ndarray, y: np.ndarray, degree: int = 3,
                    gamma: float | None = None) -> float:
    """Unbiased estimator of MMD^2 with a polynomial kernel (KID)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    d = x.shape[1]
    gamma = 1.0 / d if gamma is None else gamma
    kxx = (gamma * (x @ x.T) + 1.0) ** degree
    kyy = (gamma * (y @ y.T) + 1.0) ** degree
    kxy = (gamma * (x @ y.T) + 1.0) ** degree
    sxx = (kxx.sum() - np.trace(kxx)) / (nx * (nx - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (ny * (ny - 1))
    return float(sxx + syy - 2.0 * kxy.mean())


def kid_score(f1: np.ndarray, f2: np.ndarray, subsets: int = 50,
              subset_size: int = 100, seed: int = 20261021) -> dict:
    """KID: mean +- std over random subsets (unbiased MMD^2, x1e3 convention)."""
    rng = np.random.default_rng(seed)
    n = min(len(f1), len(f2), subset_size)
    if n < 5:
        return {"kid_x1000": float("nan"), "kid_std_x1000": float("nan")}
    vals = []
    for _ in range(subsets):
        i = rng.choice(len(f1), n, replace=False)
        j = rng.choice(len(f2), n, replace=False)
        vals.append(polynomial_mmd2(f1[i], f2[j]))
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "kid_x1000": float(np.nanmean(vals) * 1000.0),
        "kid_std_x1000": float(np.nanstd(vals) * 1000.0),
    }


# ----------------------------------------------------------------------------
# model wrappers
# ----------------------------------------------------------------------------


class MetricModels:
    """Lazily built torch models; everything runs on CUDA when available."""

    def __init__(self, device: str, want_iqa: bool = True, want_dist: bool = True,
                 iqa_metrics: tuple[str, ...] = ("niqe", "brisque", "clipiqa")):
        import torch

        self.torch = torch
        self.device = device
        self.lpips_model = None
        self.inception = None
        self.r3d = None
        self.iqa = {}
        self.iqa_names = iqa_metrics
        if want_dist:
            self._build_dist()
        if want_iqa:
            self._build_iqa()

    # -- LPIPS -------------------------------------------------------------
    def get_lpips(self):
        if self.lpips_model is None:
            import lpips as lpips_mod

            model = lpips_mod.LPIPS(net="alex", verbose=False).to(self.device).eval()
            self.lpips_model = model
        return self.lpips_model

    # -- Inception / r3d_18 -------------------------------------------------
    def _build_dist(self):
        from torchvision.models import Inception_V3_Weights, inception_v3
        from torchvision.models.video import R3D_18_Weights, r3d_18

        inc = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           aux_logits=True, transform_input=False)
        inc.fc = self.torch.nn.Identity()
        self.inception = inc.to(self.device).eval()
        r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        r3d.fc = self.torch.nn.Identity()
        self.r3d = r3d.to(self.device).eval()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.inc_mean = self.torch.from_numpy(mean).view(1, 3, 1, 1).to(self.device)
        self.inc_std = self.torch.from_numpy(std).view(1, 3, 1, 1).to(self.device)
        vmean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        vstd = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        self.v_mean = self.torch.from_numpy(vmean).view(1, 3, 1, 1, 1).to(self.device)
        self.v_std = self.torch.from_numpy(vstd).view(1, 3, 1, 1, 1).to(self.device)

    def _build_iqa(self):
        try:
            import pyiqa
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] pyiqa unavailable ({exc}); NR-IQA metrics skipped", flush=True)
            return
        for name in self.iqa_names:
            try:
                self.iqa[name] = pyiqa.create_metric(name, device=self.device)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] pyiqa metric {name} unavailable ({exc})", flush=True)

    # -- feature extraction -------------------------------------------------
    def inception_features(self, frames_rgb: list[np.ndarray], batch: int = 32) -> np.ndarray:
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames_rgb), batch):
                chunk = frames_rgb[i:i + batch]
                arr = np.stack([to_square(f, 299).transpose(2, 0, 1) for f in chunk])
                t = torch.from_numpy(arr).float().to(self.device) / 255.0
                t = (t - self.inc_mean) / self.inc_std
                out = self.inception(t)
                feats.append(out.reshape(len(chunk), -1).cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.zeros((0, 2048), np.float32)

    def fvd_features(self, frames_rgb: list[np.ndarray], clip_len: int = 16,
                     stride: int = 8, batch: int = 8) -> np.ndarray:
        torch = self.torch
        if len(frames_rgb) < clip_len:
            return np.zeros((0, 512), np.float32)
        clips = []
        for start in range(0, len(frames_rgb) - clip_len + 1, stride):
            clip = frames_rgb[start:start + clip_len]
            arr = np.stack([to_square(f, 112).transpose(2, 0, 1) for f in clip])
            clips.append(arr)
        feats = []
        with torch.no_grad():
            for i in range(0, len(clips), batch):
                arr = np.stack(clips[i:i + batch])  # (b, t, c, h, w)
                t = torch.from_numpy(arr).float().to(self.device) / 255.0
                t = t.permute(0, 2, 1, 3, 4)  # (b, c, t, h, w)
                t = (t - self.v_mean) / self.v_std
                out = self.r3d(t)
                feats.append(out.reshape(len(arr), -1).cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.zeros((0, 512), np.float32)

    # -- no-reference IQA ---------------------------------------------------
    def iqa_scores(self, frames_rgb: list[np.ndarray]) -> dict:
        torch = self.torch
        if not self.iqa:
            return {}
        arr = np.stack([resize_short_center(f, 512).transpose(2, 0, 1) for f in frames_rgb])
        t = torch.from_numpy(arr).float().to(self.device) / 255.0
        out: dict[str, float] = {}
        with torch.no_grad():
            for name, metric in self.iqa.items():
                try:
                    vals = []
                    for i in range(0, len(t), 16):
                        v = metric(t[i:i + 16])
                        vals.append(v.reshape(-1).cpu().numpy())
                    v = np.concatenate(vals)
                    v = v[np.isfinite(v)]
                    out[name] = float(v.mean()) if v.size else float("nan")
                except Exception as exc:  # pragma: no cover
                    print(f"[warn] {name} failed: {exc}", flush=True)
                    out[name] = float("nan")
        return out

    # -- paired perceptual metrics -----------------------------------------
    def lpips_pairs(self, a: list[np.ndarray], b: list[np.ndarray],
                    batch: int = 16) -> np.ndarray:
        torch = self.torch
        model = self.get_lpips()
        vals = []
        with torch.no_grad():
            for i in range(0, len(a), batch):
                aa = np.stack([to_square(f, 256).transpose(2, 0, 1) for f in a[i:i + batch]])
                bb = np.stack([to_square(f, 256).transpose(2, 0, 1) for f in b[i:i + batch]])
                ta = torch.from_numpy(aa).float().to(self.device) / 127.5 - 1.0
                tb = torch.from_numpy(bb).float().to(self.device) / 127.5 - 1.0
                vals.append(model(ta, tb).reshape(-1).cpu().numpy())
        return np.concatenate(vals) if vals else np.zeros(0)


def ssim_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Grayscale SSIM + PSNR for two uint8 RGB frames of equal size."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    sa = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    ssim = (((2 * mu_a * mu_b + c1) * (2 * sab + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2)))
    mse = float(np.mean((ga - gb) ** 2))
    psnr = 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return float(ssim.mean()), psnr


# ----------------------------------------------------------------------------
# per-video driver
# ----------------------------------------------------------------------------


def score_video(path: Path, models: MetricModels, max_frames: int,
                want_dist: bool, iqa_frames: int = 24) -> dict:
    frames, fps, total = read_video_frames(path, max_frames)
    row: dict = {
        "video": str(path),
        "fps": fps,
        "n_frames_total": total,
        "n_frames_sampled": len(frames),
        "duration_s": (total / fps) if fps else float("nan"),
    }
    if len(frames) < 3:
        row["error"] = "too_few_frames"
        return row

    # sharpness / motion on native resolution grayscale
    lap, diffs = [], []
    prev = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        lap.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g.astype(np.float32) - prev))))
        prev = g.astype(np.float32)
    row["sharpness_lapvar"] = float(np.mean(lap))
    row["frame_diff_l1"] = float(np.mean(diffs)) if diffs else float("nan")

    # temporal + reference metrics
    model_pairs = models.lpips_pairs(frames[1:], frames[:-1])
    row["lpips_temporal"] = float(np.mean(model_pairs)) if len(model_pairs) else float("nan")
    ref_pairs = models.lpips_pairs(frames[1:], [frames[0]] * (len(frames) - 1))
    row["lpips_vs_first"] = float(np.mean(ref_pairs)) if len(ref_pairs) else float("nan")
    ss, pp = [], []
    for f in frames[1:][::2]:
        s, p = ssim_psnr(to_square(f, 256), to_square(frames[0], 256))
        ss.append(s)
        pp.append(p)
    row["ssim_vs_first"] = float(np.mean(ss)) if ss else float("nan")
    row["psnr_vs_first"] = float(np.mean(pp)) if pp else float("nan")

    # no-reference IQA on a subsample (expensive per image, cheap in aggregate)
    iqa_idx = sample_indices(len(frames), iqa_frames)
    row.update(models.iqa_scores([frames[i] for i in iqa_idx]))

    if want_dist:
        row["_fid_feats"] = models.inception_features(frames)
        row["_fvd_feats"] = models.fvd_features(frames)
    return row


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SCALAR_KEYS = [
    "sharpness_lapvar", "frame_diff_l1", "lpips_temporal", "lpips_vs_first",
    "ssim_vs_first", "psnr_vs_first", "niqe", "brisque", "clipiqa", "musiq",
    "n_frames_sampled", "duration_s",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="base directory for relative video paths in the manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--shard", default="0/1", help="i/n for parallel shards")
    ap.add_argument("--no-distribution", action="store_true")
    ap.add_argument("--no-iqa", action="store_true")
    ap.add_argument("--iqa-metrics", default="niqe,brisque,clipiqa",
                    help="comma list of pyiqa metrics (musiq needs a big download)")
    ap.add_argument("--iqa-frames", type=int, default=24,
                    help="frames per video used for NR-IQA")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    entries = manifest["entries"]
    if args.limit:
        entries = entries[:args.limit]

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    entries = [e for i, e in enumerate(entries) if i % shard_n == shard_i]

    models = MetricModels(args.device, want_iqa=not args.no_iqa,
                          want_dist=not args.no_distribution,
                          iqa_metrics=tuple(m.strip() for m in args.iqa_metrics.split(",") if m.strip()))

    name = f"per_video_shard{shard_i}of{shard_n}.jsonl"
    rows = []
    t0 = time.time()
    with open(out / name, "w", encoding="utf-8") as fh:
        for k, entry in enumerate(entries):
            video = Path(entry["video"])
            if not video.is_absolute():
                video = args.repo_root / video
            try:
                row = score_video(video, models, args.max_frames,
                                  not args.no_distribution, args.iqa_frames)
            except Exception as exc:
                row = {"video": str(video), "error": f"{type(exc).__name__}: {exc}"}
            row.update({kk: entry.get(kk) for kk in ("group", "item") if kk in entry})
            # persist feature banks next to the jsonl so aggregation needs no re-decode
            tag = f"{entry['group'].replace('/', '__')}__{entry.get('item', k)}"
            for fkey, fname in (("_fid_feats", "fid"), ("_fvd_feats", "fvd")):
                feats = row.pop(fkey, None)
                if feats is not None and len(feats):
                    dest = out / "feats" / fname / f"{tag}.npy"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.save(dest, feats.astype(np.float32))
                    row[f"{fname}_feat_file"] = str(dest)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"[{k + 1}/{len(entries)}] {el:.0f}s "
                      f"({(k + 1) / max(el, 1e-9):.2f} vid/s) {video.name}", flush=True)
    print(f"done {len(entries)} videos in {time.time() - t0:.1f}s", flush=True)

    # quick per-shard summary (full aggregation happens in 42_*)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("group", "?"), []).append(r)
    summary = {}
    for g, rs in groups.items():
        summary[g] = {}
        for key in SCALAR_KEYS:
            vals = np.array([r.get(key, np.nan) for r in rs], dtype=np.float64)
            if np.isfinite(vals).any():
                m, lo, hi = bootstrap_mean_ci(vals)
                summary[g][key] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                                   "n": int(np.isfinite(vals).sum())}
    (out / f"summary_shard{shard_i}of{shard_n}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
