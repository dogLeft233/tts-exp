#!/usr/bin/env python3
"""download_vfhq_v2.py - 50 VFHQ samples: 45 liveface-aligned + 5 fallback.

Writes data/dataset_samples/vfhq/50/align.json mapping stem -> {aligned, yt_id, start_s, dur_s}.
"""
import json
import re
import subprocess
from pathlib import Path

import requests

REPO = Path("/mnt/e/Documents/tts-audio/tts-exp")
OUT = REPO / "data/dataset_samples/vfhq/50"
MIRROR = "https://hf-mirror.com/datasets"
PROXY = "http://127.0.0.1:7890"
FPS = 25

# (yt_id, clip) -> frames from liveface intersection, smallest 42 not yet downloaded
ALIGNED = [
    "2IwMM_VStsU_C1", "3qmhckbctEU_C2", "3y4NFbQ8rmc_C0", "18M9m31CIGM_C1",
    "0odeKs9H1Rg_C0", "0CQ6pdruJtE_C0", "2w8Yc554je0_C1", "1zEiz3z6470_C2",
    "32abllGf77g_C1", "3LVsNYeHML8_C2", "1aWLEMCe2S0_C1",
    "2BQlze8jMgQ_C1", "1G7bSLY3zj4_C1", "0H5hQCiVQpw_C0",
    "1dWF3vXRzo4_C0", "33rikgJsFmk_C2", "37ofnzpPnU4_C0",
    "0OV9duTbqYc_C1", "18hAMXc8gXA_C1", "0YqjNguUIHs_C2",
    "1j2_OhyTkjo_C0", "3G4N4OoxwuA_C1", "18gOAb4FTfw_C0", "2AZJp-RgPDY_C0",
    "2JflI7blQEk_C2", "2jkJmmEorzY_C1", "405lZLauvkY_C2", "2krK3bKwi4I_C2",
    "2jkJmmEorzY_C0", "1ibFO9lZnlI_C0", "0ufAYskIGmY_C0", "094y1Z2wpJg_C0",
    "3Bu5SWIjN-I_C2", "1i7yGiQp0WM_C1", "3r5zdeXOzc0_C2", "2_KYkHxV3CE_C2",
    "2U4Bp4xvWlA_C2",
]
# 5 fallback: small, YouTube-tested
FALLBACK = ["AQEr_KL4DKc_C0", "cvHBji1HgJ8_C0", "BBTCTprA6qU_C1", "8NwObke6Da4_C1", "amuHAliYofA_C1", "5DOHKfiU1hk_C0", "AvoObZZutvs_C2", "4iXDuMdHxtc_C0", "5DOHKfiU1hk_C2", "CYDuO7aJE_U_C1"]
# already downloaded aligned (reuse)
HAVE = {"0-9M9PcjsCc_C0": None, "0GRnvFxj8IU_C1": None, "0b6yn5VNuC8_C1": None}


def liveface_frames() -> dict:
    url = "https://huggingface.co/api/datasets/hyb1124/liveface_vfhq/tree/main/videos?recursive=true"
    items = requests.get(url, timeout=30).json()
    pat = re.compile(r"Clip\+(.+)\+P\d+\+C(\d+)\+F(\d+)-(\d+)\.mp4")
    out = {}
    for it in items:
        m = pat.search(it["path"])
        if m:
            out[(m.group(1), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)))
    return out


def dl(url: str, dest: Path, timeout: int = 600) -> None:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = liveface_frames()
    plan = []
    for name in ALIGNED:
        yid, c = name.rsplit("_", 1)
        plan.append((name, yid, int(c[1:]), True))
    for name in HAVE:
        yid, c = name.rsplit("_", 1)
        plan.append((name, yid, int(c[1:]), True))
    for name in FALLBACK:
        yid, c = name.rsplit("_", 1)
        plan.append((name, yid, int(c[1:]), False))

    align = {}
    yt_cache: dict[str, Path] = {}
    for i, (name, yid, clip, aligned) in enumerate(plan, 1):
        stem = f"v50_{name}"
        video_out = OUT / f"{stem}.mp4"
        if not video_out.exists():
            try:
                dl(f"{MIRROR}/saeed-5959/vfhq/resolve/main/video_original/{name}.mp4", video_out)
            except Exception as e:
                print(f"[v2] {i}/50 video FAIL {name}: {e}")
                continue
            print(f"[v2] {i}/50 video {name} ({video_out.stat().st_size//1048576}MB)")
        if yid not in yt_cache:
            yt_path = OUT / f"{yid}.orig"
            if not yt_path.exists():
                ok = False
                for attempt in range(3):
                    try:
                        subprocess.run(
                            ["yt-dlp", "--proxy", PROXY, "--js-runtimes", "none",
                             "-f", "ba/b", "-o", str(yt_path),
                             f"https://www.youtube.com/watch?v={yid}"],
                            check=True, capture_output=True, timeout=600)
                        ok = True
                        break
                    except subprocess.CalledProcessError as e:
                        print(f"[v2] yt-dlp {yid} attempt{attempt+1}: {e.stderr.decode()[-120:]}")
                if not ok:
                    yt_cache[yid] = None
                    print(f"[v2] {i}/50 SKIP audio {name}")
                    continue
            yt_cache[yid] = yt_path
        wav_out = OUT / f"{stem}_audio.wav"
        if not wav_out.exists():
            seg = frames.get((yid, clip))
            if aligned and seg:
                start = seg[0] / FPS
                dur = (seg[1] - seg[0]) / FPS
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                     "-i", str(yt_cache[yid]), "-ar", "16000", "-ac", "1", str(wav_out)],
                    check=True, capture_output=True, timeout=120)
                align[stem] = {"aligned": True, "yt_id": yid, "start_s": round(start, 2), "dur_s": round(dur, 2)}
                print(f"[v2] {i}/50 audio {name} aligned @{start:.1f}s {dur:.1f}s")
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-t", "10", "-i", str(yt_cache[yid]),
                     "-ar", "16000", "-ac", "1", str(wav_out)],
                    check=True, capture_output=True, timeout=120)
                align[stem] = {"aligned": False, "yt_id": yid, "start_s": 0.0, "dur_s": 10.0}
                print(f"[v2] {i}/50 audio {name} FALLBACK first-10s")
    (OUT / "align.json").write_text(json.dumps(align, indent=2, ensure_ascii=False))
    n_align = sum(1 for v in align.values() if v["aligned"])
    print(f"[v2] done: {len(list(OUT.glob('*.mp4')))} videos, {len(list(OUT.glob('*_audio.wav')))} wavs, aligned={n_align}/{len(align)}")


if __name__ == "__main__":
    main()
