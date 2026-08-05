#!/usr/bin/env python3
"""download_vfhq_samples.py - Fetch 50 VFHQ samples (video + aligned YouTube audio).

Pipeline per sample:
  1. HF video (saeed-5959/vfhq/video_original/<name>) via hf-mirror
  2. YouTube audio via yt-dlp (http proxy 127.0.0.1:7890), cached per video ID
  3. Alignment: look up liveface_vfhq clip "Clip+<ID>+P*+C*+F<start>-<end>" for the
     same ID+C; convert frames to seconds @25fps and cut the audio; fallback: first 10s.
Output: data/dataset_samples/vfhq/50/{stem}.mp4 + {stem}_audio.wav + {stem}_yt.{ext}
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests

REPO = Path("/mnt/e/Documents/tts-audio/tts-exp")
OUT = REPO / "data/dataset_samples/vfhq/50"
HF_API = "https://huggingface.co/api/datasets"
MIRROR = "https://hf-mirror.com/datasets"
PROXY = "http://127.0.0.1:7890"
FPS = 25

NAMES = """AQEr_KL4DKc_C0 cvHBji1HgJ8_C0 BBTCTprA6qU_C1 8NwObke6Da4_C1 amuHAliYofA_C1
5DOHKfiU1hk_C0 AvoObZZutvs_C2 4iXDuMdHxtc_C0 CRdPsa_SPlA_C1 buRmXeNJsmw_C0
aQPr81dq7Uc_C2 CYDuO7aJE_U_C1 a3m2u5x1giY_C0 2IwMM_VStsU_C1 bGZ-fjfldKE_C1
3qmhckbctEU_C2 aFXLRiScIPM_C1 cZqE0y-PlbU_C1 bEgMT-F8npw_C2 5FD1MV-DK3E_C2
DcJ4RCFlNhg_C0 AzmRt9hJ7lA_C2 5k-1yI9nxPM_C0 3y4NFbQ8rmc_C0 78h0eBipfz0_C2
_4S5R_LPNSc_C2 9Pplbkak_8o_C0 BhJTxVjU9AU_C2 95XQ2K78dSk_C0 90UUm0v1YCg_C1
18M9m31CIGM_C1 BV3qiie6-L8_C0 C2GmjWeE_DA_C2 CVKWRCEND3E_C2 a3iOHyx0oUs_C0
_BH0a6hi0nI_C0 cW1t4ioaUbg_C1 5E8S-dZuWe4_C2 8nTKnWLhzJU_C1 cYQnmdNDlK4_C0
0odeKs9H1Rg_C0 BRGO7k4OH8E_C0 C0N9XJ-rsXI_C0 AxG6UzvdkxY_C2 DB8ROEyQal0_C0
5XQe8UwGHog_C0 ahkCJ3RLyMk_C0 C1N98UKycA4_C1 Ap4Y-Wwa6bI_C0 C92ZwOyg_5g_C0""".split()


def yt_id(name: str) -> str:
    return name.rsplit("_", 1)[0]


def fetch_liveface_frames() -> dict:
    """Map (yt_id, clip) -> (start_frame, end_frame) from liveface_vfhq filenames."""
    url = f"{HF_API}/hyb1124/liveface_vfhq/tree/main/videos?recursive=true"
    items = requests.get(url, timeout=30).json()
    out = {}
    pat = re.compile(r"Clip\+(.+)\+P(\d+)\+C(\d+)\+F(\d+)-(\d+)\.mp4")
    for it in items:
        m = pat.search(it["path"])
        if m:
            out[(m.group(1), int(m.group(3)))] = (int(m.group(4)), int(m.group(5)))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = fetch_liveface_frames()
    print(f"[v50] liveface frame map: {len(frames)} clips")

    yt_cache: dict[str, Path] = {}
    for i, name in enumerate(NAMES, 1):
        stem = f"v50_{name}"
        video_out = OUT / f"{stem}.mp4"
        # 1) HF video
        if not video_out.exists():
            url = f"{MIRROR}/saeed-5959/vfhq/resolve/main/video_original/{name}.mp4"
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            video_out.write_bytes(r.content)
            print(f"[v50] {i}/50 video {name} ({len(r.content)//1048576}MB)")
        # 2) YouTube audio (cache per yt id)
        yid = yt_id(name)
        if yid not in yt_cache:
            yt_path = OUT / f"{yid}.orig"
            if not yt_path.exists():
                for attempt in range(3):
                    try:
                        subprocess.run(
                            ["yt-dlp", "--proxy", PROXY, "--js-runtimes", "none",
                             "-f", "ba/b", "-o", str(yt_path),
                             f"https://www.youtube.com/watch?v={yid}"],
                            check=True, capture_output=True, timeout=600)
                        break
                    except subprocess.CalledProcessError as e:
                        print(f"[v50] yt-dlp {yid} attempt {attempt+1} failed: "
                              f"{e.stderr.decode()[-200:]}")
                        if attempt == 2:
                            print(f"[v50] SKIP {yid} (yt-dlp failed)")
                            yt_cache[yid] = None
                            continue
                else:
                    continue
            yt_cache[yid] = yt_path
        if yt_cache[yid] is None:
            continue
        # 3) cut aligned audio
        wav_out = OUT / f"{stem}_audio.wav"
        if not wav_out.exists():
            key = (yid, int(name.rsplit("_", 1)[1][1:]))  # C0 -> 0
            seg = frames.get(key)
            if seg:
                start = seg[0] / FPS
                dur = (seg[1] - seg[0]) / FPS
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                     "-i", str(yt_cache[yid]), "-ar", "16000", "-ac", "1", str(wav_out)],
                    check=True, capture_output=True, timeout=120)
                print(f"[v50] {i}/50 audio {name} aligned @{start:.1f}s {dur:.1f}s")
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-t", "10", "-i", str(yt_cache[yid]),
                     "-ar", "16000", "-ac", "1", str(wav_out)],
                    check=True, capture_output=True, timeout=120)
                print(f"[v50] {i}/50 audio {name} fallback first-10s")
    print(f"[v50] done: {len(list(OUT.glob('*.mp4')))} videos, {len(list(OUT.glob('*_audio.wav')))} wavs")


if __name__ == "__main__":
    main()
