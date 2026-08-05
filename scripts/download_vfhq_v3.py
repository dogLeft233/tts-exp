#!/usr/bin/env python3
"""download_vfhq_v3.py - Rebuild VFHQ 50 as ALL strictly-aligned samples.

Source: liveface_vfhq clips (video, no audio) + YouTube original audio cut to the
clip's frame range (F@25fps). Output in data/dataset_samples/vfhq/50aligned/.
"""
import json
import re
import subprocess
from pathlib import Path

import requests

REPO = Path("/mnt/e/Documents/tts-audio/tts-exp")
OUT = REPO / "data/dataset_samples/vfhq/50aligned"
MIRROR = "https://hf-mirror.com/datasets"
PROXY = "http://127.0.0.1:7890"
FPS = 25
N = 50

# yt_ids already covered by the previous aligned 39 (skip them)
SKIP = {"0-9M9PcjsCc", "0GRnvFxj8IU", "0b6yn5VNuC8"}


def liveface_clips() -> list[tuple[str, int, int, int, int, str]]:
    url = "https://huggingface.co/api/datasets/hyb1124/liveface_vfhq/tree/main/videos?recursive=true"
    items = requests.get(url, timeout=60).json()
    pat = re.compile(r"Clip\+(.+)\+P\d+\+C(\d+)\+F(\d+)-(\d+)\.mp4")
    out = []
    for i in items:
        if i["type"] != "file" or not i["path"].endswith(".mp4"):
            continue
        m = pat.search(i["path"])
        if m:
            yid, clip, s, e = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if yid in SKIP:
                continue
            out.append((yid, clip, s, e, i["size"], i["path"]))
    out.sort(key=lambda x: x[4])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = liveface_clips()
    print(f"[v3] candidates: {len(clips)}, need {N}")

    chosen = clips[: N * 3]  # buffer for yt-dlp failures
    done, failed = 0, []
    yt_cache: dict[str, Path] = {}
    for yid, clip, s, e, size, path in chosen:
        if done >= N:
            break
        stem = f"v50_{yid}_C{clip}"
        video_out = OUT / f"{stem}.mp4"
        if not video_out.exists():
            try:
                r = requests.get(f"{MIRROR}/hyb1124/liveface_vfhq/resolve/main/{path}",
                                 timeout=300)
                r.raise_for_status()
                video_out.write_bytes(r.content)
            except Exception as ex:
                print(f"[v3] video FAIL {stem}: {ex}")
                continue
        if yid not in yt_cache:
            yt_path = OUT / f"{yid}.orig"
            if not yt_path.exists():
                ok = False
                for attempt in range(4):
                    try:
                        subprocess.run(
                            ["yt-dlp", "--proxy", PROXY, "--js-runtimes", "none",
                             "-f", "ba/b", "-o", str(yt_path),
                             f"https://www.youtube.com/watch?v={yid}"],
                            check=True, capture_output=True, timeout=600)
                        ok = True
                        break
                    except subprocess.CalledProcessError as ex:
                        print(f"[v3] yt-dlp {yid} a{attempt+1}: {ex.stderr.decode()[-100:]}")
                        subprocess.run(["sleep", "5"], check=True)
                if not ok:
                    failed.append(stem)
                    yt_cache[yid] = None
                    print(f"[v3] SKIP audio {stem} (yt failed)")
                    continue
            yt_cache[yid] = yt_path
        if yt_cache[yid] is None:
            continue
        wav_out = OUT / f"{stem}_audio.wav"
        if not wav_out.exists():
            start, dur = s / FPS, (e - s) / FPS
            # guard: clip must fit inside the downloaded audio
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(yt_cache[yid])],
                capture_output=True, text=True)
            try:
                audio_len = float(probe.stdout.strip())
            except ValueError:
                audio_len = 0.0
            if start + dur > audio_len + 0.5:
                print(f"[v3] SKIP {stem}: clip@{start:.1f}s beyond audio {audio_len:.1f}s")
                yt_cache[yid] = None
                (OUT / f"{yid}.orig").unlink(missing_ok=True)
                continue
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                     "-i", str(yt_cache[yid]), "-ar", "16000", "-ac", "1", str(wav_out)],
                    check=True, capture_output=True, timeout=120)
            except subprocess.CalledProcessError as ex:
                print(f"[v3] cut FAIL {stem}: {ex.stderr.decode()[-100:]}")
                failed.append(stem)
                continue
        done += 1
        if "start" in locals():
            print(f"[v3] {done}/{N} {stem} aligned @{start:.1f}s {dur:.1f}s")
        else:
            print(f"[v3] {done}/{N} {stem} (cached)")

    # manifest of what got built
    entries = {}
    for yid, clip, s, e, size, path in chosen:
        stem = f"v50_{yid}_C{clip}"
        v = OUT / f"{stem}.mp4"
        w = OUT / f"{stem}_audio.wav"
        if v.exists() and w.exists():
            entries[stem] = {"aligned": True, "yt_id": yid,
                             "start_s": round(s / FPS, 2), "dur_s": round((e - s) / FPS, 2)}
    (OUT / "align.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    print(f"[v3] done: {len(entries)} aligned, failed: {len(failed)}")


if __name__ == "__main__":
    main()
