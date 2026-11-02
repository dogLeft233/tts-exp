"""Acquire a frozen, speaker-diverse HDTF diagnostic subset without scoring it.

Requires imageio-ffmpeg and Pillow. HTTP range reads avoid downloading the 5GB
mirror archive. Original ZIP members are CRC-checked by zipfile and SHA-frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
import time
import urllib.request
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MIRROR = "global-optima-research/HDTF"
REVISION = "6f9ed465f761b8378c690be4f86f090634281584"
ARCHIVE_SIZE = 5243344849
ARCHIVE_SHA256 = "482074817bc256351b873f3b1af9aa0f4709a449b0c82ba32210885583c28a23"
URL = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/videos.zip"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("negative position")
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n - 1
        if n > 2 * 1024 * 1024:
            ranges = [(pos, min(end, pos + 2 * 1024 * 1024 - 1))
                      for pos in range(start, end + 1, 2 * 1024 * 1024)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                data = b"".join(pool.map(lambda bounds: self._read_range(*bounds), ranges))
        else:
            data = self._read_range(start, end)
        self.pos += n
        self.downloaded += n
        return data

    def _read_range(self, start, end):
        n = end - start + 1
        for attempt in range(5):
            try:
                # Different range URLs prevent intermediaries reusing a cached 206.
                request = urllib.request.Request(
                    self.url + f"?range_start={start}&range_end={end}",
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("server did not honor exact HTTP range")
                    data = response.read(n + 1)
                if len(data) != n:
                    raise ValueError("incomplete HTTP range")
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(name):
    match = re.fullmatch(r"(WDA|WRA)_(.+)_(\d{3})\.mp4", Path(name).name)
    if not match:
        return None
    # Prefix is a recording category, not identity; trailing digits are takes.
    return re.sub(r"\d+$", "", match[2])


def select_candidates(entries, seed=20261031):
    by_person = {}
    for entry in sorted(entries, key=lambda x: x.filename):
        person = identity(entry.filename)
        if person:
            by_person.setdefault(person, entry)
    pools = [[x for x in by_person.values() if Path(x.filename).name.startswith(prefix)]
             for prefix in ("WDA_", "WRA_")]
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    # Alternate categories, with each person's first filename only.
    result = []
    while any(pools):
        for pool in pools:
            if pool:
                result.append(pool.pop())
    return result


def run(ffmpeg, args):
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", *args],
                            capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])
    return result


def main():
    import imageio_ffmpeg
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speakers", type=int, default=20)
    args = parser.parse_args()
    if args.speakers < 1:
        parser.error("speakers must be positive")
    root = args.output.resolve()
    for folder in ("provenance", "original_videos", "images", "audio", "clips"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    official = json.loads(fetch("https://api.github.com/repos/MRzzm/HDTF/commits/main"))["sha"]
    metadata = {}
    for category in ("WDA", "WRA", "RD"):
        for kind in ("video_url", "annotion_time", "crop_wh", "crop_ratio", "resolution"):
            name = f"{category}_{kind}.txt"
            data = fetch(f"https://raw.githubusercontent.com/MRzzm/HDTF/{official}/HDTF_dataset/{name}")
            (root / "provenance" / name).write_bytes(data)
            metadata[name] = data.decode("utf-8-sig")
    remote = RemoteFile(URL, ARCHIVE_SIZE)
    records, rejected, originals = [], [], []
    archive = zipfile.ZipFile(remote)
    inventory = [dict(name=x.filename, size=x.file_size, crc32=x.CRC) for x in archive.infolist()]
    (root / "provenance" / "archive_inventory.json").write_text(json.dumps(inventory, indent=2))
    for entry in select_candidates(archive.infolist()):
        if len(originals) == args.speakers:
            break
        stem = Path(entry.filename).stem
        person = identity(entry.filename)
        category, take, clip_index = stem.split("_")
        source_url_line = next((line for line in metadata[f"{category}_video_url.txt"].splitlines()
                                if line.split()[0].removesuffix(".mp4") == take), None)
        source_time_line = next((line for line in metadata[f"{category}_annotion_time.txt"].splitlines()
                                 if line.split()[0].removesuffix(".mp4") == take), None)
        if not source_url_line or not source_time_line:
            rejected.append(dict(member=entry.filename, reason="not mapped to official metadata"))
            continue
        original = root / "original_videos" / f"{stem}.mp4"
        if not original.exists():
            print("Downloading", stem, entry.file_size, flush=True)
            # Read to EOF verifies the ZIP member CRC; never extract arbitrary paths.
            data = archive.read(entry)
            temporary = original.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(original)
        else:
            import binascii
            if original.stat().st_size != entry.file_size or binascii.crc32(original.read_bytes()) != entry.CRC:
                raise ValueError(f"existing original fails CRC: {original}")
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(original)], capture_output=True)
        stderr = probe.stderr.decode(errors="replace")
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
        if not m or "Audio:" not in stderr:
            rejected.append(dict(member=entry.filename, reason="missing duration or audio"))
            continue
        duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        if duration < 40:
            rejected.append(dict(member=entry.filename, reason="duration below 40 seconds", duration=duration))
            continue
        original_record = dict(speaker=person, member=entry.filename, original=str(original.relative_to(root)),
                               sha256=sha(original), crc32=entry.CRC, duration_s=duration,
                               official_url_line=source_url_line, official_time_line=source_time_line,
                               official_clip_index=int(clip_index))
        originals.append(original_record)
        for index, fraction in enumerate((0.2, 0.6), 1):
            # Integer 25fps starts; nonoverlapping ten-second windows.
            start = int((duration - 10) * fraction * 25) / 25
            sample = f"{stem}_seg{index}"
            audio = root / "audio" / f"{sample}.wav"
            image = root / "images" / f"{sample}.png"
            clip = root / "clips" / f"{sample}.mp4"
            run(ffmpeg, ["-v", "error", "-y", "-ss", str(start), "-i", str(original), "-t", "10",
                         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)])
            run(ffmpeg, ["-v", "error", "-y", "-ss", str(start), "-i", str(original),
                         "-frames:v", "1", str(image)])
            run(ffmpeg, ["-v", "error", "-y", "-ss", str(start), "-i", str(original), "-t", "10",
                         "-map", "0:v:0", "-map", "0:a:0", "-r", "25", "-c:v", "libx264", "-crf", "18",
                         "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", str(clip)])
            with wave.open(str(audio)) as wav:
                assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()) == (1, 16000, 2, 160000)
            with Image.open(image) as im:
                im.load()
                dimensions = list(im.size)
            # Full clip decode, including the audio stream, rejects damaged output.
            run(ffmpeg, ["-v", "error", "-i", str(clip), "-f", "null", "-"])
            records.append(dict(id=sample, speaker=person, source_member=entry.filename, start_s=start, duration_s=10,
                                image_size=dimensions, transcript=None, transcript_status="NOT_PROVIDED_NOT_ASR",
                                files={kind: dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)
                                       for kind, path in (("audio", audio), ("image", image), ("video", clip))}))
        print("Prepared", len(originals), person, flush=True)
        payload = dict(status="RUNNING", mirror=MIRROR, revision=REVISION, archive_url=URL,
                       archive_size=ARCHIVE_SIZE, upstream_archive_sha256=ARCHIVE_SHA256,
                       whole_archive_sha_verified=False, member_crc_verified=True, official_commit=official,
                       selection="seed20261031; named WDA/WRA identities deduplicated across prefixes/takes; alternating categories; first filename per identity; >=40s; no scores",
                       audio_processing="mono16kPCM16; no gain normalization; source videos retained",
                       window_rule="10s at floor25fps((source_duration-10)*[.2,.6]); may cut words; not final sentence-level TTS inputs",
                       strict_unseen=False, overlap_note="HDTF used in TalkLikeYou training; diagnostic domain comparison only",
                       originals=originals, records=records, rejected=rejected,
                       downloaded_range_bytes=remote.downloaded)
        (root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if len(originals) != args.speakers:
        raise RuntimeError("not enough eligible speakers")
    payload["status"] = "COMPLETE"
    (root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("COMPLETE", len(originals), "speakers", len(records), "pairs", flush=True)


if __name__ == "__main__":
    main()
