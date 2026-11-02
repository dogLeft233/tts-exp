"""Bounded Chinese-LiPS raw-FACE acquisition; no processed 96px mouth upsampling.

The author's HF upload lacks the final train.zip disk. We index available local
headers, retaining disk/byte provenance and explicitly reporting prefix coverage.
Raw FACE MP4s already contain audio. Extraction uses its common container clock;
this must be audited against WAV before adopting it as the training positive.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import shlex
import tarfile
import time
import zipfile
import zlib

DATASET_REVISION = "db96948538811029011eee44602438a26710ecd9"
BASE = f"https://huggingface.co/datasets/BAAI/Chinese-LiPS/resolve/{DATASET_REVISION}/"
DISK_SIZE = 8589934592


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


class HTTPRange(io.RawIOBase):
    """Strict range reads: a server returning a full archive is never accepted."""
    def __init__(self, url, size=None):
        import requests
        self.session = requests.Session()
        self.url = url.replace("/Chinese-LiPS/resolve/main/", f"/Chinese-LiPS/resolve/{DATASET_REVISION}/")
        self.signed_url = None
        self.size = size
        self.pos = 0
        self.bytes_received = 0

    def range(self, start, size):
        if size <= 0:
            return b""
        last_error = None
        for attempt in range(5):
            try:
                r = self.session.get(self.signed_url or self.url,
                                     headers={"Range": f"bytes={start}-{start+size-1}"},
                                     timeout=(20, 90), stream=True)
                if r.status_code in (401, 403) and self.signed_url:
                    self.signed_url = None
                    continue
                r.raise_for_status()
                content_range = r.headers.get("Content-Range", "")
                if r.status_code != 206 or not content_range.startswith(f"bytes {start}-"):
                    r.close()
                    raise ValueError(f"Range not honored: {r.status_code} {content_range}")
                self.signed_url = r.url
                self.size = int(content_range.rsplit("/", 1)[1])
                self.bytes_received += len(r.content)
                if len(r.content) != size and start + len(r.content) != self.size:
                    raise ValueError("Short HTTP range")
                return r.content
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"Range exhausted for {self.url} at {start}: {last_error}")

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size-self.pos
        data = self.range(self.pos, min(size, self.size-self.pos))
        self.pos += len(data)
        return data


def parse_header(data, offset=0, allow_descriptor=False):
    if len(data) < offset + 30:
        raise ValueError("Short ZIP header")
    a = struct.unpack_from("<4s5H3L2H", data, offset)
    if a[0] != b"PK\x03\x04":
        raise ValueError("Not a local ZIP header")
    if a[2] & 8 and not allow_descriptor:
        raise ValueError("ZIP data descriptors unsupported; cannot skip by size")
    n, e = a[-2:]
    if not 1 <= n <= 512 or len(data) < offset+30+n+e:
        raise ValueError("Invalid ZIP name/extra length")
    name = data[offset+30:offset+30+n].decode("utf8")
    return {"name": name, "method": a[3], "crc": a[6], "compressed_size": a[7],
            "size": a[8], "header_size": 30+n+e}


def find_first_header(data):
    pos = 0
    while True:
        pos = data.find(b"PK\x03\x04", pos)
        if pos < 0:
            raise ValueError("No valid train header in initial disk search")
        try:
            entry = parse_header(data, pos)
            if entry["name"].startswith("train/") and entry["method"] in (0, 8):
                return pos
        except (ValueError, UnicodeDecodeError):
            pass
        pos += 4


def scan_disk(disk, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/f"train.z{disk:02d}.state.json"
    index_path = output/f"train.z{disk:02d}.jsonl"
    remote = HTTPRange(BASE+f"train.z{disk:02d}", DISK_SIZE)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf8"))
        pos = state["next_offset"]
    else:
        pos = find_first_header(remote.range(0, 4*1024*1024))
    count = 0
    with index_path.open("a", encoding="utf8", buffering=1) as out:
        while pos+30 < DISK_SIZE:
            raw = remote.range(pos, min(2048, DISK_SIZE-pos))
            try:
                item = parse_header(raw)
            except ValueError:
                atomic_json(state_path, {"disk": disk, "next_offset": pos, "status": "invalid_header"})
                raise
            item.update(disk=disk, offset=pos, data_offset=pos+item["header_size"])
            out.write(json.dumps(item, ensure_ascii=False)+"\n")
            pos += item["header_size"]+item["compressed_size"]
            count += 1
            if count % 25 == 0 or pos >= DISK_SIZE:
                atomic_json(state_path, {"disk": disk, "next_offset": pos,
                            "status": "disk_scanned" if pos >= DISK_SIZE else "scanning",
                            "downloaded_bytes_this_process": remote.bytes_received})
                print(json.dumps({"disk": disk, "entries_this_process": count, "offset": pos}), flush=True)
    return count


def extract_entry(entry, output):
    """Extract a single ZIP member, including members crossing 8GiB disks."""
    data = bytearray()
    if "archive_url" in entry:
        remote = HTTPRange(entry["archive_url"], entry["archive_size"])
        raw = remote.range(entry["offset"], 2048)
        local = parse_header(raw, allow_descriptor=True)
        if local["name"] != entry["name"]:
            raise ValueError("Central/local ZIP name mismatch")
        packed = remote.range(entry["offset"]+local["header_size"], entry["compressed_size"])
        decoded = zlib.decompress(packed, -15) if entry["method"] == 8 else packed
        if len(decoded) != entry["size"] or zlib.crc32(decoded) & 0xffffffff != entry["crc"]:
            raise ValueError("Member CRC/size mismatch")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(decoded)
        return output
    disk, offset, left = entry["disk"], entry["data_offset"], entry["compressed_size"]
    while left:
        take = min(left, DISK_SIZE-offset)
        if disk > 8:
            raise ValueError("Member crosses into missing final archive disk")
        remote = HTTPRange(BASE+f"train.z{disk:02d}", DISK_SIZE)
        data += remote.range(offset, take)
        left -= take
        disk += 1
        offset = 0
    decoded = zlib.decompress(data, -15) if entry["method"] == 8 else bytes(data)
    if len(decoded) != entry["size"] or zlib.crc32(decoded) & 0xffffffff != entry["crc"]:
        raise ValueError("Member CRC/size mismatch")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(decoded)
    return output


def download_metadata(output):
    import requests
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        p = output/f"meta_{split}.csv"
        if not p.exists():
            r = requests.get(BASE+p.name, timeout=90)
            r.raise_for_status()
            p.write_bytes(r.content)


def index_validation(output):
    remote = HTTPRange(BASE+"val.zip")
    remote.range(0, 1)
    with zipfile.ZipFile(remote) as archive:
        entries = [{"name": e.filename, "offset": e.header_offset,
                    "compressed_size": e.compress_size, "size": e.file_size,
                    "crc": e.CRC, "method": e.compress_type,
                    "archive_url": BASE+"val.zip", "archive_size": remote.size}
                   for e in archive.infolist()]
    atomic_json(output, entries)
    print(json.dumps({"validation_entries": len(entries), "network_bytes": remote.bytes_received}), flush=True)


def process_clip(raw_path, output, metadata, detector, split="train"):
    """Sparse official S3FD + interpolated boxes; explicit deviation from dense tracking.

    We retain a clip only when its own AV stream PTS origins agree. No SyncNet
    score, offset search or semantic-model filtering determines its zero point.
    """
    import cv2
    import numpy as np
    import python_speech_features
    from scipy.io import wavfile
    from scipy.signal import medfilt
    raw_path, output = Path(raw_path), Path(output)
    clip_id = raw_path.stem.removesuffix("_FACE")
    row = metadata[clip_id]
    out = output/clip_id
    out.mkdir(parents=True, exist_ok=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                               "-of", "json", str(raw_path)], text=True))
    videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audios = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if len(videos) != 1 or not audios:
        raise ValueError("Expected one video with its own audio stream")
    v, a = videos[0], audios[0]
    vst, ast = float(v.get("start_time", 0)), float(a.get("start_time", 0))
    if abs(vst) > 0.001 or abs(ast) > 0.001:
        raise ValueError(f"Nonzero PTS origin requires explicit handling: {vst},{ast}")
    cap = cv2.VideoCapture(str(raw_path))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(float(v["duration"]), float(a["duration"]))
    if not 2 <= duration <= 31 or abs(fps-30) > 0.001:
        raise ValueError("Unexpected length/frame rate")
    frame_probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(raw_path)], text=True))
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frame_probe["frames"]])
    gaps = np.diff(pts)
    if (len(pts) != frames or len(gaps) == 0 or abs(pts[0]-vst) > 0.001
            or np.any(np.abs(gaps-1/30) > 0.0001)
            or abs(pts[-1]-pts[0]-(len(pts)-1)/30) > 0.0001):
        raise ValueError("Nonuniform/missing video timestamp support; do not silently fill with fps filter")
    pts_audit = {"decoded_frame_count": len(pts), "min_gap_seconds": float(gaps.min()),
                 "max_gap_seconds": float(gaps.max()), "last_frame_time": float(pts[-1]),
                 "clock_drift_seconds": float(pts[-1]-(len(pts)-1)/30)}
    positions = np.unique(np.linspace(0, frames-1, max(3, int(duration/2)+1)).astype(int))
    boxes = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Sparse video decode failed")
        found = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf_th=0.9, scales=[0.25])
        if len(found) != 1:
            raise ValueError(f"Sparse S3FD requires exactly one face, got {len(found)}")
        boxes.append(found[0, :4])
    cap.release()
    boxes = np.array(boxes)
    widths = boxes[:, 2]-boxes[:, 0]
    heights = boxes[:, 3]-boxes[:, 1]
    if widths.min() < 70 or heights.min() < 70 or widths.max()/widths.min() > 2:
        raise ValueError("Small/unstable face geometry")
    times = positions/fps
    nframes = int(np.floor(duration*25))
    t = np.arange(nframes)/25
    xs = np.interp(t, times, (boxes[:, 0]+boxes[:, 2])/2)
    ys = np.interp(t, times, (boxes[:, 1]+boxes[:, 3])/2)
    sizes = np.interp(t, times, np.maximum(widths, heights)/2)
    xs, ys, sizes = [medfilt(x, 13) for x in (xs, ys, sizes)]
    video_path, audio_path, mfcc_path = out/"face.mp4", out/"audio.wav", out/"mfcc.npy"
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(raw_path), "-an", "-vf", "fps=25",
                                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", "224x224", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "20", "-threads", "1", "-pix_fmt", "yuv420p", str(video_path)], stdin=subprocess.PIPE)
    count = 0
    shape = (int(v["height"]), int(v["width"]), 3)
    frame_bytes = int(np.prod(shape))
    while True:
        data = decoder.stdout.read(frame_bytes)
        if not data:
            break
        if len(data) != frame_bytes:
            raise ValueError("Partial FFmpeg raw frame")
        if count < nframes:
            frame = np.frombuffer(data, np.uint8).reshape(shape)
            bs, x, y = sizes[count], xs[count], ys[count]
            # Same crop-scale geometry as official SyncNet run_pipeline.py (cs=.4).
            pad = int(bs*1.8)+1
            image = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), constant_values=110)
            face = image[int(y+pad-bs):int(y+pad+bs*1.8), int(x+pad-bs*1.4):int(x+pad+bs*1.4)]
            encoder.stdin.write(cv2.resize(face, (224, 224)).tobytes())
        count += 1
    encoder.stdin.close()
    if decoder.wait() or encoder.wait() or count < nframes:
        raise ValueError("FFmpeg video conversion failed")
    output_probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames,r_frame_rate,width,height", "-of", "json", str(video_path)], text=True))["streams"][0]
    if int(output_probe["nb_frames"]) != nframes or output_probe["r_frame_rate"] != "25/1":
        raise ValueError("Output frame count/rate differs from manifest")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(raw_path), "-map", "0:a:0", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path)], check=True)
    sr, audio = wavfile.read(audio_path)
    mfcc = python_speech_features.mfcc(audio, sr).T.astype(np.float32)
    np.save(mfcc_path, mfcc)
    (out/"text.txt").write_text(row["TEXT"], encoding="utf8")
    record = {"id": clip_id, "speaker": clip_id.split("_")[0], "split": split,
              "video_path": str(video_path.resolve()), "audio_path": str(audio_path.resolve()),
              "mfcc_path": str(mfcc_path.resolve()), "text_path": str((out/"text.txt").resolve()),
              "text": row["TEXT"], "frame_count": nframes, "duration": nframes/25,
              "audio_samples": len(audio), "sample_rate": sr, "mfcc_shape": list(mfcc.shape),
              "video_bgr": True, "fps": 25, "raw_path": str(raw_path.resolve()),
              "phone_spans_path": None,
              "provenance": {"dataset": "BAAI/Chinese-LiPS", "dataset_revision": DATASET_REVISION, "video_pts_start": vst,
                 "audio_pts_start": ast, "audio_source": "raw_FACE_same_container_track",
                 "hardware_sync_verified": False, "detector": "official_S3FD_CPU_sparse_2sec_interpolated",
                 "dense_official_pipeline": False, "crop_scale": 0.4, "raw_probe": probe,
                 "video_pts_audit": pts_audit, "output_probe": output_probe,
                 "audio_samples_minus_stream_duration": len(audio)-round(float(a["duration"])*sr),
                 "sparse_frame_indices": positions.tolist(), "sparse_boxes": boxes.tolist()}}
    atomic_json(out/"record.json", record)
    return record


_DETECTOR = None
_METADATA = None


def init_detector(syncnet_root, metadata_path):
    import sys
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(syncnet_root).resolve()))
    from detectors.s3fd import S3FD
    old = os.getcwd()
    os.chdir(syncnet_root)
    global _DETECTOR
    global _METADATA
    _METADATA = {r["ID"]: r for r in csv.DictReader(Path(metadata_path).open(encoding="utf8"))}
    _DETECTOR = S3FD(device="cpu")
    os.chdir(old)


def process_one(arguments):
    path, output, split = arguments
    started = time.time()
    try:
        record = process_clip(path, output, _METADATA, _DETECTOR, split)
        return {"record": record, "elapsed": time.time()-started}
    except Exception as exc:
        return {"path": str(path), "reason": repr(exc), "elapsed": time.time()-started}


def approximate_durations(processed_index):
    return {Path(e["name"]).stem: max(0, e["size"]-44)/32000
            for e in processed_index if e["name"].endswith(".wav")}


def prefix_candidates(index_dir, durations, seconds_per_disk=10080, quotas=None):
    """Deterministic per-disk prefix quotas; independent of download completion."""
    chosen, coverage = [], {}
    for disk in range(1, 9):
        quota = quotas[str(disk)] if quotas else seconds_per_disk
        file = Path(index_dir)/f"train.z{disk:02d}.jsonl"
        seen, seconds = set(), 0
        if file.exists():
            for line in file.read_text(encoding="utf8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrently appended final line can be incomplete.
                if e["name"] in seen or not e["name"].endswith("_FACE.mp4"):
                    continue
                seen.add(e["name"])
                clip_id = Path(e["name"]).stem.removesuffix("_FACE")
                if clip_id not in durations:
                    raise ValueError(f"Raw clip missing processed-duration metadata: {clip_id}")
                chosen.append(e)
                seconds += durations[clip_id]
                if seconds >= quota:
                    break
        coverage[str(disk)] = {"approximate_seconds": seconds, "quota_met": seconds >= quota,
                               "quota_seconds": quota,
                               "face_entries": len(seen)}
    return chosen, coverage


def allocate_prefix_quotas(capacities, total_seconds=8*10080):
    """Water-fill equal disk quotas; exhausted low-capacity disks contribute all.

    Infinite capacity means indexing is incomplete. Allocation is deterministic
    from metadata and never depends on a synchronization/model score.
    """
    active, quotas, remaining = set(map(str, range(1, 9))), {}, float(total_seconds)
    while active:
        level = remaining/len(active)
        capped = [d for d in active if capacities.get(d, float("inf")) <= level]
        if not capped:
            quotas.update({d: level for d in active})
            break
        for disk in capped:
            quotas[disk] = capacities[disk]
            remaining -= capacities[disk]
            active.remove(disk)
    if remaining > 0.001 and not active:
        raise ValueError("Uploaded raw training prefix has insufficient total candidate duration")
    return quotas


def balanced_records(records, target_seconds, seed):
    speakers = {}
    for record in records:
        speakers.setdefault(record["speaker"], []).append(record)
    for values in speakers.values():
        values.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    speaker_order = sorted(speakers, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    selected, seconds = [], 0
    for depth in range(max(map(len, speakers.values()), default=0)):
        for speaker in speaker_order:
            if depth < len(speakers[speaker]):
                record = speakers[speaker][depth]
                selected.append(record)
                seconds += record["duration"]
                if seconds >= target_seconds:
                    return selected
    raise ValueError(f"Insufficient eligible duration: {seconds/3600:.3f}h < {target_seconds/3600:.3f}h")


def audit_wav_tracks(raw_wav_dir, processed_root, output):
    """Compare independently packaged raw WAV with FACE-container decoded PCM."""
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import correlate, correlation_lags
    results = []
    for raw in sorted(Path(raw_wav_dir).glob("*.wav")):
        canonical = Path(processed_root)/raw.stem/"audio.wav"
        if not canonical.exists():
            continue
        decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(raw), "-ac", "1", "-ar", "16000",
                                           "-f", "s16le", "pipe:1"])
        x = np.frombuffer(decoded, dtype="<i2").astype(np.float64)
        sr, y = wavfile.read(canonical)
        y = y.astype(np.float64)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        x -= x.mean()
        y -= y.mean()
        lags = correlation_lags(n, n)
        correlation = correlate(y, x, mode="full", method="fft")
        valid = np.abs(lags) <= 16000
        best = int(lags[valid][np.argmax(correlation[valid])])
        zero_corr = float(np.dot(x, y)/np.sqrt(np.dot(x, x)*np.dot(y, y)))
        chunk_lags = []
        width = min(n, 32000)
        for start in sorted({0, max(0, (n-width)//2), max(0, n-width)}):
            xx, yy = x[start:start+width], y[start:start+width]
            cc = correlate(yy, xx, mode="full", method="fft")
            ll = correlation_lags(len(yy), len(xx))
            vv = np.abs(ll) <= 1600
            chunk_lags.append(int(ll[vv][np.argmax(cc[vv])]))
        results.append({"id": raw.stem, "speaker": raw.stem.split("_")[0], "sample_rate": sr,
                        "zero_lag_waveform_correlation": zero_corr, "best_lag_samples_face_minus_wav": best,
                        "first_middle_last_lags_samples": chunk_lags,
                        "raw_wav_decoded_samples": len(decoded)//2, "face_pcm_samples": len(wavfile.read(canonical)[1])})
    atomic_json(output, {"status": "AUDITED", "results": results, "hardware_sync_truth": False,
                        "interpretation": "Packaging/audio consistency only; not independent proof of natural physical AV synchrony"})
    print(json.dumps(results), flush=True)


class PipelinePool(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, status_path, *args, **kwargs):
        self.status_path = status_path
        super().__init__(*args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            atomic_json(self.status_path, {"status": "FAILED", "reason": repr(exc_val),
                                          "updated_unix": time.time(), "training_ready": False})
            self.shutdown(wait=False, cancel_futures=True)
            return False
        return super().__exit__(exc_type, exc_val, exc_tb)


def pipeline(args):
    """Windows persistent cache -> CPU preprocessing -> volatile Ubuntu training cache.

    All remote paths are confined to this experiment's dedicated /dev/shm tree.
    Training waits for READY.json written only after actual durations and split
    isolation pass. Failed processing is recorded rather than silently relabeled.
    """
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    remote_root = args.remote_root.rstrip("/")
    if not remote_root.startswith("/dev/shm/chinese_lips20h_20261029"):
        raise ValueError("Remote root must remain in the task-owned volatile tree")
    host = args.host
    def ssh(command, **kwargs):
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True, **kwargs)
    def scp(sources, destination):
        subprocess.run(["scp", *map(str, sources), destination], check=True)
    def remote_words(words):
        return " ".join(shlex.quote(str(w)) for w in words)
    script = f"{remote_root}/chinese_lips_data_20261029.py"
    ssh(remote_words(["mkdir", "-p", remote_root]))
    scp([Path(__file__).resolve()], f"{host}:{script}")
    for split in ["train", "valid"]:
        scp([Path(args.metadata)/f"meta_{split}.csv"], f"{host}:{remote_root}/meta_{split}.csv")
    durations = approximate_durations(json.loads(Path(args.processed_index).read_text(encoding="utf8")))
    validation = [e for e in json.loads(Path(args.validation_index).read_text(encoding="utf8"))
                  if e["name"].endswith("_FACE.mp4")]
    # Official validation is complete; take a deterministic 3h candidate count
    # across all its speakers, then finalize to actual 2h after preprocessing.
    val_speakers = {}
    for entry in validation:
        val_speakers.setdefault(Path(entry["name"]).name.split("_")[0], []).append(entry)
    for entries in val_speakers.values():
        entries.sort(key=lambda e: hashlib.sha256(f"{args.seed}:{e['name']}".encode()).hexdigest())
    validation = [val_speakers[s][depth] for depth in range(max(map(len, val_speakers.values())))
                  for s in sorted(val_speakers) if depth < len(val_speakers[s])][:1100]
    processed_names = set()
    for split in ["train", "val"]:
        for p in (root/"processed"/split).glob("*/record.json"):
            processed_names.add((split, p.parent.name))
    rejected = set()
    status_path = root/"pipeline_status.json"
    pending = {}
    started_unix = time.time()
    with PipelinePool(status_path, max_workers=args.download_workers) as pool:
        # Validation can prepare immediately, independently of the growing train index.
        queues = {"val": validation, "train": []}
        scheduled = set()
        finished_splits = set()
        while len(finished_splits) < 2:
            if time.time()-started_unix > 18*3600:
                raise TimeoutError("Data acquisition exceeded 18h; READY not published")
            completed_disks = []
            for disk in map(str, range(1, 9)):
                state_path = Path(args.index)/f"train.z{int(disk):02d}.state.json"
                if state_path.exists():
                    disk_state = json.loads(state_path.read_text(encoding="utf8"))
                    if disk_state.get("status") == "invalid_header":
                        raise ValueError(f"Index scanner failed on disk {disk}; no READY")
                    if disk_state.get("status") == "disk_scanned":
                        completed_disks.append(disk)
            capacities = {}
            if completed_disks:
                _, all_coverage = prefix_candidates(args.index, durations, float("inf"))
                capacities = {d: all_coverage[d]["approximate_seconds"] for d in completed_disks}
            quotas = allocate_prefix_quotas(capacities)
            train, coverage = prefix_candidates(args.index, durations, quotas=quotas)
            queues["train"] = train
            atomic_json(root/"candidate_selection.json", {"train": train, "val": validation,
                        "train_available_prefix": True, "prefix_coverage": coverage})
            train_index_ready = all(v["quota_met"] for v in coverage.values())
            for split in ["val", "train"]:
                for entry in queues[split]:
                    if len(pending) >= args.download_workers*20:
                        break
                    clip_id = Path(entry["name"]).stem.removesuffix("_FACE")
                    key = (split, clip_id)
                    if key in scheduled or key in processed_names or key in rejected:
                        continue
                    destination = root/"raw_face"/split/f"{clip_id}_FACE.mp4"
                    def acquire(e=entry, dst=destination):
                        if not dst.exists() or dst.stat().st_size != e["size"]:
                            extract_entry(e, dst)
                        return dst
                    pending[key] = pool.submit(acquire)
                    scheduled.add(key)
            ready = {s: [] for s in ["val", "train"]}
            for key, future in list(pending.items()):
                if future.done():
                    # Network failure is terminal and remains visible; no whole-run retry.
                    ready[key[0]].append((key[1], future.result()))
            for split in ["val", "train"]:
                if not ready[split]:
                    continue
                batch = ready[split][:args.batch_size]
                raw_remote, prepared_remote = f"{remote_root}/raw/{split}", f"{remote_root}/candidates/{split}"
                ssh(remote_words(["mkdir", "-p", raw_remote, prepared_remote]))
                scp([p for _, p in batch], f"{host}:{raw_remote}/")
                meta_split = "valid" if split == "val" else split
                ssh(remote_words([args.python, script, "process", "--raw-dir", raw_remote,
                        "--output", prepared_remote, "--metadata", f"{remote_root}/meta_{meta_split}.csv",
                        "--syncnet-root", args.syncnet_root, "--split", split, "--workers", str(args.process_workers)]))
                # Retrieve metadata first, then only successful clip directories.
                batch_status = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host,
                    remote_words([args.python, "-c", "import json,pathlib;root=pathlib.Path("+repr(prepared_remote)+");print(json.dumps([p.parent.name for p in root.glob('*/record.json')]))"])], text=True)
                remote_ids = set(json.loads(batch_status))
                good = [clip_id for clip_id, _ in batch if clip_id in remote_ids]
                archive = root/f"batch-{split}.tar"
                if good:
                    with archive.open("wb") as sink:
                        ssh(remote_words(["tar", "-C", prepared_remote, "-cf", "-", *good]), stdout=sink)
                    destination = root/"processed"/split
                    destination.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(archive) as source:
                        source.extractall(destination, filter="data")
                    archive.unlink()
                for clip_id, _ in batch:
                    key = (split, clip_id)
                    pending.pop(key)
                    if clip_id in remote_ids:
                        processed_names.add(key)
                    else:
                        rejected.add(key)
                ssh(remote_words(["rm", "--", *[f"{raw_remote}/{p.name}" for _, p in batch]]))
            valid_done = all(("val", Path(e["name"]).stem.removesuffix("_FACE")) in processed_names | rejected for e in validation)
            train_done = train_index_ready and all(("train", Path(e["name"]).stem.removesuffix("_FACE")) in processed_names | rejected for e in train)
            if valid_done:
                finished_splits.add("val")
            if train_done:
                finished_splits.add("train")
            atomic_json(status_path, {"status": "PREPARING", "train_index_ready": train_index_ready,
                    "prefix_coverage": coverage, "processed": {s: sum(k[0] == s for k in processed_names) for s in queues},
                    "rejected": sorted(rejected), "pending_downloads": len(pending),
                    "updated_unix": time.time(), "training_ready": False})
            print(json.dumps(json.loads(status_path.read_text()), ensure_ascii=False), flush=True)
            if len(finished_splits) < 2:
                time.sleep(10)
    manifests, summaries = {}, {}
    for split, target in [("train", 20*3600), ("val", 2*3600)]:
        records = [json.loads(p.read_text(encoding="utf8")) for p in (root/"processed"/split).glob("*/record.json")]
        chosen = balanced_records(records, target, args.seed)
        manifests[split] = chosen
        summaries[split] = {"clips": len(chosen), "duration_seconds": sum(r["duration"] for r in chosen),
                            "speakers": sorted({r["speaker"] for r in chosen})}
    if set(summaries["train"]["speakers"]) & set(summaries["val"]["speakers"]):
        raise ValueError("Official split speaker overlap")
    cohort = root/"cohort"
    cohort.mkdir(exist_ok=True)
    for split, records in manifests.items():
        p = cohort/f"{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf8")
        ssh(remote_words(["mkdir", "-p", f"{remote_root}/cohort/{split}"]))
        scp([p], f"{host}:{remote_root}/cohort/{split}/manifest.jsonl")
    receipt = {"status": "READY", "train_manifest": f"{remote_root}/cohort/train/manifest.jsonl",
               "val_manifest": f"{remote_root}/cohort/val/manifest.jsonl", "summaries": summaries,
               "seed": args.seed, "train_selection": "equal_2.8h_disk_prefix_quotas_with_metadata_only_waterfill_for_exhausted_disks_then_hash_speaker_balanced_20h",
               "full_train_uniform": False, "disk_upload_missing_final_central_directory": True,
               "dataset_revision": DATASET_REVISION,
               "initial_acquisition": "Initial main URL index/pilot acquisition preceded revision pin; immutable sizes/CRC and API revision retained separately",
               "persistent_cache": str(root), "volatile_remote": remote_root,
               "phones_ready": False, "synchronization_ability_verified": False}
    atomic_json(cohort/"READY.json", receipt)
    scp([cohort/"READY.json"], f"{host}:{remote_root}/data_ready.json.tmp")
    ssh(remote_words(["mv", "--", f"{remote_root}/data_ready.json.tmp", f"{remote_root}/data_ready.json"]))
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def choose_entries(index_dir, count, seed):
    entries = {}
    for p in sorted(Path(index_dir).glob("train.z??.jsonl")):
        for line in p.read_text(encoding="utf8").splitlines():
            e = json.loads(line)
            if e["name"].endswith("_FACE.mp4"):
                entries[e["name"]] = e
    by_speaker = {}
    for name, e in entries.items():
        speaker = Path(name).name.split("_")[0]
        by_speaker.setdefault(speaker, []).append(e)
    for speaker in by_speaker:
        by_speaker[speaker].sort(key=lambda e: hashlib.sha256(f"{seed}:{e['name']}".encode()).hexdigest())
    # Round robin across available speakers avoids selecting only a speaker block.
    selected = []
    speakers = sorted(by_speaker, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    for depth in range(max(map(len, by_speaker.values()), default=0)):
        for speaker in speakers:
            if depth < len(by_speaker[speaker]):
                selected.append(by_speaker[speaker][depth])
                if len(selected) == count:
                    return selected
    return selected


def download_selected(args):
    output = Path(args.output)
    selected = choose_entries(args.index, args.count, args.seed)
    atomic_json(output/"selection.json", {"status": "AVAILABLE_PREFIX_NOT_FULL_TRAIN_UNIFORM",
                "seed": args.seed, "count": len(selected), "entries": selected,
                "selection_rule": "hash order within speaker; speaker-balanced round robin over indexed available prefix"})
    def one(e):
        dst = output/"raw_face"/Path(e["name"]).name
        if not dst.exists() or dst.stat().st_size != e["size"]:
            extract_entry(e, dst)
        return dst.name
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, name in enumerate(pool.map(one, selected), 1):
            if i % 10 == 0:
                print(json.dumps({"downloaded": i, "total": len(selected), "last": name}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("metadata")
    p.add_argument("--output", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--output", required=True)
    p.add_argument("--disks", default="1,2,3,4,5,6,7,8")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("download")
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=7500)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("validation-index")
    p.add_argument("--output", required=True)
    p = sub.add_parser("process")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--syncnet-root", required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("pipeline")
    p.add_argument("--output", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--processed-index", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--host", default="wjj@192.168.50.2")
    p.add_argument("--remote-root", default="/dev/shm/chinese_lips20h_20261029")
    p.add_argument("--python", default="/home/wjj/.venvs/syncnet/bin/python")
    p.add_argument("--syncnet-root", default="/home/wjj/tts-audio/tts-exp/third_party/syncnet_python")
    p.add_argument("--download-workers", type=int, default=6)
    p.add_argument("--process-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261029)
    p = sub.add_parser("audit-wav")
    p.add_argument("--raw-wav-dir", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "metadata":
        download_metadata(args.output)
    elif args.command == "scan":
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda disk: scan_disk(disk, args.output), map(int, args.disks.split(","))))
    elif args.command == "download":
        download_selected(args)
    elif args.command == "validation-index":
        index_validation(args.output)
    elif args.command == "pipeline":
        try:
            pipeline(args)
        except Exception as exc:
            atomic_json(Path(args.output)/"pipeline_status.json", {"status": "FAILED", "reason": repr(exc),
                "updated_unix": time.time(), "training_ready": False})
            raise
    elif args.command == "audit-wav":
        audit_wav_tracks(args.raw_wav_dir, args.processed_root, args.output)
    else:
        paths = sorted(Path(args.raw_dir).glob("*_FACE.mp4"))
        if args.max_clips:
            paths = paths[:args.max_clips]
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        paths = [path for path in paths if not (out/path.stem.removesuffix("_FACE")/"record.json").exists()]
        if not paths:
            return
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=init_detector,
                                                      initargs=(args.syncnet_root, args.metadata))
        with (out/"manifest.jsonl").open("a", encoding="utf8", buffering=1) as manifest, (out/"rejected.jsonl").open("a", encoding="utf8", buffering=1) as rejected:
            for result in pool.map(process_one, [(path, out, args.split) for path in paths]):
                if "record" in result:
                    record = result["record"]
                    manifest.write(json.dumps(record, ensure_ascii=False)+"\n")
                    print(json.dumps({"processed": record["id"], "duration": record["duration"], "elapsed": result["elapsed"]}), flush=True)
                else:
                    rejected.write(json.dumps(result)+"\n")
                    print(json.dumps({"rejected": result["path"], "reason": result["reason"]}), flush=True)
        pool.shutdown()


if __name__ == "__main__":
    main()
