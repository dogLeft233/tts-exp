#!/usr/bin/env python3
"""Run the repository's official SyncNet V2 evaluator on a static-TFG run.

The static Wav2Lip stage intentionally stores video-only FFV1 files and the
direct scorer supplies audio separately.  The official evaluator instead
expects one media file, so this adapter creates a per-cell Matroska file with
the original video stream copied losslessly and 16 kHz mono PCM audio, then
runs the unchanged official ``run_pipeline.py`` and ``run_syncnet.py``.

The selected cells mirror the direct-score preregistration:

    N/N, D/N, A/N, B/N, C/N,
    N/D, N/A, N/B, N/C,
    D/D, A/A, B/B, C/C, T/T.

This script is sequential by design: official face detection and SyncNet use
the same GPU.  It is resumable and records every command, hash, and failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ARMS = ("N", "T", "D", "A", "B", "C")
SELECTED_CELLS = (
    ("N", "N"),
    ("D", "N"),
    ("A", "N"),
    ("B", "N"),
    ("C", "N"),
    ("N", "D"),
    ("N", "A"),
    ("N", "B"),
    ("N", "C"),
    ("D", "D"),
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("T", "T"),
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONFIDENCE_RE = re.compile(rf"Confidence:\s+({_NUMBER})")
MIN_DIST_RE = re.compile(rf"Min dist:\s+({_NUMBER})")
OFFSET_RE = re.compile(r"AV offset:\s+(-?\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            handle.write(f"\n[TIMEOUT] after {timeout_seconds} seconds\n")
            returncode = 124
        handle.write(f"\n[returncode] {returncode}\n")
        handle.write(f"[elapsed_seconds] {time.monotonic() - started:.3f}\n")
    return returncode


def ffprobe_streams(ffprobe: Path, media: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(media)],
        capture_output=True,
        check=True,
    )
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"ffprobe returned no stream list: {media}")
    return streams


def decode_pcm16(ffmpeg: Path, media: Path) -> bytes:
    result = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def strict_mux(
    *,
    video: Path,
    audio: Path,
    output: Path,
    ffmpeg: Path,
    ffprobe: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Mux one cell and verify the decoded PCM contract.

    FFV1 is deliberately kept as a video stream copy.  The official pipeline
    will decode it to frames itself; no image re-encoding is introduced here.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial{output.suffix}")
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "matroska",
        str(temporary),
    ]
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            result = subprocess.run(
                command,
                cwd=str(output.parent),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(f"\n[returncode] {result.returncode}\n")
            handle.write(f"[elapsed_seconds] {time.monotonic() - started:.3f}\n")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed ({result.returncode}): {log_path}")

        source_streams = ffprobe_streams(ffprobe, video)
        muxed_streams = ffprobe_streams(ffprobe, temporary)
        source_videos = [s for s in source_streams if s.get("codec_type") == "video"]
        muxed_videos = [s for s in muxed_streams if s.get("codec_type") == "video"]
        muxed_audios = [s for s in muxed_streams if s.get("codec_type") == "audio"]
        if len(source_videos) != 1 or len(muxed_videos) != 1 or len(muxed_audios) != 1:
            raise RuntimeError(f"unexpected stream count after mux: {output}")
        audio_stream = muxed_audios[0]
        if (
            audio_stream.get("codec_name") != "pcm_s16le"
            or int(audio_stream.get("sample_rate", 0)) != 16000
            or int(audio_stream.get("channels", 0)) != 1
        ):
            raise RuntimeError(f"muxed audio contract mismatch: {output}")

        source_pcm = decode_pcm16(ffmpeg, audio)
        muxed_pcm = decode_pcm16(ffmpeg, temporary)
        if source_pcm != muxed_pcm:
            raise RuntimeError(f"mux changed decoded PCM: {output}")
        temporary.replace(output)
        source_video = source_videos[0]
        muxed_video = muxed_videos[0]
        video_fields = (
            "codec_name",
            "width",
            "height",
            "pix_fmt",
            "r_frame_rate",
            "avg_frame_rate",
        )
        if any(source_video.get(field) != muxed_video.get(field) for field in video_fields):
            raise RuntimeError(f"video stream metadata changed during copy: {output}")
        return {
            "media": str(output.resolve()),
            "media_sha256": sha256_file(output),
            "video_source": str(video.resolve()),
            "video_source_sha256": sha256_file(video),
            "audio_source": str(audio.resolve()),
            "audio_source_sha256": sha256_file(audio),
            "video_stream_copy_requested": True,
            "video_stream_copy_verified_by_metadata": True,
            "video_stream": {field: muxed_video.get(field) for field in video_fields},
            "audio_stream": {
                field: audio_stream.get(field)
                for field in ("codec_name", "sample_rate", "channels", "channel_layout")
            },
            "source_pcm16k_mono_sha256": hashlib.sha256(source_pcm).hexdigest(),
            "muxed_pcm16k_mono_sha256": hashlib.sha256(muxed_pcm).hexdigest(),
            "audio_pcm_exact": True,
            "audio_sample_count_after_decode": len(source_pcm) // 2,
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_score(log_path: Path) -> dict[str, float | int] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = CONFIDENCE_RE.findall(text)
    min_dist = MIN_DIST_RE.findall(text)
    offsets = OFFSET_RE.findall(text)
    if len(confidence) != 1 or len(min_dist) != 1 or len(offsets) > 1:
        return None
    return {
        "sync_c": float(confidence[0]),
        "sync_d": float(min_dist[0]),
        "av_offset": int(offsets[0]) if offsets else None,
    }


def safe_key(pair_id: str, video_arm: str, audio_arm: str, *, portrait_id: str = "3", seed: int | None = None, scope: str = "official") -> str:
    seed_part = "none" if seed is None else str(int(seed))
    raw = f"{scope}__P_{portrait_id}__S_{seed_part}__{pair_id}__V_{video_arm}__A_{audio_arm}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def load_cells(run_root: Path, *, portrait_id: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    audio_manifest = read_json(run_root / "04_audio" / "manifest.json")
    tfg_manifest = read_json(run_root / "07_tfg" / "manifest.json")
    audio_rows = audio_manifest.get("rows", [])
    if not isinstance(audio_rows, list) or not audio_rows:
        raise RuntimeError("04_audio/manifest.json has no rows")
    audio_by_pair: dict[str, dict[str, Any]] = {}
    audio_path_index: dict[str, tuple[str, str]] = {}
    for row in audio_rows:
        pair_id = str(row["pair_id"])
        paths = {arm: str(Path(str(path)).resolve()) for arm, path in row["paths"].items()}
        if set(paths) != set(ARMS):
            raise RuntimeError(f"audio arms incomplete for {pair_id}: {sorted(paths)}")
        audio_by_pair[pair_id] = {"pair_id": pair_id, "source_group": row.get("source_group"), "paths": paths}
        for arm, path in paths.items():
            audio_path_index[path] = (pair_id, arm)

    videos_by_key: dict[tuple[str, str, int | None, str], str] = {}
    for result in tfg_manifest.get("results", []):
        output = Path(str(result["output"])).resolve()
        rendered_audio = str(Path(str(result["audio"])).resolve())
        try:
            pair_id, rendered_arm = audio_path_index[rendered_audio]
        except KeyError as exc:
            raise RuntimeError(f"TFG result audio is not in 04_audio manifest: {rendered_audio}") from exc
        if not output.is_file():
            raise FileNotFoundError(output)
        stem_parts = output.stem.split("__")
        parsed_arm = str(result.get("video_arm") or (stem_parts[-1] if stem_parts else ""))
        if parsed_arm not in ARMS:
            raise RuntimeError(f"cannot parse video arm from {output}")
        if parsed_arm != rendered_arm:
            raise RuntimeError(f"TFG receipt arm mismatch for {output}: {parsed_arm} != {rendered_arm}")
        rendered_portrait = str(result.get("portrait_id", "3"))
        rendered_seed = int(result["seed"]) if result.get("seed") is not None else None
        key = (pair_id, rendered_portrait, rendered_seed, parsed_arm)
        previous = videos_by_key.get(key)
        if previous is not None and previous != str(output):
            raise RuntimeError(f"duplicate video arm for {key}")
        videos_by_key[key] = str(output)

    available_portraits = sorted({key[1] for key in videos_by_key})
    selected_portrait = str(portrait_id or (available_portraits[0] if available_portraits else "3"))
    selected_seeds = sorted({key[2] for key in videos_by_key if key[1] == selected_portrait}, key=lambda value: (-1 if value is None else value))
    selected_seed = seed if seed is not None else (selected_seeds[0] if selected_seeds else None)

    cells: list[dict[str, Any]] = []
    for pair_id in sorted(audio_by_pair):
        if {key[3] for key in videos_by_key if key[:3] == (pair_id, selected_portrait, selected_seed)} != set(ARMS):
            raise RuntimeError(f"TFG video arms incomplete for {pair_id}/{selected_portrait}/{selected_seed}")
        for video_arm, audio_arm in SELECTED_CELLS:
            cells.append(
                {
                    "pair_id": pair_id,
                    "source_group": audio_by_pair[pair_id]["source_group"],
                    "video_arm": video_arm,
                    "audio_arm": audio_arm,
                    "portrait_id": selected_portrait,
                    "seed": selected_seed,
                    "scope": "official",
                    "video": videos_by_key[(pair_id, selected_portrait, selected_seed, video_arm)],
                    "audio": audio_by_pair[pair_id]["paths"][audio_arm],
                }
            )
    return cells


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    output_root = (args.output_root or run_root / "10_official_syncnet").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    syncnet_root = args.syncnet_root.resolve()
    syncnet_python = args.syncnet_python
    model = args.model.resolve()
    ffmpeg = args.ffmpeg.resolve()
    ffprobe = args.ffprobe.resolve()
    main_protocol = run_root / "protocol.json"
    main_lock = run_root / "00_protocol" / "protocol_lock.json"
    if not main_protocol.is_file() or not main_lock.is_file():
        raise RuntimeError("PROTOCOL_LOCK_MISSING")
    locked = read_json(main_lock)
    locked_sha = str(locked.get("protocol_sha256", ""))
    actual_sha = hashlib.sha256(main_protocol.read_bytes()).hexdigest()
    if not locked_sha or locked_sha != actual_sha:
        raise RuntimeError("PROTOCOL_LOCK_INVALID")
    if not args.skip_gpu_check:
        nvidia = shutil.which("nvidia-smi")
        if nvidia is None:
            raise RuntimeError("GPU_PREFLIGHT_FAILED:nvidia-smi missing")
        query = subprocess.run([nvidia, "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        if query.returncode != 0:
            raise RuntimeError("GPU_PREFLIGHT_FAILED:nvidia-smi query failed")
        gpu_rows = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
        selected = next((row for row in gpu_rows if int(row[0].strip()) == int(args.gpu_index)), None)
        if selected is None or float(selected[1].strip()) < float(args.min_free_gpu_mib) or float(selected[2].strip()) > 10.0:
            raise RuntimeError(f"GPU_PREFLIGHT_FAILED:gpu {args.gpu_index} is not idle or has insufficient memory")
        processes = subprocess.run([nvidia, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        allowed = tuple(str(value) for value in args.allow_process_name)
        foreign = [line for line in processes.stdout.splitlines() if line.strip() and not any(pattern in line for pattern in allowed)]
        if foreign:
            raise RuntimeError(f"GPU_PREFLIGHT_FAILED:foreign processes present: {foreign}")
    cells = load_cells(run_root, portrait_id=args.portrait_id, seed=args.seed)
    if args.limit is not None:
        cells = cells[: int(args.limit)]

    protocol = {
        "schema_version": 1,
        "evaluation": "official_syncnet_v2_end_to_end",
        "run_root": str(run_root),
        "output_root": str(output_root),
        "syncnet_root": str(syncnet_root),
        "syncnet_python": str(syncnet_python),
        "syncnet_model": str(model),
        "syncnet_model_sha256": sha256_file(model),
        "run_pipeline_sha256": sha256_file(syncnet_root / "run_pipeline.py"),
        "run_syncnet_sha256": sha256_file(syncnet_root / "run_syncnet.py"),
        "syncnet_instance_sha256": sha256_file(syncnet_root / "SyncNetInstance.py"),
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
        "min_track": int(args.min_track),
        "vshift": 15,
        "worker_policy": "sequential_single_gpu",
        "audio_contract": "per-cell mux, video stream copy, 16000 Hz mono pcm_s16le, decoded PCM exact",
        "portrait_id": args.portrait_id,
        "seed": args.seed,
        "scope": "SMOKE" if args.limit is not None else "FULL",
        "selected_cells_per_pair": [f"V_{v}/A_{a}" for v, a in SELECTED_CELLS],
        "selected_cell_count": len(cells),
        "timeout_pipeline_seconds": int(args.pipeline_timeout),
        "timeout_syncnet_seconds": int(args.syncnet_timeout),
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    official_protocol_path = output_root / "protocol.json"
    if args.resume and official_protocol_path.is_file():
        previous = read_json(official_protocol_path)
        if previous.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("OFFICIAL_PROTOCOL_CHANGED; refusing to reuse old official cells")
    write_json(official_protocol_path, protocol)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    for index, cell in enumerate(cells, 1):
        pair_id = str(cell["pair_id"])
        video_arm = str(cell["video_arm"])
        audio_arm = str(cell["audio_arm"])
        key = safe_key(pair_id, video_arm, audio_arm, portrait_id=str(cell.get("portrait_id", "3")), seed=cell.get("seed"), scope=str(protocol["scope"]))
        media_path = output_root / "media" / f"{key}.mkv"
        mux_log = output_root / "logs" / f"{key}.mux.log"
        pipeline_log = output_root / "logs" / f"{key}.pipeline.log"
        sync_log = output_root / "logs" / f"{key}.syncnet.log"
        score_path = output_root / "scores" / f"{key}.json"
        reference = f"official_{key}"
        base = {
            "schema_version": 1,
            "pair_id": pair_id,
            "source_group": cell.get("source_group"),
            "video_arm": video_arm,
            "audio_arm": audio_arm,
            "portrait_id": cell.get("portrait_id"),
            "seed": cell.get("seed"),
            "scope": protocol["scope"],
            "cell_key": f"V_{video_arm}/A_{audio_arm}",
            "video": str(Path(str(cell["video"])).resolve()),
            "audio": str(Path(str(cell["audio"])).resolve()),
            "reference": reference,
            "min_track": int(args.min_track),
            "scorer": "official_syncnet_v2_end_to_end",
            "protocol_sha256": protocol["protocol_sha256"],
            "video_sha256": sha256_file(Path(str(cell["video"]))),
            "audio_sha256": sha256_file(Path(str(cell["audio"]))),
        }

        if args.resume and score_path.is_file() and media_path.is_file():
            try:
                cached = read_json(score_path)
                if cached.get("status") == "COMPLETE" and cached.get("sync_c") is not None and cached.get("protocol_sha256") == protocol["protocol_sha256"] and cached.get("video_sha256") == base["video_sha256"] and cached.get("audio_sha256") == base["audio_sha256"] and cached.get("media_sha256") == sha256_file(media_path):
                    records.append(cached)
                    print(
                        f"CACHED {index}/{len(cells)} {pair_id} {video_arm}/{audio_arm} "
                        f"C={float(cached['sync_c']):.3f}",
                        flush=True,
                    )
                    continue
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        try:
            mux = strict_mux(
                video=Path(str(cell["video"])),
                audio=Path(str(cell["audio"])),
                output=media_path,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                log_path=mux_log,
            )
            data_dir = output_root / "data" / key
            pipeline_rc = run_logged(
                [
                    str(syncnet_python),
                    str(syncnet_root / "run_pipeline.py"),
                    "--videofile",
                    str(media_path),
                    "--reference",
                    reference,
                    "--data_dir",
                    str(data_dir),
                    "--min_track",
                    str(args.min_track),
                    "--overwrite",
                ],
                cwd=syncnet_root,
                log_path=pipeline_log,
                timeout_seconds=int(args.pipeline_timeout),
            )
            if pipeline_rc != 0:
                raise RuntimeError(f"run_pipeline failed with return code {pipeline_rc}")

            sync_rc = run_logged(
                [
                    str(syncnet_python),
                    str(syncnet_root / "run_syncnet.py"),
                    "--videofile",
                    str(media_path),
                    "--reference",
                    reference,
                    "--data_dir",
                    str(data_dir),
                    "--initial_model",
                    str(model),
                ],
                cwd=syncnet_root,
                log_path=sync_log,
                timeout_seconds=int(args.syncnet_timeout),
            )
            if sync_rc != 0:
                raise RuntimeError(f"run_syncnet failed with return code {sync_rc}")
            score = parse_score(sync_log)
            if score is None:
                raise RuntimeError(f"could not parse official SyncNet output: {sync_log}")
            record = {
                **base,
                **mux,
                **score,
                "status": "COMPLETE",
                "video_sha256": base["video_sha256"],
                "audio_sha256": base["audio_sha256"],
                "media_sha256": sha256_file(media_path),
                "mux_log": str(mux_log.resolve()),
                "pipeline_log": str(pipeline_log.resolve()),
                "syncnet_log": str(sync_log.resolve()),
                "data_dir": str(data_dir.resolve()),
            }
            write_json(score_path, record)
            records.append(record)
            print(
                f"OK {index}/{len(cells)} {pair_id} {video_arm}/{audio_arm} "
                f"C={float(record['sync_c']):.3f} D={float(record['sync_d']):.3f} "
                f"offset={record['av_offset']}",
                flush=True,
            )
        except Exception as exc:  # record a cell failure and continue the batch
            failure = {
                **base,
                "status": "FAILED",
                "stage": "official_syncnet",
                "error": str(exc),
                "media": str(media_path.resolve()),
                "mux_log": str(mux_log.resolve()),
                "pipeline_log": str(pipeline_log.resolve()),
                "syncnet_log": str(sync_log.resolve()),
            }
            failures.append(failure)
            write_json(output_root / "failures" / f"{key}.json", failure)
            print(f"FAIL {index}/{len(cells)} {pair_id} {video_arm}/{audio_arm}: {exc}", flush=True)

    result = {
        "schema_version": 1,
        "evaluation": "official_syncnet_v2_end_to_end",
        "status": "COMPLETE" if args.limit is None and not failures and len(records) == len(cells) else ("PARTIAL_SCOPE" if args.limit is not None else "PARTIAL"),
        "expected": len(cells),
        "completed": len(records),
        "failed": len(failures),
        "min_track": int(args.min_track),
        "selected_cells_per_pair": [f"V_{v}/A_{a}" for v, a in SELECTED_CELLS],
        "records": records,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--syncnet-root", type=Path, default=repo / "third_party/syncnet_python")
    parser.add_argument("--syncnet-python", type=Path, default=Path("/home/wjj/.venvs/syncnet/bin/python"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/home/wjj/miniconda3/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/home/wjj/miniconda3/bin/ffprobe"))
    parser.add_argument("--min-track", type=int, default=25)
    parser.add_argument("--pipeline-timeout", type=int, default=900)
    parser.add_argument("--syncnet-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None, help="debug/smoke limit, in deterministic cell order")
    parser.add_argument("--portrait-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-gpu-mib", type=float, default=5000.0)
    parser.add_argument("--allow-process-name", action="append", default=["gnome-remote-desktop-daemon"])
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_root = args.run_root.resolve()
    if args.model is None:
        args.model = args.syncnet_root / "data/syncnet_v2.model"
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "expected": result["expected"],
                "completed": result["completed"],
                "failed": result["failed"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
