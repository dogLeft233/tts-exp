#!/usr/bin/env python3
"""Engineering-only, real Wav2Lip + original-PCM mux + fresh SyncNet probe.

This deliberately scores one historically exposed pilot utterance. It is a
runtime/mux/scorer smoke probe and must never be included in candidate
selection, development summaries, or confirmation claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "63b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
EXPECTED_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
RMS_TARGET = 0.0376838172675746
RUN_ID = "cross_tfg_audio_head_20261003_oldpilot_qc_attempt01"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(argv: list[str], *, cwd: Path, log_stem: Path, timeout: int = 1800, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_stem.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_stem.with_suffix(".stdout.log").open("wb") as out, log_stem.with_suffix(".stderr.log").open("wb") as err:
        try:
            p = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, check=False)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return {
        "argv": argv, "cwd": str(cwd), "returncode": code,
        "started_unix": started, "ended_unix": time.time(),
        "stdout_path": str(log_stem.with_suffix(".stdout.log")),
        "stdout_sha256": sha256_file(log_stem.with_suffix(".stdout.log")),
        "stderr_path": str(log_stem.with_suffix(".stderr.log")),
        "stderr_sha256": sha256_file(log_stem.with_suffix(".stderr.log")),
    }


def read_ffprobe(path: Path) -> dict[str, Any]:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        raise RuntimeError("ffprobe failed: " + p.stderr.decode("utf-8", "replace")[-1500:])
    return json.loads(p.stdout)


def decode_f32(path: Path) -> bytes:
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        raise RuntimeError("float32 audio decode failed: " + p.stderr.decode("utf-8", "replace")[-1500:])
    return p.stdout


def video_stream_sha256(path: Path) -> str:
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-hash", "sha256", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        raise RuntimeError("video stream hash failed: " + p.stderr.decode("utf-8", "replace")[-1500:])
    text = p.stdout.decode("ascii", "strict").strip()
    if not text.startswith("SHA256="):
        raise RuntimeError("ffmpeg returned no SHA-256 video stream hash")
    return text.split("=", 1)[1].lower()


def patch_official_librosa_mel() -> None:
    """Adapt only the legacy Wav2Lip positional mel call to librosa 0.10+."""
    import numpy as np
    if not hasattr(np, "complex"):
        np.complex = np.complex128  # type: ignore[attr-defined]
    import librosa
    original = librosa.filters.mel

    def compatible(*args: Any, **kwargs: Any):
        keys = ("sr", "n_fft")
        if len(args) > len(keys):
            raise TypeError("unexpected positional librosa mel arguments")
        updated = dict(kwargs)
        for key, value in zip(keys, args):
            if key in updated:
                raise TypeError(f"mel got multiple values for {key}")
            updated[key] = value
        return original(**updated)

    librosa.filters.mel = compatible


def canonicalize_with_wav2lip_loader(source: Path, wav2lip_root: Path):
    sys.path.insert(0, str(wav2lip_root))
    patch_official_librosa_mel()
    import audio as official_audio  # noqa: E402
    import numpy as np  # noqa: E402
    from scipy.io import wavfile  # noqa: E402

    x = np.asarray(official_audio.load_wav(str(source), 16000), dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("official audio.load_wav did not return finite mono audio")
    return x, wavfile


def write_float_wav(path: Path, samples: Any, wavfile: Any) -> dict[str, Any]:
    import numpy as np
    x = np.ascontiguousarray(samples, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), 16000, x)
    rate, back = wavfile.read(str(path), mmap=False)
    back = np.ascontiguousarray(back, dtype="<f4")
    raw = back.tobytes()
    if int(rate) != 16000 or back.ndim != 1 or back.size != x.size or not np.array_equal(back, x):
        raise RuntimeError("float32 WAV readback differs from source samples")
    return {"path": str(path), "file_sha256": sha256_file(path), "pcm_sha256": sha256_bytes(raw), "sample_rate_hz": int(rate), "channels": 1, "sample_count": int(back.size)}


def generator_child(args: argparse.Namespace) -> int:
    import runpy
    import numpy as np
    import torch
    wav2lip_root = Path(args.wav2lip_root).resolve()
    os.chdir(Path(args.cwd).resolve())
    sys.path.insert(0, str(wav2lip_root))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    patch_official_librosa_mel()
    inference_path = wav2lip_root / "inference.py"
    sys.argv = [str(inference_path), "--checkpoint_path", args.checkpoint,
                "--face", args.face, "--audio", args.audio,
                "--outfile", args.outfile, "--static", "True", "--fps", "25",
                "--pads", "0", "10", "0", "0"]
    runpy.run_path(str(inference_path), run_name="__main__")
    return 0


def score_with_official_syncnet(video: Path, work: Path, syncnet_root: Path, syncnet_python: Path, arm_id: str) -> dict[str, Any]:
    py = str(syncnet_python)
    data_dir = work / ("syncnet_" + arm_id)
    reference = "oldpilot_" + arm_id
    data_dir.mkdir(parents=True, exist_ok=False)
    pipeline = syncnet_root / "run_pipeline.py"
    scoring = syncnet_root / "run_syncnet.py"
    pipeline_run = run([py, str(pipeline), "--videofile", str(video), "--reference", reference,
                        "--data_dir", str(data_dir), "--facedet_scale", "0.25",
                        "--crop_scale", "0.40", "--min_track", "5", "--frame_rate", "25",
                        "--num_failed_det", "25", "--min_face_size", "100", "--overwrite"],
                       cwd=syncnet_root, log_stem=work / "logs" / f"syncnet_pipeline_{arm_id}", timeout=1800)
    if pipeline_run["returncode"] != 0:
        raise RuntimeError(f"official SyncNet tracking pipeline failed ({pipeline_run['returncode']}); see {pipeline_run['stderr_path']}")
    score_run = run([py, str(scoring), "--data_dir", str(data_dir), "--reference", reference,
                     "--initial_model", str(syncnet_root / "data" / "syncnet_v2.model"),
                     "--batch_size", "20", "--vshift", "15"],
                    cwd=syncnet_root, log_stem=work / "logs" / f"syncnet_score_{arm_id}", timeout=1800)
    if score_run["returncode"] != 0:
        raise RuntimeError(f"official SyncNet fresh scoring failed ({score_run['returncode']}); see {score_run['stderr_path']}")
    active = data_dir / "pywork" / reference / "activesd.pckl"
    if not active.is_file():
        raise RuntimeError("official SyncNet scorer returned no activesd.pckl")
    import pickle
    import numpy as np
    with active.open("rb") as stream:
        by_track = pickle.load(stream)
    curves = []
    tracks = []
    for track_index, distance in enumerate(by_track):
        matrix = np.asarray(distance, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 31:
            raise RuntimeError(f"unexpected SyncNet distance matrix shape {matrix.shape}; expected [N,31]")
        curve = matrix.mean(axis=0, dtype=np.float32)
        best32 = np.min(curve).astype(np.float32)
        median32 = np.median(curve).astype(np.float32)
        best = float(best32)
        median = float(median32)
        d0 = np.float32(curve[15])
        c_at_zero_lag = np.float32(median32 - d0)
        c_best = np.float32(median32 - best32)
        min_idx = int(curve.argmin())
        curves.append({"track_index": track_index, "query_count": int(matrix.shape[0]),
                       "lag_offsets_frames": list(range(-15, 16)),
                       "lag_distance_mean_float32": curve.tolist(), "B_float32": median,
                       "Dmin_float32": best, "D0_float32": float(d0),
                       "sync_c_best_float32": float(c_best),
                       "sync_c_at_zero_lag_B_minus_D0_float32": float(c_at_zero_lag),
                       "fixed_anchor_status": "NOT_FROZEN_FOR_ENGINEERING_QC",
                       "offset_frames": 15 - min_idx,
                       "distance_by_query_lag_shape": list(matrix.shape),
                       "distance_by_query_lag_sha256": sha256_bytes(np.ascontiguousarray(matrix, dtype="<f4").tobytes())})
        tracks.append(matrix)
    if not curves:
        raise RuntimeError("official SyncNet pipeline produced no scored face tracks")
    npz_path = work / "syncnet" / f"{arm_id}_distance_by_query_lag.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=False)
    # Retain every detected track's true per-query by lag matrix.
    np.savez_compressed(npz_path, **{f"track_{i}_distance_by_query_lag": a for i, a in enumerate(tracks)},
                        lag_offsets_frames=np.arange(-15, 16, dtype=np.int32))
    return {"protocol": "official_wav2lip_syncnet_run_pipeline_plus_run_syncnet",
            "syncnet_root": str(syncnet_root), "syncnet_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=syncnet_root, text=True).strip(),
            "weight_path": str(syncnet_root / "data" / "syncnet_v2.model"),
            "weight_sha256": sha256_file(syncnet_root / "data" / "syncnet_v2.model"),
            "tracking_settings": {"facedet_scale": 0.25, "crop_scale": 0.40, "min_track": 5, "frame_rate": 25,
                                  "num_failed_det": 25, "min_face_size": 100},
            "pipeline_run": pipeline_run, "score_run": score_run,
            "activesd_pickle_path": str(active), "activesd_pickle_sha256": sha256_file(active),
            "distance_npz_path": str(npz_path), "distance_npz_sha256": sha256_file(npz_path),
            "tracks": curves}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/pilot/natural_1.wav")
    parser.add_argument("--face", default="/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png")
    parser.add_argument("--wav2lip-root", default="/root/autodl-tmp/wav2lip-syncnet/Wav2Lip")
    parser.add_argument("--wav2lip-python", default="/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
    parser.add_argument("--syncnet-root", default="/root/autodl-tmp/wav2lip-syncnet/syncnet_python")
    parser.add_argument("--syncnet-python", default="/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python")
    parser.add_argument("--output-root", default="/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt01")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--cwd")
    parser.add_argument("--checkpoint")
    parser.add_argument("--audio")
    parser.add_argument("--outfile")
    parser.add_argument("--wav2lip-root-child", dest="wav2lip_root_child")
    parser.add_argument("--face-child", dest="face_child")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.child:
        child_ns = argparse.Namespace(wav2lip_root=args.wav2lip_root_child, cwd=args.cwd,
                                      checkpoint=args.checkpoint, face=args.face_child,
                                      audio=args.audio, outfile=args.outfile, seed=args.seed)
        return generator_child(child_ns)

    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing attempt directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    face = Path(args.face).resolve()
    wav2lip_root = Path(args.wav2lip_root).resolve()
    syncnet_root = Path(args.syncnet_root).resolve()
    checkpoint = wav2lip_root / "checkpoints" / "wav2lip_gan.pth"
    python = Path(args.wav2lip_python).resolve()
    syncnet_python = Path(args.syncnet_python).resolve()
    result: dict[str, Any] = {
        "schema_version": "cross_tfg_oldpilot_engineering_qc_v1", "run_id": RUN_ID,
        "status": "RUNNING", "engineering_only": True, "eligible_for_candidate_selection": False,
        "sample_id": "natural_1", "historically_exposed": True,
        "source_path": str(source), "face_path": str(face), "observed_unix": time.time(),
        "rms_target": RMS_TARGET, "rows": [], "failures": []}
    atomic_json(root / "engineering_qc_manifest.json", result)
    try:
        if sha256_file(source) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("natural_1 source SHA mismatch")
        if sha256_file(face) != EXPECTED_FACE_SHA256:
            raise RuntimeError("primary face source SHA mismatch")
        source_files = {"wav2lip_inference_sha256": sha256_file(wav2lip_root / "inference.py"),
                        "wav2lip_audio_sha256": sha256_file(wav2lip_root / "audio.py"),
                        "wav2lip_hparams_sha256": sha256_file(wav2lip_root / "hparams.py"),
                        "wav2lip_checkpoint_sha256": sha256_file(checkpoint)}
        x, wavfile = canonicalize_with_wav2lip_loader(source, wav2lip_root)
        original_ref = write_float_wav(root / "audio" / "natural_official_load_f32.wav", x, wavfile)
        import numpy as np
        input_rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64)))
        gain = 1.0 if input_rms <= 1e-12 else RMS_TARGET / input_rms
        xrms = x.copy() if input_rms <= 1e-12 else np.ascontiguousarray(x * np.float32(gain), dtype=np.float32)
        rms_ref = write_float_wav(root / "audio" / "rms_target_0p0376838172675746_f32.wav", xrms, wavfile)
        audio_refs = {"identity": original_ref, "identity_repeat": original_ref, "rms_fixed": rms_ref}
        result["input"] = {"source_file_sha256": sha256_file(source), "face_file_sha256": sha256_file(face),
                           "canonicalization": "pinned Wav2Lip audio.load_wav(path,16000); no pre-mel peak normalization",
                           "natural_pcm_float32": original_ref, "sample_count": int(x.size), "input_rms": input_rms,
                           "fixed_rms_gain": gain, "rms_pcm_float32": rms_ref,
                           "source_sample_rate_policy": "official librosa.load(sr=16000, default mono conversion)"}
        result["pins"] = {"wav2lip_root": str(wav2lip_root), "wav2lip_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=wav2lip_root, text=True).strip(),
                          "wav2lip_python": str(python), "wav2lip_python_version": subprocess.check_output([str(python), "-c", "import torch,librosa,numpy;print(torch.__version__+'|librosa='+librosa.__version__+'|numpy='+numpy.__version__)"] ,text=True).strip(),
                          "syncnet_python": str(syncnet_python), "syncnet_python_version": subprocess.check_output([str(syncnet_python), "-c", "import torch,numpy;print(torch.__version__+'|numpy='+numpy.__version__)"] ,text=True).strip(),
                          **source_files}
        methods = [("identity", 20261003), ("rms_fixed", 20261003), ("identity_repeat", 20261003)]
        for method_id, seed in methods:
            row_root = root / "rows" / method_id
            row_root.mkdir(parents=True, exist_ok=False)
            child_cwd = row_root / "generator_cwd"
            child_cwd.mkdir()
            audio_path = Path(audio_refs[method_id]["path"])
            generated = row_root / "generated_video.mp4"
            child_argv = [str(python), str(Path(__file__).resolve()), "--child",
                          "--wav2lip-root-child", str(wav2lip_root), "--cwd", str(child_cwd),
                          "--checkpoint", str(checkpoint), "--face-child", str(face),
                          "--audio", str(audio_path), "--outfile", str(generated), "--seed", str(seed)]
            child_env = dict(os.environ)
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
            child_env["OMP_NUM_THREADS"] = "1"
            child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            gen = run(child_argv, cwd=child_cwd, log_stem=row_root / "logs" / "wav2lip", timeout=900, env=child_env)
            if gen["returncode"] != 0 or not generated.is_file() or generated.stat().st_size == 0:
                raise RuntimeError(f"Wav2Lip {method_id} generator failed ({gen['returncode']}); see {gen['stderr_path']}")
            silent = row_root / "silent_video.mkv"
            muxed = row_root / "original_pcm_f32_mux.mkv"
            strip = run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(generated), "-map", "0:v:0", "-c:v", "copy", "-an", "-map_metadata", "-1", "-f", "matroska", str(silent)], cwd=row_root, log_stem=row_root / "logs" / "strip_audio")
            if strip["returncode"] != 0:
                raise RuntimeError(f"strip generator audio failed for {method_id}; see {strip['stderr_path']}")
            mux = run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(silent), "-i", str(original_ref["path"]), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", "-map_metadata", "0", "-f", "matroska", str(muxed)], cwd=row_root, log_stem=row_root / "logs" / "mux_original_pcm")
            if mux["returncode"] != 0:
                raise RuntimeError(f"original PCM mux failed for {method_id}; see {mux['stderr_path']}")
            expected_pcm_sha = original_ref["pcm_sha256"]
            decoded = decode_f32(muxed)
            if len(decoded) != int(original_ref["sample_count"]) * 4 or sha256_bytes(decoded) != expected_pcm_sha:
                raise RuntimeError(f"PCM_F32LE mux readback mismatch for {method_id}")
            streams = read_ffprobe(muxed).get("streams", [])
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
            if len(video_streams) != 1 or len(audio_streams) != 1:
                raise RuntimeError(f"expected one video and one audio stream in {method_id} mux")
            if audio_streams[0].get("codec_name") != "pcm_f32le" or int(audio_streams[0].get("sample_rate", 0)) != 16000 or int(audio_streams[0].get("channels", 0)) != 1:
                raise RuntimeError(f"mux audio stream is not mono PCM_F32LE 16kHz for {method_id}")
            score = score_with_official_syncnet(muxed, row_root, syncnet_root, syncnet_python, method_id)
            result["rows"].append({"row_id": f"oldpilot_natural_1_wav2lip_primary_face_seed{seed}_{method_id}",
                "split": "engineering_old_exposed", "method_id": method_id, "seed": seed,
                "natural_pcm_sha256": expected_pcm_sha, "generator_audio_path": str(audio_path),
                "generator_audio_pcm_sha256": audio_refs[method_id]["pcm_sha256"],
                "generated_video_path": str(generated), "generated_video_sha256": sha256_file(generated),
                "generated_video_elementary_stream_sha256": video_stream_sha256(generated),
                "silent_video_path": str(silent), "silent_video_sha256": sha256_file(silent),
                "muxed_video_path": str(muxed), "muxed_video_sha256": sha256_file(muxed),
                "video_stream": video_streams[0], "mux_audio_readback": {"codec": audio_streams[0].get("codec_name"),
                    "sample_rate_hz": int(audio_streams[0].get("sample_rate")), "channels": int(audio_streams[0].get("channels")),
                    "sample_count": len(decoded)//4, "pcm_float32_sha256": sha256_bytes(decoded),
                    "matches_original_natural_pcm": True},
                "render_command": gen, "strip_command": strip, "mux_command": mux,
                "fresh_syncnet": score, "status": "PASS"})
            atomic_json(root / "engineering_qc_manifest.json", result)
        if len(result["rows"]) == 3:
            a, b = result["rows"][0], result["rows"][2]
            a_hashes = [item["distance_by_query_lag_sha256"] for item in a["fresh_syncnet"]["tracks"]]
            b_hashes = [item["distance_by_query_lag_sha256"] for item in b["fresh_syncnet"]["tracks"]]
            exact_video = a["generated_video_elementary_stream_sha256"] == b["generated_video_elementary_stream_sha256"]
            exact_matrix = a_hashes == b_hashes
            result["identity_repeat_qc"] = {"status": "EXACT_MATCH" if exact_video and exact_matrix else "NONDETERMINISTIC_DIFFERENCE",
                                              "identity_generated_video_elementary_stream_sha256_equal": exact_video,
                                              "identity_syncnet_per_query_matrix_sha256_equal": exact_matrix}
        result["status"] = "PASS_ENGINEERING_ONLY"
    except BaseException as exc:
        result["status"] = "FAILED_ENGINEERING_QC"
        result["failures"].append({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    result["ended_unix"] = time.time()
    atomic_json(root / "engineering_qc_manifest.json", result)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "manifest": str(root / "engineering_qc_manifest.json")}, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS_ENGINEERING_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
