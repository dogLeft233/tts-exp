"""Frozen local natural-phase/TTS-magnitude pilot on AISHELL-1.

The 17-record speech-residual cohort is discovery. The 27-record intraphone
cohort is a different-speaker evaluation cohort. Window selection uses audio
and MFA labels only; no video or SyncNet result enters candidate construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/local_tts_spectral_repair_20260925"
DISCOVERY = ROOT / "runs/mfa_linear_heldout_speech_20260925"
EVALUATION = ROOT / "runs/mfa_linear_intraphone_time_20260925"
TOKENS = ROOT / "runs/aishell1_qwen_mfa_linear_n100_20260816/03_tokens_paired/tokens.json"
WAV2LIP = ROOT / "third_party/Wav2Lip"
SYNCNET = ROOT / "third_party/syncnet_python"
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PY = Path("/home/wjj/.venvs/syncnet/bin/python")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")

ALPHA = 0.5
WINDOW_S = 0.40
FADE_S = 0.04
MIN_PHONE_S = 0.06
SILENCE_GUARD_S = 0.12
SHAM_GAP_S = 0.70
MIN_FLUX_ADVANTAGE = 0.0
ARMS = ("target", "sham")
REPEAT_CONTROL_IDS = ("a1_003", "a1_004", "a1_005")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def pcm(path: Path, *, allow_float: bool = False) -> np.ndarray:
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or (info.subtype != "PCM_16" and not (allow_float and info.subtype == "FLOAT")):
        raise ValueError(f"expected 16 kHz mono PCM16: {path}: {info}")
    if info.subtype == "FLOAT":
        values = sf.read(path, dtype="float64")[0]
        if not np.isfinite(values).all() or np.max(np.abs(values)) >= 1:
            raise ValueError(f"invalid float donor: {path}")
        return np.rint(values * 32768).astype(np.int16)
    return sf.read(path, dtype="int16")[0]


def write_pcm(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.int16), 16000, subtype="PCM_16")


def run(argv: list[str], log: Path, *, cwd: Path = ROOT) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def matching_token_pairs(natural: list[dict], tts: list[dict]) -> dict[int, int]:
    def speech(tokens: list[dict]) -> list[tuple[int, str]]:
        return [(i, str(token["token"])) for i, token in enumerate(tokens)
                if not token["is_silence"] and not token["is_unknown_speech"]]

    n, t = speech(natural), speech(tts)
    matcher = SequenceMatcher(None, [label for _, label in n], [label for _, label in t], autojunk=False)
    return {n[ni + step][0]: t[ti + step][0]
            for ni, ti, size in matcher.get_matching_blocks() for step in range(size)}


def flux_curve(values: np.ndarray) -> np.ndarray:
    import torch

    wave = torch.as_tensor(values.astype(np.float64) / 32768.0)
    spec = torch.stft(wave, n_fft=512, hop_length=160, win_length=512,
                      window=torch.hann_window(512, periodic=True, dtype=torch.float64),
                      center=True, pad_mode="reflect", return_complex=True)
    logmag = torch.log(torch.abs(spec).clamp_min(1e-7)).numpy()
    flux = np.mean(np.maximum(0.0, np.diff(logmag, axis=1, prepend=logmag[:, :1])), axis=0)
    return np.convolve(flux, np.ones(3) / 3, mode="same")


def choose_windows(natural_tokens: list[dict], tts_tokens: list[dict],
                   natural_pcm: np.ndarray, donor_pcm: np.ndarray) -> dict:
    """Rank matched speech transitions without visual or SyncNet information."""
    matched = matching_token_pairs(natural_tokens, tts_tokens)
    fn, fd = flux_curve(natural_pcm), flux_curve(donor_pcm)
    duration = len(natural_pcm) / 16000
    pauses = [(float(t["start_s"]), float(t["end_s"])) for t in natural_tokens if t["is_silence"]]
    candidates = []
    for i in range(len(natural_tokens) - 1):
        left, right = natural_tokens[i], natural_tokens[i + 1]
        if i not in matched or matched.get(i + 1) != matched[i] + 1:
            continue
        if left["is_silence"] or right["is_silence"] or left["is_unknown_speech"] or right["is_unknown_speech"]:
            continue
        if min(float(left["duration_s"]), float(right["duration_s"])) < MIN_PHONE_S:
            continue
        center = (float(left["end_s"]) + float(right["start_s"])) / 2
        lo, hi = center - WINDOW_S / 2, center + WINDOW_S / 2
        if lo < 0.25 or hi > duration - 0.25:
            continue
        if any(max(lo - SILENCE_GUARD_S, a) < min(hi + SILENCE_GUARD_S, b) for a, b in pauses):
            continue
        sample_lo, sample_hi = round(lo * 16000), round(hi * 16000)
        nrms = float(np.sqrt(np.mean((natural_pcm[sample_lo:sample_hi].astype(np.float64) / 32768) ** 2)))
        drms = float(np.sqrt(np.mean((donor_pcm[sample_lo:sample_hi].astype(np.float64) / 32768) ** 2)))
        if nrms < 0.003 or drms < 0.003 or not 0.5 <= drms / nrms <= 2.0:
            continue
        center_frame = round(center * 100)
        local = slice(max(0, center_frame - 4), min(len(fn), center_frame + 5))
        advantage = float(np.max(fd[local]) - np.max(fn[local]))
        candidates.append({"center_s": center, "start_s": lo, "end_s": hi,
                           "boundary_index": i, "left_phone": left["token"], "right_phone": right["token"],
                           "flux_advantage": advantage, "natural_rms": nrms, "donor_rms": drms})
    candidates.sort(key=lambda c: (-c["flux_advantage"], c["center_s"]))
    target = next((c for c in candidates if c["flux_advantage"] > MIN_FLUX_ADVANTAGE), None)
    sham = None
    if target is not None:
        eligible = [c for c in candidates if abs(c["center_s"] - target["center_s"]) >= SHAM_GAP_S]
        sham = min(eligible, key=lambda c: (c["flux_advantage"], c["center_s"]), default=None)
    return {"target": target, "sham": sham, "candidate_count": len(candidates),
            "top_candidates": candidates[:5]}


def natural_phase_blend(natural_pcm: np.ndarray, donor_pcm: np.ndarray) -> np.ndarray:
    import torch

    if natural_pcm.shape != donor_pcm.shape:
        raise ValueError("natural/donor sample clocks differ")
    n = torch.as_tensor(natural_pcm.astype(np.float64) / 32768)
    d = torch.as_tensor(donor_pcm.astype(np.float64) / 32768)
    window = torch.hann_window(1024, periodic=True, dtype=torch.float64)
    kwargs = {"n_fft": 1024, "hop_length": 256, "win_length": 1024, "window": window,
              "center": True, "pad_mode": "reflect", "return_complex": True}
    ns, ds = torch.stft(n, **kwargs), torch.stft(d, **kwargs)
    nm, dm = torch.abs(ns).clamp_min(1e-7), torch.abs(ds).clamp_min(1e-7)
    magnitude = torch.exp((1 - ALPHA) * torch.log(nm) + ALPHA * torch.log(dm))
    result = torch.istft(magnitude * (ns / nm), n_fft=1024, hop_length=256,
                         win_length=1024, window=window, center=True,
                         length=natural_pcm.size)
    return result.numpy()


def local_mask(size: int, region: dict | None) -> np.ndarray:
    mask = np.zeros(size, dtype=np.float64)
    if region is None:
        return mask
    a = max(0, round(region["start_s"] * 16000))
    b = min(size, round(region["end_s"] * 16000))
    fade = min(round(FADE_S * 16000), (b - a) // 2)
    if b <= a or fade < 1:
        raise ValueError("invalid local window")
    mask[a:b] = 1.0
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade, endpoint=False))
    mask[a:a + fade] = ramp
    mask[b - fade:b] = ramp[::-1]
    return mask


def render_audio(natural_pcm: np.ndarray, blend: np.ndarray, region: dict | None) -> tuple[np.ndarray, dict]:
    original = natural_pcm.astype(np.float64) / 32768.0
    mask = local_mask(len(original), region)
    if region is None:
        return natural_pcm.copy(), {"edited_samples": 0, "residual_scale": 0.0, "rms_scale": 1.0}
    active = mask > 0
    n_rms = np.sqrt(np.mean(original[active] ** 2))
    b_rms = np.sqrt(np.mean(blend[active] ** 2))
    if n_rms <= 0 or b_rms <= 0:
        raise ValueError("zero-energy edit window")
    rms_scale = float(n_rms / b_rms)
    residual = mask * (blend * rms_scale - original)
    scale = 1.0
    if np.max(np.abs(original + residual)) >= 0.999:
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = (low + high) / 2
            if np.max(np.abs(original + middle * residual)) < 0.999:
                low = middle
            else:
                high = middle
        scale = low
    y = np.rint(np.clip(original + scale * residual, -1, 32767 / 32768) * 32768).astype(np.int16)
    y[~active] = natural_pcm[~active]
    if not np.array_equal(y[~active], natural_pcm[~active]):
        raise AssertionError("unaltered samples changed")
    return y, {"edited_samples": int(np.count_nonzero(y != natural_pcm)),
               "mask_samples": int(np.count_nonzero(active)), "residual_scale": scale,
               "rms_scale": rms_scale,
               "edit_rms": float(np.sqrt(np.mean(((y.astype(np.float64) - natural_pcm) / 32768) ** 2)))}


def prepare() -> None:
    source = (("discovery", DISCOVERY), ("evaluation", EVALUATION))
    records = read(TOKENS)["records"]
    rows = []
    speakers = {}
    for split, root in source:
        manifest = read(root / "manifest.json")
        speakers[split] = {r["speaker"] for r in manifest["rows"]}
        for row in manifest["rows"]:
            sample_id = row["id"]
            n_path, d_path = Path(row["natural"]), Path(row["arms"]["F"]["audio"])
            n_pcm, d_pcm = pcm(n_path), pcm(d_path, allow_float=True)
            if len(n_pcm) != len(d_pcm) or sample_id not in records:
                raise ValueError(f"invalid source clock/alignment: {sample_id}")
            source_video = root / "video" / sample_id / "N" / "video.mp4"
            item = {"id": sample_id, "speaker": row["speaker"], "split": split,
                    "natural": str(n_path), "donor": str(d_path), "face": row["face"],
                    "natural_video": str(source_video), "target_frames": row["target_frames"],
                    "samples": len(n_pcm), "natural_sha256": sha(n_path), "donor_sha256": sha(d_path),
                    "tokens_sha256": sha(TOKENS)}
            item["selection"] = choose_windows(records[sample_id]["natural"]["tokens"],
                                               records[sample_id]["tts"]["tokens"], n_pcm, d_pcm)
            rows.append(item)
    if speakers["discovery"] & speakers["evaluation"]:
        raise ValueError("discovery/evaluation speaker sets overlap")
    write(OUT / "protocol.json", {"question": "Can a fixed audio-only transition selector transfer local TTS spectral magnitude into natural-phase speech and improve natural-track SyncNet?",
                                  "discovery_count": sum(r["split"] == "discovery" for r in rows),
                                  "evaluation_count": sum(r["split"] == "evaluation" for r in rows),
                                  "alpha": ALPHA, "window_s": WINDOW_S, "fade_s": FADE_S,
                                  "minimum_flux_advantage": MIN_FLUX_ADVANTAGE,
                                  "selection": "largest positive donor-minus-natural spectral-flux peak at a matched speech transition; sham is least-ranked transition >=0.7s away; no score access",
                                  "primary": "evaluation mean paired official Sync-C on same natural PCM, target minus natural video; speaker-cluster bootstrap CI",
                                  "secondary": "target minus sham, Sync-D, local fixed-natural-lag distance, audio change and quality"})
    write(OUT / "manifest.json", {"rows": rows})
    print(f"prepared {len(rows)} rows; target selected {sum(r['selection']['target'] is not None for r in rows)}", flush=True)


def audio() -> None:
    rows = read(OUT / "manifest.json")["rows"]
    for row in rows:
        n_pcm, d_pcm = pcm(Path(row["natural"])), pcm(Path(row["donor"]), allow_float=True)
        if sha(Path(row["natural"])) != row["natural_sha256"] or sha(Path(row["donor"])) != row["donor_sha256"]:
            raise ValueError(f"source changed: {row['id']}")
        blended = natural_phase_blend(n_pcm, d_pcm)
        for arm in ARMS:
            region = row["selection"][arm]
            output, detail = render_audio(n_pcm, blended, region)
            path = OUT / "audio" / row["id"] / f"{arm}.wav"
            write_pcm(path, output)
            if not np.array_equal(pcm(path), output):
                raise AssertionError(f"PCM roundtrip failed: {path}")
            row.setdefault("arms", {})[arm] = {"audio": str(path), "sha256": sha(path), **detail}
        print("audio", row["id"], row["split"], row["arms"]["target"]["edited_samples"], flush=True)
    write(OUT / "manifest.json", {"rows": rows})


def audit() -> None:
    """Verify target reachability at the actual frozen Wav2Lip mel frontend."""
    if str(WAV2LIP) not in sys.path:
        sys.path.insert(0, str(WAV2LIP))
    import audio as wav2lip_audio

    rows = []
    for row in read(OUT / "manifest.json")["rows"]:
        n = pcm(Path(row["natural"]))
        natural_mel = wav2lip_audio.melspectrogram(n.astype(np.float32) / 32768)
        arms = {}
        for arm in ARMS:
            output = pcm(Path(row["arms"][arm]["audio"]))
            region = row["selection"][arm]
            if len(output) != len(n):
                raise AssertionError("sample count changed")
            outside = local_mask(len(n), region) == 0
            if not np.array_equal(output[outside], n[outside]):
                raise AssertionError(f"PCM outside edit changed: {row['id']}/{arm}")
            mel = wav2lip_audio.melspectrogram(output.astype(np.float32) / 32768)
            if mel.shape != natural_mel.shape:
                raise AssertionError("mel shape changed")
            times = np.arange(mel.shape[1]) / 80
            difference = np.abs(mel - natural_mel)
            if region is None:
                inside = np.zeros(len(times), dtype=bool)
            else:
                inside = (times >= region["start_s"]) & (times <= region["end_s"])
            far = ~inside if region is None else ((times < region["start_s"] - .10) | (times > region["end_s"] + .10))
            arms[arm] = {"edit_mel_mae": float(difference[:, inside].mean()) if inside.any() else 0.0,
                         "far_mel_mae": float(difference[:, far].mean()) if far.any() else 0.0,
                         "peak": float(np.max(np.abs(output.astype(np.int32))) / 32768),
                         "outside_mask_pcm_identical": True}
        rows.append({"id": row["id"], "speaker": row["speaker"], "split": row["split"], "arms": arms})
        print("audit", row["id"], arms["target"]["edit_mel_mae"], flush=True)
    write(OUT / "audio_audit.json", {"rows": rows,
                                     "contract": "16 kHz mono PCM16, exact natural sample clock, outside-mask PCM identity, actual Wav2Lip mel change"})


def render(*, split: str | None = None, sample_id: str | None = None) -> None:
    from scripts.experiments.mfa_linear_video_retiming.generation import (
        canonicalize_tail,
    )

    rows = read(OUT / "manifest.json")["rows"]
    selected = [row for row in rows if (split is None or row["split"] == split)
                and (sample_id is None or row["id"] == sample_id)]
    if not selected:
        raise ValueError("no records matched render filter")
    import torch

    if torch.cuda.is_available():
        probe = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        if probe.returncode != 0:
            raise RuntimeError(f"GPU available to torch but resource probe failed: {probe.stderr.strip()}")
        for line in probe.stdout.splitlines():
            used, total, utilization = [int(x.strip()) for x in line.split(",")]
            if used / total > 0.50 or utilization > 70:
                raise RuntimeError(f"GPU already occupied: {used}/{total} MiB, {utilization}%")
    elif os.getloadavg()[0] > 0.60 * (os.cpu_count() or 1):
        raise RuntimeError(f"GPU unavailable and CPU heavily loaded: load={os.getloadavg()[0]:.1f}")
    for row in selected:
        matched = row.get("render_baseline", False)
        if matched and not torch.cuda.is_available():
            raise RuntimeError("matched render protocol requires CUDA for every arm")
        if not matched and not Path(row["natural_video"]).is_file():
            raise FileNotFoundError(f"baseline video pending: {row['natural_video']}")
        repeat = ("N_repeat",) if matched and row["id"] in REPEAT_CONTROL_IDS else ()
        for arm in (("N", *repeat, *ARMS) if matched else ARMS):
            natural_arm = arm in ("N", "N_repeat")
            if not natural_arm and row["selection"][arm] is None:
                continue  # Identity by protocol; score the N video in this arm.
            dest = OUT / "video" / row["id"] / arm
            raw, normalized, final = dest / "raw.mp4", dest / "normalized.mkv", dest / "video.mp4"
            if not raw.is_file():
                work = OUT / "work" / row["id"] / arm
                (work / "temp").mkdir(parents=True, exist_ok=True)
                dest.mkdir(parents=True, exist_ok=True)
                inference = EVALUATION / "inference_cached.py" if matched else WAV2LIP / "inference.py"
                boxes_args = ["--boxes_input", str(EVALUATION / "face_boxes_132.json")] if matched else []
                run([str(WAV2LIP_PY), str(inference), "--checkpoint_path",
                     str(WAV2LIP / "checkpoints/wav2lip_gan.pth"), "--face", row["face"],
                     "--audio", row["natural"] if natural_arm else row["arms"][arm]["audio"], "--outfile", str(raw),
                     "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth", *boxes_args],
                    OUT / "logs" / f"{row['id']}_{arm}_render.log", cwd=work)
            if not normalized.is_file():
                detail = canonicalize_tail(raw, normalized, target_frame_count=row["target_frames"], ffmpeg=FFMPEG)
                write(dest / "normalized.json", detail)
            if not final.is_file():
                run([str(FFMPEG), "-y", "-v", "error", "-i", str(normalized), "-i", row["natural"],
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "16",
                     "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "25", "-frames:v",
                     str(row["target_frames"]), "-c:a", "aac", "-b:a", "192k", str(final)],
                    OUT / "logs" / f"{row['id']}_{arm}_mux.log")
            print("render", row["id"], arm, flush=True)


def crop(row: dict, arm: str) -> tuple[Path, Path]:
    identity = arm == "N" or (arm in ARMS and row["selection"][arm] is None)
    video = Path(row["natural_video"]) if identity else OUT / "video" / row["id"] / arm / "video.mp4"
    root = OUT / "pipeline" / row["id"] / arm
    ref = f"local_{row['id']}_{arm}"
    crop_path = root / "pycrop" / ref / "00000.avi"
    pcm_path = root / "pyavi" / ref / "audio.wav"
    if not crop_path.is_file() or not pcm_path.is_file():
        run([str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(video), "--reference", ref,
             "--data_dir", str(root), "--min_track", "100", "--overwrite"],
            OUT / "logs" / f"{row['id']}_{arm}_pipeline.log", cwd=SYNCNET)
    if not crop_path.is_file() or not pcm_path.is_file():
        raise FileNotFoundError(f"pipeline crop missing: {row['id']}/{arm}")
    return crop_path, pcm_path


def score(*, split: str | None = None, sample_id: str | None = None, device: str = "cpu") -> None:
    from scripts.experiments.mfa_linear_vocoder_wav2lip_split import score_matrix
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    engine = SyncNetEngine(batch_size=16, device=device)
    try:
        rows = [row for row in read(OUT / "manifest.json")["rows"]
                if (split is None or row["split"] == split) and (sample_id is None or row["id"] == sample_id)]
        if not rows:
            raise ValueError("no records matched score filter")
        for row in rows:
            result_path = OUT / "scores" / f"{row['id']}.json"
            if result_path.is_file():
                continue
            repeat = ("N_repeat",) if row.get("render_baseline") and row["id"] in REPEAT_CONTROL_IDS else ()
            assets = {arm: crop(row, arm) for arm in ("N", *ARMS, *repeat)}
            pcm_hashes = {arm: hashlib.sha256(pcm(wav).tobytes()).hexdigest() for arm, (_, wav) in assets.items()}
            if len(set(pcm_hashes.values())) != 1:
                raise ValueError(f"different decoded natural tracks: {row['id']}: {pcm_hashes}")
            audio_features, _ = engine.extract_audio(assets["N"][1])
            visual = {}
            crop_frames = {}
            for arm, (video_crop, _) in assets.items():
                visual[arm], meta = engine.extract_visual(video_crop)
                crop_frames[arm] = meta["frame_count"]
            if len(set(crop_frames.values())) != 1:
                raise ValueError(f"crop frame mismatch: {row['id']}: {crop_frames}")
            if len({item.shape[0] for item in visual.values()}) != 1:
                raise ValueError(f"visual support mismatch: {row['id']}")
            matrices = {arm: engine.distance_matrix(item, audio_features) for arm, item in visual.items()}
            cells = {arm: score_matrix(matrix) for arm, matrix in matrices.items()}
            base_lag_index = 15 - cells["N"]["offset"]
            for arm in ARMS:
                region = row["selection"][arm]
                if region is None:
                    cells[arm]["local_fixed_lag_distance"] = None
                    continue
                centers = (np.arange(matrices[arm].shape[0]) + 2) / 25
                mask = (centers >= region["start_s"]) & (centers <= region["end_s"])
                if not np.any(mask):
                    raise ValueError("no local SyncNet windows")
                cells[arm]["local_fixed_lag_distance"] = float(matrices[arm][mask, base_lag_index].mean())
                cells[arm]["natural_local_fixed_lag_distance"] = float(matrices["N"][mask, base_lag_index].mean())
            write(result_path, {"id": row["id"], "speaker": row["speaker"], "split": row["split"],
                                "natural_pcm_sha256": pcm_hashes["N"], "crop_frames": crop_frames,
                                "score_device": device, "cells": cells})
            print("score", row["id"], {k: v["sync_c"] for k, v in cells.items()}, flush=True)
    finally:
        engine.close()


def cluster_interval(rows: list[dict], field: str, *, baseline: str = "N", draws: int = 10000,
                     weighting: str = "speaker") -> dict:
    """Resample whole speakers; keep the utterance and speaker estimands explicit."""
    if weighting not in ("speaker", "utterance"):
        raise ValueError(f"unknown weighting: {weighting}")
    by_speaker: dict[str, list[float]] = {}
    for row in rows:
        cells = row["cells"]
        value = (cells["target"][field] - cells[baseline][field]) if field != "sync_d" else (cells[baseline][field] - cells["target"][field])
        by_speaker.setdefault(row["speaker"], []).append(value)
    if not by_speaker:
        raise ValueError("no scored records")
    ordered = [v for _, v in sorted(by_speaker.items())]
    groups = np.array([np.mean(v) for v in ordered])
    weights = np.array([len(v) if weighting == "utterance" else 1 for v in ordered])
    rng = np.random.default_rng(20260925)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    bootstrap = (groups[indices] * weights[indices]).sum(axis=1) / weights[indices].sum(axis=1)
    return {"mean": float(np.average(groups, weights=weights)), "weighting": weighting,
            "ci95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "positive_speakers": int(np.sum(groups > 0)), "speaker_count": len(groups)}


def analyze() -> None:
    manifest = [row for row in read(OUT / "manifest.json")["rows"] if row["split"] == "evaluation"]
    results = []
    for row in manifest:
        path = OUT / "scores" / f"{row['id']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(read(path))
    summary = {"records": len(results), "speakers": len({r["speaker"] for r in results}),
               "selected": sum(r["selection"]["target"] is not None for r in manifest),
               "sync_c_target_minus_n": cluster_interval(results, "sync_c"),
               "sync_c_target_minus_sham": cluster_interval(results, "sync_c", baseline="sham"),
               "sync_d_improvement": cluster_interval(results, "sync_d"),
               "sync_c_target_wins": sum(r["cells"]["target"]["sync_c"] > r["cells"]["N"]["sync_c"] for r in results),
               "sync_c_sham_wins": sum(r["cells"]["sham"]["sync_c"] > r["cells"]["N"]["sync_c"] for r in results)}
    summary["utterance_weighted_primary"] = cluster_interval(results, "sync_c", weighting="utterance")
    summary["utterance_weighted_target_minus_sham"] = cluster_interval(results, "sync_c", baseline="sham", weighting="utterance")
    summary["utterance_weighted_sync_d_improvement"] = cluster_interval(results, "sync_d", weighting="utterance")
    summary["weighting_note"] = "Protocol primary is utterance-weighted; legacy top-level intervals are speaker-equal sensitivity estimates. Both bootstrap entire speakers."
    summary["natural_repeat_control"] = [{"id": r["id"],
                                           "delta_sync_c": r["cells"]["N_repeat"]["sync_c"] - r["cells"]["N"]["sync_c"]}
                                          for r in results if "N_repeat" in r["cells"]]
    summary["render_control"] = ("matched frozen-box fresh renders" if all(r.get("render_baseline") for r in manifest)
                                 else "CONFOUNDED: reused frozen-box N versus fresh-detection target/sham; not causal evidence")
    summary["mean_sync_c"] = {arm: float(np.mean([r["cells"][arm]["sync_c"] for r in results]))
                              for arm in ("N", *ARMS)}
    summary["local_fixed_natural_lag_diagnostic"] = {}
    for arm in ARMS:
        gains = [r["cells"][arm]["natural_local_fixed_lag_distance"] - r["cells"][arm]["local_fixed_lag_distance"]
                 for r in results if r["cells"][arm].get("local_fixed_lag_distance") is not None]
        summary["local_fixed_natural_lag_diagnostic"][arm] = {
            "selected_records": len(gains), "mean_distance_improvement": float(np.mean(gains)) if gains else None,
            "improved_records": sum(v > 0 for v in gains),
            "note": "Descriptive selected-window diagnostic, positive is better; not Sync-C or an independent truth metric."}
    provenance = {"script_sha256": sha(Path(__file__)), "manifest_sha256": sha(OUT / "manifest.json"),
                  "wav2lip_checkpoint_sha256": sha(WAV2LIP / "checkpoints/wav2lip_gan.pth")}
    write(OUT / "analysis.json", {"summary": summary, "per_record": results, "provenance": provenance,
                                  "interpretation_limit": "The evaluation utterances are disjoint and speaker-disjoint from the 17 discovery utterances, but speakers overlap older 10-record project diagnostics; this is an exploratory internal replication."})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audio", "audit", "render", "score", "analyze"))
    parser.add_argument("--split", choices=("discovery", "evaluation"))
    parser.add_argument("--sample-id")
    parser.add_argument("--matched-render", action="store_true",
                        help="Use separately stored matched frozen-box renders, including a fresh natural baseline")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.matched_render:
        parent = OUT
        OUT = parent / "matched_render"
        if args.stage not in ("render", "score", "analyze"):
            parser.error("matched-render applies only to render, score, analyze")
        if not (OUT / "manifest.json").is_file():
            manifest = read(parent / "manifest.json")
            manifest["rows"] = [r for r in manifest["rows"] if r["split"] == "evaluation"]
            for row in manifest["rows"]:
                row["prior_natural_video"] = row["natural_video"]
                row["natural_video"] = str(OUT / "video" / row["id"] / "N" / "video.mp4")
                row["render_baseline"] = True
            write(OUT / "manifest.json", manifest)
            protocol = read(parent / "protocol.json")
            protocol.update({"correction": "Parent result is confounded by fresh vs frozen face detection. All arms here rerendered with the same frozen boxes and inference entry point; audio selection and parameters unchanged.",
                             "boxes_sha256": sha(EVALUATION / "face_boxes_132.json"),
                             "inference_sha256": sha(EVALUATION / "inference_cached.py"),
                             "score_device": args.device})
            write(OUT / "protocol.json", protocol)
    if args.stage in ("render", "score"):
        kwargs = {"device": args.device} if args.stage == "score" else {}
        globals()[args.stage](split=args.split, sample_id=args.sample_id, **kwargs)
    else:
        if args.split is not None or args.sample_id is not None:
            parser.error("filters apply only to render and score")
        globals()[args.stage]()


if __name__ == "__main__":
    main()
