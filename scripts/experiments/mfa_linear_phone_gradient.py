"""Phone-aligned MFA-linear timing-gradient mechanism experiment (n=3).

The previous repair fell back to word timing.  This experiment makes that
failure observable: phones are paired only inside explicitly matched word
intervals, using a fixed IPA-to-articulatory-class alignment.  It then warps
the historical MFA-linear waveform and its Wav2Lip mel with the same gradual
clock: 0, .25, .5, .75, 1 of the way from M's phone boundaries to natural
boundaries.  No score-dependent selection is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PARENT = REPO / "runs/mfa_linear_mechanism_pilot_20260913"
NATURAL_PARENT = REPO / "runs/natural_clock_pilot_20260913_v2"
ROOT = REPO / "runs/mfa_linear_phone_gradient_20260913"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
IMAGE = REPO / "data/data/image/3.png"
IMAGE_RGB_SHA256 = "bd5659ec3560bea57c34aa98d97ec9f60916a85a9414e70a3cf7a538db3f3903"
BOX = (138, 90, 357, 387)
CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
MODEL = SYNCNET / "data/syncnet_v2.model"
SCORE_BOX = NATURAL_PARENT / "score_box_3.json"
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIDS = ("lrs3_6WeS1bXRBOk_00006", "lrs3_6ul2TSvUDog_00007", "lrs3_6wk4dkYSrV0_00006")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as h:
        if (h.getframerate(), h.getnchannels(), h.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"not 16 kHz mono PCM16: {path}")
        return np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").copy()


def put_pcm(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(16000)
        h.writeframes(np.asarray(values, dtype="<i2").tobytes())


def job(cmd: list[str], log: Path, *, gpu: bool = False) -> None:
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free disk; leave existing artifacts intact")
    if gpu:
        fields = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        if int(fields[0]) < 8000 or int(fields[1]) > 20:
            raise RuntimeError(f"GPU is not safely idle: {fields}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            cmd,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


_INTERVAL = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:?\s*xmin\s*=\s*([0-9.eE+-]+)\s*xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)


def tier(path: Path, name: str) -> list[Interval]:
    raw = path.read_text(encoding="utf-8")
    marker = f'name = "{name}"'
    start = raw.index(marker)
    end = raw.find("item [", start + len(marker))
    section = raw[start:] if end < 0 else raw[start:end]
    values = [
        Interval(float(a), float(b), label.strip().lower())
        for a, b, label in _INTERVAL.findall(section)
    ]
    if not values:
        raise ValueError(f"empty {name} tier: {path}")
    return values


def speech(items: list[Interval]) -> list[Interval]:
    return [item for item in items if item.label not in {"", "sil", "sp"}]


def phone_class(label: str) -> str:
    """Fixed broad classes absorb MFA's IPA allophones without word fallback."""
    x = (
        label.replace("ː", "")
        .replace("ʰ", "")
        .replace("ʲ", "")
        .replace("ʷ", "")
        .replace("̪", "")
    )
    if x in {"a", "ɑ", "ɒ", "ɐ", "æ"}:
        return "v_open"
    if x in {
        "e",
        "ɛ",
        "eː",
        "ej",
        "ə",
        "ɜ",
        "ɚ",
        "ɝ",
        "ɪ",
        "i",
        "iː",
        "ʉ",
        "ʉː",
        "ʊ",
        "o",
        "ow",
        "əw",
    }:
        return "v_midclose"
    if x in {"aj", "aw"}:
        return "v_diph"
    if x in {"p", "b", "m", "mʲ"}:
        return "labial"
    if x in {"f", "v", "ʋ", "w"}:
        return "labiodorsal"
    if x in {
        "t",
        "d",
        "ð",
        "n",
        "l",
        "ɫ",
        "s",
        "z",
        "c",
        "cʰ",
        "cʷ",
        "ʈ",
        "ɖ",
        "ɟ",
        "ɲ",
        "ʎ",
    }:
        return "coronal"
    if x in {"k", "ɡ", "ŋ", "h", "x"}:
        return "dorsal"
    if x in {"tʃ", "dʒ", "ʃ", "ʒ", "ɹ", "r", "j"}:
        return "sibilant_rhotic"
    return f"raw:{x}"


def lcs_words(
    natural: list[Interval], candidate: list[Interval]
) -> list[tuple[int, int]]:
    """Monotone exact-word matching; blank MFA intervals are excluded first."""
    n, m = speech(natural), speech(candidate)
    score = [[0] * (len(m) + 1) for _ in range(len(n) + 1)]
    move = [[""] * (len(m) + 1) for _ in range(len(n) + 1)]
    for i in range(len(n) - 1, -1, -1):
        for j in range(len(m) - 1, -1, -1):
            choices = [(score[i + 1][j], "n"), (score[i][j + 1], "m")]
            if n[i].label == m[j].label:
                choices.append((score[i + 1][j + 1] + 1, "both"))
            score[i][j], move[i][j] = max(
                choices, key=lambda item: (item[0], item[1] == "both")
            )
    pairs, i, j = [], 0, 0
    while i < len(n) and j < len(m):
        choice = move[i][j]
        if choice == "both":
            pairs.append((i, j))
            i += 1
            j += 1
        elif choice == "m":
            j += 1
        else:
            i += 1
    return [(natural.index(n[i]), candidate.index(m[j])) for i, j in pairs]


def phones_in_word(phones: list[Interval], word: Interval) -> list[Interval]:
    return [
        phone
        for phone in speech(phones)
        if phone.start < word.end - 1e-6 and phone.end > word.start + 1e-6
    ]


def align_word_phones(
    natural: list[Interval], candidate: list[Interval]
) -> list[tuple[int, int]]:
    """Needleman-Wunsch with fixed articulatory classes; no arbitrary index pairing."""
    score = np.zeros((len(natural) + 1, len(candidate) + 1), dtype=np.int32)
    step = [[""] * (len(candidate) + 1) for _ in range(len(natural) + 1)]
    score[:, 0] = -np.arange(len(natural) + 1)
    score[0, :] = -np.arange(len(candidate) + 1)
    for i in range(1, len(natural) + 1):
        step[i][0] = "n"
    for j in range(1, len(candidate) + 1):
        step[0][j] = "m"
    for i in range(1, len(natural) + 1):
        for j in range(1, len(candidate) + 1):
            a, b = (
                phone_class(natural[i - 1].label),
                phone_class(candidate[j - 1].label),
            )
            # Different broad classes are deliberately not pairable.
            options = [(score[i - 1, j] - 1, "n"), (score[i, j - 1] - 1, "m")]
            if a == b:
                options.append(
                    (
                        score[i - 1, j - 1]
                        + (2 if natural[i - 1].label == candidate[j - 1].label else 1),
                        "both",
                    )
                )
            score[i, j], step[i][j] = max(
                options, key=lambda item: (int(item[0]), item[1] == "both")
            )
    pairs, i, j = [], len(natural), len(candidate)
    while i or j:
        choice = step[i][j]
        if choice == "both":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif choice == "n":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def phone_anchors(
    natural_grid: Path, candidate_grid: Path, duration_s: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_words, m_words = tier(natural_grid, "words"), tier(candidate_grid, "words")
    n_phones, m_phones = tier(natural_grid, "phones"), tier(candidate_grid, "phones")
    word_pairs = lcs_words(n_words, m_words)
    anchors: list[tuple[float, float]] = [(0.0, 0.0), (duration_s, duration_s)]
    paired, n_total, m_total = 0, 0, 0
    traces = []
    for ni, mi in word_pairs:
        ns, ms = (
            phones_in_word(n_phones, n_words[ni]),
            phones_in_word(m_phones, m_words[mi]),
        )
        n_total += len(ns)
        m_total += len(ms)
        local = align_word_phones(ns, ms)
        paired += len(local)
        traces.append(
            {
                "word": n_words[ni].label,
                "natural_phone_count": len(ns),
                "candidate_phone_count": len(ms),
                "paired": len(local),
            }
        )
        for a, b in local:
            # Pair starts and ends: timing within a paired phone remains linear.
            anchors.extend([(ms[b].start, ns[a].start), (ms[b].end, ns[a].end)])
    anchors.sort(key=lambda item: (item[0], item[1]))
    # Collapse source-coordinate duplicates deterministically; endpoints are immutable.
    source, target = [], []
    for x, y in anchors:
        if source and abs(x - source[-1]) < 1e-7:
            target[-1] = max(target[-1], y)
        else:
            source.append(x)
            target.append(y)
    s, t = np.asarray(source), np.asarray(target)
    if len(s) < 8 or np.any(np.diff(s) <= 0) or np.any(np.diff(t) < 0):
        raise ValueError("phone anchors are insufficient or non-monotone")
    coverage_n = paired / max(1, n_total)
    coverage_m = paired / max(1, m_total)
    meta = {
        "matched_words": len(word_pairs),
        "natural_speech_words": len(speech(n_words)),
        "candidate_speech_words": len(speech(m_words)),
        "paired_phones": paired,
        "natural_speech_phones_in_matched_words": n_total,
        "candidate_speech_phones_in_matched_words": m_total,
        "natural_phone_coverage": coverage_n,
        "candidate_phone_coverage": coverage_m,
        "word_trace": traces,
    }
    if min(coverage_n, coverage_m) < 0.70:
        raise ValueError(f"phone mapping coverage below frozen 0.70 threshold: {meta}")
    return s, t, meta


def inverse_warp(
    values: np.ndarray,
    source_s: np.ndarray,
    natural_s: np.ndarray,
    strength: float,
    *,
    rate: float,
) -> np.ndarray:
    """Output lives on natural clock; inverse map samples M at blended anchors."""
    blended = (1.0 - strength) * source_s + strength * natural_s
    clock = np.arange(len(values), dtype=np.float64) / rate
    source_clock = np.interp(clock, blended, source_s)
    sampled = np.interp(
        source_clock * rate,
        np.arange(len(values)),
        np.asarray(values, dtype=np.float64),
    )
    return np.clip(np.rint(sampled), -32768, 32767).astype(np.int16)


def warp_mel(
    mel: np.ndarray, source_s: np.ndarray, natural_s: np.ndarray, strength: float
) -> np.ndarray:
    """Same physical time map, at Wav2Lip's 80 Hz mel centers."""
    blended = (1.0 - strength) * source_s + strength * natural_s
    target = (np.arange(mel.shape[1], dtype=np.float64) + 0.5) / 80.0
    source = np.interp(target, blended, source_s)
    result = np.empty_like(mel, dtype=np.float32)
    positions = source * 80.0 - 0.5
    grid = np.arange(mel.shape[1], dtype=np.float64)
    for row in range(mel.shape[0]):
        result[row] = np.interp(positions, grid, mel[row]).astype(np.float32)
    return result


def freeze() -> list[dict]:
    parent = json.loads((PARENT / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    for record in parent["records"]:
        if record["sample_id"] not in SIDS:
            continue
        candidate_grid = (
            PARENT / "repair_alignment/textgrids" / f"{record['sample_id']}.TextGrid"
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "natural": record["natural"],
                "mfa_linear": record["mfa_linear"],
                "natural_textgrid": record["natural_textgrid"],
                "candidate_textgrid": str(candidate_grid),
                "source_video": record["source_video"],
                "natural_sha256": sha(Path(record["natural"])),
                "mfa_linear_sha256": sha(Path(record["mfa_linear"])),
                "natural_textgrid_sha256": sha(Path(record["natural_textgrid"])),
                "candidate_textgrid_sha256": sha(candidate_grid),
            }
        )
    if [row["sample_id"] for row in rows] != list(SIDS):
        raise ValueError("frozen sample ordering changed")
    protocol = {
        "experiment": "mfa_linear_phone_gradient",
        "status": "frozen_before_generation",
        "sample_ids": list(SIDS),
        "strengths": list(STRENGTHS),
        "paths": ["waveform", "mel_direct"],
        "image": str(IMAGE),
        "image_rgb_sha256": IMAGE_RGB_SHA256,
        "box": list(BOX),
        "score_box": str(SCORE_BOX),
        "mapping": "within exact matched words, fixed broad IPA-class Needleman-Wunsch; unmatched phones do not become artificial pairs; require >=0.70 coverage both streams",
        "primary": "candidate audio on original natural video; candidate-driven static Wav2Lip video with candidate audio",
        "records": rows,
        "scope": "n=3, one static image, fixed strength grid; exploratory mechanism screen only",
    }
    write_json(ROOT / "protocol.json", protocol)
    return rows


def make_candidates(records: list[dict]) -> None:
    sys.path.insert(0, str(WAV2LIP))
    import audio

    for row in records:
        sid = row["sample_id"]
        n, m = pcm(Path(row["natural"])), pcm(Path(row["mfa_linear"]))
        if len(n) != len(m):
            raise ValueError("historical M and N length contract changed")
        source_s, natural_s, mapping = phone_anchors(
            Path(row["natural_textgrid"]),
            Path(row["candidate_textgrid"]),
            len(n) / 16000.0,
        )
        write_json(
            ROOT / "mapping" / f"{sid}.json",
            {
                "source_anchor_s": source_s.tolist(),
                "natural_anchor_s": natural_s.tolist(),
                **mapping,
            },
        )
        target_mel = np.asarray(
            audio.melspectrogram(m.astype(np.float32) / 32768.0), dtype=np.float32
        )
        for strength in STRENGTHS:
            name = f"P{round(strength * 100):03d}"
            audio_path = ROOT / "audio" / sid / f"{name}.wav"
            if not audio_path.exists():
                put_pcm(
                    audio_path,
                    inverse_warp(m, source_s, natural_s, strength, rate=16000.0),
                )
            mel_path = ROOT / "mels" / sid / f"{name}.npy"
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            if not mel_path.exists():
                np.save(
                    mel_path,
                    warp_mel(target_mel, source_s, natural_s, strength),
                    allow_pickle=False,
                )
            write_json(
                ROOT / "audio" / sid / f"{name}.json",
                {
                    "strength": strength,
                    "method": "phone-boundary blended inverse waveform warp",
                    "mapping": str(ROOT / "mapping" / f"{sid}.json"),
                    "sha256": sha(audio_path),
                    "samples": len(n),
                },
            )


def render_static(records: list[dict]) -> None:
    renderer = REPO / "scripts/experiments/static_image_bridge/render_worker.py"
    direct = REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"
    for row in records:
        sid = row["sample_id"]
        out = ROOT / "videos" / sid
        out.mkdir(parents=True, exist_ok=True)
        boxes = out / "boxes.json"
        if not boxes.exists():
            write_json(boxes, [list(BOX)] * 1000)
        for strength in STRENGTHS:
            name = f"P{round(strength * 100):03d}"
            wave = out / f"W_{name}.mkv"
            if not wave.exists():
                job(
                    [
                        str(WAV2LIP_PY),
                        str(renderer),
                        "--image",
                        str(IMAGE),
                        "--image-rgb-sha256",
                        IMAGE_RGB_SHA256,
                        "--audio",
                        str(ROOT / "audio" / sid / f"{name}.wav"),
                        "--box",
                        *map(str, BOX),
                        "--checkpoint",
                        str(CHECKPOINT),
                        "--ffmpeg",
                        "ffmpeg",
                        "--outfile",
                        str(wave),
                        "--result",
                        str(out / f"W_{name}.json"),
                        "--batch-size",
                        "4",
                    ],
                    ROOT / "logs" / f"render_{sid}_W_{name}.log",
                    gpu=True,
                )
            direct_out = out / f"M_{name}.mp4"
            if not direct_out.exists():
                job(
                    [
                        str(WAV2LIP_PY),
                        str(direct),
                        "render",
                        "--checkpoint",
                        str(CHECKPOINT),
                        "--face",
                        str(IMAGE),
                        "--mel",
                        str(ROOT / "mels" / sid / f"{name}.npy"),
                        "--outfile",
                        str(direct_out),
                        "--boxes-input",
                        str(boxes),
                        "--face-det-batch-size",
                        "4",
                        "--wav2lip-batch-size",
                        "4",
                        "--nosmooth",
                    ],
                    ROOT / "logs" / f"render_{sid}_M_{name}.log",
                    gpu=True,
                )


def score_static(records: list[dict]) -> None:
    scorer = REPO / "scripts/experiments/static_image_bridge/score_worker.py"
    for row in records:
        sid = row["sample_id"]
        for strength in STRENGTHS:
            name = f"P{round(strength * 100):03d}"
            audio_path = ROOT / "audio" / sid / f"{name}.wav"
            request = ROOT / "requests" / f"{sid}_{name}.json"
            write_json(request, {name: str(audio_path)})
            for prefix, suffix in (("W", ".mkv"), ("M", ".mp4")):
                arm, out = f"{prefix}_{name}", ROOT / "matrices" / sid
                done = out / f"{arm}__{name}__worker.json"
                if not done.exists():
                    job(
                        [
                            str(SYNCNET_PY),
                            str(scorer),
                            "--video",
                            str(ROOT / "videos" / sid / f"{arm}{suffix}"),
                            "--video-arm",
                            arm,
                            "--sample-id",
                            sid,
                            "--score-box",
                            str(SCORE_BOX),
                            "--audio-json",
                            str(request),
                            "--model",
                            str(MODEL),
                            "--output-dir",
                            str(out),
                            "--batch-size",
                            "20",
                        ],
                        ROOT / "logs" / f"score_{sid}_{arm}.log",
                        gpu=True,
                    )


def score_real(records: list[dict]) -> None:
    """One SyncNet model/load per record; source crops are shared over strengths."""
    import torch

    from scripts.experiments.local_swap_minimal_replay.media import (
        lock_source_crop,
    )
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        make_face_detector,
    )
    from scripts.experiments.static_image_bridge.score_worker import (
        audio_embedding,
        visual_embedding,
    )

    sys.path.insert(0, str(SYNCNET))
    from SyncNetInstance import SyncNetInstance

    support = {
        r["sample_id"]: r["support"]["primary"]
        for r in json.loads((NATURAL_PARENT / "protocol.json").read_text())["records"]
    }
    torch.set_num_threads(2)
    detector = make_face_detector()
    model = SyncNetInstance(device="cuda")
    model.loadParameters(str(MODEL))
    model.eval()
    for row in records:
        sid = row["sample_id"]
        source = Path(row["source_video"])
        _, crops, cropmeta = lock_source_crop(source, detector)
        visual, _ = visual_embedding(np.stack(crops), model, "cuda", torch)
        windows = np.asarray(support[sid])
        output = ROOT / "real" / sid
        output.mkdir(parents=True, exist_ok=True)
        results = {}
        for strength in STRENGTHS:
            name = f"P{round(strength * 100):03d}"
            aud, count, _ = audio_embedding(
                ROOT / "audio" / sid / f"{name}.wav", model, "cuda", torch
            )
            mat = np.full((min(len(visual), len(aud)), 31), np.nan)
            for lag in range(-15, 16):
                mat[windows, lag + 15] = torch.nn.functional.pairwise_distance(
                    visual[windows].float(), aud[windows + lag].float()
                ).numpy()
            path = output / f"{name}.npy"
            np.save(path, mat, allow_pickle=False)
            results[name] = {
                **metric(mat, windows),
                "matrix": str(path),
                "audio_count": count,
            }
        write_json(
            output / "result.json",
            {
                "sample_id": sid,
                "source_video": str(source),
                "source_sha256": sha(source),
                "crop": cropmeta,
                "support": windows.tolist(),
                "scores": results,
            },
        )


def metric(matrix: np.ndarray, support: np.ndarray | list[int]) -> dict:
    curve = np.asarray(matrix)[np.asarray(support)].mean(axis=0)
    k = int(np.argmin(curve))
    median = float(np.median(curve))
    return {
        "C": median - float(curve[k]),
        "D": float(curve[k]),
        "median": median,
        "lag": int(k - 15),
        "curve": [float(x) for x in curve],
    }


def existing_curve_diagnosis() -> dict:
    parent = json.loads((PARENT / "analysis.json").read_text())
    answer = []
    for row in parent["records"]:
        metric_row = row["metrics"]
        n, m = metric_row["V_N/A_N"], metric_row["V_M/A_M"]
        answer.append(
            {
                "sample_id": row["sample_id"],
                "delta_C": m["C"] - n["C"],
                "delta_D": m["D"] - n["D"],
                "lag_change": m["offset"] - n["offset"],
                "interpretation": "C includes both best-lag depth and off-lag curve shape; this compact historical JSON lacks curves, so full curve decomposition is deferred to newly stored matrices.",
            }
        )
    return {
        "historical_summary": answer,
        "limitation": "prior analysis.json stored C/D/offset only, not curve vectors; current run saves full matrices for all arms.",
    }


def analyse(records: list[dict]) -> None:
    support = {
        r["sample_id"]: np.asarray(r["support"]["primary"])
        for r in json.loads((NATURAL_PARENT / "protocol.json").read_text())["records"]
    }
    rows = []
    for row in records:
        sid = row["sample_id"]
        static, real = (
            {},
            json.loads((ROOT / "real" / sid / "result.json").read_text())["scores"],
        )
        for strength in STRENGTHS:
            name = f"P{round(strength * 100):03d}"
            static[name] = {}
            for prefix in ("W", "M"):
                worker = json.loads(
                    (
                        ROOT
                        / "matrices"
                        / sid
                        / f"{prefix}_{name}__{name}__worker.json"
                    ).read_text()
                )
                static[name][prefix] = metric(np.load(worker["matrix"]), support[sid])
        rows.append(
            {
                "sample_id": sid,
                "static": static,
                "real": real,
                "mapping": json.loads((ROOT / "mapping" / f"{sid}.json").read_text()),
            }
        )
    summary = {}
    for context, paths in (("static", ("W", "M")), ("real", ("W",))):
        for path in paths:
            base = np.asarray(
                [
                    (
                        row[context]["P000"][path]
                        if context == "static"
                        else row[context]["P000"]
                    )["C"]
                    for row in rows
                ]
            )
            base_d = np.asarray(
                [
                    (
                        row[context]["P000"][path]
                        if context == "static"
                        else row[context]["P000"]
                    )["D"]
                    for row in rows
                ]
            )
            base_median = np.asarray(
                [
                    (
                        row[context]["P000"][path]
                        if context == "static"
                        else row[context]["P000"]
                    )["median"]
                    for row in rows
                ]
            )
            for strength in STRENGTHS:
                name = f"P{round(strength * 100):03d}"
                selected = [
                    (
                        row[context][name][path]
                        if context == "static"
                        else row[context][name]
                    )
                    for row in rows
                ]
                now = np.asarray([value["C"] for value in selected])
                now_d = np.asarray([value["D"] for value in selected])
                now_median = np.asarray([value["median"] for value in selected])
                delta = now - base
                summary[f"{context}_{path}_{name}_minus_P000"] = {
                    "mean_delta_C": float(delta.mean()),
                    "mean_delta_curve_median": float((now_median - base_median).mean()),
                    "mean_D_improvement": float((base_d - now_d).mean()),
                    "values": delta.tolist(),
                    "positive": int((delta > 0).sum()),
                }
    write_json(ROOT / "existing_curve_diagnosis.json", existing_curve_diagnosis())
    write_json(ROOT / "analysis.json", {"rows": rows, "summary": summary})
    lines = [
        "# MFA-linear phone timing gradient",
        "",
        "固定3条、单图；MFA-linear音频分别向自然音素边界移动0/25/50/75/100%。每个强度用同一坐标场生成波形臂(W)与直接mel臂(M)，并同时在自然视频和静态生成视频上评分。自然视频只评可播放的波形候选；mel臂只用于静态生成机制对照。",
        "",
        "|路径/场景/强度|平均 ΔSync-C|曲线中位数变化|D改善|正向|",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in summary.items():
        lines.append(
            f"|{key}|{value['mean_delta_C']:+.3f}|{value['mean_delta_curve_median']:+.3f}|{value['mean_D_improvement']:+.3f}|{value['positive']}/3|"
        )
    lines += [
        "",
        "音素配对只发生在LCS精确匹配词内，使用冻结IPA宽类序列比对；映射覆盖率见analysis.json。若coverage不足0.70会停止，而不会回退成word-level。直接mel只用于机制对照，不是可播放音频方案。",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    records = freeze()
    if args.real_only:
        score_real(records)
        return
    make_candidates(records)
    render_static(records)
    score_static(records)
    # The real-video scorer shares SyncNet's Python dependencies, which are
    # intentionally isolated from Wav2Lip's render environment.
    job(
        [str(SYNCNET_PY), str(Path(__file__).resolve()), "--real-only"],
        ROOT / "logs" / "score_real.log",
        gpu=True,
    )
    analyse(records)
    print(ROOT / "report.md")


if __name__ == "__main__":
    main()
