"""Small, explicit audio interventions; no score-based fitting or time warp."""

from __future__ import annotations

import argparse
import math
import re
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import signal

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.audio import (
    write_pcm16,
)
from scripts.experiments.static_image_bridge.common import (
    file_sha256,
    read_json,
    read_pcm16,
    write_json_atomic,
)

FP_ROOT = Path(
    "/home/wjj/.cache/nvidia-fastpitch-pilot-20260913/PyTorch/SpeechSynthesis/FastPitch"
)
MODEL_ROOT = Path("/home/wjj/.cache/natural-clock-pilot-models")


def encode(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid waveform")
    if np.max(np.abs(values)) >= 1:
        raise ValueError("clipping: do not silently normalize or retry")
    return np.rint(values * 32768).clip(-32768, 32767).astype("<i2")


def equalize(pcm, strength_db=2.0):
    """Fixed broad +2 dB high shelf, natural phase; global RMS restored."""
    x = pcm.astype(np.float64) / 32768
    freq, _, z = signal.stft(
        x, fs=16000, nperseg=800, noverlap=600, boundary="zeros", padded=True
    )
    shelf = 0.5 * (1 + np.tanh(np.log2(np.maximum(freq, 1) / 1500)))
    gain = 10 ** (strength_db * shelf / 20)
    _, y = signal.istft(
        z * gain[:, None], fs=16000, nperseg=800, noverlap=600, boundary=True
    )
    y = y[: len(x)]
    scale = np.sqrt(np.mean(x * x) / max(np.mean(y * y), 1e-20))
    y *= scale
    return encode(y), {
        "strength_db": strength_db,
        "knee_hz": 1500,
        "rms_scale": float(scale),
        "no_time_warp": True,
    }


# English MFA IPA inventory -> canonical CMU phones. Allophony/stress is not exact.
IPA = dict(
    zip(
        ["b", "d", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "v", "w", "z"],
        ["B", "D", "F", "HH", "Y", "K", "L", "M", "N", "P", "S", "T", "V", "W", "Z"],
    )
)
IPA.update(
    {
        "a": "AE1",
        "aj": "AY1",
        "aw": "AW1",
        "e": "EH1",
        "ej": "EY1",
        "i": "IY1",
        "o": "OW1",
        "æ": "AE1",
        "ð": "DH",
        "ŋ": "NG",
        "ɑ": "AA1",
        "ɒ": "AA1",
        "ɔ": "AO1",
        "ə": "AH0",
        "əw": "OW1",
        "ɚ": "ER0",
        "ɛ": "EH1",
        "ɜ": "ER1",
        "ɡ": "G",
        "ɪ": "IH1",
        "ɫ": "L",
        "ɹ": "R",
        "ʃ": "SH",
        "ʊ": "UH1",
        "u": "UW1",
        "θ": "TH",
        "ʒ": "ZH",
        "tʃ": "CH",
        "dʒ": "JH",
        "c": "K",
        "ɟ": "G",
        "ɖ": "D",
        "ʈ": "T",
        "ɲ": "N",
        "ʎ": "L",
        "ʋ": "V",
    }
)


def phone_intervals(path):
    text = path.read_text().split('name = "phones"', 1)[1]
    rows = [
        (float(a), float(b), label)
        for a, b, label in re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',
            text,
        )
    ]
    if (
        not rows
        or rows[0][0] != 0
        or any(abs(a[1] - b[0]) > 1e-6 for a, b in pairwise(rows))
    ):
        raise ValueError("MFA coverage/gap error")
    return rows


def canonical_phone(label):
    if not label:
        return " "
    cleaned = re.sub("[ːʰʲʷˈˌ]", "", label)
    if cleaned not in IPA:
        raise ValueError(f"unsupported IPA phone: {label}")
    return "@" + IPA[cleaned]


def duration_grid(intervals, samples):
    if abs(intervals[-1][1] - samples / 16000) > 0.002:
        raise ValueError("MFA duration differs from natural audio")
    edges = np.rint(
        np.array([a for a, _, _ in intervals] + [samples / 16000]) * 22050 / 256
    ).astype(int)
    edges[-1] = math.ceil(samples * 22050 / 16000 / 256)
    d = np.diff(edges)
    if np.any(d < 0) or d.sum() != edges[-1]:
        raise ValueError("invalid duration grid")
    return edges, d


def fastpitch_audios(protocol, root):
    import librosa
    import torch

    sys.path.insert(0, str(FP_ROOT))
    from common.layers import TacotronSTFT
    from common.text.symbols import get_symbols
    from fastpitch.model import FastPitch, regulate_len
    from hifigan.models import Generator

    torch.set_num_threads(2)
    torch.manual_seed(20260913)
    fp_path = MODEL_ROOT / "nvidia_fastpitch_210824.pt"
    hg_path = MODEL_ROOT / "hifigan_gen_checkpoint_10000_ft.pt"
    ck = torch.load(fp_path, map_location="cpu", weights_only=False)
    model = FastPitch(**ck["config"]).eval().cuda()
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in ck["state_dict"].items()}, strict=True
    )
    ck = torch.load(hg_path, map_location="cpu", weights_only=False)
    hg = Generator(ck["config"]).eval().cuda()
    hg.load_state_dict(ck["generator"], strict=True)
    symbols = {s: i for i, s in enumerate(get_symbols())}
    frontend = TacotronSTFT().eval()
    for r in protocol["records"]:
        sid = r["sample_id"]
        pcm, _ = read_pcm16(Path(r["audio"]["N"]["path"]))
        intervals = phone_intervals(Path(r["textgrid"]))
        edges, durations = duration_grid(intervals, len(pcm))
        phones = [canonical_phone(t) for _, _, t in intervals]
        x = signal.resample_poly(pcm.astype(np.float64) / 32768, 441, 320).astype(
            np.float32
        )
        f0, voiced, _ = librosa.pyin(
            x, sr=22050, fmin=50, fmax=600, frame_length=1024, hop_length=256
        )
        f0 = np.nan_to_num(f0)
        with torch.inference_mode():
            mel_n = frontend.mel_spectrogram(torch.from_numpy(x)[None])[0]
        energy = np.linalg.norm(mel_n.numpy(), axis=0)
        pitch_mean = float(model.pitch_mean[0]) or 218.14
        pitch_std = float(model.pitch_std[0]) or 67.24
        pitches, energies = [], []
        for j, (_, _, label) in enumerate(intervals):
            lo, hi = edges[j : j + 2]
            local = f0[lo:hi]
            valid = local[local > 0]
            pitches.append(
                float((valid.mean() - pitch_mean) / pitch_std)
                if label and len(valid)
                else 0.0
            )
            local_e = energy[lo:hi]
            energies.append(float(np.log1p(local_e.mean())) if len(local_e) else 0.0)
        ids = torch.tensor([[symbols[s] for s in phones]], device="cuda")
        d = torch.tensor(durations[None], device="cuda")
        pitch = torch.tensor([[pitches]], device="cuda", dtype=torch.float32)
        energy_t = torch.tensor([[energies]], device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            enc, _ = model.encoder(ids, conditioning=0)
            enc = enc + model.pitch_emb(pitch).transpose(1, 2)
            enc = enc + model.energy_emb(energy_t).transpose(1, 2)
            expanded, lens = regulate_len(d, enc)
            dec, _ = model.decoder(expanded, lens)
            mel = model.proj(dec).permute(0, 2, 1)
            y22 = hg(mel)[0, 0].cpu().numpy()
        y = signal.resample_poly(y22, 320, 441)[: len(pcm)]
        if len(y) != len(pcm):
            raise ValueError("synthesis underflow")
        out = root / "audio" / sid / "FASTPITCH.wav"
        write_pcm16(out, encode(y))
        write_json_atomic(
            out.with_suffix(".json"),
            {
                "method": "FastPitch external natural MFA durations, phone-averaged natural F0 and log-mel-norm energy",
                "sample_id": sid,
                "sha256": file_sha256(out),
                "model_sha256": file_sha256(fp_path),
                "vocoder_sha256": file_sha256(hg_path),
                "phones": phones,
                "source_intervals": intervals,
                "durations_frames": durations.tolist(),
                "pitch_normalized": pitches,
                "energy_log1p": energies,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "mel_frames": int(lens[0]),
                "output_samples": len(y),
                "raw_output_samples_22k": len(y22),
                "voiced_ratio": float(voiced.mean()),
                "max_boundary_quantization_ms": float(
                    np.max(
                        np.abs(
                            edges[:-1] * 256 / 22050
                            - np.array([a for a, _, _ in intervals])
                        )
                    )
                    * 1000
                ),
                "limitations": "single LJSpeech speaker, approximate IPA-to-CMU/allophony/stress, MFA labels estimated; not voice cloning or verified output alignment",
                "postprocess": "22.05k->16k resample; right crop at most one mel hop; no time warp; no loudness normalization",
            },
        )
        print("FASTPITCH_READY", sid, flush=True)


def dac_audios(protocol, root):
    import torch

    from scripts.experiments.lrs3_dac16k_codec_identity.audio import (
        _model_forward,
        load_dac_model,
    )
    from scripts.experiments.lrs3_dac16k_codec_identity.config import (
        DAC_CHECKPOINT,
        DAC_SOURCE_ROOT,
    )

    torch.set_num_threads(2)
    model = load_dac_model(DAC_SOURCE_ROOT, DAC_CHECKPOINT, "cuda")
    for r in protocol["records"]:
        pcm, _ = read_pcm16(Path(r["audio"]["N"]["path"]))
        result, qc = _model_forward(model, pcm, "cuda")
        out = root / "audio" / r["sample_id"] / "DAC.wav"
        write_pcm16(out, encode(result["waveform"]))
        write_json_atomic(
            out.with_suffix(".json"),
            {
                "method": "DAC16k full-codebook identity",
                "model_sha256": file_sha256(DAC_CHECKPOINT),
                "sha256": file_sha256(out),
                **result["provenance"],
                **qc,
            },
        )
        print("DAC_READY", r["sample_id"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["dac", "fastpitch"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    protocol = read_json(args.root / "protocol.json")
    (dac_audios if args.method == "dac" else fastpitch_audios)(protocol, args.root)


if __name__ == "__main__":
    main()
