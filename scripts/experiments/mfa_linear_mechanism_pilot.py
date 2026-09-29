"""Small, frozen mechanism screen for the MFA-linear replacement hypothesis.

This is deliberately a diagnostic, not an optimiser: three pre-existing LRS3
records and the already frozen frontal image are used without score selection.
It asks (1) whether the historical MFA-linear waveform is a positive control in
this static setting, (2) whether one output-alignment-based local repair helps,
and (3) whether a target Wav2Lip mel differs from a Griffin-Lim waveform route.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PARENT = REPO / "runs/natural_clock_pilot_20260913_v2"
MFA_PARENT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
ROOT = REPO / "runs/mfa_linear_mechanism_pilot_20260913"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
IMAGE = REPO / "data/data/image/3.png"
IMAGE_RGB_SHA256 = "bd5659ec3560bea57c34aa98d97ec9f60916a85a9414e70a3cf7a538db3f3903"
CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
MODEL = SYNCNET / "data/syncnet_v2.model"
SIDS = ("lrs3_6WeS1bXRBOk_00006", "lrs3_6ul2TSvUDog_00007", "lrs3_6wk4dkYSrV0_00006")
BOX = (138, 90, 357, 387)
SCORE_BOX = REPO / "runs/natural_clock_pilot_20260913_v2/score_box_3.json"
ARMS = ("N", "M", "M_REPAIR", "M_GL", "M_DIRECT_FIXED")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as h:
        if (h.getframerate(), h.getnchannels(), h.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"not 16k mono PCM16: {path}")
        return np.frombuffer(h.readframes(h.getnframes()), dtype="<i2")


def put_pcm(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(16000)
        h.writeframes(np.asarray(values, dtype="<i2").tobytes())


def run(cmd: list[str], log: Path, *, gpu: bool = False) -> None:
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free; leave all existing artifacts intact")
    if gpu:
        state = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        if int(state[0]) < 8000 or int(state[1]) > 20:
            raise RuntimeError(f"GPU is not safely idle: {state}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        result = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False,
                                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2"})
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def freeze() -> list[dict]:
    protocol = json.loads((PARENT / "protocol.json").read_text())
    by_id = {r["sample_id"]: r for r in protocol["records"]}
    records = []
    for sid in SIDS:
        r = by_id[sid]
        n = Path(r["audio"]["N"]["path"])
        m = MFA_PARENT / "02_candidate_audio_retry3/audio" / f"{sid}.wav"
        if not n.is_file() or not m.is_file():
            raise FileNotFoundError((n, m))
        records.append({"sample_id": sid, "transcript": r["transcript"], "natural": str(n), "mfa_linear": str(m),
                        "natural_textgrid": r["textgrid"], "source_video": r["source_video"],
                        "natural_sha256": sha(n), "mfa_linear_sha256": sha(m)})
    frozen = {"experiment": "mfa_linear_mechanism_pilot", "status": "frozen_before_scoring", "sample_ids": list(SIDS),
              "image": str(IMAGE), "image_container_sha256": sha(IMAGE), "image_rgb_sha256": IMAGE_RGB_SHA256, "generation_box": list(BOX), "score_box": str(SCORE_BOX),
              "records": records, "arms": list(ARMS),
              "questions": ["historical MFA-linear static positive control", "one MFA output-alignment local repair", "direct target mel versus Griffin-Lim waveform/re-extract route"],
              "scope": "n=3, one existing image, descriptive diagnostic; no tuning or generalization claim"}
    write_json(ROOT / "protocol.json", frozen)
    return records


def align_and_repair(records: list[dict]) -> None:
    from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
        AlignmentError,
        build_frame_mapping,
        parse_textgrid,
    )
    corpus, aligned = ROOT / "repair_alignment/input", ROOT / "repair_alignment/textgrids"
    if not aligned.exists():
        corpus.mkdir(parents=True, exist_ok=True)
        for r in records:
            shutil.copy2(r["mfa_linear"], corpus / f"{r['sample_id']}.wav")
            (corpus / f"{r['sample_id']}.lab").write_text(r["transcript"].lower() + "\n")
        run(["mfa", "align", "--clean", "--overwrite", str(corpus), "english_us_mfa", "english_mfa", str(aligned), "--single_speaker", "--num_jobs", "3"], ROOT / "logs/mfa_realign.log")
    for r in records:
        n, m = pcm(Path(r["natural"])), pcm(Path(r["mfa_linear"]))
        natural_tokens, m_tokens = parse_textgrid(Path(r["natural_textgrid"])), parse_textgrid(aligned / f"{r['sample_id']}.TextGrid")
        try:
            mapping, stats = build_frame_mapping(int(np.ceil(len(n) / 640)), int(np.ceil(len(m) / 640)), natural_tokens, m_tokens)
            alignment_level = "phone"
        except AlignmentError:  # MFA allophones can differ despite identical words; use the coarser aligned word clock.
            natural_tokens = word_intervals(Path(r["natural_textgrid"]))
            m_tokens = word_intervals(aligned / f"{r['sample_id']}.TextGrid")
            mapping, stats = build_frame_mapping(int(np.ceil(len(n) / 640)), int(np.ceil(len(m) / 640)), natural_tokens, m_tokens)
            alignment_level = "word_fallback_after_phone_label_mismatch"
        # A single fixed piecewise-linear coordinate field, derived only from output MFA alignment.
        target = np.arange(len(n), dtype=np.float64)
        anchors_t = np.array([0.0] + [(x.natural_frame_index + .5) * 640 for x in mapping] + [float(len(n) - 1)])
        anchors_s = np.array([0.0] + [((x.left_frame_index + x.interpolation_alpha) + .5) * 640 for x in mapping] + [float(len(m) - 1)])
        source = np.interp(target, anchors_t, anchors_s)
        repaired = np.interp(source, np.arange(len(m)), m.astype(np.float64))
        repaired = np.clip(np.rint(repaired), -32768, 32767).astype(np.int16)
        output = ROOT / "audio" / r["sample_id"] / "M_REPAIR.wav"
        put_pcm(output, repaired)
        write_json(output.with_suffix(".json"), {"source": r["mfa_linear"], "output_alignment": str(aligned / f"{r['sample_id']}.TextGrid"),
            "method": "single output-MFA-derived piecewise-linear waveform coordinate repair; no score selection", "alignment_level": alignment_level, "mapping": stats,
            "input_samples": len(m), "output_samples": len(repaired), "sha256": sha(output)})


def mel_and_gl(records: list[dict]) -> None:
    sys.path.insert(0, str(WAV2LIP))
    import audio
    import librosa
    for r in records:
        m = pcm(Path(r["mfa_linear"])).astype(np.float32) / 32768.0
        mel = np.asarray(audio.melspectrogram(m), dtype=np.float32)
        mpath = ROOT / "mels" / f"{r['sample_id']}_M.npy"; mpath.parent.mkdir(parents=True, exist_ok=True); np.save(mpath, mel, allow_pickle=False)
        # This intentionally uses a non-learned inversion: it is a mechanism contrast, not an audio proposal.
        gl = librosa.feature.inverse.mel_to_audio(mel, sr=16000, n_fft=800, hop_length=200, win_length=800, window="hann", center=True,
                                                  power=1.0, n_iter=64, fmin=55, fmax=7600, htk=True, norm="slaney")
        gl = signal.resample(gl, len(m)).astype(np.float32) if len(gl) != len(m) else gl.astype(np.float32)
        output = ROOT / "audio" / r["sample_id"] / "M_GL.wav"; put_pcm(output, np.clip(np.rint(gl * 32768), -32768, 32767).astype(np.int16))
        write_json(output.with_suffix(".json"), {"target_mel": str(mpath), "target_mel_sha256": sha(mpath), "method": "librosa mel_to_audio Griffin-Lim 64 iterations then exact-length resample", "sha256": sha(output)})


def word_intervals(path: Path) -> list[dict]:
    """Read non-empty word intervals; add edge silence so every frame has an owner."""
    text = path.read_text(encoding="utf-8")
    section = text.split('name = "words"', 1)[1].split('item [', 1)[0]
    pairs = [
        (float(a), float(b), label.strip().lower())
        for a, b, label in re.findall(
            r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"',
            section,
        )
    ]
    if not pairs:
        raise ValueError(f"no words in {path}")
    result = []
    for start, end, label in pairs:
        result.append({"label": label, "start_s": start, "end_s": end})
    return result


def materialize(records: list[dict]) -> None:
    for r in records:
        base = ROOT / "audio" / r["sample_id"]; base.mkdir(parents=True, exist_ok=True)
        for arm, src in (("N", Path(r["natural"])), ("M", Path(r["mfa_linear"]))):
            dest = base / f"{arm}.wav"
            if not dest.exists(): shutil.copy2(src, dest)


def render_and_score(records: list[dict]) -> None:
    renderer = REPO / "scripts/experiments/static_image_bridge/render_worker.py"
    scorer = REPO / "scripts/experiments/static_image_bridge/score_worker.py"
    direct = REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"
    for r in records:
        sid = r["sample_id"]; base = ROOT / "audio" / sid
        videos = ROOT / "videos" / sid; videos.mkdir(parents=True, exist_ok=True)
        for arm in ("N", "M", "M_REPAIR", "M_GL"):
            output = videos / f"{arm}.mkv"
            if not output.exists():
                run([str(WAV2LIP_PY), str(renderer), "--image", str(IMAGE), "--image-rgb-sha256", IMAGE_RGB_SHA256, "--audio", str(base / f"{arm}.wav"),
                     "--box", *map(str, BOX), "--checkpoint", str(CHECKPOINT), "--ffmpeg", "ffmpeg", "--outfile", str(output),
                     "--result", str(videos / f"{arm}.worker.json"), "--batch-size", "4"], ROOT / f"logs/render_{sid}_{arm}.log", gpu=True)
        direct_out = videos / "M_DIRECT_FIXED.mp4"
        if not direct_out.exists():
            boxes = videos / "M_DIRECT_FIXED.boxes.json"
            write_json(boxes, [list(BOX)] * 1000)
            run([str(WAV2LIP_PY), str(direct), "render", "--checkpoint", str(CHECKPOINT), "--face", str(IMAGE),
                 "--mel", str(ROOT / "mels" / f"{sid}_M.npy"), "--outfile", str(direct_out), "--boxes-input", str(boxes),
                 "--face-det-batch-size", "4", "--wav2lip-batch-size", "4", "--nosmooth"],
                ROOT / f"logs/render_{sid}_M_DIRECT_FIXED.log", gpu=True)
        audios = {arm: str(base / f"{arm}.wav") for arm in ("N", "M", "M_REPAIR", "M_GL")}
        request = ROOT / "requests" / f"{sid}.json"; write_json(request, audios)
        for arm in ARMS:
            video = direct_out if arm == "M_DIRECT_FIXED" else videos / f"{arm}.mkv"
            out = ROOT / "matrices" / sid
            done = [out / f"{arm}__{a}__worker.json" for a in audios]
            if not all(p.exists() for p in done):
                run([str(SYNCNET_PY), str(scorer), "--video", str(video), "--video-arm", arm, "--sample-id", sid,
                     "--score-box", str(SCORE_BOX), "--audio-json", str(request), "--model", str(MODEL), "--output-dir", str(out), "--batch-size", "20"],
                    ROOT / f"logs/score_{sid}_{arm}.log", gpu=True)


def metrics(matrix: np.ndarray, support: list[int]) -> dict:
    curve = matrix[np.asarray(support)].mean(0); k = int(np.argmin(curve))
    return {"C": float(np.median(curve) - curve[k]), "D": float(curve[k]), "offset": int(15 - k)}


def analyse(records: list[dict]) -> None:
    protocol = json.loads((PARENT / "protocol.json").read_text()); support = {r["sample_id"]: r["support"]["primary"] for r in protocol["records"]}
    rows = []
    for r in records:
        sid = r["sample_id"]; local = {}
        for v in ARMS:
            for a in ("N", "M", "M_REPAIR", "M_GL"):
                worker = json.loads((ROOT / "matrices" / sid / f"{v}__{a}__worker.json").read_text())
                local[f"V_{v}/A_{a}"] = metrics(np.load(worker["matrix"]), support[sid])
        rows.append({"sample_id": sid, "metrics": local})
    comparisons = {"M_native_minus_N": [], "M_replacement_minus_N": [], "repair_native_minus_M": [], "direct_M_minus_waveform_M": [], "direct_M_minus_GL_native": []}
    for row in rows:
        m = row["metrics"]
        comparisons["M_native_minus_N"].append(m["V_M/A_M"]["C"] - m["V_N/A_N"]["C"])
        comparisons["M_replacement_minus_N"].append(m["V_M/A_N"]["C"] - m["V_N/A_N"]["C"])
        comparisons["repair_native_minus_M"].append(m["V_M_REPAIR/A_M_REPAIR"]["C"] - m["V_M/A_M"]["C"])
        comparisons["direct_M_minus_waveform_M"].append(m["V_M_DIRECT_FIXED/A_M"]["C"] - m["V_M/A_M"]["C"])
        comparisons["direct_M_minus_GL_native"].append(m["V_M_DIRECT_FIXED/A_M"]["C"] - m["V_M_GL/A_M_GL"]["C"])
    summary = {key: {"values": values, "mean": float(np.mean(values)), "positive": int(sum(x > 0 for x in values))} for key, values in comparisons.items()}
    write_json(ROOT / "analysis.json", {"records": rows, "comparisons": summary})
    report = ["# MFA-linear mechanism pilot", "", "固定前3条LRS3与既有3号静态正脸图。所有结果仅为n=3机制筛查。", "", "|比较|平均 ΔSync-C|正向样本|", "|---|---:|---:|"]
    report += [f"|{k}|{v['mean']:+.3f}|{v['positive']}/3|" for k, v in summary.items()]
    report += ["", "M为历史MFA-linear；M_REPAIR为对M重新MFA后按自然时间的一次固定局部坐标修复。本批phone标签因MFA异音标注无法逐项匹配，三条均预先回退到同一份输出MFA word clock，故它不是phone级修复的有效检验。M_DIRECT_FIXED将M的Wav2Lip mel直接输入生成器，并与波形臂共用同一冻结人脸框；M_GL是同一mel经64轮Griffin-Lim重建、再由Wav2Lip提mel。direct-mel只用于机制定位，不能视为可播放音频方案。"]
    (ROOT / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    records = freeze(); materialize(records); align_and_repair(records); mel_and_gl(records); render_and_score(records); analyse(records)
    print(ROOT / "report.md")


if __name__ == "__main__":
    main()
