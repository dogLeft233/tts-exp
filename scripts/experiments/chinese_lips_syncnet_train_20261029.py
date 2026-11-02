"""Controlled random-initialization SyncNet pilot; only negative loss weights differ.

This is an experimental training recipe, NOT a reproduction of the unpublished
SyncNet V2 training history. Importing this module needs numpy but not torch.
Canonical inputs are 224-square BGR uint8 at 25 fps and official int16-PCM
python_speech_features MFCCs (13 x time at 100 Hz). No per-window normalization.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

LAGS = tuple(range(-25, -5)) + tuple(range(6, 26))
PHONE_SILENCE = {"sil", "sp", "<sil>", "silence", ""}
PHONE_UNKNOWN = {"<unk>", "unk", "unknown", "None", "spn"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        rows = obj if isinstance(obj, list) else obj["rows"] if "rows" in obj else [obj]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip ids")
    required = {"id", "speaker", "split", "video_path", "mfcc_path", "frame_count", "audio_path"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"missing fields for {row.get('id')}: {required-set(row)}")
    train = {r["speaker"] for r in rows if r["split"] == "train"}
    val = {r["speaker"] for r in rows if r["split"] in {"val", "validation", "test"}}
    if train & val:
        raise ValueError(f"speaker leakage: {sorted(train & val)}")
    return rows


def phone_spans(row):
    spans = row.get("phone_spans")
    if spans is None and row.get("phone_spans_path"):
        spans = read_json(row["phone_spans_path"])
        if isinstance(spans, dict):
            spans = spans.get("phone_spans", spans.get("phones"))
    if not spans:
        raise ValueError(f"{row['id']}: real forced-aligned phone spans required")
    normalized = []
    for p in spans:
        if isinstance(p, dict):
            start, end = p["start"], p["end"]
            label = p.get("phone", p.get("label"))
        else:
            start, end, label = p
        if label is None or not (math.isfinite(float(start)) and math.isfinite(float(end))):
            raise ValueError("invalid phone span")
        if float(end) <= float(start):
            raise ValueError("nonpositive phone span")
        normalized.append((float(start), float(end), str(label)))
    normalized.sort()
    if any(a[1] > b[0] + .001 for a, b in zip(normalized, normalized[1:])):
        raise ValueError("overlapping phone alignment")
    return normalized


def phones_at_window(spans, frame, samples=10):
    """Equal-time samples cover the entire 200 ms window, not its center phone."""
    result = []
    for t in frame / 25 + (np.arange(samples) + .5) * (.2 / samples):
        labels = [label for start, end, label in spans if start <= t < end]
        if len(labels) != 1 or labels[0] in PHONE_UNKNOWN:
            return None
        result.append(labels[0])
    return result


def phone_similarity(a, b):
    if a is None or b is None or len(a) != 10 or len(b) != 10:
        raise ValueError("ten known full-window phone samples required")
    return sum(x == y for x, y in zip(a, b)) / 10


def voiced(phones):
    return int(sum(p not in PHONE_SILENCE for p in phones) >= len(phones) / 2)


def energy_bin(pcm, frame):
    segment = np.asarray(pcm[frame * 640:frame * 640 + 3200], np.float64)
    if len(segment) != 3200:
        raise ValueError("audio energy window not complete")
    rms = np.sqrt(np.mean(segment * segment)) / 32768
    dbfs = 20 * np.log10(max(rms, 1e-8))
    return int(np.clip(np.floor((dbfs + 80) / 10), 0, 7))


def max_window_start(row, mfcc_steps=None, audio_samples=None):
    maximum = int(row["frame_count"]) - 5
    if mfcc_steps is not None:
        maximum = min(maximum, (int(mfcc_steps) - 20) // 4)
    if audio_samples is not None:
        # Official psf: 25ms frame, 10ms hop. Twenty MFCC steps occupy
        # 19*160+400=3440 real samples, not merely 200ms=3200 samples.
        maximum = min(maximum, (int(audio_samples) - 3440) // 640)
    return maximum


def balancing_weights(strata, similarity_bins, max_weight=4.):
    """Keep both conditions on identical support and each stratum's total mass.

    Exact half/half weighting is allowed only when both classes have >=1/(2cap)
    empirical mass. Other strata are excluded from BOTH conditions, not clipped.
    This explicitly changes the estimand to common support and reports coverage.
    """
    if max_weight < 1:
        raise ValueError("max_weight must be >=1")
    strata = np.asarray(strata)
    labels = np.asarray(similarity_bins, np.int8)
    if strata.ndim != 2 or len(strata) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid strata or labels")
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[tuple(int(x) for x in s)].append(i)
    weights = np.zeros(len(labels), np.float32)
    audit = []
    for key, ix in sorted(groups.items()):
        ix = np.asarray(ix, np.int64)
        counts = np.bincount(labels[ix], minlength=2)
        eligible = counts.min() > 0 and len(ix) / (2 * counts.min()) <= max_weight
        if eligible:
            for group in (0, 1):
                weights[ix[labels[ix] == group]] = len(ix) / (2 * counts[group])
        mass_error = abs(float(weights[ix].sum(dtype=np.float64)) - len(ix)) if eligible else None
        if eligible and mass_error > 1e-5 * len(ix):
            raise AssertionError("negative stratum weight mass not preserved")
        audit.append(dict(stratum=list(key), total=len(ix), dissimilar=int(counts[0]),
                          similar=int(counts[1]), retained=bool(eligible),
                          mass_error=mass_error,
                          weighted_similar_fraction=.5 if eligible else None))
    keep = weights > 0
    return keep, weights, audit


def prepare_pairs(manifest, output, seed=1729, queries_per_clip=4, max_weight=4., threshold=.5):
    from scipy.io import wavfile
    rows = read_manifest(manifest)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "clips.json", rows)
    protocol = dict(status="FROZEN_BEFORE_TRAINING", source_sha256=sha(__file__),
        manifest_sha256=sha(manifest), seed=seed, queries_per_clip=queries_per_clip,
        lags_frames=LAGS, candidate_clock="video q vs audio q+lag; positive lag selects later audio",
        phone_similarity="ten equal-time phone labels across entire .2s window; exact agreement",
        phone_similarity_threshold=threshold, max_weight=max_weight,
        strata=["signed_lag", "positive_voiced", "negative_voiced", "positive_dBFS_bin", "negative_dBFS_bin"],
        support="drop entire strata from both arms unless both categories exist and exact half/half weight <=cap",
        loss="0.5*(mean(d_pos**2)+mean(w_neg*relu(margin-d_neg)**2))",
        main_endpoint="physical zero raw negative Euclidean; no per-clip peak correction",
        validation_gate=dict(auc_at_least=.8, absolute_peak_mae_ms_at_most=80,
                             within_one_frame_at_least=.7, pcm_shift_pass_fraction_at_least=.8,
                             pcm_shift_frames=[-4, 4], pcm_shift_tolerance_frames=1),
        pool_quality_gate=dict(both_voiced_pairs_at_least=10000, retained_train_speakers_at_least=30,
            validation_speakers_at_least=5, retained_candidate_fraction_at_least=.1,
            retained_both_voiced_fraction_at_least=.5),
        official_recipe_reproduction=False, acoustic_augmentation=False)
    write_json(out / "protocol.json", protocol)
    rng = np.random.default_rng(seed)
    arrays = defaultdict(list)
    val_queries, errors = [], []
    for clip_index, row in enumerate(rows):
        try:
            mfcc = np.load(row["mfcc_path"], mmap_mode="r")
            if mfcc.ndim != 2 or mfcc.shape[0] != 13 or not np.isfinite(mfcc).all():
                raise ValueError("MFCC must be finite 13 x T")
            sr, pcm = wavfile.read(row["audio_path"])
            if sr != 16000 or pcm.ndim != 1 or pcm.dtype != np.int16:
                raise ValueError("PCM must be mono int16 16000Hz")
            maximum = max_window_start(row, mfcc.shape[1], len(pcm))
            candidates = np.arange(25, maximum - 25 + 1)
            if len(candidates) == 0:
                raise ValueError("clip too short for complete +/-1000ms support")
            qs = sorted(rng.choice(candidates, size=min(queries_per_clip, len(candidates)), replace=False).tolist())
            if row["split"] != "train":
                val_queries.extend(dict(clip_index=clip_index, q=q) for q in qs)
                continue
            spans = phone_spans(row)
            for q in qs:
                positive = phones_at_window(spans, q)
                if positive is None:
                    continue
                for lag in LAGS:
                    negative = phones_at_window(spans, q + lag)
                    if negative is None:
                        continue
                    similarity = phone_similarity(positive, negative)
                    arrays["clip_index"].append(clip_index)
                    arrays["q"].append(q)
                    arrays["lag"].append(lag)
                    arrays["similar"].append(int(similarity >= threshold))
                    arrays["similarity"].append(similarity)
                    arrays["strata"].append([lag, voiced(positive), voiced(negative), energy_bin(pcm, q), energy_bin(pcm, q + lag)])
        except (ValueError, OSError, KeyError) as exc:
            errors.append(dict(id=row["id"], error=str(exc)))
    if not arrays["q"] or not val_queries:
        write_json(out / "preparation_failure.json", dict(errors=errors, negatives=len(arrays["q"]), val_queries=len(val_queries)))
        raise ValueError("no eligible training negatives or no independent validation queries")
    keep, weights, strata = balancing_weights(arrays["strata"], arrays["similar"], max_weight)
    if not keep.any():
        write_json(out / "preparation_failure.json", dict(errors=errors, strata=strata))
        raise ValueError("no common support for negative intervention")
    payload = {k: np.asarray(v, np.float32 if k == "similarity" else np.int32)[keep] for k, v in arrays.items()}
    payload["weight_n1"] = weights[keep]
    np.savez_compressed(out / "pairs.npz", **payload)
    write_json(out / "validation_queries.json", val_queries)
    audit = dict(candidate_negatives=len(keep), retained_negatives=int(keep.sum()),
                 retained_fraction=float(keep.mean()), strata=strata, excluded_clips=errors,
                 baseline_mean_weight=1., intervention_mean_weight=float(weights[keep].mean()),
                 intervention_max_weight=float(weights.max()),
                 train_clips=len(set(payload["clip_index"].tolist())),
                 val_queries=len(val_queries), pairs_sha256=sha(out / "pairs.npz"),
                 clips_sha256=sha(out / "clips.json"),
                 validation_queries_sha256=sha(out / "validation_queries.json"))
    retained_clips = sorted(set(payload["clip_index"].tolist()))
    retained_speakers = sorted({rows[i]["speaker"] for i in retained_clips})
    val_speakers = sorted({rows[q["clip_index"]]["speaker"] for q in val_queries})
    speech = (payload["strata"][:, 1] == 1) & (payload["strata"][:, 2] == 1)
    candidate_strata = np.asarray(arrays["strata"])
    by_vad = []
    for pv in (0, 1):
        for nv in (0, 1):
            candidate_mask = (candidate_strata[:, 1] == pv) & (candidate_strata[:, 2] == nv)
            retained_mask = (payload["strata"][:, 1] == pv) & (payload["strata"][:, 2] == nv)
            den = int(candidate_mask.sum())
            by_vad.append(dict(positive_voiced=pv, negative_voiced=nv, candidates=den,
                retained=int(retained_mask.sum()), retained_fraction=float(retained_mask.sum() / den) if den else None))
    audit.update(retained_clip_ids=[rows[i]["id"] for i in retained_clips],
        retained_speakers=retained_speakers, validation_speakers=val_speakers,
        retained_clip_hours=sum(rows[i]["frame_count"] for i in retained_clips) / 25 / 3600,
        retained_both_voiced_pairs=int(speech.sum()), retained_both_voiced_fraction=float(speech.mean()),
        by_vad=by_vad,
        similar_effective_fraction=float(np.sum(payload["weight_n1"] * payload["similar"]) / payload["weight_n1"].sum()))
    checks = dict(speech_pairs=int(speech.sum()) >= 10000,
        train_speakers=len(retained_speakers) >= 30, val_speakers=len(val_speakers) >= 5,
        retained_fraction=float(keep.mean()) >= .1, speech_fraction=float(speech.mean()) >= .5)
    audit["quality_gate"] = dict(checks=checks, passed=all(checks.values()),
        status="PASS" if all(checks.values()) else "CONTROL_FAILED",
        scope="metadata eligibility before training; does not depend on TTS or model scores")
    # The group index pass in balancing_weights checks each pair once (avoid
    # an O(number_of_strata * number_of_pairs) full-mask audit on 20h pools).
    mass_errors = [item["mass_error"] for item in strata if item["retained"]]
    audit["max_stratum_mass_error"] = max(mass_errors, default=0.)
    write_json(out / "weight_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"strata", "excluded_clips"}}), flush=True)


def import_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    return torch


def load_model(source_path, seed=1729, checkpoint=None, device="cpu"):
    torch = import_torch()
    torch.manual_seed(seed)
    module_spec = importlib.util.spec_from_file_location("controlled_official_syncnet", source_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.S(num_layers_in_fc_layers=1024)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.to(device)


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ClipCache:
    def __init__(self, rows, max_clips=6):
        self.rows, self.max_clips, self.cache = rows, max_clips, OrderedDict()
        self.audio_cache = OrderedDict()

    def audio_window(self, clip_index, q):
        clip_index, q = int(clip_index), int(q)
        if clip_index not in self.audio_cache:
            value = np.asarray(np.load(self.rows[clip_index]["mfcc_path"]), np.float32)
            if value.ndim != 2 or value.shape[0] != 13 or not np.isfinite(value).all():
                raise ValueError("MFCC must be finite13xT")
            self.audio_cache[clip_index] = value
        self.audio_cache.move_to_end(clip_index)
        while len(self.audio_cache) > 128:
            self.audio_cache.popitem(last=False)
        a = self.audio_cache[clip_index][:, 4 * q:4 * q + 20]
        if q < 0 or a.shape != (13, 20):
            raise ValueError("incomplete audio window; padding forbidden")
        return a[None].copy()

    def get(self, clip_index):
        import cv2
        if clip_index in self.cache:
            self.cache.move_to_end(clip_index)
            return self.cache[clip_index]
        row = self.rows[clip_index]
        cap = cv2.VideoCapture(row["video_path"])
        if abs(cap.get(cv2.CAP_PROP_FPS) - 25) > .001:
            raise ValueError("input video fps !=25")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError("input video must be 224x224 BGR uint8")
            frames.append(frame)
        cap.release()
        if len(frames) != int(row["frame_count"]):
            raise ValueError("decoded frame count differs from manifest")
        self.audio_window(clip_index, 0)
        value = (np.stack(frames), self.audio_cache[clip_index])
        self.cache[clip_index] = value
        while len(self.cache) > self.max_clips:
            self.cache.popitem(last=False)
        return value

    def window(self, clip_index, q, lag):
        frames, _ = self.get(int(clip_index))
        v = frames[int(q):int(q) + 5]
        a = self.audio_window(clip_index, int(q) + int(lag))
        if v.shape != (5, 224, 224, 3):
            raise ValueError("incomplete AV window; padding forbidden")
        return np.transpose(v, (3, 0, 1, 2)).astype(np.float32), a


def auc_from_scores(positive, negative):
    """Tie-correct Mann-Whitney AUC, implemented without sklearn dependency."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        raise ValueError("AUC needs positive and negative scores")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def summarize_validation(zero_distances, negative_distances, curve_distances, curve_lags):
    d0, dn, curves = np.asarray(zero_distances), np.asarray(negative_distances), np.asarray(curve_distances)
    peaks = np.asarray(curve_lags)[curves.argmin(axis=1)]
    auc = auc_from_scores(-d0, -dn)
    mae = float(np.abs(peaks).mean() * 40)
    near = float((np.abs(peaks) <= 1).mean())
    return dict(auc=auc, positive_distance=float(d0.mean()), negative_distance=float(dn.mean()),
                absolute_peak_mae_ms=mae, within_one_frame=near, queries=len(d0),
                gate_pass=bool(auc >= .8 and mae <= 80 and near >= .7),
                gate_scope="held-out same-recording AV physical-zero vs wrong-time; no TTS selection",
                peak_lags=peaks.tolist())


def evaluate(model, rows, queries, device, max_queries=128, batch_size=32):
    torch = import_torch()
    import torch.nn.functional as F
    model.eval()
    cache = ClipCache(rows)
    lags = np.arange(-10, 11)
    chosen = sorted(queries, key=lambda q: (int(q["clip_index"]), q["q"]))
    if len(chosen) > max_queries:
        speakers = {rows[int(q["clip_index"])]["speaker"] for q in chosen}
        if max_queries < len(speakers):
            raise ValueError("validation budget cannot cover every held-out speaker")
        selected = set()
        for speaker in sorted(speakers):
            selected.add(next(i for i, q in enumerate(chosen) if rows[int(q["clip_index"])]["speaker"] == speaker))
        remaining = [i for i in range(len(chosen)) if i not in selected]
        extra = np.linspace(0, len(remaining) - 1, max_queries - len(selected), dtype=int)
        ix = sorted(selected | {remaining[i] for i in extra})
        chosen = [chosen[i] for i in ix]
    curves, zeros, negatives, details, pcm_controls = [], [], [], [], []
    shifted_mfcc_cache = {}
    with torch.inference_mode():
        for query in chosen:
            clip, q = int(query["clip_index"]), int(query["q"])
            video, _ = cache.window(clip, q, 0)
            v = model.forward_lip(torch.from_numpy(video[None]).to(device))
            audio = np.stack([cache.audio_window(clip, q + int(lag)) for lag in lags])
            values = []
            for start in range(0, len(audio), batch_size):
                a = model.forward_aud(torch.from_numpy(audio[start:start + batch_size]).to(device))
                distance = F.pairwise_distance(v.expand(len(a), -1), a, p=2, eps=1e-6)
                values.extend(distance.cpu().tolist())
            curve = np.asarray(values)
            curves.append(curve)
            zeros.append(curve[10])
            negatives.extend(curve[np.abs(lags) >= 6].tolist())
            details.append(dict(id=rows[clip]["id"], speaker=rows[clip]["speaker"], q=q))
            # Real waveform shift, followed by the official MFCC frontend again.
            # All evaluated samples remain well inside real original support.
            if clip not in shifted_mfcc_cache:
                shifted_mfcc_cache = {clip: shifted_mfccs(rows[clip]["audio_path"])}
            base_peak = int(lags[curve.argmin()])
            for shift, mfcc in shifted_mfcc_cache[clip].items():
                audio_shift = np.stack([mfcc[:, 4 * (q + lag):4 * (q + lag) + 20][None] for lag in lags])
                shifted_values = []
                for start in range(0, len(audio_shift), batch_size):
                    a = model.forward_aud(torch.from_numpy(audio_shift[start:start + batch_size]).to(device))
                    shifted_values.extend(F.pairwise_distance(v.expand(len(a), -1), a, p=2, eps=1e-6).cpu().tolist())
                shifted_peak = int(lags[np.argmin(shifted_values)])
                error = shifted_peak - base_peak - shift
                pcm_controls.append(dict(id=rows[clip]["id"], q=q, shift_frames=shift,
                    baseline_peak=base_peak, shifted_peak=shifted_peak,
                    relative_error_frames=error, passed=bool(abs(error) <= 1)))
    result = summarize_validation(zeros, negatives, curves, lags)
    result["query_details"] = details
    result["speakers"] = sorted({d["speaker"] for d in details})
    result["curve_lags"] = lags.tolist()
    result["curves"] = np.asarray(curves).tolist()
    result["pcm_controls"] = pcm_controls
    result["pcm_shift_pass_fraction"] = sum(x["passed"] for x in pcm_controls) / len(pcm_controls)
    result["gate_pass_without_pcm_controls"] = result["gate_pass"]
    result["gate_pass"] = bool(result["gate_pass"] and result["pcm_shift_pass_fraction"] >= .8)
    model.train()
    return result


def shifted_mfccs(audio_path):
    from scipy.io import wavfile
    import python_speech_features as psf
    sr, pcm = wavfile.read(audio_path)
    if sr != 16000 or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError("PCM controls require canonical mono16kint16")
    answer = {}
    for shift in (-4, 4):
        samples = abs(shift) * 640
        if shift > 0:
            shifted = np.concatenate([np.zeros(samples, pcm.dtype), pcm[:-samples]])
        else:
            shifted = np.concatenate([pcm[samples:], np.zeros(samples, pcm.dtype)])
        answer[shift] = psf.mfcc(shifted, sr).T.astype(np.float32)
    return answer


def contrastive_loss(audio_positive, audio_negative, video, weights, margin):
    torch = import_torch()
    import torch.nn.functional as F
    positive = F.pairwise_distance(video, audio_positive, p=2, eps=1e-6)
    negative = F.pairwise_distance(video, audio_negative, p=2, eps=1e-6)
    loss = .5 * (positive.square().mean() + (weights * torch.relu(margin - negative).square()).mean())
    return loss, positive, negative


def save_checkpoint(path, payload):
    torch = import_torch()
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(args):
    torch = import_torch()
    root, out = Path(args.pairs), Path(args.output)
    out.mkdir(parents=True, exist_ok=args.resume)
    rows = read_manifest(root / "clips.json")
    queries = read_json(root / "validation_queries.json")
    pairs = np.load(root / "pairs.npz")
    audit = read_json(root / "weight_audit.json")
    if not audit.get("quality_gate", {}).get("passed", False):
        raise ValueError("frozen negative pool fails metadata quality gate; formal two-arm training prohibited")
    for name in ("pairs.npz", "clips.json", "validation_queries.json"):
        key = {"pairs.npz": "pairs_sha256", "clips.json": "clips_sha256",
               "validation_queries.json": "validation_queries_sha256"}[name]
        if sha(root / name) != audit[key]:
            raise ValueError(f"frozen pool identity mismatch: {name}")
    count = len(pairs["q"])
    if count < args.batch_size:
        raise ValueError("fewer retained pairs than a batch")
    if any(rows[int(i)]["split"] != "train" for i in np.unique(pairs["clip_index"])):
        raise ValueError("nontraining clips in training pairs")
    if any(rows[int(q["clip_index"])]["split"] == "train" for q in queries):
        raise ValueError("training clips in validation queries")
    if not all(len(pairs[k]) == count for k in pairs.files):
        raise ValueError("training array lengths differ")
    if not np.isfinite(pairs["weight_n1"]).all() or np.any(pairs["weight_n1"] <= 0):
        raise ValueError("invalid frozen negative weights")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = load_model(args.model_source, args.seed, device=args.device)
    initial_digest = state_digest(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    protocol = dict(vars(args), initial_model_state_sha256=initial_digest,
                    trainer_source_sha256=sha(__file__),
                    model_source_sha256=sha(args.model_source), pairs_sha256=sha(root / "pairs.npz"),
                    pool_protocol_sha256=sha(root / "protocol.json"),
                    selection="validation gate_pass first, then max AUC, then min absolute peak MAE; TTS never consulted",
                    init="official S constructor random parameters, fresh BN, BOTH encoders train")
    step, best = 0, (False, -float("inf"), -float("inf"))
    if args.resume:
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        for k in ("seed", "condition", "batch_size", "lr", "margin", "steps", "grad_clip",
                  "val_every", "val_queries", "cpu_threads", "pairs_sha256", "model_source_sha256",
                  "trainer_source_sha256", "pool_protocol_sha256", "selection"):
            if protocol[k] != state["protocol"][k]:
                raise ValueError(f"resume protocol mismatch: {k}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for k, v in item.items():
                if torch.is_tensor(v):
                    item[k] = v.to(args.device)
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, best = state["step"], tuple(state["best"])
    else:
        write_json(out / "protocol.json", protocol)
        save_checkpoint(out / "initial.pt", dict(model=model.state_dict(), protocol=protocol, step=0))
        baseline = evaluate(model, rows, queries, args.device, args.val_queries, args.batch_size * 2)
        write_json(out / "validation_000000.json", baseline)
    model.train()
    cache = ClipCache(rows, args.cache_clips)
    started, last_log = time.time(), time.time()
    log_path = out / "train.jsonl"
    # A per-epoch fixed permutation permits exact replay from step without saving millions of indexes.
    current_epoch, order = None, None
    for step in range(step + 1, args.steps + 1):
        offset = (step - 1) * args.batch_size
        epoch, position = divmod(offset, count - count % args.batch_size)
        if epoch != current_epoch:
            order = np.random.default_rng(args.seed + 100003 * epoch).permutation(count)
            current_epoch = epoch
        ix = order[position:position + args.batch_size]
        batch = [cache.window(pairs["clip_index"][i], pairs["q"][i], pairs["lag"][i]) for i in ix]
        video = torch.from_numpy(np.stack([x[0] for x in batch])).to(args.device)
        negative = np.stack([x[1] for x in batch])
        positive = np.stack([cache.audio_window(pairs["clip_index"][i], pairs["q"][i]) for i in ix])
        aud = torch.from_numpy(np.concatenate([positive, negative])).to(args.device)
        weights = torch.from_numpy(pairs["weight_n1"][ix] if args.condition == "N1" else np.ones(len(ix), np.float32)).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        af = model.forward_aud(aud)
        vf = model.forward_lip(video)
        loss, pd, nd = contrastive_loss(af[:len(ix)], af[len(ix):], vf, weights, args.margin)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step{step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            record = dict(step=step, loss=float(loss.detach()), positive_distance=float(pd.detach().mean()),
                          negative_distance=float(nd.detach().mean()), grad_norm=float(grad),
                          active_negative_fraction=float((nd.detach() < args.margin).float().mean()),
                          positive_loss=float(pd.detach().square().mean()),
                          negative_loss=float((weights * torch.relu(args.margin - nd.detach()).square()).mean()),
                          batch_pair_indices=ix.tolist(), mean_negative_weight=float(weights.mean()),
                          elapsed_seconds=time.time() - started, utc_unix=time.time())
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            last_log = time.time()
        if step % args.val_every == 0 or step == args.steps:
            metric = evaluate(model, rows, queries, args.device, args.val_queries, args.batch_size * 2)
            write_json(out / f"validation_{step:06d}.json", metric)
            value = (metric["gate_pass"], metric["auc"], -metric["absolute_peak_mae_ms"])
            improved = value > best
            if improved:
                best = value
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), protocol=protocol,
                           step=step, best=best, validation=metric, torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [])
            save_checkpoint(out / "last.pt", payload)
            if improved:
                save_checkpoint(out / "best.pt", {k: v for k, v in payload.items() if k != "optimizer"})
            write_json(out / "status.json", dict(status="COMPLETE" if step == args.steps else "RUNNING",
                step=step, steps=args.steps, latest_validation=metric, best_gate_pass=best[0], best_validation_auc=best[1],
                latest_log_unix=last_log, elapsed_seconds=time.time() - started))
            print("VALIDATION", step, metric["auc"], metric["absolute_peak_mae_ms"], metric["gate_pass"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-pairs")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--queries-per-clip", type=int, default=4)
    p.add_argument("--max-weight", type=float, default=4.)
    p.add_argument("--similarity-threshold", type=float, default=.5)
    p = sub.add_parser("train")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--condition", choices=["N0", "N1"], required=True)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=10.)
    p.add_argument("--grad-clip", type=float, default=10.)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-queries", type=int, default=128)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--cache-clips", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-pairs":
        prepare_pairs(args.manifest, args.output, args.seed, args.queries_per_clip, args.max_weight, args.similarity_threshold)
    else:
        train(args)


if __name__ == "__main__":
    main()
