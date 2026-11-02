"""Native cloud-cohort evaluation of controlled, freshly trained SyncNet models.

This is an exploratory bridge to the old 20-source cohort, not an independent
confirmation set. Every encoder is forwarded again from canonical PCM and the
existing official face track; old V2 embeddings are never loaded. The primary
endpoint is physical-zero negative Euclidean distance. Search is auxiliary.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

LAGS = np.arange(-15, 16, dtype=np.int64)
ENDPOINTS = ("zero", "best", "background", "contrast", "zero_contrast", "search_uplift")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_speaker(row):
    """Known AISHELL IDs encode their actual source speaker, not face identity."""
    if row.get("speaker_id"):
        return str(row["speaker_id"]), "explicit_speaker_id"
    match = re.fullmatch(r"BAC009(S\d{4})W\d+", str(row.get("sample_id", row.get("source", ""))))
    if match:
        return match.group(1), "AISHELL_source_id_Sdddd"
    return str(row.get("sample_id", row.get("source", ""))), "source_only_unknown_speaker"


def validate_inputs(rows, expected_sources=20, expected_draws=3):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate media ID")
    groups = defaultdict(list)
    for row in rows:
        if not ID_RE.fullmatch(str(row["id"])):
            raise ValueError("unsafe media ID")
        q0, q1 = int(row["query_start"]), int(row["query_end_exclusive"])
        if q0 < 15 or q1 <= q0 or q1 + 15 > int(row["sync_windows"]):
            raise ValueError("common query support is incomplete")
        if q1 - q0 != int(row["query_count"]):
            raise ValueError("query count mismatch")
        groups[row["sample_id"]].append(row)
    if len(groups) != expected_sources:
        raise ValueError(f"expected {expected_sources} sources, received {len(groups)}")
    for source, group in groups.items():
        if sum(r["arm"] == "N" for r in group) != 1 or len(group) != expected_draws + 1:
            raise ValueError(f"expected one N plus {expected_draws} T media: {source}")
        if len({r["query_count"] for r in group}) != 1:
            raise ValueError("N/T common-window counts differ")
        if len({bool(r["strict_reference"]) for r in group}) != 1:
            raise ValueError("strict-source membership differs by arm")
        if len({source_speaker(r) for r in group}) != 1:
            raise ValueError("source speaker differs by arm")
    return groups


def score_matrix(audio, video, queries):
    audio, video = np.asarray(audio, np.float64), np.asarray(video, np.float64)
    queries = np.asarray(queries, np.int64)
    if audio.shape != video.shape or audio.ndim != 2 or audio.shape[1] != 1024:
        raise ValueError("expected equal 1024-dimensional encoder outputs")
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise ValueError("non-finite encoder outputs")
    if queries.size == 0 or queries.min() < 15 or queries.max() + 15 >= len(audio):
        raise ValueError("lag curve would use out-of-support features")
    # Same geometric definition as official F.pairwise_distance: eps is added
    # to each coordinate before the norm. Float64 only stabilizes aggregation.
    return np.stack([-np.linalg.norm(video[queries] - audio[queries + lag] + 1e-6, axis=1)
                     for lag in LAGS], axis=1)


def reduce_curve(matrix):
    matrix = np.asarray(matrix, np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(LAGS) or not np.isfinite(matrix).all():
        raise ValueError("expected finite query-by-31 matrix")
    curve = matrix.mean(axis=0)
    zero, best, background = float(curve[15]), float(curve.max()), float(np.median(curve))
    return dict(zero=zero, best=best, background=background,
                contrast=best-background, zero_contrast=zero-background,
                search_uplift=best-zero, peak_lag_frames=int(LAGS[curve.argmax()]),
                curve=curve.tolist())


def gate_passes(payload):
    """Fail closed; a successful process exit or low loss is not a capability gate."""
    gate = payload.get("gate", payload)
    if "gate_pass" in gate:
        return gate["gate_pass"] is True and gate.get("status", "PASS") == "PASS"
    if gate.get("status") == "PASS" and gate.get("passed", True) is True:
        return True
    return gate.get("passed") is True and gate.get("status", "PASS") == "PASS"


def make_pairs(items):
    natural = {r["source"]: r for r in items if r["arm"] == "N"}
    if len(natural) != sum(r["arm"] == "N" for r in items):
        raise ValueError("duplicate natural source")
    pairs = []
    for t in items:
        if t["arm"] == "N":
            continue
        n = natural[t["source"]]
        pairs.append(dict(id=t["id"], source=t["source"], speaker=t["speaker"],
                          speaker_provenance=t["speaker_provenance"],
                          strict_reference=t["strict_reference"], natural_id=n["id"],
                          delta={f: t["endpoints"][f] - n["endpoints"][f] for f in ENDPOINTS}))
    return sorted(pairs, key=lambda p: p["id"])


def bootstrap_source_equal(values, clusters, replicates=10000, seed=20261029):
    """Resample source-speaker clusters; keep all sources and draws together.

    The estimand is an equal-source mean, so each replicated speaker contributes
    every source they own and the replicate is divided by its source count.
    """
    keys = sorted(values)
    if not keys:
        raise ValueError("empty source set")
    by_cluster = defaultdict(list)
    for key in keys:
        by_cluster[clusters[key]].append(float(values[key]))
    labels = sorted(by_cluster)
    sums = np.array([sum(by_cluster[c]) for c in labels])
    counts = np.array([len(by_cluster[c]) for c in labels])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(labels), (replicates, len(labels)))
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return dict(mean=float(np.mean([values[k] for k in keys])),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                clusters=len(labels), sources=len(keys), bootstrap_replicates=replicates,
                cluster_unit="source_speaker_or_source_if_unknown",
                estimand="equal_source_mean; all T draws averaged within source")


def summarize_pairs(pairs, replicates=10000):
    sources = sorted({p["source"] for p in pairs})
    if not sources:
        return dict(sources=0, pairs=0, metrics={})
    clusters = {s: next(p["speaker"] for p in pairs if p["source"] == s) for s in sources}
    result = dict(sources=len(sources), pairs=len(pairs), metrics={})
    for endpoint in ENDPOINTS:
        values = {s: float(np.mean([p["delta"][endpoint] for p in pairs if p["source"] == s])) for s in sources}
        win_values = {s: float(np.mean([p["delta"][endpoint] > 0 for p in pairs if p["source"] == s])) for s in sources}
        d = [p["delta"][endpoint] for p in pairs]
        result["metrics"][endpoint] = dict(
            score_delta_in_this_model_units=bootstrap_source_equal(values, clusters, replicates),
            tts_winrate=bootstrap_source_equal(win_values, clusters, replicates),
            tts_higher=sum(v > 0 for v in d), natural_higher=sum(v < 0 for v in d),
            ties=sum(v == 0 for v in d), total_pairs=len(d), source_means=values)
    return result


def compare_models(pair_sets, replicates=10000):
    if set(pair_sets) != {"N0", "N1"}:
        raise ValueError("cross-condition comparison requires exactly N0 and N1")
    indexed = {m: {p["id"]: p for p in ps} for m, ps in pair_sets.items()}
    if set(indexed["N0"]) != set(indexed["N1"]):
        raise ValueError("models evaluated different T/N pairs")
    changes = []
    for key in sorted(indexed["N0"]):
        a, b = indexed["N0"][key], indexed["N1"][key]
        if (a["source"], a["speaker"], a["strict_reference"]) != (b["source"], b["speaker"], b["strict_reference"]):
            raise ValueError("source metadata differs across models")
        changes.append(dict(id=key, source=a["source"], speaker=a["speaker"],
                            strict_reference=a["strict_reference"],
                            N0_tts_win=a["delta"]["zero"] > 0, N1_tts_win=b["delta"]["zero"] > 0,
                            win_difference=int(b["delta"]["zero"] > 0)-int(a["delta"]["zero"] > 0)))
    result = {}
    for name, selected in views(changes):
        sources = sorted({p["source"] for p in selected})
        if not sources:
            continue
        values = {s: float(np.mean([p["win_difference"] for p in selected if p["source"] == s])) for s in sources}
        clusters = {s: next(p["speaker"] for p in selected if p["source"] == s) for s in sources}
        result[name] = dict(
            zero_winrate_N1_minus_N0=bootstrap_source_equal(values, clusters, replicates),
            pairs=len(selected), changed_tts_win_status=sum(p["N0_tts_win"] != p["N1_tts_win"] for p in selected),
            N0_only_tts=sum(p["N0_tts_win"] and not p["N1_tts_win"] for p in selected),
            N1_only_tts=sum(p["N1_tts_win"] and not p["N0_tts_win"] for p in selected),
            no_cross_model_distance_subtraction=True)
    return dict(summary=result, paired_win_changes=changes,
                training_seed_inference="one paired seed is a feasibility pilot; media are not independent training repetitions")


def views(pairs):
    return [("all20", pairs), ("strict12", [p for p in pairs if p["strict_reference"]]),
            ("extended8", [p for p in pairs if not p["strict_reference"]])]


def paired_training_identity(checkpoints):
    if set(checkpoints) != {"N0", "N1"}:
        raise ValueError("expected paired N0/N1 checkpoints")
    fields = ("seed", "initial_model_state_sha256", "model_source_sha256", "pairs_sha256",
              "trainer_source_sha256", "pool_protocol_sha256", "batch_size", "lr", "margin", "steps",
              "grad_clip", "val_every", "val_queries", "selection")
    result = {}
    for key in fields:
        a, b = checkpoints["N0"].get(key), checkpoints["N1"].get(key)
        if a is None or a != b:
            raise ValueError(f"paired training condition mismatch: {key}")
        result[key] = a
    for condition, protocol in checkpoints.items():
        if protocol.get("condition") != condition:
            raise ValueError("checkpoint condition is mislabeled")
    return result


def freeze(args):
    import torch
    rows = read(args.inputs)
    validate_inputs(rows)
    if sum(r["arm"] == "N" and r["strict_reference"] for r in rows) != 12:
        raise ValueError("strict support does not contain 12 sources")
    models = read(args.models)
    if len(models) != 2 or {m["id"] for m in models} != {"N0", "N1"}:
        raise ValueError("models JSON must have exactly N0 and N1")
    frozen = []
    training_protocols = {}
    for m in models:
        checkpoint, gate_path = Path(m["checkpoint"]), Path(m["gate_path"])
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        training_protocols[m["id"]] = checkpoint_data.get("protocol", {})
        checkpoint_validation = checkpoint_data.get("validation", {})
        gate = read(gate_path)
        if not checkpoint_validation or checkpoint_validation != gate:
            raise ValueError("gate metrics are not the selected checkpoint's own validation")
        frozen.append(dict(m, checkpoint_sha256=sha(checkpoint),
                           gate_sha256=sha(gate_path), capability_pass=gate_passes(gate),
                           checkpoint_step=checkpoint_data.get("step")))
    paired_identity = paired_training_identity(training_protocols)
    all_pass = all(m["capability_pass"] for m in frozen)
    if not all_pass and not args.diagnostic:
        raise ValueError("one or both capability gates failed/missing; preference interpretation is disabled")
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "inputs.json", rows)
    write(args.out / "models.json", frozen)
    write(args.out / "protocol.json", dict(
        status="FROZEN_BEFORE_PREFERENCE_FORWARD", capability_pass=all_pass,
        interpretation="EXPLORATORY_PREFERENCE" if all_pass else "DIAGNOSTIC_CAPABILITY_FAILED",
        inputs_path=str(args.inputs.resolve()), inputs_sha256=sha(args.inputs),
        models_path=str(args.models.resolve()), models_config_sha256=sha(args.models),
        paired_training_identity=paired_identity,
        source_sha256=sha(__file__), created_unix=time.time(),
        primary="physical zero: negative raw 1024D Euclidean, eps1e-6; no per-clip timing correction",
        auxiliary="31 real-support feature-index lags +/-600ms; best-background Sync-C-like contrast, not official model benchmark",
        windows="exact frozen per-source equal N+3T common windows from report73",
        audio="canonical int16 mono PCM16k from sample0; official python_speech_features.mfcc defaults",
        video="existing official SyncNet face-track AVI -> official ffmpeg JPEG defaults -> cv2 BGR; encoder re-forwarded",
        previous_embeddings_reused=False, scratch="one media at a time, uniquely owned temp directory removed in finally",
        statistics="source-equal T draw means; AISHELL source-speaker clustered bootstrap; same paired cluster resamples across conditions",
        limitations="old cohort is exploratory; raw distances have condition-specific scales; one paired training seed is not causal replication"))
    shutil.copy2(__file__, args.out / "evaluator_source_frozen.py")
    write(args.out / "status.json", dict(status="FROZEN", media=len(rows), models=2))


def load_net(repo, checkpoint, device):
    import torch
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Controlled trainer saves an explicit model state. Loading arbitrary model
    # objects or silent partial state matching would obscure initialization.
    if not isinstance(data, dict) or "model" not in data:
        raise ValueError("controlled checkpoint must contain model state")
    raw_source = data.get("protocol", {}).get("model_source")
    if not isinstance(raw_source, str) or not Path(raw_source).is_absolute():
        raise ValueError("controlled checkpoint requires an absolute frozen model_source")
    source = Path(raw_source)
    if sha(source) != data.get("protocol", {}).get("model_source_sha256"):
        raise ValueError("official architecture source changed since training")
    spec = importlib.util.spec_from_file_location("controlled_preference_syncnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.S()
    net.load_state_dict(data["model"], strict=True)
    return net.to(device).eval(), data


def encode_item(net, row, device, batch_size, scratch_parent):
    import cv2
    import python_speech_features as psf
    from scipy.io import wavfile
    import torch
    if sha(row["audio_path"]) != row["audio_sha256"] or sha(row["sync_crop"]) != row["sync_crop_sha256"]:
        raise ValueError("canonical PCM or face-track identity changed")
    sr, pcm = wavfile.read(row["audio_path"])
    if sr != 16000 or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError("expected canonical PCM16k mono int16")
    cap = cv2.VideoCapture(row["sync_crop"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(fps-25) > 1e-6:
        raise ValueError("face track must be 25fps")
    n = int(row["sync_windows"])
    mfcc = psf.mfcc(pcm, sr).T.astype(np.float32)
    if mfcc.shape[1] < 4*(n-1)+20:
        raise ValueError("MFCC support is too short")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=row["id"]+"_", dir=scratch_parent))
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", row["sync_crop"],
                        "-threads", "1", "-f", "image2", str(scratch / "%06d.jpg")], check=True)
        files = sorted(scratch.glob("*.jpg"))
        if len(files) < n+4:
            raise ValueError("face-track frame support is too short")
        frames = np.stack([cv2.imread(str(p)) for p in files[:n+4]])
        if frames.shape[1:] != (224, 224, 3):
            raise ValueError("official face track must decode as 224x224 BGR")
        a, v = [], []
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                indices = range(start, min(start+batch_size, n))
                xa = np.stack([mfcc[:, 4*i:4*i+20] for i in indices])[:, None]
                xv = np.stack([frames[i:i+5].transpose(3, 0, 1, 2) for i in indices]).astype(np.float32)
                a.append(net.forward_aud(torch.from_numpy(xa).to(device)).cpu().numpy())
                v.append(net.forward_lip(torch.from_numpy(xv).to(device)).cpu().numpy())
        return np.concatenate(a), np.concatenate(v)
    finally:
        # Only the freshly generated directory returned by mkdtemp is removed.
        shutil.rmtree(scratch)


def run(args):
    import torch
    protocol = read(args.out / "protocol.json")
    if protocol["source_sha256"] != sha(__file__):
        raise ValueError("evaluator source changed after freeze")
    if not protocol["capability_pass"] and not args.diagnostic:
        raise ValueError("capability failed; supply --diagnostic only for explicitly labeled diagnostics")
    rows = read(args.out / "inputs.json")
    validate_inputs(rows)
    models = read(args.out / "models.json")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    for model in models:
        if sha(model["checkpoint"]) != model["checkpoint_sha256"] or sha(model["gate_path"]) != model["gate_sha256"]:
            raise ValueError("checkpoint or capability gate changed after freeze")
        net, checkpoint = load_net(args.repo, model["checkpoint"], args.device)
        dest = args.out / model["id"]
        dest.mkdir(exist_ok=True)
        write(dest / "model_receipt.json", dict(
            checkpoint_sha256=model["checkpoint_sha256"],
            official_architecture_sha256=checkpoint["protocol"]["model_source_sha256"],
            official_architecture_path=checkpoint["protocol"]["model_source"],
            seed=checkpoint["protocol"]["seed"], condition=checkpoint["protocol"]["condition"],
            checkpoint_step=checkpoint.get("step"), torch_version=torch.__version__,
            eval_mode=not net.training, input_embeddings_reused=False))
        items = []
        for row in rows:
            artifact = dest / f"{row['id']}.npz"
            meta = artifact.with_suffix(".json")
            if artifact.exists() and meta.exists():
                item = read(meta)
                if item["checkpoint_sha256"] != model["checkpoint_sha256"] or item["npz_sha256"] != sha(artifact):
                    raise ValueError("resumed item identity mismatch")
            else:
                audio, video = encode_item(net, row, args.device, args.batch_size, args.out / "scratch")
                queries = np.arange(row["query_start"], row["query_end_exclusive"])
                matrix = score_matrix(audio, video, queries)
                np.savez_compressed(artifact, audio=audio, video=video, queries=queries, lags=LAGS, matrix=matrix)
                speaker, provenance = source_speaker(row)
                item = dict(id=row["id"], source=row["sample_id"], arm=row["arm"],
                            strict_reference=row["strict_reference"], speaker=speaker,
                            speaker_provenance=provenance, query_count=len(queries),
                            query_start=int(queries[0]), query_end_exclusive=int(queries[-1]+1),
                            endpoints=reduce_curve(matrix), checkpoint_sha256=model["checkpoint_sha256"],
                            npz_sha256=sha(artifact))
                write(meta, item)
            items.append(item)
            write(args.out / "status.json", dict(status="SCORING", model=model["id"],
                  scored=len(items), total=len(rows), pid=os.getpid(), elapsed_s=time.time()-started))
            print("PREFERENCE_ITEM", model["id"], len(items), row["id"], flush=True)
        write(dest / "items.json", items)
        del net
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    analyze(args)


def analyze(args):
    protocol = read(args.out / "protocol.json")
    models = read(args.out / "models.json")
    pair_sets = {}
    summaries = {}
    for model in models:
        dest = args.out / model["id"]
        items = read(dest / "items.json")
        if len(items) != 80:
            raise ValueError("expected all 80 media before analysis")
        pairs = make_pairs(items)
        if len(pairs) != 60:
            raise ValueError("expected 60 T/N pairs")
        pair_sets[model["id"]] = pairs
        write(dest / "pairs.json", pairs)
        summaries[model["id"]] = {name: summarize_pairs(ps, args.bootstrap_replicates) for name, ps in views(pairs)}
    comparison = compare_models(pair_sets, args.bootstrap_replicates)
    write(args.out / "summary.json", dict(
        status="COMPLETE_EXPLORATORY" if protocol["capability_pass"] else "DIAGNOSTIC_CAPABILITY_FAILED",
        capability_pass=protocol["capability_pass"], primary_endpoint="zero",
        model_summaries=summaries, condition_comparison=comparison,
        no_cross_model_distance_subtraction=True,
        limitations=protocol["limitations"]))
    write(args.out / "status.json", dict(status="COMPLETE", capability_pass=protocol["capability_pass"],
          interpretation="EXPLORATORY_PREFERENCE" if protocol["capability_pass"] else "DIAGNOSTIC_ONLY"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "run", "analyze"])
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--inputs", type=Path)
    p.add_argument("--models", type=Path, help="JSON list: id N0/N1, checkpoint, gate_path")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    if args.stage == "freeze":
        if args.inputs is None or args.models is None:
            p.error("freeze requires --inputs and --models")
        freeze(args)
    elif args.stage == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
