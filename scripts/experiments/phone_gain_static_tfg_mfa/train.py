"""Fair FIT-only training for the three conditional gain-head arms."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.phone_separability_enhancement.audio import BandGainRenderer, make_protected_mask, validate_distortion
from scripts.experiments.phone_separability_enhancement.teacher import FrozenPhoneTeacher, pool_hidden

from .assets import read_pcm16
from .conditioning import Conditioning, PhoneVocabulary, build_conditioning
from .config import canonical_hash, state_dict_hash, write_json
from .model import build_model


def _torch():
    import torch

    return torch


def _sample_condition(sample: Mapping[str, Any], *, frame_count: int, vocabulary: PhoneVocabulary, duration_stats: Mapping[str, float], mode: str, sample_count: int, sample_rate: int, hop_length: int) -> Conditioning:
    value = sample.get("conditioning")
    if isinstance(value, Conditioning) and value.phone_ids.size == int(frame_count):
        if value.mode != mode:
            raise ValueError("sample conditioning mode does not match arm")
        return value
    return build_conditioning(sample["tokens"], sample_count=sample_count, frame_count=frame_count, vocabulary=vocabulary, duration_stats=duration_stats, protected_mask=sample["protected_mask"], edit_mask=sample.get("edit_mask"), mode=mode, sample_rate=sample_rate, hop_length=hop_length)


def _phone_loss(hidden: Any, frame_times: Any, tokens: Sequence[Mapping[str, Any]], support_entries: Sequence[Mapping[str, Any]], centroids: Mapping[str, Sequence[float]], *, target_margin: float = 0.05, temperature: float = 0.1) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    pooled = pool_hidden(hidden, frame_times, tokens, view="core")
    labels = sorted(str(label) for label in centroids)
    if len(labels) < 2:
        return hidden.new_zeros(()), []
    center = torch.as_tensor(np.stack([np.asarray(centroids[label], dtype=np.float32) for label in labels]), device=hidden.device, dtype=hidden.dtype)
    center = center / torch.linalg.vector_norm(center, dim=1, keepdim=True).clamp_min(1e-8)
    values: dict[str, list[Any]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for entry in support_entries:
        token_id = str(entry.get("natural_token_id", f":{entry.get('natural_token_index')}"))
        pooled_row = pooled.get(token_id)
        label = str(entry.get("label", ""))
        if pooled_row is None or label not in centroids:
            continue
        embedding = pooled_row["embedding"]
        scores = torch.matmul(center, embedding)
        target = labels.index(label)
        wrong = torch.cat([scores[:target], scores[target + 1:]])
        margin = scores[target] - wrong.max()
        values[label].append(margin)
        rows.append({"support_key": entry.get("support_key"), "label": label, "margin": margin, "token_id": token_id})
    if not values:
        return hidden.new_zeros(()), rows
    label_losses = []
    for label in sorted(values):
        margins = torch.stack(values[label])
        label_losses.append(torch.nn.functional.softplus((float(target_margin) - margins) / float(temperature)).mean())
    return torch.stack(label_losses).mean(), rows


def _normalize_features(features: Any, feature_stats: Mapping[str, Any] | None) -> Any:
    if not feature_stats:
        return features
    torch = _torch()
    mean = torch.as_tensor(feature_stats["mean"], dtype=features.dtype, device=features.device).reshape(1, -1, 1)
    std = torch.as_tensor(feature_stats["std"], dtype=features.dtype, device=features.device).reshape(1, -1, 1).clamp_min(1e-6)
    if features.shape[1] != mean.shape[1]:
        raise ValueError("FIT feature statistics channel count differs from renderer")
    return (features - mean) / std


def loss_for_sample(model: Any, sample: Mapping[str, Any], teacher: FrozenPhoneTeacher, centroids: Mapping[str, Sequence[float]], renderer: BandGainRenderer, *, vocabulary: PhoneVocabulary, duration_stats: Mapping[str, float], feature_stats: Mapping[str, Any] | None = None, mode: str, device: str, phone_weight: float = 1.0, keep_weight: float = 1.0, tv_weight: float = 0.01, sample_rate: int = 16000, hop_length: int = 128) -> tuple[Any, dict[str, Any], Any]:
    torch = _torch()
    source_pcm = np.asarray(sample["audio_pcm"], dtype=np.int16).reshape(-1)
    x = torch.as_tensor(source_pcm.astype(np.float32) / 32768.0, device=device)
    features, _ = renderer.band_features(x)
    features = _normalize_features(features, feature_stats)
    condition = _sample_condition(sample, frame_count=int(features.shape[-1]), vocabulary=vocabulary, duration_stats=duration_stats, mode=mode, sample_count=source_pcm.size, sample_rate=sample_rate, hop_length=hop_length)
    ids = torch.as_tensor(condition.phone_ids, dtype=torch.long, device=device).unsqueeze(0)
    timing = torch.as_tensor(condition.timing, dtype=features.dtype, device=device).unsqueeze(0)
    raw_gain = model(features, ids, timing, mode=mode)
    edit_mask = torch.as_tensor(condition.edit_mask if condition.edit_mask is not None else ~condition.protected_mask, dtype=features.dtype, device=device)
    y, render_meta = renderer.render(x, raw_gain, edit_mask)
    hidden, frame_times, _ = teacher.encode(y, layer=int(sample.get("layer", 6)))
    phone, phone_rows = _phone_loss(hidden, frame_times, sample["tokens"], sample.get("support_entries", []), centroids)
    edit = edit_mask
    keep = ((y - x) * edit).square().sum() / x.square().sum().clamp_min(1e-8)
    normalized = raw_gain / float(renderer.max_gain_db)
    tv_time = normalized[:, :, 1:].sub(normalized[:, :, :-1]).square().mean() if normalized.shape[-1] > 1 else normalized.new_zeros(())
    tv_freq = normalized[:, 1:, :].sub(normalized[:, :-1, :]).square().mean() if normalized.shape[1] > 1 else normalized.new_zeros(())
    tv = tv_time + tv_freq
    total = float(phone_weight) * phone + float(keep_weight) * keep + float(tv_weight) * tv
    return total, {"phone_loss": float(phone.detach().cpu()), "keep_loss": float(keep.detach().cpu()), "tv_loss": float(tv.detach().cpu()), "phone_tokens": len(phone_rows), "render": render_meta, "condition_mode": mode, "candidate": y, "source": x, "condition": condition}, y


def _balanced_order(samples: Sequence[Mapping[str, Any]], seed: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[str(sample.get("source_group", ""))].append(index)
    rng = random.Random(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    order: list[int] = []
    while any(groups.values()):
        for key in sorted(list(groups)):
            if groups[key]:
                order.append(groups[key].pop())
    return order


def _save_checkpoint(path: Path, model: Any, optimizer: Any, *, seed: int, optimizer_updates: int, microbatches: int, mode: str, contract_hash: str, epoch: int = 0, cursor: int = 0, elapsed_seconds: float = 0.0) -> str:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 3, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": int(seed), "optimizer_updates": int(optimizer_updates), "microbatches": int(microbatches), "mode": mode, "contract_hash": str(contract_hash), "epoch": int(epoch), "cursor": int(cursor), "elapsed_seconds": float(elapsed_seconds), "model_hash": state_dict_hash(model.state_dict()), "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate()}
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return str(path)


def train_arm(samples: Sequence[Mapping[str, Any]], teacher: FrozenPhoneTeacher, centroids: Mapping[str, Sequence[float]], *, mode: str, vocabulary: PhoneVocabulary, duration_stats: Mapping[str, float], seed: int, output_dir: str | Path, network: Mapping[str, Any], training: Mapping[str, Any], audio: Mapping[str, Any], device: str, resume: bool = False, contract: Mapping[str, Any] | None = None, feature_stats: Mapping[str, Any] | None = None) -> dict[str, Any]:
    torch = _torch()
    if mode not in {"AUDIO_FEATURES", "BOUNDARY_TIME", "MFA_PHONE_TIME"}:
        raise ValueError(f"unknown arm mode: {mode}")
    if not samples:
        return {"status": "NO_TRAINING_SAMPLES", "mode": mode, "seed": int(seed), "optimizer_updates": 0}
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    random.seed(int(seed))
    model = build_model(vocab_size=len(vocabulary.labels), embedding_dim=int(network.get("embedding_dim", 16)), audio_channels=int(audio.get("n_bands", 24)), hidden_channels=int(network.get("hidden_channels", 64)), max_gain_db=float(audio.get("max_gain_db", 6.0)), dilations=tuple(int(value) for value in network.get("residual_dilations", [1, 2, 4]))).to(device)
    with torch.no_grad():
        model.embedding.weight[vocabulary.sil_id].zero_()
        model.embedding.weight[vocabulary.unk_id].zero_()
    renderer = BandGainRenderer(sample_rate=int(audio.get("sample_rate", 16000)), n_fft=int(audio.get("n_fft", 512)), win_length=int(audio.get("win_length", 512)), hop_length=int(audio.get("hop_length", 128)), n_bands=int(audio.get("n_bands", 24)), max_gain_db=float(audio.get("max_gain_db", 6.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 1e-4)), weight_decay=float(training.get("weight_decay", 1e-4)))
    accumulation = int(training.get("gradient_accumulation", 4))
    max_updates = int(training.get("max_optimizer_updates", 400))
    checkpoint_every = int(training.get("checkpoint_every_updates", 25))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract_payload = {"mode": mode, "seed": int(seed), "vocabulary_hash": vocabulary.hash, "duration_stats": dict(duration_stats), "feature_stats": dict(feature_stats or {}), "network": dict(network), "training": dict(training), "audio": dict(audio), **dict(contract or {})}
    contract_hash = canonical_hash(contract_payload)
    epoch = 0
    cursor = 0
    order = _balanced_order(samples, seed + epoch)
    history: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    updates = 0
    microbatches = 0
    step_paths: list[Path] = []
    payload: dict[str, Any] = {}
    if resume:
        result_path = output / "result.json"
        if result_path.is_file():
            import json

            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("status") == "COMPLETE" and previous.get("contract_hash") == contract_hash and int(previous.get("optimizer_updates", 0)) >= max(int(training.get("max_optimizer_updates", 400)), 0):
                return previous
            history = list(previous.get("history", []))
        step_paths = sorted(output.glob("step_*.pt"))
        if step_paths:
            latest = step_paths[-1]
            payload = torch.load(str(latest), map_location=device, weights_only=False)
            if int(payload.get("schema_version", 0)) < 3 or payload.get("contract_hash") != contract_hash or payload.get("mode") != mode or int(payload.get("seed", -1)) != int(seed):
                raise ValueError(f"checkpoint contract mismatch: {latest}")
            if payload.get("model_hash") != state_dict_hash(payload["model"]):
                raise ValueError(f"checkpoint model hash mismatch: {latest}")
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            updates = int(payload.get("optimizer_updates", 0))
            microbatches = int(payload.get("microbatches", updates * int(training.get("gradient_accumulation", 4))))
            epoch = int(payload.get("epoch", 0))
            cursor = int(payload.get("cursor", 0))
            order = _balanced_order(samples, seed + epoch)
            if cursor < 0 or cursor > len(order):
                raise ValueError("checkpoint cursor is outside the locked epoch order")
            checkpoint_paths = [str(path) for path in step_paths]
            if "rng_torch" in payload:
                torch.set_rng_state(payload["rng_torch"].cpu())
            if "rng_numpy" in payload:
                np.random.set_state(payload["rng_numpy"])
            if "rng_python" in payload:
                random.setstate(payload["rng_python"])
            if "rng_cuda" in payload and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(payload["rng_cuda"])
    if not checkpoint_paths:
        checkpoint_paths = [_save_checkpoint(output / "step_0000.pt", model, optimizer, seed=seed, optimizer_updates=0, microbatches=0, mode=mode, contract_hash=contract_hash, epoch=epoch, cursor=cursor)]
    prior_elapsed = 0.0
    if step_paths:
        prior_elapsed = float(payload.get("elapsed_seconds", 0.0))
    started = time.monotonic() - max(0.0, prior_elapsed)
    optimizer.zero_grad(set_to_none=True)
    status = "COMPLETE"
    valid_since_cycle = False
    skipped_samples = 0
    while updates < max_updates:
        if not order:
            status = "NO_VALID_PHONE_TARGET"
            break
        if cursor >= len(order):
            cursor = 0
            epoch += 1
            order = _balanced_order(samples, seed + epoch)
            valid_since_cycle = False
        sample = samples[order[cursor]]
        cursor += 1
        cycle_end = cursor >= len(order)
        total, diagnostics, _ = loss_for_sample(model, sample, teacher, centroids, renderer, vocabulary=vocabulary, duration_stats=duration_stats, feature_stats=feature_stats, mode=mode, device=device, phone_weight=float(training.get("phone_weight", 1.0)), keep_weight=float(training.get("keep_weight", 1.0)), tv_weight=float(training.get("tv_weight", 0.01)), sample_rate=renderer.sample_rate, hop_length=renderer.hop_length)
        if int(diagnostics.get("phone_tokens", 0)) <= 0:
            optimizer.zero_grad(set_to_none=True)
            skipped_samples += 1
            if cycle_end:
                if not valid_since_cycle:
                    status = "NO_VALID_PHONE_TARGET"
                    break
                cursor = 0
                epoch += 1
                order = _balanced_order(samples, seed + epoch)
                valid_since_cycle = False
            continue
        valid_since_cycle = True
        (total / max(accumulation, 1)).backward()
        microbatches += 1
        if microbatches % accumulation == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
            if not bool(torch.isfinite(grad_norm)):
                status = "ENGINEERING_FAILURE"
                break
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.embedding.weight[vocabulary.sil_id].zero_()
                model.embedding.weight[vocabulary.unk_id].zero_()
            updates += 1
            if updates == 1 or updates % checkpoint_every == 0 or updates == max_updates:
                history.append({"optimizer_update": updates, "microbatches": microbatches, "loss": float(total.detach().cpu()), "phone_loss": diagnostics["phone_loss"], "keep_loss": diagnostics["keep_loss"], "tv_loss": diagnostics["tv_loss"], "phone_tokens": diagnostics["phone_tokens"]})
            if updates % checkpoint_every == 0 or updates == max_updates:
                checkpoint_paths.append(_save_checkpoint(output / f"step_{updates:04d}.pt", model, optimizer, seed=seed, optimizer_updates=updates, microbatches=microbatches, mode=mode, contract_hash=contract_hash, epoch=epoch, cursor=cursor, elapsed_seconds=time.monotonic() - started))
        if time.monotonic() - started >= float(training.get("timeout_s", 7200)):
            status = "BUDGET_LIMITED"
            break
        if cycle_end:
            cursor = 0
            epoch += 1
            order = _balanced_order(samples, seed + epoch)
            valid_since_cycle = False
    if microbatches % accumulation:
        optimizer.zero_grad(set_to_none=True)
    elapsed = max(0.0, time.monotonic() - started)
    last = _save_checkpoint(output / "last.pt", model, optimizer, seed=seed, optimizer_updates=updates, microbatches=microbatches, mode=mode, contract_hash=contract_hash, epoch=epoch, cursor=cursor, elapsed_seconds=elapsed)
    result = {"schema_version": 3, "status": status, "mode": mode, "seed": int(seed), "contract_hash": contract_hash, "optimizer_updates": int(updates), "microbatches": int(microbatches), "epoch": int(epoch), "cursor": int(cursor), "elapsed_seconds": elapsed, "skipped_samples": int(skipped_samples), "checkpoint": last, "checkpoint_paths": checkpoint_paths, "history": history, "model_hash": state_dict_hash(model.state_dict())}
    write_json(output / "result.json", result)
    return result


def load_checkpoint(model: Any, checkpoint: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return payload


def make_training_sample(row: Mapping[str, Any], support_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pcm, metadata = read_pcm16(row["natural"]["audio_path"] if "natural" in row else row["audio_path"])
    tokens = row["natural"]["tokens"] if "natural" in row else row["tokens"]
    protection = make_protected_mask(tokens, pcm.size)
    return {"pair_id": row["pair_id"], "source_group": row["source_group"], "analysis_split": row["analysis_split"], "audio_pcm": pcm, "audio_meta": metadata, "tokens": tokens, "protected_mask": protection["protected"], "edit_mask": protection["mask"], "support_entries": list(support_entries)}


__all__ = ["load_checkpoint", "loss_for_sample", "make_training_sample", "train_arm"]
