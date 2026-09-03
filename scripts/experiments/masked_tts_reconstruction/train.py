"""Fixed CPU training arms and shared sampler schedules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import (
    BATCH_SIZE,
    CPU_THREADS,
    EVAL_GROUPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    SEEDS,
    TRAIN_GROUPS,
    TRAIN_STEPS,
    WEIGHT_DECAY,
)
from .model import MaskedNaturalReconstructor, reconstruction_loss
from .protocol import canonical_json, sha256_bytes, sha256_text, write_json


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


CPU_CONFIGURED = False


def set_deterministic_cpu() -> dict[str, Any]:
    global CPU_CONFIGURED
    torch.set_num_threads(CPU_THREADS)
    if not CPU_CONFIGURED:
        torch.set_num_interop_threads(1)
        CPU_CONFIGURED = True
    torch.use_deterministic_algorithms(True)
    torch.set_default_device("cpu")
    return {"device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True}


def make_schedule(
    mask_manifest: Mapping[str, Any],
    seed: int,
    *,
    train_groups: Sequence[str] = TRAIN_GROUPS,
    train_steps: int = TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    groups = tuple(str(group) for group in train_groups)
    if not groups or train_steps < 1 or batch_size < 1:
        raise ValueError("schedule dimensions must be positive")
    masks = [row for row in mask_manifest["masks"] if row["prototype_split"] == "train"]
    by_group: dict[str, dict[str, list[Mapping[str, Any]]]] = {group: {} for group in groups}
    for row in masks:
        group = str(row["source_group"])
        if group in by_group:
            by_group[group].setdefault(str(row["sample_id"]), []).append(row)
    if any(not by_group[group] for group in groups):
        raise ValueError("every frozen train group must have masks")
    for group in groups:
        for sid in by_group[group]:
            by_group[group][sid].sort(key=lambda row: int(row["canonical_index"]))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    schedule: list[dict[str, Any]] = []
    for step in range(int(train_steps)):
        for position in range(int(batch_size)):
            group_index = int(rng.integers(0, len(groups)))
            group = groups[group_index]
            sample_ids = sorted(by_group[group], key=lambda value: value.encode("utf-8"))
            record_index = int(rng.integers(0, len(sample_ids)))
            sample_id = sample_ids[record_index]
            choices = by_group[group][sample_id]
            mask_index = int(rng.integers(0, len(choices)))
            row = choices[mask_index]
            schedule.append({
                "step": step + 1,
                "position": position,
                "group_index": group_index,
                "source_group": group,
                "record_index": record_index,
                "sample_id": sample_id,
                "mask_index": mask_index,
                "mask_sha256": str(row["mask_sha256"]),
            })
    return schedule


def schedule_hash(schedule: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(list(schedule)))


def _batch(examples: Mapping[str, Mapping[str, np.ndarray]], rows: Sequence[Mapping[str, Any]], *, zero_tts: bool) -> dict[str, torch.Tensor]:
    values = [examples[str(row["mask_sha256"])] for row in rows]
    batch: dict[str, torch.Tensor] = {}
    for key in ("natural_mel", "masked_support", "target_core", "target"):
        batch[key] = torch.from_numpy(np.stack([np.asarray(item[key], dtype=np.float32) for item in values], axis=0))
    tts = np.stack([np.asarray(item["tts_features"], dtype=np.float32) for item in values], axis=0)
    if zero_tts:
        tts[...] = 0.0
    batch["tts_features"] = torch.from_numpy(tts)
    return batch


def train_arm(
    examples: Mapping[str, Mapping[str, np.ndarray]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    arm: str,
    initial_state: Mapping[str, torch.Tensor],
    output_path: Path | None = None,
    binding: Mapping[str, Any] | None = None,
    steps: int = TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    if arm not in {"full_correct", "nat_only", "paired_tts", "phone_centroid"}:
        raise ValueError("unsupported training arm")
    if steps < 1 or batch_size < 1 or len(schedule) != int(steps) * int(batch_size):
        raise ValueError("schedule length does not match training dimensions")
    set_deterministic_cpu()
    model = MaskedNaturalReconstructor()
    model.load_state_dict(clone_state(initial_state), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[dict[str, float]] = []
    for step in range(int(steps)):
        rows = schedule[step * int(batch_size):(step + 1) * int(batch_size)]
        batch = _batch(examples, rows, zero_tts=arm == "nat_only")
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["natural_mel"], batch["masked_support"], batch["target_core"], batch["tts_features"])
        loss = reconstruction_loss(prediction, batch["target"], batch["target_core"])
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        losses.append({key: float(value.detach().cpu()) for key, value in loss.items()})
    if not losses or not all(np.isfinite(value) for row in losses for value in row.values()):
        raise FloatingPointError(f"non-finite training loss for {arm} seed {seed}")
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "arm": arm,
        "optimizer_step": int(steps),
        "schedule_sha256": schedule_hash(schedule),
        "initial_state_sha256": state_hash(initial_state),
        "model_config": model.config(),
        "thread_config": {"device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True},
        "binding": dict(binding or {}),
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "loss_trace_sha256": sha256_text(canonical_json(losses)),
        "state_dict": model.state_dict(),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def load_checkpoint(path: Path) -> tuple[MaskedNaturalReconstructor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MaskedNaturalReconstructor()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def run_overfit_smoke(examples: Mapping[str, Mapping[str, np.ndarray]], schedule: Sequence[Mapping[str, Any]], *, seed: int = 20260901, steps: int = 40) -> dict[str, Any]:
    if steps < 2:
        raise ValueError("overfit smoke needs at least two steps")
    set_deterministic_cpu()
    torch.manual_seed(seed)
    model = MaskedNaturalReconstructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    selected = list(examples)[:2]
    if len(selected) < 2:
        raise ValueError("overfit smoke requires two examples")
    rows = [{"mask_sha256": key} for key in selected]
    trace: list[float] = []
    for _ in range(steps):
        batch = _batch(examples, rows * (BATCH_SIZE // len(rows)), zero_tts=False)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["natural_mel"], batch["masked_support"], batch["target_core"], batch["tts_features"])
        loss = reconstruction_loss(prediction, batch["target"], batch["target_core"])["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        trace.append(float(loss.detach()))
    return {"status": "GO", "steps": steps, "initial_loss": trace[0], "final_loss": trace[-1], "finite": bool(np.isfinite(trace).all()), "decreasing": bool(trace[-1] < trace[0]), "parameter_count": model.parameter_count}


def train_all(
    examples: Mapping[str, Mapping[str, np.ndarray]],
    mask_manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    seeds: Sequence[int] = SEEDS,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_cpu()
    results: dict[str, Any] = {}
    for seed in seeds:
        torch.manual_seed(int(seed))
        initial = MaskedNaturalReconstructor()
        initial_state = clone_state(initial.state_dict())
        initial_hash = state_hash(initial_state)
        schedule = make_schedule(mask_manifest, int(seed))
        write_json(output_dir / str(seed) / "schedule.json", {"schema_version": 1, "seed": int(seed), "entries": schedule, "sha256": schedule_hash(schedule)})
        seed_dir = output_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        arms = {}
        for arm in ("full_correct", "nat_only"):
            checkpoint = seed_dir / arm / "checkpoint.pt"
            arms[arm] = train_arm(examples, schedule, seed=int(seed), arm=arm, initial_state=initial_state, output_path=checkpoint, binding=binding)
        results[str(seed)] = {"seed": int(seed), "initial_state_sha256": initial_hash, "schedule_sha256": schedule_hash(schedule), "arms": arms, "thread_config": {"device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True}}
    return {"schema_version": 1, "seeds": results, "arm_count": len(results) * 2}
