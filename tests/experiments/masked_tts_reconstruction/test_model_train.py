from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.experiments.masked_tts_reconstruction.model import MaskedNaturalReconstructor, forward_batch, reconstruction_loss
from scripts.experiments.masked_tts_reconstruction.train import make_schedule, state_hash
from scripts.experiments.masked_tts_reconstruction.config import TRAIN_GROUPS


def test_model_contract_and_parameter_cap() -> None:
    torch.manual_seed(0)
    model = MaskedNaturalReconstructor()
    assert model.parameter_count < 1_500_000
    batch = {
        "natural_mel": torch.zeros(2, 96, 80),
        "masked_support": torch.zeros(2, 96, 1),
        "target_core": torch.zeros(2, 96, 1),
        "tts_features": torch.zeros(2, 96, 1024),
        "target": torch.zeros(2, 96, 80),
    }
    batch["target_core"][:, 20:24] = 1
    output = forward_batch(model, batch)
    assert output.shape == (2, 96, 80)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError):
        forward_batch(model, {**batch, "phone_id": torch.zeros(2, 1)})


def test_velocity_excludes_core_boundaries() -> None:
    prediction = torch.zeros(1, 96, 80)
    target = torch.zeros(1, 96, 80)
    prediction[:, 19] = 100
    prediction[:, 24] = 100
    core = torch.zeros(1, 96, 1)
    core[:, 20:24] = 1
    losses = reconstruction_loss(prediction, target, core)
    assert float(losses["velocity"]) == 0.0
    assert float(losses["patch"]) == 0.0


def test_schedule_is_reproducible_and_group_balanced_shape() -> None:
    masks = []
    for group_index, group in enumerate(TRAIN_GROUPS):
        masks.append({"source_group": group, "sample_id": f"sample{group_index}", "mask_sha256": f"mask{group_index}", "canonical_index": group_index, "prototype_split": "train"})
    manifest = {"masks": masks}
    first = make_schedule(manifest, 20260901)
    second = make_schedule(manifest, 20260901)
    assert first == second
    assert len(first) == 600 * 16
    assert state_hash({"x": torch.ones(2)}) == state_hash({"x": torch.ones(2)})
