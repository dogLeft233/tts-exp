from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phone_alignment_target_audit import (  # noqa: E402
    _hard_dtw_path,
    _pool_target,
)


def test_hard_dtw_path_is_local_monotonic_and_covers_source() -> None:
    cost = torch.tensor(
        [
            [0.0, 3.0, 4.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
        ]
    )

    path = _hard_dtw_path(cost, band_ratio=1.0)

    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert all(path[index][0] <= path[index + 1][0] for index in range(len(path) - 1))
    assert all(path[index][1] <= path[index + 1][1] for index in range(len(path) - 1))
    assert {i for i, _ in path} == {0, 1, 2}


def test_phone_pool_target_repeats_one_vector_per_source_span() -> None:
    natural = torch.zeros(4, 2)
    tts = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
    spans = [
        ({"label": "a"}, torch.tensor([0, 1]), torch.tensor([0, 1])),
        ({"label": "b"}, torch.tensor([2, 3]), torch.tensor([2])),
    ]

    target, mask, stats = _pool_target(natural, tts, spans)

    assert mask.tolist() == [True, True, True, True]
    assert torch.equal(target[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(target[1], torch.tensor([2.0, 3.0]))
    assert torch.equal(target[2], torch.tensor([9.0, 9.0]))
    assert stats["used_spans"] == 2
