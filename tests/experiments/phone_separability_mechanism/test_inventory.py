from __future__ import annotations

from scripts.experiments.phone_separability_mechanism.inventory import build_overlap_graph, freeze_splits


def test_freeze_splits_is_independent_of_input_order() -> None:
    groups = ["g3", "g1", "g2", "g4", "g5"]
    first = freeze_splits(groups, fit_groups=3, dev_groups=2)
    second = freeze_splits(list(reversed(groups)), fit_groups=3, dev_groups=2)
    assert first == second
    assert set(first["fit"]).isdisjoint(first["dev"])


def test_overlap_graph_connects_arms_and_duplicate_pcm() -> None:
    assets = [
        {"asset_id": "n", "source_group": "g1", "pcm_sha256": "p1"},
        {"asset_id": "t", "source_group": "g1", "pcm_sha256": "p2"},
        {"asset_id": "alias", "source_group": "g2", "pcm_sha256": "p1"},
    ]
    graph = build_overlap_graph(assets)
    components = list(graph["components"].values())
    assert any(set(row) == {"n", "t", "alias"} for row in components)
