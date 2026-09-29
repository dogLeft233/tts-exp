from __future__ import annotations

from scripts.experiments.phoneme_tfg_association.protocol import PAIR_TYPES, freeze_protocol, select_blocks, validate_receipt


def _rows(count: int = 36):
    audit = []
    features = []
    refs = []
    for index in range(count):
        sid = f"s{index:03d}"
        group = f"g{index:03d}"
        audit.append({"sample_id": sid, "source_group": group})
        features.append({"sample_id": sid, "source_group": group, "eligible": True, "primary_values": {}})
        refs.append({"sample_id": sid, "source_group": group, "status": "ready", "source_video": f"{sid}.mp4", "box_top_bottom_left_right": [0, 224, 0, 224], "crop_sha256": "x"})
    return audit, features, refs


def test_selection_is_order_invariant_and_balanced():
    audit, features, refs = _rows()
    cfg = {"seed": 20260921, "source_groups": 36, "pair_replicates": 6, "tts_order": [*("qwen_cloud", "qwen_local", "index_tts2", "cosyvoice2")], "max_unique_wav2lip_videos": 108, "primary_tfg": "wav2lip"}
    left = select_blocks(audit, features, refs, cfg)
    right = select_blocks(list(reversed(audit)), list(reversed(features)), list(reversed(refs)), cfg)
    assert [(row["sample_id"], row["pair"]) for row in left] == [(row["sample_id"], row["pair"]) for row in right]
    assert {tuple(row["pair"]) for row in left} == set(PAIR_TYPES)
    assert all(sum(tuple(row["pair"]) == pair for row in left) == 6 for pair in PAIR_TYPES)
    assert len(left) == 36


def test_freeze_budget_and_receipt_contract(tmp_path):
    audit, features, refs = _rows()
    cfg = {"seed": 1, "source_groups": 36, "pair_replicates": 6, "max_unique_wav2lip_videos": 108, "primary_tfg": "wav2lip"}
    blocks = select_blocks(audit, features, refs, cfg)
    protocol = freeze_protocol(tmp_path, blocks, cfg)
    assert protocol["video_budget"]["planned_unique_wav2lip_videos"] == 108
    cell = protocol["cells"][0]
    validate_receipt({"protocol_id": protocol["protocol_id"], "protocol_hash": cell["protocol_hash"], "cell_key": cell["cell_key"], "sample_id": cell["sample_id"], "source_group": cell["source_group"], "tfg": cell["tfg"], "arm": cell["arm"], "status": "failed"}, cell)


def test_optional_ditto_uses_two_blocks_per_pair(tmp_path):
    audit, features, refs = _rows()
    cfg = {
        "seed": 1,
        "source_groups": 36,
        "pair_replicates": 6,
        "max_unique_wav2lip_videos": 108,
        "max_unique_ditto_videos": 36,
        "primary_tfg": "wav2lip",
        "enable_ditto_replication": True,
    }
    blocks = select_blocks(audit, features, refs, cfg)
    protocol = freeze_protocol(tmp_path, blocks, cfg)
    pair_by_block = {row["block_id"]: "|".join(str(item) for item in row["pair"]) for row in blocks}
    ditto_pairs = [pair_by_block[row["block_id"]] for row in protocol["ditto_cells"] if row["arm"] == "natural"]
    assert len(ditto_pairs) == 12
    assert all(ditto_pairs.count("|".join(pair)) == 2 for pair in PAIR_TYPES)
