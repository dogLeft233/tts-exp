import importlib.util
import json
from pathlib import Path
import struct

import pytest

SOURCE = Path(__file__).parents[1]/"scripts/experiments/chinese_lips_data_20261029.py"
spec = importlib.util.spec_from_file_location("lips_data", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def header(name="train/1/FACE/a_FACE.mp4", flags=0):
    encoded = name.encode()
    return struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, flags, 8, 0, 0, 123, 25, 30, len(encoded), 0)+encoded


def test_header_retains_offsets_and_crc():
    e = mod.parse_header(header())
    assert e["compressed_size"] == 25
    assert e["crc"] == 123
    assert e["header_size"] == len(header())


def test_skip_unknown_descriptor_size_is_rejected():
    with pytest.raises(ValueError, match="data descriptors"):
        mod.parse_header(header(flags=8))
    assert mod.parse_header(header(flags=8), allow_descriptor=True)["name"].endswith("_FACE.mp4")


def test_disk_boundary_search_does_not_take_random_pk():
    data = b"audioPK\x03\x04garbage"+header()
    assert mod.find_first_header(data) == len(b"audioPK\x03\x04garbage")


def test_balanced_selection_deduplicates_and_is_deterministic(tmp_path):
    entries = [{"name": f"train/{s}/FACE/{s}_X_{i}_FACE.mp4"} for s in [1, 2, 3] for i in range(5)]
    (tmp_path/"train.z01.jsonl").write_text("\n".join(map(json.dumps, entries+entries[:1])), encoding="utf8")
    a = mod.choose_entries(tmp_path, 6, 3)
    b = mod.choose_entries(tmp_path, 6, 3)
    assert a == b
    assert len({e["name"] for e in a}) == 6
    assert sorted([sum(Path(e["name"]).name.startswith(f"{s}_") for e in a) for s in [1, 2, 3]]) == [2, 2, 2]


def test_actual_duration_finalizer_balances_speakers():
    records = [{"id": f"{s}_{i}", "speaker": str(s), "duration": 11.0} for s in [1, 2, 3] for i in range(5)]
    chosen = mod.balanced_records(records, 60, 7)
    assert sum(r["duration"] for r in chosen) == 66
    assert {r["speaker"] for r in chosen} == {"1", "2", "3"}
    with pytest.raises(ValueError, match="Insufficient"):
        mod.balanced_records(records, 1000, 7)


def test_prefix_quota_is_disk_ordered_not_completion_order(tmp_path):
    entries = [{"name": f"train/1/FACE/1_X_{i}_FACE.mp4", "offset": i*100} for i in range(3)]
    (tmp_path/"train.z01.jsonl").write_text("\n".join(map(json.dumps, entries))+"\n{partial", encoding="utf8")
    chosen, coverage = mod.prefix_candidates(tmp_path, {f"1_X_{i}": 7 for i in range(3)}, 10)
    assert len(chosen) == 2
    assert coverage["1"]["quota_met"]
    assert coverage["1"]["approximate_seconds"] == 14
    assert not coverage["2"]["quota_met"]


def test_exhausted_disk_deficit_is_redistributed_without_scores():
    quotas = mod.allocate_prefix_quotas({"4": 0}, 8*100)
    assert quotas["4"] == 0
    assert sum(quotas.values()) == pytest.approx(800)
    assert quotas["1"] == pytest.approx(800/7)
    with pytest.raises(ValueError, match="insufficient"):
        mod.allocate_prefix_quotas({str(i): 20 for i in range(1, 9)}, 800)
