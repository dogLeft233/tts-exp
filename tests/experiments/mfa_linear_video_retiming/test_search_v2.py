from __future__ import annotations

import numpy as np

from scripts.experiments.mfa_linear_video_retiming.retime import build_map
from scripts.experiments.mfa_linear_video_retiming.search_v2 import choose_routes, final_reasons, propose_maps, select_final


def _metrics(c: float, d: float, d0: float, offset: int) -> dict:
    return {"sync_c": c, "sync_d": d, "d0": d0, "offset": offset,
            "local_windows": [{"d0": d0, "offset": offset}], "local_median_d0": d0,
            "local_abs_offset_q90": abs(offset)}


CFG = {"noninferiority_sync_c": .05, "noninferiority_sync_d": .05,
       "minimum_d0_improvement": .02, "final_d0_slack": .02}


def test_true_aligned_candidate_can_pass_with_small_c_loss() -> None:
    baseline = _metrics(5.0, 14.0, 16.0, -3)
    identity = {"score_kind": "true_fixed_crop", "label": "identity", "metrics": baseline,
                "map_sha256": "identity", "regularization": 0.0}
    aligned = {"score_kind": "true_fixed_crop", "label": "local", "metrics": _metrics(4.97, 14.01, 15.8, 0),
               "map_sha256": "aligned", "regularization": 1.0}
    proxy = {**aligned, "score_kind": "proxy_embedding", "map_sha256": "proxy"}
    assert final_reasons(aligned, baseline, CFG) == []
    assert "NOT_TRUE_FORWARD" in final_reasons(proxy, baseline, CFG)
    assert select_final([identity, proxy, aligned], baseline, CFG)["map_sha256"] == "aligned"


def test_high_c_unaligned_candidate_is_rejected() -> None:
    baseline = _metrics(5.0, 14.0, 16.0, -3)
    high = {"score_kind": "true_fixed_crop", "label": "candidate", "metrics": _metrics(7.0, 13.0, 14.0, 2),
            "map_sha256": "high", "regularization": 0.0}
    assert "OFFSET_OUTSIDE_ONE_FRAME" in final_reasons(high, baseline, CFG)


def test_map_proposal_contains_nonidentity_neighbours() -> None:
    mapping = build_map(134, 130)
    proposals = propose_maps({"map": mapping}, .25)
    assert proposals
    assert all(item["map_sha256"] != mapping["map_sha256"] for _, item in proposals)


def test_vectorized_coverage_keeps_naive_route_order() -> None:
    identity = build_map(134, 130)
    maps = {identity["map_sha256"]: identity}
    for _, mapping in propose_maps({"map": identity}, .25):
        maps[mapping["map_sha256"]] = mapping
    rows = []
    for index, mapping in enumerate(list(maps.values())[:35]):
        rows.append({"map_sha256": mapping["map_sha256"], "map": mapping,
                     "regularization": float(index % 4),
                     "proxy_metrics": {"d0": float(index % 9), "offset": index % 3 - 1,
                                       "sync_c": float((index * 7) % 13)}})
    support = np.arange(15, 108, dtype=np.int64)
    actual = choose_routes(rows, support, counts=(3, 3, 5), proxy=True)
    chosen: list[dict] = []
    def d0(row: dict) -> tuple:
        m = row["proxy_metrics"]
        return (m["d0"], abs(m["offset"]), -m["sync_c"], row["regularization"], row["map_sha256"])
    def c(row: dict) -> tuple:
        m = row["proxy_metrics"]
        return (abs(m["offset"]) > 1, -m["sync_c"], m["d0"], row["regularization"], row["map_sha256"])
    for row in sorted(rows, key=d0):
        if len(chosen) == 3:
            break
        chosen.append(row)
    for row in sorted(rows, key=c):
        if len(chosen) == 6:
            break
        if row not in chosen:
            chosen.append(row)
    while len(chosen) < 11:
        remaining = [row for row in rows if row not in chosen]
        def distance(a: dict, b: dict) -> float:
            x = np.asarray(a["map"]["delta"], dtype=np.float32)[support]
            y = np.asarray(b["map"]["delta"], dtype=np.float32)[support]
            return float(np.mean(np.abs(x - y), dtype=np.float32))
        chosen.append(min(remaining, key=lambda row: (-min(distance(row, old) for old in chosen), *d0(row))))
    assert [row["map_sha256"] for row in actual] == [row["map_sha256"] for row in chosen]
