from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.check import np_distance_matrix, np_metrics
from scripts.experiments.mfa_linear_video_retiming.check_v2 import _proxy_score, _reasons, verify_search_v2
from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError, canonical_json_sha256, file_sha256, write_json
from scripts.experiments.mfa_linear_video_retiming.retime import build_map, global_seed


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    base = tmp_path / "03_search" / "1"
    base.mkdir(parents=True)
    visual = np.zeros((60, 1024), dtype=np.float32)
    audio = np.zeros((60, 1024), dtype=np.float32)
    emb = base / "embedding.npz"
    audio_emb = base / "audio.npz"
    np.savez_compressed(emb, visual=visual)
    np.savez_compressed(audio_emb, audio=audio)
    support = list(range(15, 45))
    score = np_metrics(np_distance_matrix(visual, audio, 60), support)
    score["conservative_window_count"] = 60
    score["local_windows"] = [{"row_start": 15, "row_stop_exclusive": 40,
                               "d0": score["d0"], "offset": score["offset"]}]
    maps = [("identity", build_map(64, 60))] + [
        (f"global_seed_{amount:+d}", global_seed(64, 60, amount)) for amount in (-3, -2, -1, 1, 2, 3)]
    true = {}
    proxy = []
    for label, mapping in maps:
        key = mapping["map_sha256"]
        true[key] = {"label": label, "stage": "CALIBRATION" if label == "identity" else "ROOT",
                     "map_sha256": key, "map": mapping, "score_kind": "true_fixed_crop",
                     "metrics": score, "regularization": float(sum(abs(x) for x in mapping["delta"])),
                     "embedding_path": str(emb), "embedding_sha256": file_sha256(emb)}
        proxy_score = _proxy_score(visual, audio, np.asarray(mapping["q"]), np.asarray(support))
        proxy_score["curve"] = proxy_score["curve"].tolist()
        proxy.append({"map_sha256": key, "map": mapping, "score_kind": "proxy_embedding",
                      "stage": "A", "regularization": true[key]["regularization"], "proxy_metrics": proxy_score})
    identity = true[maps[0][1]["map_sha256"]]
    config = {"protocol": "mfa_linear_video_retiming_v2",
              "search": {"proxy_unique_maps": 20000, "max_seconds_per_sample": 1200,
                         "noninferiority_sync_c": .05, "noninferiority_sync_d": .05,
                         "minimum_d0_improvement": .02, "final_d0_slack": .02}}
    calibration = {"request_fingerprint": "fixture", "baseline_candidate": identity,
                   "natural_audio_embedding_path": str(audio_emb), "frozen_support": support}
    calibration_path = base / "state.json"
    write_json(calibration_path, calibration, self_hash=True)
    chunk = base / "proxy_chunks" / "chunk_00000.json"
    chunk.parent.mkdir()
    write_json(chunk, {"rows": proxy}, self_hash=True)
    state = {"protocol": config["protocol"], "calibration_state_sha256": file_sha256(calibration_path),
             "request_hash": canonical_json_sha256({"request": "fixture", "search": config["search"],
                                                     "calibration_sha256": file_sha256(calibration_path)}),
             "proxy_attempted": 7, "proxy_completed": 7, "stage_a_proxy_count": 7,
             "stage_b_proxy_count": 0, "proxy_chunks": [{"name": "proxy_chunks/chunk_00000.json",
                                                    "sha256": file_sha256(chunk), "count": 7}],
             "true_attempted": 6, "true_completed": 6, "true_candidates": true,
             "shortlist_a": [], "shortlist_b": [], "elapsed_search_seconds": 1.0,
             "final_gate": {key: _reasons(row, score, config["search"]) for key, row in true.items()},
             "selected_map_sha256": identity["map_sha256"]}
    state_path = base / "state_v2.json"
    write_json(state_path, state, self_hash=True)
    global_best = min([row for row in true.values() if row["stage"] == "ROOT"],
                      key=lambda row: (row["metrics"]["d0"], abs(row["metrics"]["offset"]),
                                       -row["metrics"]["sync_c"], row["regularization"], row["map_sha256"]))
    write_json(base / "result.json", {"protocol": config["protocol"], "selected": identity,
                                       "global_control": global_best}, self_hash=True)
    return tmp_path, {"records": [{"sample_id": "1"}]}, config


def test_checker_accepts_independently_recomputed_fixture(tmp_path: Path) -> None:
    root, frozen, config = _fixture(tmp_path)
    assert verify_search_v2(root, frozen, config)[0]["status"] == "PASS"


@pytest.mark.parametrize("tamper", ["proxy_metric", "shortlist", "selected_role", "gate"])
def test_checker_rejects_tampered_search_evidence(tmp_path: Path, tamper: str) -> None:
    root, frozen, config = _fixture(tmp_path)
    base = root / "03_search" / "1"
    state_path = base / "state_v2.json"
    state = json.loads(state_path.read_text())
    if tamper == "proxy_metric":
        chunk = base / state["proxy_chunks"][0]["name"]
        data = json.loads(chunk.read_text())
        data.pop("artifact_sha256")
        data["rows"][0]["proxy_metrics"]["sync_c"] += 1.0
        write_json(chunk, data, self_hash=True)
        state["proxy_chunks"][0]["sha256"] = file_sha256(chunk)
    elif tamper == "shortlist":
        state["shortlist_a"] = ["fabricated"]
    elif tamper == "selected_role":
        result_path = base / "result.json"
        result = json.loads(result_path.read_text())
        result.pop("artifact_sha256")
        result["selected"]["score_kind"] = "proxy_embedding"
        write_json(result_path, result, self_hash=True)
    else:
        state["final_gate"] = {}
    state.pop("artifact_sha256")
    write_json(state_path, state, self_hash=True)
    with pytest.raises(ProtocolError):
        verify_search_v2(root, frozen, config)
