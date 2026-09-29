from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError, file_sha256, verify_json, write_json
from scripts.experiments.mfa_linear_video_retiming.retime import build_map
from scripts.experiments.mfa_linear_video_retiming.run import (
    _is_ordered_prefix,
    _available_ram_bytes,
    _request,
    _syncnet_request,
    run_stage,
    stage_official,
    _timed_stage,
)
from scripts.experiments.mfa_linear_video_retiming.search_worker import (
    _valid_state_or_none,
    state_compatible,
)


def _request_fixture(tmp_path: Path):
    m_video = tmp_path / "M.mkv"
    n_video = tmp_path / "N.mkv"
    n_audio = tmp_path / "N.wav"
    m_audio = tmp_path / "M.wav"
    syncnet = tmp_path / "syncnet.model"
    legacy = tmp_path / "legacy_worker.py"
    for path, content in ((m_video, b"M-video"), (n_video, b"N-video"), (n_audio, b"N-audio"),
                          (m_audio, b"M-audio"), (syncnet, b"syncnet"), (legacy, b"legacy scorer")):
        path.write_bytes(content)
    record = {"sample_id": "1", "paired_key": "key1", "speaker_id": "speaker1",
              "audio": {"natural": {"path": str(n_audio), "sha256": file_sha256(n_audio)},
                        "mfa_linear": {"path": str(m_audio), "sha256": file_sha256(m_audio)}}}
    portrait = {"path": str(tmp_path / "face.png"), "rgb_pixel_sha256": "f" * 64,
                "score_box": {"box": [1, 2, 3, 4]}}
    frozen = {"portrait_bindings": {"portraits": {"3": portrait}}}
    generated_record = {"portraits": {"3": {
        "frame_count": 60, "valid_frame_count": 58,
        "N": {"canonical_video": str(n_video)},
        "M": {"canonical_video": str(m_video)},
    }}}
    config = {"repo_root": str(tmp_path),
              "paths": {"syncnet_root": str(tmp_path / "syncnet"),
                        "syncnet_model": str(syncnet), "legacy_score_worker": str(legacy)},
              "models": {"syncnet_batch_size": 4, "syncnet_model_sha256": file_sha256(syncnet)},
              "search": {"max_candidates": 256}}
    return config, frozen, generated_record, record, m_video, syncnet, legacy


def test_request_fingerprint_invalidates_audio_box_videos_and_model_code(tmp_path: Path) -> None:
    config, frozen, generated, record, m_video, syncnet, legacy = _request_fixture(tmp_path)

    def fingerprint() -> str:
        return _request(config, frozen, generated, tmp_path, record, "3", "search", "frozen-run-fingerprint")["request_fingerprint"]

    base = fingerprint()
    original_audio_sha = record["audio"]["natural"]["sha256"]
    record["audio"]["natural"]["sha256"] = "a" * 64
    assert fingerprint() != base
    record["audio"]["natural"]["sha256"] = original_audio_sha

    original_box = frozen["portrait_bindings"]["portraits"]["3"]["score_box"]["box"]
    frozen["portrait_bindings"]["portraits"]["3"]["score_box"]["box"] = [2, 2, 3, 4]
    assert fingerprint() != base
    frozen["portrait_bindings"]["portraits"]["3"]["score_box"]["box"] = original_box

    m_video.write_bytes(b"changed M video")
    assert fingerprint() != base
    m_video.write_bytes(b"M-video")
    assert fingerprint() == base

    syncnet.write_bytes(b"changed SyncNet checkpoint")
    assert fingerprint() != base
    syncnet.write_bytes(b"syncnet")
    legacy.write_bytes(b"changed scorer code")
    assert fingerprint() != base


def test_state_compatibility_checks_both_frozen_fingerprints() -> None:
    state = {"schema_version": 1, "request_fingerprint": "request-a", "scorer_fingerprint": "sync-a"}

    assert state_compatible(state, "request-a", "sync-a")
    assert not state_compatible(state, "request-b", "sync-a")
    assert not state_compatible(state, "request-a", "sync-b")


def test_resume_drops_partial_embedding_file_and_never_keeps_stale_scores(tmp_path: Path) -> None:
    embedding = tmp_path / "candidate.npz"
    embedding.write_bytes(b"embedding")
    state_path = tmp_path / "state.json"
    request = {"request_fingerprint": "request-a"}
    state = {"schema_version": 1, "request_fingerprint": "request-a", "scorer_fingerprint": "sync-a",
             "status": "SEARCHING", "candidates": {"candidate": {
                 "embedding_path": str(embedding), "embedding_sha256": file_sha256(embedding),
                 "metrics": {"sync_c": 99.0}}}, "cache_invalidations": []}
    write_json(state_path, state, self_hash=True)

    assert _valid_state_or_none(state_path, request, "sync-a")["candidates"]["candidate"]["metrics"]["sync_c"] == 99.0
    embedding.write_bytes(b"partial or replaced")
    resumed = _valid_state_or_none(state_path, request, "sync-a")
    assert resumed is not None
    assert resumed["candidates"] == {}
    assert resumed["cache_invalidations"][-1]["reason"] == "embedding_missing_or_hash_mismatch"
    with pytest.raises(ProtocolError, match="FINGERPRINT_MISMATCH"):
        _valid_state_or_none(state_path, {"request_fingerprint": "changed-audio-or-box"}, "sync-a")


def test_official_rows_may_resume_only_as_an_ordered_prefix() -> None:
    expected = ["p3_s1_vN_aN", "p3_s1_vM_aN", "p3_s1_vR_aN"]

    assert _is_ordered_prefix([], expected)
    assert _is_ordered_prefix(expected[:2], expected)
    assert _is_ordered_prefix(expected, expected)
    assert not _is_ordered_prefix([expected[0], expected[2]], expected)
    assert not _is_ordered_prefix(expected + ["unexpected"], expected)


def test_official_stage_resumes_the_missing_suffix_without_rescoring_passed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.experiments.mfa_linear_video_retiming.official as official
    import scripts.experiments.mfa_linear_video_retiming.run as runner

    run_dir = tmp_path / "run"
    sealed_path = run_dir / "04_sealed" / "manifest.json"
    transfer_path = run_dir / "05_transfer" / "manifest.json"
    write_json(sealed_path, {"status": "SEALED", "records": []}, self_hash=True)
    write_json(transfer_path, {"status": "COMPLETE", "records": []}, self_hash=True)
    cells = []
    for arm in ("N", "GLOBAL"):
        video = tmp_path / f"{arm}.mkv"
        audio = tmp_path / f"{arm}.wav"
        video.write_bytes(arm.encode())
        audio.write_bytes((arm + "-audio").encode())
        cells.append({"sample_id": "1", "paired_key": "key1", "speaker_id": "speaker1",
                      "portrait_id": "3", "video_arm": arm, "audio_role": "N",
                      "video": str(video), "audio": str(audio), "status": "PASS"})
    monkeypatch.setattr(runner, "_official_cells", lambda *args, **kwargs: cells)
    first_key = "p3_s1_vN_aN"
    first_result_path = run_dir / "06_official" / "cells" / first_key / "result.json"
    syncnet = tmp_path / "syncnet.model"
    syncnet.write_bytes(b"syncnet")
    first_result = {"status": "PASS", "cell_key": first_key,
                    "video": cells[0]["video"], "audio": cells[0]["audio"],
                    "syncnet_model_sha256": file_sha256(syncnet),
                    "video_sha256": file_sha256(cells[0]["video"]),
                    "audio_sha256": file_sha256(cells[0]["audio"])}
    write_json(first_result_path, first_result, self_hash=True)
    old_row = {"cell_key": first_key, "result_path": str(first_result_path), "status": "PASS",
               "sample_id": "1", "paired_key": "key1", "speaker_id": "speaker1", "portrait_id": "3",
               "video_arm": "N", "audio_role": "N", "video": cells[0]["video"], "audio": cells[0]["audio"],
               "video_sha256": file_sha256(cells[0]["video"]), "audio_sha256": file_sha256(cells[0]["audio"])}
    write_json(run_dir / "06_official" / "manifest.json", {"status": "RUNNING", "rows": [old_row]}, self_hash=True)
    calls = []

    def fake_score(config, cell, *, run_dir, resume, max_elapsed_seconds):
        calls.append(cell["video_arm"])
        key = f"p{cell['portrait_id']}_s{cell['sample_id']}_v{cell['video_arm']}_a{cell['audio_role']}"
        curve = {"d0": 1.0, "support_columns": 40}
        payload = {"status": "PASS", "cell_key": key,
                   "video": cell["video"], "audio": cell["audio"],
                   "syncnet_model_sha256": file_sha256(syncnet),
                   "video_sha256": file_sha256(cell["video"]), "audio_sha256": file_sha256(cell["audio"]),
                   "official_sync_c": 1.0, "official_sync_d": 2.0, "official_offset": 0,
                   "recomputed_official_curve": curve,
                   "activesd": {"path": str(tmp_path / "activesd.pckl"), "sha256": "a" * 64},
                   "crop": {"frame_count": 40}}
        write_json(run_dir / "06_official" / "cells" / key / "result.json", payload, self_hash=True)
        return payload

    monkeypatch.setattr(official, "score_official_cell", fake_score)
    result = stage_official({"paths": {"syncnet_model": str(syncnet)}}, run_dir, {"records": []}, smoke=True)

    assert result["status"] == "COMPLETE"
    assert calls == ["GLOBAL"]
    assert [row["cell_key"] for row in result["rows"]] == [first_key, "p3_s1_vGLOBAL_aN"]
    assert result["rows"][1]["speaker_id"] == "speaker1"


def test_failed_timed_gpu_stage_still_consumes_the_persisted_budget(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = {"budget": {"active_gpu_seconds": 0.0}, "stage_active_gpu_seconds": {}}
    write_json(protocol_path, protocol, self_hash=True)

    def fail() -> None:
        raise RuntimeError("simulated worker failure")

    with pytest.raises(RuntimeError, match="simulated"):
        _timed_stage(protocol_path, protocol, "search", fail)
    saved = verify_json(protocol_path, self_hash=True)
    assert saved["budget"]["active_gpu_seconds"] > 0.0
    assert saved["stage_active_gpu_seconds"]["search"] > 0.0


def test_syncnet_timeout_at_global_budget_is_reported_as_budget_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"paths": {"syncnet_python": "/venv/syncnet/bin/python"}, "repo_root": str(tmp_path),
              "timeouts": {"syncnet_seconds": 30}}
    request = {"mode": "search", "state_path": str(tmp_path / "1" / "state.json"),
               "result_path": str(tmp_path / "1" / "result.json"), "sample_id": "1"}
    observed: dict[str, object] = {}

    def timeout(*args, **kwargs):
        observed["command"] = args[0]
        raise __import__("subprocess").TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("scripts.experiments.mfa_linear_video_retiming.run.subprocess.run", timeout)
    with pytest.raises(ProtocolError, match="BUDGET_LIMITED"):
        _syncnet_request(config, request, log=tmp_path / "worker.log", timeout_seconds=5)
    assert "run_request(sys.argv[1])" in observed["command"][2]


def test_audit_resource_wait_is_reported_without_a_traceback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import scripts.experiments.mfa_linear_video_retiming.run as runner

    def wait(*args, **kwargs):
        raise ProtocolError("RESOURCE_WAIT:RAM_BELOW_MINIMUM")

    monkeypatch.setattr(runner, "_load_run_context", wait)

    assert run_stage(Path("unused.yaml"), "run", "audit") == 2
    assert "audit RESOURCE_WAIT: RESOURCE_WAIT:RAM_BELOW_MINIMUM" in capsys.readouterr().err


def test_memory_fallback_uses_memavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1000000 kB\nMemFree: 1000 kB\nMemAvailable: 900000 kB\n", encoding="ascii")
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert _available_ram_bytes(meminfo) == 900000 * 1024
