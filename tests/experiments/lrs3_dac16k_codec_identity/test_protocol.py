from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.lrs3_dac16k_codec_identity.common import (
    assert_run_root_compatible,
    canonical_json_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def test_self_hashed_json_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_self_hashed_json(path, {"protocol_id": "p", "value": 1})
    assert verify_self_hashed_json(path)["value"] == 1
    payload = json.loads(path.read_text())
    payload["value"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_self_hashed_json(path)


def test_run_root_rejects_unbound_populated_root(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("x")
    with pytest.raises(ValueError, match="no protocol marker"):
        assert_run_root_compatible(tmp_path, "p")


def test_run_root_rejects_changed_protocol(tmp_path: Path) -> None:
    marker = tmp_path / "00_protocol" / "protocol.json"
    write_self_hashed_json(marker, {"protocol_id": "old"})
    with pytest.raises(ValueError, match="another protocol"):
        assert_run_root_compatible(tmp_path, "new")


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})
