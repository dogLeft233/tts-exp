from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.experiments.tts_evidence_temporal_patch import protocol


def _inventory(group_count: int = 20) -> dict:
    cells = []
    for index in range(group_count):
        group = f"g{index:02d}"
        base = {"source_group": group, "sample_id": f"s{index:02d}", "cell_key": f"{group}::s{index:02d}::natural", "arm": "natural", "audio": "n.wav", "textgrid": "n.TextGrid"}
        cells.append(base)
        for model in protocol.MODELS:
            cells.append({"source_group": group, "sample_id": f"s{index:02d}", "cell_key": f"{group}::s{index:02d}::{model}", "arm": model, "audio": f"{model}.wav", "textgrid": f"{model}.TextGrid"})
    return {"cells": cells}


def test_selection_has_disjoint_quotas_and_does_not_use_outcomes() -> None:
    inventory = _inventory()
    inventory["official_gain"] = object()
    inventory["silhouette"] = object()
    selected = protocol.select_patch_cohort(inventory)
    assert len(selected["rows"]) == 16
    assert len({row["source_group"] for row in selected["rows"]}) == 16
    assert {row["stage"] for row in selected["rows"]} == {"technical", "science"}
    assert selected["selection_inputs"].endswith("support_only")


def test_cache_key_changes_when_any_frozen_input_changes() -> None:
    first = protocol.cache_key(inputs={"audio": "a"}, parameters={"lambda": 0.5}, code={"worker": "x"}, environment={"python": "p"})
    second = protocol.cache_key(inputs={"audio": "b"}, parameters={"lambda": 0.5}, code={"worker": "x"}, environment={"python": "p"})
    assert first != second


def test_gpu_query_is_fail_closed_on_any_incomplete_command() -> None:
    calls = []

    def failing_runner(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="permission denied")

    with pytest.raises(protocol.ResourceUnknown):
        protocol.query_gpu_state(runner=failing_runner)
    assert len(calls) == 3


def test_gpu_query_collects_pmon_and_compute_pids() -> None:
    def runner(command):
        if any(str(item).startswith("--query-gpu") for item in command):
            return subprocess.CompletedProcess(command, 0, stdout="0, GPU-1, 0, 0\n", stderr="")
        if any(str(item).startswith("--query-compute-apps") for item in command):
            return subprocess.CompletedProcess(command, 0, stdout="123, python, 10\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="# gpu pid type sm mem enc dec\n0 456 C 0 0 0 0\n", stderr="")

    state = protocol.query_gpu_state(target_uuid="GPU-1", runner=runner)
    assert {row["pid"] for row in state["processes"]} == {"123", "456"}
