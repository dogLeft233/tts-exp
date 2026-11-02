from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import unittest


MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
               "cross_tfg_audio_head_20261003" /
               "fit40_spsa_twenty_pair_coordinator_attempt01.py")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_twenty_pair_coordinator_attempt01", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
coordinator_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator_mod)


def _write_session_manifest(path: Path) -> None:
    body = {"schema": coordinator_mod.SESSION_MANIFEST_SCHEMA,
            "status": "OPEN", "session_id": "synthetic-session"}
    body["manifest_sha256"] = coordinator_mod._sha(coordinator_mod._canonical(body))
    raw = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(raw)


@dataclass(frozen=True)
class _FakeProposal:
    proposal_id: str
    method_id: str
    side: str
    iteration: int
    pair_id: str
    theta: tuple[float, ...]


class _FakePair:
    def __init__(self, iteration: int):
        self.iteration = iteration
        self.pair_id = "pair-%02d" % iteration
        self.batch_id = "batch-%02d" % iteration
        self.plus = _FakeProposal("plus-%02d" % iteration, "method-plus-%02d" % iteration,
                                  "plus", iteration, self.pair_id, (1.0, 2.0))
        self.minus = _FakeProposal("minus-%02d" % iteration, "method-minus-%02d" % iteration,
                                   "minus", iteration, self.pair_id, (0.0, 1.0))


class _FakeSession:
    def __init__(self):
        self.manifest = {"session_id": "synthetic-session"}
        self.controller = SimpleNamespace(iteration=0)
        self.halted = False
        self.pending_pair = None
        self.events = [{"event_type": "SESSION_INITIALIZED", "event_sha256": "init"}]

    def ask_pair(self):
        if self.pending_pair is None:
            self.pending_pair = _FakePair(self.controller.iteration)
            self.events.append({"event_type": "PAIR_ASKED", "pair_id": self.pending_pair.pair_id,
                                "iteration": self.pending_pair.iteration,
                                "event_sha256": "ask-%02d" % self.pending_pair.iteration})
        return self.pending_pair

    def tell(self, pair: _FakePair, status: str):
        self.events.append({"event_type": "PAIR_TOLD", "pair_id": pair.pair_id,
                            "iteration": pair.iteration,
                            "tell_result": {"status": status},
                            "event_sha256": "tell-%02d" % pair.iteration})
        self.controller.iteration = pair.iteration + 1
        self.pending_pair = None
        self.halted = status != "UPDATED"


def _make_coordinator(temp: Path):
    repo = Path(__file__).resolve().parents[2]
    session_dir = temp / "session"
    session_dir.mkdir()
    _write_session_manifest(session_dir / "session_manifest.json")
    python = Path(sys.executable).resolve(strict=True)
    return coordinator_mod.TwentyPairCoordinator.create(
        coordinator_dir=temp / "coordinator", session_dir=session_dir,
        repo_root=repo, python_executable=python), session_dir


def _fake_runner(session: _FakeSession, mode: str = "UPDATED", *, calls: list | None = None):
    def run(argv, directory):
        if calls is not None:
            calls.append(list(argv))
        intent_path = directory / "launch_intent.json"
        assert intent_path.is_file(), "durable intent must precede child launch"
        intent, _raw, intent_pin = coordinator_mod._read_self_json(
            intent_path, "test intent", "intent_sha256")
        pair = session.pending_pair
        assert pair is not None and pair.pair_id == intent["pair_id"]
        start = coordinator_mod._seal({
            "schema": coordinator_mod.PAIR_START_SCHEMA, "status": "STARTED",
            "coordinator_id": intent["coordinator_id"], "child_pid": 12345,
            "child_start_ticks": 99, "started_utc": "2026-10-10T00:00:00.000000Z",
            "argv": list(argv)}, "start_sha256")
        start_pin = coordinator_mod._write_json_exclusive(directory / "process_start.json",
                                                          start, "start_sha256")
        updated = mode == "UPDATED"
        tell_status = "UPDATED" if updated else "NO_UPDATE" if mode == "NO_UPDATE" else "UPDATED"
        child_json = {"status": "PAIR_TOLD_UPDATED" if updated else "PAIR_TOLD_NOT_UPDATED",
                      "session_id": "synthetic-session", "iteration": pair.iteration,
                      "pair_id": pair.pair_id, "tell_result": {"status": tell_status}}
        stdout = (json.dumps(child_json, separators=(",", ":")) + "\n").encode()
        stderr = b""
        (directory / "child.stdout").write_bytes(stdout)
        (directory / "child.stderr").write_bytes(stderr)
        if mode != "UNKNOWN":
            session.tell(pair, tell_status)
        return {"pid": 12345, "start_ticks": 99,
                "return_code": 0 if updated else 2,
                "started_utc": "2026-10-10T00:00:00.000000Z",
                "ended_utc": "2026-10-10T00:00:01.000000Z",
                "launch_pin": start_pin,
                "stdout_path": str((directory / "child.stdout").resolve()),
                "stdout_bytes": len(stdout), "stdout_sha256": coordinator_mod._sha(stdout),
                "stderr_path": str((directory / "child.stderr").resolve()),
                "stderr_bytes": len(stderr), "stderr_sha256": coordinator_mod._sha(stderr),
                "stdout_json": child_json, "exception": None}
    return run


class TwentyPairCoordinatorTests(unittest.TestCase):
    def test_twenty_updated_pairs_are_intent_first_and_resume_verified(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            coordinator, _session_dir = _make_coordinator(temp)
            session = _FakeSession()
            calls = []
            result = coordinator.run(session_loader=lambda: session,
                                     child_runner=_fake_runner(session, calls=calls))
            self.assertEqual(result["status"], "COMPLETE_20_UPDATED")
            self.assertEqual(result["completed_pairs"], 20)
            self.assertEqual(len(calls), 20)
            for index in range(1, 21):
                directory = coordinator.root / "pairs" / ("pair_%02d" % index)
                self.assertTrue((directory / "launch_intent.json").is_file())
                self.assertTrue((directory / "process_start.json").is_file())
                receipt, _raw, _pin = coordinator_mod._read_self_json(
                    directory / "process_receipt.json", "test receipt", "receipt_sha256")
                self.assertEqual(receipt["status"], "COMPLETE_UPDATED")
                self.assertEqual(receipt["child_exit_code"], 0)
            replay = coordinator.run(session_loader=lambda: session,
                                     child_runner=lambda *_: self.fail("completed pairs must not rerun"))
            self.assertEqual(replay["status"], "COMPLETE_20_UPDATED")

    def test_no_update_is_terminal_and_stops_before_second_pair(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, _session_dir = _make_coordinator(Path(td))
            session = _FakeSession()
            calls = []
            result = coordinator.run(session_loader=lambda: session,
                                     child_runner=_fake_runner(session, "NO_UPDATE", calls=calls))
            self.assertEqual(result["status"], "NO_UPDATE")
            self.assertEqual(result["completed_pairs"], 0)
            self.assertEqual(len(calls), 1)
            again = coordinator.run(session_loader=lambda: session,
                                    child_runner=lambda *_: self.fail("NO_UPDATE must not rerun"))
            self.assertEqual(again["status"], "NO_UPDATE")

    def test_unconfirmed_child_result_is_unknown_and_never_retried(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, _session_dir = _make_coordinator(Path(td))
            session = _FakeSession()
            calls = []
            result = coordinator.run(session_loader=lambda: session,
                                     child_runner=_fake_runner(session, "UNKNOWN", calls=calls))
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertEqual(len(calls), 1)
            again = coordinator.run(session_loader=lambda: session,
                                    child_runner=lambda *_: self.fail("UNKNOWN must not rerun"))
            self.assertEqual(again["status"], "UNKNOWN")

    def test_existing_unreceipted_intent_is_read_back_not_relaunched(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, session_dir = _make_coordinator(Path(td))
            session = _FakeSession()
            pair = session.ask_pair()
            repo = Path(__file__).resolve().parents[2]
            directory = coordinator._pair_dir(1)
            intent = coordinator._make_pair_intent(index=1, pair=pair, session=session,
                                                   repo=repo, session_dir=session_dir)
            coordinator_mod._write_json_exclusive(directory / "launch_intent.json", intent,
                                                  "intent_sha256")
            calls = []
            result = coordinator.run(session_loader=lambda: session,
                                     child_runner=_fake_runner(session, calls=calls))
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertEqual(calls, [])
            readback = coordinator.root / "pairs" / "pair_01" / "unknown_readback.json"
            self.assertTrue(readback.is_file())
            second = coordinator.run(session_loader=lambda: session,
                                     child_runner=lambda *_: self.fail("ambiguous intent is non-dispatchable"))
            self.assertEqual(second["status"], "UNKNOWN")

    def test_matching_live_pid_start_ticks_is_reported_running(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, session_dir = _make_coordinator(Path(td))
            session = _FakeSession()
            pair = session.ask_pair()
            repo = Path(__file__).resolve().parents[2]
            directory = coordinator._pair_dir(1)
            intent = coordinator._make_pair_intent(index=1, pair=pair, session=session,
                                                   repo=repo, session_dir=session_dir)
            coordinator_mod._write_json_exclusive(directory / "launch_intent.json", intent,
                                                  "intent_sha256")
            pid, ticks = os.getpid(), 777
            start = {"schema": coordinator_mod.PAIR_START_SCHEMA, "status": "STARTED",
                     "coordinator_id": coordinator.manifest["coordinator_id"],
                     "child_pid": pid, "child_start_ticks": ticks,
                     "started_utc": "2026-10-10T00:00:00.000000Z", "argv": []}
            coordinator_mod._write_json_exclusive(directory / "process_start.json", start,
                                                  "start_sha256")
            calls = []
            with patch.object(coordinator_mod, "_proc_start_ticks",
                              side_effect=lambda observed_pid, proc_root=Path("/proc"):
                              ticks if observed_pid == pid else None):
                result = coordinator.run(session_loader=lambda: session,
                                         child_runner=lambda *args: calls.append(args))
            self.assertEqual(result["status"], "RUNNING")
            self.assertEqual(calls, [])
            self.assertFalse((directory / "unknown_readback.json").exists())

    @unittest.skipUnless(os.name == "posix", "Linux /proc process start ticks are required")
    def test_actual_popen_captures_pid_ticks_and_numeric_return_code(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, session_dir = _make_coordinator(Path(td))
            session = _FakeSession()
            pair = session.ask_pair()
            repo = Path(__file__).resolve().parents[2]
            directory = coordinator._pair_dir(1)
            intent = coordinator._make_pair_intent(index=1, pair=pair, session=session,
                                                   repo=repo, session_dir=session_dir)
            coordinator_mod._write_json_exclusive(directory / "launch_intent.json", intent,
                                                  "intent_sha256")
            child_stdout = json.dumps({"status": "CHILD_FIXTURE_ONLY"}, separators=(",", ":"))
            argv = [sys.executable, "-c", "print(" + repr(child_stdout) + ")"]
            observed = coordinator._child_command(argv, directory)
            self.assertEqual(observed["return_code"], 0)
            self.assertIs(type(observed["pid"]), int)
            self.assertIs(type(observed["start_ticks"]), int)
            self.assertEqual(observed["stdout_json"], {"status": "CHILD_FIXTURE_ONLY"})
            start, _raw, _pin = coordinator_mod._read_self_json(
                directory / "process_start.json", "test process start", "start_sha256")
            self.assertEqual(start["child_pid"], observed["pid"])
            self.assertEqual(start["child_start_ticks"], observed["start_ticks"])

    def test_create_only_and_fixed_twenty_pair_limit(self):
        with tempfile.TemporaryDirectory() as td:
            coordinator, _session_dir = _make_coordinator(Path(td))
            with self.assertRaises(coordinator_mod.CoordinatorError):
                coordinator_mod.TwentyPairCoordinator.create(
                    coordinator_dir=coordinator.root,
                    session_dir=coordinator.manifest["session_dir"],
                    repo_root=coordinator.manifest["repo_root"],
                    python_executable=coordinator.manifest["python_runtime"]["resolved_path"])
            with self.assertRaises(coordinator_mod.CoordinatorError):
                coordinator.run(session_loader=lambda: _FakeSession(), child_runner=lambda *_: {}, max_pairs=19)


if __name__ == "__main__":
    unittest.main()
