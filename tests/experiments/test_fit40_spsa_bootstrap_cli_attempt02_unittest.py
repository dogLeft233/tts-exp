from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest


MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
               "cross_tfg_audio_head_20261003" / "fit40_spsa_bootstrap_cli_attempt02.py")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_bootstrap_cli_attempt02", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def _valid_documents():
    collector_pin = {"path": "/repo/collector05.py", "bytes": 100, "sha256": "a" * 64}
    index_pin = {"path": "/handoff/index.json", "bytes": 200, "sha256": "b" * 64,
                 "self_sha256": "c" * 64}
    family_pin = {"path": "/repo/family.json", "bytes": 300, "sha256": "d" * 64,
                  "self_sha256": "e" * 64}
    digest = "f" * 64
    index = {
        "schema": bootstrap.INGEST_SCHEMA,
        "status": "READY_FOR_LOCAL_INGEST_NOT_SELECTION",
        "expected_row_count": 720,
        "model_ids": list(bootstrap.MODEL_ORDER),
        "sample_ids": [f"sample{i:02d}" for i in range(40)],
        "models": [{"model_id": model} for model in bootstrap.MODEL_ORDER],
        "raw_matrix_npz_file_ids": {f"r{i}": f"npz{i}" for i in range(720)},
        "official_source_frame_map_file_ids": {f"r{i}": f"map{i}" for i in range(720)},
        "scientific_input_pins_sha256": digest,
    }
    family = {"scientific_input_pins_sha256": digest}
    proofs = ([{"proof_origin": "HISTORICAL_PREFIX_TERMINAL",
                "historical_exit_code_available": False, "historical_exit_code": None}
               for _ in range(34)]
              + [{"proof_origin": "ACTUAL_QUEUE_CHILD_RC", "child_exit_code": 0}
                 for _ in range(86)])
    handoff = {
        "schema": bootstrap.HANDOFF_SCHEMA,
        "status": "PASS_FULL_720_TERMINAL_READBACK_READY_FOR_SELECTION_INGEST",
        "selection_performed": False,
        "scientific_initialization_performed": False,
        "expected_rows": 720,
        "verified_rows": 720,
        "rows_by_model": {model: 240 for model in bootstrap.MODEL_ORDER},
        "scientific_input_pins_sha256": digest,
        "initial_ingest_index": index_pin,
        "composed_family_plan": family_pin,
        "historical_prefix": {"blocks": 34, "child_exit_code_available": False,
                               "child_exit_code": None},
        "actual_queue": {"new_blocks": 86, "new_rows": 516,
                          "process_rc_required": 0, "process_rc_verified_count": 86,
                          "paused_models": [], "active_claim": None},
        "source_block_proofs": proofs,
        "source_code_pins": [collector_pin],
    }
    pin_bundle = {
        "handoff_pin": {"path": "/handoff/terminal.json", "bytes": 400, "sha256": "1" * 64},
        "index_pin": index_pin,
        "family_pin": family_pin,
        "collector_pin": collector_pin,
    }
    return handoff, index, family, pin_bundle


class BootstrapHandoffGateTest(unittest.TestCase):
    def test_accepts_complete_actual_handoff_and_keeps_historical_rc_unknown(self) -> None:
        handoff, index, family, pins = _valid_documents()
        bootstrap._verify_handoff_documents(handoff, index, family, **pins)

    def test_rejects_missing_live_exit_proof(self) -> None:
        handoff, index, family, pins = _valid_documents()
        handoff["source_block_proofs"][-1]["child_exit_code"] = None
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._verify_handoff_documents(handoff, index, family, **pins)

    def test_rejects_fabricated_historical_return_code(self) -> None:
        handoff, index, family, pins = _valid_documents()
        handoff["source_block_proofs"][0]["historical_exit_code"] = 0
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._verify_handoff_documents(handoff, index, family, **pins)

    def test_rejects_incomplete_paused_or_active_queue(self) -> None:
        for edit in (
            lambda h: h.update(verified_rows=719),
            lambda h: h["actual_queue"].update(paused_models=[bootstrap.MODEL_ORDER[0]]),
            lambda h: h["actual_queue"].update(active_claim={"pid": 1}),
        ):
            handoff, index, family, pins = _valid_documents()
            edit(handoff)
            with self.subTest(handoff=handoff):
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap._verify_handoff_documents(handoff, index, family, **pins)

    def test_rejects_changed_index_and_wrong_collector_source(self) -> None:
        handoff, index, family, pins = _valid_documents()
        index["raw_matrix_npz_file_ids"].pop("r0")
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._verify_handoff_documents(handoff, index, family, **pins)
        handoff, index, family, pins = _valid_documents()
        expected_collector_pin = dict(pins["collector_pin"])
        handoff["source_code_pins"][0]["sha256"] = "0" * 64
        pins["collector_pin"] = expected_collector_pin
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._verify_handoff_documents(handoff, index, family, **pins)

    def test_actual_source_closure_matches_frozen_pins(self) -> None:
        root = Path(__file__).resolve().parents[2]
        closure = bootstrap._verify_source_closure(root)
        self.assertEqual(set(closure), set(bootstrap.SOURCE_PINS))

    def test_cli_help_is_parse_only(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            bootstrap.main(["--help"])
        self.assertEqual(caught.exception.code, 0)


def _write_self_hashed(path: Path, document: dict, field: str) -> dict:
    document[field] = bootstrap._self_hash(document, field)
    raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": bootstrap._sha(raw), "self_sha256": document[field]}


class _FakeSession:
    def __init__(self, session_root: Path, tell_status: str = "UPDATED",
                 journal_complete: bool = True) -> None:
        self.root = session_root
        self.manifest = {
            "session_id": "synthetic-session-test-only",
            "composed_family_plan": {"path": "/family.json", "sha256": "a" * 64},
            "initial_selector_receipt": {"path": "/receipt.json", "sha256": "b" * 64},
            "execution_package_manifest": {"path": "/package.json", "sha256": "c" * 64},
        }
        self.events = [{"event_type": "test-only"}]
        self.dispatch_intents = set()
        self._tell_status = tell_status
        self._journal_complete = journal_complete
        self.durable_tell_written = False
        self.pair = SimpleNamespace(
            plus=SimpleNamespace(proposal_id="test-plus", side="plus"),
            minus=SimpleNamespace(proposal_id="test-minus", side="minus"),
            iteration=1, pair_id="test-pair",
        )

    @classmethod
    def resume(cls, *, session_dir: Path):
        return _FAKE_SESSION

    def ask_pair(self):
        return self.pair

    def run_pending_pair(self, dispatcher):
        self.durable_tell_written = True
        self.dispatcher_bound = dispatcher.bound
        return SimpleNamespace(status=self._tell_status)

    def controller_journal_document(self):
        count = 20 if self._journal_complete else 19
        return {
            "schema": bootstrap.INITIAL_RECEIPT_SCHEMA.replace(
                "initial_family_selector_receipt", "spsa_controller_journal"),
            "status": ("COMPLETE_20_ASK_TELL_PAIRS" if self._journal_complete
                       else "INCOMPLETE_OR_HALTED"),
            "entries": [{"tell_result": {"status": "UPDATED"}} for _ in range(count)],
        }

    def write_controller_journal_exclusive(self, path: Path):
        document = self.controller_journal_document()
        document["schema"] = "fit40_spsa_controller_journal_v1"
        return _write_self_hashed(path, document, "journal_sha256")


_FAKE_SESSION = None


class _FakeSessionModule:
    JOURNAL_SCHEMA = "fit40_spsa_controller_journal_v1"
    DurableSPSASession = _FakeSession
    PointOutcome = object

    @staticmethod
    def _plain(value):
        return {"status": value.status}


class _FakeDispatcher:
    def __init__(self, **kwargs):
        self.bound = False

    def bind_pair_packet(self, pair, packet_index):
        self.bound = pair.pair_id == "test-pair" and packet_index.is_file()


class _FakeDispatcherModule:
    PointBlockDispatcher = _FakeDispatcher


class BootstrapMainOutcomeTest(unittest.TestCase):
    def _invoke_with_fake_modules(self, argv, fake_session):
        global _FAKE_SESSION
        _FAKE_SESSION = fake_session
        modules = {"session": _FakeSessionModule,
                   "dispatcher": _FakeDispatcherModule,
                   "packet_builder": object(), "collector": object()}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(bootstrap, "_load_frozen_modules", return_value=modules), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = bootstrap.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_run_pair_exit_code_requires_updated_tell_and_preserves_result(self) -> None:
        with self.subTest(status="UPDATED"):
            with self._run_pair_fixture("UPDATED") as (args, session):
                code, stdout, stderr = self._invoke_with_fake_modules(args, session)
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                result = json.loads(stdout)
                self.assertEqual(result["status"], "PAIR_TOLD_UPDATED")
                self.assertEqual(result["tell_result"], {"status": "UPDATED"})
                self.assertTrue(session.durable_tell_written)
                self.assertTrue(session.dispatcher_bound)
        with self.subTest(status="NO_UPDATE"):
            with self._run_pair_fixture("NO_UPDATE") as (args, session):
                code, stdout, stderr = self._invoke_with_fake_modules(args, session)
                self.assertEqual(code, 2)
                self.assertEqual(stderr, "")
                result = json.loads(stdout)
                self.assertEqual(result["status"], "PAIR_TOLD_NOT_UPDATED")
                self.assertEqual(result["tell_result"], {"status": "NO_UPDATE"})
                self.assertTrue(session.durable_tell_written)

    def test_seal_journal_create_only_requires_complete_twenty_updated_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_root = root / "session"
            session_root.mkdir()
            _write_self_hashed(session_root / "session_manifest.json",
                               {"schema": "test-only", "session_id": "synthetic"},
                               "manifest_sha256")
            session = _FakeSession(session_root, journal_complete=True)
            journal_path = root / "controller_journal.json"
            args = ["--repo-root", str(root), "seal-journal", "--session-dir",
                    str(session_root), "--journal-out", str(journal_path)]
            code, stdout, stderr = self._invoke_with_fake_modules(args, session)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(stderr, "")
            result = json.loads(stdout)
            self.assertEqual(result["status"], "COMPLETE_20_ASK_TELL_PAIRS")
            self.assertEqual(result["entry_count"], 20)
            self.assertTrue(journal_path.is_file())
            with self.assertRaises(FileExistsError):
                session.write_controller_journal_exclusive(journal_path)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_root = root / "session"
            session_root.mkdir()
            _write_self_hashed(session_root / "session_manifest.json",
                               {"schema": "test-only", "session_id": "synthetic"},
                               "manifest_sha256")
            session = _FakeSession(session_root, journal_complete=False)
            journal_path = root / "must_not_be_created.json"
            args = ["--repo-root", str(root), "seal-journal", "--session-dir",
                    str(session_root), "--journal-out", str(journal_path)]
            code, stdout, stderr = self._invoke_with_fake_modules(args, session)
            self.assertEqual(code, 2)
            self.assertFalse(journal_path.exists())
            self.assertIn("complete 20-pair UPDATED journal", stderr)

    @contextlib.contextmanager
    def _run_pair_fixture(self, tell_status):
        global _FAKE_SESSION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_root = root / "session"
            session_root.mkdir()
            _write_self_hashed(session_root / "session_manifest.json",
                               {"schema": "test-only", "session_id": "synthetic"},
                               "manifest_sha256")
            packet_dir = root / "packet"
            packet_dir.mkdir()
            (packet_dir / "packet_index.json").write_text("{}", encoding="utf-8")
            fake = _FakeSession(session_root, tell_status=tell_status)
            _FAKE_SESSION = fake
            args = ["--repo-root", str(root), "run-pair", "--session-dir",
                    str(session_root), "--packet-dir", str(packet_dir),
                    "--dispatch-root", str(root / "dispatch")]
            yield args, fake


if __name__ == "__main__":
    unittest.main()
