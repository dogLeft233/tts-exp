from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
               "cross_tfg_audio_head_20261003" / "fit40_spsa_bootstrap_cli_attempt01.py")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_bootstrap_cli_attempt01", MODULE_PATH)
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


if __name__ == "__main__":
    unittest.main()
