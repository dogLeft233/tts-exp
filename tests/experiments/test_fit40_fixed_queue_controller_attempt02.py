from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.cross_tfg_audio_head_20261003.fit40_fixed_queue_controller_attempt02 import (
    BLOCK_ROWS,
    EVENT_SCHEMA,
    METHODS,
    MODEL_ORDER,
    PROCESS_SCHEMA,
    FrozenBundle,
    FixedFitQueueController,
    ModelPinSpec,
    QueueError,
    SourcePin,
    _verify_outer_launch_record,
    canonical,
    current_process_identity,
    load_pinned_bundles,
    sha256,
)


def fake_bundles() -> tuple[FrozenBundle, ...]:
    bundles = []
    natural = {
        "file_sha256": "1" * 64,
        "pcm_sha256": "2" * 64,
        "sample_count": 80000,
        "sample_rate_hz": 16000,
        "channels": 1,
        "subtype": "PCM_16",
    }
    for model_index, model_id in enumerate(MODEL_ORDER):
        rows = []
        samples = []
        for clip in range(40):
            sample_id = f"BAC009S{clip:04d}W0001"
            samples.append(sample_id)
            for method in METHODS:
                rows.append({
                    "row_id": f"{model_id}:{sample_id}:{method}",
                    "model_id": model_id,
                    "method_id": method,
                    "sample_id": sample_id,
                    "speaker_id": f"S{clip:04d}",
                    "seed": 20261003,
                    "face_id": "portrait-v1",
                    "split": "fit",
                    "source_media_kind": "static_portrait",
                    "natural_pcm": dict(natural),
                    "score_audio": dict(natural),
                    "source_face": {"file_sha256": "3" * 64},
                    "source_video": {"file_sha256": None},
                })
        spec = ModelPinSpec(
            model_id=model_id,
            plan=SourcePin(f"plan-{model_index}", 1, str(model_index + 1) * 64),
            planned_rows=SourcePin(f"rows-{model_index}", 1, str(model_index + 4) * 64),
            runtime_binding=SourcePin(f"runtime-{model_index}", 1, str(model_index + 7) * 64),
            launch_sources=(SourcePin(f"launch-{model_index}", 1, str(model_index + 9) * 64),),
        )
        bundles.append(FrozenBundle(
            spec=spec,
            plan={"schema_version": "test"},
            rows=tuple(rows),
            row_ids=tuple(row["row_id"] for row in rows),
            sample_ids=tuple(samples),
            scientific_input_pins_sha256="a" * 64,
        ))
    return tuple(bundles)


class FixedQueueControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        bundles = fake_bundles()
        for bundle in bundles:
            bundle.plan["execution"] = {
                "output_root": str(self.root / "output" / bundle.spec.model_id)
            }
        self.controller = FixedFitQueueController(bundles, self.root / "queue")
        self.owner_pid, self.owner_ticks = current_process_identity()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def gates(self, starts: dict[str, int] | None = None):
        starts = starts or {model: 0 for model in MODEL_ORDER}
        return {
            model: {
                "status": "PASS",
                "model_id": model,
                "block_start_index": starts[model],
                "plan_raw_sha256": self.controller.bundles[model].spec.plan.sha256,
                "evidence_raw_sha256": "b" * 64,
            }
            for model in starts
        }

    def absent(self, starts: dict[str, int] | None = None):
        starts = starts or {model: 0 for model in MODEL_ORDER}
        return {
            model: {
                "status": "PASS",
                "exists": False,
                "model_id": model,
                "block_start_index": starts[model],
                "plan_raw_sha256": self.controller.bundles[model].spec.plan.sha256,
                "output_leaf": str(
                    Path(self.controller.bundles[model].plan["execution"]["output_root"])
                    / f"block_{starts[model] // 6:03d}_attempt01"
                ),
                "evidence_raw_sha256": "c" * 64,
            }
            for model in starts
        }

    def claim(self):
        return self.controller.claim_next(
            ready_model_ids=MODEL_ORDER,
            resource_gates=self.gates(),
            absent_leaf_proofs=self.absent(),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            nonce="d" * 32,
        )

    def write_terminal_files(self, claim, *, exit_code=0, corrupt_method=False):
        output = Path(claim["output_leaf"])
        output.mkdir(parents=True, exist_ok=True)
        bundle = self.controller.bundles[claim["model_id"]]
        expected = bundle.row_ids[claim["block_start_index"]:claim["block_start_index"] + BLOCK_ROWS]
        rows = []
        for row_id, planned in zip(expected, bundle.rows[claim["block_start_index"]:claim["block_start_index"] + BLOCK_ROWS]):
            rows.append({
                "row_id": row_id,
                "model_id": claim["model_id"],
                "method_id": ("wrong_method" if corrupt_method and not rows else planned["method_id"]),
                "sample_id": planned["sample_id"],
                "seed": planned["seed"],
                "status": "PASS",
            })
        ledger_raw = b"".join(canonical(row) + b"\n" for row in rows)
        ledger_path = output / "results.jsonl"
        ledger_path.write_bytes(ledger_raw)
        counts = {"FAILED": 0, "INCOMPLETE": 0, "PASS": BLOCK_ROWS}
        method_counts = [
            {"method_id": method, "model_id": claim["model_id"], "planned_rows": 1,
             "result_rows": 1, "status_counts": {"FAILED": 0, "INCOMPLETE": 0, "PASS": 1}}
            for method in METHODS
        ]
        manifest = {
            "schema_version": "cross-tfg-audio-eval-v1",
            "status": "BLOCK_COMPLETE",
            "block_index": claim["block_index"],
            "block_start_index": claim["block_start_index"],
            "block_rows": BLOCK_ROWS,
            "block_sample_id": bundle.rows[claim["block_start_index"]]["sample_id"],
            "planned_rows": 240,
            "parent_planned_rows": 240,
            "result_rows": BLOCK_ROWS,
            "plan_raw_sha256": bundle.spec.plan.sha256,
            "output_leaf": str(output.resolve()),
            "planned_rows_sha256": bundle.spec.planned_rows.sha256,
            "results_jsonl_sha256": sha256(ledger_raw),
            "status_counts": counts,
            "counts_by_model_method": method_counts,
        }
        manifest_raw = canonical(manifest) + b"\n"
        manifest_path = output / "result_manifest.json"
        manifest_path.write_bytes(manifest_raw)
        status = {
            "schema_version": "cross_tfg_audio_head_stagea_v1",
            "status": "BLOCK_COMPLETE",
            "block_index": claim["block_index"],
            "block_start_index": claim["block_start_index"],
            "block_rows": BLOCK_ROWS,
            "completed_rows": BLOCK_ROWS,
            "planned_rows": 240,
            "plan_raw_sha256": bundle.spec.plan.sha256,
            "planned_rows_sha256": bundle.spec.planned_rows.sha256,
            "pid": 7100,
            "result_manifest_sha256": sha256(manifest_raw),
            "status_counts": counts,
        }
        status_raw = canonical(status) + b"\n"
        status_path = output / "control" / "run_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_bytes(status_raw)
        process = {
            "schema": PROCESS_SCHEMA,
            "claim_nonce": claim["claim_nonce"],
            "model_id": claim["model_id"],
            "block_index": claim["block_index"],
            "block_start_index": claim["block_start_index"],
            "plan_raw_sha256": bundle.spec.plan.sha256,
            "output_leaf": str(Path(claim["output_leaf"])),
            "terminal_status": "EXITED",
            "child_pid": 7100,
            "child_start_ticks": 80900,
            "exit_code": exit_code,
            "run_status_raw_sha256": sha256(status_raw),
            "result_manifest_raw_sha256": sha256(manifest_raw),
            "results_jsonl_raw_sha256": sha256(ledger_raw),
        }
        process["self_sha256"] = sha256(canonical(process))
        process_path = output / "process_terminal.json"
        process_path.write_bytes(canonical(process) + b"\n")
        return process_path, output

    def finish_pass(self, claim):
        process, output = self.write_terminal_files(claim)
        return self.controller.finish_claim(
            claim,
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            process_receipt_path=process,
            output_dir=output,
        )

    def test_exact_repository_allowlist_loads_all_three_frozen_plans(self):
        repo_root = Path(__file__).resolve().parents[2]
        bundles = load_pinned_bundles(repo_root)
        self.assertEqual(tuple(bundle.spec.model_id for bundle in bundles), MODEL_ORDER)
        self.assertEqual(tuple(len(bundle.rows) for bundle in bundles), (240, 240, 240))
        self.assertEqual(len({bundle.scientific_input_pins_sha256 for bundle in bundles}), 1)

    def test_detached_stdout_uses_full_receipt_for_row_and_source_pins(self):
        bundle = self.controller.bundles[MODEL_ORDER[1]]
        bundle.plan["run_id"] = "test-muse-run"
        claim = {
            "block_start_index": 0,
            "block_index": 0,
            "block_rows": BLOCK_ROWS,
            "row_ids": list(bundle.row_ids[:BLOCK_ROWS]),
            "output_leaf": str(self.root / "muse" / "block_000_attempt01"),
        }
        output = Path(claim["output_leaf"]).resolve()
        launch_pin = bundle.spec.launch_sources[0]

        def make_receipt(rows_sha: str) -> dict:
            receipt = {
                "schema": "musetalk_fit40_detached_block_launch_receipt_v1",
                "status": "STARTED_DETACHED", "run_id": "test-muse-run",
                "pid": 1234, "start_ticks": 5678, "block_index": 0,
                "block_start": 0, "block_rows": BLOCK_ROWS,
                "row_ids": claim["row_ids"], "output_root": str(output),
                "launcher": {"bytes": launch_pin.bytes, "sha256": launch_pin.sha256},
                "plan": {"bytes": bundle.spec.plan.bytes, "sha256": bundle.spec.plan.sha256},
                "planned_rows": {"bytes": bundle.spec.planned_rows.bytes, "sha256": rows_sha},
                "binding": {"bytes": bundle.spec.runtime_binding.bytes,
                            "sha256": bundle.spec.runtime_binding.sha256},
            }
            receipt["self_sha256"] = sha256(canonical(receipt))
            return receipt

        receipt_path = self.root / "launch_receipt.json"
        good = make_receipt(bundle.spec.planned_rows.sha256)
        good_raw = canonical(good)
        receipt_path.write_bytes(good_raw)
        stdout_record = {
            "status": "STARTED_DETACHED", "pid": 1234, "start_ticks": 5678,
            "run_id": "test-muse-run", "block_index": 0, "block_start": 0,
            "block_rows": BLOCK_ROWS, "output_root": str(output),
            "launch_receipt": str(receipt_path.resolve()),
            "launch_receipt_bytes": len(good_raw), "launch_receipt_sha256": sha256(good_raw),
        }
        self.assertEqual(
            _verify_outer_launch_record(stdout_record, claim, bundle, 1234, 5678, launch_pin.sha256),
            receipt_path,
        )

        bad_path = self.root / "launch_receipt_bad_rows.json"
        bad = make_receipt("f" * 64)
        bad_raw = canonical(bad)
        bad_path.write_bytes(bad_raw)
        bad_stdout = dict(stdout_record)
        bad_stdout.update({
            "launch_receipt": str(bad_path.resolve()),
            "launch_receipt_bytes": len(bad_raw),
            "launch_receipt_sha256": sha256(bad_raw),
        })
        with self.assertRaises(QueueError):
            _verify_outer_launch_record(bad_stdout, claim, bundle, 1234, 5678, launch_pin.sha256)

    def test_round_robin_advances_only_after_exact_six_pass_rows(self):
        first = self.claim()
        self.assertEqual(first["model_id"], MODEL_ORDER[0])
        pass_event = self.finish_pass(first)
        self.assertEqual(pass_event["event"], "BLOCK_PASS")
        self.assertEqual(self.controller.snapshot()["next_start_by_model"][MODEL_ORDER[0]], 6)
        second = self.controller.claim_next(
            ready_model_ids=MODEL_ORDER,
            resource_gates=self.gates({MODEL_ORDER[0]: 6, MODEL_ORDER[1]: 0, MODEL_ORDER[2]: 0}),
            absent_leaf_proofs=self.absent({MODEL_ORDER[0]: 6, MODEL_ORDER[1]: 0, MODEL_ORDER[2]: 0}),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            nonce="e" * 32,
        )
        self.assertEqual(second["model_id"], MODEL_ORDER[1])

    def test_one_model_failure_pauses_it_and_round_robin_serves_next_model(self):
        muse = self.controller.claim_next(
            ready_model_ids=MODEL_ORDER,
            resource_gates=self.gates(),
            absent_leaf_proofs=self.absent(),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            nonce="f" * 32,
        )
        self.assertEqual(muse["model_id"], MODEL_ORDER[0])
        self.finish_pass(muse)
        muse_start = 0
        ditto_start = 0
        ready = MODEL_ORDER
        claim = self.controller.claim_next(
            ready_model_ids=ready,
            resource_gates=self.gates({MODEL_ORDER[1]: muse_start, MODEL_ORDER[2]: ditto_start}),
            absent_leaf_proofs=self.absent({MODEL_ORDER[1]: muse_start, MODEL_ORDER[2]: ditto_start}),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            nonce="1" * 32,
        )
        self.assertEqual(claim["model_id"], MODEL_ORDER[1])
        process, output = self.write_terminal_files(claim, exit_code=1)
        outcome = self.controller.finish_claim(
            claim,
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            process_receipt_path=process,
            output_dir=output,
        )
        self.assertEqual(outcome["event"], "BLOCK_PAUSED_FAILURE")
        snapshot = self.controller.snapshot()
        self.assertIn(MODEL_ORDER[1], snapshot["paused_models"])
        next_claim = self.controller.claim_next(
            ready_model_ids=ready,
            resource_gates=self.gates({MODEL_ORDER[0]: 6, MODEL_ORDER[2]: 0}),
            absent_leaf_proofs=self.absent({MODEL_ORDER[0]: 6, MODEL_ORDER[2]: 0}),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            nonce="2" * 32,
        )
        self.assertEqual(next_claim["model_id"], MODEL_ORDER[2])

    def test_bad_result_row_pauses_model_instead_of_advancing(self):
        claim = self.claim()
        process, output = self.write_terminal_files(claim, corrupt_method=True)
        event = self.controller.finish_claim(
            claim,
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
            process_receipt_path=process,
            output_dir=output,
        )
        self.assertEqual(event["event"], "BLOCK_PAUSED_FAILURE")
        self.assertIn("result row does not match frozen plan identity", event["failure_reason"])
        self.assertEqual(self.controller.snapshot()["next_start_by_model"][MODEL_ORDER[0]], 0)

    def test_missing_or_wrong_live_gate_never_claims_a_block(self):
        claim = self.controller.claim_next(
            ready_model_ids=MODEL_ORDER,
            resource_gates={},
            absent_leaf_proofs=self.absent(),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
        )
        self.assertIsNone(claim)
        gates = self.gates()
        gates[MODEL_ORDER[0]]["plan_raw_sha256"] = "0" * 64
        claim = self.controller.claim_next(
            ready_model_ids=[MODEL_ORDER[0]],
            resource_gates=gates,
            absent_leaf_proofs=self.absent(),
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
        )
        self.assertIsNone(claim)

    def test_active_nonce_lock_blocks_parallel_claims_and_checks_owner(self):
        claim = self.claim()
        with self.assertRaises(QueueError):
            self.claim()
        process, output = self.write_terminal_files(claim)
        with self.assertRaises(QueueError):
            self.controller.finish_claim(
                claim,
                owner_pid=self.owner_pid + 1,
                owner_start_ticks=self.owner_ticks,
                process_receipt_path=process,
                output_dir=output,
            )

    def test_leaf_reuse_is_never_claimed(self):
        claim = self.claim()
        self.finish_pass(claim)
        gates = self.gates({MODEL_ORDER[0]: 6})
        leaves = self.absent({MODEL_ORDER[0]: 6})
        leaves[MODEL_ORDER[0]]["output_leaf"] = claim["output_leaf"]
        next_claim = self.controller.claim_next(
            ready_model_ids=[MODEL_ORDER[0]],
            resource_gates=gates,
            absent_leaf_proofs=leaves,
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_ticks,
        )
        self.assertIsNone(next_claim)


if __name__ == "__main__":
    unittest.main()
