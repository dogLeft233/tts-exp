"""CPU-only structure tests for the tell-derived full-center dispatcher."""

from __future__ import annotations

import unittest

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit40_spsa_center_dispatcher_attempt01 as center_dispatch,
    fit_only_spsa_selection_attempt06 as selector,
)


MODELS = list(center_dispatch.MODEL_ORDER)
THETA = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4)


def _center(step: int = 5) -> selector.SnapshotSpec:
    return selector.SnapshotSpec(
        method_id=f"fit40_d0_spsa_center_k{step:02d}",
        artifact_sha256="a" * 64,
        binding_sha256="b" * 64,
        source_step=step,
        theta=THETA,
        point_kind="spsa_center",
    )


def _packet(step: int = 5):
    center = _center(step)
    method_id = center.method_id
    samples = [f"sample-{index:02d}" for index in range(40)]
    rows = []
    for sample in samples:
        natural = {"path": f"/fit/source_wav/{sample}.wav", "pcm_sha256": "c" * 64}
        for model in MODELS:
            rows.append({
                "row_id": f"{sample}-{model}-{method_id}",
                "split": "fit", "sample_id": sample, "model_id": model,
                "method_id": method_id, "natural_pcm": natural,
                "score_audio": dict(natural),
            })
    config = {
        "method": "fit40_spsa_center", "schema": "fit40_spsa_center_transform_v1",
        "method_id": method_id, "point_kind": "spsa_center", "source_step": step,
        "theta": list(THETA), "artifact_sha256": center.artifact_sha256,
        "binding_sha256": center.binding_sha256,
    }
    scope = {
        "scope_kind": "spsa_full_center", "point_kind": "spsa_center",
        "source_step": step, "method_id": method_id,
        "artifact_sha256": center.artifact_sha256,
        "binding_sha256": center.binding_sha256,
        "run_id": f"run-center-{step}",
    }
    plan = {
        "run_id": scope["run_id"], "rows": rows,
        "methods": [{"method_id": method_id, "config": config}],
        "models": [{"model_id": model} for model in MODELS],
    }
    return center, scope, plan, rows


class CenterDispatcherStructureTests(unittest.TestCase):
    def test_full_center_plan_preserves_40_by_3_fit_order(self):
        center, scope, plan, rows = _packet()
        center_dispatch.validate_center_plan_shape(center, scope, plan, rows)
        self.assertEqual(len(rows), 120)
        self.assertTrue(all(row["score_audio"] == row["natural_pcm"] for row in rows))

    def test_each_predeclared_center_step_is_accepted(self):
        for step in center_dispatch.CENTER_STEPS:
            with self.subTest(step=step):
                center, scope, plan, rows = _packet(step)
                center_dispatch.validate_center_plan_shape(center, scope, plan, rows)

    def test_rejects_partial_or_reordered_denominator(self):
        center, scope, plan, rows = _packet()
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, plan, rows[:-1])
        shuffled = list(rows)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, plan, shuffled)

    def test_rejects_nonfit_roster_or_changed_original_score_audio(self):
        center, scope, plan, rows = _packet()
        dev_rows = [dict(row) for row in rows]
        dev_rows[0]["split"] = "dev"
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, plan, dev_rows)
        changed_audio = [dict(row) for row in rows]
        changed_audio[0]["score_audio"] = {"path": "/tmp/generated.wav"}
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, plan, changed_audio)

    def test_rejects_theta_artifact_binding_model_and_step_tampering(self):
        center, scope, plan, rows = _packet()
        bad_plan = dict(plan)
        bad_method = dict(plan["methods"][0])
        bad_config = dict(bad_method["config"])
        bad_config["theta"] = [1.0] + list(THETA[1:])
        bad_method["config"] = bad_config
        bad_plan["methods"] = [bad_method]
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, bad_plan, rows)

        bad_scope = dict(scope, source_step=10)
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, bad_scope, plan, rows)

        bad_models = dict(plan, models=list(reversed(plan["models"])))
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.validate_center_plan_shape(center, scope, bad_models, rows)

    def test_center_spec_must_be_from_exact_completed_tell(self):
        center = _center(5)

        class Tell:
            status = "UPDATED"
            iteration = 5
            full_center_score_due = True
            full_center_spec = center

        class Controller:
            iteration = 20
            _pending = None
            halted = False
            history = [Tell() for _ in range(20)]

        journal = {
            "schema": "fit40_spsa_controller_journal_v1",
            "status": "COMPLETE_20_ASK_TELL_PAIRS", "metric_backend": "torch_cpu",
            "entries": [{"tell_result": {"status": "UPDATED", "iteration": 5,
                                           "full_center_spec": center_dispatch._plain(center)}}
                        for _ in range(20)],
        }
        journal["journal_sha256"] = center_dispatch._self_hash(journal, "journal_sha256")

        class Session:
            controller = Controller()

        self.assertEqual(center_dispatch.center_spec_from_verified_history(Session(), journal, 5), center)
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.center_spec_from_verified_history(Session(), journal, 6)
        journal["entries"][4]["tell_result"]["full_center_spec"]["theta"][0] += 1.0
        journal["journal_sha256"] = center_dispatch._self_hash(journal, "journal_sha256")
        with self.assertRaises(center_dispatch.CenterDispatchError):
            center_dispatch.center_spec_from_verified_history(Session(), journal, 5)


if __name__ == "__main__":
    unittest.main()
