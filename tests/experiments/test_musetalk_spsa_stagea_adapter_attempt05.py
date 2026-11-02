from scripts.experiments.cross_tfg_audio_head_20261003 import (
    musetalk_spsa_stagea_adapter_attempt05 as adapter,
)


def test_composed_plan_keeps_model_descriptor_separate_from_source_protocol():
    descriptor_generator = {
        "renderer": "musetalk",
        "runtime_binding": "binding.json",
    }
    source_generator = {
        "audio_frontend": "frozen-wav2lip-frontend",
        "frame_count_formula": "floor(N*25/16000)",
    }
    plan = {
        "models": [{"model_id": adapter.MUSE_MODEL_ID,
                    "generator": descriptor_generator}],
        "execution": {"source_model_protocols": {
            adapter.MUSE_MODEL_ID: {"generator": source_generator},
        }},
        "protocol": {"generators_by_model": {
            adapter.MUSE_MODEL_ID: source_generator,
        }},
    }

    model, generator = adapter._resolve_muse_protocol(plan)

    assert model["generator"] == descriptor_generator
    assert generator == source_generator


def test_composed_protocol_map_must_match_exact_source_protocol():
    plan = {
        "models": [{"model_id": adapter.MUSE_MODEL_ID,
                    "generator": {"renderer": "musetalk"}}],
        "execution": {"source_model_protocols": {
            adapter.MUSE_MODEL_ID: {"generator": {"audio_frontend": "source"}},
        }},
        "protocol": {"generators_by_model": {
            adapter.MUSE_MODEL_ID: {"audio_frontend": "different"},
        }},
    }

    try:
        adapter._resolve_muse_protocol(plan)
    except adapter.MuseSPSAAdapterError as exc:
        assert "differs from source protocol" in str(exc)
    else:
        raise AssertionError("mismatched composed generator map was accepted")
