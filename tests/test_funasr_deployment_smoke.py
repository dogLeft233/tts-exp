import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).parents[1] / "scripts/funasr_deployment_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_normalization_retains_original_character_mapping():
    assert smoke.normalize("你 好，啊。") == {"original": "你 好，啊。", "normalized": "你好啊", "original_indices": [0, 2, 4]}


def test_validator_rejects_unbound_and_out_of_range_boundaries():
    assert smoke.validate([[0, 100], [90, 250]], ["你"], 200) == [
        "nonmonotonic_or_overlap", "out_of_audio_range", "token_timestamp_count_mismatch"]
    assert smoke.validate([], [], 200) == ["empty_timestamps"]
    assert smoke.validate([[0, 100], [100, 200]], ["你", "好"], 200) == []
    assert smoke.validate([[-90, 100], [100, 200]], ["你", "好"], 200) == ["out_of_audio_range"]
