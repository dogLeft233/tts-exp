import importlib.util
import json
from pathlib import Path


def test_extract_sync_scores_uses_current_repo_and_retains_condition(tmp_path):
    p = Path(__file__).resolve().parents[1] / "scripts/44_extract_sync_scores.py"
    spec = importlib.util.spec_from_file_location("video_sync_scores", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.REPO = tmp_path
    target = tmp_path / "runs/leaptalk_eval/04_eval/scores_leaptalk.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps([{"cond": "natural_raw", "i": 1, "c": 4.125},
                                  {"cond": "tts_raw", "i": 1, "c": 5.25},
                                  {"cond": "tts_raw", "i": 2, "c": None}]))
    assert mod.a1() == {"A1/natural_raw": {"1": 4.125}, "A1/tts_raw": {"1": 5.25}}
