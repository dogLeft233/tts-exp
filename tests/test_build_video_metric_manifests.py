import importlib.util
from pathlib import Path


def load():
    p = Path(__file__).resolve().parents[1] / "scripts/42_build_video_metric_manifests.py"
    spec = importlib.util.spec_from_file_location("video_manifest", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_manifests_follow_selected_repository_and_skip_absent_models(tmp_path):
    mod = load(); mod.REPO = tmp_path
    p = tmp_path / "results/imtalker/natural_raw/1.mp4"
    p.parent.mkdir(parents=True); p.write_bytes(b"video")
    assert mod.family_a2() == [{"group": "A2/imtalker/natural_raw", "item": "1",
                                "video": "results/imtalker/natural_raw/1.mp4"}]


def test_leaptalk_items_are_sorted_numerically(tmp_path):
    mod = load(); mod.REPO = tmp_path
    folder = tmp_path / "runs/leaptalk_multiset/tts_raw"; folder.mkdir(parents=True)
    for name in ["10", "2", "1"]: (folder / (name + ".mp4")).touch()
    assert [e["item"] for e in mod.family_a1()] == ["1", "2", "10"]
