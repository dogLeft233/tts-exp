import importlib.util
from pathlib import Path


def test_report_preserves_paired_delta_direction_and_precision():
    p = Path(__file__).resolve().parents[1] / "scripts/45_render_video_metric_report.py"
    spec = importlib.util.spec_from_file_location("video_metric_report", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    record = {"a": "natural", "b": "tts", "paired": {"niqe": {"delta": -0.125}}}
    assert mod.pair_delta({"pairs": [record]}, "natural", "tts") is record
    assert mod.pair_delta({"pairs": [record]}, "tts", "natural") == {}
    assert mod.fmt(5.1254) == "5.125"
    assert mod.fmt(-0.1254, plus=True) == "-0.125"
