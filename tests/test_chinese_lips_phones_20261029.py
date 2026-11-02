import importlib.util
from pathlib import Path

import pytest
import json
import zipfile


spec = importlib.util.spec_from_file_location("lipsphones", Path(__file__).parents[1] / "scripts/experiments/chinese_lips_phones_20261029.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_segmentation_does_not_hide_unknown_or_punctuation():
    assert m.segment("你，好世界！", {"你", "好", "世界"}) == (["你", "好", "世界"], [])
    assert m.segment("你好X", {"你", "好"}) == (["你", "好", "<unk>"], ["X"])


def test_grid_phone_tier_not_word_tier(tmp_path):
    p = tmp_path / "a.TextGrid"
    p.write_text('''item [1]:
    name = "words"
    intervals [1]:
        xmin = 0
        xmax = 0.5
        text = "你好"
item [2]:
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = "n"
    intervals [2]:
        xmin = 0.2
        xmax = 0.5
        text = "i˧˥"
''', encoding="utf-8")
    result = m.parse_textgrid(p)
    assert [(x["start"], x["end"], x["phone"]) for x in result] == [(0, .2, "n"), (.2, .5, "i")]
    assert result[-1]["raw_phone"] == "i˧˥"


def test_bad_alignment_rejected(tmp_path):
    p = tmp_path / "bad.TextGrid"
    p.write_text('''item [1]:
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.4
        text = "a"
    intervals [2]:
        xmin = 0.3
        xmax = 0.5
        text = "b"
''', encoding="utf-8")
    with pytest.raises(ValueError, match="Overlapping"):
        m.parse_textgrid(p)


def test_dictionary_eligibility_matches_actual_model_inventory(tmp_path):
    acoustic = tmp_path / "model.zip"
    with zipfile.ZipFile(acoustic, "w") as z:
        z.writestr("model/meta.json", json.dumps({"phones": ["a", "b"]}))
    d = tmp_path / "in.dict"
    d.write_text("ab\t0.9\t0.5\t1.0\t1.0\ta b\nbad\t1.0\t0.0\t1.0\t1.0\tx\na\ta\n", encoding="utf-8")
    out = tmp_path / "out.dict"
    report = m.compatible_dictionary(d, acoustic, out)
    assert report["accepted_pronunciations"] == 2
    assert report["rejected_pronunciations"] == 1
    assert m.dictionary_words(out) == {"ab", "a"}
