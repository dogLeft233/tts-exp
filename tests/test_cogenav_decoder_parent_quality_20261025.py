import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('parent_quality', Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_parent_quality_20261025.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_true_levenshtein_handles_substitution_deletion_and_insertion():
    assert module.edit_distance('今天很好','今天真好') == 1
    assert module.edit_distance('今天很好','今天') == 2
    assert module.edit_distance('今天很好','今天天气很好') == 2
    assert module.edit_distance('','abc') == 3


def test_text_normalization_preserves_numbers_and_han():
    assert module.normalized(' 今天，１２３！ A ') == '今天123a'
