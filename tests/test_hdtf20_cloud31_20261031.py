import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('hdtf_cloud', Path(__file__).parents[1] / 'scripts/experiments/hdtf20_cloud31_20261031.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def test_requested_model_and_asr_response_contract():
    assert cloud.ASR == 'qwen-audio-3.1-asr-flash'
    assert cloud.TTS == 'qwen-audio-3.1-tts-plus'
    assert cloud.extract_text({'output': {'text': ' Hello  world. '}}) == 'Hello world.'
    assert cloud.extract_text({'output': {'output': {'text': 'English.'}}}) == 'English.'
    with pytest.raises(ValueError):
        cloud.extract_text({'output': {'text': ''}})


def test_qc_distinguishes_added_or_missing_words():
    assert cloud.word_error('Hello, WORLD!', 'hello world') == 0
    assert cloud.word_error('one two three four', 'one three four') == .25
    assert cloud.word_error('one two', 'one two three') == .5
