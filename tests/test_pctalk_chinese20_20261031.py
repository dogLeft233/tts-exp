"""Check the actual upstream window-count contract and bounded stage scope."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'scripts/experiments/pctalk_chinese20_20261031'
spec=importlib.util.spec_from_file_location('pctalk_prepare',S/'prepare.py')
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

def test_native_frame_formula_matches_official_float_expression():
    for samples in range(16000,160000,23):
        expected=max(1,int(max(1,1+samples//200-16)/80*25))
        assert prepare.native_frames(samples)==expected

def test_only_syncnet_stage_and_no_hdtf_resume():
    controller=(S/'controller.py').read_text()
    assert "stages=['syncnet']" in controller
    assert "elif stage==" not in controller
    for path in S.glob('*.py'):
        assert 'hdtf20' not in path.read_text()
