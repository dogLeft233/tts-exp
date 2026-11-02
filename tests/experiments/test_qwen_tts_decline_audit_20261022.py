import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

SPEC=importlib.util.spec_from_file_location('decline_audit',Path(__file__).parents[2]/'scripts/experiments/qwen_tts_decline_audit_20261022.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


@pytest.mark.parametrize('reference,hypothesis,distance',[
    ('','甲乙',2),('甲乙','',2),('高周转就是牺牲利润','高周转就是牺牲利润',0),
    ('九龙仓成为股东','九龙成为大股东',2),('甲','甲甲甲甲',3),
])
def test_cer_edit_distance_keeps_deletions_and_repeated_hallucinations(reference,hypothesis,distance):
    assert M.edit_distance(reference,hypothesis)==distance


def test_normalization_preserves_content_but_removes_formatting():
    cc=SimpleNamespace(convert=lambda s:s.replace('萬','万'))
    assert M.normalize(' ＡＢＣ，\n萬！ ',cc)=='abc万'
    assert M.normalize('十二',cc)!=M.normalize('12',cc)
