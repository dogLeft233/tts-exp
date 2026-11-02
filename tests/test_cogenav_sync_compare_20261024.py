"""Natural reuse must not inflate independent source counts."""
import importlib.util
from pathlib import Path
import sys

scripts = Path(__file__).parents[1] / "scripts/experiments"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cogenav_compare", scripts / "cogenav_sync_compare_20261024.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_source_grouping_and_paired_draws():
    rows = []
    for source in ["s1", "s2"]:
        for arm, draws, value in [("N", [0], 0.), ("L06", [1, 2, 3], 1.), ("L17", [1, 2, 3], 2.)]:
            for draw in draws:
                rows.append(dict(id=f"{source}_{arm}_{draw}", sample_id=source, arm=arm, draw=draw,
                                 qualified=True, score_float32=value, eq1=-value,
                                 zero_common=-value, max_score=-value, zero_excess=-value, peak_excess=-value))
    pairs, groups, families = m.compare(rows)
    assert len(pairs) == 18 and len(groups) == 6
    for family in families.values():
        assert family["sources"] == 2 and family["pairs"] == 6
        assert family["eq1_sign_conflicts"] == 6
        assert family["source_mean_eq1_sign_conflicts"] == 2


def test_missing_measurement_does_not_create_a_pair():
    rows=[]
    for arm,draws in [('N',[0]),('L06',[1,2]),('L17',[1,2,3])]:
        for draw in draws:
            rows.append(dict(id=f'{arm}_{draw}',sample_id='s1',arm=arm,draw=draw,qualified=True,
                             **{f:1. for f in ['score_float32','eq1','zero_common','max_score','zero_excess','peak_excess']}))
    pairs,groups,families=m.compare(rows,allow_missing=True)
    assert len(pairs)==7
    assert families['L06-N']['pairs']==2
    assert families['L17-N']['pairs']==3
    assert families['L17-L06']['pairs']==2
