import importlib.util
from pathlib import Path


def module():
    p=Path(__file__).parents[1]/'scripts/experiments/echomimic_v3_official_control_20261030/control.py'
    spec=importlib.util.spec_from_file_location('official_control',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_selection_does_not_follow_scores_or_input_order():
    m=module();rows=[]
    for strict in (True,False):
        for i in range(8):
            for arm,draw in [('N',0),('T',1),('T',2),('T',3)]:
                rows.append(dict(source=f'{strict}_{i}',arm=arm,draw=draw,strict_reference=strict,samples=32000+i*640))
    selected=m.select_rows(list(reversed(rows)))
    assert {r['source'] for r in selected}=={f'{s}_{i}' for s in (True,False) for i in (0,3,7)}
    assert len(selected)==12 and all(r['draw'] in (0,1) for r in selected)


def test_shared_support_uses_shorter_audio_without_padding():
    m=module()
    rows=[dict(source='short',samples=40000),dict(source='short',samples=80000),
          dict(source='long',samples=96000),dict(source='long',samples=128000)]
    assert m.common_support(rows)=={'short':61,'long':81}
    assert m.native_frames(95984)==149


def test_scoring_repair_does_not_change_model_cli_or_audio():
    import sys
    m=module();sys.modules['control']=m
    p=Path(__file__).parents[1]/'scripts/experiments/echomimic_v3_official_control_20261030/score_repair.py'
    spec=importlib.util.spec_from_file_location('scoring_repair',p)
    repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
    clip=[str(m.B/'tools/bin/ffmpeg'),'-i','parent.mp4','-an','-frames:v','81','-c:v','copy','visual.mp4']
    fixed=repair.repaired_command(clip)
    assert fixed[fixed.index('-frames:v')+1]=='81' and fixed[fixed.index('-crf')+1]=='0'
    assert fixed[fixed.index('-bf')+1]=='0'
    scorer=['python','gpu_guard.py','run_syncnet.py','--reference','x']
    assert repair.repaired_command(scorer)==scorer
    audio=[str(m.B/'tools/bin/ffmpeg'),'-i','x.wav','-f','s16le','-']
    assert repair.repaired_command(audio)==audio
