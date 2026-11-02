"""Verify native support boundaries, missing-pair policy and Holm family size."""
from pathlib import Path
import importlib.util

S=Path(__file__).resolve().parents[1]/'scripts/experiments/pctalk_chinese20_nine_20261031'
def load(name):
    spec=importlib.util.spec_from_file_location('pctalk_nine_'+name,S/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_exact_classifier_frame_boundary_and_complete_sources():
    rows=[dict(id=f'a{i}',source='a',samples=80000,actual_video_frames=125) for i in range(4)]
    rows.append(dict(id='b',source='b',samples=79999,actual_video_frames=125))
    value=load('prepare').support(rows)
    assert value['classifier_eligible_media']==4 and value['classifier_complete_sources']==['a']

def test_missing_tts_draw_excludes_entire_source():
    rows=[dict(id='n',source='a',speaker='s',arm='N',strict_reference=True)]
    rows += [dict(id=f't{i}',source='a',speaker='s',arm='T',strict_reference=True) for i in range(3)]
    pairs,missing=load('analyze').make_pairs(rows,dict(n=1.,t0=2.,t1=3.))
    assert not pairs and missing[0]['unmeasurable_ids']==['t2']

def test_nine_hypothesis_family_and_no_new_syncnet():
    a=load('analyze');assert len(a.METRICS)==9
    assert abs(a.holm([.001]+[1.]*8)[0]-.009)<1e-15
    source=(S/'controller.py').read_text()
    assert "stages=['cogenav'" in source and "stage=='syncnet'" not in source

def test_resume_adds_wrapper_gpu_binding_without_changing_science():
    original=(S/'controller.py').read_text()
    fixed=(S/'controller_launch02.py').read_text()
    assert fixed==original.replace('env=dict(os.environ,GPU_METRICS_PATH=',
                                  'env=dict(os.environ,GPU_UUID=GPU,GPU_METRICS_PATH=')
