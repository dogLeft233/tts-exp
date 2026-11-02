"""The deferred job must never start on an incomplete or failed PC-Talk run."""
import importlib.util,json,sys
from pathlib import Path
import pytest
import subprocess

ROOT=Path(__file__).resolve().parents[1]/'scripts/experiments/dystream_lips_full104_20261031'

def test_worker_directory_does_not_shadow_standard_library_queue():
    result=subprocess.run([sys.executable,'-c','from queue import Queue; assert Queue().empty()'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_native_moviepy_export_can_repeat_one_tail_frame():
    e=load('native_export')
    assert e.exported_frames(249)==250
    assert e.exported_frames(295)==295
    assert all(n<=e.exported_frames(n)<=n+1 for n in range(125,334))

def load(name):
    spec=importlib.util.spec_from_file_location('dystream_full_'+name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        if name!='prepare':mp.setitem(sys.modules,'prepare',load('prepare'))
        spec.loader.exec_module(module)
    return module

def acceptance(root):
    (root/'orchestrator_complete').touch()
    for name in ['independent_audit.json','independent_statistics_audit.json','delivery_receipt.json','syncnet_geometry_audit.json']:
        (root/name).write_text(json.dumps(dict(status='PASS')))

def test_queue_waits_for_complete_marker_and_all_audits(tmp_path):
    q=load('wait_for_pctalk')
    assert q.dependency_state(tmp_path)=='WAITING_PARENT'
    (tmp_path/'orchestrator_complete').touch()
    assert q.dependency_state(tmp_path)=='PARENT_ACCEPTANCE_MISSING'
    acceptance(tmp_path)
    assert q.dependency_state(tmp_path)=='READY_FOR_TERMINAL_CHECK'

@pytest.mark.parametrize('failure',['orchestrator','audit'])
def test_failure_overrides_even_a_complete_marker(tmp_path,failure):
    q=load('wait_for_pctalk');acceptance(tmp_path)
    if failure=='orchestrator':(tmp_path/'orchestrator_failed').touch()
    else:(tmp_path/'independent_audit.json').write_text(json.dumps(dict(status='FAIL')))
    assert q.dependency_state(tmp_path)==('PARENT_FAILED' if failure=='orchestrator' else 'PARENT_ACCEPTANCE_FAILED')

@pytest.mark.parametrize('samples',[49936,640*154,640*154+639,640*332])
def test_native_frame_count_matches_official_sliding_window(samples):
    p=load('prepare')
    # Independent enumeration of official inference sliding windows; the native
    # prefix is removed after concatenating one output frame per valid window.
    for prefix in [8,14,30]:
        total=(samples+prefix*640)//640;window=prefix+2
        emitted=sum(1 for start in range(total) if min(start+window,total)-start==window)
        assert p.native_frames(samples)==emitted
    assert 640<=samples-p.native_frames(samples)*640<1280
