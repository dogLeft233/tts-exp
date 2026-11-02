"""One-shot fresh entry qualification, F persistence, then independent formal repeat."""
import importlib.util,json,os,sys,tarfile,time,traceback
from pathlib import Path
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1')
def main():
    sys.path.insert(0,str(OUT/'sources'))
    import patrick_rms_repeat_controller_v1 as c
    t=c.t
    assert not (OUT/'receipt.json').exists() and not (OUT/'bootstrap_receipt.json').exists()
    contract=json.loads((OUT/'contract.json').read_text(encoding='utf-8'))
    for n,s in contract['sources'].items():assert t.sha(OUT/'sources'/n)==s
    def save(stage,**extra):
        t.atomic_json(OUT/'bootstrap_receipt.json',dict(pid=os.getpid(),stage=stage,version=c.VERSION,**extra))
    job='jobs/qualification_entry_attempt01'
    command=['flock','-n',t.REMOTE+'/preparation_GPU.lock','bash',t.REMOTE+'/runtime_exec.sh',t.RUN+'/sources/qualify_patrick_rms_repeat_entry_v1.py']
    code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import start_job;print(json.dumps(start_job(Path(%r),%r)))'%(t.REMOTE,t.RUN+'/'+job,command)
    result=json.loads(t.remote_python(code));save('fresh_GPU_entry_qualification',remote_job=result)
    while result['state'] in ('starting','running'):
        time.sleep(10);result=c.status(job);save('fresh_GPU_entry_qualification',remote_job=result)
    assert result['state']=='completed','lost entry job; no replay'
    t.persist(job)
    exists=json.loads(t.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))'%(t.RUN+'/qualification')))
    if exists:t.persist('qualification')
    assert result['returncode']==0,'entry qualification failed; no formal training'
    with tarfile.open(OUT/'qualification/artifacts.tar.gz','r:gz') as stream:
        raw=stream.extractfile('result.json').read()
    q=json.loads(raw)
    assert q['passed'] and q['contract_sha256']==t.sha(OUT/'contract.json') and q['total_attempts']==6
    assert q['initial_unchanged'] and q['actual_sampling_equal']
    (OUT/'qualification/result_source_exact.json').write_bytes(raw)
    (OUT/'qualification/result.json').write_bytes(raw)
    save('entry_passed_starting_formal',qualification=q)
    c.run()
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'bootstrap_failure.json').write_text(json.dumps(dict(pid=os.getpid(),traceback=traceback.format_exc(),no_automatic_restart=True),indent=2),encoding='utf-8')
        raise
