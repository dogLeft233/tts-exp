"""Observe the single pinned in-flight job, persist its epoch, then stop scheduling."""
import importlib.util,json,os,sys,time,traceback,hashlib,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1')
REV=OUT/'controller_revision04_pause_for_patrick_fine_v1'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('frozen_trial_pause',OUT/'sources/other_identities_trial_v1.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def verify_closed(relative):
    d=OUT/relative;v=json.loads((d/'verification.json').read_text())
    assert v['passed'] and v['all_archived_member_bytes_verified']
    assert t.sha(d/'manifest.json')==v['manifest_sha256'] and t.sha(d/'artifacts.tar.gz')==v['archive_sha256']
    rows={r['path']:r for r in json.loads((d/'manifest.json').read_text())}
    seen=set()
    with tarfile.open(d/'artifacts.tar.gz','r|gz') as stream:
        for m in stream:
            assert m.isfile() and m.name in rows and m.name not in seen
            b=stream.extractfile(m).read();r=rows[m.name]
            assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];seen.add(m.name)
    for n,r in rows.items():
        if n not in seen:
            assert 'workspace/checkpoints/' in n
            assert (d/n).stat().st_size==r['bytes'] and t.sha(d/n)==r['sha256'];seen.add(n)
    assert seen==set(rows)
    return v

def run():
    request=json.loads((REV/'request.json').read_text())
    assert request['source_sha256']==t.sha(Path(__file__))
    assert request['identity']=='ChrisVanHollen' and request['epoch']==2 and request['gain_db']==-6
    with t.exclusive_lock(REV/'observer.lock'):
        assert not (REV/'paused.json').exists()
        state=dict(pid=os.getpid(),status='waiting_pinned_active_epoch',request=request,no_new_training_jobs=True)
        def save():t.atomic_json(REV/'receipt.json',state)
        save()
        while True:
            result=t.remote_action('status',request['identity'],gain=request['gain_db'],epoch=request['epoch'])
            assert result['pid']==request['remote_job_pid'] and result['command_sha256']==request['command_sha256']
            state['remote_job']=result;save()
            if result['state'] not in ['starting','running']:break
            time.sleep(15)
        assert result['state']=='completed','lost job is never replayed'
        for relative in [request['relative'].replace('training/','jobs/',1),request['relative']]:
            if not (OUT/relative/'verification.json').exists():t.persist(relative)
            verify_closed(relative)
        assert result['returncode']==0,'failed job preserved; no fine authorization from failed pause'
        latest={}
        for gain,epoch in [(0,2),(-6,2),(6,1)]:
            relative=f'training/ChrisVanHollen/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01'
            verify_closed(relative)
            r=json.loads((OUT/relative/'epoch_receipt.json').read_text());assert r['passed']
            latest[str(gain)]=dict(epoch=epoch,checkpoint_path=r['checkpoint_path'],checkpoint_sha256=r['checkpoint_sha256'],receipt_path=str(OUT/relative/'epoch_receipt.json'),successful_updates=r['cumulative_successful_updates'])
        code="from pathlib import Path;import json;print(json.dumps(Path(%r).exists()))"%(t.RUN+'/jobs/ChrisVanHollen/seed42/gain_+6/coarse/ep0002/attempt01/command.json')
        assert not json.loads(t.remote_python(code)),'unexpected future job; audit rather than stop GPU'
        state.update(status='paused_at_verified_epoch_boundary',passed=True,latest_by_gain=latest,no_optimizer_replay=True,original_trial_preserved=True)
        save();t.atomic_json(REV/'paused.json',state)

if __name__=='__main__':
    try:run()
    except BaseException:
        t.atomic_json(REV/'failure.json',dict(pid=os.getpid(),status='failed_no_automatic_replay',traceback=traceback.format_exc()));raise
