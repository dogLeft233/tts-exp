"""Audited continuation after a host-only POSIX checkpoint path failure."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

OUT = Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1')
REV = OUT / 'controller_revision02_posix_handoff_v1'
RUN = '/dev/shm/synctalk-multispeaker-data-20261017/trials/patrick_ave_rms_coarse_v1'
ORDER = [(epoch, gain) for epoch in range(1, 11) for gain in (0, -6, 6)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def remote_checkpoint(path):
    assert '\\' not in path, 'Remote paths must retain POSIX separators'
    p = PurePosixPath(path)
    root = PurePosixPath(RUN) / 'training'
    assert p.is_absolute() and '..' not in p.parts
    p.relative_to(root)
    assert p.name.startswith('ngp_ep') and p.suffix == '.pth'
    return p

def validate_prefix(rows):
    keys = [(r['epoch'], r['gain_db']) for r in rows]
    assert keys and keys == ORDER[:len(keys)], 'Only an exact completed chronological prefix may continue'
    assert len(set(keys)) == len(keys)
    for r in rows:
        assert r['passed'] and r['global_step'] == 6000 * r['epoch']
        assert r['epoch_attempts'] == 6000 and not r['old_fine_resumed']
        assert r['epoch_successful_updates'] + r['epoch_amp_skips'] == 6000
        remote_checkpoint(r['checkpoint_path'])
    return keys

def load_parent():
    sys.path.insert(0, str(OUT / 'sources'))
    spec = importlib.util.spec_from_file_location('rms_original_frozen_controller', OUT / 'sources/patrick_rms_controller_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def rotate(t, prior, current):
    if prior is None:
        return
    old = remote_checkpoint(prior['checkpoint_path'])
    new = remote_checkpoint(current['checkpoint_path'])
    assert old != new and prior['epoch'] < current['epoch']
    assert prior['gain_db'] == current['gain_db'] and prior['epoch'] < 10
    local_old = OUT / old.relative_to(PurePosixPath(RUN)).as_posix()
    local_new = OUT / new.relative_to(PurePosixPath(RUN)).as_posix()
    assert sha(local_old) == prior['checkpoint_sha256'] and sha(local_new) == current['checkpoint_sha256']
    assert json.loads((local_new.parent.parent.parent / 'verification.json').read_text(encoding='utf-8'))['passed']
    # This version only fixes host serialization; identical lease and SHA gates remain.
    code = '''import fcntl,os,hashlib,json
from pathlib import Path
old=Path(%r);new=Path(%r);root=Path(%r)
assert old.resolve().is_relative_to(root/'training') and new.resolve().is_relative_to(root/'training')
assert hashlib.sha256(old.read_bytes()).hexdigest()==%r and hashlib.sha256(new.read_bytes()).hexdigest()==%r
fd=os.open(old,os.O_RDWR)
try:
 fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);old.unlink()
finally:
 fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
print(json.dumps(dict(passed=True)))
''' % (old.as_posix(), new.as_posix(), RUN, prior['checkpoint_sha256'], current['checkpoint_sha256'])
    intent = dict(old_sha256=prior['checkpoint_sha256'], superseded_by=current['checkpoint_sha256'], path_serialization='PurePosixPath.as_posix', revision='controller_revision02_posix_handoff_v1')
    t.atomic_json(local_old.parent.parent.parent / 'checkpoint_retirement_revision02_intent.json', intent)
    assert json.loads(t.remote_python(code))['passed']
    local_old.unlink()
    t.atomic_json(local_old.parent.parent.parent / 'checkpoint_cache_retirement.json', dict(intent, passed=True))

def run():
    parent = load_parent()
    t = parent.t
    handoff = json.loads((REV / 'handoff.json').read_text(encoding='utf-8'))
    assert handoff['passed'] and handoff['failure_kind'] == 'host_only_posix_path_serialization'
    assert handoff['source_sha256'] == sha(Path(__file__))
    assert sha(OUT / 'contract.json') == handoff['training_contract_sha256']
    assert sha(OUT / 'controller_failure.json') == handoff['original_failure_sha256']
    assert json.loads((REV / 'qualification.json').read_text(encoding='utf-8'))['passed']
    audit_path = REV / 'audit.json'
    assert sha(audit_path) == handoff['audit_sha256']
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['passed'] and audit['no_training_process_active'] and audit['no_future_jobs']
    assert len(audit['completed']) == 4
    contract = json.loads((OUT / 'contract.json').read_text(encoding='utf-8'))
    def guard():
        assert sha(OUT / 'contract.json') == handoff['training_contract_sha256']
        for name, digest in contract['sources'].items():
            assert sha(OUT / 'sources' / name) == digest
        assert not (OUT / 'pause_request.json').exists()
    guard()
    rows = []
    for item in audit['completed']:
        path = OUT / item['receipt_relative']
        assert sha(path) == item['receipt_sha256']
        r = json.loads(path.read_text(encoding='utf-8'))
        r['_receipt'] = RUN + '/' + item['receipt_relative']
        rows.append(r)
    keys = validate_prefix(rows)
    with t.exclusive_lock(OUT / 'controller.lock'):
        assert not (REV / 'receipt.json').exists(), 'No automatic restart'
        state = dict(pid=os.getpid(), status='audited_continuation', version=parent.VERSION, controller_revision='controller_revision02_posix_handoff_v1', handoff_sha256=sha(REV / 'handoff.json'), completed=[dict(epoch=r['epoch'], gain_db=r['gain_db'], receipt=audit['completed'][i]['receipt_relative']) for i, r in enumerate(rows)], current=None, failures=[], old_fine_resumed=False)
        def save():
            t.atomic_json(REV / 'receipt.json', state)
        save()
        previous = {gain: None for gain in (0, -6, 6)}
        epochs = []
        for epoch in range(1, 11):
            receipts = {}
            traces = {}
            reference = None
            for gain in (0, -6, 6):
                guard()
                rel = f'training/PatrickLeahy/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01'
                if (epoch, gain) in keys:
                    r = rows[keys.index((epoch, gain))]
                else:
                    job = rel.replace('training/', 'jobs/', 1)
                    assert not (OUT / rel).exists() and not (OUT / job).exists()
                    exists = json.loads(t.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists() or Path(%r).exists()))' % (RUN + '/' + rel, RUN + '/' + job)))
                    assert not exists, 'Any existing job requires audit, never replay'
                    actual_rel, actual_job, result = parent.launch(gain, epoch, previous[gain], reference)
                    assert (actual_rel, actual_job) == (rel, job)
                    state.update(status='training', current=dict(epoch=epoch, gain_db=gain, relative=rel, job_relative=job), remote_job=result)
                    save()
                    while result['state'] in ('starting', 'running'):
                        time.sleep(30)
                        result = parent.status(job)
                        state['remote_job'] = result
                        save()
                    assert result['state'] == 'completed', 'Lost job: stop without replay'
                    t.persist(job)
                    exists = json.loads(t.remote_python('import json;from pathlib import Path;print(json.dumps(Path(%r).exists()))' % (RUN + '/' + rel)))
                    if exists:
                        t.persist(rel)
                    if result['returncode'] != 0:
                        state.update(status='terminal_failure_preserved_no_new_jobs')
                        state['failures'].append(dict(epoch=epoch, gain_db=gain, execution=result))
                        save()
                        raise RuntimeError('Training terminal; artifacts preserved, no new jobs')
                    r = json.loads((OUT / rel / 'epoch_receipt.json').read_text(encoding='utf-8'))
                    assert r['passed'] and r['normalization_contract_sha256'] == handoff['training_contract_sha256']
                    r['_receipt'] = RUN + '/' + rel + '/epoch_receipt.json'
                    rotate(t, previous[gain], r)
                    state['completed'].append(dict(epoch=epoch, gain_db=gain, receipt=rel + '/epoch_receipt.json'))
                    save()
                previous[gain] = r
                receipts[gain] = r
                traces[gain] = OUT / rel / 'attempts.jsonl'
                if gain == 0:
                    reference = RUN + '/' + rel + '/attempts.jsonl'
            block = t.qualify_epoch(receipts, traces)
            lr = [[json.loads(line)['optimizer_lr'] for line in Path(traces[gain]).read_text(encoding='utf-8').splitlines()] for gain in (0, -6, 6)]
            assert lr[0] == lr[1] == lr[2]
            block['actual_lr_equal'] = True
            epochs.append(block)
            block_path = OUT / f'blocks/coarse_ep{epoch:04d}.json'
            if block_path.exists():
                assert json.loads(block_path.read_text(encoding='utf-8')) == block
            else:
                t.atomic_json(block_path, block)
        phase = t.qualify_phase(epochs, previous)
        t.atomic_json(OUT / 'blocks/coarse_qualification.json', phase)
        ready = dict(passed=True, identity='PatrickLeahy', version=parent.VERSION, phase=phase, final_receipts={str(g): r for g, r in previous.items()}, contract_sha256=handoff['training_contract_sha256'], no_training_job_active=True, normalization=contract['normalization'], new_27_grid_required=True, controller_handoff_sha256=sha(REV / 'handoff.json'))
        t.atomic_json(OUT / 'coarse_ready_for_27_grid.json', ready)
        t.publish('coarse_ready_for_27_grid.json', ready)
        state.update(status='coarse10_complete_ready_for_new27', current=None, phase=phase)
        save()

if __name__ == '__main__':
    try:
        run()
    except BaseException:
        REV.mkdir(exist_ok=True)
        (REV / 'controller_failure.json').write_text(json.dumps(dict(pid=os.getpid(), status='stopped_no_automatic_replay', traceback=traceback.format_exc()), indent=2), encoding='utf-8')
        raise
