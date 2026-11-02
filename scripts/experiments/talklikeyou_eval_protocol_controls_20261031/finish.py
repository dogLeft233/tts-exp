"""Wait for the existing producer, then audit and archive its complete run."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

B = Path('/home/data2/wangjiajun/av-evaluation')
RUN = B/'jobs/talklikeyou_syncnet_protocol_controls_20261031_attempt01'
PYTHON = B/'envs/eval-base/bin/python'
PRODUCER_PID = int(os.environ.get('PRODUCER_PID', '11201'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def main():
    os.sched_setaffinity(0, {0, 1, 2, 3})
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    assert not (RUN/'terminal_receipt.json').exists()
    source = RUN/'postprocess_sources_revision02'
    source.mkdir(exist_ok=True)
    paths = [B/'talklikeyou_protocol_analyze_20261031_revision02.py', B/'talklikeyou_protocol_audit_20261031_revision02.py', Path(__file__)]
    pins = {}
    for p in paths:
        shutil.copy2(p, source/p.name)
        pins[str(p)] = sha(p)
    write(RUN/'postprocess_source_pins_revision02.json', pins)
    deadline = time.time()+12*3600
    while json.loads((RUN/'status.json').read_text())['status'] != 'FORWARDS_COMPLETE':
        if time.time() > deadline:
            raise RuntimeError('Producer wait exceeded12h')
        if not Path(f'/proc/{PRODUCER_PID}/stat').exists():
            raise RuntimeError('Original producer exited before completion')
        time.sleep(10)
    for label, script in [('analysis', paths[0]), ('independent_audit', paths[1])]:
        assert sha(script) == pins[str(script)]
        with (RUN/(label+'.log')).open('w') as output:
            result = subprocess.run([str(PYTHON), str(script), '--run', str(RUN)],
                stdout=output, stderr=subprocess.STDOUT)
        write(RUN/(label+'_exit.json'), dict(returncode=result.returncode, ended=time.time()))
        assert result.returncode == 0, label+' failed'
    protocol = json.loads((RUN/'protocol.json').read_text())
    inputs = RUN/'source_inputs'
    inputs.mkdir(exist_ok=True)
    for row in protocol['unique_rows']:
        for suffix, origin in [('.mp4', row['original_mp4']), ('.wav', row['audio_path'])]:
            dest = inputs/(row['id']+suffix)
            shutil.copy2(origin, dest)
            assert sha(dest) == sha(origin)
    science = RUN/'upstream_sources'
    shutil.copytree(B/'scorers/syncnet', science, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pth', '*.model', '*.avi'), dirs_exist_ok=True)
    stages = [json.loads(p.read_text()) for p in (RUN/'stages').glob('*.json')]
    assert all(p['returncode'] == 0 for p in stages)
    write(RUN/'terminal_receipt.json', dict(status='COMPLETE', producer_pid=PRODUCER_PID,
        unique_clips=41, scientific_stage_exit0=len(stages), analysis_exit0=True, independent_audit_exit0=True,
        recovery='Initial producer11201 completed27 items then stopped at resource gate before item28 scoring. Revision02 waits for resources and reuses validated completed pipeline stages. Auxiliary duplicate launch12360 was refused before forward. Original logs and failures preserved.', ended=time.time()))
    files = []
    for p in sorted(RUN.rglob('*')):
        if p.is_file():
            files.append(dict(path=str(p.relative_to(RUN)), bytes=p.stat().st_size, sha256=sha(p)))
    write(RUN/'artifact_manifest.json', dict(files=files, count=len(files), bytes=sum(p['bytes'] for p in files)))
    archive = RUN.with_suffix('.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        tar.add(RUN, arcname=RUN.name)
    receipt = dict(status='ARCHIVED', archive=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), scientific_files=len(files), scientific_bytes=sum(p['bytes'] for p in files), ended=time.time())
    write(RUN.with_suffix('.archive_receipt.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(RUN/'finish_failure.json', dict(error=repr(error), time=time.time()))
        raise
