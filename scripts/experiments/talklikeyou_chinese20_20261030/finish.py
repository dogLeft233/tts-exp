"""Run verification and packaging after all science stages have actually exited."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

from prepare import B, RUN, sha, write


def main():
    lock = (RUN/'finish.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    source = Path(__file__).parent
    while not (RUN/'stages/analysis.json').exists():
        for failure in ['failure.json', 'generation_failure.json']:
            assert not (RUN/failure).exists(), (RUN/failure).read_text() if (RUN/failure).exists() else ''
        time.sleep(20)
    assert json.loads((RUN/'stages/analysis.json').read_text())['returncode'] == 0
    stages = [
        ('audit', B/'envs/eval-base/bin/python', 'audit.py', ['--run', str(RUN)]),
        ('peavs_batch20', B/'envs/peavs/bin/python', 'peavs_batch_sensitivity.py', []),
        ('compare_models', B/'envs/eval-base/bin/python', 'compare_models.py', []),
        ('plot', B/'envs/eval-base/bin/python', 'plot.py', ['--run', str(RUN)]),
        ('package', B/'envs/eval-base/bin/python', 'package.py', []),
    ]
    for name, interpreter, script, arguments in stages:
        record = RUN/'stages'/('finish_'+name+'.json')
        assert not record.exists(), 'Do not replay finished postprocessing without investigation'
        write(RUN/'finish_status.json', dict(status='RUNNING', stage=name, updated=time.time()))
        command = [str(interpreter), '-u', str(source/script), *arguments]
        log = RUN/'logs'/('finish_'+name+'.log')
        with log.open('w') as output:
            code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
        write(record, dict(returncode=code, command=command, log_sha256=sha(log), ended=time.time()))
        assert code == 0, name+' failed; inspect '+str(log)
    write(RUN/'finish_status.json', dict(status='COMPLETE', updated=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(RUN/'finish_failure.json', dict(error=repr(error), time=time.time()))
        raise
