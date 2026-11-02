import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

spec=importlib.util.spec_from_file_location('decoder_job',Path(__file__).parents[1]/'scripts/experiments/cogenav_decoder_pilot_20261025/run_job.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_worker_failure_exit_code_is_preserved():
    with tempfile.TemporaryDirectory() as directory:
        result=module.run_phase('fake',directory,[sys.executable,'-c','import sys; print("intentional failure");sys.exit(3)'],timeout=10)
        assert result['status']=='CRASHED' and result['exit_code']==3
        saved=json.loads((Path(directory)/'fake_exit.json').read_text())
        assert saved['pid']>0 and saved['exit_code']==3
        assert 'intentional failure' in (Path(directory)/'fake.log').read_text()
