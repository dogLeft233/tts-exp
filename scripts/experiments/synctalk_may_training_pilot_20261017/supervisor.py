"""One locked sequential worker; never relaunch an existing executed arm."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, save_json, sha

lock=(ROOT/'supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'supervisor_receipt.json').exists(), 'existing execution identity must not be replaced'
identity={'pid':os.getpid(),'proc_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
          'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'protocol_sha256':sha(ROOT/'protocol.json'),
          'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'execution_order_db':[0,-6,6]}
save_json(ROOT/'supervisor_receipt.json',identity)
try:
    for gain in identity['execution_order_db']:
        subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),'--gain',str(gain)],check=True)
    save_json(ROOT/'training_complete.json',{'passed':True,'training_arms':3,'steps_each':60000})
    if (ROOT/'postprocess.py').exists():
        subprocess.run([sys.executable,'-u',str(ROOT/'postprocess.py')],check=True)
    else:
        save_json(ROOT/'status.json',{'phase':'training_complete_awaiting_postprocess','training_arms':3})
except BaseException as e:
    save_json(ROOT/'supervisor_failure.json',{'error':repr(e),'type':type(e).__name__,
                                             'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    raise
