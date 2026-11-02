from pathlib import Path
import time,subprocess,sys
ROOT=Path(__file__).resolve().parent
log=Path('/dev/shm/synctalk-may-gain-inference-20261016/logs/build_import_attempt04.log')
deadline=time.time()+300
while time.time()<deadline:
 if log.exists() and 'ALL_ENCODER_IMPORT_PASS' in log.read_text():break
 time.sleep(3)
else:raise RuntimeError('build not qualified; no training')
sys.exit(subprocess.call(['bash',str(ROOT/'run_env.sh'),str(ROOT/'qualify.py')]))
