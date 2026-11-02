"""Runs frozen C only after B gate and pre-C content operational binding."""
from pathlib import Path
import subprocess,sys
src=Path(__file__).resolve().parent;run=Path(sys.argv[1]);py='/home/wjj/.venvs/qwen3-tts-1p7b/bin/python';sync='/home/wjj/.venvs/syncnet/bin/python';asr='/home/wjj/.venvs/tts-asr-audit-20261022/bin/python'
jobs=[('C_prepare',py,'confirmation.py',['prepare']),
      ('C_generate_L06',py,'confirmation.py',['generate','--model','L06']),
      ('C_generate_L17',py,'confirmation.py',['generate','--model','L17']),
      ('C_audio',sync,'canonical.py',[]),('C_ASR',asr,'content_asr.py',[]),
      ('C_downstream_prepare',sync,'downstream_plan.py',['prepare']),
      ('C_downstream_execute',sync,'downstream_plan.py',['execute']),
      ('C_analysis',py,'analyze_confirmation.py',[])]
for label,python,script,args in jobs:
    subprocess.run([py,str(src/'execute.py'),str(run),label,python,str(src/script),*args,'--run',str(run)],check=True)
