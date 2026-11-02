"""Independent CPU follow-ups, each with durable child receipts."""
from pathlib import Path
import subprocess,sys
src=Path(__file__).resolve().parent;run=Path(sys.argv[1]);py='/home/wjj/.venvs/qwen3-tts-1p7b/bin/python';sync='/home/wjj/.venvs/syncnet/bin/python'
jobs=[('B_CPU_audit',py,'audit_bank.py'),('B_source_summary',py,'summarize_bank.py'),('B_score_description',py,'describe_scores.py'),('B_plot',sync,'plot_bank.py')]
children=[]
for name,python,script in jobs:
    children.append((name,subprocess.Popen([py,str(src/'execute.py'),str(run),name,python,str(src/script),'--run',str(run)])))
failed=[]
for name,p in children:
    if p.wait()!=0:failed.append(name)
if failed:raise RuntimeError('CPU follow-up failed: '+','.join(failed))
