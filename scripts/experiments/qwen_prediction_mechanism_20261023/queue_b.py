"""Frozen B queue; preserves explicit stage exits and stops on failure."""
from pathlib import Path
import subprocess,sys
src=Path(__file__).resolve().parent;run=Path(sys.argv[1]);py=sys.executable
for name in ['L06','L17']:
    subprocess.run([py,str(src/'execute.py'),str(run),'B_'+name,py,str(src/'bank.py'),'run','--model',name,'--run',str(run)],check=True)
subprocess.run([py,str(src/'execute.py'),str(run),'B_analysis',py,str(src/'analyze_bank.py'),'--run',str(run)],check=True)
