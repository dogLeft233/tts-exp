"""Sequential A control queue; stops on any nonzero child exit."""
from pathlib import Path
import subprocess,sys
src=Path(__file__).resolve().parent;run=Path(sys.argv[1]);py=sys.executable
jobs=[('A_L17','mechanism.py','L17'),('A_qualification_L06','qualify.py','L06'),('A_qualification_L17','qualify.py','L17')]
for label,script,model in jobs:
    subprocess.run([py,str(src/'execute.py'),str(run),label,py,str(src/script),'--model',model,'--run',str(run)],check=True)
