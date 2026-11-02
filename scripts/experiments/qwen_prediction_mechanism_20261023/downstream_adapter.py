"""Only adds outer experiment-budget guards to the unchanged official runner."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen17_level_comparison10_20261023 import downstream as original
from scripts.experiments.qwen_prediction_mechanism_20261023.budget import guard

run=Path(sys.argv[sys.argv.index('--run')+1]).resolve().parent
command=original.checked_stage_command
event=original.Runner.event
def checked(*args,**kwargs):
    guard(run)
    return command(*args,**kwargs)
def observed(self,name,**kwargs):
    if name=='cell_started':guard(run)
    return event(self,name,**kwargs)
original.checked_stage_command=checked
original.Runner.event=observed
original.main()
