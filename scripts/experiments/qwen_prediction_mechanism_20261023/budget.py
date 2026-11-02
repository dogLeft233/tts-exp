"""Conservative serial-GPU wall upper bound, with a full-unit reserve."""
from pathlib import Path
import json,time,shutil

class BudgetStop(RuntimeError):pass

def guard(run,reserve=900):
    run=Path(run);first=run.parent/'qwen_prediction_mechanism_20261023_attempt01/A_L06/started.json'
    start=json.loads(first.read_text())['unix'];elapsed=time.time()-start
    # Counting the entire serial experiment wall clock as GPU time is an upper
    # bound. It cannot understate GPU use, and deliberately includes loading/CPU.
    if elapsed+reserve>=8*3600:raise BudgetStop('conservative total GPU-time upper bound near8h; no new unit')
    if elapsed+reserve>=12*3600:raise BudgetStop('total wall budget near12h; no new unit')
    if shutil.disk_usage(run).free<12*1024**3:raise BudgetStop('persistent disk free below12GiB; no new unit')
    return {'elapsed_wall_s':elapsed,'conservative_GPU_time_upper_bound_s':elapsed,'reserve_s':reserve}
