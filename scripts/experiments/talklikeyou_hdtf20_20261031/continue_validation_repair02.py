"""Continue from a successful first model forward after fixing validation only."""
import json,os,fcntl,time
from prepare import B,D,RUN,GPU,sha,write
from orchestrate import stage

def main():
    lock=(RUN/'orchestrator_repair02.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((RUN/'generation/WDA_DickDurbin_000_seg1_N/worker_result.json').read_text());assert previous['status']=='PASS'
    write(RUN/'validation_resolution_repair02.json',dict(status='CONTINUING',reason='Old fixed512 assertion was invalid for native816 source composition',
        preserved_worker_sha256=sha(RUN/'generation/WDA_DickDurbin_000_seg1_N/worker_result.json'),first_forward_rerun=False,
        first_gpu_uuid='GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1',remaining_gpu_uuid=GPU,
        gpu_change_reason='Another job occupied original GPU after initial selection; remaining job moved to idle A6000, no foreign process changed',time=time.time()))
    p=B/'envs/eval-base/bin/python'
    stage('generation_smoke_repair02',D/'env/bin/python','generate.py','--smoke')
    stage('generation',D/'env/bin/python','generate.py')
    stage('scoring',p,'controller.py')
    stage('audit',p,'audit_precision_replay.py','--run',RUN)
    stage('plot',p,'plot.py')
    stage('package',p,'package.py')
    write(RUN/'orchestrator_status.json',dict(status='COMPLETE',stage='DONE',pid=os.getpid(),updated=time.time(),continuation='validation_repair02'))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'orchestrator_failure_repair02.json',dict(error=repr(e),time=time.time()));raise
