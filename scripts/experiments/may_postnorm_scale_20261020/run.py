"""One autonomous, bounded workflow; first true failure prevents every later job."""
from pathlib import Path
import json,sys,os,time,subprocess,traceback,shutil
from protocol import OUT,REMOTE,sha,write
from host import run_job,copy_tree,SSH
def cpu(script,*args):
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).parent/script),*args],check=True)
def main():
    assert not (OUT/'workflow_launcher.json').exists(),'never restart an existing workflow'
    assert shutil.disk_usage(OUT).free>3*1024**3
    write(OUT/'workflow_launcher.json',dict(host_pid=os.getpid(),source_sha256=sha(__file__),started_unix=time.time(),no_automatic_restart=True))
    p=json.loads((OUT/'protocol.json').read_text());done=[]
    try:
        cpu('deploy.py');cpu('review_baselines.py')
        for arm in p['arms']:
            done.append(run_job('qualify_'+arm['id'],'train.py',['qualify',arm['id']],mode='qualification',arm=arm['id']))
        cpu('review_training.py','qualification');write(OUT/'qualification_complete.json',dict(passed=True,successes=4,skip=0,formal_warmstart=False))
        for arm in p['arms']:
            done.append(run_job('train_'+arm['id'],'train.py',['train',arm['id']],mode='training',arm=arm['id']))
        cpu('review_training.py','training');write(OUT/'training_complete.json',dict(passed=True,new_models=2,new_successful_updates=24000,jobs=done))
        for arm in p['arms']+p['imported_arms']:
            done.append(run_job('render_'+arm['id'],'evaluate.py',['render',arm['id']]))
            copy_tree('renders/'+arm['id'],OUT/'remote/renders'/arm['id'])
        done.append(run_job('score_all','evaluate.py',['score']));copy_tree('scores',OUT/'remote/scores')
        r=subprocess.run(SSH+['python3 '+REMOTE+'/review_media_clock.py '+REMOTE],capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace')
        b=subprocess.check_output(SSH+['cat '+REMOTE+'/media_clock_review.json']);assert json.loads(b)['passed'];(OUT/'media_clock_review.json').write_bytes(b)
        write(OUT/'postprocess_complete.json',dict(passed=True,new_cells=10,new_matrices=20,imported_cells=2,imported_matrices=4,jobs=done))
        cpu('analysis.py');cpu('report.py')
        write(OUT/'workflow_receipt.json',dict(status='numerical_delivery_ready_pending_visual_and_memory',passed=True,host_pid=os.getpid(),jobs=done))
        write(OUT/'active_controller.json',dict(status='numerical_delivery_ready_pending_visual_and_memory',host_pid=os.getpid(),all_scientific_jobs_complete=True))
    except BaseException as e:
        write(OUT/'workflow_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_jobs=done,new_jobs_stopped=True,host_pid=os.getpid()))
        raise
if __name__=='__main__':main()
