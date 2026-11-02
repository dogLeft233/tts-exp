"""Sequential private-directory experiment controller; never manages foreign jobs."""
from pathlib import Path
import json, os, time, subprocess, sys, fcntl
from prepare import B, RUN, sha, write

S=Path(__file__).resolve().parent
GPU='GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def run(cmd, name, cwd=None):
    log=RUN/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        env=dict(os.environ,GPU_UUID=GPU,GPU_METRICS_PATH=str(RUN/'logs'/(name+'_gpu_metrics.json')),
            PEAVS_FEATURE_METRICS_PATH=str(RUN/'logs'/'peavs_features_gpu_metrics.json'))
        result=subprocess.run([str(x) for x in cmd],stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,env=env)
    write(RUN/'stages'/(name+'.json'),dict(command=list(map(str,cmd)),returncode=result.returncode,log_sha256=sha(log),ended=time.time()))
    if result.returncode:raise RuntimeError('Failed '+name+'; see '+str(log))

def resources(stage):
    while True:
        free,util=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(','))
        if free>=20000 and util<=85:return
        write(RUN/'status.json',dict(stage=stage,status='WAITING_GPU',free_mib=free,utilization=util,updated=time.time()))
        time.sleep(45)

def main():
    lock=(RUN/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (RUN/'generation_complete.json').exists():
        pid=int((RUN/'generate.pid').read_text());os.kill(pid,0)
        time.sleep(30)
    gen=json.loads((RUN/'generation_complete.json').read_text());assert gen['status']=='PASS' and gen['completed']==len(json.loads((RUN/'inputs.json').read_text())['rows'])
    generation_lock=(RUN/'experiment.lock').open('w')
    fcntl.flock(generation_lock,fcntl.LOCK_EX)
    rows=json.loads((RUN/'inputs.json').read_text())['rows'];seen={};unique=[];aliases={}
    for r in rows:
        r['video_path']=str(RUN/'videos'/(r['id']+'.mp4'));r['video_sha256']=sha(r['video_path']);r['actual_video_frames']=json.loads((RUN/'generation'/r['id']/'receipt.json').read_text())['frames']
        key=(r['audio_sha256'],r['video_sha256'])
        if key in seen:aliases[r['id']]=seen[key]
        else:seen[key]=r['id'];unique.append(r)
    write(RUN/'scoring_inputs.json',dict(rows=rows,unique_rows=unique,aliases=aliases,
        policy='Scores of exactly identical audio+video artifacts may be referenced for model-independent per-clip scorers; PEAVS lexically sorted full paired cohort batch is preserved'))
    envnames={'avs_u':'cogenav','synchformer':'synchformer','sparsesync':'synchformer','mtd':'eval-base','vocalist':'eval-base'}
    stages=['cogenav','avs_u','mtd','vocalist','synchformer','sparsesync','syncreward','peavs']
    for stage in stages:
        done=RUN/'stages'/(stage+'_complete.json')
        if done.exists():continue
        resources(stage);write(RUN/'status.json',dict(stage=stage,status='RUNNING',updated=time.time()))
        if stage=='cogenav':
            target=RUN/'cogenav';target.mkdir(exist_ok=True)
            for name in ['upstream','weights','autoavsr_detector.py','autoavsr_video_process.py','20words_mean_face.npy']:
                dest=target/name
                if not dest.exists():dest.symlink_to(B/'assets/cogenav'/name)
            manifest=dict(rows=[dict(r,sample_id=r['source'],qualified=False,score_float32=None,features={'duration_s':r['duration_s']}) for r in unique])
            write(target/'input_manifest.json',manifest)
            run(['flock','-n',B/'tmp/scorer.lock',B/'envs/cogenav/bin/python','-u',B/'gpu_guard.py',S/'cogenav_support_adapter.py','--run',target,'--manifest',target/'input_manifest.json','--sources','105','--record-long-clips'],stage)
        elif stage in envnames:
            run([B/'envs'/envnames[stage]/'bin/python',B/'gpu_guard.py',S/'batch_adapters.py',stage],stage)
        elif stage=='syncreward':
            # Unlike the two classifiers, official SyncReward explicitly supports
            # short native clips by repeating tail video and zero-padding segments.
            # Preserve that upstream inference definition; never pad input media.
            reused=[json.loads(line) for line in (RUN/'syncreward/reused_scores.jsonl').read_text().splitlines()]
            reused_ids={Path(r['video_path']).stem for r in reused}
            eligible=[r for r in unique if r['id'] not in reused_ids]
            write(RUN/'syncreward/eligibility.json',dict(eligible=[r['id'] for r in eligible],unmeasurable=[],
                upstream_padding_ids=[r['id'] for r in unique if r['samples']<76000 or r['actual_video_frames']<119],
                full_native_span_ids=[r['id'] for r in unique if r['samples']>=76000 and r['actual_video_frames']>=119],
                rule='All native clips satisfy official loader minimum. Upstream window_segments pads short segments internally; input media unchanged. Report full4.75s native-support sensitivity separately.'))
            run(['bash',B/'run_scorer.sh',stage,*[r['video_path'] for r in eligible],'--output',RUN/'syncreward/new_scores.jsonl','--encode_batch_size','1','--precision','bf16'],stage)
            fresh=[json.loads(line) for line in (RUN/'syncreward/new_scores.jsonl').read_text().splitlines()]
            combined=reused+fresh
            assert len(combined)==len(unique)==len({Path(r['video_path']).stem for r in combined})
            (RUN/'syncreward/scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
        elif stage=='peavs':
            ordered=sorted([r for r in rows if r['paired_available']],key=lambda r:Path(r['video_path']).name)
            unpaired=[r for r in rows if not r['paired_available']]
            assert len(ordered)==208 and len(unpaired)==1
            write(RUN/'peavs/batch_composition.json',dict(batch_size=64,batches=[[r['id'] for r in ordered[i:i+64]] for i in range(0,len(ordered),64)],
                unpaired_separate_batch=[r['id'] for r in unpaired],input_videos=[r['video_path'] for r in ordered],no_reused_pilot_scores=True))
            results=[]
            for label,items in [('paired',ordered),('unpaired',unpaired)]:
                resultfile=RUN/'peavs'/(label+'_native_scores.json')
                if not resultfile.exists():
                    run(['bash',B/'run_scorer.sh',stage,*[r['video_path'] for r in items],'--output',resultfile,'--metric_batch_size','64'],stage+'_'+label)
                result=json.loads(resultfile.read_text())
                assert len(result['scores'])==len(items)
                results.extend(result['scores'])
            write(RUN/'peavs/scores.json',dict(scores=results,metric_batch_size=64,paired_media=208,unpaired_media_separate=1,
                warning='PEAVS batch composition dependent; unpaired N scores are descriptive only, separate batch1.'))
        write(done,dict(status='COMPLETE',ended=time.time()))
    write(RUN/'status.json',dict(status='SCORING_COMPLETE',models=9,updated=time.time()))
    run([B/'envs/eval-base/bin/python',S/'analyze.py','--run',RUN],'analysis')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(RUN/'failure.json',dict(error=repr(error),time=time.time()));raise
