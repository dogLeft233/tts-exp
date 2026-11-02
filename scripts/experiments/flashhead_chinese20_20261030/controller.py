"""Sequential private-directory experiment controller; never manages foreign jobs."""
from pathlib import Path
import json, os, time, subprocess, sys, fcntl
from prepare import B, RUN, sha, write

S=Path(__file__).resolve().parent
GPU='GPU-de7ff5ff-3510-739f-917c-cb98892de759'

def run(cmd, name, cwd=None):
    log=RUN/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        result=subprocess.run([str(x) for x in cmd],stdout=stream,stderr=subprocess.STDOUT,cwd=cwd)
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
    gen=json.loads((RUN/'generation_complete.json').read_text());assert gen['status']=='PASS' and gen['completed']==80
    generation_lock=(RUN/'experiment.lock').open('w')
    fcntl.flock(generation_lock,fcntl.LOCK_EX)
    rows=json.loads((RUN/'inputs.json').read_text())['rows'];seen={};unique=[];aliases={}
    for r in rows:
        r['video_path']=str(RUN/'videos'/(r['id']+'.mp4'));r['video_sha256']=sha(r['video_path'])
        key=(r['audio_sha256'],r['video_sha256'])
        if key in seen:aliases[r['id']]=seen[key]
        else:seen[key]=r['id'];unique.append(r)
    write(RUN/'scoring_inputs.json',dict(rows=rows,unique_rows=unique,aliases=aliases,
        policy='Scores of exactly identical audio+video artifacts may be referenced for model-independent per-clip scorers; PEAVS all80 batch composition is preserved'))
    envnames={'avs_u':'cogenav','synchformer':'synchformer','sparsesync':'synchformer','mtd':'eval-base','vocalist':'eval-base'}
    stages=['syncnet','cogenav','avs_u','mtd','vocalist','synchformer','sparsesync','syncreward','peavs']
    for stage in stages:
        done=RUN/'stages'/(stage+'_complete.json')
        if done.exists():continue
        resources(stage);write(RUN/'status.json',dict(stage=stage,status='RUNNING',updated=time.time()))
        if stage=='syncnet':
            import pickle,numpy as np
            import torch
            scores=[]
            for r in unique:
                resultfile=RUN/'syncnet'/(r['id']+'.json')
                if resultfile.exists():scores.append(json.loads(resultfile.read_text()));continue
                work=RUN/'syncnet/work';ref=r['id']
                # Preserve the previous Chinese protocol: lossless PCM MKV and min_track25.
                # The generic deployment wrapper uses pipeline defaults and cannot pass
                # pipeline-only flags to run_syncnet.py, so invoke the two official CLIs.
                media=RUN/'syncnet/inputs'/(ref+'.mkv');media.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-y','-copyts','-i',r['video_path'],'-i',r['audio_path'],
                    '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le','-ar','16000','-ac','1','-avoid_negative_ts','disabled',str(media)],check=True)
                import soundfile as sf
                decoded=subprocess.check_output([str(B/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',str(media),'-map','0:a:0','-f','s16le','-'])
                assert decoded==sf.read(r['audio_path'],dtype='int16')[0].astype('<i2').tobytes()
                for ext in [r['video_path'],str(media)]:
                    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',ext]))
                    pts=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
                    if ext==r['video_path']:expected_pts=pts
                    else:assert np.allclose(pts,expected_pts,atol=1e-6,rtol=0)
                common=['--videofile',media,'--reference',ref,'--data_dir',work]
                python=B/'envs/eval-base/bin/python';source=B/'scorers/syncnet'
                run([python,B/'gpu_guard.py',source/'run_pipeline.py',*common,'--min_track','25','--overwrite'],stage+'_pipeline_'+ref,source)
                run([python,B/'gpu_guard.py',source/'run_syncnet.py',*common,'--initial_model',source/'data/syncnet_v2.model'],stage+'_'+ref,source)
                dist=pickle.loads((work/'pywork'/ref/'activesd.pckl').read_bytes())
                assert len(dist)==1,'Expected single generated face track'
                z=np.asarray(dist[0],dtype=np.float32);assert z.ndim==2 and z.shape[1]==31
                np.save(RUN/'syncnet'/(ref+'_distances.npy'),z)
                curve=torch.from_numpy(z).T.mean(1);d,k=torch.min(curve,0);median=torch.median(curve)
                score=dict(id=ref,status='PASS',C=float(median-d),D=float(d),B=float(median),offset=int(15-k),windows=len(z))
                write(resultfile,score);scores.append(score)
            write(RUN/'syncnet/scores.json',scores)
        elif stage=='cogenav':
            target=RUN/'cogenav';target.mkdir(exist_ok=True)
            for name in ['upstream','weights','autoavsr_detector.py','autoavsr_video_process.py','20words_mean_face.npy']:
                dest=target/name
                if not dest.exists():dest.symlink_to(B/'assets/cogenav'/name)
            manifest=dict(rows=[dict(r,sample_id=r['source'],qualified=False,score_float32=None,features={'duration_s':r['duration_s']}) for r in unique])
            write(target/'input_manifest.json',manifest)
            run(['bash',B/'run_scorer.sh',stage,'--run',target,'--manifest',target/'input_manifest.json','--sources','20','--record-long-clips'],stage)
        elif stage in envnames:
            run([B/'envs'/envnames[stage]/'bin/python',B/'gpu_guard.py',S/'batch_adapters.py',stage],stage)
        elif stage=='syncreward':
            # Unlike the two classifiers, official SyncReward explicitly supports
            # short native clips by repeating tail video and zero-padding segments.
            # Preserve that upstream inference definition; never pad input media.
            eligible=unique
            write(RUN/'syncreward/eligibility.json',dict(eligible=[r['id'] for r in eligible],unmeasurable=[],
                upstream_padding_ids=[r['id'] for r in unique if r['samples']<76000],
                full_native_span_ids=[r['id'] for r in unique if r['samples']>=76000],
                rule='All native clips satisfy official loader minimum. Upstream window_segments pads short segments internally; input media unchanged. Report full4.75s native-support sensitivity separately.'))
            run(['bash',B/'run_scorer.sh',stage,*[r['video_path'] for r in eligible],'--output',RUN/'syncreward/scores.jsonl','--encode_batch_size','1','--precision','bf16'],stage)
        elif stage=='peavs':
            ordered=sorted(rows,key=lambda r:Path(r['video_path']).name)
            write(RUN/'peavs/batch_composition.json',dict(batch_size=64,batches=[[r['id'] for r in ordered[:64]],[r['id'] for r in ordered[64:]]],input_videos=[r['video_path'] for r in ordered]))
            run(['bash',B/'run_scorer.sh',stage,*[r['video_path'] for r in ordered],'--output',RUN/'peavs/scores.json','--metric_batch_size','64'],stage)
        write(done,dict(status='COMPLETE',ended=time.time()))
    write(RUN/'status.json',dict(status='SCORING_COMPLETE',models=9,updated=time.time()))
    run([B/'envs/eval-base/bin/python',S/'analyze.py','--run',RUN],'analysis')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(RUN/'failure.json',dict(error=repr(error),time=time.time()));raise
