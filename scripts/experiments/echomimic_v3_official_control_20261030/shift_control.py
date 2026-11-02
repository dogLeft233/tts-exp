"""Real SyncNet offset controls, isolated from experimental score cells."""
import json
import os
import pickle
import subprocess
import numpy as np
import soundfile as sf
import torch
from control import B,RUN,read,write


def main():
    scores=read(RUN/'scores.json');plan=read(RUN/'plan.json');out=[]
    ff=B/'tools/bin/ffmpeg';source=B/'scorers/syncnet';python=B/'envs/eval-base/bin/python'
    for r in plan['rows'][:2]:
        reference=next(s for s in scores if s['condition']=='baseline' and s['id']==r['id'])
        baseline=RUN/'syncnet'/('baseline__'+r['id'])
        pcm,sr=sf.read(baseline/'audio.wav',dtype='int16')
        for shift in (-10,10):
            key=r['id']+('_m10' if shift<0 else '_p10');root=RUN/'offset_controls'/key
            root.mkdir(parents=True)
            audio=root/'shifted.wav';sf.write(audio,np.roll(pcm,shift*640),sr,subtype='PCM_16')
            media=root/'input.mkv'
            subprocess.run([str(ff),'-nostdin','-v','error','-i',str(baseline/'visual.mp4'),'-i',str(audio),
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le',str(media)],check=True)
            common=['--videofile',str(media),'--reference',key,'--data_dir',str(root/'work')]
            for script,extra in [('run_pipeline.py',['--min_track','25','--overwrite']),('run_syncnet.py',['--initial_model',str(source/'data/syncnet_v2.model')])]:
                with (root/(script+'.log')).open('x') as log:
                    subprocess.run([str(python),str(B/'gpu_guard.py'),str(source/script),*common,*extra],cwd=source,
                        stdout=log,stderr=subprocess.STDOUT,check=True,
                        env=dict(os.environ,GPU_METRICS_PATH=str(root/(script+'_gpu.json'))))
            dist=pickle.loads((root/'work/pywork'/key/'activesd.pckl').read_bytes());assert len(dist)==1
            z=np.asarray(dist[0],dtype=np.float32);np.save(root/'distances.npy',z)
            curve=torch.from_numpy(z).T.mean(1);d,k=curve.min(0);offset=int(15-k)
            rec=dict(id=r['id'],diagnostic_audio_roll_frames=shift,offset=offset,
                baseline_offset=reference['offset'],offset_change=offset-reference['offset'],C=float(curve.median()-d))
            # Search-lag confidence need not decrease for a recoverable time shift.
            rec['magnitude_pass']=abs(abs(rec['offset_change'])-10)<=1
            write(root/'result.json',rec);out.append(rec);print(rec,flush=True)
    groups={r['id']:[s for s in out if s['id']==r['id']] for r in plan['rows'][:2]}
    passed=all(s['magnitude_pass'] for s in out) and all(v[0]['offset_change']*v[1]['offset_change']<0 for v in groups.values())
    write(RUN/'offset_control_summary.json',dict(status='PASS' if passed else 'CONTROL_FAILED',rows=out,
        rule='Real +/-400ms circular diagnostic waveform roll; fitted offset changes about10frames in opposite directions. No claim that searched Sync-C must decrease. Never used in experiment aggregates.'))
    assert passed,'Offset control failed; preserve results and disclose limitation'


if __name__=='__main__':main()
