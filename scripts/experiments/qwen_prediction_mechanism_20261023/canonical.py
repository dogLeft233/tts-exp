"""Same-RMS PCM bridge for all unique confirmation audio; retain failures."""
from pathlib import Path
import argparse,hashlib,sys,subprocess,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha

def main(run):
    import numpy as np,soundfile as sf
    p=read(run/'C_protocol.json');out=run/'C_audio';out.mkdir();result=[]
    for row in p['rows']:
        if row['duplicate_of']:result.append({'id':row['id'],'status':'EXACT_REUSE','duplicate_of':row['duplicate_of']});continue
        record=read(run/('C_generation_'+row['model'])/row['id']/'result.json')
        if record['status']=='FAILED':result.append({'id':row['id'],'status':'FAILED','generation_failure':record});continue
        unit=out/row['id'];unit.mkdir();source=read(row['old_record']) if row['reuse_L06_baseline'] else record['raw']
        sourcepath=source['path'];digest=source['sha256'];assert sha(sourcepath)==digest
        try:
            f32=unit/'float16.wav';cmd=['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',sourcepath,'-ac','1','-ar','16000','-c:a','pcm_f32le',str(f32)]
            cp=subprocess.run(cmd,capture_output=True,text=True,timeout=60);save(unit/'resample_receipt.json',{'command':cmd,'returncode':cp.returncode,'stderr':cp.stderr});cp.check_returncode()
            x,sr=sf.read(f32,dtype='float32');assert sr==16000 and x.ndim==1 and len(x)>0 and np.isfinite(x).all()
            rms=float(np.sqrt(np.mean(x.astype(float)**2)));assert rms>0
            gain=p['target_rms']/rms;y=x*np.float32(gain);assert np.max(abs(y))<1.,'cannot normalize without clipping'
            dest=unit/'canonical.wav';sf.write(dest,y,16000,subtype='PCM_16');pcm,_=sf.read(dest,dtype='int16');assert len(pcm)==len(x)
            if row['reuse_L06_baseline']:
                assert sha(dest)==row['old_audio']['sha256'],'old baseline PCM bridge'
                assert np.array_equal(pcm,sf.read(row['old_audio']['path'],dtype='int16')[0])
            r={'id':row['id'],'status':'COMPLETE','sample_id':row['sample_id'],'speaker_id':row['speaker_id'],'role':'T'+str(row['draw']),'path':str(dest),'sha256':sha(dest),'samples':len(pcm),'sr':16000,'pcm_sha256':hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),'source_path':sourcepath,'source_sha256':digest,'gain':gain,'rms_before':rms,'rms_after':float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),'EOS':record['EOS'],'duration_ratio':record['duration_ratio'],'qualified_EOS_duration':bool(record['EOS'] and .5<=record['duration_ratio']<=2.),'baseline_PCM_exact':bool(row['reuse_L06_baseline'])}
        except BaseException as e:r={'id':row['id'],'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}
        save(unit/'result.json',r);result.append(r)
    save(run/'C_audio_manifest.json',{'records':result,'count':len(result),'unique_successes':sum(r['status']=='COMPLETE' for r in result)})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
