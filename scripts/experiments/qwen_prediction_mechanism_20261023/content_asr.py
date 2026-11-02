"""Original Whisper-small CPU int8 content diagnostic; no hints or VAD."""
from pathlib import Path
import argparse,os,sys,time,traceback,wave,hashlib,signal,importlib.metadata
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha

def main(run):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    from scripts.experiments.qwen_tts_decline_audit_20261022 import normalize,edit_distance
    p=read(run/'C_protocol.json');samples={s['sample_id']:s for s in p['samples']};rows=read(run/'C_audio_manifest.json')['records']
    directory=ROOT/'runs/mfa_linear_content_pause_prosody_mapping_20260925/asr_model_small';pins={str(f):sha(f) for f in directory.iterdir() if f.is_file()}
    assert pins==read(ROOT/'runs/qwen_tts_decline_audit_20261022_attempt01/protocol.json')['ASR']['model_files']
    packages={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','numpy','opencc-python-reimplemented']}
    assert packages==read(ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01/asr/runtime.json')['packages']
    out=run/'C_ASR';out.mkdir();save(out/'runtime.json',{'executable':sys.executable,'model_files':pins,'packages':packages,'device':'cpu','compute_type':'int8','settings':{'language':'zh','beam_size':5,'temperature':0,'vad_filter':False,'condition_on_previous_text':False,'initial_prompt':None}})
    model=WhisperModel(str(directory),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True);cc=OpenCC('t2s');result=[]
    for row in rows:
        if row['status']!='COMPLETE':continue
        try:
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('ASR unit300s')));signal.alarm(300)
            assert sha(row['path'])==row['sha256']
            with wave.open(row['path']) as f:
                assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,16000);raw=f.readframes(f.getnframes())
            assert hashlib.sha256(raw).hexdigest()==row['pcm_sha256']
            x=np.frombuffer(raw,dtype='<i2').astype(np.float32)/np.float32(32768)
            segments,info=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
            ss=[{'start':s.start,'end':s.end,'text':s.text} for s in segments];text=''.join(s['text'] for s in ss);ref=normalize(samples[row['sample_id']]['transcript'],cc);hyp=normalize(text,cc);distance=edit_distance(ref,hyp)
            r={'id':row['id'],'status':'COMPLETE','audio_sha':row['sha256'],'text':text,'normalized':hyp,'reference':ref,'edit_distance':distance,'CER':distance/len(ref),'segments':ss,'duration_s':info.duration}
        except BaseException as e:r={'id':row['id'],'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}
        finally:signal.alarm(0)
        save(out/(row['id']+'.json'),r);result.append(r);print('ASR',len(result),row['id'],r['status'],flush=True)
    save(out/'results.json',{'records':result,'count':len(result),'human_verified':False})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
