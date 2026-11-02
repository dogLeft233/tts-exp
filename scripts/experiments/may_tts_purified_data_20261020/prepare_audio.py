"""Prepare train-only native Qwen resynthesis; never launches generator training.

Secret arrives on stdin, is kept in memory, never enters metadata or command lines.
Original and TTS amplitudes are preserved. Length conversion is resample_poly only.
"""
from pathlib import Path
import base64, hashlib, io, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

OUT=Path('F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1')
PARENT=Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
BASELINE=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017')
BASE='https://dashscope.aliyuncs.com/api/v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def stats(x):
 x=np.asarray(x,dtype=np.float64);assert x.size and np.isfinite(x).all()
 return dict(samples=len(x),rms=float(np.sqrt(np.mean(x*x))),peak=float(np.max(np.abs(x))),mean=float(x.mean()))
def prepare():
 assert not OUT.exists(),'Fresh run required; no silent overwrites'
 OUT.mkdir(parents=True)
 m=json.loads((PARENT/'manifest.json').read_text());src=PARENT/'source/May/aud.wav'
 expected=next(r['sha256'] for r in m['source_entries'] if r['path']=='source/May/aud.wav')
 assert sha(src)==expected
 x,sr=sf.read(src,dtype='float64',always_2d=True);assert sr==16000 and np.array_equal(x[:,0],x[:,1])
 x=x[:4200*640,0];assert len(x)==4200*640
 # Fixed acoustic rule, not ASR/SyncNet outcome selection. Frame-aligned cuts.
 frame_energy=np.mean(x.reshape(-1,640)**2,axis=1)
 smooth=np.convolve(frame_energy,np.ones(5)/5,mode='same');cuts=[0]
 while 4200-cuts[-1]>400:
  target=cuts[-1]+350;lo=target-50;hi=min(target+51,4200-200)
  cuts.append(lo+int(np.argmin(smooth[lo:hi])))
 cuts.append(4200)
 chunks=[]
 for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
  d=OUT/'chunks'/f'{i:03d}';d.mkdir(parents=True)
  p=d/'natural.wav';sf.write(p,x[a*640:b*640],sr,subtype='PCM_16')
  y,_=sf.read(p,dtype='float64');assert np.array_equal(y,x[a*640:b*640])
  chunks.append(dict(id=i,source_start_frame=a,source_end_frame_exclusive=b,natural_samples=len(y),natural_sha256=sha(p)))
 protocol=dict(schema='may_tts_purified_data_v1',status='AUDIO_PREPARING',baseline_report=50,
  baseline_protocol_sha256=sha(BASELINE/'protocol.json'),baseline_initialization_sha256=sha(BASELINE/'initialization.pt'),
  baseline_frame_order_sha256=sha(BASELINE/'frame_order.npy'),source_audio_sha256=expected,
  source_support_frames=[0,4200],source_supervision_frames=[25,4175],
  untouched_validation_frames=[4275,4825],untouched_test_frames=[4925,6025],
  asr_model='qwen3-asr-flash',tts_model='qwen3-tts-vc-2026-01-22',language='English',
  voice_reference='each train-only natural chunk, self cloning',audio_normalization=False,
  gain_db=0,denoise=False,clip=False,limiter=False,remove_mean=False,tts_native_timing=True,
  sample_rate_conversion='scipy.signal.resample_poly 24000->16000, no amplitude normalization',
  purification=dict(method='NAS SyncDrift unchanged estimate_correction',mode='window',window_sec=.5,stride_sec=.1,search_min_ms=-1000,search_max_ms=1000,median_filter_size=3,min_valid_feats=8,flat_score_epsilon=1e-4,
    initial_video_clock='per chunk uniform natural frame mapping to native TTS duration',boundary='no mapping across chunks or into validation/test'),
  train_authorized=False,generator_training_jobs=0,chunks=chunks)
 save(OUT/'protocol.json',protocol)
 sf.write(OUT/'natural_train_support.wav',x,sr,subtype='PCM_16')
 return protocol
def request(key,route,payload,receipt):
 response=requests.post(BASE+route,headers={'Authorization':'Bearer '+key},json=payload,timeout=180)
 try:v=response.json()
 except Exception:v={'code':'non_json','status_code':response.status_code}
 # Responses do not contain authentication headers or input base64.
 save(receipt,dict(http_status=response.status_code,body=v))
 if response.status_code!=200:raise RuntimeError('API failed: HTTP '+str(response.status_code)+' code='+str(v.get('code')))
 return v
def transcribe(key,path,receipt):
 return request(key,'/services/aigc/multimodal-generation/generation',dict(model='qwen3-asr-flash',input=dict(messages=[dict(role='user',content=[dict(audio='data:audio/wav;base64,'+base64.b64encode(path.read_bytes()).decode())])]),parameters=dict(asr_options=dict(language='en',enable_itn=False))),receipt)['output']['choices'][0]['message']['content'][0]['text']
def work(key,row):
 d=OUT/'chunks'/f"{row['id']:03d}";natural=d/'natural.wav'
 text=transcribe(key,natural,d/'asr_response.json');assert text.strip()
 (d/'transcript.txt').write_text(text,encoding='utf-8')
 voice=request(key,'/services/audio/tts/customization',dict(model='qwen-voice-enrollment',input=dict(action='create',target_model='qwen3-tts-vc-2026-01-22',preferred_name=f"mayprep{row['id']:03d}",audio=dict(data='data:audio/wav;base64,'+base64.b64encode(natural.read_bytes()).decode()))),d/'voice_response.json')['output']['voice']
 v=request(key,'/services/aigc/multimodal-generation/generation',dict(model='qwen3-tts-vc-2026-01-22',input=dict(text=text,voice=voice,language_type='English')),d/'tts_response.json')
 response=requests.get(v['output']['audio']['url'],timeout=120);response.raise_for_status()
 (d/'tts_native.wav').write_bytes(response.content)
 raw,sr=sf.read(io.BytesIO(response.content),dtype='float64',always_2d=True);assert raw.shape[1]==1 and sr==24000
 raw=raw[:,0];stats(raw)
 y=resample_poly(raw,2,3);sf.write(d/'tts_16k.wav',y,16000,subtype='DOUBLE')
 back,_=sf.read(d/'tts_16k.wav',dtype='float64');assert np.array_equal(back,y)
 ttext=transcribe(key,d/'tts_native.wav',d/'tts_asr_response.json')
 (d/'tts_transcript.txt').write_text(ttext,encoding='utf-8')
 result=dict(**row,natural_text=text,tts_asr_text=ttext,voice=voice,native_sample_rate=sr,native=stats(raw),resampled=stats(y),native_sha256=sha(d/'tts_native.wav'),resampled_sha256=sha(d/'tts_16k.wav'),amplitude_normalization=False)
 save(d/'receipt.json',result);print(json.dumps(dict(chunk=row['id'],done=True,tts_seconds=len(y)/16000)),flush=True)
 return result
def main():
 p=prepare();key=sys.stdin.readline().strip();assert key.startswith('sk-')
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(lambda r:work(key,r),p['chunks']))
  joined=np.concatenate([sf.read(OUT/'chunks'/f"{r['id']:03d}"/'tts_16k.wav',dtype='float64')[0] for r in rows])
  sf.write(OUT/'tts_train_native_clock.wav',joined,16000,subtype='DOUBLE')
  save(OUT/'audio_receipt.json',dict(passed=True,chunks=rows,tts=stats(joined),sha256=sha(OUT/'tts_train_native_clock.wav'),normalization=False,training_jobs=0))
 except Exception as e:
  save(OUT/'audio_failure.json',dict(type=type(e).__name__,message=str(e).replace(key,'[REDACTED]'),training_jobs=0));raise
 finally:key=None
if __name__=='__main__':main()
