"""Cloud Qwen-Audio3.1 only; native output amplitude/timing, auditable revision."""
from pathlib import Path
import base64,io,json,sys,requests,re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from prepare_audio import OUT,sha,save,stats
DEST=OUT/'cloud31_v1'
BASE='https://dashscope.aliyuncs.com/api/v1'
def call(key,route,payload,path):
 assert not path.exists()
 r=requests.post(BASE+route,headers={'Authorization':'Bearer '+key,'X-DashScope-SSE':'disable'},json=payload,timeout=180)
 v=r.json();save(path,dict(http_status=r.status_code,body=v))
 if r.status_code!=200:raise RuntimeError(str(r.status_code)+' '+str(v.get('code'))+' '+str(v.get('message')))
 return v
def asr(key,path,out,sr):
 uri='data:audio/wav;base64,'+base64.b64encode(path.read_bytes()).decode()
 p={'model':'qwen-audio-3.1-asr-flash','input':{'messages':[{'role':'user','content':[{'type':'input_audio','input_audio':{'data':uri}}]}]},'parameters':{'format':'wav','sample_rate':str(sr),'language_hints':['en']}}
 v=call(key,'/services/aigc/multimodal-generation/generation',p,out)
 text=v['output'].get('text') or v['output'].get('output',{}).get('text')
 assert isinstance(text,str) and text.strip();return text
def word_error(ref,hyp):
 a=re.findall(r"[a-z0-9']+",ref.lower());b=re.findall(r"[a-z0-9']+",hyp.lower());v=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  w=[i]
  for j,y in enumerate(b,1):w.append(min(w[-1]+1,v[j]+1,v[j-1]+(x!=y)))
  v=w
 return v[-1]/max(1,len(a))
def work(key,r):
 d=DEST/'chunks'/f"{r['id']:03d}";d.mkdir(parents=True)
 natural=OUT/'chunks'/f"{r['id']:03d}"/'natural.wav';assert sha(natural)==r['natural_sha256']
 if r['id']==0:
  import shutil
  p=OUT/'qwen_audio31_smoke_v1/asr_response_attempt02.json';shutil.copyfile(p,d/'natural_asr.json')
  text=json.loads(p.read_text())['body']['output']['text']
 else:text=asr(key,natural,d/'natural_asr.json',16000)
 (d/'transcript.txt').write_text(text,encoding='utf-8')
 uri='data:audio/wav;base64,'+base64.b64encode(natural.read_bytes()).decode()
 clone=dict(model='voice-enrollment',input=dict(action='create_voice',target_model='qwen-audio-3.1-tts-flash',prefix=f'may31{r["id"]:03d}',url=uri,language_hints=['en'],enable_preprocess=False,enable_volume_normalization='false'))
 voice=call(key,'/services/audio/tts/customization',clone,d/'voice_response.json')['output']['voice_id']
 payload=dict(model='qwen-audio-3.1-tts-flash',input=dict(text=text,voice=voice,format='wav',sample_rate=24000,language_hints=['en'],seed=42,rate=1.0,pitch=1.0))
 save(d/'request_metadata.json',dict(model=payload['model'],input=payload['input'],voice_preprocess=False,voice_volume_normalization=False,reference_sha256=sha(natural),secret_omitted=True))
 v=call(key,'/services/audio/tts/SpeechSynthesizer',payload,d/'tts_response.json')
 resp=requests.get(v['output']['audio']['url'],timeout=120);resp.raise_for_status();(d/'tts_native.wav').write_bytes(resp.content)
 raw,sr=sf.read(io.BytesIO(resp.content),dtype='float64',always_2d=True);assert raw.shape[1]==1 and sr==24000
 raw=raw[:,0];y=resample_poly(raw,2,3);stats(raw);stats(y)
 sf.write(d/'tts_16k.wav',y,16000,subtype='DOUBLE');back,_=sf.read(d/'tts_16k.wav',dtype='float64');assert np.array_equal(back,y)
 hyp=asr(key,d/'tts_native.wav',d/'tts_asr.json',sr);(d/'tts_transcript.txt').write_text(hyp,encoding='utf-8')
 row=dict(**r,text=text,tts_asr_text=hyp,word_error_rate=word_error(text,hyp),native=stats(raw),resampled=stats(y),native_sha256=sha(d/'tts_native.wav'),resampled_sha256=sha(d/'tts_16k.wav'),native_sr=sr,normalized=False)
 save(d/'receipt.json',row);print(json.dumps(dict(id=r['id'],done=True,seconds=len(y)/16000,wer=row['word_error_rate'])),flush=True)
 return row
def main():
 key=sys.stdin.readline().strip();assert not DEST.exists();DEST.mkdir()
 p=json.loads((OUT/'protocol.json').read_text());p.update(status='CLOUD31_PREPARING',asr_model='qwen-audio-3.1-asr-flash',tts_model='qwen-audio-3.1-tts-flash',cloud_only=True,local_tts_allowed=False,voice_preprocess=False,voice_volume_normalization=False,tts_rate=1.0,tts_pitch=1.0,tts_seed=42)
 p['acceptance']=dict(sync_c_minimum_exclusive=5.0,each_chunk_guard20=True,equal_chunk_mean=True,no_score_selection_or_silent_drop=True,no_training_until_acceptance=True)
 save(DEST/'protocol.json',p)
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(lambda r:work(key,r),p['chunks']))
  save(DEST/'audio_receipt.json',dict(passed=True,tts_ready=True,asr_model=p['asr_model'],tts_model=p['tts_model'],chunks=rows,source_seconds=168,tts_seconds=sum(r['resampled']['samples'] for r in rows)/16000,normalization=False,local_tts_calls=0,training_jobs=0))
 except Exception as e:
  save(DEST/'failure.json',dict(type=type(e).__name__,message=str(e).replace(key,'[REDACTED]'),training_jobs=0));raise
if __name__=='__main__':main()
