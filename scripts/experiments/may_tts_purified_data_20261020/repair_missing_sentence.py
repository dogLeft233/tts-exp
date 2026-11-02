"""Text-content repair only: preserve failed chunk, resynthesize its sentences."""
from pathlib import Path
import sys,json,re,io,requests
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from prepare_audio import OUT,sha,save,stats
from cloud31_audio import call,asr,word_error,DEST
def main():
 key=sys.stdin.readline().strip();old=DEST/'chunks/001';d=DEST/'chunk001_content_repair_v1';d.mkdir(exist_ok=False)
 r=json.loads((old/'receipt.json').read_text());voice=json.loads((old/'voice_response.json').read_text())['body']['output']['voice_id']
 assert r['word_error_rate']==.4
 parts=[x.strip() for x in re.split(r'(?<=[.!?])\s+',r['text'].strip()) if x.strip()];assert len(parts)==2
 waves=[]
 for i,text in enumerate(parts):
  p=dict(model='qwen-audio-3.1-tts-flash',input=dict(text=text,voice=voice,format='wav',sample_rate=24000,language_hints=['en'],seed=42,rate=1.0,pitch=1.0))
  save(d/f'part{i}_request_metadata.json',p);v=call(key,'/services/audio/tts/SpeechSynthesizer',p,d/f'part{i}_tts_response.json')
  resp=requests.get(v['output']['audio']['url'],timeout=120);resp.raise_for_status();(d/f'part{i}_native.wav').write_bytes(resp.content)
  y,sr=sf.read(io.BytesIO(resp.content),dtype='float64');assert sr==24000 and y.ndim==1;waves.append(y)
 y=np.concatenate(waves);sf.write(d/'tts_native.wav',y,24000,subtype='DOUBLE');z=resample_poly(y,2,3);sf.write(d/'tts_16k.wav',z,16000,subtype='DOUBLE')
 hyp=asr(key,d/'tts_native.wav',d/'tts_asr.json',24000);wer=word_error(r['text'],hyp)
 row=dict(r,tts_asr_text=hyp,word_error_rate=wer,native=stats(y),resampled=stats(z),native_sha256=sha(d/'tts_native.wav'),resampled_sha256=sha(d/'tts_16k.wav'),repair='sentencewise_same_voice_same_model_no_audio_normalization',original_receipt_sha256=sha(old/'receipt.json'))
 save(d/'receipt.json',row);print(json.dumps(dict(wer=wer,seconds=len(z)/16000)))
 assert wer==0,'Do not silently accept content mismatch'
 manifest=json.loads((DEST/'audio_receipt.json').read_text());manifest['chunks']=[row if x['id']==1 else x for x in manifest['chunks']]
 manifest['tts_seconds']=sum(x['resampled']['samples'] for x in manifest['chunks'])/16000
 manifest['content_repair']=dict(id=1,path=d.name,original_retained=True,reason='Two independent ASR models confirm first sentence missing; no Sync-C selection')
 save(DEST/'audio_receipt_revision02.json',manifest)
if __name__=='__main__':main()
