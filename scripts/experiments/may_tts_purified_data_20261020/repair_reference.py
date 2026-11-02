"""Repair confirmed text omission using an already qualified train-only May voice."""
from pathlib import Path
import sys,json,io,requests
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from prepare_audio import OUT,sha,save,stats
from cloud31_audio import call,asr,word_error,DEST
def main():
 key=sys.stdin.readline().strip();old=DEST/'chunks/001';d=DEST/'chunk001_content_repair_v2';d.mkdir(exist_ok=False)
 r=json.loads((old/'receipt.json').read_text());voicepath=DEST/'chunks/000/voice_response.json';voice=json.loads(voicepath.read_text())['body']['output']['voice_id']
 payload=dict(model='qwen-audio-3.1-tts-flash',input=dict(text=r['text'],voice=voice,format='wav',sample_rate=24000,language_hints=['en'],seed=42,rate=1.0,pitch=1.0))
 save(d/'request_metadata.json',dict(payload=payload,reference_chunk=0,reference_source_frames=[0,348],voice_response_sha256=sha(voicepath),reason='Confirmed missing sentence; same May train-only alternative reference, not score tuning'))
 v=call(key,'/services/audio/tts/SpeechSynthesizer',payload,d/'tts_response.json')
 resp=requests.get(v['output']['audio']['url'],timeout=120);resp.raise_for_status();(d/'tts_native.wav').write_bytes(resp.content)
 y,sr=sf.read(io.BytesIO(resp.content),dtype='float64');assert sr==24000 and y.ndim==1
 z=resample_poly(y,2,3);sf.write(d/'tts_16k.wav',z,16000,subtype='DOUBLE')
 hyp=asr(key,d/'tts_native.wav',d/'tts_asr.json',24000);wer=word_error(r['text'],hyp)
 row=dict(r,tts_asr_text=hyp,word_error_rate=wer,native=stats(y),resampled=stats(z),native_sha256=sha(d/'tts_native.wav'),resampled_sha256=sha(d/'tts_16k.wav'),repair='same_cloud_model_May_train000_reference_no_normalization',original_receipt_sha256=sha(old/'receipt.json'))
 save(d/'receipt.json',row);print(json.dumps(dict(wer=wer,seconds=len(z)/16000)))
 assert wer==0
 m=json.loads((DEST/'audio_receipt.json').read_text());m['chunks']=[row if x['id']==1 else x for x in m['chunks']];m['tts_seconds']=sum(x['resampled']['samples'] for x in m['chunks'])/16000
 m['content_repair']=dict(id=1,path=d.name,original_retained=True,reason='Confirmed missing sentence; same May train-only reference000, no Sync-C selection')
 save(DEST/'audio_receipt_revision02.json',m)
if __name__=='__main__':main()
