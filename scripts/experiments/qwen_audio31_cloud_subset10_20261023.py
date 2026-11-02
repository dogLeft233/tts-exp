"""CPU-only cloud self-cloning pilot. Credentials arrive through stdin only.

Ten frozen references, three calls per person, no retries or output selection.
Preserves provider bytes and uses the existing NRT canonical/RMS contract.
"""
from __future__ import annotations
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
import soundfile as sf

MODEL = 'qwen-audio-3.1-tts-plus'
BASE = 'https://dashscope.aliyuncs.com/api/v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def canonical_values(values,target):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid canonical wave')
    rms=float(np.sqrt(np.mean(x.astype(float)**2)))
    if rms<=0:raise ValueError('Silent output')
    gain=target/rms
    if float(np.abs(x).max()*gain)>=1:raise ValueError('RMS normalization clips; do not adjust target')
    return x*np.float32(gain),rms,gain


def safe_url(url):
    u=urlsplit(url);return urlunsplit((u.scheme,u.netloc,u.path,'',''))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--key-stdin',action='store_true',required=True)
    args=ap.parse_args();run=args.run
    key=sys.stdin.readline().strip()
    if not key:raise RuntimeError('Empty runtime credential')
    os.environ['DASHSCOPE_API_KEY']=key
    def redact(s):return str(s).replace(key,'[REDACTED]')
    p=json.loads((run/'protocol.json').read_text('utf8'))
    if len(p['samples'])!=10 or p['cloud_model']!=MODEL or p['draws_per_speaker']!=3:raise ValueError('Frozen cloud layout changed')
    if (run/'cloud_completion.json').exists():raise FileExistsError('No silent rerun')
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    records=[];calls=[];start=time.monotonic();failed=None
    def event(kind,**kw):
        j=dict(event=kind,elapsed_s=time.monotonic()-start,pid=os.getpid(),**kw)
        line=json.dumps(j,ensure_ascii=False,allow_nan=False)
        with (run/'cloud_events.jsonl').open('a',encoding='utf8') as f:f.write(line+'\n')
        print(line,flush=True)
    def post(endpoint,payload,stem):
        begin=time.monotonic()
        response=requests.post(BASE+endpoint,headers=headers,json=payload,timeout=(15,120))
        try:body=response.json()
        except ValueError:body={'message':redact(response.text[:1000])}
        receipt={'endpoint':BASE+endpoint,'http_status':response.status_code,'request_id':body.get('request_id'),
                 'usage':body.get('usage'),'elapsed_s':time.monotonic()-begin,'code':body.get('code'),
                 'message':redact(body.get('message','')),'model':payload['model'],'attempt':1,'no_retry':True}
        write(run/'api_receipts'/(stem+'.json'),receipt);calls.append(receipt)
        if response.status_code!=200 or body.get('code'):raise RuntimeError('API '+stem+' failed: '+json.dumps(receipt,ensure_ascii=False))
        return body
    try:
        for row in p['samples']:
            sid=row['sample_id'];reference=Path(row['input24_path'])
            if sha(reference)!=row['input24_sha256']:raise ValueError('Reference SHA changed '+sid)
            data=reference.read_bytes()
            payload={'model':'voice-enrollment','input':{'action':'create_voice','target_model':MODEL,
                     'prefix':'nrts'+row['speaker_id'][1:],'audio':'data:audio/wav;base64,'+base64.b64encode(data).decode(),
                     'language_hints':['zh'],'enable_preprocess':False,'enable_volume_normalization':'false',
                     'max_prompt_audio_length':10.0}}
            body=post('/services/audio/tts/customization',payload,'enroll_'+sid)
            voice=body['output']['voice_id']
            write(run/'voices'/(sid+'.json'),{'sample_id':sid,'voice_id':voice,'reference_sha256':row['input24_sha256'],
                   'target_model':MODEL,'request_id':body.get('request_id')})
            text=''.join(row['transcript'].split())
            for draw in (1,2,3):
                if time.monotonic()-start>3600:raise TimeoutError('Cloud generation hard timeout')
                role='C'+str(draw);stem=sid+'_'+role
                body=post('/services/audio/tts/SpeechSynthesizer',{'model':MODEL,'input':{'text':text,'voice':voice,'format':'wav','sample_rate':24000}},stem)
                audio=body['output']['audio'];url=audio['url'] if isinstance(audio,dict) else audio
                downloaded=requests.get(url,timeout=(15,120));downloaded.raise_for_status()
                dest=run/'provider_audio'/sid/(role+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(downloaded.content)
                info=sf.info(dest)
                if info.frames==0:raise ValueError('Empty provider audio')
                canonical=run/'audio'/sid/(role+'_16k_pcm16.wav');canonical.parent.mkdir(parents=True,exist_ok=True)
                floatpath=canonical.with_name(role+'_16k_float.wav')
                cmd=['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',str(dest),'-ac','1','-ar','16000','-c:a','pcm_f32le',str(floatpath)]
                rr=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                write(run/'canonical_receipts'/(stem+'.json'),{'command':cmd,'returncode':rr.returncode,'stderr':redact(rr.stderr)})
                if rr.returncode:raise RuntimeError('Canonicalization failed '+stem)
                x,sr=sf.read(floatpath,dtype='float32')
                if sr!=16000:raise ValueError('Unexpected canonical sample rate')
                y,rms,gain=canonical_values(x,p['target_rms']);sf.write(canonical,y,sr,subtype='PCM_16')
                pcm,_=sf.read(canonical,dtype='int16')
                if len(pcm)!=len(x):raise ValueError('Canonical sample count changed')
                rec={'sample_id':sid,'speaker_id':row['speaker_id'],'role':role,'draw_index':draw,'cloud_model':MODEL,
                     'voice_id':voice,'reference_audio_sha256':row['input24_sha256'],'target_text':text,'provider_audio':str(dest),
                     'provider_sha256':sha(dest),'provider_sample_rate':info.samplerate,'provider_frames':info.frames,
                     'provider_channels':info.channels,'provider_subtype':info.subtype,'request_id':body.get('request_id'),
                     'finish_reason':body.get('output',{}).get('finish_reason'),'url_without_query':safe_url(url),'usage':body.get('usage'),
                     'path':str(canonical),'sha256':sha(canonical),'pcm_sha256':hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                     'samples':len(pcm),'sr':sr,'rms_before':rms,'gain':gain,'rms_after':float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),
                     'duration_ratio':len(pcm)/sr/row['duration_s'],'seed_control':'API has no requested seed; three calls, not three independent speakers',
                     'no_time_editing':True}
                write(run/'audio_records'/(stem+'.json'),rec);records.append(rec);event('cloud_draw_complete',sample_id=sid,role=role,completed=len(records),planned=30)
        write(run/'audio_manifest.json',{'status':'COMPLETE','records':records,'count':len(records),'protocol_sha256':sha(run/'protocol.json')})
    except BaseException as exc:
        failed={'type':type(exc).__name__,'message':redact(exc),'traceback':redact(traceback.format_exc())}
        write(run/'cloud_failure.json',failed);event('failed',failure=failed)
    finally:
        os.environ.pop('DASHSCOPE_API_KEY',None)
        result={'status':'COMPLETE' if len(records)==30 and failed is None else 'INCOMPLETE','completed':len(records),
                'planned':30,'API_calls':len(calls),'enrollments_planned':10,'syntheses_planned':30,
                'elapsed_seconds':time.monotonic()-start,'failure':failed,'no_retry':True,'no_training':True,'pid':os.getpid(),
                'secret_persisted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(run/'cloud_completion.json',result);event('finished',status=result['status'],completed=len(records))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
