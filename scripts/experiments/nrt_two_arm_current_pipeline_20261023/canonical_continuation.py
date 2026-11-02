"""Authorized CPU continuation: one frozen three-sample PCM saturation exception.

All generation and original attempt bodies remain unchanged. Other 79 cells
retain the original no-clipping gate and fixed RMS. No GPU or model imports.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import numpy as np
import soundfile as sf

BASE_SHA='4afbb4fc1d217a4d37450e5ca5b810c08724f9fde094dad197ac584c535ac3ee'
EXCEPTION_KEY='BAC009S0908W0121_T2'
EXCEPTION_FLOAT_SHA='00dda8f475abdba88a7fd8158cb7504e67cfd072d031c05712919e0c6471f46c'
INDICES=[364180,364181,364182]
TARGET=.0376838172675746

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def normalize(x,key,float_sha):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('finite mono nonempty input required')
    rms=float(np.sqrt(np.mean(x.astype(float)**2)))
    if rms<=0:raise ValueError('silent input')
    gain=TARGET/rms;y=x*np.float32(gain);exception=None
    if key==EXCEPTION_KEY:
        if float_sha!=EXCEPTION_FLOAT_SHA:raise ValueError('exception source differs')
        clipped=np.clip(y,np.float32(-1),np.float32(32767/32768))
        idx=np.flatnonzero(y!=clipped).tolist()
        if idx!=INDICES:raise ValueError('exception must change exactly frozen three samples')
        exception=dict(policy='explicit user-authorized three-sample saturation exception, no limiter or gain change',
             indices_zero_based=idx,values_before=[float(y[i]) for i in idx],values_after=[float(clipped[i]) for i in idx],
             float_sha256=float_sha,changed_samples=3,changed_fraction=3/len(y),gain_unchanged=True,
             RMS_before_clip=float(np.sqrt(np.mean(y.astype(float)**2))),RMS_after_clip=float(np.sqrt(np.mean(clipped.astype(float)**2))))
        y=clipped
    elif float(abs(x).max()*gain)>=1:raise ValueError('unapproved clipping; retain original strict gate')
    return y,rms,gain,exception

def pcm_in_memory(x):
    b=io.BytesIO();sf.write(b,x,16000,format='WAV',subtype='PCM_16');b.seek(0)
    q,sr=sf.read(b,dtype='int16');assert sr==16000
    return q

def run(args):
    run=Path(args.run).resolve();p=Path(args.protocol).resolve()
    if p!=run/'protocol.json' or sha(p)!=args.protocol_sha256:raise ValueError('frozen protocol binding')
    cfg=read(p)
    if cfg['target_rms']!=TARGET or cfg['expected_cells']!=80 or cfg['roles']!=['N','R','T1','T2','T3']:
        raise ValueError('fixed current pipeline settings')
    if cfg['clipping_amendment']['indices_zero_based']!=INDICES or cfg['clipping_amendment']['key']!=EXCEPTION_KEY:
        raise ValueError('amendment scope differs')
    for path,h in cfg['source_pins'].items():
        if sha(path)!=h:raise ValueError('source/input/weight pin changed: '+path)
    if str(Path(__file__).resolve()) not in cfg['source_pins']:raise ValueError('adapter source pin required')
    copied=run/'source_attempt01';audit=read(copied/'canonical_preflight/audit.json')
    copy_manifest=read(run/'source_copy_manifest.json');old=Path(copy_manifest['source_run'])
    for row in copy_manifest['files']:
        if sha(run/row['copy_relative_path'])!=row['sha256']:raise ValueError('copied failure/source body changed')
    raw=read(run/'raw_manifest.json')['records'];lookup={(r['sample_id'],r['role']):r for r in raw}
    expected={(r['sample_id'],role) for r in cfg['samples'] for role in cfg['roles']}
    if len(raw)!=80 or set(lookup)!=expected:raise ValueError('all frozen80 raw cells required')
    out=[];reused=0;exceptions=0
    write(run/'receipts/canonical_launch.json',dict(pid=os.getpid(),command=[sys.executable,*sys.argv],source_sha256=sha(__file__),protocol_sha256=sha(p),CPU_only=True,no_training=True))
    for a in audit['records']:
        key=a['key'];r=lookup[a['sample_id'],a['role']]
        float_path=copied/Path(a['float_path']).relative_to(old)
        if sha(float_path)!=a['float_sha256'] or sha(r['path'])!=r['sha256']:raise ValueError('float/raw binding changed')
        x,sr=sf.read(float_path,dtype='float32')
        if sr!=16000 or len(x)!=a['samples']:raise ValueError('frozen full16k clock changed')
        y,rms,gain,exception=normalize(x,key,a['float_sha256']);expected_pcm=pcm_in_memory(y)
        dest=run/'audio'/r['sample_id']/f"{r['role']}_16k_pcm16.wav";dest.parent.mkdir(parents=True,exist_ok=True)
        bridge=None
        if a['existing_canonical'] is not None:
            existing=copied/Path(a['existing_canonical']['path']).relative_to(old)
            if sha(existing)!=a['existing_canonical']['sha256']:raise ValueError('reused PCM file changed')
            q,s=sf.read(existing,dtype='int16')
            if s!=16000 or not np.array_equal(q,expected_pcm):raise ValueError('reused PCM does not equal original fixed-gain serialization')
            with dest.open('xb') as f:f.write(existing.read_bytes())
            if sha(dest)!=sha(existing):raise ValueError('copy byte equivalence failed')
            bridge=dict(original_path=a['existing_canonical']['path'],copied_path=str(existing),sha256=sha(existing),byte_exact=True,PCM_exact=True)
            reused+=1
        else:
            if dest.exists():raise ValueError('refuse overwrite')
            sf.write(dest,y,16000,subtype='PCM_16')
        pcm,s=sf.read(dest,dtype='int16')
        if s!=16000 or len(pcm)!=len(x) or not np.array_equal(pcm,expected_pcm):raise ValueError('PCM readback mismatch')
        actual_rms=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2)))
        if exception:
            exceptions+=1;exception.update(PCM_actual_RMS=actual_rms,PCM_RMS_change_dB=float(20*np.log10(actual_rms/exception['RMS_before_clip'])))
        record=dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],role=r['role'],cohort=cfg['cohort'],path=str(dest),sha256=sha(dest),
             samples=len(pcm),sr=16000,source_path=r['path'],source_sha256=r['sha256'],resampled_float_path=str(float_path),resampled_float_sha256=a['float_sha256'],
             rms_before=rms,gain=gain,rms_after=actual_rms,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
             original63_PCM_bridge=bridge,normalization_exception=exception,no_time_edit=True)
        out.append(record);write(run/'receipts'/f"canonical_{key}.json",record)
        print('CANONICAL',len(out),80,key,'REUSED' if bridge else 'SERIALIZED','EXCEPTION' if exception else 'STRICT',flush=True)
    if len(out)!=80 or reused!=63 or exceptions!=1:raise ValueError('continuation coverage mismatch')
    write(run/'audio_manifest.json',dict(records=out,count=80,protocol_sha256=sha(p),original63_PCM_byte_exact_count=63,three_sample_exception_count=1))
    write(run/'receipts/canonical_completion.json',dict(status='COMPLETE',pid=os.getpid(),count=80,original63_PCM_byte_exact_count=63,three_sample_exception_count=1,no_TTS_generation=True,CPU_only=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    run(parser.parse_args())
