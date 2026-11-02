"""Independent arithmetic/text/PCM readback of posthoc diagnostics; no inference."""
from pathlib import Path
import hashlib
import json
import unicodedata
import wave
import numpy as np
from opencc import OpenCC

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/qwen_tts_decline_audit_20261022_attempt01'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-12


def distance(ref,hyp):
    table=[[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):table[i][0]=i
    for j in range(len(hyp)+1):table[0][j]=j
    for i in range(1,len(ref)+1):
        for j in range(1,len(hyp)+1):table[i][j]=min(table[i-1][j]+1,table[i][j-1]+1,table[i-1][j-1]+int(ref[i-1]!=hyp[j-1]))
    return table[-1][-1]


def main():
    p=read(RUN/'protocol.json');src=p['source_analysis'];assert sha(src['path'])==src['sha256'];a=read(src['path'])
    influence=read(RUN/'influence.json');caps=read(RUN/'posthoc_cap_intervals.json');n=0
    for policy in ['common','native_full','equal_count_spread']:
        rows=influence[policy]['rows']
        for baseline,key in [('N','T_minus_N'),('R','T_minus_R')]:
            deltas=[];modified=[]
            for item,stored in zip(a['records'],rows):
                assert item['sample_id']==stored['sample_id'];c=item['policies'][policy]
                ts=[c[f'T{i}']['C'] for i in range(1,4)];delta=sum(ts)/3-c[baseline]['C'];deltas.append(delta);close(delta,stored[key])
                modified.append((sum(ts[:2])/2 if item['speaker_id']=='S0011' else sum(ts)/3)-c[baseline]['C'])
            s=influence[policy]['summaries'][key];close(np.mean(deltas),s['mean']);close(np.median(deltas),s['median'])
            for idx,loo in enumerate(s['leave_one_out']):close((sum(deltas)-deltas[idx])/9,loo['remaining_mean'])
            t=np.array(modified);rng=np.random.default_rng(20261022);bootstrap=t[rng.integers(0,10,(10000,10))].mean(axis=1)
            cap=caps[policy][key];close(t,cap['values']);close(t.mean(),cap['mean'])
            close(np.quantile(bootstrap,[.025,.975]),cap['CI95']);close(np.quantile(bootstrap,[.0125,.9875]),cap['CI97_5']);n+=1
    assert n==6
    ar=read(RUN/'asr_attempt02/results.json');ab={(r['sample_id'],r['role']):r for r in ar['records']}
    assert len(ab)==len(p['jobs'])==50
    cc=OpenCC('t2s')
    def norm(s):return ''.join(c for c in cc.convert(unicodedata.normalize('NFKC',s).lower()) if not c.isspace() and not unicodedata.category(c).startswith('P'))
    for job in p['jobs']:
        assert sha(job['path'])==job['sha256'];r=ab[job['sample_id'],job['role']];assert r['audio_sha256']==job['sha256']
        with wave.open(job['path'],'rb') as w:
            assert w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==16000
            pcm=w.readframes(w.getnframes())
        assert hashlib.sha256(pcm).hexdigest()==r['pcm_sha256'];assert norm(job['transcript'])==r['reference'] and norm(r['raw'])==r['normalized']
        dist=distance(r['reference'],r['normalized']);assert dist==r['edit_distance'];close(dist/len(r['reference']),r['CER'])
    vad=read(RUN/'cap_vad_sensitivity/results.json')['records'];assert len(vad)==5
    for row in vad:
        close(sum(s['end']-s['start'] for s in row['spans_s']),row['VAD_speech_s'])
        close(sum(max(0,s['end']-max(6,s['start'])) for s in row['spans_s']),row['VAD_after_6s_speech_s'])
        close(distance(norm(row['reference']),norm(row['raw']))/len(norm(row['reference'])),row['CER'])
    counts={}
    for directory in ['qwen_full_tts_extension_20261022_attempt01','qwen_codec_roundtrip_20261022_attempt01']:
        parent=ROOT/'runs'/directory;seal=read(parent/'artifact_manifest.json')
        for record in seal['files']:
            f=parent/record['path'];assert f.stat().st_size==record['bytes'] and sha(f)==record['sha256']
        counts[directory]=len(seal['files'])
    result=dict(status='PASS',ASR50_PCM_and_CER_readback=True,VAD5_timestamps_and_CER=True,all_6_cap_mean_bootstrap_intervals_recomputed=True,
                all_speaker_influence_and_leave_one_out_recomputed=True,unchanged_original_sealed_files=counts,
                source_sha256=sha(__file__),ASR_truth_not_human_validated=True,posthoc_not_primary=True)
    with (RUN/'independent_readback.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
