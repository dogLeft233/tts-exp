"""Independent CPU readback of all30 new codec and90 full-TTS raw payloads."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(r):
    assert sha(r['path'])==r['sha256'],r['path']
    return Path(r['path'])


def main():
    p=read(RUN/'protocol.json');generation=RUN/'generation';samples={s['sample_id']:s for s in p['samples'] if s['cohort']=='new30'}
    rrows=read(generation/'codec_manifest.json')['records'];trows=read(generation/'tts_manifest.json')['records']
    assert len(rrows)==30 and len(trows)==90
    rmap={};seen=set();missing_eos=[]
    for r in rrows:
        sid=r['sample_id'];assert sid in samples and sid not in rmap;rmap[sid]=r
        assert r['input24_sha256']==samples[sid]['input24_sha256'] and sha(r['input24_path'])==r['input24_sha256']
        code=np.load(bound(r['codes']),allow_pickle=False);wave,sr=sf.read(bound(r),dtype='float32')
        assert code.ndim==2 and code.shape[1]==16 and code.dtype.kind in 'iu' and code.min()>=0
        assert code[:,0].max()<4096 and code[:,1:].max()<2048 and sr==24000 and np.isfinite(wave).all()
        assert len(wave)==len(code)*1920==r['samples']
        assert hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==r['float_pcm_sha256']
    for r in trows:
        sid=r['sample_id'];idx=r['seed_index'];assert sid in samples and idx in (1,2,3) and (sid,idx) not in seen
        seen.add((sid,idx));assert r['seed']==p['seed_bases'][idx-1]+100*int(samples[sid]['speaker_id'][1:])
        target=''.join(samples[sid]['transcript'].split());assert r['target_text']==r['reference_text']==target
        ref=np.load(bound(r['reference_codes']),allow_pickle=False);new=np.load(bound(r['new_codes']),allow_pickle=False)
        joined=np.load(bound(r['decode_concat_codes']),allow_pickle=False);assert np.array_equal(ref,np.load(rmap[sid]['codes']['path'],allow_pickle=False))
        assert np.array_equal(joined,np.concatenate([ref,new])) and new.ndim==2 and new.shape[1]==16
        full,sr=sf.read(bound(r['full_decoded']),dtype='float32');wave,wsr=sf.read(bound(r),dtype='float32')
        assert sr==wsr==24000 and np.isfinite(full).all() and full.ndim==wave.ndim==1
        cut=int(len(ref)/len(joined)*len(full));assert cut==r['official_ratio_cut_samples'] and np.array_equal(full[cut:],wave)
        assert len(wave)==r['samples'] and hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==r['float_pcm_sha256']
        seq=np.load(bound(r['talker_sequences']),allow_pickle=False);present=bool(np.any(seq==r['talker_effective_kwargs']['eos_token_id']))
        assert present==r['eos_in_talker_sequences'] and r['same_concat_repeat_exact']
        assert r['talker_effective_kwargs']['max_new_tokens']==512
        if not present:missing_eos.append(dict(sample_id=sid,seed_index=idx,duration_s=len(wave)/24000))
    assert seen=={(sid,i) for sid in samples for i in (1,2,3)}
    bridge=read(generation/'reference_bridge.json');assert bridge['status']=='PASS' and bridge['encode_codes_exact'] and bridge['raw_R_float_exact']
    old=bridge['parent_output'];assert sha(old['path'])==old['file_sha256']
    before,sr=sf.read(old['path'],dtype='float32');after,rs=sf.read(bound(bridge['output']),dtype='float32');assert sr==rs and np.array_equal(before,after)
    before=np.load(bridge['parent_codes']['path'],allow_pickle=False);after=np.load(bound(bridge['codes']),allow_pickle=False);assert np.array_equal(before,after)
    result=dict(status='PASS',new_R_count=30,new_TTS_count=90,all_fixed_sources_and_seeds_preserved=True,version_bridge_exact=True,
                all_codes_full_FLOAT_official_cuts_PCM_readback=True,nonEOS=missing_eos,no_model_forward=True,source_sha256=sha(__file__))
    with (RUN/'independent_generation_readback.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
