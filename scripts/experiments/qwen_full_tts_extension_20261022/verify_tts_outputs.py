"""Independent CPU readback of the retained TTS codes, cuts and float audio."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bound(record):
    p=Path(record['path'])
    assert sha(p)==record['sha256'],str(p)
    return p


def main():
    protocol=read(RUN/'protocol.json');tts=read(RUN/'tts/tts_manifest.json')
    assert tts['manifest_sha256']==sha(RUN/'protocol.json')
    assert len(tts['records'])==30 and len(tts['context_controls'])==10
    parent=Path(protocol['parent_run']);binding=read(parent/'execution_bindings.json')
    prior={r['sample_id']:r for r in read(parent/binding['codec_directory']/'results.json')['rows']}
    samples={r['sample_id']:r for r in protocol['samples']};seen=set();eos_absent=[];rounding=[]
    for row in tts['records']:
        sid=row['sample_id'];idx=row['seed_index'];key=(sid,idx)
        assert sid in samples and idx in (1,2,3) and key not in seen
        seen.add(key)
        assert row['speaker_id']==samples[sid]['speaker_id']
        assert row['seed']==protocol['seed_bases'][idx-1]+100*int(row['speaker_id'][1:])
        transcript=''.join(samples[sid]['transcript'].split())
        assert row['target_text']==row['reference_text']==transcript
        ref=np.load(bound(row['reference_codes']),allow_pickle=False)
        new=np.load(bound(row['new_codes']),allow_pickle=False)
        joined=np.load(bound(row['decode_concat_codes']),allow_pickle=False)
        assert np.array_equal(ref,np.load(prior[sid]['codes']['path'],allow_pickle=False))
        assert np.array_equal(joined,np.concatenate((ref,new),axis=0))
        assert row['reference_steps']==len(ref) and row['new_steps']==len(new) and row['total_decode_steps']==len(joined)
        full,sr=sf.read(bound(row['full_decoded']),dtype='float32')
        wave,wr=sf.read(bound(row),dtype='float32')
        assert sr==wr==24000 and full.ndim==wave.ndim==1 and np.isfinite(full).all()
        cut=int(len(ref)/len(joined)*len(full))
        assert cut==row['official_ratio_cut_samples'] and np.array_equal(full[cut:],wave)
        assert len(wave)==row['samples'] and hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==row['float_pcm_sha256']
        assert row['same_concat_repeat_exact']
        sequence=np.load(bound(row['talker_sequences']),allow_pickle=False)
        eos=row['talker_effective_kwargs']['eos_token_id']
        present=bool(np.any(sequence==eos))
        assert present==row['eos_in_talker_sequences']
        if not present:eos_absent.append(key)
        assert row['talker_effective_kwargs']['max_new_tokens']==512
        bound(row['diagnostics']['newcodes_only'])
        rounding.append(cut-len(ref)*1920)
    assert seen=={(sid,idx) for sid in samples for idx in (1,2,3)}
    context_exact=0
    for c in tts['context_controls']:
        full,sr=sf.read(bound(c['full_output']),dtype='float32')
        out,orr=sf.read(bound(c['output']),dtype='float32')
        joined=np.load(bound(c['codes']),allow_pickle=False)
        p=prior[c['sample_id']];ref=np.load(p['codes']['path'],allow_pickle=False)
        assert np.array_equal(joined,np.concatenate((ref,ref),axis=0))
        cut=int(len(ref)/len(joined)*len(full))
        assert sr==orr==24000 and cut==c['official_ratio_cut_samples'] and np.array_equal(full[cut:],out)
        raw,rr=sf.read(p['output']['path'],dtype='float32');common=min(len(out),len(raw))
        exact=np.array_equal(out,raw);mae=float(np.abs(out[:common].astype(float)-raw[:common].astype(float)).mean())
        assert exact==c['context_vs_parent_R']['array_exact']
        assert abs(mae-c['context_vs_parent_R']['overlap_mae'])<1e-15
        context_exact+=int(exact)
    bridge=read(RUN/'codec_bridge/bridge_results.json')
    assert bridge['status']=='PASS' and len(bridge['records'])==10
    for item in bridge['records']:
        assert sha(item['parent_R_path'])==item['parent_R_file_sha256']
        assert sha(item['new_path'])==item['new_file_sha256']
        before,bs=sf.read(item['parent_R_path'],dtype='float32')
        after,ass=sf.read(item['new_path'],dtype='float32')
        assert bs==ass==24000 and np.array_equal(before,after)
    result=dict(status='PASS',count_TTS=30,count_context_diagnostics=10,context_R_exact_count=context_exact,
                parent_R_redecode_exact_count=10,
                eos_absent=eos_absent,cut_rounding_difference_samples=rounding,
                checked=['planned cohort and all seeds','reference codes equal parent','concat codes','official cuts and full FLOAT PCM exact',
                         'EOS from retained token sequence','context diagnostic waveform and MAE','bound payload hashes','10 parent R redecodes exact'],
                no_new_forward=True,source_sha256=sha(__file__),tts_manifest_sha256=sha(RUN/'tts/tts_manifest.json'))
    with (RUN/'independent_tts_readback.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
