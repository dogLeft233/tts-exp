"""Seal independent speakers, audio, axis and design before any new generation."""
import hashlib,json,shutil,sys,wave
from pathlib import Path
import numpy as np
from core import fit_common,check_disjoint,CONDITIONS

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'runs/p1_gain_dose_response_20261016_attempt02_readback/final'
CAL=ROOT/'runs/recovered_20260930/remote/calibration_20260930'
OUT=Path('C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/run')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'input').mkdir()
    m=json.loads((BASE/'encoder_manifest.json').read_text(encoding='utf8'))
    rows=[]; fit=[]
    for row in m['speakers']:
        sid=row['speaker_id']; arrays=[]; hashes={}
        for cond in ['RAW','DOSE_P3']:
            p=BASE/'encoder'/sid/(cond+'_encoder.npy')
            assert sha(p)==row['conditions'][cond]['encoder_sha256']
            arrays.append(np.load(p,allow_pickle=False)); hashes[cond]=sha(p)
        rows.append((sid,*arrays)); fit.append({'speaker':sid,'hashes':hashes})
    g=fit_common(rows)
    np.save(OUT/'input/common.npy',g,allow_pickle=False)
    # Fresh runtime bridge on the first and last fit speaker; no refitting.
    bridge=[]
    for row in [m['speakers'][0],m['speakers'][-1]]:
        sid=row['speaker_id']; data={}
        for cond in ['RAW','DOSE_P3']:
            p=BASE/'encoder'/sid/(cond+'_mel_chunks.npy')
            assert sha(p)==row['conditions'][cond]['mel_chunks_sha256']
            data[cond+'_chunks']=np.load(p,allow_pickle=False)[:8]
            data[cond+'_z']=np.load(BASE/'encoder'/sid/(cond+'_encoder.npy'),allow_pickle=False)[:8]
        name='bridge_'+sid+'.npz'; np.savez(OUT/'input'/name,**data)
        bridge.append({'file':name,'sha256':sha(OUT/'input'/name)})
    s=json.loads((CAL/'calibration_summary.json').read_text(encoding='utf8'))
    selected={}
    for pid in sorted(s['clips']): selected.setdefault(pid.split('_')[0],pid)
    check_disjoint([r['speaker'] for r in fit],list(selected))
    test=[]
    for sid,pid in sorted(selected.items()):
        wav=ROOT/'runs/recovered_20260930/local/qwen_source_axis/hf_selected/audio'/sid/(pid.split('_',1)[1]+'.wav')
        assert sha(wav)==s['clips'][pid]['natural']['sha256']
        with wave.open(str(wav),'rb') as f:
            assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,16000)
            x=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
        assert max(abs(x))*10**(3/20)<1
        dest=OUT/'input'/(pid+'.wav'); shutil.copyfile(wav,dest)
        test.append({'speaker':sid,'id':pid,'file':dest.name,'sha256':sha(dest),'samples':len(x)})
    face=ROOT/'runs/recovered_20260930/independent_validation/musetalk_native_attempt15_readback_attempt01/extracted_attempt01/inputs/primary_face.png'
    assert sha(face)==s['method']['face']['sha256']
    shutil.copyfile(face,OUT/'input/face.png')
    protocol={'status':'SEALED_BEFORE_NEW_GENERATION','dose_db':3.0,'conditions':list(CONDITIONS),
        'fit_speakers':fit,'test_speakers':test,'speaker_disjoint':True,'test_selection':'first lexicographic CAL clip per speaker S0052-S0061, no endpoint selection',
        'test_history':'previously inspected calibration cohort, held out from axis fit, not never-seen confirmation',
        'common':'g=equal-speaker mean_t(unit(z_plus3)-unit(z_raw)); constant g, no test gain information',
        'residual':'unit(z_plus3)-unit(z_raw)-g; oracle diagnostic, not an orthogonal residual',
        'map':'z_condition=r_raw*unit(u_raw+component), float64 math then float32',
        'orthant':'primary interventions retain negative coordinates; explicit NONNEG sensitivity clips at zero before unitization; no silent clipping',
        'primary':'fixed original natural PCM; shared guard20 video-query support; Sync-C',
        'secondary':['Dmin','D0','B','Cfixed at BASE optimum','FULL support','negative input fraction','angular dose'],
        'replication_claim':'require actual gain positive and common positive paired 95% lower bounds; common-minus-gain and common-minus-FULL_DIRECTION 95% intervals inside [-0.05,0.05]; nonneg sensitivity must agree for practical transfer claim',
        'statistic':'10 speakers, paired bootstrap 20000, seed20261018; descriptive pilot, no best-dose or sample selection',
        'resource_policy':'local RTX4060 only after >=3500MiB free; batch4, torch allocator cap 0.40; between clips recheck >=2000MiB external reserve; >=1GiB disk free; no Ubuntu GPU use',
        'bridge':bridge,'bridge_tolerance':{'relative_l2':1e-4,'max_abs':1e-4},
        'face':s['method']['face'],'fit_manifest_sha256':sha(BASE/'encoder_manifest.json'),
        'common_sha256':sha(OUT/'input/common.npy'),
        'weights':{'wav2lip':'ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8',
                   'syncnet':'961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'},
        'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    pointer=ROOT/'runs/gain_shared_direction_20261018'
    pointer.mkdir(exist_ok=False)
    (pointer/'artifact_location.json').write_text(json.dumps({'path':str(OUT),'protocol_sha256':sha(OUT/'protocol.json')},indent=2),encoding='utf8')
    print(json.dumps({'fit':len(fit),'test':len(test),'videos':len(test)*len(CONDITIONS),'common_norm':float(np.linalg.norm(g)),'output':str(OUT)}))

if __name__=='__main__': main()
