"""Independent CPU reconstruction of known-noise mixtures, moments and contrasts."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main(out):
    protocol=json.loads((out/'protocol.json').read_text());complete=json.loads((out/'completion.json').read_text())
    assert complete['status']=='COMPLETED' and complete['forwards']==449
    assert sha(out/'protocol.json')==complete['protocol_sha256']
    rows=json.loads((out/'noise_response_rows.json').read_text());calls=json.loads((out/'forward_calls.json').read_text())
    controls=json.loads((out/'controls.json').read_text());assert len(calls)==449
    assert len([r for r in controls if r['kind']=='epsilon_compensation_all_stages_exact' and r['passed']])==84
    assert len([r for r in controls if r['kind']=='repeatA_all_stages_exact' and r['passed']])==1
    sources={};noises={};powers={};label_cache={};thresholds={};checked_samples=0;checked_pairs=0;maximum_error=0.;input_hash_checks=0
    for pin in protocol['input_pins']:
        s=np.load(out/f"{pin['split']}_native.npy")
        assert sha(pin['source_audio'])==pin['source_sha256']
        raw,sr=sf.read(pin['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);expected=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert np.array_equal(s,expected) and digest(s)==pin['native_sha256']
        sources[pin['split']]=s
        powers[pin['split']]=np.r_[0.,np.cumsum(np.pad(s.astype(float)**2,(400,400)))]
        centers=np.arange((len(s)-400)//320+1)*320+199
        energy=np.sqrt((powers[pin['split']][centers+800]-powers[pin['split']][centers])/800)
        thresholds[pin['split']]=np.quantile(energy,[.2,.8])
        assert np.array_equal(thresholds[pin['split']],pin['thresholds'])
    for pin in protocol['noise_pins']:
        path=out/pin['path'];assert sha(path)==pin['sha256'];noise=np.load(path)
        rng=np.random.default_rng(pin['generator_seed']);v=rng.standard_normal(pin['samples'])
        if pin['kind']=='pink':
            f=np.fft.rfft(v);power=np.arange(len(f),dtype=float);power[0]=1
            f=f/np.sqrt(power);f[0]=0;v=np.fft.irfft(f,n=len(v))
        v=v-v.mean();v=v/np.sqrt((v*v).mean());unit=v.astype(np.float32)
        expected=(unit*np.float32(pin['native_rms']*10**(-30/20))).astype(np.float32)
        assert np.array_equal(expected,noise)
        measured=20*np.log10(np.sqrt(np.mean(sources[pin['split']].astype(float)**2))/np.sqrt(np.mean(noise.astype(float)**2)))
        assert abs(measured-30)<1e-5 and abs(measured-pin['added_noise_snr_db'])<1e-12
        noises[(pin['split'],pin['kind'],pin['seed'])]=noise
    for call in calls:
        s=sources[call['split']];start=call['chunk']*320000;end=min(start+320080,len(s));x=s[start:end]
        if call['label']=='anchor1':y=x
        elif call['label']=='anchorhalf':y=x*.5
        else:
            case,arm=call['label'].split('/');kind,seed=case.split('_');n=noises[(call['split'],kind,int(seed))][start:end]
            if arm in ['A','repeatA']:y=x+n
            elif arm in ['B','B_compensated']:y=(x+n)*.5
            elif arm=='C':y=x+n*.5
            elif arm=='D':y=x*.5+n
            else:raise ValueError(arm)
        assert digest(y)==call['input_sha256'];input_hash_checks+=1
    row_map={(r['split'],r['chunk'],r['kind'],r['seed'],r['arm'],r['stage'],r['stratum']):r for r in rows}
    for split in ['train','val','test']:
        for folder in sorted((out/split).glob('chunk*')):
            chunk=int(folder.name[5:])
            for kind in protocol['noise_types']:
                for seed in protocol['seeds']:
                    case=folder/f'{kind}_{seed}'
                    with np.load(case/'A/samples.npz') as aa,np.load(case/'B_compensated/samples.npz') as cc:
                        for stage in protocol['stages']:assert np.array_equal(aa[stage],cc[stage])
                    assert np.array_equal(np.load(case/'A/last.npy'),np.load(case/'B_compensated/last.npy'))
                    for arm in ['A','B','C','D']:
                        anchor=folder/('anchor1' if arm in ['A','C'] else 'anchorhalf')
                        with np.load(anchor/'samples.npz') as base,np.load(case/arm/'samples.npz') as changed,\
                             np.load(case/arm/'frame_moments.npz') as fm,np.load(case/arm/'channel_moments.npz') as cm:
                            for stage in protocol['stages']:
                                m=fm[stage];ch=cm[stage];idx=base[stage+'__indices']
                                a=base[stage].astype(float);b=changed[stage].astype(float);delta=b-a
                                expected=np.stack([(a*a).sum(1),(b*b).sum(1),(delta*delta).sum(1),
                                                   (a*b).sum(1),abs(delta).max(1)],axis=1)
                                assert np.allclose(expected,m[idx],rtol=2e-13,atol=2e-12)
                                checked_samples+=len(idx);checked_pairs+=1
                                for column in [0,1,3]:
                                    target={0:2,1:3,3:4}[column]
                                    assert np.allclose(m[:,column].sum(),ch[:,target].sum(),rtol=1e-12)
                                labels=fm[stage+'__stratum']
                                label_key=(split,chunk,stage)
                                if label_key not in label_cache:
                                    centers=chunk*320000+np.arange(len(m))*(5 if stage=='first_gn' else 320)+(4 if stage=='first_gn' else 199)
                                    power=powers[split];energy=np.sqrt((power[centers+800]-power[centers])/800)
                                    lo,hi=thresholds[split]
                                    label_cache[label_key]=np.where(energy<=lo,0,np.where(energy>hi,2,1))
                                assert np.array_equal(labels,label_cache[label_key])
                                for value,stratum in [(-1,'all'),(0,'low20'),(1,'middle60'),(2,'high20')]:
                                    selected=m if value==-1 else m[labels==value]
                                    if not len(selected):continue
                                    r=row_map[(split,chunk,kind,seed,arm,stage,stratum)]
                                    xx,yy,dd,xy=selected[:,:4].sum(0);relative=np.sqrt(dd/xx)
                                    valid=(selected[:,0]>0)&(selected[:,1]>0)
                                    cos=np.clip(selected[valid,3]/np.sqrt(selected[valid,0]*selected[valid,1]),-1,1)
                                    direction=np.sqrt(np.maximum(0,2-2*cos).mean())
                                    maximum_error=max(maximum_error,abs(relative-r['relative_noise_change']),abs(direction-r['unit_noise_change']))
                                    assert np.allclose([relative,direction],[r['relative_noise_change'],r['unit_noise_change']],atol=1e-13,rtol=1e-12)
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ['split','kind','seed','arm','stage','stratum'])
        groups.setdefault(key,[]).append(row)
    aggregated=[]
    for key,group in groups.items():
        xx=sum(r['xx'] for r in group);dd=sum(r['dd'] for r in group)
        unit=sum(r['unit_delta_square_sum'] for r in group);frames=sum(r['unit_valid_frames'] for r in group)
        aggregated.append(dict(zip(['split','kind','seed','arm','stage','stratum'],key),
            chunks=len(group),frames=frames,relative_noise_change=float(np.sqrt(dd/xx)),
            unit_noise_change=float(np.sqrt(unit/frames))))
    contrasts=[]
    for split in ['train','val','test']:
        for kind in protocol['noise_types']:
            for seed in protocol['seeds']:
                for stage in protocol['stages']:
                    for stratum in ['all','low20','middle60','high20']:
                        selected={r['arm']:r for r in aggregated if (r['split'],r['kind'],r['seed'],r['stage'],r['stratum'])==
                                  (split,kind,seed,stage,stratum)}
                        for numerator,denominator,label in [('B','A','whole_scale_half'),('C','A','only_noise_half'),('D','B','restore_noise_at_half_content')]:
                            a,b=selected[numerator],selected[denominator]
                            contrasts.append(dict(split=split,kind=kind,seed=seed,stage=stage,stratum=stratum,contrast=label,
                                unit_response_ratio=a['unit_noise_change']/b['unit_noise_change'],
                                relative_l2_response_ratio=a['relative_noise_change']/b['relative_noise_change']))
    summary=[]
    for split in ['train','val','test']:
        for kind in protocol['noise_types']:
            for label in ['whole_scale_half','only_noise_half','restore_noise_at_half_content']:
                selected=[r for r in contrasts if (r['split'],r['kind'],r['stage'],r['stratum'],r['contrast'])==
                          (split,kind,'last','all',label)]
                values=[r['unit_response_ratio'] for r in selected]
                summary.append(dict(split=split,kind=kind,contrast=label,seeds=len(values),
                    median_unit_response_ratio=float(np.median(values)),min_ratio=min(values),max_ratio=max(values)))
    for name,values in [('aggregate',aggregated),('contrasts',contrasts),('summary',summary)]:
        write(out/f'{name}.json',values)
        with (out/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    qa=dict(passed=True,generated_noise_arrays_exact=18,calibrated_snr_db=30,input_hash_checks=input_hash_checks,
        stage_pairs=checked_pairs,raw_sample_vector_pairs=checked_samples,maximum_metric_rebuild_error=maximum_error,
        compensated_noise_cases_exact=84,training_updates=0,new_gpu_calls=0,
        note='Consumer does not import producer; original May is not a noise-free reference')
    write(out/'independent_qa.json',qa)
    print(json.dumps(qa,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
