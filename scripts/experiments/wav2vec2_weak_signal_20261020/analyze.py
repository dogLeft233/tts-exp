"""CPU independent moments/sample readback and stratified aggregation."""
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np
import soundfile as sf


def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def aggregate(rows,keys):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in keys);groups.setdefault(key,[]).append(row)
    output=[]
    for key,group in groups.items():
        xx=sum(r['xx'] for r in group);yy=sum(r['yy'] for r in group)
        dd=sum(r['dd'] for r in group);xy=sum(r['xy'] for r in group)
        output.append(dict(zip(keys,key),chunks=len(group),frames=sum(r['frames'] for r in group),
            norm_ratio=float(np.sqrt(yy/xx)),gain_fit=xy/xx,relative_l2=float(np.sqrt(dd/xx)),
            shape_residual=float(np.sqrt(max(0,yy-xy*xy/xx)/xx)),xx=xx,yy=yy,dd=dd,xy=xy))
    return output


def main(root):
    completion=json.loads((root/'completion.json').read_text());assert completion['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'layer_rows.json').read_text())
    channel_rows=json.loads((root/'channel_rows.json').read_text())
    controls=json.loads((root/'controls.json').read_text())
    calls=json.loads((root/'forward_calls.json').read_text())
    pins=json.loads((root/'input_pins.json').read_text())
    assert completion['forwards']==len(calls)==44 and completion['chunks']==14
    assert len([c for c in controls if c['kind']=='epsilon_compensated_all_postGN_stages_exact' and c['passed']])==14
    assert len([c for c in controls if c['kind']=='repeat_all_stages_exact' and c['passed']])==2
    errors=[];sample_rows=0;checked_channels=0;stage_pairs=0;gnratios={};gnfit=[]
    for split in ['train','val','test']:
        z=np.load(root/f'{split}_native_input.npy')
        pin=next(p for p in pins if p['split']==split)
        assert hashlib.sha256(Path(pin['source_audio']).read_bytes()).hexdigest()==pin['source_sha256']
        raw,sr=sf.read(pin['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);reconstructed=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert np.array_equal(z,reconstructed)
        assert hashlib.sha256(z.tobytes()).hexdigest()==pin['native_sha256']
        assert hashlib.sha256((z*.5).tobytes()).hexdigest()==pin['half_sha256']
        power=np.r_[0.,np.cumsum(np.pad(z.astype(float)**2,(400,400)))]
        refcenters=np.arange((len(z)-400)//320+1)*320+199
        refenergy=np.sqrt((power[refcenters+800]-power[refcenters])/800)
        thresholds=np.quantile(refenergy,[.2,.8]);assert np.array_equal(thresholds,pin['thresholds'])
        gndata=[]
        for folder in sorted((root/split).glob('chunk*')):
            chunk=int(folder.name[5:]);meta=json.loads((folder/'stage_metadata.json').read_text())
            with np.load(folder/'paired_frame_moments.npz') as fm,np.load(folder/'paired_channel_moments.npz') as cm,\
                 np.load(folder/'baseline/stage_samples.npz') as bs,np.load(folder/'half/stage_samples.npz') as ls,\
                 np.load(folder/'half_epsilon_compensated/stage_samples.npz') as cs:
                for name,description in meta.items():
                    stage_pairs+=1;frames=fm[name];ch=cm[name];n,channels=description['shape']
                    assert frames.shape==(n,5) and ch.shape==(channels,5)
                    idx=bs[name+'__indices'];a=bs[name].astype(float);b=ls[name].astype(float);d=b-a
                    expected=np.stack([np.einsum('ij,ij->i',a,a),np.einsum('ij,ij->i',b,b),
                        np.einsum('ij,ij->i',d,d),np.einsum('ij,ij->i',a,b),np.max(abs(d),axis=1)],axis=1)
                    assert np.allclose(expected,frames[idx],rtol=2e-13,atol=2e-12)
                    sample_rows+=len(idx)
                    assert np.allclose(frames[:,0].sum(),ch[:,2].sum(),rtol=1e-12)
                    assert np.allclose(frames[:,1].sum(),ch[:,3].sum(),rtol=1e-12)
                    assert np.allclose(frames[:,3].sum(),ch[:,4].sum(),rtol=1e-12)
                    if name=='conv0_linear':assert np.array_equal(ls[name],bs[name]*.5)
                    else:assert np.array_equal(bs[name],cs[name])
                    labels=fm[name+'__acoustic_stratum'];centers=fm[name+'__source_sample_centers']
                    assert len(labels)==len(centers)==n and set(np.unique(labels)).issubset({0,1,2})
                    energy=np.sqrt((power[centers+800]-power[centers])/800)
                    assert np.array_equal(labels,np.where(energy<=thresholds[0],0,np.where(energy>thresholds[1],2,1)))
                    for key,label in [(-1,'all'),(0,'low20'),(1,'middle60'),(2,'high20')]:
                        m=frames if key==-1 else frames[labels==key]
                        if not len(m):continue
                        row=next(r for r in rows if (r['split'],r['chunk'],r['stage'],r['stratum'])==(split,chunk,name,label))
                        xx,yy,dd,xy=m[:,:4].sum(0)
                        assert np.allclose([row['xx'],row['yy'],row['dd'],row['xy']],[xx,yy,dd,xy],rtol=1e-13)
                        error=abs(row['norm_ratio']-np.sqrt(yy/xx));errors.append(error);assert error<1e-12
                    vx=ch[:,2]/n-(ch[:,0]/n)**2;vy=ch[:,3]/n-(ch[:,1]/n)**2
                    q20,q80=np.quantile(vx,[.2,.8]);cbin=np.where(vx<=q20,0,np.where(vx>q80,2,1))
                    assert np.array_equal(cbin,cm[name+'__baseline_response_stratum'])
                    checked_channels+=channels
                    for key,label in enumerate(['low20','middle60','high20']):
                        use=cbin==key
                        row=next(r for r in channel_rows if (r['split'],r['chunk'],r['stage'],r['stratum'])==(split,chunk,name,label))
                        assert abs(row['centered_norm_ratio']-np.sqrt(vy[use].sum()/vx[use].sum()))<1e-11
            with np.load(folder/'gn_channels.npz') as gn:
                ratio=np.sqrt(gn['half_centered_energy']/gn['base_centered_energy'])
                predicted=np.sqrt((gn['variance']+1e-5)/(gn['variance']+4e-5))
                assert np.array_equal(ratio,gn['observed_ratio'])
                assert np.array_equal(predicted,gn['predicted_ratio'])
                gndata.append({k:gn[k].copy() for k in gn.files})
                gnfit.append(dict(split=split,chunk=chunk,max_affine_fit_error=float(gn['per_channel_fit_maxerror'].max()),
                    max_ratio_formula_error=float(abs(ratio-predicted).max())))
            b=np.load(folder/'baseline/last_hidden_state.npy');c=np.load(folder/'half_epsilon_compensated/last_hidden_state.npy')
            assert np.array_equal(b,c)
        gnratios[split]=gndata
    layer=aggregate(rows,['split','stage','stratum'])
    cent=[]
    for split in ['train','val','test']:
        for stage in dict.fromkeys(r['stage'] for r in channel_rows):
            for label in ['low20','middle60','high20']:
                r=[r for r in channel_rows if (r['split'],r['stage'],r['stratum'])==(split,stage,label)]
                xx=sum(t['centered_xx'] for t in r);yy=sum(t['centered_yy'] for t in r)
                cent.append(dict(split=split,stage=stage,stratum=label,centered_norm_ratio=float(np.sqrt(yy/xx)),
                    centered_xx=xx,centered_yy=yy,chunk_channel_members=sum(t['channels'] for t in r)))
    gnsummary=[]
    for split,data in gnratios.items():
        for index,label in enumerate(['low20','middle60','high20']):
            ratios=[];vlist=[]
            for d in data:
                v=d['variance'];lo,hi=np.quantile(v,[.2,.8]);g=np.where(v<=lo,0,np.where(v>hi,2,1));use=g==index
                ratios.extend(d['observed_ratio'][use]);vlist.extend(v[use])
            gnsummary.append(dict(split=split,stratum=label,chunk_channel_members=len(ratios),
                median_ratio=float(np.median(ratios)),min_ratio=float(min(ratios)),max_ratio=float(max(ratios)),
                median_attenuation_pct=float((1-np.median(ratios))*100),
                min_variance=float(min(vlist)),max_variance=float(max(vlist))))
    result=dict(layer=layer,channel_centered=cent,first_GN_prevariance=gnsummary,gn_formula_fit=gnfit,
        note='Norm ratios measure representation response, not isolated noise; strata fixed using baseline only')
    result['cache_aligned_endpoint']=[]
    for split in ['train','val','test']:
        a=np.concatenate([np.load(p/'baseline/last_hidden_state.npy') for p in sorted((root/split).glob('chunk*'))])
        b=np.concatenate([np.load(p/'half/last_hidden_state.npy') for p in sorted((root/split).glob('chunk*'))])
        original=np.load(Path(protocol['parent'])/f'features/wav2vec2_base/{split}_s1.00.npy').reshape(-1,768)
        low=np.load(Path(protocol['parent'])/f'features/wav2vec2_base/{split}_s0.50.npy').reshape(-1,768)
        assert np.array_equal(a[:len(original)],original) and np.array_equal(b[:len(low)],low)
        da=a[:len(original)].astype(float);db=b[:len(original)].astype(float)
        result['cache_aligned_endpoint'].append(dict(split=split,retained_frames=len(original),
            dropped_encoder_tail_frames=len(a)-len(original),cache_exact=True,
            relative_l2=float(np.linalg.norm(db-da)/np.linalg.norm(da))))
    write(root/'analysis.json',result)
    for name,values in [('layer_summary',layer),('channel_summary',cent),('gn_variance_summary',gnsummary)]:
        with (root/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    qa=dict(passed=True,new_training_updates=0,new_gpu_calls=0,stage_pairs=stage_pairs,
        raw_sample_vectors=sample_rows,checked_channel_moment_rows=checked_channels,
        norm_ratio_reconstruction_max_error=max(errors),compensated_14_full_endpoint_arrays_exact=True,
        compensated_all_14_all_stage_sample_vectors_exact=True,
        max_GN_observed_ratio_formula_error=max(r['max_ratio_formula_error'] for r in gnfit),
        max_GN_temporal_constant_affine_fit_error=max(r['max_affine_fit_error'] for r in gnfit))
    write(root/'independent_qa.json',qa)
    print(json.dumps(qa,indent=2));print(json.dumps(gnsummary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
