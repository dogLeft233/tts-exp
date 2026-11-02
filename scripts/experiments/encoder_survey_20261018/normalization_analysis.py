"""Data-only paired analysis and direct last-frame reconstruction for norm ablation."""
from pathlib import Path
import json,sys,csv,hashlib
OLD=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
OUT=Path('F:/Datasets/tts-exp/encoder_normalization_ablation_20261019')
sys.path.insert(0,str(OLD/'deps'))
import numpy as np
from run import unit,geometry_summary,write
from prepare import sha

def summarize(records):
    stat=geometry_summary(records)
    vectors=np.asarray(stat['speaker_deltas']);local=[]
    for si,sp in enumerate(stat['speaker_ids']):
        axis=unit(np.delete(vectors,si,axis=0).mean(0));sent=[]
        for r in records:
            if r['speaker']!=sp:continue
            d=r['tokens'];valid=np.linalg.norm(d,axis=1)>1e-12
            if valid.any():sent.append(float(np.mean(unit(d[valid])@axis)))
        if sent:local.append(float(np.mean(sent)))
    stat['local_loso_cos_mean']=float(np.mean(local)) if local else None
    del stat['speaker_deltas']
    return stat

def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    oldres=json.loads((OLD/'analysis/results.json').read_text(encoding='utf8'))
    assert sha(OLD/'analysis/results.json')==p['old_results_sha256']
    rows=[];checks=[];maxnorm=0.;maxframe=0.;maxphone=0.;files=[]
    for s in p['models']:
        mid=s['id'];root=OUT/'models'/mid;oldroot=OLD/'models'/mid
        if not (root/'receipt.json').exists() or json.loads((root/'receipt.json').read_text(encoding='utf8'))['status']!='COMPLETED':
            root=OUT/'recovery_attempt03/models'/mid
        r=json.loads((root/'receipt.json').read_text(encoding='utf8'))
        assert r['status']=='COMPLETED' and r['forward_count']==64 and len(r['clips'])==20
        assert r['source_sha256']==p['producer_sha256'] and r['protocol_sha256']==sha(OUT/'protocol.json')
        assert sha(oldroot/'receipt.json')==s['old_receipt_sha256']
        assert all(q['maxabs']<=1e-5 and q['relative_l2']<=1e-6 for q in r['normalized_replay'])
        assert not r['loading_info']['missing_keys'] and not r['loading_info']['mismatched_keys']
        oldreceipt=json.loads((oldroot/'receipt.json').read_text(encoding='utf8'))
        assert r['original_processor']==oldreceipt['preprocessor']
        stats=json.loads((root/'clip_statistics.json').read_text(encoding='utf8'))
        data={};olddata={};selected=sorted({0,1,2,r['layer_count']//2,r['layer_count']-1})
        for row in p['rows']:
            rec=next(c for c in r['clips'] if c['id']==row['id'])
            f=root/rec['file'];assert sha(f)==rec['sha256'];files.append(str(f))
            with np.load(f,allow_pickle=False) as z:data[row['id']]={k:z[k] for k in z.files}
            oldrec=next(c for c in oldreceipt['clips'] if c['id']==row['id'])
            f=oldroot/oldrec['file'];assert sha(f)==oldrec['sha256']
            with np.load(f,allow_pickle=False) as z:olddata[row['id']]={k:z[k] for k in z.files if '_directions_' in k or '_phone_indices' in k}
            d=data[row['id']]
            for key,x in d.items():
                assert np.isfinite(x).all()
                if '_directions_' in key:
                    error=float(abs(np.linalg.norm(x.astype(float),axis=-1)-1).max());maxnorm=max(maxnorm,error);assert error<2e-7
            for ci,arm in enumerate(p['arms']):
                assert rec['conditions'][ci]['samples']==row['samples'][arm]
                assert rec['conditions'][ci]['pcm_sha256']==row['pcm_sha256'][arm]
                if s['kind']=='wave':
                    assert rec['conditions'][ci]['processor']['input_values']['sha256']==row['pcm_sha256'][arm]
                assert d['last_frames_'+str(ci)].shape[0]==len(d['times_'+str(ci)])
                assert np.array_equal(d['times_0'],d['times_'+str(ci)])
            li=r['layer_count']-1
            us=[unit(d['last_frames_'+str(ci)]) for ci in range(3)]
            for ep,ca,cb in [('GAIN_P3',0,1),('GAIN_M3_SIGNED',2,0)]:
                chord=float(np.sqrt(np.mean(np.sum((us[cb]-us[ca])**2,axis=1))))
                expected=next(t['chord_rms'] for t in stats if (t['id'],t['domain'],t['layer'],t['endpoint'])==(row['id'],'full_phone',li,ep))
                error=abs(chord-expected);maxframe=max(maxframe,error);assert error<1e-12
            # Independent phone pool reconstruction from saved actual last-layer frames.
            phones={ph['index']:ph for ph in row['phones']}
            for domain in ['full_phone','inner50']:
                expected=next(c['common_phone_indices'] for c in p['coverage'][domain] if c['id']==row['id'])
                assert d[domain+'_phone_indices'].tolist()==expected
                for ci in range(3):
                    reconstructed=[]
                    for idx in expected:
                        lo,hi=phones[idx]['natural']
                        if domain=='inner50':lo,hi=lo+.25*(hi-lo),hi-.25*(hi-lo)
                        take=(d['times_'+str(ci)]>=lo)&(d['times_'+str(ci)]<hi)
                        reconstructed.append(unit(us[ci][take].mean(0)))
                    err=float(abs(np.asarray(reconstructed)-d[f'{domain}_directions_l{li}'][ci]).max())
                    maxphone=max(maxphone,err);assert err<8e-8
        for domain in ['full_phone','inner50']:
            for li in selected:
                for ep,ca,cb,oldca,oldcb in [('GAIN_P3',0,1,2,4),('GAIN_M3_SIGNED',2,0,5,2)]:
                    records=[];oldrecords=[];baselinecos=[]
                    for row in p['rows']:
                        d=data[row['id']];od=olddata[row['id']]
                        expected=d[domain+'_phone_indices'].tolist();oid=od[domain+'_phone_indices'].tolist()
                        take=[oid.index(i) for i in expected]
                        a=d[f'{domain}_directions_l{li}'].astype(float)
                        b=od[f'{domain}_directions_l{li}'][:,take,:].astype(float)
                        delta=a[cb]-a[ca];oldelta=b[oldcb]-b[oldca]
                        records.append(dict(id=row['id'],speaker=row['speaker'],delta=delta.mean(0),tokens=delta))
                        oldrecords.append(dict(id=row['id'],speaker=row['speaker'],delta=oldelta.mean(0),tokens=oldelta))
                        baselinecos.append(float(np.mean(np.sum(unit(a[0])*unit(b[2]),axis=-1))))
                    off=summarize(records);on=summarize(oldrecords)
                    old=next(t for t in oldres['results'] if (t['model'],t['domain'],t['layer'],t['endpoint'])==(mid,domain,li,ep))
                    assert abs(on['pairwise_cos_mean']-old['pairwise_cos_mean'])<1e-10
                    assert abs(on['loso_cos_mean']-old['loso_cos_mean'])<1e-10
                    ns=[t for t in stats if (t['domain'],t['layer'],t['endpoint'])==(domain,li,ep)]
                    assert len(ns)==20
                    chord=float(np.mean([t['chord_rms'] for t in ns]));repeat=r['repeat_chord_rms'][li]
                    off.update(mean_chord_rms_native=chord,repeat_chord_rms=repeat,
                        direction_resolved=bool(chord>=1e-5 and chord>10*repeat),
                        mean_radius_ratio=float(np.mean([t['radius_ratio'] for t in ns])),
                        mean_phone_angle_deg=float(np.mean([t['mean_phone_angle_deg'] for t in ns])))
                    on.update(mean_chord_rms_native=old['mean_chord_rms_native'],direction_resolved=old['direction_resolved'],
                        mean_radius_ratio=old['native_radius_ratio'],repeat_chord_rms=old['repeat_chord_rms'])
                    # A second, direct pairwise computation verifies aggregation of new saved vectors.
                    sp=sorted({t['speaker'] for t in records});v=np.stack([np.mean([t['delta'] for t in records if t['speaker']==w],0) for w in sp])
                    pair=[float(np.dot(unit(v[i]),unit(v[j]))) for i in range(10) for j in range(i)]
                    assert abs(float(np.mean(pair))-off['pairwise_cos_mean'])<1e-12
                    rows.append(dict(model=mid,domain=domain,layer=li,layer_fraction=li/(r['layer_count']-1),endpoint=ep,
                        normalize_on=on,normalize_off=off,off_on_fixed_phone_cos=float(np.mean(baselinecos)),
                        common_phone_count=sum(c['common_phones'] for c in p['coverage'][domain])))
        checks.append(dict(model=mid,root=str(root),receipt_sha256=sha(root/'receipt.json'),forwards=r['forward_count'],
            normalized_replay=r['normalized_replay'],peak_allocated=r['peak_allocated'],peak_reserved=r['peak_reserved']))
        print('VERIFIED_SCIENCE',mid,flush=True)
        del data,olddata
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=False)
    (analysis/'normalization_analysis.py').write_bytes(Path(__file__).read_bytes())
    write(analysis/'results.json',dict(status='COMPLETE',protocol_sha256=sha(OUT/'protocol.json'),
        old_results_sha256=sha(OLD/'analysis/results.json'),results=rows,controls=checks))
    qa=dict(status='PASS_DATA_ONLY',models=8,new_forwards=512,files=len(files),
        max_direction_norm_error=maxnorm,max_last_frame_chord_error=maxframe,max_last_phone_reconstruction_error=maxphone,
        geometry_rows=len(rows),source_sha256=sha(__file__),results_sha256=sha(analysis/'results.json'))
    write(analysis/'qa.json',qa);print(json.dumps(qa),flush=True)
    last=[t for t in rows if t['domain']=='full_phone' and t['layer_fraction']==1 and t['endpoint']=='GAIN_P3']
    table=[]
    for t in last:
        on=t['normalize_on'];off=t['normalize_off']
        table.append(dict(model=t['model'],on_chord=on['mean_chord_rms_native'],off_chord=off['mean_chord_rms_native'],
            on_resolved=on['direction_resolved'],off_resolved=off['direction_resolved'],
            on_paircos=on['pairwise_cos_mean'],off_paircos=off['pairwise_cos_mean'],off_loso=off['loso_cos_mean'],
            off_localcos=off['local_loso_cos_mean'],off_on_fixed_cos=t['off_on_fixed_phone_cos']))
    write(analysis/'last_table.json',table)
    with (analysis/'table.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(table,indent=2))
    sys.path.insert(0,str(OLD/'plotdeps'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(8)
    for offset,key,label,color in [(-.18,'normalize_on','Native normalization','#377eb8'),(.18,'normalize_off','Input normalization off','#e66101')]:
        axes[0].bar(x+offset,[t[key]['mean_chord_rms_native'] for t in last],.35,label=label,color=color)
        axes[1].bar(x+offset,[t[key]['pairwise_cos_mean'] if t[key]['direction_resolved'] else np.nan for t in last],.35,color=color)
    axes[0].set_yscale('log');axes[0].axhline(1e-5,ls='--',color='gray',lw=1);axes[0].legend(fontsize=8)
    axes[0].set_title('Gain +3 dB: final-layer unit-vector change');axes[0].set_ylabel('RMS chord (original frame support)')
    axes[1].set_title('Across-speaker direction cosine');axes[1].set_ylabel('Mean cosine; unresolved native bars omitted');axes[1].axhline(0,color='gray',lw=1)
    for ax in axes:ax.set_xticks(x,[t['model'] for t in last],rotation=40,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same weights / PCM / clock; CAL20, 10 male speakers; no TFG or SyncNet run')
    for ext in ['png','svg','pdf']:fig.savefig(analysis/('normalization_ablation.'+ext),dpi=160)
    plt.close(fig)

if __name__=='__main__':main()
