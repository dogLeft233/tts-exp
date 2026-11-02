"""Recompute AVS boundary/content intervention results without producer imports."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def cosine(a,v):
    a,v=np.asarray(a,dtype='float64'),np.asarray(v,dtype='float64')
    assert a.shape==v.shape and np.isfinite(a).all() and np.isfinite(v).all()
    return np.einsum('ij,ij->i',a,v)/(np.sqrt(np.einsum('ij,ij->i',a,a))*np.sqrt(np.einsum('ij,ij->i',v,v)))


def ci(values):
    a=np.array(values,dtype=float)
    if not len(a):return dict(mean=None,ci95=None,sources=0)
    assert np.isfinite(a).all()
    i=np.random.default_rng(20261028).integers(0,len(a),(10000,len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(np.take(a,i).mean(1),[.025,.975]).tolist(),sources=len(a))


def audit(root):
    items=read(root/'items.json');inputs={r['id']:r for r in read(root/'inputs.json')}
    assert len(items)==len(inputs)==80
    protocol=read(root/'protocol.json')
    assert sha(root/'source_frozen.py')==protocol['script_sha256']
    rows=[];max_error=0.;valid_landmarks=0;total_frames=0;max_native_parity=0.;checked_pcm=0
    regions_by_name=['head','center','tail']
    conditions=['native','reflect','repeat','crop_head','crop_mid','crop_tail']
    for item in items:
        ident=item['id'];row=inputs[ident];path=root/'items'/f'{ident}.npz'
        assert sha(path)==item['npz_sha256']
        with np.load(path,allow_pickle=False) as z:
            f=item['frames'];total_frames+=f
            assert z['input_audio'].shape==(f,104) and z['input_video'].shape==(f,88,88)
            assert np.isfinite(z['input_audio']).all() and np.isfinite(z['input_video']).all()
            assert np.array_equal(z['source_frame_indices'],np.arange(f))
            t=(f-5)//2
            assert np.array_equal(z['target_indices'],np.arange(t,t+5))
            assert np.array_equal(z['crop_starts'],[t,t-10,t-20])
            assert np.array_equal(z['target_local_starts'],[0,10,20])
            regions=np.stack([np.arange(15),np.arange((f-15)//2,(f-15)//2+15),np.arange(f-15,f)])
            assert np.array_equal(z['segment_indices'],regions)
            for key,mode in [('reflect','reflect'),('repeat','edge')]:
                mapping=np.pad(np.arange(f),(75,75),mode=mode)
                assert np.array_equal(z[key+'_map'],mapping)
                assert np.array_equal(mapping[75:75+f],np.arange(f))
                for modality in ['audio','video']:
                    whole=z['input_'+modality]
                    assert np.array_equal(whole[mapping][75:75+f],whole)
            for start,local in zip(z['crop_starts'],z['target_local_starts']):
                assert start>=0 and start+25<=f
                for modality in ['audio','video']:
                    whole=z['input_'+modality]
                    assert np.array_equal(whole[start:start+25][local:local+5],whole[t:t+5])
            valid=z['lip_valid'];points=z['lip_points']
            assert len(valid)==f and points.shape==(f,4,2)
            assert np.isfinite(points[valid]).all()
            lp=points[valid].astype(float)
            aperture=np.sqrt(np.sum((lp[:,0]-lp[:,1])**2,axis=1))/np.sqrt(np.sum((lp[:,2]-lp[:,3])**2,axis=1))
            assert np.allclose(aperture,z['lip_ratio'][valid],atol=1e-6,rtol=1e-6)
            valid_landmarks+=int(valid.sum())
            assert int(valid.sum())==item['valid_landmark_frames']
            assert np.isfinite(z['frame_rms_ratio']).all() and (z['frame_rms_ratio']>=0).all()
            if Path(row['audio_path']).exists():
                from scipy.io import wavfile
                sr,pcm=wavfile.read(row['audio_path']);assert sr==16000 and pcm.dtype==np.int16
                p=pcm.astype(float);rms=np.sqrt(np.mean(p*p))
                energy=np.array([np.sqrt(np.mean(p[i*640:(i+1)*640]**2))/rms for i in range(f)])
                assert np.allclose(energy,z['frame_rms_ratio'],rtol=0,atol=1e-12)
                checked_pcm+=1
            scores={c:cosine(z[c+'_audio'],z[c+'_video']) for c in conditions}
            assert all(len(scores[c])==25 for c in conditions if c.startswith('crop'))
            assert len(scores['native'])==f and len(scores['reflect'])==len(scores['repeat'])==f+150
            regional={c:{name:float(scores[c][reg+(0 if c=='native' else 75)].mean())
                         for name,reg in zip(regions_by_name,regions)} for c in ['native','reflect','repeat']}
            for c in regional:
                for name in regions_by_name:
                    max_error=max(max_error,abs(regional[c][name]-item['regional_scores'][c][name]))
            target={c:float(scores[c][start:start+5].mean()) for c,start in zip(conditions[3:],[0,10,20])}
            for c in target:max_error=max(max_error,abs(target[c]-item['target_scores'][c]))
            values={}
            for c in ['reflect','repeat']:
                for region in regions_by_name:values[c+'_'+region+'_change']=regional[c][region]-regional['native'][region]
                values[c+'_edge_minus_center_change']=(values[c+'_head_change']+values[c+'_tail_change'])/2-values[c+'_center_change']
            values.update(crop_head_minus_mid=target['crop_head']-target['crop_mid'],
                          crop_tail_minus_mid=target['crop_tail']-target['crop_mid'],
                          crop_edges_minus_mid=(target['crop_head']+target['crop_tail'])/2-target['crop_mid'])
            states={};conditional={}
            for energy_threshold in [.05,.1,.2]:
                for lip_threshold in [.03,.05,.08]:
                    setting=f'E{energy_threshold}_L{lip_threshold}';states[setting]={};conditional[setting]={}
                    quiet=z['frame_rms_ratio']<energy_threshold;closed=z['lip_ratio']<lip_threshold
                    masks={'quiet_closed':quiet&closed,'quiet_open':quiet&~closed,
                           'active_closed':~quiet&closed,'active_open':~quiet&~closed}
                    for state,mask in masks.items():
                        original_mask=mask&valid
                        target_mask=original_mask[t:t+5]
                        cell=dict(target_frames=int(target_mask.sum()))
                        for c,local in [('crop_head',0),('crop_tail',20)]:
                            cell[c+'_minus_mid']=float((scores[c][local:local+5]-scores['crop_mid'][10:15])[target_mask].mean()) if target_mask.any() else None
                        ix=np.r_[regions[0],regions[2]];ex=ix[original_mask[ix]];mx=regions[1][original_mask[regions[1]]]
                        cell.update(edge_frames=len(ex),center_frames=len(mx))
                        for c in ['reflect','repeat']:
                            edge=float((scores[c][ex+75]-scores['native'][ex]).mean()) if len(ex) else None
                            center=float((scores[c][mx+75]-scores['native'][mx]).mean()) if len(mx) else None
                            cell[c+'_edge_change']=edge;cell[c+'_center_change']=center
                            cell[c+'_edge_minus_center_change']=edge-center if edge is not None and center is not None else None
                        conditional[setting][state]=cell
                    if valid[t:t+5].all():
                        for c in ['crop_head','crop_tail']:
                            cells=conditional[setting].values()
                            weighted=sum(x['target_frames']*x[c+'_minus_mid'] for x in cells if x['target_frames'])/5
                            assert abs(weighted-values[c+'_minus_mid'])<1e-12
                    for name,ix in [('edge',np.r_[regions[0],regions[2]]),('center',regions[1])]:
                        states[setting][name]={}
                        for state,mask in masks.items():
                            selected=ix[mask[ix]&valid[ix]]
                            states[setting][name][state]=dict(frames=len(selected),score=float(scores['native'][selected].mean()) if len(selected) else None)
            reference=root.parent/'common_window_curves_cloud_20261028_attempt01/items'/f'{ident}.npz'
            with np.load(reference,allow_pickle=False) as old:
                for modality in ['audio','video']:
                    error=float(np.max(np.abs(z['native_'+modality]-old['avs_'+modality])))
                    max_native_parity=max(max_native_parity,error)
                    assert error<=1e-5
                    assert abs(error-item['native_parity'][modality+'_max_error'])<1e-12
            rows.append(dict(id=ident,source=item['source'],arm=item['arm'],strict_reference=item['strict_reference'],
                             values=values,states=states,regional=regional,target=target,conditional=conditional))
    assert max_error<1e-10
    saved=read(root/'summary.json');summary={};max_stat_error=0.
    for view,subset in [('all20',rows),('strict12',[r for r in rows if r['strict_reference']])]:
        summary[view]={}
        for arm in ['N','C']:
            rs=[r for r in subset if r['arm']==arm];sources=sorted({r['source'] for r in rs})
            effects={field:ci([np.mean([r['values'][field] for r in rs if r['source']==source]) for source in sources]) for field in rs[0]['values']}
            state_stats={}
            for setting in rs[0]['states']:
                state_stats[setting]={}
                for state in ['quiet_closed','quiet_open','active_closed','active_open']:
                    deltas=[];coverage={}
                    for source in sources:
                        vals=[]
                        for r in rs:
                            if r['source']!=source:continue
                            e,c=r['states'][setting]['edge'][state],r['states'][setting]['center'][state]
                            if e['frames'] and c['frames']:vals.append(e['score']-c['score'])
                        if vals:deltas.append(np.mean(vals))
                        coverage[source]=len(vals)
                    state_stats[setting][state]=dict(**ci(deltas),matched_clip_counts=coverage)
            actual=dict(clips=len(rs),sources=len(sources),effects=effects,state_edge_minus_center=state_stats)
            expected=saved[view][arm]
            for field,stats in effects.items():
                old=expected['effects'][field]
                max_stat_error=max(max_stat_error,abs(stats['mean']-old['mean']),float(np.abs(np.array(stats['ci95'])-old['ci95']).max()))
            for setting,ss in state_stats.items():
                for state,stats in ss.items():
                    old=expected['state_edge_minus_center'][setting][state]
                    assert stats['sources']==old['sources'] and stats['matched_clip_counts']==old['matched_clip_counts']
                    if stats['sources']:
                        max_stat_error=max(max_stat_error,abs(stats['mean']-old['mean']),float(np.abs(np.array(stats['ci95'])-old['ci95']).max()))
                    else:assert stats['mean'] is None and old['mean'] is None
            summary[view][arm]=actual
    assert max_stat_error<1e-10
    result=dict(status='PASS',clips=80,sources=20,total_frames=total_frames,valid_landmark_frames=valid_landmarks,
                max_endpoint_error=max_error,max_stat_error=max_stat_error,max_native_parity_independently_verified=max_native_parity,
                canonical_pcm_energy_checked_clips=checked_pcm,
                original_source_sha256=sha(root/'source_frozen.py'),audit_source_sha256=sha(__file__),
                scope='Independent feature/mapping/aperture/cosine/state/cluster-bootstrap reconstruction, not a second GPU model implementation')
    save(root/'parent_independent_audit.json',result)
    save(root/'parent_reconstructed_rows.json',rows)
    conditional_summary={}
    for arm in ['N','C']:
        rs=[r for r in rows if r['arm']==arm];sources=sorted({r['source'] for r in rs})
        conditional_summary[arm]={}
        for setting in rs[0]['conditional']:
            conditional_summary[arm][setting]={}
            for state in rs[0]['conditional'][setting]:
                fields=[f for f in rs[0]['conditional'][setting][state] if not f.endswith('_frames')]
                conditional_summary[arm][setting][state]={}
                for field in fields:
                    by_source=[];coverage={}
                    for source in sources:
                        values=[r['conditional'][setting][state][field] for r in rs if r['source']==source and r['conditional'][setting][state][field] is not None]
                        coverage[source]=len(values)
                        if values:by_source.append(np.mean(values))
                    conditional_summary[arm][setting][state][field]=dict(**ci(by_source),matched_clip_counts=coverage)
    save(root/'content_stratified_interventions.json',dict(status='COMPLETE_ADDITIONAL_DIAGNOSTIC',
         amendment='target_state_analysis_amendment.json',new_gpu_forwards=0,summary=conditional_summary,
         caveat='Original energy/aperture masks fixed across encoding conditions. Active means non-low-energy, not verified speech. Limited-support cells retain NA/coverage.'))
    print(json.dumps(result,indent=2))
    return rows


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.4,4.4),constrained_layout=True)
    for arm,color,label in [('N','#2875b0','Natural'),('C','#d46727','Cloud TTS')]:
        rs=[r for r in rows if r['arm']==arm];sources=sorted({r['source'] for r in rs})
        means=[];lo=[];hi=[]
        for c in ['crop_head','crop_mid','crop_tail']:
            stats=ci([np.mean([r['target'][c] for r in rs if r['source']==s]) for s in sources])
            means.append(stats['mean']);lo.append(stats['ci95'][0]);hi.append(stats['ci95'][1])
        axes[0].plot(range(3),means,label=label,color=color,marker='o')
        axes[0].fill_between(range(3),lo,hi,alpha=.15,color=color)
    axes[0].set(title='Identical 5-frame target; fixed 25-frame inputs',ylabel='AVS_u target cosine')
    axes[0].set_xticks(range(3),['Target at start','At center','At end'])
    axes[0].legend()
    for ax,arm,name in zip(axes[1:],['N','C'],['Natural','Cloud TTS']):
        rs=[r for r in rows if r['arm']==arm];sources=sorted({r['source'] for r in rs})
        for condition,color,label in [('native','#356da4','Native'),('reflect','#bd6139','Reflect context'),('repeat','#449260','Repeat context')]:
            means=[np.mean([np.mean([r['regional'][condition][region] for r in rs if r['source']==s]) for s in sources]) for region in ['head','center','tail']]
            ax.plot(range(3),means,marker='o',label=label,color=color)
        ax.set(title=f'{name}: same original frames',ylabel='AVS_u regional cosine')
        ax.set_xticks(range(3),['Original start','Original center','Original end']);ax.legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Boundary/context diagnostics; 20 sources. Target content unchanged; context is manipulated.',fontsize=12)
    fig.savefig(root/'boundary_context_diagnostics.png',dpi=170)
    fig.savefig(root/'boundary_context_diagnostics.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    rows=audit(args.run)
    if args.plot:plot(args.run,rows)
