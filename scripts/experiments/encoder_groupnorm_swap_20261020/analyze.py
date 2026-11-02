"""CPU audit plus within/cross-encoder feature-direction comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,csv,sys,shutil,hashlib
import numpy as np
from geometry import directions,changes

ROOT=Path('F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1')
DELIVERY=Path('C:/Users/dogle/Documents/Codex/2026-10-20/new-chat/outputs/encoder_groupnorm_swap')
ARMS=('native','swapped','self_permuted')
KEY_STAGES=('conv0_groupnorm','conv6_activation','projection_linear','transformer_04','transformer_08','transformer_09','transformer_12','last_hidden_state')
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    assert json.loads((ROOT/'completion.json').read_text())['status']=='COMPLETED'
    calls=json.loads((ROOT/'forward_calls.json').read_text());rows=json.loads((ROOT/'layer_metrics.json').read_text())
    controls=json.loads((ROOT/'controls.json').read_text());assert all(x['passed'] for x in controls)
    assert len(calls)==40 and len(rows)==832
    checks=0;vectors=0;maxerror=0;full_last=0
    for call in calls:
        if call['repeat']:continue
        folder=ROOT/call['folder']
        kinds=['scale_response'] if call['scale']!=1 else [] if call['arm']=='native' else ['baseline_drift']
        for kind in kinds:
            reference=ROOT/call['encoder']/call['split']/('native' if kind=='baseline_drift' else call['arm'])/'scale1'
            selected=[x for x in rows if x['encoder']==call['encoder'] and x['split']==call['split'] and x['arm']==call['arm'] and x['scale']==call['scale'] and x['kind']==kind]
            assert len(selected)==26
            with np.load(folder/(kind+'_moments.npz')) as moments,np.load(folder/'stage_samples.npz') as sample,np.load(reference/'stage_samples.npz') as base:
                for row in selected:
                    name=row['stage'];m=moments[name];a,b,d,dot,mx=m.T
                    assert np.isfinite(m).all() and np.allclose(d,a+b-2*dot,rtol=1e-5,atol=1e-8)
                    error=abs(float(np.sqrt(d.sum()/a.sum()))-row['relative_l2']);assert error<1e-13;maxerror=max(maxerror,error)
                    valid=(a>0)&(b>0);cos=np.clip(dot[valid]/np.sqrt(a[valid]*b[valid]),-1,1)
                    assert abs(float(cos.mean())-row['mean_cosine'])<1e-13
                    indices=np.linspace(0,len(m)-1,min(128,len(m)),dtype=int)
                    x,y=base[name].astype(float),sample[name].astype(float)
                    for col,actual in [(0,np.sum(x*x,axis=1)),(1,np.sum(y*y,axis=1)),(2,np.sum((x-y)**2,axis=1)),(3,np.sum(x*y,axis=1)),(4,np.max(abs(x-y),axis=1))]:
                        assert np.allclose(actual,m[indices,col],rtol=1e-12,atol=1e-10)
                    checks+=1;vectors+=len(indices)
                    if name=='last_hidden_state':
                        x,y=np.load(reference/'last_hidden_state.npy').astype(float),np.load(folder/'last_hidden_state.npy').astype(float)
                        assert abs(np.linalg.norm(y-x)/np.linalg.norm(x)-row['relative_l2'])<1e-13;full_last+=1
    assert checks==832
    # Per-channel parameter transplant verified directly from persisted tensors.
    parent=Path(json.loads((ROOT/'protocol.json').read_text())['parent']);permutation=np.load(ROOT/'channel_permutation.npy')
    params={}
    for mid in ('hubert_base','wav2vec2_base'):
        with np.load(parent/mid/'train/eps_1e-05_scale1/norm_channel_stats.npz') as p:params[mid]=[p['gamma'].astype(np.float32),p['beta'].astype(np.float32)]
    for mid in params:
        donor=next(x for x in params if x!=mid)
        for split in ('train','test'):
            for arm in ARMS:
                expected=params[donor] if arm=='swapped' else [x[permutation] for x in params[mid]] if arm=='self_permuted' else params[mid]
                with np.load(ROOT/mid/split/arm/'scale1/applied_parameters.npz') as p:
                    assert np.array_equal(p['gamma'],expected[0]) and np.array_equal(p['beta'],expected[1])
    predictions=json.loads((ROOT/'groupnorm_predictions.json').read_text());assert len(predictions)==24
    pred_error=max(abs(x['predicted']-x['actual']) for x in predictions);assert pred_error<1e-6
    qa=dict(passed=True,forwards=40,layer_comparisons=checks,independent_sampled_vectors=vectors,
        independent_full_last_state_checks=full_last,relative_l2_reconstruction_maxerror=maxerror,
        groupnorm_formula_prediction_maxerror=pred_error,parameter_transplants_exact=True,all_controls_passed=True,
        original_full_states_restored=True,training_updates=0,numpy_version=np.__version__,python=sys.version,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(ROOT/'qa.json',qa)
    with (ROOT/'layer_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    sampled={}
    for mid in params:
        for split in ('train','test'):
            for arm in ARMS:
                for scale in (1.,.5,2.):
                    with np.load(ROOT/mid/split/arm/f'scale{scale:g}/stage_samples.npz') as z:
                        sampled[mid,split,arm,scale]={k:z[k].astype(float) for k in z.files}
    within=[];response=[];cross=[];cross_response=[]
    for mid in params:
        for split in ('train','test'):
            native=sampled[mid,split,'native',1.]
            for arm in ('swapped','self_permuted'):
                base=sampled[mid,split,arm,1.]
                for stage in base:
                    within.append(dict(encoder=mid,split=split,arm=arm,stage=stage,sampled_frames=len(base[stage]),**directions(native[stage],base[stage])))
                    for scale in (.5,2.):
                        response.append(dict(encoder=mid,split=split,arm=arm,stage=stage,scale=scale,**changes(native[stage],sampled[mid,split,'native',scale][stage],base[stage],sampled[mid,split,arm,scale][stage])))
    for split in ('train','test'):
        for h_arm in ARMS:
            for w_arm in ARMS:
                h=sampled['hubert_base',split,h_arm,1.];w=sampled['wav2vec2_base',split,w_arm,1.]
                for stage in KEY_STAGES:
                    cross.append(dict(split=split,hubert_arm=h_arm,wav2vec2_arm=w_arm,stage=stage,**directions(h[stage],w[stage])))
                    for scale in (.5,2.):
                        cross_response.append(dict(split=split,hubert_arm=h_arm,wav2vec2_arm=w_arm,stage=stage,scale=scale,**changes(h[stage],sampled['hubert_base',split,h_arm,scale][stage],w[stage],sampled['wav2vec2_base',split,w_arm,scale][stage])))
    for name,value in [('within_direction_geometry',within),('within_scale_direction_alignment',response),('cross_encoder_geometry',cross),('cross_encoder_scale_direction_alignment',cross_response)]:write(ROOT/(name+'.json'),value)
    terminal=dict(baseline_drift=[x for x in rows if x['stage']=='last_hidden_state' and x['kind']=='baseline_drift'],
        scale_response=[x for x in rows if x['stage']=='last_hidden_state' and x['kind']=='scale_response'],
        within_direction_geometry=[x for x in within if x['stage']=='last_hidden_state'],
        within_scale_direction_alignment=[x for x in response if x['stage']=='last_hidden_state'],
        cross_encoder_geometry=[x for x in cross if x['stage']=='last_hidden_state'],
        cross_encoder_scale_direction_alignment=[x for x in cross_response if x['stage']=='last_hidden_state'])
    write(ROOT/'terminal_summary.json',terminal)
    plots(rows,within,cross)
    DELIVERY.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','qa.json','terminal_summary.json','layer_metrics.csv','within_direction_geometry.json','within_scale_direction_alignment.json','cross_encoder_geometry.json','cross_encoder_scale_direction_alignment.json','swap_diagnostic.png']:
        shutil.copyfile(ROOT/name,DELIVERY/name)
    print(json.dumps(qa,indent=2))
    print('TERMINAL_SCALE_HALF',json.dumps([dict(encoder=x['encoder'],split=x['split'],arm=x['arm'],relative_l2=x['relative_l2'],angle=x['mean_angle_degrees']) for x in terminal['scale_response'] if x['scale']==.5],indent=2))
    print('BASELINE_DRIFT',json.dumps([dict(encoder=x['encoder'],split=x['split'],arm=x['arm'],cos=x['mean_cosine'],angle=x['mean_angle_degrees']) for x in terminal['baseline_drift']],indent=2))
    print('CROSS',json.dumps([x for x in terminal['cross_encoder_geometry'] if x['hubert_arm']!='self_permuted' and x['wav2vec2_arm']!='self_permuted'],indent=2))

def plots(rows,within,cross):
    sys.path.insert(0,'E:/Documents/tts-exp/.codex/scratch/report25_plotdeps_attempt01')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=['conv0_groupnorm','conv6_activation','projection_linear','transformer_04','transformer_08','transformer_12']
    fig,axes=plt.subplots(3,1,figsize=(11,12),layout='constrained')
    colors={'hubert_base':'#1565c0','wav2vec2_base':'#c44422'}
    for mid in colors:
        for arm,style in [('native','-'),('swapped','--'),('self_permuted',':')]:
            selected=[next(x for x in rows if x['encoder']==mid and x['split']=='test' and x['arm']==arm and x['kind']=='scale_response' and x['scale']==.5 and x['stage']==stage) for stage in stages]
            axes[0].plot(range(len(stages)),[x['mean_angle_degrees'] for x in selected],style+'o',color=colors[mid],label=mid+' / '+arm)
        for arm,style in [('swapped','--'),('self_permuted',':')]:
            selected=[next(x for x in rows if x['encoder']==mid and x['split']=='test' and x['arm']==arm and x['kind']=='baseline_drift' and x['stage']==stage) for stage in stages]
            axes[1].plot(range(len(stages)),[x['mean_angle_degrees'] for x in selected],style+'o',color=colors[mid],label=mid+' / '+arm)
    for h,w,style in [('native','native','-'),('swapped','native','--'),('native','swapped',':'),('swapped','swapped','-.')]:
        selected=[next(x for x in cross if x['split']=='test' and x['hubert_arm']==h and x['wav2vec2_arm']==w and x['stage']==stage) for stage in stages]
        axes[2].plot(range(len(stages)),[x['unit_direction_CKA'] for x in selected],style+'o',label=f'H={h}, W={w}')
    for ax in axes:
        ax.set_xticks(range(len(stages)),['First GN','Conv7','Projection','T4','T8','T12']);ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    axes[0].set_yscale('log');axes[0].set_ylabel('Mean angle (degrees)');axes[0].set_title('RMS scale 0.5 versus 1 within each arm (test chunk)')
    axes[1].set_ylabel('Mean angle (degrees)');axes[1].set_title('Direction drift caused by changing GN parameters at unchanged RMS')
    axes[2].set_ylabel('Unit-direction CKA');axes[2].set_ylim(0,1);axes[2].set_title('Cross-model temporal geometry; no learned coordinate alignment')
    fig.savefig(ROOT/'swap_diagnostic.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
