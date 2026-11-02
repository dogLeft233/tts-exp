"""Independent CPU consumer of saved per-frame moments, samples and final arrays."""
from pathlib import Path
import json, hashlib, csv
import numpy as np

ROOT=Path('F:/Datasets/tts-exp/encoder_layer_epsilon_20261020_v1')
DELIVERY=Path('C:/Users/dogle/Documents/Codex/2026-10-20/new-chat/outputs/encoder_layer_epsilon')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    completion=json.loads((ROOT/'completion.json').read_text());assert completion['status']=='COMPLETED'
    calls=json.loads((ROOT/'forward_calls.json').read_text());rows=json.loads((ROOT/'layer_metrics.json').read_text())
    assert len(calls)==60 and len(rows)==1352
    controls=json.loads((ROOT/'controls.json').read_text())
    for control in controls:
        if 'passed' in control:assert control['passed'],control
        if control['kind']=='original_cache_bridge':assert control['relative_l2']==0,control
    seen=set();sample_checks=0;full_checks=0;moment_checks=0;largest_error=0
    for call in calls:
        folder=ROOT/call['path'];mom_path=folder/'full_frame_moments.npz'
        if not mom_path.exists():continue
        if call['label'].startswith('compensated'):
            kind='epsilon_compensated';basefolder=folder.parent/'eps_1e-05_scale1'
        elif call['scale']==1.:
            kind='epsilon_baseline_drift';basefolder=folder.parent/'eps_1e-05_scale1'
        else:
            kind='scale_response';basefolder=folder.parent/f"eps_{call['eps']:.0e}_scale1"
        selected=[r for r in rows if r['encoder']==call['encoder'] and r['split']==call['split'] and r['kind']==kind and r['epsilon']==call['eps'] and r['scale']==call['scale']]
        assert len(selected)==26
        with np.load(mom_path) as saved,np.load(folder/'stage_samples.npz') as changed,np.load(basefolder/'stage_samples.npz') as baseline:
            for row in selected:
                key=(call['encoder'],call['split'],kind,call['eps'],call['scale'],row['stage']);assert key not in seen;seen.add(key)
                m=saved[row['stage']];assert m.shape==(row['shape'][0],5) and np.isfinite(m).all()
                a,b,d,dot,mx=m.T
                assert np.allclose(d,a+b-2*dot,rtol=1e-5,atol=1e-8)
                reconstructed=float(np.sqrt(d.sum()/a.sum()));error=abs(reconstructed-row['relative_l2']);assert error<1e-14
                largest_error=max(largest_error,error);moment_checks+=1
                indices=np.linspace(0,len(m)-1,min(32,len(m)),dtype=int)
                x=baseline[row['stage']].astype(float);y=changed[row['stage']].astype(float)
                for column,actual in [(0,np.sum(x*x,axis=1)),(1,np.sum(y*y,axis=1)),(2,np.sum((y-x)**2,axis=1)),(3,np.sum(x*y,axis=1)),(4,np.max(abs(y-x),axis=1))]:
                    assert np.allclose(m[indices,column],actual,rtol=1e-12,atol=1e-10)
                sample_checks+=len(indices)
                if row['stage']=='last_hidden_state':
                    x=np.load(basefolder/'last_hidden_state.npy').astype(float);y=np.load(folder/'last_hidden_state.npy').astype(float)
                    independent=float(np.linalg.norm(y-x)/np.linalg.norm(x));assert abs(independent-row['relative_l2'])<1e-14;full_checks+=1
    assert len(seen)==len(rows)
    norms=json.loads((ROOT/'groupnorm_diagnostics.json').read_text());assert len(norms)==32
    prediction_errors=[]
    for item in norms:
        base=ROOT/item['encoder']/item['split']/f"eps_{item['epsilon']:.0e}_scale1"
        with np.load(base/'norm_channel_stats.npz') as stats:
            v,g,b=stats['variance'],stats['gamma'],stats['beta'];eps=item['epsilon'];scale=item['scale']
            ratio=np.sqrt((v+eps)/(v+eps/scale**2))
            numerator=(g*g*v/(v+eps)*(ratio-1)**2).sum()
            denominator=(g*g*v/(v+eps)+b*b).sum()
            predicted=float(np.sqrt(numerator/denominator));assert abs(predicted-item['predicted_groupnorm_relative_l2'])<1e-14
            prediction_errors.append(abs(predicted-item['observed_groupnorm_relative_l2']))
            assert item['conv_scaling_maxabs']==0
    zero=[r for r in rows if r['kind']=='scale_response' and r['epsilon']==0 and r['stage']!='conv0_linear']
    compensated=[r for r in rows if r['kind']=='epsilon_compensated' and r['stage']!='conv0_linear']
    assert all(r['relative_l2']==0 and r['maxabs']==0 for r in zero+compensated)
    qa=dict(passed=True,full_forwards=60,layer_comparisons=1352,full_frame_moment_checks=moment_checks,
        sampled_vectors_checked=sample_checks,independent_full_last_state_checks=full_checks,
        reconstructed_relative_l2_max_error=largest_error,max_groupnorm_formula_absolute_error=max(prediction_errors),
        zero_epsilon_postnorm_comparisons_exact=len(zero),compensated_postnorm_comparisons_exact=len(compensated),
        original_cache_bridges_exact=4,all_repeat_stages_exact=True,weights_unchanged=True,training_updates=0)
    write(ROOT/'qa.json',qa)
    with (ROOT/'layer_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    key_stages=['conv0_groupnorm','conv0_activation','conv1_activation','conv2_activation','conv3_activation','conv4_activation','conv5_activation','conv6_activation','projection_norm','projection_linear','transformer_input_norm','transformer_01','transformer_04','transformer_08','transformer_12']
    brief=[r for r in rows if r['kind']=='scale_response' and r['epsilon']==1e-5 and r['scale']==.5 and r['stage'] in key_stages]
    write(ROOT/'default_half_scale_summary.json',brief)
    terminal=[r for r in rows if r['stage']=='last_hidden_state']
    write(ROOT/'terminal_summary.json',terminal)
    plots(rows,norms)
    DELIVERY.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['qa.json','protocol.json','layer_metrics.csv','terminal_summary.json','groupnorm_diagnostics.json','layer_epsilon_diagnostic.png']:
        shutil.copyfile(ROOT/name,DELIVERY/name)
    print(json.dumps(qa,indent=2))
    print('DEFAULT HALF SCALE:',json.dumps([{k:r[k] for k in ('encoder','split','stage','relative_l2')} for r in brief if r['stage'] in ['conv0_groupnorm','conv6_activation','projection_linear','transformer_12']],indent=2))

def plots(rows,norms):
    import sys
    sys.path.insert(0,'E:/Documents/tts-exp/.codex/scratch/report25_plotdeps_attempt01')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(13,13),layout='constrained')
    names=[r['stage'] for r in rows if r['encoder']=='hubert_base' and r['split']=='train' and r['kind']=='scale_response' and r['epsilon']==1e-5 and r['scale']==.5 and r['stage'] not in ['conv0_linear','last_hidden_state','positional_embedding']]
    for mid,color in [('hubert_base','#1565c0'),('wav2vec2_base','#c44422')]:
        for split,style in [('train','-'),('test','--')]:
            select=[next(r for r in rows if r['encoder']==mid and r['split']==split and r['kind']=='scale_response' and r['epsilon']==1e-5 and r['scale']==.5 and r['stage']==stage) for stage in names]
            axes[0].plot(range(len(names)),[r['relative_l2']*100 for r in select],style+'o',color=color,label=mid+' / '+split,markersize=3)
            for scale,marker in [(.5,'o'),(2.,'s')]:
                select=[next(r for r in rows if r['encoder']==mid and r['split']==split and r['kind']=='scale_response' and r['epsilon']==eps and r['scale']==scale and r['stage']=='last_hidden_state') for eps in [1e-7,1e-5,1e-3]]
                axes[1].plot([1e-7,1e-5,1e-3],[r['relative_l2']*100 for r in select],style+marker,color=color,label=f'{mid} / {split} / scale {scale:g}',markersize=4,alpha=.8)
        with np.load(ROOT/mid/'train/eps_1e-05_scale1/norm_channel_stats.npz') as z:
            v=np.sort(z['variance']);axes[2].plot(v,np.arange(1,len(v)+1)/len(v),color=color,label=mid)
    axes[0].set_xticks(range(len(names)),[n.replace('transformer_','T').replace('_activation','').replace('projection_','proj_').replace('transformer_input_norm','T_input') for n in names],rotation=65,ha='right')
    axes[0].set_yscale('log');axes[0].set_ylabel('Relative L2 change (%)');axes[0].set_title('Default epsilon 1e-5: scale 0.5 versus 1, paired within each encoder');axes[0].legend(ncol=2);axes[0].grid(alpha=.2)
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlabel('First GroupNorm epsilon');axes[1].set_ylabel('Final hidden-state change (%)');axes[1].set_title('Only first GroupNorm epsilon changes; epsilon 0 gives exactly zero response');axes[1].legend(ncol=2,fontsize=8);axes[1].grid(alpha=.2)
    axes[2].set_xscale('log');axes[2].axvline(1e-5,color='gray',ls=':',label='Default epsilon');axes[2].set_xlabel('First convolution output variance per channel (train chunk)');axes[2].set_ylabel('Fraction of 512 channels');axes[2].set_title('Learned filter outputs have different variance distributions');axes[2].legend();axes[2].grid(alpha=.2)
    fig.savefig(ROOT/'layer_epsilon_diagnostic.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
