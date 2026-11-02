"""Independent CPU audit and paired dose-response analysis; no producer imports."""
from pathlib import Path
import json,hashlib,csv
import numpy as np

ROOT=Path('F:/Datasets/tts-exp/encoder_gn_baseline_preserving_20261020_v1')
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def unit(x):return x/np.linalg.norm(x,axis=1)[:,None]
def anorm(x):return float(np.sqrt(np.einsum('ij,ij->',x,x,dtype=np.float64)))

def main():
    assert json.loads((ROOT/'completion.json').read_text())['status']=='COMPLETED'
    rows=json.loads((ROOT/'layer_metrics.json').read_text());calls=json.loads((ROOT/'forward_calls.json').read_text())
    controls=json.loads((ROOT/'controls.json').read_text());assert len(calls)==160 and len(rows)==1280 and all(x['passed'] for x in controls)
    controls_kind={kind:sum(c['kind']==kind for c in controls) for kind in {c['kind'] for c in controls}}
    summary=[];row_checks=0;sample_vectors=0;max_l2error=0.;full_last_checks=0;formula_max=0.;dose_max=0.;probe_hashes=0
    for mid in ('hubert_base','wav2vec2_base'):
        for split in ('train','test'):
            folder=ROOT/mid/split
            with np.load(folder/'mediator_reference.npz') as f:
                y0=f['gn'];conv=f['conv'];gamma=f['gamma'];beta=f['beta'];natural={.5:f['natural_half'],2.:f['natural_double']}
            mu=conv.mean(axis=0,dtype=np.float64);var=conv.var(axis=0,dtype=np.float64);assert np.all(var>0)
            indices=np.linspace(0,len(y0)-1,64,dtype=int);baseNorm=anorm(y0)
            reference=np.load(folder/'baseline/last_hidden_state.npy').astype(float)
            samples=np.load(folder/'baseline/stage_samples.npz')
            own=gamma.copy();other='wav2vec2_base' if mid=='hubert_base' else 'hubert_base'
            p=json.loads((ROOT/'protocol.json').read_text());parent=Path(p['parent'])
            with np.load(parent/other/'train/eps_1e-05_scale1/norm_channel_stats.npz') as donor:donor_gamma=donor['gamma']
            permutation=np.load(ROOT/'gamma_permutation.npy')
            selected_calls=[c for c in calls if c['encoder']==mid and c['split']==split]
            by_label={c['label']:c for c in selected_calls}
            for call in selected_calls:
                target=folder/call['label'];selected=[r for r in rows if r['encoder']==mid and r['split']==split and r['label']==call['label']];assert len(selected)==8
                with np.load(target/'frame_moments.npz') as moments,np.load(target/'stage_samples.npz') as values:
                    for row in selected:
                        name=row['stage'];m=moments[name];a,b,d,dot,mx,ud=m.T
                        assert np.isfinite(m).all() and np.allclose(d,a+b-2*dot,rtol=1e-5,atol=1e-8)
                        reconstructed=float(np.sqrt(d.sum()/a.sum()));error=abs(reconstructed-row['relative_l2']);assert error<1e-13;max_l2error=max(max_l2error,error)
                        assert abs(np.sqrt(ud.mean())-row['unit_change_rms'])<1e-13
                        ix=np.linspace(0,len(m)-1,min(64,len(m)),dtype=int)
                        x=samples[name].astype(float);y=values[name].astype(float)
                        checks=[np.sum(x*x,axis=1),np.sum(y*y,axis=1),np.sum((x-y)**2,axis=1),np.sum(x*y,axis=1),np.max(abs(x-y),axis=1),np.sum((unit(x)-unit(y))**2,axis=1)]
                        for k,z in enumerate(checks):assert np.allclose(m[ix,k],z,rtol=1e-12,atol=1e-10)
                        row_checks+=1;sample_vectors+=len(ix)
                        if name=='last':
                            full=np.load(target/'last_hidden_state.npy').astype(float)
                            assert abs(np.linalg.norm(full-reference)/np.linalg.norm(reference)-row['relative_l2'])<1e-13
                            assert abs(np.linalg.norm(unit(full)-unit(reference))/np.sqrt(len(full))-row['unit_change_rms'])<1e-13;full_last_checks+=1
                    gn=next(x for x in selected if x['stage']=='gn');last=next(x for x in selected if x['stage']=='last')
                    meta={k:v for k,v in call.items() if k not in ['encoder','split','folder','seconds','last_sha256']}
                    summary.append(dict(encoder=mid,split=split,**meta,gn_relative_l2=gn['relative_l2'],gn_unit_change_rms=gn['unit_change_rms'],
                        terminal_relative_l2=last['relative_l2'],terminal_unit_change_rms=last['unit_change_rms'],terminal_mean_angle_degrees=last['mean_angle_degrees'],
                        relative_input_to_unit_output_gain=None if gn['relative_l2']==0 else last['unit_change_rms']/gn['relative_l2'],
                        direction_input_to_unit_output_gain=None if gn['unit_change_rms']==0 else last['unit_change_rms']/gn['unit_change_rms']))
                    if call['family'] in ('matched_natural','random_probe'):
                        error=abs(gn['relative_l2']/call['dose']-1);dose_max=max(dose_max,error);assert error<5e-4
                    if call['family']=='natural':assert np.array_equal(values['gn'],natural[call['scale']][indices])
                    if call['family']=='anchored_epsilon' or call['family']=='anchored_gamma':
                        eps=call.get('epsilon',1e-5);g=gamma if call['family']=='anchored_epsilon' else donor_gamma if call['gamma']=='donor_raw_index' else own[permutation]
                        invdiff=1/np.sqrt(var+eps/call['scale']**2)-1/np.sqrt(var+eps)
                        predicted=y0[indices].astype(float)+(conv[indices].astype(float)-mu[None,:])*g[None,:]*invdiff[None,:]
                        error=float(abs(predicted-values['gn']).max());formula_max=max(formula_max,error);assert error<5e-5
                    if call['family'] in ('baseline','baseline_repeat','anchor_control') or call['family']=='anchored_epsilon' and call['epsilon']==0:
                        assert all(r['relative_l2']==0 and r['unit_change_rms']==0 for r in selected)
            # Independently regenerate both probe families and check all 64 sampled targets.
            ledger=json.loads((folder/'probe_ledger.json').read_text())
            for probe in ledger:
                rng=np.random.default_rng(probe['seed'])
                direction=rng.standard_normal(y0.shape,dtype=np.float32) if probe['family']=='iid_gaussian' else ((y0.astype(float)-beta[None,:])*rng.standard_normal(512)[None,:]).astype(np.float32)
                assert hashlib.sha256(direction.tobytes()).hexdigest()==probe['direction_sha256'];probe_hashes+=1
                for dose in (3e-4,1e-3):
                    for sign in (-1,1):
                        label=f"probe_{probe['family']}_{probe['seed']}_{dose:.0e}_{sign:+d}"
                        with np.load(folder/label/'stage_samples.npz') as saved:
                            expected=(y0[indices].astype(float)+direction[indices].astype(float)*(sign*dose*baseNorm/anorm(direction))).astype(np.float32)
                            assert np.array_equal(expected,saved['gn'])
            for scale in (.5,2.):
                direction=natural[scale].astype(float)-y0.astype(float)
                for dose in (1e-4,3e-4,1e-3,3e-3):
                    with np.load(folder/f'matched_natural_{scale:g}_{dose:.0e}/stage_samples.npz') as saved:
                        expected=(y0[indices].astype(float)+direction[indices]*(dose*baseNorm/anorm(direction))).astype(np.float32)
                        assert np.array_equal(expected,saved['gn'])
            samples.close();del conv,y0,natural
    assert row_checks==1280 and full_last_checks==160
    qa=dict(passed=True,full_forwards=160,layer_comparisons=row_checks,sampled_vectors=sample_vectors,full_last_state_checks=full_last_checks,
        controls=controls_kind,relative_l2_reconstruction_maxerror=max_l2error,gn_formula_sample_maxabs=formula_max,
        achieved_relative_dose_maxrelativeerror=dose_max,probe_directions_regenerated_fullhash=probe_hashes,
        all_probe_and_matched_sample_targets_bitexact=True,all_baseline_and_epsilon_zero_states_exact=True,training_updates=0,
        numpy_version=np.__version__,consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(ROOT/'qa.json',qa);write(ROOT/'endpoint_summary.json',summary)
    fields=list(dict.fromkeys(k for r in summary for k in r))
    with (ROOT/'endpoint_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    pairs=[]
    for split in ('train','test'):
        for h in [x for x in summary if x['encoder']=='hubert_base' and x['split']==split]:
            w=next(x for x in summary if x['encoder']=='wav2vec2_base' and x['split']==split and x['label']==h['label'])
            if h['terminal_unit_change_rms']>0:
                pairs.append(dict(split=split,label=h['label'],family=h['family'],gn_relative_ratio_w_over_h=w['gn_relative_l2']/h['gn_relative_l2'],
                    terminal_direction_ratio_w_over_h=w['terminal_unit_change_rms']/h['terminal_unit_change_rms'],
                    conditional_gain_ratio_w_over_h=w['relative_input_to_unit_output_gain']/h['relative_input_to_unit_output_gain'],
                    conditional_direction_gain_ratio_w_over_h=w['direction_input_to_unit_output_gain']/h['direction_input_to_unit_output_gain']))
    write(ROOT/'paired_model_ratios.json',pairs)
    probe_summary=[];odd_even=[]
    for mid in ('hubert_base','wav2vec2_base'):
        for split in ('train','test'):
            base=unit(np.load(ROOT/mid/split/'baseline/last_hidden_state.npy').astype(float))
            for family in ('iid_gaussian','channel_modulated'):
                for dose in (3e-4,1e-3):
                    members=[x for x in summary if x['encoder']==mid and x['split']==split and x['family']=='random_probe' and x['probe_family']==family and x['dose']==dose]
                    gain=[x['relative_input_to_unit_output_gain'] for x in members]
                    probe_summary.append(dict(encoder=mid,split=split,probe_family=family,dose=dose,conditions=len(members),gain_mean=float(np.mean(gain)),gain_min=min(gain),gain_max=max(gain),
                        mean_terminal_unit_change_rms=float(np.mean([x['terminal_unit_change_rms'] for x in members]))))
                    for seed in (20261020,20261021):
                        values=[]
                        for sign in (-1,1):values.append(unit(np.load(ROOT/mid/split/f'probe_{family}_{seed}_{dose:.0e}_{sign:+d}/last_hidden_state.npy').astype(float)))
                        odd=(values[1]-values[0])/2;even=(values[1]+values[0])/2-base
                        odd_even.append(dict(encoder=mid,split=split,probe_family=family,dose=dose,seed=seed,
                            odd_gain=float(np.linalg.norm(odd)/np.sqrt(len(base))/dose),even_to_odd_ratio=float(np.linalg.norm(even)/np.linalg.norm(odd))))
    write(ROOT/'random_probe_summary.json',probe_summary);write(ROOT/'probe_odd_even.json',odd_even)
    print(json.dumps(qa,indent=2))
    print('TEST NATURAL AND MATCHED',json.dumps([x for x in summary if x['split']=='test' and (x['family']=='natural' and x['scale']==.5 or x['family']=='matched_natural' and x['scale']==.5 and x['dose']==1e-3)],indent=2))
    print('PROBE_GAIN',json.dumps([x for x in probe_summary if x['dose']==1e-3],indent=2))

if __name__=='__main__':main()
