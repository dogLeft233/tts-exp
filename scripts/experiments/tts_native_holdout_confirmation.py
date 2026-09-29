"""Frozen new-speaker native confirmation: feature extraction precedes index sealing and scoring."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.tts_independent_visual import read,write,sha
from scripts.experiments.tts_native_midpoint import coordinates,transform,distances,curve_score
from scripts.experiments.tts_native_permutation import generate_perms,gram,curve_metrics,FIELDS
from scripts.experiments.tts_native_boundary_audit import stats
OUT=ROOT/'runs/tts_native_holdout_confirmation_20260926'
ARMS=('N','Q1','Q2')


def verify_protocol():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().split()[0]
    for path,h in p['parent_hashes'].items():assert sha(path)==h,path
    return p


def input_rows():
    p=verify_protocol();rows=[]
    for orig in p['rows']:
        r=dict(orig);r['audio']=dict(orig['audio']);r['tts']={}
        for arm in ARMS[1:]:
            receipt=OUT/'tts'/r['id']/(arm+'.json')
            if receipt.exists():
                z=read(receipt);r['tts'][arm]=z
                if z['status']=='COMPLETE':r['audio'][arm]={'path':z['path'],'sha256':z['sha256']}
        rows.append(r)
    return rows


def extract():
    # Parent strengthened HOLD to cover new SyncNet embeddings as well as scores.
    assert read(OUT/'leakage_audit.json')['score_gate']=='PASS', 'leakage HOLD: no new SyncNet features'
    import cv2,torch,soundfile as sf
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model,chunk_mels
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids,gpu_lease
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    from scripts.experiments.tts_native_gain_attribution import config as sc
    from scripts.experiments.tts_prepost_mel import MODEL
    from scripts.experiments.tts_raw_video_transfer import render
    sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    p=verify_protocol();rows=input_rows();im=p['images'][0]
    assert im['id']=='3' and sha(im['path'])==im['sha256']
    assert sha(MODEL)==read(ROOT/'runs/tts_prepost_mel_20260926/protocol.json')['model_sha256']
    write(OUT/'frontend_protocol.json',{'model_sha256':sha(MODEL),'mel_frontend_sha256':sha(audio.__file__),'syncnet_weight_sha256':sha(sc.SYNCNET_MODEL),'syncnet_definition_sha256':sha(sc.SYNCNET_DEFINITION),'syncnet_instance_sha256':sha(sc.SYNCNET_INSTANCE),'feature_worker_sha256':sha(sc.REPO/'scripts/experiments/tts_native_gain_attribution/syncnet.py'),'image':im,'mode':'feature extraction only, no distance or score computation','joint_length':'min(F, samples//640)-5'})
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    with gpu_lease(gpu_peak_bytes=8<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=8<<30) as gate:
        write(OUT/'feature_compute_gate.json',gate)
        assert not set(gpu_compute_pids())-{os.getpid()}
        model=_load_model(MODEL,'cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
        frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0
        it=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
        try:
            for r in rows:
                sid=r['id']
                for arm in ARMS:
                    dest=OUT/'features'/sid/arm
                    if dest.with_suffix('.json').exists():continue
                    if arm not in r['audio']:
                        write(dest.with_suffix('.json'),{'id':sid,'arm':arm,'speaker':r['speaker'],'status':'TTS_FAILED','tts':r['tts'].get(arm)});continue
                    assert not set(gpu_compute_pids())-{os.getpid()}
                    info=r['audio'][arm];assert sha(info['path'])==info['sha256']
                    ae,am=engine.extract_audio(info['path'])
                    mel=audio.melspectrogram(audio.load_wav(info['path'],16000)).astype(np.float32);chunks=chunk_mels(mel,25);frames=[]
                    for start in range(0,len(chunks),32):
                        mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).cuda()
                        with torch.inference_mode():pred=model(mt,it.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
                        for fo in pred:
                            full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));frames.append(crop_zero_padded(full,im['score_box']['box']))
                    vp=OUT/'videos'/sid/(arm+'.avi');render(np.stack(frames),np.arange(len(frames))/25,vp);del frames
                    ve,vm=engine.extract_visual(vp);L=min(vm['frame_count'],am['sample_count']//640)-5
                    assert 0<L<=min(len(ve),len(ae))
                    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),visual=ve,audio=ae)
                    write(dest.with_suffix('.json'),{'id':sid,'speaker':r['speaker'],'arm':arm,'status':'COMPLETE','joint_L':L,'path':str(dest.with_suffix('.npz')),'sha256':sha(dest.with_suffix('.npz')),'audio':info,'audio_metadata':am,'visual_metadata':vm,'video_path':str(vp),'video_sha256':sha(vp),'mel_shape':list(mel.shape),'tts':r['tts'].get(arm)})
                print('features',sid,flush=True)
            first=rows[0]['id'];repeat={}
            for arm in ARMS:
                z=read(OUT/'features'/first/(arm+'.json'));saved=np.load(z['path']);a,am=engine.extract_audio(z['audio']['path']);v,vm=engine.extract_visual(z['video_path']);repeat[arm]={'audio_max_error':float(abs(a-saved['audio']).max()),'visual_max_error':float(abs(v-saved['visual']).max())}
            write(OUT/'feature_repeat_controls.json',repeat)
        finally:engine.close()


def seal():
    p=verify_protocol();assert not (OUT/'scores').exists()
    leak=read(OUT/'leakage_audit.json');assert leak['score_gate']=='PASS'
    records=[];seals={};supports=[]
    for r in p['rows']:
        cells=[]
        for arm in ARMS:
            path=OUT/'features'/r['id']/(arm+'.json');z=read(path);seals[str(path)]=sha(path)
            if z['status']=='COMPLETE':
                assert sha(z['path'])==z['sha256'];seals[z['path']]=z['sha256'];cells.append(z);records.append(z)
        eligible=len(cells)==3 and all(z['joint_L']>40 for z in cells)
        supports.append({'id':r['id'],'speaker':r['speaker'],'eligible':eligible,'lengths':{z['arm']:z['joint_L'] for z in cells},'reason':None if eligible else 'missing_arm_or_joint_L_not_above_40'})
        if not eligible:continue
        for z in cells:
            L=z['joint_L'];k=3;J=np.arange(max(0,-k),min(L,L-k));region=np.full(L,-1,dtype=np.int32);region[J]=np.where(J<20,0,np.where(J<L-20,1,2))
            pi,sizes=generate_perms(L,J,region,np.zeros(L,dtype=np.int32),f'holdout|{r["id"]}|{z["arm"]}|{k}','whole',256)
            assert np.all(region[pi[:,J]]==region[J]) and np.all(np.sort(pi[:,20:L-20],axis=1)==np.arange(20,L-20))
            path=OUT/'permutations'/r['id']/(z['arm']+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,whole=pi,J=J,region=region);seals[str(path)]=sha(path)
    write(OUT/'support.json',supports);write(OUT/'feature_records.json',records);write(OUT/'permutation_seal.json',seals)
    write(OUT/'score_lock.json',{'status':'sealed_before_any_distance_scoring','protocol_sha256':sha(OUT/'protocol.json'),'leakage_audit_sha256':sha(OUT/'leakage_audit.json'),'support_sha256':sha(OUT/'support.json'),'records_sha256':sha(OUT/'feature_records.json'),'permutation_seal_sha256':sha(OUT/'permutation_seal.json'),'analysis_code_sha256':sha(__file__),'repeats':256,'eligible_records':sum(s['eligible'] for s in supports),'eligible_speakers':len({s['speaker'] for s in supports if s['eligible']})})
    print('score indices sealed',sum(s['eligible'] for s in supports),flush=True)


def features(z,geom):
    a=np.load(z['path']);L=z['joint_L'];v,t=a['visual'][:L],a['audio'][:L]
    if geom=='unit':v=(v.astype(float)/np.linalg.norm(v.astype(float),axis=1)[:,None]).astype(np.float32);t=(t.astype(float)/np.linalg.norm(t.astype(float),axis=1)[:,None]).astype(np.float32)
    return v,t


def native_policy(matrix,policy,k=3):
    L=len(matrix)
    if policy=='valid':
        curve=np.array([matrix[np.maximum(0,-s):np.minimum(L,L-s),s+15].mean() for s in range(-15,16)])
        return dict(zip(FIELDS,curve_metrics(curve,k)[0].tolist()))
    guard=int(policy[5:]);m=matrix[guard:L-guard] if guard else matrix
    if not len(m):return None
    return dict(zip(FIELDS,curve_metrics(m.mean(0),k)[0].tolist()))


def analyze():
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    torch.set_num_threads(2);p=verify_protocol();lock=read(OUT/'score_lock.json');assert sha(__file__)==lock['analysis_code_sha256']
    for path,h in read(OUT/'permutation_seal.json').items():assert sha(path)==h,path
    records=read(OUT/'feature_records.json');rr={(z['id'],z['arm']):z for z in records};support=read(OUT/'support.json');ids=[r['id'] for r in support if r['eligible']]
    for z in records:
        dest=OUT/'scores/native'/z['id']/(z['arm']+'.json')
        if dest.exists():continue
        v,a=features(z,'raw');matrix=SyncNetEngine.distance_matrix(v,a)
        policies={q:native_policy(matrix,q) for q in ('guard0','valid','guard15','guard20') if q in ('guard0','valid') or len(v)>2*int(q[5:])}
        write(dest,{'id':z['id'],'speaker':z['speaker'],'arm':z['arm'],'L':len(v),'policies':policies})
    for geom in ('raw','unit'):
        for sid in ids:
            for ta in ('Q1','Q2'):
                dest=OUT/'scores/pairs'/geom/sid/(ta+'.json')
                if dest.exists():continue
                kresults={};arrays={};controls={'identity':0.,'anchor64':0.,'anchor32':0.,'gram_vs_direct':0.,'permutation_anchor':0.}
                for k in (2,3,4):
                    c={a:coordinates(*features(rr[sid,a],geom),k) for a in ('N',ta)};alpha=c['N']['sigma']/c[ta]['sigma'];scores={};perms={}
                    for label,arm,aa in [('N_base','N',1.),('T_base',ta,1.),('N_M','N',1/alpha),('T_M',ta,alpha)]:
                        cc=c[arm];v,a=transform(cc,aa,1.);dm=distances(v,a);s=curve_score(dm,k);s['C_anchor']=s['B']-s['D_anchor'];scores[label]=s
                        if label.endswith('_base'):controls['identity']=max(controls['identity'],float(np.max(abs(v-cc['v']))),float(np.max(abs(a-cc['a']))))
                        if label.endswith('_M'):
                            J=cc['t'];old=cc['a'][J+k]-cc['v'][J];new=a[J+k]-v[J];controls['anchor64']=max(controls['anchor64'],float(abs(new-old).max()));controls['anchor32']=max(controls['anchor32'],float(abs((a[J+k].astype(np.float32)-v[J].astype(np.float32))-old).max()))
                        if k!=3:continue
                        J=cc['t'];lo=int(J[0]);I=cc['i'];q=I[:,None]+np.arange(-15,16)-k;d=gram(v[J],a[J+k]);ident=d[I[:,None]-lo,q-lo];controls['gram_vs_direct']=max(controls['gram_vs_direct'],float(abs(ident-dm).max()))
                        pi=np.load(OUT/'permutations'/sid/(arm+'.npz'))['whole'];curves=np.stack([d[x[I,None]-lo,x[q]-lo].mean(0) for x in pi]);values=curve_metrics(curves,k);controls['permutation_anchor']=max(controls['permutation_anchor'],float(abs(values[:,3]-s['D_anchor']).max()));perms[label]=dict(zip(FIELDS,values.mean(0).tolist()));arrays[label]=curves
                        del d
                    kresults[str(k)]={'alpha_T2N':alpha,'sigma_N':c['N']['sigma'],'sigma_T':c[ta]['sigma'],'rho_N':c['N']['rho'],'rho_T':c[ta]['rho'],'scores':scores,'permuted':perms}
                dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**arrays)
                write(dest,{'id':sid,'speaker':rr[sid,'N']['speaker'],'T_arm':ta,'geometry':geom,'k_results':kresults,'controls':controls,'curves_sha256':sha(dest.with_suffix('.npz'))})
            print('analyzed',geom,sid,flush=True)
    summarize()


def summarize():
    p=read(OUT/'protocol.json');supports=read(OUT/'support.json');eligible={r['id'] for r in supports if r['eligible']};spk={r['id']:r['speaker'] for r in p['rows']};native={};out={}
    for r in p['rows']:
        sid=r['id'];native[sid]={a:read(OUT/'scores/native'/sid/(a+'.json')) for a in ARMS if (OUT/'scores/native'/sid/(a+'.json')).exists()}
    eos_ids={r['id'] for r in p['rows'] if all(read(OUT/'tts'/r['id']/(a+'.json')).get('termination',{}).get('stop_reason')=='EOS' for a in ('Q1','Q2'))}
    for support in ('maximum','common_guard20','common_guard20_all_Q_EOS'):
        out[support]={}
        for policy in ('guard0','valid','guard15','guard20'):
            ids=[sid for sid,rr in native.items() if (support=='maximum' or sid in eligible) and (support!='common_guard20_all_Q_EOS' or sid in eos_ids) and len(rr)==3 and all(policy in r['policies'] for r in rr.values())]
            result={'ids':ids,'n':len(ids),'speakers':len({spk[s] for s in ids}),'contrasts':{}}
            for field in FIELDS:
                vals=[np.mean([native[s][a]['policies'][policy][field] for a in ('Q1','Q2')])-native[s]['N']['policies'][policy][field] for s in ids];result['contrasts'][field]=stats(vals,[spk[s] for s in ids]) if ids else None
            result['arms']={a:{f:stats([native[s][a]['policies'][policy][f] for s in ids],[spk[s] for s in ids]) for f in FIELDS} for a in ARMS} if ids else {}
            out[support][policy]=result
    medians={a:float(np.median([r[a]['policies']['guard20']['best_lag'] for r in native.values() if a in r and 'guard20' in r[a]['policies']])) for a in ARMS}
    write(OUT/'calibration_transfer_gate.json',{'frozen_k0':3,'arm_best_lag_medians':medians,'passed':max(medians.values())-min(medians.values())<=1 and all(abs(x-3)<=1 for x in medians.values()),'policy':'do not alter k or support even on failure'})
    result={}
    for geom in ('raw','unit'):
        for k in (2,3,4):
            by={sid:[read(OUT/'scores/pairs'/geom/sid/(a+'.json'))['k_results'][str(k)] for a in ('Q1','Q2')] for sid in sorted(eligible)}
            contrasts=defaultdict(dict)
            for field in FIELDS:
                vals=defaultdict(list)
                for sid,pairs in by.items():
                    qs=defaultdict(list)
                    for r in pairs:
                        s=r['scores'];b=s['T_base'][field]-s['N_base'][field];tm=s['T_M'][field]-s['N_base'][field];nm=s['T_base'][field]-s['N_M'][field]
                        q={'original':b,'T2N_effect':s['T_M'][field]-s['T_base'][field],'T2N_residual':tm,'N2T_effect':s['N_M'][field]-s['N_base'][field],'N2T_residual':nm}
                        if k==3:
                            z=r['permuted'];perm=z['T_base'][field]-z['N_base'][field];joint=z['T_M'][field]-z['N_base'][field];rev=z['T_base'][field]-z['N_M'][field]
                            q.update(whole_gap=perm,whole_effect=perm-b,T2N_joint=joint,T2N_interaction=joint-tm-perm+b,N2T_joint=rev,N2T_interaction=rev-nm-perm+b,N_whole_effect=z['N_base'][field]-s['N_base'][field],T_whole_effect=z['T_base'][field]-s['T_base'][field])
                        for name,value in q.items():qs[name].append(value)
                    for name,value in qs.items():vals[name].append(float(np.mean(value)))
                for name,values in vals.items():contrasts[name][field]=stats(values,[spk[s] for s in by])
            coefficients={f:stats([np.mean([r[f] for r in pairs]) for pairs in by.values()],[spk[s] for s in by]) for f in ('alpha_T2N','sigma_N','sigma_T','rho_N','rho_T')}
            result[f'{geom}/k{k}']={'ids':list(by),'contrasts':contrasts,'coefficients':coefficients}
    write(OUT/'native_summary.json',out);write(OUT/'mechanism_summary.json',result)
    generation=[]
    for r in p['rows']:
        generation.append({'id':r['id'],'speaker':r['speaker'],'natural_seconds':r['natural_seconds'],'tts':{a:read(OUT/'tts'/r['id']/(a+'.json')) for a in ('Q1','Q2')},'common_guard20':r['id'] in eligible})
    write(OUT/'generation_distribution.json',{'rows':generation,'primary_anomalies_retained':True,'all_Q_EOS_ids':sorted(eos_ids),'other_ids':sorted(set(native)-eos_ids)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('extract','seal','analyze','summarize'));globals()[p.parse_args().stage]()
