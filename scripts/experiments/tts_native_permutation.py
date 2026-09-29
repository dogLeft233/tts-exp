"""Paired temporal permutation at fixed native query/lag support, CPU only."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.tts_native_boundary_audit import read,write,sha,stats
from scripts.experiments.tts_native_midpoint import inputs,coordinates,transform

MID=ROOT/'runs/tts_native_midpoint_20260926'
SOURCE=ROOT/'runs/tts_chinese_instance_primed_20260926'
OUT=ROOT/'runs/tts_native_permutation_20260926'
FIELDS=('C','B','D','D_anchor','C_anchor','best_lag')


def phone_intervals(path):
    text=Path(path).read_text().split('name = "phones"',1)[1].split('item [',1)[0]
    matches=re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',text)
    spans=[(float(a),float(b),label) for a,b,label in matches if float(b)>float(a)]
    assert spans and all(spans[j][1]<=spans[j+1][0]+1e-8 for j in range(len(spans)-1))
    return spans


def labels_at(times,spans):
    ends=np.array([s[1] for s in spans]);labels=[]
    for time in times:
        j=int(np.searchsorted(ends,time,side='right'))
        if j<len(spans) and spans[j][0]<=time<spans[j][1]:
            labels.append('occ_'+str(j)+'_'+spans[j][2])
        else:
            labels.append('gap_'+str(j))
    unique={x:i for i,x in enumerate(dict.fromkeys(labels))}
    return np.array([unique[x] for x in labels],dtype=np.int32),list(unique)


def generate_perms(n,domain,region,phone,seed_key,mode,repeats=256):
    pi=np.tile(np.arange(n,dtype=np.int32),(repeats,1))
    keys=region[domain] if mode=='whole' else region[domain]*(phone.max()+1)+phone[domain]
    blocks=[domain[keys==k] for k in sorted(set(keys))]
    for b in range(repeats):
        seed=int.from_bytes(hashlib.sha256(f'20260926|{seed_key}|{mode}|{b}'.encode()).digest()[:16],'big')
        rng=np.random.Generator(np.random.PCG64(seed))
        for block in blocks:
            pi[b,block]=rng.permutation(block)
    return pi,[len(b) for b in blocks]


def freeze():
    if (OUT/'protocol.json').exists():raise ValueError('already frozen')
    parent=read(MID/'protocol.json')
    manifest={r['id']:r for r in read(SOURCE/'manifest.json')['rows']}
    seals={};assets={};records=[]
    for group,g in parent['groups'].items():
        ids=g['eligible_ids'] if group=='dynamic_cloud' else g['eval_ids']
        for r in g['records']:
            sid,arm=r['id'],r['arm']
            if sid not in ids:continue
            n,k=r['joint_L'],g['k0'];domain=np.arange(max(0,-k),min(n,n-k))
            region=np.full(n,-1,dtype=np.int32);region[domain]=np.where(domain<20,0,np.where(domain<n-20,1,2))
            a='C' if arm=='T' else arm;grid=manifest[sid].get('grids',{}).get(a)
            phone=np.full(n,-1,dtype=np.int32);spans=None
            if grid and Path(grid['path']).exists():
                assert sha(grid['path'])==grid['sha256'];assets[grid['path']]=grid['sha256']
                spans=phone_intervals(grid['path'])
                phone[domain],vocab=labels_at((domain+k)*.04+.1075,spans)
            else:vocab=[]
            key=f'{group}/{sid}/{arm}';dest=OUT/'permutations'/f'{key}.npz'
            arrays={'domain':domain,'region':region,'phone_labels':phone};meta={}
            for mode in ('whole','phone'):
                if mode=='phone' and spans is None:continue
                pi,sizes=generate_perms(n,domain,region,phone,f'{r["panel"]}|{sid}|{arm}|{k}',mode)
                assert np.all(region[pi[:,domain]]==region[domain])
                if mode=='phone':assert np.all(phone[pi[:,domain]]==phone[domain])
                assert np.all(np.sort(pi[:,20:n-20],axis=1)==np.arange(20,n-20))
                arrays[mode]=pi
                meta[mode]={'block_sizes':sizes,'moved_J_first64':float(np.mean(pi[:64,domain]!=domain)),
                            'moved_I_first64':float(np.mean(pi[:64,20:n-20]!=np.arange(20,n-20))),
                            'moved_J_256':float(np.mean(pi[:,domain]!=domain)),
                            'block_quantiles':np.quantile(sizes,[0,.25,.5,.75,1]).tolist()}
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
            receipt={'key':key,'n':n,'k0':k,'labels':vocab,'spans':spans,'grid':grid,'modes':meta,'path':str(dest),'sha256':sha(dest)}
            write(dest.with_suffix('.json'),receipt)
            seals[str(dest)]=sha(dest);seals[str(dest.with_suffix('.json'))]=sha(dest.with_suffix('.json'))
            records.append(receipt)
    write(OUT/'permutation_seal.json',seals);write(OUT/'annotation_hashes.json',assets)
    write(OUT/'protocol.json',{'status':'frozen_before_permutation_scores','parent':parent,'records':records,
        'parent_sha256':sha(MID/'protocol.json'),'permutation_seal_sha256':sha(OUT/'permutation_seal.json'),
        'annotation_manifest_sha256':sha(SOURCE/'manifest.json'),'code_sha256_at_freeze':sha(Path(__file__)),
        'repeats':'First64; all256 permutations presealed. If any reported group C/C_anchor gap, permutation effect or interaction differs by >.02 between first32 and64, all groups/modes/geometries use256.',
        'labels':'(t+k0)*.04+.1075; own MFA occurrence, including each silence interval; each uncovered continuous gap distinct; region x occurrence blocks. Center label does not fix complete 200ms context.',
        'score':'D(V_pi(i),Aaligned_pi(i+s-k0)), i20:L-20, s=-15..15. Same pi for raw/unit and M conditions, N shared across Q1/Q2.',
        'distance':'Float32 reconstructed embeddings, float64 Gram squared distance with eps1e-6; independently checked against direct float32 vector norm and old native matrices. One fullJ matrix at a time.',
        'metrics':FIELDS,'statistics':'Per-permutation C before repeat mean; then Q instances within record, speaker equal,20k PCG64(20260926),99CI.',
        'interaction':'(M+perm gap - M-only gap) - (perm gap - original gap); reverse N->T sensitivity also included.',
        'limits':'Measures score dependence on representation time organization; no physical reordered audio/video, no true-phone claim or new support gate.'})
    print('sealed',len(records),'arms',len(seals),'files; phone available',sum('phone' in r['modes'] for r in records))


def gram(v,a):
    v,a=np.asarray(v,dtype=np.float32).astype(float),np.asarray(a,dtype=np.float32).astype(float)
    vv=(v*v).sum(1);aa=(a*a).sum(1);ve=v.sum(1);ae=a.sum(1)
    out=np.empty((len(v),len(a)),dtype=np.float64)
    for start in range(0,len(v),128):
        stop=min(start+128,len(v));q=vv[start:stop,None]+aa[None,:]-2*v[start:stop]@a.T
        q+=2e-6*(ve[start:stop,None]-ae[None,:])+v.shape[1]*1e-12
        out[start:stop]=np.sqrt(np.maximum(q,0))
    return out


def curve_metrics(curves,k):
    z=np.atleast_2d(curves);j=z.argmin(1);d=z[np.arange(len(z)),j];b=np.median(z,axis=1);anchor=z[:,k+15]
    return np.stack([b-d,b,d,anchor,b-anchor,j-15],axis=1)


def build_matrices():
    protocol=read(OUT/'protocol.json')
    for p,h in read(OUT/'permutation_seal.json').items():assert sha(p)==h
    for p,h in read(OUT/'annotation_hashes.json').items():assert sha(p)==h
    for p,h in read(ROOT/'runs/tts_native_boundary_audit_20260926/source_hashes.json').items():assert sha(p)==h
    controls=[]
    for group,g in protocol['parent']['groups'].items():
        ids=g['eligible_ids'] if group=='dynamic_cloud' else g['eval_ids']
        rows={(r['id'],r['arm']):r for r in g['records']}
        for geom in ('raw','unit'):
            for sid in ids:
                for ta in g['arms'][1:]:
                    dest=OUT/'matrices'/group/geom/sid/ta
                    if (dest/'receipt.json').exists():continue
                    c={a:coordinates(*inputs(rows[sid,a],geom),g['k0']) for a in ('N',ta)}
                    ratio=c['N']['sigma']/c[ta]['sigma'];metadata={};errors={'identity_replay':0.,'anchor_multiset':0.,'energy_invariance':0.}
                    baseline=read(MID/'pairs'/group/geom/sid/f'{ta}.json')
                    for label,(arm,alpha) in {'N_base':('N',1.),'T_base':(ta,1.),'N_M':('N',1/ratio),'T_M':(ta,ratio)}.items():
                        cc=c[arm];v,a=transform(cc,alpha,1.);t=cc['t'];lo=int(t[0]);k=g['k0'];ix=cc['i'];q=ix[:,None]+np.arange(-15,16)-k
                        d=gram(v[t],a[t+k]);dest.mkdir(parents=True,exist_ok=True);np.save(dest/f'{label}.npy',d)
                        ident=d[ix[:,None]-lo,q-lo].mean(0);metrics=curve_metrics(ident,k)[0]
                        expected=baseline['scores'][label]
                        error=max(abs(metrics[j]-expected[f]) for j,f in enumerate(FIELDS) if f in expected)
                        errors['identity_replay']=max(errors['identity_replay'],float(error));assert error<2e-5
                        pp=np.load(OUT/'permutations'/group/sid/f'{arm}.npz')
                        energies=((cc['M'][ix]-cc['mu'])**2).sum(1);renergy=(cc['R'][ix]**2).sum(1)
                        diag=d[ix-lo,ix-lo]
                        for mode in ('whole','phone'):
                            if mode not in pp:continue
                            pi=pp[mode]
                            assert np.array_equal(np.sort(d[pi[:,ix]-lo,pi[:,ix]-lo],axis=1),np.broadcast_to(np.sort(diag),(len(pi),len(ix))))
                            order=pi[:,ix]-20
                            ee=max(float(np.max(abs(energies[order].mean(1)-energies.mean()))),float(np.max(abs(renergy[order].mean(1)-renergy.mean()))))
                            errors['energy_invariance']=max(errors['energy_invariance'],ee)
                        rng=np.random.default_rng(20260926);probe=rng.integers(len(t),size=(512,2));values=d[probe[:,0],probe[:,1]]
                        np.savez_compressed(dest/f'{label}_probes.npz',indices=probe,values=values)
                        metadata[label]={'arm':arm,'alpha':alpha,'lo':lo,'L':len(v),'identity_curve':ident.tolist(),'identity_metrics':dict(zip(FIELDS,metrics.tolist())),
                                         'matrix_sha256':sha(dest/f'{label}.npy'),'probes_sha256':sha(dest/f'{label}_probes.npz')}
                        del d
                    write(dest/'receipt.json',{'group':group,'geometry':geom,'id':sid,'speaker':rows[sid,'N']['speaker'],'T_arm':ta,'k0':g['k0'],'labels':metadata,'controls':errors})
                    controls.append(errors)
            print('matrices',group,geom,flush=True)
    write(OUT/'matrix_controls.json',{k:max(r[k] for r in controls) for k in controls[0]} if controls else {'resumed':True})


def score(repeats):
    for rp in sorted((OUT/'matrices').glob('*/*/*/*/receipt.json')):
        info=read(rp);dest=OUT/'scores'/str(repeats)/info['group']/info['geometry']/info['id']/f"{info['T_arm']}.npz"
        if dest.exists():continue
        arrays={}
        for label,meta in info['labels'].items():
            pp=np.load(OUT/'permutations'/info['group']/info['id']/f"{meta['arm']}.npz")
            ix=np.arange(20,meta['L']-20);q=ix[:,None]+np.arange(-15,16)-info['k0'];lo=meta['lo']
            d=np.load(rp.parent/f'{label}.npy',mmap_mode='r')
            arrays[label+'_identity']=np.array(meta['identity_curve'])[None,:]
            for mode in ('whole','phone'):
                if mode not in pp:continue
                curves=[]
                for pi in pp[mode][:repeats]:curves.append(d[pi[ix,None]-lo,pi[q]-lo].mean(0))
                arrays[label+'_'+mode]=np.array(curves)
            del d
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
    print('scored permutations',repeats,flush=True)


def summarize(repeats):
    protocol=read(OUT/'protocol.json');out={};mc={};pair_cache={}
    for group,g in protocol['parent']['groups'].items():
        for geom in ('raw','unit'):
            for support in (('eval','full74_bridge') if group=='dynamic_cloud' else (('eval','exclude_known_failure') if group=='static_local' else ('eval',))):
                ids=g['eligible_ids'] if support=='full74_bridge' else g['eval_ids']
                if support=='exclude_known_failure':ids=[sid for sid in ids if sid!='a1_005']
                for mode in ('whole','phone'):
                    rr={};missing=[]
                    for sid in ids:
                        pairs=[]
                        for ta in g['arms'][1:]:
                            path=OUT/'scores'/str(repeats)/group/geom/sid/f'{ta}.npz'
                            with np.load(path) as data:
                                if any(label+'_'+mode not in data for label in ('N_base','T_base','N_M','T_M')):break
                                pair={}
                                for length in (32,64,repeats):
                                    v={key:curve_metrics(data[key][:length],g['k0']).mean(0) for key in data.files if key.endswith('_identity') or key.endswith('_'+mode)}
                                    f00=v['T_base_identity']-v['N_base_identity'];f01=v['T_base_'+mode]-v['N_base_'+mode]
                                    f10=v['T_M_identity']-v['N_base_identity'];f11=v['T_M_'+mode]-v['N_base_'+mode]
                                    r10=v['T_base_identity']-v['N_M_identity'];r11=v['T_base_'+mode]-v['N_M_'+mode]
                                    pair[length]={'original':f00,'permuted':f01,'permutation_change':f01-f00,'M_only':f10,'M_permuted':f11,
                                                  'interaction':f11-f10-f01+f00,'M_effect_after_permutation':f11-f01,
                                                  'reverse_M_only':r10,'reverse_M_permuted':r11,'reverse_interaction':r11-r10-f01+f00,
                                                  'N_permutation_change':v['N_base_'+mode]-v['N_base_identity'],
                                                  'T_permutation_change':v['T_base_'+mode]-v['T_base_identity']}
                                pairs.append(pair)
                        if len(pairs)==len(g['arms'])-1:rr[sid]=pairs
                        else:missing.append(sid)
                    if not rr:continue
                    speakers=[next(r['speaker'] for r in g['records'] if r['id']==sid) for sid in rr]
                    result={'ids':list(rr),'missing':missing,'contrasts':{},'mc32_vs64':{}}
                    for contrast in next(iter(rr.values()))[0][repeats]:
                        result['contrasts'][contrast]={}
                        for j,field in enumerate(FIELDS):
                            values=[np.mean([x[repeats][contrast][j] for x in pairs]) for pairs in rr.values()]
                            result['contrasts'][contrast][field]=stats(values,speakers)
                            if field in ('C','C_anchor'):
                                delta=[np.mean([x[64][contrast][j]-x[32][contrast][j] for x in pairs]) for pairs in rr.values()]
                                st=stats(delta,speakers)
                                result['mc32_vs64'][contrast+'/'+field]=st['mean']
                                mc[f'{group}/{geom}/{support}/{mode}/{contrast}/{field}']=st['mean']
                    out[f'{group}/{geom}/{support}/{mode}']=result
    write(OUT/f'summary_{repeats}.json',out)
    worst=max(mc,key=lambda k:abs(mc[k]));gate={'initial_repeats':64,'max_abs_group_32vs64':abs(mc[worst]),'worst':worst,'signed_difference':mc[worst],'escalate_all_256':abs(mc[worst])>.02}
    write(OUT/f'mc_gate_{repeats}.json',gate)
    print(gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','matrices','score','summarize'));p.add_argument('--repeats',type=int,default=64,choices=(64,256));a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='matrices':build_matrices()
    elif a.stage=='score':score(a.repeats)
    else:summarize(a.repeats)
