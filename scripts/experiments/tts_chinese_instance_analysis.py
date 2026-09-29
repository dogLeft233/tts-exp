"""Fixed native and same-event 3x3/4x4 analyses; no outcome-dependent tuning."""
from collections import Counter
import argparse
import itertools
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,ARMS,read,write,sha
from scripts.experiments.tts_negative_pool import PHASES,ACENTER,dist,norm,rank
from scripts.experiments.tts_phone_video_transfer import phones
from scripts.experiments.tts_pcm_residual import cluster

def stats(rows):
    names=sorted({k for r in rows for k in r['effects']})
    return {k:cluster([r['effects'][k] for r in rows if k in r['effects']],[r['speaker'] for r in rows if k in r['effects']]) for k in names}

def load(r,im,a):
    f=BASE/'scores'/im/r['id']/(a+'.npz')
    if not f.exists():return None
    assert sha(f)==read(f.with_suffix('.json'))['sha256']
    return np.load(f)

def calibrate():
    p=read(BASE/'protocol.json');groups={a:[] for a in ARMS};assets={}
    for r in p['rows']:
        if r['split']!='calibration':continue
        for a in ARMS:
            f=BASE/'scores/3'/r['id']/(a+'.json')
            if f.exists() and '20' in read(f)['metrics']:groups[a].append(-read(f)['metrics']['20']['offset']);assets[str(f)]=sha(f)
    primary=sum([groups[a] for a in ('N','Q1','Q2')],[])
    medians={a:float(np.median(v)) for a,v in groups.items() if v}
    write(BASE/'calibration.json',{'k':int(np.median(primary)),'medians':medians,'groups':groups,'transfer_gate':max(medians[a] for a in ('N','Q1','Q2'))-min(medians[a] for a in ('N','Q1','Q2'))<=1,'assets':assets,'C_excluded_from_selection':True})

def native():
    p=read(BASE/'protocol.json');result={}
    for im in ('3','6','9'):
        rows=[];excluded=[];maximal=[]
        for r in p['rows']:
            if r['split']!='evaluation' or (im!='3' and r['id'] not in p['sensitivity_ids']):continue
            paths={a:BASE/'scores'/im/r['id']/(a+'.json') for a in ARMS}
            if not all(f.exists() for f in paths.values()):excluded.append(r['id']);continue
            cells={a:read(f)['metrics'] for a,f in paths.items()};eff={};wide={}
            for guard in ('0','15','20'):
                if all(guard in cells[a] for a in ('N','Q1','Q2')):wide[guard+'_GQ']=(cells['Q1'][guard]['C']+cells['Q2'][guard]['C'])/2-cells['N'][guard]['C']
                if all(guard in cells[a] for a in ('N','C')):wide[guard+'_GC']=cells['C'][guard]['C']-cells['N'][guard]['C']
                if not all(guard in z for z in cells.values()):continue
                cs={a:cells[a][guard]['C'] for a in ARMS}
                gq=(cs['Q1']+cs['Q2'])/2-cs['N'];gc=cs['C']-cs['N']
                eff.update({f'{guard}_GQ':gq,f'{guard}_GC':gc,f'{guard}_GQ-GC':gq-gc,**{f'{guard}_level_{a}':v for a,v in cs.items()}})
            rows.append({'id':r['id'],'speaker':r['speaker'],'effects':eff})
            maximal.append({'id':r['id'],'speaker':r['speaker'],'effects':wide})
            if not eff:excluded.append({'id':r['id'],'reason':'guard20_rows_insufficient_in_one_or_more_arms','arms':[a for a,z in cells.items() if '20' not in z]})
        result[im]={'analysis':stats(rows),'rows':rows,'excluded':excluded,'maximal_pair_analysis':stats(maximal),'maximal_pair_rows':maximal}
    write(BASE/'native.json',result)

def support(r,arms,lens,k):
    ph={a:phones(r['grids'][a]['path']) for a in arms}
    if any([v[0] for v in ph[a]]!=[v[0] for v in ph[arms[0]]] for a in arms):return {'id':r['id'],'speaker':r['speaker'],'eligible':False,'reason':'phone_sequence_mismatch'}
    events=[]
    for n,p in enumerate(ph[arms[0]]):
        events.append({'event':f'phone_{n}','phone':p[0],'span':{a:list(ph[a][n][1:]) for a in arms}})
        if n+1<len(ph[arms[0]]):events.append({'event':f'gap_{n}','phone':'__gap__','span':{a:[ph[a][n][2],ph[a][n+1][1]] for a in arms}})
    nodes=[];rejected=Counter()
    for e in events:
        if any(e['span'][a][1]-e['span'][a][0]<.08-1e-9 for a in arms):rejected['duration']+=1;continue
        for phase in PHASES:
            indices={};errors={};valid=True
            for a in arms:
                lo,hi=e['span'][a];t=lo+phase*(hi-lo);j=int(np.floor((t-ACENTER)*25+.5));actual=j/25+ACENTER
                indices[a]=j;errors[a]=actual-t
                valid &= lo<=actual<hi and abs(actual-t)<=.02000001 and 0<=j<lens[a] and 0<=j-k<lens[a]
            if valid:nodes.append({**e,'phase':phase,'j':indices,'errors':errors})
            else:rejected['grid_extent']+=1
    queries=[]
    for qi,q in enumerate(nodes):
        if q['phone']=='__gap__':continue
        if not all(20<=q['j'][a]-k<lens[a]-20 for a in arms):rejected['guard']+=1;continue
        donors=[di for di,d in enumerate(nodes) if di!=qi and all(3<=abs(d['j'][a]-q['j'][a])<=15 for a in arms)]
        if len(donors)<5:rejected['donors']+=1;continue
        queries.append({'node':qi,'donors':donors})
    occurrences=len({nodes[q['node']]['event'] for q in queries})
    return {'id':r['id'],'speaker':r['speaker'],'eligible':len(queries)>=8 and occurrences>=5,'nodes':nodes,'queries':queries,'n_occurrences':occurrences,'rejected':dict(rejected),'lengths':lens,'reason':None if len(queries)>=8 and occurrences>=5 else 'query_or_occurrence_support'}

def contrasts(q):
    stt=.5*(q['Q1Q1']+q['Q2Q2']-q['Q1Q2']-q['Q2Q1'])
    snt=.25*(2*q['NN']+q['Q1Q1']+q['Q2Q2']-q['NQ1']-q['Q1N']-q['NQ2']-q['Q2N'])
    return {'STT':stt,'SNT':snt,'SNT-STT':snt-stt,
            'Tdiag-NN':.5*(q['Q1Q1']+q['Q2Q2'])-q['NN'],
            'Toffdiag-NN':.5*(q['Q1Q2']+q['Q2Q1'])-q['NN'],
            **{'q_'+k:v for k,v in q.items()}}

def mechanism():
    p=read(BASE/'protocol.json');manifest=read(BASE/'manifest.json');k0=read(BASE/'calibration.json')['k']
    for delta in (0,-1,1):
        k=k0+delta
        for label,arms in [('primary',('N','Q1','Q2')),('mfa_only',('N','Q1','Q2')),('strict',('N','Q1','Q2')),('cloud4',ARMS)]:
            supports=[];cells=[]
            for r in manifest['rows']:
                if r['split']!='evaluation':continue
                if not all(a in r['grids'] for a in arms):supports.append({'id':r['id'],'speaker':r['speaker'],'eligible':False,'reason':'missing_grid'});continue
                mats={a:load(r,'3',a) for a in arms}
                if any(m is None for m in mats.values()):supports.append({'id':r['id'],'speaker':r['speaker'],'eligible':False,'reason':'missing_score'});continue
                for a in arms:assert sha(r['grids'][a]['path'])==r['grids'][a]['sha256']
                lens={a:len(mats[a]['matrix']) for a in arms};s=support(r,arms,lens,k)
                s['mfa_event_eligible']=s['eligible'];s['quality']={}
                for a in arms:
                    af=BASE/'quality/asr'/r['id']/(a+'.json')
                    stop=read(BASE/'tts'/r['id']/(a+'.json')).get('termination',{}).get('stop_reason') if a.startswith('Q') else ('not_applicable' if a=='N' else 'stop_unknown')
                    s['quality'][a]={'cer':read(af)['cer'] if af.exists() else None,'stop':stop}
                if label!='mfa_only':
                    limit=0 if label=='strict' else .10
                    qpass=all(z['cer'] is not None and z['cer']<=limit and (not a.startswith('Q') or z['stop']=='EOS') for a,z in s['quality'].items())
                    if not qpass and s['eligible']:s.update(eligible=False,reason='quality_gate')
                supports.append(s)
                if not s['eligible']:continue
                audio=np.load(BASE/'audio_features'/(r['id']+'.npz'))
                for im in ('3','6','9'):
                    if im!='3' and r['id'] not in p['sensitivity_ids']:continue
                    vm={a:load(r,im,a)['visual'][:lens[a]] for a in arms}
                    assert all(len(vm[a])==lens[a] for a in arms)
                    for geo in ('raw','unit'):
                        aa={a:audio[a][:lens[a]].astype(float) for a in arms};vv={a:vm[a].astype(float) for a in arms}
                        if geo=='unit':aa={a:norm(v) for a,v in aa.items()};vv={a:norm(v) for a,v in vv.items()}
                        pairs={}
                        for v,a in itertools.product(arms,repeat=2):
                            values=[]
                            for query in s['queries']:
                                node=s['nodes'][query['node']];i=node['j'][v]-k;j=node['j'][a];donors=[s['nodes'][di]['j'][a] for di in query['donors']]
                                pos=float(dist(vv[v][i],aa[a][j]));neg=dist(vv[v][i],aa[a][donors]);values.append([pos,float(neg.mean()),float(neg.mean()-pos),rank(neg,pos)])
                            pairs[v+a]=dict(zip(('positive','negative','margin','rank'),np.mean(values,axis=0).tolist()))
                        effects={}
                        for metric in ('positive','negative','margin','rank'):
                            effects.update({metric+'_'+name:val for name,val in contrasts({pair:vals[metric] for pair,vals in pairs.items()}).items()})
                        cells.append({'id':r['id'],'speaker':r['speaker'],'image':im,'geometry':geo,'queries':len(s['queries']),'effects':effects})
            result={im:{geo:stats([c for c in cells if c['image']==im and c['geometry']==geo]) for geo in ('raw','unit')} for im in ('3','6','9')}
            out=BASE/'mechanism'/f'{label}_k{k}'
            write(out/'support.json',supports);write(out/'cells.json',cells);write(out/'analysis.json',result)
            native_rows=read(BASE/'native.json')['3']['rows'];ids={s['id'] for s in supports if s['eligible']}
            write(out/'native_on_support.json',stats([r for r in native_rows if r['id'] in ids]))
            print(label,k,'support',sum(s['eligible'] for s in supports),flush=True)

def audit_support():
    p=read(BASE/'protocol.json');k=read(BASE/'calibration.json')['k'];result={}
    for label in ('primary','mfa_only','strict','cloud4'):
        ss=read(BASE/'mechanism'/f'{label}_k{k}'/'support.json');eligible=[s for s in ss if s['eligible']]
        result[label]={'records':len(ss),'eligible':len(eligible),'speakers':len({s['speaker'] for s in eligible}),'mfa_event_eligible':sum(s.get('mfa_event_eligible',False) for s in ss),'queries':sum(len(s.get('queries',[])) for s in eligible),'reasons':dict(Counter(s.get('reason') or 'eligible' for s in ss)),
                       'quality_before_event':sum(all(z.get('cer') is not None and z['cer']<=(0 if label=='strict' else .1) and (not a.startswith('Q') or z['stop']=='EOS') for a,z in s.get('quality',{}).items()) for s in ss if s.get('quality'))}
    groups={kk:{s['id']:s for s in read(BASE/'mechanism'/f'primary_k{kk}'/'support.json') if s['eligible']} for kk in (k-1,k,k+1)}
    common=set.intersection(*(set(x) for x in groups.values()));query_counts={}
    for sid in sorted(common):
        keys=[]
        for ss in groups.values():
            s=ss[sid];keys.append({(s['nodes'][q['node']]['event'],s['nodes'][q['node']]['phase']) for q in s['queries']})
        query_counts[sid]=len(set.intersection(*keys))
    result['lag_common']={'ids':sorted(common),'query_counts':query_counts,'note':'lag sensitivity uses its own frozen annotation/index support, never chooses winner; common IDs/query counts descriptive'}
    # Count quality independently of alignment availability, so the gate order is visible.
    for label,arms in [('primary',('N','Q1','Q2')),('cloud4',ARMS),('strict',('N','Q1','Q2'))]:
        rows=[]
        for r in p['rows']:
            if r['split']!='evaluation':continue
            passed=True
            for a in arms:
                f=BASE/'quality/asr'/r['id']/(a+'.json')
                passed &= f.exists() and read(f)['cer']<=(0 if label=='strict' else .1)
                if a.startswith('Q'):passed &= read(BASE/'tts'/r['id']/(a+'.json')).get('termination',{}).get('stop_reason')=='EOS'
            if passed:rows.append(r)
        result[label]['independent_quality_pass_ids']=[r['id'] for r in rows];result[label]['independent_quality_pass_speakers']=len({r['speaker'] for r in rows})
    quality_rows=[]
    for r in p['rows']:
        if r['split']!='evaluation':continue
        cer={a:read(BASE/'quality/asr'/r['id']/(a+'.json'))['cer'] for a in ARMS}
        failed=[a for a in ('N','Q1','Q2') if cer[a]>.1]
        stops={a:read(BASE/'tts'/r['id']/(a+'.json'))['termination']['stop_reason'] for a in ('Q1','Q2')}
        quality_rows.append({'id':r['id'],'speaker':r['speaker'],'cer':cer,'asr_failed_arms':failed,'stops':stops})
    result['quality_detail']={'rows':quality_rows,
        'asr_failure_combinations':dict(Counter('+'.join(r['asr_failed_arms']) or 'none' for r in quality_rows)),
        'by_arm':{a:{'asr_failure_count':sum(r['cer'][a]>.1 for r in quality_rows),'cer_quantiles':dict(zip(('min','q25','median','q75','q90','max'),np.quantile([r['cer'][a] for r in quality_rows],[0,.25,.5,.75,.9,1]).tolist()))} for a in ARMS},
        'by_speaker':{sp:{'total':sum(r['speaker']==sp for r in quality_rows),'asr_eos_pass':sum(r['speaker']==sp and not r['asr_failed_arms'] and all(z=='EOS' for z in r['stops'].values()) for r in quality_rows)} for sp in sorted({r['speaker'] for r in quality_rows})},
        'known_generation_stop_failures':[{'id':r['id'],'arm':a,'stop':z} for r in quality_rows for a,z in r['stops'].items() if z!='EOS'],
        'interpretation':'ASR disagreement is a proxy failure, not proof of actual misreading. Context-capacity truncation is independently observed generation failure.'}
    write(BASE/'support_audit.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('calibrate','native','mechanism','audit_support'));globals()[p.parse_args().stage]()
