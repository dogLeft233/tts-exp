"""Independent CPU readback: no pipeline/statistics imports or model forwards."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import soundfile as sf
import json,hashlib,subprocess,io,datetime,traceback,unicodedata

RUN=Path('/home/wjj/tts-audio/tts-exp/runs/qwen_nrt_expansion40_20261022_attempt01')
ROLES=('N','R','T1','T2','T3');POLICIES=('common','native_full','equal_count_spread');METRICS=('C','B','Dmin','D0','offset')
files={};errors=[]


def check(condition,message):
    if not condition:errors.append(message);raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def filecheck(path,expected=None):
    path=Path(path);key=str(path)
    if key not in files:files[key]=sha(path)
    if expected is not None:check(files[key]==expected,'file SHA mismatch: '+key)
    return files[key]


def read(path):return json.loads(Path(path).read_text())


def close(a,b,name,tol=1e-12):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64);check(a.shape==b.shape,name+' shape')
    err=float(np.max(abs(a-b))) if a.size else 0.;check(err<=tol,f'{name} error {err}');return err


def bootstrap(values):
    a=np.asarray(values,dtype=np.float64);rng=np.random.Generator(np.random.PCG64(20261022))
    indices=rng.integers(0,len(a),(10000,len(a)));draws=np.add.reduce(a[indices],axis=1)/len(a)
    return dict(n_speakers=len(a),mean=float(np.mean(a)),CI95=np.percentile(draws,[2.5,97.5]).tolist(),
                CI97_5=np.percentile(draws,[1.25,98.75]).tolist(),values=a.tolist(),positive=int(np.sum(a>0)),negative=int(np.sum(a<0)))


def compare_summary(a,b,name):
    for key in ('n_speakers','positive','negative'):check(a[key]==b[key],name+' '+key)
    return max(close(a[key],b[key],name+' '+key) for key in ('mean','CI95','CI97_5','values'))


def summarize_matrix(matrix,selection):
    curve=np.add.reduce(matrix[selection],axis=0,dtype=np.float64)/len(selection);j=int(np.argmin(curve))
    b=float(np.median(curve));d=float(curve[j])
    return dict(C=b-d,B=b,Dmin=d,D0=float(curve[15]),offset=15-j,curve=curve.tolist(),query_count=len(selection),best_lag_at_boundary=j in (0,30))


def edit_distance(reference,hypothesis):
    """Full 2D character DP, independently implemented from ASR's rolling rows."""
    table=np.empty((len(reference)+1,len(hypothesis)+1),dtype=np.int64)
    table[:,0]=np.arange(len(reference)+1);table[0,:]=np.arange(len(hypothesis)+1)
    for i in range(1,len(reference)+1):
        for j in range(1,len(hypothesis)+1):
            table[i,j]=min(table[i-1,j]+1,table[i,j-1]+1,table[i-1,j-1]+int(reference[i-1]!=hypothesis[j-1]))
    return int(table[-1,-1])


def normalized_bulk(texts):
    # Only OpenCC string conversion in its existing CPU environment; no ASR/model load.
    initial=[unicodedata.normalize('NFKC',s).lower() for s in texts]
    command=['/home/wjj/.venvs/tts-asr-audit-20261022/bin/python','-c',
             "import json,sys;from opencc import OpenCC;c=OpenCC('t2s');print(json.dumps([c.convert(s) for s in json.load(sys.stdin)],ensure_ascii=False))"]
    converted=json.loads(subprocess.check_output(command,input=json.dumps(initial,ensure_ascii=False).encode()))
    return [''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith('P')) for s in converted]


def main():
    check((RUN/'analysis.json').is_file() and (RUN/'asr/results.json').is_file() and (RUN/'asr/vad_results.json').is_file(),'wait for complete downstream+ASR')
    p=read(RUN/'protocol.json');analysis=read(RUN/'analysis.json')
    check(filecheck(RUN/'protocol.json')=='70ee7630c9022258fbf4d97b375a9479294046b51f63ca673f3480a42452e4b1','frozen protocol')
    filecheck(RUN/'analysis.json');parent=Path(p['parent_run']);pilot=Path(p['pilot_tts_run'])
    seals=[]
    for root,count in [(parent,2740),(pilot,4333)]:
        seal_path=root/'artifact_manifest.json';binding=p['reused_bindings'][str(seal_path)]
        filecheck(seal_path,binding['sha256']);seal=read(seal_path);check(len(seal['files'])==seal['file_count']==count,'whole seal count')
        total=0
        for row in seal['files']:
            path=root/row['path'];check(path.stat().st_size==row['bytes'],'sealed file size');filecheck(path,row['sha256']);total+=row['bytes']
        check(total==seal['total_bytes'],'seal total bytes');seals.append(dict(root=str(root),file_count=count,total_bytes=total,sha256=sha(seal_path)))
    for path,binding in p['reused_bindings'].items():filecheck(path,binding['sha256'])
    samples={s['sample_id']:s for s in p['samples']};check(len(samples)==40,'40 frozen samples')
    for s in samples.values():
        filecheck(s['source_path'],s['source_sha256']);filecheck(s['input24_path'],s['input24_sha256'])
        filecheck(s['transcript_path'],s['transcript_sha256'])
    print('BOTH_SEALS_7073_FILES_PASS',flush=True)
    audio=read(RUN/'audio_manifest.json');video=read(RUN/'video_manifest.json');features=read(RUN/'feature_manifest.json')
    for name in ['audio_manifest.json','video_manifest.json','feature_manifest.json','generation/tts_manifest.json','generation/codec_manifest.json',
                 'generation/completion.json','generation/reference_bridge.json','asr/results.json','asr/vad_results.json','asr/vad_followup_protocol.json','asr/runtime.json']:
        filecheck(RUN/name)
    check(read(RUN/'generation/reference_bridge.json')['status']=='PASS','old S0002 encode/decode bridge')
    all_tts=read(pilot/'tts/tts_manifest.json')['records']+read(RUN/'generation/tts_manifest.json')['records']
    check(len(all_tts)==120,'all120 predetermined T draws retained')
    tb={(r['sample_id'],f"T{r['seed_index']}"):r for r in all_tts}
    rb={r['sample_id']:r for r in read(RUN/'generation/codec_manifest.json')['records']}
    ab={(r['sample_id'],r['role']):r for r in audio['records']};vb={(r['sample_id'],r['role']):r for r in video['records']};fb={(r['sample_id'],r['role']):r for r in features['records']}
    expected={(sid,role) for sid in samples for role in ROLES}
    check(len(audio['records'])==len(video['records'])==len(features['records'])==200 and set(ab)==set(vb)==set(fb)==expected,'all200 unique groups')
    check(len(tb)==120 and len(rb)==30,'raw manifest counts')
    check(video['weights_sha256']==p['downstream']['weights_sha256'],'W2L weights')
    check(video['inference_sha256']=='42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9','same generation source')
    check(features['weights_sha256']=='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442','same SyncNet weights')
    filecheck(p['downstream']['face'],p['downstream']['face_sha256'])
    audio_checks=[];new_feature_count=0
    for (sid,role),a in ab.items():
        v=vb[sid,role];f=fb[sid,role];s=samples[sid];is_new=s['cohort']=='new30'
        check(a['reused']==v['reused']==f['reused']==(not is_new),'reuse flags')
        check(a['cohort']==v['cohort']==f['cohort']==s['cohort'],'cohort flags')
        filecheck(a['path'],a['sha256']);pcm,psr=sf.read(a['path'],dtype='int16')
        check(psr==16000 and len(pcm)==a['samples'],'PCM source sample count')
        check(hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==a['pcm_sha256'],'PCM source payload SHA')
        filecheck(v['path'],v['sha256']);check(v['audio_sha256']==a['sha256'],'video audio SHA')
        filecheck(f['audio_path'],f['audio_sha256']);filecheck(f['visual_path'],f['visual_sha256'])
        check(f['audio_meta']['pcm_sha256']==a['pcm_sha256'] and f['audio_meta']['sample_count']==len(pcm),'feature PCM binding')
        if not is_new:continue
        raw_path=s['input24_path'] if role=='N' else rb[sid]['path'] if role=='R' else tb[sid,role]['path']
        check(a['source_path']==raw_path,'canonical raw origin')
        filecheck(a['source_path'],a['source_sha256']);x,sr=sf.read(Path(a['path']).with_name(f'{role}_16k_float.wav'),dtype='float32')
        check(sr==16000 and np.isfinite(x).all(),'finite16k float')
        raw24,sr24=sf.read(raw_path,dtype='float32');check(sr24==24000 and np.isfinite(raw24).all(),'raw24k finite')
        data=subprocess.check_output(['/usr/bin/ffmpeg','-nostdin','-v','error','-i',raw_path,'-ac','1','-ar','16000','-c:a','pcm_f32le','-f','f32le','pipe:1'])
        check(np.array_equal(np.frombuffer(data,dtype='<f4'),x),'independent FFmpeg sample exact')
        rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=p['target_rms']/rms
        close(rms,a['rms_before'],'RMS before');close(gain,a['gain'],'gain')
        normalized=x*np.float32(gain);check(float(abs(normalized).max())<1,'no prequantization clipping')
        memory=io.BytesIO();sf.write(memory,normalized,16000,format='WAV',subtype='PCM_16');memory.seek(0);expected_pcm,_=sf.read(memory,dtype='int16')
        check(np.array_equal(pcm,expected_pcm) and len(pcm)==len(x),'exact untrimmed normalized PCM')
        post=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2)));close(post,a['rms_after'],'post RMS')
        cmd=v['receipt']['command'];check(v['receipt']['returncode']==0,'render exit')
        for flag,value in [('--audio',a['path']),('--face',p['downstream']['face']),('--wav2lip_batch_size','4'),('--face_det_batch_size','4')]:
            check(cmd[cmd.index(flag)+1]==value,'render argument '+flag)
        check('--nosmooth' in cmd and cmd[cmd.index('--checkpoint_path')+1].endswith('third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),'generation options')
        new_feature_count+=2
        for value in f['repeat'].values():check(value<=1e-5,'new feature repeat control')
        meta=f['frames_meta'];jpgs=sorted((RUN/'decoded_frames'/sid/role).glob('*.jpg'))
        check(len(jpgs)==meta['frame_count']==len(meta['frame_sha256']),'decoded frame count')
        for frame,h in zip(jpgs,meta['frame_sha256']):filecheck(frame,h)
        audio_checks.append(dict(sample_id=sid,role=role,samples16=len(pcm),samples24=len(raw24),PCM_exact=True,rms_error=abs(post-p['target_rms'])))
    check(len(audio_checks)==150 and new_feature_count==300,'new150 canonical/video and300 feature arrays')
    print('AUDIO_VIDEO150_FEATURE300_BINDINGS_PASS',flush=True)
    # Reliability is independent of SyncNet score or ASR content.
    reliability={}
    for key,t in tb.items():
        sid,role=key;index=int(role[1:]);actual=p['seed_bases'][index-1]+100*int(samples[sid]['speaker_id'][1:])
        check(t['seed']==actual,'fixed speaker seed');ratio=t['samples']/t['sample_rate_hz']/samples[sid]['duration_s']
        reliability[key]=dict(qualified=t['eos_in_talker_sequences'] is True and .5<=ratio<=2.,EOS=t['eos_in_talker_sequences'],duration_ratio=ratio,seed=actual,seed_index=index)
    saved_rel={(r['sample_id'],r['role']):r for r in analysis['all120_draws_reliability']};check(set(saved_rel)==set(reliability),'all120 reliability coverage')
    for key,wanted in reliability.items():
        for field,value in wanted.items():
            if field=='duration_ratio':close(value,saved_rel[key][field],'duration ratio')
            else:check(value==saved_rel[key][field],'reliability '+field)
    ar={r['sample_id']:r for r in analysis['records']};unmeasurable={r['sample_id']:r for r in analysis['unmeasurable']}
    check(set(ar).isdisjoint(unmeasurable) and set(ar)|set(unmeasurable)==set(samples),'measurable coverage explicit')
    old={r['sample_id']:r for r in read(pilot/'analysis.json')['records']}
    rebuilt=[];matrix_checks=[];max_matrix=0.;max_scalar=0.;feature_count=0;metric_cells=0;bridge_max=0.
    for sid,s in samples.items():
        matrices={};counts={};arrays={}
        for role in ROLES:
            f=fb[sid,role];a=np.load(f['audio_path'],allow_pickle=False);v=np.load(f['visual_path'],allow_pickle=False);feature_count+=2
            check(a.ndim==v.ndim==2 and a.shape[1]==v.shape[1]==1024 and np.isfinite(a).all() and np.isfinite(v).all(),'finite actual1024 embeddings')
            nframes=f['frames_meta']['frame_count'];nmfcc=f['audio_meta']['mfcc_shape'][0]
            check(len(v)==nframes-4 and (len(a)-1)*4+20<=nmfcc,'real receptive fields')
            n=min(len(a),len(v));counts[role]=n;arrays[role]=(a,v)
            q=np.arange(15,max(15,n-15),dtype=np.int64);j=q[:,None]+np.arange(-15,16)[None,:]
            if len(q):check(j.min()>=0 and j.max()<n and q.max()+4<nframes,'q+-15 no padding')
            delta=np.asarray(v[q,None,:]-a[j]+np.float32(1e-6),dtype=np.float32)
            independent=np.sqrt(np.add.reduce(delta*delta,axis=2,dtype=np.float32),dtype=np.float32).astype(np.float64)
            path=RUN/'distances'/sid/f'{role}_native.npy';filecheck(path);saved=np.load(path,allow_pickle=False)
            err=close(independent,saved,'NumPy raw distances',1e-5);max_matrix=max(max_matrix,err);matrices[role]=saved
            matrix_checks.append(dict(sample_id=sid,role=role,shape=list(saved.shape),numpy_max_error=err,no_padding=True))
        if min(counts.values())<=30:
            check(sid in unmeasurable and unmeasurable[sid]['counts']==counts,'unmeasurable empty real support');continue
        check(sid in ar and ar[sid]['counts']==counts,'condition feature counts');r=ar[sid]
        native={role:np.arange(15,counts[role]-15) for role in ROLES};k=min(map(len,native.values()));common=np.arange(15,15+k)
        supports={'common':{role:common for role in ROLES},'native_full':native,
                  'equal_count_spread':{role:q[np.rint(np.linspace(0,len(q)-1,k)).astype(np.int64)] for role,q in native.items()}}
        policies={}
        for policy,roles in supports.items():
            policies[policy]={}
            for role,q in roles.items():
                stored=r['policies'][policy][role];check(len(q)==len(np.unique(q)) and np.array_equal(q,stored['query_indices']),'true unique query support')
                check(q.min()-15>=0 and q.max()+15<counts[role],'true all-lag index support')
                z=summarize_matrix(matrices[role],q-15);metric_cells+=1
                check(z['query_count']==stored['query_count'] and z['best_lag_at_boundary']==stored['best_lag_at_boundary'],'query count/boundary')
                for field in (*METRICS,'curve'):max_scalar=max(max_scalar,close(z[field],stored[field],'saved curve/scalar'))
                if s['cohort']=='pilot10':
                    for metric in METRICS:bridge_max=max(bridge_max,close(z[metric],old[sid]['policies'][policy][role][metric],'pilot all-support bridge'))
                policies[policy][role]=z
        check(r['common_query_count']==k and r['low_common_query_warning']==(k<25),'low common support warning')
        for role in ROLES[2:]:check(r['draw_reliability'][role]==reliability[sid,role],'per-record reliability')
        rebuilt.append(dict(sample_id=sid,speaker_id=s['speaker_id'],cohort=s['cohort'],policies=policies,common_queries=k))
    check(feature_count==400 and len(matrix_checks)==200,'400 arrays/200 matrices')
    close(analysis['pilot_bridge_max_error'],bridge_max,'pilot bridge max')
    print('MATRIX200_SUPPORT_METRICS_PASS',metric_cells,max_matrix,flush=True)
    summary_count=0;conditional_count=0;max_summary=0.;summaries={}
    for cohort in ('new30','all40','pilot10'):
        rows=[r for r in rebuilt if cohort=='all40' or r['cohort']==cohort];wanted=analysis['cohorts'][cohort]
        check(wanted['measurable_speakers']==len(rows),'cohort denominator')
        good=[r for r in rows if any(reliability[r['sample_id'],role]['qualified'] for role in ROLES[2:])]
        missing=[r['sample_id'] for r in rows if r not in good]
        check(wanted['conditional_coverage']==len(good) and wanted['conditional_missing']==missing,'conditional explicit coverage')
        summaries[cohort]={}
        for policy in POLICIES:
            summaries[cohort][policy]={}
            for metric in METRICS:
                x=np.asarray([[r['policies'][policy][role][metric] for role in ROLES] for r in rows],dtype=np.float64)
                if not len(rows):continue
                t=np.add.reduce(x[:,2:],axis=1)/3.;contrasts={'T-N':t-x[:,0],'T-R':t-x[:,1],'R-N':x[:,1]-x[:,0]}
                w=wanted['summaries'][policy][metric];results={}
                for name,values in contrasts.items():
                    z=bootstrap(values);max_summary=max(max_summary,compare_summary(z,w['contrasts'][name],'pooled '+cohort));summary_count+=1;results[name]=z
                    if metric=='C':
                        loo=w['leave_one_out'][name];check(len(loo)==len(rows),'symmetric LOO coverage')
                        for index,item in enumerate(loo):
                            check(item['omitted']==rows[index]['sample_id'],'LOO order');close(item['mean'],float(np.delete(values,index).mean()),'LOO mean')
                for index in range(3):
                    for name,base in [('T-N',0),('T-R',1)]:
                        z=bootstrap(x[:,index+2]-x[:,base]);max_summary=max(max_summary,compare_summary(z,w['per_seed'][f'T{index+1}'][name],'per-seed '+cohort));summary_count+=1
                for index,role in enumerate(ROLES):close(float(x[:,index].mean()),w['means'][role],'condition group mean')
                close(float(x[:,2:].mean()),w['T_mean'],'T group mean');supplement={}
                cw=wanted['conditional_generation'][policy][metric]
                if good:
                    conditional=[]
                    for r in good:
                        selected=[r['policies'][policy][role][metric] for role in ROLES[2:] if reliability[r['sample_id'],role]['qualified']]
                        tmean=float(np.mean(selected));conditional.append([tmean-r['policies'][policy]['N'][metric],tmean-r['policies'][policy]['R'][metric]])
                    conditional=np.asarray(conditional)
                    for index,name in enumerate(('T-N','T-R')):
                        z=bootstrap(conditional[:,index]);max_summary=max(max_summary,compare_summary(z,cw['contrasts'][name],'conditional '+cohort));conditional_count+=1;supplement[name]=z
                else:check(cw['n_speakers']==0,'conditional no-person coverage')
                summaries[cohort][policy][metric]=dict(contrasts=results,conditional=supplement)
    check(summary_count==405 and conditional_count==90,'405 main/perseed +90 conditional bootstraps')
    main=summaries['new30']['common']['C']['contrasts']
    expected_decision={name:main[name]['CI97_5'][0]>0 for name in ('T-N','T-R')} if not any(r['cohort']=='new30' for r in unmeasurable.values()) else None
    check(analysis['two_main_new30_bonferroni_positive']==expected_decision,'two-main adjusted intervals')
    print('BOOTSTRAP405_PLUS90_PASS',max_summary,flush=True)
    asr=read(RUN/'asr/results.json');vad=read(RUN/'asr/vad_results.json');vp=read(RUN/'asr/vad_followup_protocol.json')
    asr_rows=asr['records'];new_expected={(sid,role) for sid,s in samples.items() if s['cohort']=='new30' for role in ROLES}
    check(asr['count']==len(asr_rows)==150 and {(r['sample_id'],r['role']) for r in asr_rows}==new_expected,'ASR all150 retained')
    bad={sid for (sid,role),rel in reliability.items() if samples[sid]['cohort']=='new30' and not rel['qualified']}
    check(vp['speakers']==sorted(bad),'VAD metadata-only trigger')
    check(vad['count']==len(vad['records']) and {(r['sample_id'],r['role']) for r in vad['records']}=={(sid,role) for sid in bad for role in ROLES},'VAD all-five diagnostic coverage')
    all_asr=asr_rows+vad['records'];strings=[]
    for r in all_asr:strings.extend([samples[r['sample_id']]['transcript'],r['raw']])
    normalized=normalized_bulk(strings);cer_checks=[]
    for index,r in enumerate(all_asr):
        sid,role=r['sample_id'],r['role'];a=ab[sid,role];check(r['audio_sha256']==a['sha256'],'ASR source file')
        ref,hyp=normalized[2*index:2*index+2];check(ref==r['reference'] and hyp==r['normalized'],'ASR normalized text')
        distance=edit_distance(ref,hyp);close(r['CER'],distance/len(ref),'CER independent2D DP')
        if index<150:
            check(r['pcm_sha256']==a['pcm_sha256'] and r['edit_distance']==distance,'ASR PCM/edit distance')
            check(''.join(segment['text'] for segment in r['segments'])==r['raw'],'ASR full segment text')
            saved=read(RUN/'asr/records'/sid/f'{role}.json');check(saved==r,'ASR per-record readback')
        else:
            close(r['speech_s'],sum(span['end']-span['start'] for span in r['spans_s']),'VAD duration')
        cer_checks.append(dict(sample_id=sid,role=role,VAD_exploratory=index>=150,CER=distance/len(ref)))
    print('ASR150_AND_VAD_CER_READBACK_PASS',len(all_asr),flush=True)
    return dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
                analysis_sha256=sha(RUN/'analysis.json'),parent_seals=seals,
                checks=dict(new_canonical150_PCM_exact=True,new_video150_bound=True,new_features300_bound=True,
                            feature_arrays400=True,self_matrices200=True,actual_all_lag_no_padding=True,metric_cells=metric_cells,
                            main_perseed_bootstraps=summary_count,conditional_bootstraps=conditional_count,all120_draws_retained=True,
                            pilot_all_support_endpoint_bridge=True,ASR150_CER_independent_DP=True,VAD_metadata_only_diagnostic=True),
                numeric_errors=dict(numpy_matrix_max=max_matrix,curves_and_scalars_max=max_scalar,bootstrap_max=max_summary,pilot_bridge_max=bridge_max),
                canonical_audio=audio_checks,matrix_checks=matrix_checks,summary=summaries,ASR_CER_checks=cer_checks,
                common_query_counts={r['sample_id']:r['common_queries'] for r in rebuilt},unmeasurable=list(unmeasurable.values()),
                boundaries=['Expansion uses existing development corpus; new30 is not guaranteed independent of training or historical exposure.',
                            'Full-ICL T versus R includes reference decoder context; not isolated LM-learning-objective causality.',
                            'Same physical support count or positions do not align phonemes between independent utterances.',
                            'All planned draws retained; EOS/duration-qualified analysis is a conditional diagnostic, not replacement main.',
                            'Whisper CER/VAD are automated proxies, not human truth, naturalness or mouth accuracy.',
                            'Inherited pilot execution overlap remains disclosed in its sealed original record. This audit cannot prove no stochastic influence.'],
                checked_files=files,errors=errors)


if __name__=='__main__':
    destination=RUN/'independent_downstream_audit.json';check(not destination.exists(),'audit output exists: never overwrite/retry silently')
    try:result=main()
    except BaseException as exc:
        result=dict(status='FAILED',errors=errors,exception=repr(exc),traceback=traceback.format_exc(),checked_files=files)
        with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        raise
    with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('INDEPENDENT_DOWNSTREAM_AUDIT_SHA',sha(destination),flush=True)
