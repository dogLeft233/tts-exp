"""Frozen exploratory reward probes for existing canonical audio; no generation."""
import os
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[_v]='1'
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import argparse, hashlib, json, pickle, sys, time, datetime
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

R=Path('/home/wjj/tts-audio/tts-exp'); OUT=R/'runs/syncnet_preference_probe_20261024_attempt01'
MODEL=Path('/home/wjj/.cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a')
PRIMARY=['N-L06','N-L17','L17-L06']; WITHIN=['L06-within','L17-within']; MODELS=['M0','M1','M2','M3']
ACOUSTIC=['duration_s','quiet_fraction','envelope_cv','spectral_flux','centroid_hz','high_band_fraction']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,v):
    p=OUT/name;assert not p.exists(),p
    p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def audit():
    manifest=read(OUT/'manifest.json'); rows=manifest['rows'];err=0;norm=[]
    for x in rows:
        for field,h in [('audio_path','audio_sha256'),('video_path','video_sha256'),('score_path','score_sha256'),('activesd_path','activesd_sha256')]:assert sha(x[field])==x[h]
        info=sf.info(x['audio_path']);assert info.samplerate==16000 and info.channels==1 and info.subtype=='PCM_16'
        curve=np.asarray(pickle.loads(Path(x['activesd_path']).read_bytes())[0]);assert curve.dtype==np.float32 and curve.shape[1]==31
        mu=curve.mean(0); c=float(np.median(mu)-mu.min());d=float(mu.min());b=float(np.median(mu))
        err=max(err,abs(c-x['score_printed']));assert abs(c-x['score_printed'])<=.000501
        x.update(score_float32=c,score_d=d,score_b=b)
        norm.append(x['features']['rms'])
    # All 20 source/arm chains cite one shared target; quantization changes realized RMS slightly.
    op=read(R/'runs/qwen17_level_comparison10_20261023_attempt02/protocol.json');cp=read(R/'runs/qwen_prediction_mechanism_20261023_attempt02/C_protocol.json')
    target=.0376838172675746;assert abs(op['target_rms']-target)<1e-12 and abs(cp['target_rms']-target)<1e-12
    assert max(abs(np.asarray(norm)-target))<1e-5
    put('input_manifest.json',dict(rows=rows,normalization_target=target,actual_rms_range=[min(norm),max(norm)],bridge_max_error=err,protocol_sha256=sha(OUT/'protocol.json'),encoder_pins={str(p):sha(p) for p in [MODEL/'config.json',MODEL/'preprocessor_config.json',MODEL/'pytorch_model.bin']},code_sha256=sha(__file__)))
    print('AUDIT',len(rows),'bridge_max',err,flush=True)

def extract():
    import torch,transformers
    from transformers import HubertModel,Wav2Vec2FeatureExtractor
    torch.set_num_threads(4);torch.manual_seed(20261024)
    rows=read(OUT/'input_manifest.json')['rows']; processor=Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL),local_files_only=True)
    model=HubertModel.from_pretrained(str(MODEL),local_files_only=True).float().eval().to('cuda')
    vectors=[];metadata=[];start=time.time()
    for i,x in enumerate(rows):
        assert sha(x['audio_path'])==x['audio_sha256']
        y,sr=sf.read(x['audio_path'],dtype='float32');assert sr==16000 and y.ndim==1
        inp=processor(y,sampling_rate=sr,return_tensors='pt',return_attention_mask=True)
        assert int(inp['attention_mask'].sum())==len(y) and inp['input_values'].shape[-1]==len(y)
        with torch.no_grad():
            h=model(inp['input_values'].cuda(),attention_mask=inp['attention_mask'].cuda()).last_hidden_state[0]
            valid=int(model._get_feat_extract_output_lengths(torch.tensor(len(y))).item());assert len(h)==valid
            h=h[:valid].float();v=torch.cat([h.mean(0),h.std(0,unbiased=False)]).cpu().numpy();assert np.isfinite(v).all()
        vectors.append(v);metadata.append(dict(id=x['id'],input_samples=len(y),valid_hidden_frames=valid,hidden_dimensions=h.shape[-1],pooled_dimensions=len(v)))
        print('ENCODER',i+1,len(rows),x['id'],valid,'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(OUT/'hubert_pooled.npz',ids=np.asarray([x['id'] for x in rows]),pooled=np.asarray(vectors,dtype=np.float32))
    put('encoder_metadata.json',dict(model=str(MODEL),layer='final',precision='FP32',normalization=processor.to_dict(),frames=metadata,torch=torch.__version__,transformers=transformers.__version__,elapsed_s=time.time()-start,weights_sha256=sha(MODEL/'pytorch_model.bin')))

def make_pairs(rows,strict=False):
    ix={x['id']:i for i,x in enumerate(rows)};by={}
    for x in rows:by.setdefault(x['sample_id'],{})[(x['arm'],x['draw'])]=x
    out=[]
    for sid,cells in by.items():
        ds=[d for d in [1,2,3] if not strict or all(cells[(a,d)]['qualified'] for a in ['L06','L17'])]
        def add(a,da,b,db,family):
            x,y=cells[a,da],cells[b,db]
            out.append(dict(source=sid,speaker=x['speaker_id'],group=x['group_id'],cohort=x['cohort'],family=family,a=ix[x['id']],b=ix[y['id']],a_id=x['id'],b_id=y['id'],a_draw=da,b_draw=db,target=x['score_float32']-y['score_float32']))
        for d in ds:
            add('N',0,'L06',d,'N-L06');add('N',0,'L17',d,'N-L17');add('L17',d,'L06',d,'L17-L06')
        for arm in ['L06','L17']:
            for i,d in enumerate(ds):
                for e in ds[i+1:]:add(arm,d,arm,e,arm+'-within')
    return out

def matrices(rows,hidden,train_ids):
    """All fitted transformations use exactly the unique training clip IDs."""
    tr=np.asarray(sorted(train_ids));arms=np.asarray([[float(x['arm']==a) for a in ['N','L06','L17']] for x in rows])
    ac=np.asarray([[x['features'][f] for f in ACOUSTIC] for x in rows]); support=np.log1p([x['score_windows'] for x in rows])[:,None]
    hs=StandardScaler().fit(hidden[tr]);hp=PCA(n_components=8,svd_solver='full').fit(hs.transform(hidden[tr]));z=hp.transform(hs.transform(hidden))
    raw={'M0':arms,'M1':np.c_[arms,ac,support],'M2':np.c_[arms,z,np.log(ac[:,0]),support],'M3':np.c_[arms,ac,support,z]}
    transforms={}; arrays={}
    for m,x in raw.items():
        sc=StandardScaler().fit(x[tr]);arrays[m]=sc.transform(x);transforms[m]=sc
    return arrays,dict(train_clip_indices=tr.tolist(),train_clip_ids=[rows[i]['id'] for i in tr],hidden_scaler=hs,hidden_pca=hp,model_scalers=transforms)

def pair_weights(pairs):
    counts={}
    for p in pairs:counts[p['source'],p['family']]=counts.get((p['source'],p['family']),0)+1
    return np.asarray([1/counts[p['source'],p['family']] for p in pairs])
def fit_model(pairs,z,alpha):
    x=np.asarray([z[p['a']]-z[p['b']] for p in pairs]);y=np.asarray([p['target'] for p in pairs]);return Ridge(alpha=alpha,fit_intercept=False).fit(x,y,sample_weight=pair_weights(pairs))
def predict(pairs,z,model):return model.predict(np.asarray([z[p['a']]-z[p['b']] for p in pairs]))
def source_metrics(pairs,pred,families=PRIMARY,threshold=0):
    parts={}
    for p,yhat in zip(pairs,pred):
        if p['family'] not in families or abs(p['target'])<threshold:continue
        key=(p['source'],p['family']);target=p['target']; correct=None if target==0 else (.5 if abs(yhat)<=1e-9 else float((yhat>0)==(target>0)))
        parts.setdefault(key,[]).append((abs(yhat-target),correct))
    per={}
    for (sid,f),v in parts.items():
        per.setdefault(sid,{})[f]=dict(mae=float(np.mean([x[0] for x in v])),accuracy=float(np.mean([x[1] for x in v if x[1] is not None])) if any(x[1] is not None for x in v) else None,pairs=len(v))
    sources={sid:dict(mae=float(np.mean([v['mae'] for v in f.values()])),accuracy=float(np.mean([v['accuracy'] for v in f.values() if v['accuracy'] is not None])) if any(v['accuracy'] is not None for v in f.values()) else None,families=f) for sid,f in per.items()}
    return dict(mae=float(np.mean([v['mae'] for v in sources.values()])) if sources else None,accuracy=float(np.mean([v['accuracy'] for v in sources.values() if v['accuracy'] is not None])) if any(v['accuracy'] is not None for v in sources.values()) else None,sources=sources,pairs=sum(len(v) for v in parts.values()))

def fit_all(rows,h,pairs,name):
    predictions={m:np.zeros(len(pairs)) for m in MODELS};folds=[];groups=sorted({p['group'] for p in pairs})
    for gi,group in enumerate(groups):
        train=[p for p in pairs if p['group']!=group];testix=[i for i,p in enumerate(pairs) if p['group']==group];test=[pairs[i] for i in testix]
        assert not ({p['source'] for p in train}&{p['source'] for p in test})
        scores={m:{alpha:[] for alpha in [1.,10.,100.]} for m in MODELS}
        inner=GroupKFold(5)
        for it,iv in inner.split(train,groups=[p['group'] for p in train]):
            t=[train[i] for i in it];v=[train[i] for i in iv];clips={p[k] for p in t for k in ['a','b']};z,_=matrices(rows,h,clips)
            for m in MODELS:
                for alpha in scores[m]:
                    fit=fit_model(t,z[m],alpha);pr=predict(v,z[m],fit);scores[m][alpha].extend(x['mae'] for x in source_metrics(v,pr)['sources'].values())
        alphas={m:min(scores[m],key=lambda a:(round(float(np.mean(scores[m][a])),12),-a)) for m in MODELS}
        clips={p[k] for p in train for k in ['a','b']};z,processors=matrices(rows,h,clips);models={};fold=dict(group=group,train_sources=sorted({p['source'] for p in train}),test_sources=sorted({p['source'] for p in test}),alphas=alphas,inner_mae={m:{str(a):float(np.mean(v)) for a,v in scores[m].items()} for m in MODELS})
        for m in MODELS:
            model=fit_model(train,z[m],alphas[m]);pr=predict(test,z[m],model);predictions[m][testix]=pr;models[m]=model
            swapped=[dict(p,a=p['b'],b=p['a']) for p in test];assert np.max(abs(predict(swapped,z[m],model)+pr))<1e-10
        # Structural leakage and heldout invariance tests: fit transformers after a heldout-only alteration.
        if gi==0:
            hh=h.copy();held={p[k] for p in test for k in ['a','b']};hh[list(held)]+=123.
            zz,pp=matrices(rows,hh,clips)
            assert np.array_equal(processors['hidden_scaler'].mean_,pp['hidden_scaler'].mean_)
            assert np.array_equal(processors['hidden_pca'].components_,pp['hidden_pca'].components_)
            for m in MODELS:assert np.max(abs(z[m][list(clips)]-zz[m][list(clips)]))<1e-10
        path=OUT/'folds'/f'{name}_{group}.pkl';path.parent.mkdir(exist_ok=True);assert not path.exists()
        path.write_bytes(pickle.dumps(dict(processors=processors,models=models,fold=fold)))
        numeric=dict(train_indices=np.asarray(processors['train_clip_indices']),hidden_scaler_mean=processors['hidden_scaler'].mean_,hidden_scaler_scale=processors['hidden_scaler'].scale_,pca_mean=processors['hidden_pca'].mean_,pca_components=processors['hidden_pca'].components_)
        for m in MODELS:
            numeric.update({m+'_scaler_mean':processors['model_scalers'][m].mean_,m+'_scaler_scale':processors['model_scalers'][m].scale_,m+'_coef':models[m].coef_,m+'_alpha':np.asarray(alphas[m])})
        np.savez_compressed(path.with_suffix('.npz'),**numeric)
        folds.append(fold);print('FIT',name,gi+1,len(groups),group,alphas,flush=True)
    oof=[dict(p,predictions={m:float(predictions[m][i]) for m in MODELS}) for i,p in enumerate(pairs)]
    summaries={}
    for m in MODELS:
        pr=predictions[m];summaries[m]={'primary':source_metrics(pairs,pr),'threshold_.1':source_metrics(pairs,pr,threshold=.1),'threshold_.3':source_metrics(pairs,pr,threshold=.3),
          'families':{f:source_metrics(pairs,pr,[f]) for f in PRIMARY+WITHIN},'cohorts':{c:source_metrics([p for p in pairs if p['cohort']==c],pr[[i for i,p in enumerate(pairs) if p['cohort']==c]]) for c in ['old','new']}}
    dif=np.asarray([summaries['M1']['primary']['sources'][s]['mae']-summaries['M3']['primary']['sources'][s]['mae'] for s in sorted(summaries['M1']['primary']['sources'])]);rng=np.random.default_rng(20261024);boot=dif[rng.integers(len(dif),size=(10000,len(dif)))].mean(1)
    result=dict(summaries=summaries,mae_improvement=float(dif.mean()),descriptive_CI95=np.quantile(boot,[.025,.975]).tolist(),folds=folds,pair_count=len(pairs))
    put(name+'_oof.json',oof);put(name+'_results.json',result);return result

def analyze():
    rows=read(OUT/'input_manifest.json')['rows'];h=np.load(OUT/'hubert_pooled.npz');assert list(h['ids'])==[x['id'] for x in rows]
    with threadpool_limits(limits=1):
        full=fit_all(rows,h['pooled'],make_pairs(rows),'full');strict=fit_all(rows,h['pooled'],make_pairs(rows,True),'qualified')
    s=full['summaries'];acc=s['M3']['threshold_.1']['accuracy']-s['M1']['threshold_.1']['accuracy'];cohort={c:s['M1']['cohorts'][c]['mae']-s['M3']['cohorts'][c]['mae'] for c in ['old','new']}
    gate={'mae_improvement_at_least_.05':full['mae_improvement']>=.05,'accuracy_improvement_at_least_.05':acc>=.05,'cohorts_nonnegative':all(x>=0 for x in cohort.values()),'qualified_direction_positive':strict['mae_improvement']>0}
    put('gate.json',dict(pass_gate=all(gate.values()),checks=gate,accuracy_improvement=acc,cohort_mae_improvements=cohort,qualification_improvement=strict['mae_improvement']))
    report(full,strict)

def report(full,strict):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gate=read(OUT/'gate.json');im=read(OUT/'input_manifest.json');lines=['# 归一化三臂音频的原生 SyncNet 偏好探针','',f"推进门：{'通过' if gate['pass_gate'] else '未通过'}。全量M3−M1的MAE改善（正值有利）={full['mae_improvement']:+.3f}，描述性95%区间 {full['descriptive_CI95']}。",'', '| 模型 | Primary MAE | 符号准确率 | gap绝对值≥0.1准确率 |','|---|---:|---:|---:|']
    for m in MODELS:
        s=full['summaries'][m];lines.append(f"| {m} | {s['primary']['mae']:.3f} | {s['primary']['accuracy']:.3f} | {s['threshold_.1']['accuracy']:.3f} |")
    lines+=['',f"old/new 的 M1−M3 MAE改善：{gate['cohort_mae_improvements']}；共同资格重拟合改善 {strict['mae_improvement']:+.3f}。",'', '## Material Passport','',f"20来源、20speaker/text连通组；N20、L06/L17各60，共140已有canonical音频。全量300pair，主比较180pair；共同资格{strict['pair_count']}pair。每source/family等权，N每source仅一条，不作为三个独立样本。",f"同一目标RMS={im['normalization_target']}，实测范围{im['actual_rms_range']}；140 mono16k PCM16，媒体及31lag曲线SHA重验；首track float32精确C与打印值桥接最大误差 {im['bridge_max_error']:.9f}。",'冻结HuBERT base-ls960最终层、eval/no_grad FP32，整段音频无截尾；处理器自带归一化如encoder_metadata.json。时间mean/std只池化有效帧。PCA8为训练折内表示压缩，并非音素概念或可解释声学特征。', '每outer留一speaker/text连通组，inner GroupKFold5；所有draw同组，unique训练clip fit标准化/PCA，alpha仅1/10/100。保存40个fold模型/处理器以及全pair OOF。M0即仅arm身份、由训练折估计的比较器。', '零生成器训练；本实验训练了小型线性reward probe。标签只是原有官方Sync-C分差，不是人类偏好。native各自(audio, generated video)评分，audio-only预测针对固定现有TFG链的原生排序，不能隔离评价器纯偏好与生成端响应，也不能判断嘴型质量或因果机制。', '全部20来源已有结果被查看，本研究是探索性组外预测。共享训练折的OOF bootstrap为描述性区间，不是独立确认推断。未过门不能声称不存在可解释机制；按协议停止复杂模型/SAE，不自动扩搜索。', '', '完整family/cohort/source和支持数量见full_results.json / qualified_results.json；逐pair预测见*_oof.json，所有处理器见folds/。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    ids=sorted(full['summaries']['M1']['primary']['sources']);vals=[full['summaries']['M1']['primary']['sources'][s]['mae']-full['summaries']['M3']['primary']['sources'][s]['mae'] for s in ids]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(ids)),vals);ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(ids)),[s[6:11] for s in ids],rotation=70);ax.set_ylabel('OOF MAE improvement: M1 - M3');fig.tight_layout();fig.savefig(OUT/'source_mae_improvement.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','extract','analyze']);args=p.parse_args();start=time.time()
    put(args.stage+'_started.json',dict(utc=utc(),pid=os.getpid(),command=sys.argv,code_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json')))
    try:
        globals()[args.stage]();put(args.stage+'_finished.json',dict(utc=utc(),elapsed_s=time.time()-start,exitcode=0))
    except Exception as e:
        put(args.stage+'_failed.json',dict(utc=utc(),elapsed_s=time.time()-start,error=repr(e)));raise
if __name__=='__main__':main()
