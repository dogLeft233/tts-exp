"""Manifest-driven cache consumer; prepare/CPU-bridge never allocates a GPU.

The inherited 20-source cache is referenced by immutable paths and hashes.
Only a separate parent release can execute new frozen SyncNet or FA work.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,os,pickle,shutil,sys,time,wave,subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
    p=Path(p);assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def normalized(s):return ''.join(s.split())
def validate_generation_contract(g,s,p):
    """Pure identity/sampling validation; no synthesis or preference selection."""
    assert g['sample_id']==s['sample_id'] and g['speaker_id']==s['speaker_id'] and g['model']=='L17'
    d=int(g['draw_index']);assert 1<=d<=3 and g['role']==f'L17_norm{d}'
    assert p['seed_bases']==[2026102201,2026102202,2026102203]
    assert p['generation_kwargs']==dict(do_sample=True,temperature=.9,top_k=50,top_p=1.,max_new_tokens=512)
    assert p['non_streaming_mode'] is False and p['no_training'] is True
    assert g['seed']==p['seed_bases'][d-1]+100*int(s['speaker_id'][1:])
    text=normalized(s['transcript']);assert g['target_text']==g['reference_text']==text
    assert g['reference_audio_sha256']==s['input24_sha256']
    core=g['core_explicit_kwargs'];effective=g['talker_effective_kwargs']
    for k,v in p['generation_kwargs'].items():assert core[k]==v and effective[k]==v,k
    assert core['non_streaming_mode'] is False and core['languages']==['Chinese']
    for k,v in dict(subtalker_dosample=True,subtalker_top_k=50,subtalker_top_p=1.,subtalker_temperature=.9,repetition_penalty=1.05).items():assert core[k]==v,k
    return dict(expected_seed=g['seed'],draw=d,text=text,reference_sha256=s['input24_sha256'])

def producer_qa(args):
    """CPU-only acceptance snapshot; missing cells remain pending intentions."""
    import numpy as np
    import soundfile as sf
    root=Path(args.producer_root);out=Path(args.output_root);out.mkdir(parents=True,exist_ok=False);p=read(root/'protocol.json')
    assert p['scope']=='missing60_L17_only' and p['no_training'] is True and p['target_rms']==.0376838172675746
    put(out/'qa_protocol.json',dict(schema='producer_cpu_independent_qa_v1',frozen_utc=utc(),producer_root=str(root),producer_protocol_sha256=sha(root/'protocol.json'),consumer_code_sha256=sha(__file__),gpu=False,new_FA=False,new_generation=False,
        thresholds=dict(canonical_rms_abs=1e-5,metadata_rms_abs=1e-12,gain_target_abs=1e-12,score_stdout_abs=.000501),EOS='Retain true and false; compare claimed EOS to saved token array, never filter on score',partial='Only cells with official_completion receipt become QA candidates; no complete consumer manifest required for interim snapshot'))
    done=[];pending=[];failed=[];cache={}
    def pin(path,expected=None):
        path=str(path)
        if path not in cache:cache[path]=sha(path)
        if expected is not None:assert cache[path]==expected,path
        return cache[path]
    for s in p['samples']:
        sid=s['sample_id'];cell=root/'cells'/sid;completion=cell/'receipts/official_completion.json'
        if not completion.exists():pending.extend(f'{sid}_L17_{d}' for d in [1,2,3]);continue
        cp=read(cell/'protocol.json')
        try:
            assert cp['samples']==[s] and cp['target_rms']==p['target_rms']
            pin(s['source_path'],s['source_sha256']);pin(s['transcript_path'],s['transcript_sha256']);pin(s['input24_path'],s['input24_sha256'])
            assert normalized(Path(s['transcript_path']).read_text())==normalized(s['transcript'])
            synth_task=next(x for x in p['tasks'] if x['sample_id']==sid and x['stage']=='synthesize');assert synth_task['protocol_sha256']==sha(cell/'protocol.json')
            am=read(cell/'audio_manifest.json');vm=read(cell/'video_manifest.json');rm=read(cell/'raw_manifest.json');assert am['protocol_sha256']==vm['protocol_sha256']==rm['protocol_sha256']==sha(cell/'protocol.json')
            assert am['count']==3 and vm['count']==3 and rm['count']==3
            aud={x['role']:x for x in am['records']};vid={x['role']:x for x in vm['records']};raw={x['role']:x for x in rm['records']}
            oi=read(cell/'official/input_manifest.json');omap={x['role']:x for x in oi['records']};official_protocol=read(cell/'official/protocol.json');assert oi['protocol_sha256']==sha(cell/'official/protocol.json')
        except Exception as e:
            failed.extend({'id':f'{sid}_L17_{d}','stage':'cell_bindings','error':repr(e)} for d in [1,2,3]);continue
        for d in [1,2,3]:
            uid=f'{sid}_L17_{d}';role=f'L17_norm{d}'
            try:
                gp=cell/'generation'/f'draw{d}.json';g=read(gp);assert g==raw[role]
                gen=validate_generation_contract(g,s,cp);a,v=aud[role],vid[role]
                for field in ['path','sha256']:assert a['source_'+field]==g[field]
                pin(g['path'],g['sha256']);pin(a['path'],a['sha256']);pin(v['path'],v['sha256'])
                rx,rsr=sf.read(g['path'],dtype='float32');assert rsr==g['sample_rate_hz']==24000 and rx.ndim==1 and len(rx)==g['samples'] and np.isfinite(rx).all()
                assert hashlib.sha256(rx.astype('<f4').tobytes()).hexdigest()==g['float_pcm_sha256']
                pcm,sr=sf.read(a['path'],dtype='int16');info=sf.info(a['path']);assert sr==a['sr']==16000 and info.channels==1 and info.subtype=='PCM_16' and len(pcm)==a['samples']
                rms=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2)));peak=float(np.max(abs(pcm.astype(float)/32768)))
                assert abs(rms-p['target_rms'])<=1e-5 and abs(rms-a['rms_after'])<=1e-12 and abs(a['rms_before']*a['gain']-p['target_rms'])<=1e-12
                assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==a['pcm_sha256'] and peak<1 and not np.any(pcm==-32768) and not np.any(pcm==32767)
                assert abs(len(pcm)/16000-len(rx)/24000)<=1/16000
                arrays={}
                for key in ['reference_codes','new_codes','talker_sequences']:
                    desc=g[key];pin(desc['path'],desc['sha256']);arr=np.load(desc['path'],allow_pickle=False);assert list(arr.shape)==desc['shape'] and str(arr.dtype)==desc['dtype'];arrays[key]=arr
                actual_eos=bool(np.any(arrays['talker_sequences']==g['talker_effective_kwargs']['eos_token_id']));assert actual_eos==g['eos_in_talker_sequences']
                assert v['audio_sha256']==a['sha256'] and v['sample_id']==a['sample_id']==sid and v['speaker_id']==a['speaker_id']==s['speaker_id']
                pin(cp['downstream']['wav2lip_checkpoint'],v['weights_sha256']) if 'wav2lip_checkpoint' in cp['downstream'] else None
                face_command=v['receipt']['command'];face=face_command[face_command.index('--face')+1];inference=face_command[1];weights=face_command[face_command.index('--checkpoint_path')+1]
                pin(face,v['face_sha256']);pin(inference,v['inference_sha256']);pin(weights,v['weights_sha256']);assert v['receipt']['returncode']==0 and not v['receipt']['timed_out']
                assert face_command[face_command.index('--audio')+1]==a['path']
                sp=cell/'official/scores'/f'{sid}_{role}.json';score=read(sp);unit=dict(id=uid,sample_id=sid,speaker_id=s['speaker_id'],arm='L17',draw=d,text=gen['text'],score_path=str(sp),score_sha256=sha(sp));bound=parse_score(unit)
                assert bound['audio_path']==a['path'] and bound['audio_sha256']==a['sha256'] and bound['video_path']==v['path'] and bound['video_sha256']==v['sha256']
                assert score['role']==role and omap[role]['canonical_audio_sha256']==a['sha256'];pin(score['input_mkv'],score['input_mkv_sha256'])
                dist=np.asarray(pickle.loads(Path(bound['activesd_path']).read_bytes())[0]);assert dist.dtype==np.float32 and dist.ndim==2 and dist.shape[1]==31
                mu=dist.mean(0);C=float(np.median(mu)-mu.min());D=float(mu.min());B=float(np.median(mu));assert abs(C-score['sync_c'])<=.000501 and abs(D-score['sync_d'])<=.000501 and int(15-mu.argmin())==score['av_offset']
                done.append(dict(unit,status='CPU_QA_PASS_MEDIA_ONLY',canonical_audio_path=a['path'],canonical_audio_sha256=a['sha256'],video_path=v['path'],video_sha256=v['sha256'],seed=g['seed'],target_text=g['target_text'],reference_audio_sha256=g['reference_audio_sha256'],eos_in_talker_sequences=actual_eos,
                  norm_rms=rms,norm_peak=peak,raw_samples=len(rx),canonical_samples=len(pcm),raw_float_sha256=g['float_pcm_sha256'],full_official_rows=len(dist),score_float32=C,score_D_float32=D,score_B_float32=B,score_stdout_error=abs(C-score['sync_c']),generation_path=str(gp),generation_sha256=sha(gp),cell_protocol_sha256=sha(cell/'protocol.json'),official_input_manifest_path=str(cell/'official/input_manifest.json'),official_input_manifest_sha256=sha(cell/'official/input_manifest.json')))
            except Exception as e:failed.append(dict(id=uid,stage='draw_QA',error=repr(e)))
    final=root/'consumer_manifest.json';final_status='NOT_YET_PRESENT'
    if final.exists():
        f=read(final);assert f['status']=='COMPLETE_MEDIA_ONLY' and f['protocol_sha256']==sha(root/'protocol.json')
        indexed={x['id']:x for x in f['rows']};assert len(indexed)==60
        for x in done:
            y=indexed[x['id']]
            for k in ['sample_id','speaker_id','seed','target_text','reference_audio_sha256','eos_in_talker_sequences','canonical_audio_path','canonical_audio_sha256','video_path','video_sha256','score_path','score_sha256']:assert y[k]==x[k],k
            assert y['status']=='COMPLETE' and y['arm']=='L17' and y['draw']==x['draw'] and y['split']=='fit'
        final_status='PRESENT_IDENTITY_QA_COMPLETE' if len(done)==60 and not failed else 'PRESENT_BUT_NOT_ACCEPTED'
    put(out/'producer_cpu_qa.json',dict(schema='producer_cpu_independent_acceptance_v1',utc=utc(),status='PASS_PARTIAL_COMPLETED_ONLY' if pending and not failed else ('PASS_ALL60_MEDIA_ONLY' if len(done)==60 and not failed else 'QA_FAILURE_RETAINED'),producer_final_manifest_status=final_status,producer_manifest_sha256=sha(final) if final.exists() else None,
        intention=60,passed_count=len(done),pending_count=len(pending),failed_count=len(failed),passed=done,pending_ids=pending,failed=failed,source_pins=cache,QA_protocol_sha256=sha(out/'qa_protocol.json'),consumer_source_sha256=sha(__file__),no_GPU_forward=True,no_new_FA=True,no_training=True))
    print(json.dumps(dict(passed=len(done),pending=len(pending),failed=failed,producer_final_manifest=final_status),ensure_ascii=False))
def parse_score(unit,check_hashes=True):
    p=Path(unit['score_path']);s=read(p)
    if unit.get('score_sha256'):assert sha(p)==unit['score_sha256']
    assert s.get('track_count')==1,'Current contract accepts unique first official track; multi-track needs reviewed adapter'
    t=s['tracks'][0]['metadata'];assert len(s['tracks'])==1 and s['status']=='COMPLETE'
    cmd=s['pipeline_receipt']['command'];data=Path(cmd[cmd.index('--data_dir')+1]);pre=data/'pyavi'/s['reference']/'audio.wav'
    b=dict(id=unit['id'],sample_id=unit['sample_id'],speaker_id=unit['speaker_id'],arm=unit['arm'],draw=unit['draw'],text=unit['text'],
        score_path=str(p),score_sha256=sha(p),audio_path=s['canonical_audio_path'],audio_sha256=s['canonical_audio_sha256'],
        video_path=s['video_path'],video_sha256=s['source_video_sha256'],activesd_path=s['activesd_path'],activesd_sha256=s['activesd_sha256'],
        crop_path=t['crop_file'],crop_sha256=t['crop_sha256'],track=t,pretrack_audio_path=str(pre),pretrack_audio_sha256=sha(pre),source_frames=s['frame_count'],intermediate_probe=s['official_intermediate_AVI_probe'])
    assert s['sample_id']==unit['sample_id'];assert s['speaker_id']==unit['speaker_id']
    assert s['mux_PCM_exact'] and s['decoded_frames_exact'] and s['frame_PTS_exact']
    for receipt in ['pipeline_receipt','syncnet_receipt']:
        assert s[receipt]['returncode']==0 and not s[receipt]['timed_out']
    assert int(t['track_start_frame'])==0,'Inherited engine prefix binding assumes track start0; stop rather than silently mis-map'
    if check_hashes:
        for k,h in [('audio_path','audio_sha256'),('video_path','video_sha256'),('activesd_path','activesd_sha256'),('crop_path','crop_sha256')]:assert sha(b[k])==b[h],k
        with wave.open(b['audio_path']) as a:
            assert a.getframerate()==16000 and a.getnchannels()==1 and a.getsampwidth()==2
    return b
def identity(b):return tuple(b[k] for k in ['audio_sha256','score_sha256','crop_sha256','activesd_sha256'])
def cache_index(cache_root):
    cache_root=Path(cache_root);seal=read(cache_root/'final_manifest.json');sealed={x['relative_path']:x['sha256'] for x in seal['files']}
    rows=read(cache_root/'input_manifest.json')['rows'];idx={}
    for b in rows:
        item=cache_root/'items'/b['id'];rec=read(item/'replay.json');assert rec['pass_replay'] and rec['distance_replay_exact'] and rec['mfcc_exact']
        files={}
        for n in ['sequence.npz','scoring_audio.wav','replay.json','frame_manifest.json']:
            p=item/n;rel=str(p.relative_to(cache_root));assert sha(p)==sealed[rel];files[n]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        idx[identity(b)]=dict(id=b['id'],files=files,windows=rec['windows'],cache_root=str(cache_root),seal_sha256=sha(cache_root/'final_manifest.json'))
    engine=cache_root/'executed_source.py';assert sha(engine)==sealed['executed_source.py']
    env=read(cache_root/'environment.json')
    return idx,dict(path=str(engine),sha256=sha(engine),official_source_pins=read(cache_root/'protocol.json')['source_pins'],tolerances=read(cache_root/'protocol.json')['tolerances'],required_environment={k:env[k] for k in ['executable','python','torch','numpy','scipy','opencv']})
def assemble(args):
    """One dataset input snapshot: old ready materials and explicit missing cells."""
    nrt=read(args.legacy_nrt_manifest)['records'];samples=read(args.sample_protocol)['samples'];old=read(Path(args.reuse_root)/'input_manifest.json')['rows'];existing={(x['sample_id'],x['arm'],x['draw']):x for x in old};units=[]
    nrtidx={(x['sample_id'],x['role']):x for x in nrt};producer=Path(args.producer_root)
    producer_rows=read(args.producer_manifest)['rows'] if args.producer_manifest else []
    pidx={(x['sample_id'],x['arm'],int(x['draw'])):x for x in producer_rows};assert len(pidx)==len(producer_rows)
    qa_rows=read(args.producer_qa)['passed'] if args.producer_qa else []
    qa_idx={(x['sample_id'],x['arm'],int(x['draw'])):x for x in qa_rows}
    for sample in samples:
        sid=sample['sample_id']
        for arm,draw in [('N',0)]+[(a,d) for a in ['L06','L17'] for d in [1,2,3]]:
            eligible=True
            x=existing.get((sid,arm,draw))
            if x is not None:score=Path(x['score_path']);origin='existing_verified140'
            elif arm in ['N','L06']:
                role='N' if arm=='N' else f'T{draw}';assert (sid,role) in nrtidx;score=Path(args.legacy_nrt_manifest).parent/'scores'/f'{sid}_{role}.json';origin='existing_nrt40'
            else:
                score_base=producer/'cells'/sid if args.producer_layout=='cells' else producer
                score=score_base/'official/scores'/f'{sid}_L17_norm{draw}.json';origin='new_L17_missing60_producer'
                if args.producer_manifest:
                    punit=pidx.get((sid,arm,draw));assert punit is not None,'Producer final manifest omits a target cell'
                    assert punit['status']=='COMPLETE';score=Path(punit['score_path']);assert score.exists() and sha(score)==punit['score_sha256']
                if args.producer_qa:
                    qa=qa_idx.get((sid,arm,draw));eligible=qa is not None
                    if qa is not None:assert sha(score)==qa['score_sha256']
            ready=eligible and score.exists();units.append(dict(id=f'{sid}_{arm}_{draw}',sample_id=sid,speaker_id=sample['speaker_id'],arm=arm,draw=draw,text=normalized(sample['transcript']),score_path=str(score),score_sha256=sha(score) if ready else None,status='ready' if ready else 'pending',origin=origin))
    assert len(units)==280 and len({x['id'] for x in units})==280 and len({x['sample_id'] for x in units})==40
    out=Path(args.output_root);out.mkdir(exist_ok=False,parents=True)
    source_files=[args.legacy_nrt_manifest,args.sample_protocol,str(Path(args.reuse_root)/'final_manifest.json')]+([args.producer_manifest] if args.producer_manifest else [])+([args.producer_qa] if args.producer_qa else [])
    put(out/'dataset_input_manifest.json',dict(schema='syncnet_audio_dataset_input_v1',created_utc=utc(),project_root=str(Path(args.project_root)),units=units,reuse_root=str(Path(args.reuse_root)),producer_root=str(producer),producer_layout=args.producer_layout,producer_manifest=args.producer_manifest,source_pins={str(Path(p)):sha(p) for p in source_files}))
    print(json.dumps(dict(manifest=str(out/'dataset_input_manifest.json'),ready=sum(x['status']=='ready' for x in units),pending=sum(x['status']=='pending' for x in units))))
def prepare(args):
    man=read(args.manifest);out=Path(args.output_root);out.mkdir(exist_ok=True,parents=True)
    assert len({x['id'] for x in man['units']})==len(man['units'])
    idx,engine=cache_index(man['reuse_root']);rows=[]
    fa_path=Path(man['reuse_root'])/'alignment.json'
    fa_idx={x['audio_sha256']:dict(path=str(fa_path),file_sha256=sha(fa_path),id=x['id'],audio_sha256=x['audio_sha256'],truth='UNVERIFIED') for x in read(fa_path)['audio']}
    additional_fa_pins={}
    for ip in getattr(args,'fa_index',[]) or []:
        additional_fa_pins[str(Path(ip))]=sha(ip)
        for f in read(ip)['rows']:
            assert sha(f['fa_path'])==f['fa_file_sha256']
            candidate=next(x for x in read(f['fa_path'])['rows'] if x['id']==f['id']);assert candidate['audio_sha256']==f['audio_sha256']
            value=dict(path=f['fa_path'],file_sha256=f['fa_file_sha256'],id=f['id'],audio_sha256=f['audio_sha256'],truth='UNVERIFIED')
            if f['audio_sha256'] in fa_idx:assert fa_idx[f['audio_sha256']]==value,'Ambiguous duplicated FA binding'
            fa_idx[f['audio_sha256']]=value
    for unit in man['units']:
        if unit['status']=='pending':
            rows.append(dict(unit,status='PENDING_MEDIA_OR_OFFICIAL_SCORE',feature_status='pending',fa_status='pending',materialized_cache=False));continue
        b=parse_score(unit);old=idx.get(identity(b));fa=fa_idx.get(b['audio_sha256'])
        rows.append(dict(b,status='READY_MEDIA',feature_status='REUSED_VERIFIED_CACHE' if old else 'PENDING_NEW_EXTRACTION_RELEASE',fa_status='REUSED_VERIFIED_STRUCTURE_ONLY' if fa else 'PENDING_NEW_FA_RELEASE',reuse=old,fa_reuse=fa,materialized_cache=False))
    counts=collections.Counter(x['feature_status'] for x in rows);fcounts=collections.Counter(x['fa_status'] for x in rows)
    fa_pins=read(Path(man['reuse_root'])/'fa_started.json')['model_pins']
    fa_engine=dict(model_root=os.path.commonpath(list(fa_pins)),model_pins=fa_pins,source_pin_path=str(Path(man['reuse_root'])/'fa_started.json'),source_pin_sha256=sha(Path(man['reuse_root'])/'fa_started.json'))
    put(out/'prepared_manifest.json',dict(schema='syncnet_audio_dataset_prepared_v1',created_utc=utc(),manifest_path=str(Path(args.manifest)),manifest_sha256=sha(args.manifest),project_root=man['project_root'],rows=rows,engine=engine,fa_engine=fa_engine,additional_fa_index_pins=additional_fa_pins,counts=dict(counts),fa_counts=dict(fcounts),source_code_sha256=sha(__file__),status='PREPARED_ONLY_NO_NEW_FORWARD_OR_FA',no_training=True,gpu_used=False))
    put(out/'prepare_receipt.json',dict(command=sys.argv,pid=os.getpid(),utc=utc(),exitcode=0,counts=dict(counts),fa_counts=dict(fcounts),pending_not_complete=True,cache_copied=False,gpu_used=False,free_bytes=shutil.disk_usage(out).free))
    print(json.dumps(dict(counts=dict(counts),fa_counts=dict(fcounts)),ensure_ascii=False))
def cpu_bridge(args):
    # Actual CPU transformations only: no net construction and no cuda calls.
    import numpy as np
    from scipy.io import wavfile
    import python_speech_features as psf
    import torch
    man=read(args.manifest);engine=man['engine'];weights=[p for p in engine['official_source_pins'] if p.endswith('syncnet_v2.model')];assert len(weights)==1;assert sha(weights[0])==engine['official_source_pins'][weights[0]]
    state=torch.load(weights[0],map_location='cpu',weights_only=True);W=state['netfcaud.3.weight'].numpy();bias=state['netfcaud.3.bias'].numpy();wanted=args.ids.split(',');result=[]
    for uid in wanted:
        x=next(x for x in man['rows'] if x['id']==uid);assert x['feature_status']=='REUSED_VERIFIED_CACHE';reuse=x['reuse'];z=np.load(reuse['files']['sequence.npz']['path']);sr,y=wavfile.read(reuse['files']['scoring_audio.wav']['path']);assert sr==16000 and y.dtype==np.int16
        mfcc=psf.mfcc(y,sr).T.astype(np.float32);assert np.array_equal(mfcc,z['mfcc']);err=float(np.max(abs(z['audio512']@W.T+bias-z['audio1024'])))
        assert err<=1e-5;original=np.asarray(pickle.loads(Path(x['activesd_path']).read_bytes())[0]);assert np.array_equal(original,z['distance'])
        result.append(dict(id=uid,mfcc_exact=True,original_curve_exact=True,audio512_projection_max_abs=err,windows=len(original),gpu_forward=False))
    put(Path(args.output_root)/'cpu_bridge_receipt.json',dict(status='PASS_CPU_CONTRACT_ONLY_NOT_NEW_GPU_BRIDGE',rows=result,weights_sha256=sha(weights[0]),gpu_used=False,prepared_manifest_sha256=sha(args.manifest),command=sys.argv,utc=utc()))
    print(json.dumps(result))
def validate_release(args,key,ids):
    release=read(args.release);assert release.get(key) is True
    assert release['prepared_manifest_sha256']==sha(args.manifest)
    assert release['consumer_source_sha256']==sha(__file__)
    assert set(release['allowed_unit_ids'])==set(ids),'Exact IDs must be released; no implicit expansion'
    return release

def process_alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True
def conditional_readiness(contract):
    """Cheap CPU-only observations; no frozen-model import or construction."""
    root=Path(contract['producer_root']);reasons=[];observed={}
    for name in ['media_completion.json','consumer_manifest.json','media_cpu_acceptance.json','producer_gpu_release.json','executor_attempts/001/exit.json']:
        p=root/name
        if not p.exists():reasons.append('missing:'+name)
        else:observed[name]=read(p)
    if reasons:return dict(ready=False,reasons=reasons,observed_files=list(observed))
    completion=observed['media_completion.json'];exitrec=observed['executor_attempts/001/exit.json']
    if completion.get('status')!='MEDIA_COMPLETE_FEATURE_HOLD' or completion.get('completed_tasks')!=80 or completion.get('expected_new_media')!=60 or completion.get('no_training') is not True:reasons.append('media_completion_contract')
    if exitrec.get('returncode')!=0 or exitrec.get('status')!='MEDIA_COMPLETE_FEATURE_HOLD':reasons.append('producer_executor_exit')
    if exitrec.get('pid') and process_alive(exitrec['pid']):reasons.append('producer_executor_still_alive')
    if (root.parent/'.audio_data_supplement_gpu.lock').exists():reasons.append('producer_gpu_lock_present')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);compute=[]
    for line in query.stdout.strip().splitlines():
        fields=[s.strip() for s in line.split(',')];pid=int(fields[0]);name=fields[1];memory=float(fields[2])
        try:actual=os.readlink('/proc/'+str(pid)+'/exe')
        except OSError:actual=None
        allowed=name=='/usr/libexec/gnome-remote-desktop-daemon' and actual==name and memory<=1024
        if not allowed:reasons.append('non_allowlisted_GPU_process:'+str(pid))
        compute.append(dict(pid=pid,reported_name=name,verified_executable=actual,MiB=memory,allowed=allowed))
    free=shutil.disk_usage(contract['project_root']).free;needed=int(contract['minimum_free_reserve_bytes'])+int(contract['max_next_stage_budget_bytes'])
    if free<needed:reasons.append('disk_below_reserve_plus_stage_budget')
    return dict(ready=not reasons,reasons=reasons,compute=compute,free_bytes=free,required_free_bytes=needed,utc=utc(),producer_gpu_release_sha256=sha(root/'producer_gpu_release.json'))

def conditional(args):
    """Parent-approved conditional activation; never runs before all gates pass."""
    authorization=read(args.release);contract=read(authorization['contract_path'])
    assert authorization['status']=='AUTHORIZED_CONDITIONAL_NO_TRAINING'
    assert authorization['contract_sha256']==sha(authorization['contract_path'])
    assert authorization['consumer_source_sha256']==sha(__file__)==contract['consumer_source_sha256']
    assert authorization['allow_materialize_narrow_feature_and_fa_releases'] is True
    assert authorization['allow_training'] is False
    root=Path(contract['producer_root']);assert sha(root/'protocol.json')==contract['producer_protocol_sha256']
    out=Path(args.output_root);assert out.resolve()==Path(contract['conditional_output_root']).resolve();out.mkdir(exist_ok=False,parents=True);start=time.time();prior=None
    put(out/'conditional_launch.json',dict(pid=os.getpid(),command=sys.argv,utc=utc(),parent_authorization_sha256=sha(args.release),contract_sha256=sha(authorization['contract_path']),consumer_source_sha256=sha(__file__)))
    while True:
        state=conditional_readiness(contract);key=tuple(state['reasons'])
        if key!=prior:
            print('CONDITIONAL',json.dumps(state,ensure_ascii=False),flush=True);prior=key
        if state['ready']:break
        if time.time()-start>contract['maximum_wait_seconds']:
            put(out/'conditional_hold_timeout.json',dict(status='HOLD_NO_EXECUTION',state=state,utc=utc()));return
        time.sleep(15)
    put(out/'pre_activation_gate.json',state)
    qaout=out/'producer_independent_cpu_QA';producer_qa(argparse.Namespace(producer_root=str(root),output_root=str(qaout)))
    qa=read(qaout/'producer_cpu_qa.json');assert qa['status']=='PASS_ALL60_MEDIA_ONLY' and qa['producer_final_manifest_status']=='PRESENT_IDENTITY_QA_COMPLETE' and qa['passed_count']==60
    baseline=read(contract['original220_baseline_path']);assert sha(contract['original220_baseline_path'])==contract['original220_baseline_sha256']
    material=out/'dataset280';assemble(argparse.Namespace(legacy_nrt_manifest=contract['legacy_nrt_manifest'],sample_protocol=contract['sample_protocol'],reuse_root=contract['reuse_root'],producer_root=str(root),producer_layout='cells',producer_manifest=str(root/'consumer_manifest.json'),producer_qa=str(qaout/'producer_cpu_qa.json'),output_root=str(material),project_root=contract['project_root']))
    prepare(argparse.Namespace(manifest=str(material/'dataset_input_manifest.json'),output_root=str(material),fa_index=[contract['fa80_index']]))
    prepared=read(material/'prepared_manifest.json');old={x['id']:x for x in baseline['rows']};new={x['id']:x for x in prepared['rows']}
    for uid,b in old.items():
        for k in baseline['identity_fields']:assert new[uid][k]==b[k],(uid,k)
    assert prepared['counts']=={'REUSED_VERIFIED_CACHE':140,'PENDING_NEW_EXTRACTION_RELEASE':140}
    assert prepared['fa_counts']=={'REUSED_VERIFIED_STRUCTURE_ONLY':220,'PENDING_NEW_FA_RELEASE':60}
    feature_ids=[x['id'] for x in prepared['rows'] if x['feature_status']=='PENDING_NEW_EXTRACTION_RELEASE'];fa_ids=[x['id'] for x in prepared['rows'] if x['fa_status']=='PENDING_NEW_FA_RELEASE']
    assert set(feature_ids)==set(contract['new_feature_intention_ids']) and set(fa_ids)==set(contract['new_FA_intention_ids'])
    again=conditional_readiness(contract);assert again['ready'];put(out/'immediate_pre_execution_gate.json',again)
    lock=Path(contract['project_root'])/'runs/.syncnet_audio_dataset_cache_gpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for stage,ids,interpreter,modelroot,timeout in [('extract',feature_ids,prepared['engine']['required_environment']['executable'],None,3600),('fa',fa_ids,contract['fa_interpreter'],prepared['fa_engine']['model_root'],900)]:
            threshold=contract['minimum_free_reserve_bytes']+(contract['max_next_stage_budget_bytes'] if stage=='extract' else 64*1024**2);assert shutil.disk_usage(out).free>=threshold
            dest=out/('features_new140' if stage=='extract' else 'FA_new60');dest.mkdir()
            narrow=out/(stage+'_conditional_narrow_release.json');put(narrow,dict(allow_frozen_syncnet_forward=stage=='extract',allow_cpu_fa=stage=='fa',allow_training=False,prepared_manifest_sha256=sha(material/'prepared_manifest.json'),consumer_source_sha256=sha(__file__),allowed_unit_ids=ids,parent_conditional_authorization_sha256=sha(args.release),activation_gate_sha256=sha(out/'immediate_pre_execution_gate.json')))
            cmd=[interpreter,str(Path(__file__)),stage,'--manifest',str(material/'prepared_manifest.json'),'--output-root',str(dest),'--release',str(narrow)]
            if modelroot:cmd+=['--fa-model-root',modelroot]
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
            if stage=='fa':env['CUDA_VISIBLE_DEVICES']='-1'
            with (dest/'stdout.log').open('w') as so,(dest/'stderr.log').open('w') as se:
                child=subprocess.Popen(cmd,stdout=so,stderr=se,env=env);put(dest/'launch_receipt.json',dict(pid=child.pid,command=cmd,utc=utc(),source_sha256=sha(__file__)))
                try:rc=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:child.kill();child.wait();rc=124
            put(dest/'exit_receipt.json',dict(exitcode=rc,utc=utc()));assert rc==0,stage
            if stage=='extract':assert read(dest/'extract_receipt.json')['status']=='PASS_NEW_EXTRACTIONS'
    finally:
        assert lock.parent.resolve()==(Path(contract['project_root'])/'runs').resolve() and lock.read_text()==str(os.getpid());lock.unlink()
    put(out/'conditional_completion.json',dict(status='INCREMENTAL140_FEATURE_AND60_FA_COMPLETE_PENDING_PARENT_AUDIT',utc=utc(),prepared_manifest_sha256=sha(material/'prepared_manifest.json'),feature_ids=feature_ids,fa_ids=fa_ids,no_training=True,original140_not_rerun=True,original220_identity_preserved=True))
def extract(args):
    man=read(args.manifest);rows=[x for x in man['rows'] if x['feature_status']=='PENDING_NEW_EXTRACTION_RELEASE'];ids=[x['id'] for x in rows];validate_release(args,'allow_frozen_syncnet_forward',ids)
    out=Path(args.output_root);assert shutil.disk_usage(out).free>=5*1024**3
    engine=man['engine'];assert sha(engine['path'])==engine['sha256']
    assert str(Path(sys.executable))==engine['required_environment']['executable'],'Use the original scoring interpreter'
    for p,h in engine['official_source_pins'].items():assert sha(p)==h
    # Import the exact sealed, already-validated forward engine only after release.
    spec=importlib.util.spec_from_file_location('sealed_replay_engine',engine['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for key,value in [('torch',module.torch.__version__),('numpy',module.np.__version__),('scipy',module.scipy.__version__),('opencv',module.cv2.__version__)]:assert value==engine['required_environment'][key],key
    module.OUT=out;module.R=Path(man['project_root']);module.OFF=module.R/'third_party/syncnet_python'
    put(out/'protocol.json',dict(schema='syncnet_audio_dataset_frozen_replay_v1',tolerances=engine['tolerances'],engine_path=engine['path'],engine_sha256=engine['sha256'],prepared_manifest_sha256=sha(args.manifest),release_path=str(args.release),release_sha256=sha(args.release),training=False,only_new_ids=ids))
    start=time.time();put(out/'extract_started.json',dict(command=sys.argv,pid=os.getpid(),utc=utc(),consumer_source_sha256=sha(__file__),engine_sha256=engine['sha256']))
    s=module.SyncNetInstance(device='cuda');s.loadParameters(str(module.OFF/'data/syncnet_v2.model'));s.__S__.eval();module.torch.set_num_threads(2)
    recs=[]
    for x in rows:
        assert shutil.disk_usage(out).free>=12*1024**3+20*1024**2,'Preserve12GiB plus next-item scratch budget'
        recs.append(module.replay_one(x,s,False))
    put(out/'extract_receipt.json',dict(status='PASS_NEW_EXTRACTIONS' if all(x['pass_replay'] for x in recs) else 'FAIL_NEW_EXTRACTIONS',rows=recs,elapsed_s=time.time()-start,utc=utc(),exitcode=0,training=False,reused_original140_not_rerun=True))
def fa(args):
    man=read(args.manifest);rows=[x for x in man['rows'] if x['fa_status']=='PENDING_NEW_FA_RELEASE'];validate_release(args,'allow_cpu_fa', [x['id'] for x in rows])
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from funasr import AutoModel
    torch.set_num_threads(2);start=time.time();model=Path(args.fa_model_root);assert str(model)==man['fa_engine']['model_root']
    pins={str(p):sha(p) for p in model.rglob('*') if p.is_file()};assert pins==man['fa_engine']['model_pins'];aligner=AutoModel(model=str(model),device='cpu',disable_update=True);result=[]
    for x in rows:
        assert sha(x['audio_path'])==x['audio_sha256'];a=aligner.generate(input=(x['audio_path'],x['text']),data_type=('sound','text'));tokens=normalized(a[0].get('text',''));ts=a[0].get('timestamp',[]);issues=[];last=0
        with wave.open(x['audio_path']) as f:duration=f.getnframes()/16
        if tokens!=x['text'] or len(tokens)!=len(ts):issues.append('text_or_count')
        for lo,hi in ts:
            if not 0<=lo<=hi<=duration:issues.append('range')
            if lo<last:issues.append('nonmonotonic')
            last=hi
        result.append(dict(id=x['id'],audio_sha256=x['audio_sha256'],text=x['text'],tokens=list(tokens),timestamps_ms=ts,issues=sorted(set(issues)),truth='UNVERIFIED',long_intervals=[{'index':i,'interval_ms':v} for i,v in enumerate(ts) if v[1]-v[0]>1000],uncovered_tail_ms=duration-(ts[-1][1] if ts else 0)))
    put(Path(args.output_root)/'fa_new_receipt.json',dict(status='CPU_ALIGNMENT_STRUCTURE_ONLY',rows=result,weights_pins=pins,elapsed_s=time.time()-start,training=False,gpu_used=False,prepared_manifest_sha256=sha(args.manifest),command=sys.argv))
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','prepare','cpu-bridge','extract','fa','producer-qa','conditional']);p.add_argument('--manifest');p.add_argument('--output-root',required=True);p.add_argument('--project-root');p.add_argument('--legacy-nrt-manifest');p.add_argument('--sample-protocol');p.add_argument('--reuse-root');p.add_argument('--producer-root');p.add_argument('--producer-layout',choices=['cells','root'],default='cells');p.add_argument('--producer-manifest');p.add_argument('--producer-qa');p.add_argument('--fa-index',action='append',default=[]);p.add_argument('--release');p.add_argument('--fa-model-root');p.add_argument('--ids',default='BAC009S0002W0122_N_0,BAC009S0002W0122_L06_1,BAC009S0002W0122_L17_1');a=p.parse_args()
    try:globals()[a.stage.replace('-','_')](a)
    except Exception as e:
        if a.stage=='conditional' and (Path(a.output_root)/'conditional_launch.json').exists() and not (Path(a.output_root)/'conditional_failure_HOLD.json').exists():
            put(Path(a.output_root)/'conditional_failure_HOLD.json',dict(status='HOLD_FAILURE_NO_RETRY_OR_SCOPE_EXPANSION',utc=utc(),error=repr(e),consumer_source_sha256=sha(__file__)))
        raise
if __name__=='__main__':main()
