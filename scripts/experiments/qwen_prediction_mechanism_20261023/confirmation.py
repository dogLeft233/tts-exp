"""Bounded independent-of-calibration confirmation; no training or score search."""
from pathlib import Path
import argparse,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha,setup,seed,trace_full,persist_trace
from scripts.experiments.qwen_prediction_mechanism_20261023.budget import guard

POLICIES=['baseline','coarse','fine','both']

def prepare(run):
    gate=read(run/'B_gate_and_calibration.json');assert gate['status']=='ADVANCE_C'
    assert read(run/'B_independent_CPU_audit.json')['status']=='PASS'
    oldroot=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01';old=read(oldroot/'protocol.json')
    bank=read(run/'B_protocol.json');exclude={s['sample_id'] for s in bank['samples']}
    samples=sorted([s for s in old['samples'] if s['sample_id'] not in exclude],key=lambda s:s['sample_id'])[:10]
    assert len(samples)==10 and len({s['speaker_id'] for s in samples})==10
    aa=read(oldroot/'audio_manifest.json')['records'];rows=[];pins={str(oldroot/'protocol.json'):sha(oldroot/'protocol.json')}
    d=read(ROOT/'runs/qwen17_level_comparison10_20261023_attempt02/protocol.json')['downstream']
    for key in ['model','weights_sha256','face','face_sha256','fps','options']:
        assert old['downstream'][key]==d[key],('old baseline downstream differs',key)
    oldvideos=read(oldroot/'video_manifest.json')
    assert oldvideos['weights_sha256']==d['weights_sha256'] and oldvideos['inference_sha256']==d['inference_sha256']
    oldofficial=ROOT/'runs/qwen_nrt_official_rescore_20261023_attempt01/protocol.json'
    official=read(oldofficial);assert official['min_track']==25
    pins[str(oldofficial)]=sha(oldofficial);pins[str(oldroot/'video_manifest.json')]=sha(oldroot/'video_manifest.json')
    for f,digest in official['source_and_weight_pins'].items():
        assert sha(f)==digest,('original official source/weight pin',f);pins[f]=digest
    for sample in samples:
        assert 3<=sample['duration_s']<=8 and ''.join(sample['transcript'].split())
        for key in ['source','input24','transcript']:
            path=sample[key+'_path'];digest=sample[key+'_sha256'];assert sha(path)==digest;pins[path]=digest
        for draw in range(1,4):
            a=next(a for a in aa if a['sample_id']==sample['sample_id'] and a['role']=='T'+str(draw))
            recpath=Path(a['source_path']).parent/'record.json';rec=read(recpath)
            text=''.join(sample['transcript'].split())
            assert rec['sample_id']==sample['sample_id'] and rec['target_text']==rec['reference_text']==text
            assert rec['seed']==2026102200+draw+100*int(sample['speaker_id'][1:])
            assert rec['path']==a['source_path'] and rec['sha256']==a['source_sha256']
            for key in ['new_codes','reference_codes','talker_sequences']:
                value=rec[key];assert sha(value['path'])==value['sha256'];pins[value['path']]=value['sha256']
            scorepath=ROOT/f'runs/qwen_nrt_official_rescore_20261023_attempt01/scores/{sample["sample_id"]}_T{draw}.json'
            score=read(scorepath);assert score['status']=='COMPLETE' and score['canonical_audio_sha256']==a['sha256']
            v=next(v for v in oldvideos['records'] if v['sample_id']==sample['sample_id'] and v['role']=='T'+str(draw))
            assert v['audio_sha256']==a['sha256'] and v['sha256']==score['source_video_sha256']
            cmd=score['pipeline_receipt']['command'];assert cmd[cmd.index('--min_track')+1]=='25'
            for path,digest in [(score['video_path'],score['source_video_sha256']),
                    (score['input_mkv'],score['input_mkv_sha256']),
                    (score['tracks_pickle_path'],score['tracks_pickle_sha256']),
                    (score['activesd_path'],score['activesd_sha256'])]:
                assert sha(path)==digest; pins[path]=digest
            for path,digest in score['checks'].items():assert sha(path)==digest;pins[path]=digest
            for track in score['tracks']:
                meta=track['metadata'];assert sha(meta['crop_file'])==meta['crop_sha256'];pins[meta['crop_file']]=meta['crop_sha256']
            assert sha(a['path'])==a['sha256'] and sha(a['source_path'])==a['source_sha256']
            for f in [recpath,scorepath,Path(a['path']),Path(a['source_path'])]:pins[str(f)]=sha(f)
            for model in ['L06','L17']:
                known={}
                for policy in POLICIES:
                    temps=[.9,.9]
                    for i,category in enumerate(['coarse','fine']):
                        c=gate['temperatures'][model][category]
                        if policy in [category,'both'] and c['status']=='MATCHED':temps[i]=c['temperature']
                    key=tuple(temps);identity=f'{sample["sample_id"]}_{model}_{policy}_{draw}'
                    duplicate=known.get(key);known.setdefault(key,identity)
                    rows.append({'id':identity,'sample_id':sample['sample_id'],'speaker_id':sample['speaker_id'],'model':model,'policy':policy,'draw':draw,'seed':rec['seed'],'temperatures':temps,'duplicate_of':duplicate,'reuse_L06_baseline':model=='L06' and policy=='baseline','old_audio':a,'old_record':str(recpath),'old_score':str(scorepath),'old_score_sha':sha(scorepath)})
    source=Path(__file__);pins[str(source)]=sha(source)
    for file in ['B_gate_and_calibration.json','B_independent_CPU_audit.json','B_protocol.json','frozen_design.md']:
        pins[str(run/file)]=sha(run/file)
    test=ROOT/'tests/test_qwen_prediction_mechanism_20261023.py';pins[str(test)]=sha(test)
    coarse_protocol=ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json'
    pins[str(coarse_protocol)]=sha(coarse_protocol)
    for f,digest in read(coarse_protocol)['pins'].items():
        assert sha(f)==digest,f;pins[f]=digest
    assert read(run/'C_content_addendum.json')['status']=='FROZEN_BEFORE_C'
    pins[str(run/'C_content_addendum.json')]=sha(run/'C_content_addendum.json')
    for f in Path(__file__).parent.glob('*.py'):pins[str(f)]=sha(f)
    olddown=ROOT/'scripts/experiments/qwen17_level_comparison10_20261023/downstream.py';pins[str(olddown)]=sha(olddown)
    normalize_source=ROOT/'scripts/experiments/qwen_tts_decline_audit_20261022.py';pins[str(normalize_source)]=sha(normalize_source)
    for f,digest in read(ROOT/'runs/qwen17_level_comparison10_20261023_attempt02/protocol.json')['source_pins'].items():
        assert sha(f)==digest;pins[f]=digest
    manifest={'status':'FROZEN_BEFORE_C','samples':samples,'rows':rows,'pins':pins,'B_gate_sha':sha(run/'B_gate_and_calibration.json'),'downstream':d,'target_rms':.0376838172675746,'generation_kwargs':old['generation_kwargs'],'seed_bases':old['seed_bases'],'logical_units':240,'unique_units':sum(r['duplicate_of'] is None for r in rows),'new_syntheses_max':sum(r['duplicate_of'] is None and not r['reuse_L06_baseline'] for r in rows),'selection':'first sample_id-sorted ten original40 after excluding B10; require unchanged original qualified input materials; no score selection','content_qualification':'EOS True and original duration-ratio [0.5,2.0]; ASR settings unchanged; serious content operationalization requires frozen addendum before execution','comparisons':['P1=L06_both-L06_baseline','P2=(L17_baseline-L06_baseline)-(L17_both-L06_both)'],'source_bootstrap':{'n':20000,'seed':2026102308,'CI':.975},'limits':{'new_TTS':240,'new_video':240,'new_score':240,'per_unit_s':900,'total_GPU_s':28800,'total_wall_s':43200}}
    save(run/'C_protocol.json',manifest)

def generate(model,run):
    import numpy as np,soundfile as sf,torch
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    p=read(run/'C_protocol.json');assert read(run/'C_content_addendum.json')['status']=='FROZEN_BEFORE_C'
    for f,digest in p['pins'].items():assert sha(f)==digest,f
    guard(run)
    w=setup(model,read(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json'))
    out=run/('C_generation_'+model);out.mkdir();records=[]
    save(out/'started.json',{'pid':os.getpid(),'unix':time.time(),'protocol_sha':sha(run/'C_protocol.json')})
    try:
        for sample in p['samples']:
            sid=sample['sample_id'];x,sr=sf.read(sample['input24_path'],dtype='float32');text=''.join(sample['transcript'].split())
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('reference prompt900s')));signal.alarm(900)
            try:
                with torch.inference_mode():prompt=w.create_voice_clone_prompt(ref_audio=(x,sr),ref_text=text,x_vector_only_mode=False)
            finally:signal.alarm(0)
            ref=h.codes(prompt[0].ref_code);sampledir=out/sid;sampledir.mkdir();refrec=h.save_array(sampledir/'reference_codes.npy',ref)
            old=read(next(r['old_record'] for r in p['rows'] if r['sample_id']==sid))
            assert np.array_equal(ref,np.load(old['reference_codes']['path']))
            for row in [r for r in p['rows'] if r['sample_id']==sid and r['model']==model]:
                guard(run)
                unit=out/row['id'];unit.mkdir();start=time.monotonic();save(unit/'started.json',{'pid':os.getpid(),'unix':time.time(),'id':row['id']})
                if row['duplicate_of']:
                    result={'id':row['id'],'status':'EXACT_REUSE','duplicate_of':row['duplicate_of']};save(unit/'result.json',result);records.append(result);continue
                if row['reuse_L06_baseline']:
                    rec=read(row['old_record']);forced=np.load(rec['new_codes']['path']);decode=w.model.speech_tokenizer.decode
                    w.model.speech_tokenizer.decode=lambda encoded:([np.zeros(24000,dtype=np.float32)],24000)
                    seed(row['seed']);kwargs=dict(p['generation_kwargs']);kwargs['max_new_tokens']=len(forced)+1
                    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('baseline TF unit900s')));signal.alarm(900)
                    try:
                        with torch.inference_mode(),h.capture_official(w) as cap,trace_full(w,forced) as tr:
                            w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**kwargs)
                        assert np.array_equal(cap['new_codes'][0],forced);persist_trace(unit/'trace.npz',tr)
                    finally:w.model.speech_tokenizer.decode=decode;signal.alarm(0)
                    result={'id':row['id'],'status':'OLD_BASELINE_REUSE','record':row['old_record'],'record_sha':sha(row['old_record']),'old_audio':row['old_audio'],'old_score':row['old_score'],'EOS':rec['eos_in_talker_sequences'],'duration_ratio':rec['duration_s']/sample['duration_s'],'entropy_trace_provenance':'complete historical native-code teacher-forced reconstruction, A native/TF qualified; not newly synthesized baseline','elapsed_s':time.monotonic()-start};save(unit/'result.json',result);records.append(result);continue
                cap=tr=None
                try:
                    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('confirmation unit900s')));signal.alarm(900)
                    kwargs=dict(p['generation_kwargs']);kwargs.update(temperature=row['temperatures'][0],subtalker_temperature=row['temperatures'][1]);seed(row['seed'])
                    with torch.inference_mode(),h.capture_official(w) as cap,trace_full(w) as tr:
                        waves,rate=w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**kwargs)
                    np.save(unit/'returned_wave_unvalidated.npy',np.asarray(waves[0]))
                    codes=cap['new_codes'][0];new=h.save_array(unit/'codes.npy',codes);h.save_array(unit/'sequences.npy',cap['talker_sequences']);persist_trace(unit/'trace.npz',tr)
                    wave=h.waveform(waves[0],rate);raw=h.save_wave(unit/'raw_float.wav',wave);joined=cap['decode_codes'][0]
                    assert np.array_equal(joined,np.concatenate([ref,codes]));cut=h.official_cut(len(ref),len(joined),len(cap['full_waves'][0]));assert np.array_equal(wave,cap['full_waves'][0][cut:])
                    assert cap['talker_kwargs']['temperature']==row['temperatures'][0] and cap['talker_kwargs']['subtalker_temperature']==row['temperatures'][1]
                    eos=bool(np.any(cap['talker_sequences']==w.model.config.talker_config.codec_eos_token_id));ratio=len(wave)/rate/sample['duration_s']
                    result={'id':row['id'],'status':'COMPLETE','raw':raw,'codes':new,'reference_codes':refrec,'EOS':eos,'duration_ratio':ratio,'qualified_EOS_duration':eos and .5<=ratio<=2.,'elapsed_s':time.monotonic()-start,'talker_kwargs':cap['talker_kwargs'],'official_ratio_cut':cut}
                except BaseException as e:
                    result={'id':row['id'],'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-start}
                    try:
                        if tr is not None and not (unit/'trace.npz').exists():persist_trace(unit/'partial_trace.npz',tr)
                        if cap is not None and cap.get('new_codes') and not (unit/'codes.npy').exists():np.save(unit/'partial_codes.npy',cap['new_codes'][0])
                    except BaseException as secondary:result['partial_capture_error']=repr(secondary)
                finally:signal.alarm(0)
                save(unit/'result.json',result);records.append(result);print(row['id'],result['status'],flush=True)
        save(out/'completion.json',{'status':'COMPLETE_WITH_RECORDED_FAILURES','records':records,'count':len(records)})
    except BaseException as e:save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'records':records});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','generate']);ap.add_argument('--model');ap.add_argument('--run',required=True);a=ap.parse_args()
    if a.stage=='prepare':prepare(Path(a.run))
    else:generate(a.model,Path(a.run))
