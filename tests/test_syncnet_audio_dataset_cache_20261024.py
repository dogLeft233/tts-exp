"""Consumer preparation is CPU-only, id-specific, and treats missing media as pending."""
import importlib.util,json,tempfile,argparse
from pathlib import Path
P=Path(__file__).parents[1]/'scripts/experiments/syncnet_audio_dataset_cache_20261024.py'
sp=importlib.util.spec_from_file_location('consumer',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def test_identity_binds_curve_and_crop_not_just_audio():
    a={'audio_sha256':'a','score_sha256':'s','crop_sha256':'c','activesd_sha256':'d'};b=dict(a,crop_sha256='other');assert m.identity(a)!=m.identity(b)
def test_pending_row_requires_no_media_or_gpu():
    assert 'torch' not in m.__dict__ and 'SyncNetInstance' not in m.__dict__
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);old=d/'old';old.mkdir();(old/'alignment.json').write_text('{"audio":[]}');(old/'fa_started.json').write_text(json.dumps({'model_pins':{str(old/'model.bin'):'h'}}));inp=d/'input.json';out=d/'out'
        inp.write_text(json.dumps({'units':[{'id':'one','status':'pending','score_path':'/not/exists'}],'reuse_root':str(old),'project_root':str(d)}))
        original=m.cache_index;m.cache_index=lambda _: ({},{})
        try:m.prepare(argparse.Namespace(manifest=str(inp),output_root=str(out)))
        finally:m.cache_index=original
        plan=json.loads((out/'prepared_manifest.json').read_text());assert plan['counts']=={'pending':1} and plan['rows'][0]['materialized_cache'] is False
def test_parent_release_exact_ids_and_source():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);manifest=d/'input.json';manifest.write_text('{}');release=d/'release.json'
        release.write_text(json.dumps({'allow_frozen_syncnet_forward':True,'prepared_manifest_sha256':m.sha(manifest),'consumer_source_sha256':m.sha(P),'allowed_unit_ids':['a']}));a=argparse.Namespace(release=str(release),manifest=str(manifest));assert m.validate_release(a,'allow_frozen_syncnet_forward',['a'])
        try:m.validate_release(a,'allow_frozen_syncnet_forward',['a','b'])
        except AssertionError:pass
        else:raise AssertionError('unreleased b accepted')
def test_generation_identity_rejects_seed_and_reference_and_text_drift():
    s={'sample_id':'BAC009S0017W0121','speaker_id':'S0017','transcript':'不 少 平台','input24_sha256':'ref'}
    p={'seed_bases':[2026102201,2026102202,2026102203],'generation_kwargs':dict(do_sample=True,temperature=.9,top_k=50,top_p=1.,max_new_tokens=512),'non_streaming_mode':False,'no_training':True}
    core=dict(p['generation_kwargs'],non_streaming_mode=False,languages=['Chinese'],subtalker_dosample=True,subtalker_top_k=50,subtalker_top_p=1.,subtalker_temperature=.9,repetition_penalty=1.05)
    g={'sample_id':s['sample_id'],'speaker_id':'S0017','model':'L17','draw_index':1,'role':'L17_norm1','seed':2026103901,'target_text':'不少平台','reference_text':'不少平台','reference_audio_sha256':'ref','core_explicit_kwargs':core,'talker_effective_kwargs':dict(p['generation_kwargs'])}
    assert m.validate_generation_contract(g,s,p)['expected_seed']==2026103901
    for key,value in [('seed',2026103902),('reference_audio_sha256','other'),('target_text','换文本')]:
        try:m.validate_generation_contract(dict(g,**{key:value}),s,p)
        except AssertionError:pass
        else:raise AssertionError('identity drift accepted: '+key)
def test_conditional_gate_holds_missing_producer_busy_gpu_and_low_disk():
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);root=td/'producer';root.mkdir();(root/'executor_attempts/001').mkdir(parents=True)
        c={'producer_root':str(root),'project_root':str(td),'minimum_free_reserve_bytes':12*1024**3,'max_next_stage_budget_bytes':512*1024**2}
        assert not m.conditional_readiness(c)['ready']
        data={'media_completion.json':{'status':'MEDIA_COMPLETE_FEATURE_HOLD','completed_tasks':80,'expected_new_media':60,'no_training':True},'consumer_manifest.json':{'status':'COMPLETE_MEDIA_ONLY'},'media_cpu_acceptance.json':{'status':'COMPLETE'},'producer_gpu_release.json':{'released':True},'executor_attempts/001/exit.json':{'returncode':0,'status':'MEDIA_COMPLETE_FEATURE_HOLD','pid':123}}
        for name,obj in data.items():(root/name).write_text(json.dumps(obj))
        saved=(m.subprocess.run,m.os.readlink,m.shutil.disk_usage,m.process_alive)
        state={'gpu':'2488, /usr/libexec/gnome-remote-desktop-daemon, 313','free':13*1024**3,'producer_alive':False}
        m.subprocess.run=lambda *a,**k:SimpleNamespace(stdout=state['gpu'])
        m.os.readlink=lambda _: '/usr/libexec/gnome-remote-desktop-daemon'
        m.shutil.disk_usage=lambda _:SimpleNamespace(free=state['free'])
        m.process_alive=lambda _:state['producer_alive']
        try:
            assert m.conditional_readiness(c)['ready']
            state['producer_alive']=True;assert 'producer_executor_still_alive' in m.conditional_readiness(c)['reasons'];state['producer_alive']=False
            state['free']=12*1024**3+512*1024**2-1;assert 'disk_below_reserve_plus_stage_budget' in m.conditional_readiness(c)['reasons'];state['free']=13*1024**3
            state['gpu']='9999, /tmp/python, 1000';assert not m.conditional_readiness(c)['ready']
        finally:m.subprocess.run,m.os.readlink,m.shutil.disk_usage,m.process_alive=saved
