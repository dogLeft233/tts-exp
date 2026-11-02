"""SparseSync official-example parity gate, native 5s scoring and independent audit.

The author's offset classifier is unchanged. P(offset=0) is our continuous
preference endpoint; it is not the author's dataset-level accuracy statistic.
"""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path('/home/wjj/tts-audio/tts-exp')
DEPLOY = Path('/home/wjj/av-sync-eval-20261028')
SOURCE = DEPLOY/'SparseSync-source'
OLD = ROOT/'runs/cloud_mtd_synchformer_20261028_attempt01'
RUN = ROOT/'runs/sparsesync_official_cloud_rescore_20261028_attempt01'
EXP = '22-07-13T22-25-49'
WEIGHTS = DEPLOY/'weights/sparsesync'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def summarize(pairs):
    import numpy as np
    if not pairs:
        return {'pairs': 0, 'status': 'UNMEASURABLE'}
    d = np.array([p['delta'] for p in pairs])
    sc = np.array([p['delta_sync_c'] for p in pairs])
    groups = {}
    for p in pairs:
        groups.setdefault(p['source'], []).append(p['delta'])
    means = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(20261028)
    bs = means[rng.integers(0, len(means), (10000, len(means)))].mean(1)
    return dict(pairs=len(pairs), sources=len(means),
                natural_mean=float(np.mean([p['N'] for p in pairs])),
                tts_mean=float(np.mean([p['T'] for p in pairs])),
                mean_delta=float(d.mean()), source_bootstrap_ci95=np.percentile(bs,[2.5,97.5]).tolist(),
                tts_wins=int((d>0).sum()), natural_wins=int((d<0).sum()), ties=int((d==0).sum()),
                tts_win_fraction=float((d>0).mean()), conflict_count=int((d*sc<0).sum()),
                conflict_fraction=float((d*sc<0).mean()),
                sync_tts_metric_natural=int(((sc>0)&(d<0)).sum()),
                sync_natural_metric_tts=int(((sc<0)&(d>0)).sum()),
                sync_c_mean_delta=float(sc.mean()), sync_c_tts_wins=int((sc>0).sum()))


def shift_pcm(x, amount):
    import numpy as np
    out = np.zeros_like(x)
    if amount > 0:
        out[amount:] = x[:-amount]
    elif amount < 0:
        out[:amount] = x[-amount:]
    else:
        out[:] = x
    return out


def freeze():
    rows = read(OLD/'inputs.json')
    assert len(rows)==80
    eligible = [r for r in rows if r['eligible_5s']]
    protocol = dict(status='FROZEN_BEFORE_MODEL_SCORES',cohort='previous Mandarin 20-source cloud TTS native cohort',
        model='official LRS3 NoFaceCrop 21-class '+EXP,
        source=read(SOURCE/'source_receipt.json')['commit'],
        weights_bytes=653037857,
        primary='P(offset=0), softmax output from official example; project scalar endpoint, not benchmark accuracy',
        secondary='negative probability-weighted absolute offset seconds',
        support='unmodified official 5s test transforms; same frame-grid center as prior Synchformer; exclude short; no padding or stretching',
        precision='FP32 as official scripts/example.py; no AMP',
        audio='reuse existing normalized native 16k waveform via float-PCM MOV and official torchvision reader; assert exact original samples; no new RMS',
        init='only extractor initialization ckpt_path=None, followed by full checkpoint strict=True; no source edits',
        parity='execute official example.main; skip reconstruction visualization only; compare inputs, logits and probabilities to direct batch path before scoring',
        controls='all eligible natural fixed videos, full PCM shifted +/-6400 samples with zero-fill, no search',
        control_gate='P0 declines on >=80% of physical shifts; also report predicted-offset relative error, never tune after preference results',
        stats='strict sign/ties separate; paired native support only; bootstrap 10000 source clusters seed20261028',
        excluded_short=len(rows)-len(eligible), eligible_media=len(eligible))
    for name,obj in [('protocol.json',protocol),('inputs.json',rows),('sync_pairs.json',read(OLD/'sync_pairs.json'))]:
        p=RUN/name
        if p.exists():assert read(p)==obj, name+' already frozen differently'
        else:write(p,obj)
    snapshot=RUN/'source_snapshot'
    for f in ['scripts/example.py','scripts/train_utils.py','dataset/dataset_utils.py','dataset/transforms.py',
              'model/sync_model.py','model/modules/feature_selector.py','model/modules/feature_extractors.py','source_receipt.json']:
        dst=snapshot/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/f,dst)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    shutil.copyfile(WEIGHTS/f'cfg-{EXP}.yaml',snapshot/f'cfg-{EXP}.yaml')
    print('FROZEN',protocol['eligible_media'],flush=True)


def load():
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)
    import scripts.example as official
    cfg=OmegaConf.load(WEIGHTS/f'cfg-{EXP}.yaml')
    init=copy.deepcopy(cfg)
    init.model.params.afeat_extractor.params.ckpt_path=None
    init.model.params.vfeat_extractor.params.ckpt_path=None
    model,_=official.get_model(init,torch.device('cuda'))
    ckpt=torch.load(WEIGHTS/f'{EXP}.pt',map_location='cpu',weights_only=False)
    receipt=model.load_state_dict(ckpt['model'],strict=True)
    assert not receipt.missing_keys and not receipt.unexpected_keys
    del ckpt
    model.eval()
    transform=official.get_transforms(cfg)['test']
    grid=official.make_class_grid(-cfg.data.max_off_sec,cfg.data.max_off_sec,cfg.model.params.transformer.params.num_offset_cls)
    write(RUN/'model_load.json',dict(strict=True,parameters=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__,device=torch.cuda.get_device_name(),precision='FP32',
        full_checkpoint_bytes=(WEIGHTS/f'{EXP}.pt').stat().st_size,
        config_unchanged_except_preload_paths=True,grid=grid.tolist()))
    return official,cfg,model,transform,grid


def media(row):
    # This exact adapter was already decoded by official Synchformer reader.
    p=OLD/'synchformer_media'/(row['id']+'.mov')
    assert p.is_file(), p
    return p


def tensors(official,transform,row,amount=0):
    import numpy as np
    import soundfile as sf
    import torch
    rgb,audio,meta=official.get_video_and_audio(str(media(row)),get_meta=True)
    ref,sr=sf.read(row['audio_path'],dtype='float32')
    assert sr==16000 and np.array_equal(audio.numpy(),ref), 'Reader changed PCM'
    assert meta['video']['fps']==[25] and meta['audio']['framerate']==[16000]
    item=dict(video=rgb,audio=torch.from_numpy(shift_pcm(ref,amount)),meta=copy.deepcopy(meta),
        path=str(media(row)),split='test',targets=dict(v_start_i_sec=row['center_start_5s']/25,
        offset_sec=0.,vggsound_target=0,vggsound_label='PLACEHOLDER'))
    item=transform(item)
    aud,vid,targets=official.prepare_inputs(torch.utils.data.default_collate([item]),torch.device('cuda'))
    assert tuple(vid.shape)==(1,125,3,224,224) and tuple(aud.shape)==(1,1,257,626)
    return aud,vid,targets


def output(logits,grid):
    import torch
    z=logits.detach().float().cpu().flatten()
    assert z.shape==(21,) and torch.isfinite(z).all()
    p=z.softmax(0)
    zero=int(grid.abs().argmin())
    return dict(logits=z.tolist(),probabilities=p.tolist(),p_zero=float(p[zero]),
        negative_expected_abs_offset=float(-(p*grid.abs()).sum()),predicted_offset=float(grid[z.argmax()]))


def parity(official,cfg,model,transform,grid,row):
    """Run unmodified official main with instrumentation outside model/transforms."""
    import torch
    dest=SOURCE/'logs/sync_models'/EXP;dest.mkdir(parents=True,exist_ok=True)
    for name in [f'cfg-{EXP}.yaml',f'{EXP}.pt']:
        p=dest/name
        if not p.exists():p.symlink_to(WEIGHTS/name)
    capture={};recorders=[]
    orig={k:getattr(official,k) for k in ['get_model','prepare_inputs','decode_single_video_prediction','reconstruct_video_from_input','Recorder']}
    def capture_inputs(*args,**kwargs):
        a,v,t=orig['prepare_inputs'](*args,**kwargs)
        capture['audio']=a.cpu().clone();capture['video']=v.cpu().clone()
        return a,v,t
    def capture_decode(z,g,item):
        capture['logits']=z.detach().cpu().clone()
        capture['probabilities']=orig['decode_single_video_prediction'](z,g,item).cpu().clone()
        return capture['probabilities']
    def recorder(*args,**kwargs):
        r=orig['Recorder'](*args,**kwargs);recorders.append(r);return r
    official.get_model=lambda *args:(model,model)
    official.prepare_inputs=capture_inputs
    official.decode_single_video_prediction=capture_decode
    official.reconstruct_video_from_input=lambda *args:None
    official.Recorder=recorder
    stdout=io.StringIO()
    try:
        with torch.inference_mode(),contextlib.redirect_stdout(stdout):
            official.main(SimpleNamespace(exp_name=EXP,vid_path=str(media(row)),vfps=25,afps=16000,
                input_size=256,device='cuda',offset_sec=0.,v_start_i_sec=row['center_start_5s']/25))
    finally:
        for r in recorders:r.eject()
        for k,v in orig.items():setattr(official,k,v)
    aud,vid,targets=tensors(official,transform,row)
    with torch.inference_mode():_,z=model(vid,aud,targets)
    report=dict(status='PASS',id=row['id'],official_main_executed=True,
        skipped_only_visual_reconstruction=True,strict_checkpoint_reload_in_main=True,
        audio_max_abs_diff=float((capture['audio']-aud.cpu()).abs().max()),
        video_max_abs_diff=float((capture['video']-vid.cpu()).abs().max()),
        logits_max_abs_diff=float((capture['logits']-z.cpu()).abs().max()),
        probabilities_max_abs_diff=float((capture['probabilities'].flatten()-z.cpu().softmax(-1).flatten()).abs().max()))
    assert report['audio_max_abs_diff']==report['video_max_abs_diff']==0
    assert report['logits_max_abs_diff']<=1e-5 and report['probabilities_max_abs_diff']<=1e-6
    (RUN/'official_example_stdout.txt').write_text(stdout.getvalue())
    write(RUN/'official_parity.json',report)
    print('OFFICIAL_PARITY_PASS',report,flush=True)


def run():
    import torch
    torch.set_num_threads(4);torch.manual_seed(20261028)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    official,cfg,model,transform,grid=load()
    rows=[r for r in read(RUN/'inputs.json') if r['eligible_5s']]
    parity(official,cfg,model,transform,grid,rows[0])
    assert read(RUN/'official_parity.json')['status']=='PASS'
    # Controls precede formal preference scoring.
    controls=[]
    with torch.inference_mode():
        for r in [r for r in rows if r['arm']=='N']:
            outs={}
            for amount in [0,-6400,6400]:
                aud,vid,targets=tensors(official,transform,r,amount)
                _,z=model(vid,aud,targets);outs[str(amount)]=output(z,grid)
            for amount in [-6400,6400]:
                n,s=outs['0'],outs[str(amount)]
                expected=-amount/16000
                observed=s['predicted_offset']-n['predicted_offset']
                controls.append(dict(id=r['id'],shift_samples=amount,native=n,shifted=s,
                    p_zero_lower=s['p_zero']<n['p_zero'],expected_relative_offset=expected,
                    observed_relative_offset=observed,relative_offset_error=abs(observed-expected)))
            write(RUN/'known_shift_controls.json',dict(status='RUNNING',rows=controls))
            print('CONTROL',len(controls),r['id'],flush=True)
    lower=sum(c['p_zero_lower'] for c in controls)
    cr=dict(status='PASS' if lower>=.8*len(controls) else 'CONTROL_FAILED',rows=controls,
        p_zero_lower=lower,total=len(controls),fraction=lower/len(controls),
        offset_within_one_bin=sum(c['relative_offset_error']<=.20001 for c in controls))
    write(RUN/'known_shift_controls.json',cr)
    # A failed control stays visible; scores may still diagnose model behavior.
    scores={}
    with torch.inference_mode():
        for r in rows:
            aud,vid,targets=tensors(official,transform,r)
            _,z=model(vid,aud,targets)
            scores[r['id']]=dict(id=r['id'],source=r['sample_id'],arm=r['arm'],
                pcm_exact=True,start_frame=r['center_start_5s'],output=output(z,grid))
            write(RUN/'scores.json',scores)
            write(RUN/'status.json',dict(status='RUNNING',scored=len(scores),total=len(rows)))
            print('SCORED',len(scores),len(rows),r['id'],flush=True)
    write(RUN/'status.json',dict(status='SCORES_COMPLETE_PENDING_AUDIT',scored=len(scores),control_status=cr['status']))


def audit():
    """Independent NumPy softmax and paired reductions from saved raw logits."""
    import numpy as np
    scores=read(RUN/'scores.json');rows=read(RUN/'inputs.json')
    expected={r['id'] for r in rows if r['eligible_5s']}
    assert set(scores)==expected
    grid=np.array(read(RUN/'model_load.json')['grid']);zero=np.abs(grid).argmin()
    mx=0
    independent={}
    for key,r in scores.items():
        o=r['output'];z=np.array(o['logits'],dtype=np.float64)
        e=np.exp(z-z.max());p=e/e.sum()
        fields=dict(p_zero=float(p[zero]),negative_expected_abs_offset=float(-(p*abs(grid)).sum()))
        mx=max(mx,float(np.max(abs(p-np.array(o['probabilities'])))))
        for f,v in fields.items():assert abs(v-o[f])<2e-6
        assert o['predicted_offset']==float(grid[z.argmax()])
        independent[key]=fields
    assert mx<2e-6
    controls=read(RUN/'known_shift_controls.json')
    control_ids={r['id'] for r in rows if r['eligible_5s'] and r['arm']=='N'}
    assert len(controls['rows'])==2*len(control_ids)
    for c in controls['rows']:
        assert c['id'] in control_ids and c['shift_samples'] in [-6400,6400]
        assert np.max(abs(np.array(c['native']['logits'])-np.array(scores[c['id']]['output']['logits'])))<1e-5
        assert c['expected_relative_offset']==-c['shift_samples']/16000
        assert c['p_zero_lower']==(c['shifted']['p_zero']<c['native']['p_zero'])
    nmap={r['source']:key for key,r in scores.items() if r['arm']=='N'}
    pairsets={};summary={}
    for field in ['p_zero','negative_expected_abs_offset']:
        ps=[]
        for p in read(RUN/'sync_pairs.json'):
            if p['source'] not in nmap or p['id'] not in scores:continue
            n=scores[nmap[p['source']]]['output'][field];t=scores[p['id']]['output'][field]
            ps.append(dict(id=p['id'],source=p['source'],strict_reference=p['strict_reference'],N=n,T=t,
                delta=t-n,delta_sync_c=p['delta']['SyncC']))
        pairsets[field]=ps
        summary[field]=dict(all=summarize(ps),strict12=summarize([p for p in ps if p['strict_reference']]))
    diagnostics={}
    for arm in ['N','C']:
        group=[r for r in scores.values() if r['arm']==arm]
        if not group:continue
        prob=np.array([r['output']['p_zero'] for r in group])
        offsets={}
        for r in group:
            key=f"{r['output']['predicted_offset']:.2f}"
            offsets[key]=offsets.get(key,0)+1
        diagnostics[arm]=dict(media=len(group),p_zero_min=float(prob.min()),p_zero_median=float(np.median(prob)),
            p_zero_max=float(prob.max()),p_zero_over_099=int((prob>.99).sum()),predicted_offset_counts=offsets)
    write(RUN/'paired_scores.json',pairsets)
    precision=[]
    for p in pairsets['p_zero']:
        d=independent[p['id']]['p_zero']-independent[nmap[p['source']]]['p_zero']
        precision.append(dict(id=p['id'],delta_p0_fp32=p['delta'],delta_p0_float64=d,
            same_sign=bool(np.sign(d)==np.sign(p['delta']))))
    write(RUN/'saturation_precision_diagnostic.json',dict(status='PASS',posthoc_diagnostic=True,
        float64_sign_agree=sum(int(p['same_sign']) for p in precision),pairs=len(precision),
        min_abs_fp32_delta=min(abs(p['delta_p0_fp32']) for p in precision),rows=precision,
        controls_relative_errors=[c['relative_offset_error'] for c in controls['rows']]))
    write(RUN/'summary.json',dict(status='COMPLETE',views=summary,controls={k:v for k,v in controls.items() if k!='rows'},
        media=len(scores),excluded_short=80-len(scores),native_diagnostics=diagnostics,
        comparison_scope='Sync-C is the prior full-native-clip score on matched IDs; SparseSync sees center 5s',
        interpretation='scorer preference on native 5s support; no visual ground truth claim'))
    write(RUN/'independent_audit.json',dict(status='PASS',media=len(scores),softmax_max_abs_diff=mx,
        paired_scores=sum(len(p) for p in pairsets.values()),official_parity=read(RUN/'official_parity.json')))
    write(RUN/'status.json',dict(status='COMPLETE_OFFICIAL_PARITY_AND_AUDIT',scored=len(scores),control_status=controls['status']))
    shutil.copyfile(__file__,RUN/'analysis_source_snapshot.py')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','audit'])
    args=p.parse_args();dict(freeze=freeze,run=run,audit=audit)[args.stage]()
