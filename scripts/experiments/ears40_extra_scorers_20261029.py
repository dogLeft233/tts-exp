"""EARS40 adapters for unchanged AVS_u replication, Synchformer and SparseSync.

Inputs and scientific choices are frozen by the parent experiment protocol.
Each CLI invocation owns only one model, avoiding upstream module-name collisions.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/wjj/tts-audio/tts-exp')
RUN = ROOT/'runs/ears40_w2l_nine_scorers_20261029_attempt01'
DEPLOY = Path('/home/wjj/av-sync-eval-20261028')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def validate_rows(rows):
    if len(rows) != 80 or len({r['id'] for r in rows}) != 80:
        raise ValueError('Exactly 80 unique media required')
    sources = {}
    for row in rows:
        sources.setdefault(row['sample_id'], []).append(row['arm'])
        if row['arm'] not in ('N', 'T'):
            raise ValueError('Expected N/T arm')
        if row['eligible_5s'] != (row['native_frames'] >= 125):
            raise ValueError('Invalid native support eligibility')
        if row['eligible_5s'] and row['center_start_5s'] != (row['native_frames']-125)//2:
            raise ValueError('Expected frame-grid center5s')
    if len(sources) != 40 or any(sorted(v) != ['N', 'T'] for v in sources.values()):
        raise ValueError('One natural and one TTS per 40 speakers required')
    return rows


def prepare(model):
    rows = validate_rows(read(RUN/'inputs.json'))
    root = RUN/model; root.mkdir(exist_ok=True)
    for name in ['inputs.json', 'protocol.json']:
        src, dest = RUN/name, root/name
        if dest.exists():
            assert dest.read_bytes() == src.read_bytes(), 'Frozen artifact changed'
        else: shutil.copyfile(src, dest)
    for row in rows:
        for kind in ('audio', 'video'):
            assert sha(row[kind+'_path']) == row[kind+'_sha256'], row['id']
    adapter_files={'avs_u':'avs_u_zh_pilot_20261027.py',
        'synchformer':'cloud_mtd_synchformer_20261028.py',
        'sparsesync':'sparsesync_official_cloud_rescore_20261028.py'}
    reused=Path(__file__).with_name(adapter_files[model])
    write(root/'adapter_receipt.json', dict(script_sha256=sha(__file__),
        reused_adapter=str(reused),reused_adapter_sha256=sha(reused),
        parent_protocol_sha256=sha(RUN/'protocol.json'), inputs_sha256=sha(RUN/'inputs.json'),
        model=model, independent_sources=40, real_pcm_controls='all40 N +/-6400 samples zero-fill'))
    return root, rows


def control_summary(rows):
    if not rows:
        return dict(status='UNMEASURABLE', total=0, lower=0)
    lower = sum(bool(r['lower']) for r in rows)
    return dict(status='PASS' if lower >= .8*len(rows) else 'CONTROL_FAILED',
                total=len(rows), lower=lower, fraction=lower/len(rows), required_fraction=.8, rows=rows)


def assert_complete_controls(rows, inputs):
    expected={(r['id'],s) for r in inputs if r['arm']=='N' and r['eligible_5s'] for s in (-6400,6400)}
    actual=[(r['id'],r['shift_samples']) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==expected,'Incomplete/duplicate physical controls'


def pretrim_official_center(item, start_frame):
    """Select the frozen integer center before the author's ten-second cap.

    Full PCM is already shifted, if applicable. Both streams use exact indices.
    Author transforms still execute, with their local offset anchor at zero.
    """
    assert item['meta']['video']['fps']==[25]
    assert item['meta']['audio']['framerate']==[16000]
    start_frame=int(start_frame);a_start=start_frame*640
    assert len(item['video'])>=start_frame+125 and len(item['audio'])>=a_start+80000
    selected=dict(item,video=item['video'][start_frame:start_frame+125],
                  audio=item['audio'][a_start:a_start+80000],targets=dict(item['targets']))
    selected['targets']['v_start_i_sec']=0.
    return selected


def install_center_preselection(old, function_name, anchor, center_expression, root):
    """A bounded wrapper transformation; author transforms/model remain unchanged."""
    source=inspect.getsource(getattr(old,function_name))
    assert source.count(anchor)==1
    indent=' '*(len(anchor)-len(anchor.lstrip()))
    new=indent+'item=pretrim_official_center(item, '+center_expression+')\n'+anchor
    transformed=source.replace(anchor,new)
    dest=root/'repairs/01_integer_center_before_equalify10'/('transformed_'+function_name+'.py')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_text()==transformed
    else:dest.write_text(transformed)
    old.pretrim_official_center=pretrim_official_center
    exec(compile(transformed,str(dest),'exec'),old.__dict__)


def prepare_canonical_cache(root, rows):
    """Preserve original scores; invalidate only statically wrong integer windows."""
    repair=root/'repairs/01_integer_center_before_equalify10';repair.mkdir(parents=True,exist_ok=True)
    audit=[dict(id=r['id'],frozen_v_start=r['center_start_5s'],
        frozen_a_start=r['center_start_5s']*640,
        legacy_v_start=int((r['center_start_5s']/25)*25),
        legacy_a_start=int((r['center_start_5s']/25)*16000)) for r in rows]
    for r in audit:r['legacy_window_differs']=r['frozen_v_start']!=r['legacy_v_start'] or r['frozen_a_start']!=r['legacy_a_start']
    path=repair/'index_audit.json'
    if path.exists():assert read(path)==audit
    else:write(path,audit)
    marker=repair/'canonical_cache_prepared.json'
    if marker.exists():return
    path=root/'synchformer_scores.json'
    if path.exists():
        parity=read(repair/'pre_post_tensor_parity.json')
        assert parity['status']=='PASS' and parity['legacy_media']==24 and parity['identical_reused_inputs']==44
        original=path.read_bytes();scores=json.loads(original)
        wrong={r['id'] for r in audit if r['legacy_window_differs']}
        replace=sorted(set(scores)&wrong)
        legacy=repair/'legacy_synchformer_scores_24.json'
        assert not legacy.exists();path.rename(legacy)
        write(path,{k:v for k,v in scores.items() if k not in wrong})
        write(marker,dict(status='CANONICAL_CACHE_READY',legacy_scores=str(legacy),
            legacy_scores_sha256=hashlib.sha256(original).hexdigest(),legacy_count=len(scores),
            reused_count=len(scores)-len(replace),required_replacement_ids=replace,
            reason='Frozen integer center: author seconds-to-frame float floor was one frame early; all other successful tensors proved equal',
            cap_failure='EqualifyFromRight defaults to first10seconds before center5s; preselect exact frozen center5s before unmodified transforms'))
    else:write(marker,dict(status='NO_PRIOR_SCORES',reason='Preselect frozen center before original Equalify10s cap'))


def ensure_media(row):
    """Original classifier adapter: RGB256 video plus exact float-PCM audio."""
    root = RUN/'classifier_media'; root.mkdir(exist_ok=True)
    media = root/(row['id']+'.mov')
    if not media.exists():
        tmp = media.with_suffix('.tmp.mov')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',row['video_path'],'-i',row['audio_path'],
            '-map','0:v:0','-map','1:a:0','-vf','fps=25,scale=256:256','-c:v','libx264','-crf','18',
            '-pix_fmt','yuv420p','-c:a','pcm_f32le','-ar','16000','-ac','1',str(tmp)], check=True)
        tmp.replace(media)
    return media


def run_synchformer():
    import torch
    import cloud_mtd_synchformer_20261028 as old
    root, rows = prepare('synchformer')
    torch.set_num_threads(4);torch.manual_seed(20261029)
    torch.backends.cudnn.benchmark=False
    # Execute the already audited 5s adapter with only run/input redirection.
    old.RUN = root
    prepare_canonical_cache(root,rows)
    install_center_preselection(old,'synchformer_run','                item=transforms[seconds](item)',"row['center_start_5s']",root)
    folder = root/'synchformer_media'
    if not folder.exists(): folder.symlink_to(RUN/'classifier_media', target_is_directory=True)
    for row in rows:
        if row['eligible_5s']: ensure_media(row)
    old.synchformer_run()
    scores = read(root/'synchformer_scores.json'); controls=[]
    for key,r in scores.items():
        native=r['outputs']['5']['0']
        for shift in (-6400,6400) if r['arm']=='N' else ():
            other=r['outputs']['5'][str(shift)]
            controls.append(dict(id=key,shift_samples=shift,native=native,shifted=other,
                lower=other['p_zero']<native['p_zero'],
                relative_offset_error=abs(other['predicted_offset']-native['predicted_offset']+shift/16000)))
    assert len(scores)==len(rows)==80
    assert_complete_controls(controls,rows)
    write(root/'known_shift_controls.json',control_summary(controls))
    write(root/'scores.json',{k:dict(id=k,source=r['source'],arm=r['arm'],output=r['outputs']['5']['0'],
        pcm_exact=r['pcm_exact']) for k,r in scores.items()})
    write(root/'status.json',dict(status='COMPLETE',scored=len(scores),
        excluded_short=len(rows)-len(scores),controls=control_summary(controls)['status']))


def run_sparse():
    import numpy as np
    import torch
    import sparsesync_official_cloud_rescore_20261028 as old
    root, rows = prepare('sparsesync'); old.RUN=root; old.media=ensure_media
    prepare_canonical_cache(root,rows)
    install_center_preselection(old,'tensors','    item=transform(item)',"row['center_start_5s']",root)
    torch.set_num_threads(4); torch.manual_seed(20261029)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
    eligible=[r for r in rows if r['eligible_5s']]
    official,cfg,model,transform,grid=old.load()
    if not (root/'official_parity.json').exists():
        sparse_center_parity(old,official,cfg,model,transform,grid,eligible[0],root)
    assert read(root/'official_parity.json')['status']=='PASS'
    scores=read(root/'scores.json') if (root/'scores.json').exists() else {}
    controls=read(root/'known_shift_controls.json')['rows'] if (root/'known_shift_controls.json').exists() else []
    done_controls={(r['id'],r['shift_samples']) for r in controls}
    with torch.inference_mode():
        for row in eligible:
            key=row['id']
            if key not in scores:
                a,v,t=old.tensors(official,transform,row)
                _,z=model(v,a,t)
                scores[key]=dict(id=key,source=row['sample_id'],arm=row['arm'],pcm_exact=True,
                    start_frame=row['center_start_5s'],output=old.output(z,grid))
                write(root/'scores.json',scores)
            if row['arm']=='N':
                native=scores[key]['output']
                for shift in (-6400,6400):
                    if (key,shift) in done_controls:continue
                    a,v,t=old.tensors(official,transform,row,shift)
                    _,z=model(v,a,t);other=old.output(z,grid)
                    controls.append(dict(id=key,shift_samples=shift,native=native,shifted=other,
                        lower=other['p_zero']<native['p_zero'],
                        relative_offset_error=abs(other['predicted_offset']-native['predicted_offset']+shift/16000)))
                    done_controls.add((key,shift))
                write(root/'known_shift_controls.json',control_summary(controls))
            write(root/'status.json',dict(status='RUNNING',scored=len(scores),total=len(eligible),last=key))
            print('SPARSE',len(scores),len(eligible),key,flush=True)
    # Independent NumPy FP64 softmax validates scalar extraction without another forward.
    assert len(scores)==len(eligible)==80
    assert_complete_controls(controls,rows)
    max_error=0.
    for r in scores.values():
        z=np.asarray(r['output']['logits'],dtype=np.float64);p=np.exp(z-z.max());p/=p.sum()
        max_error=max(max_error,float(np.max(abs(p-np.asarray(r['output']['probabilities'])))))
    assert max_error<2e-6
    write(root/'independent_softmax_audit.json',dict(status='PASS',max_probability_error=max_error,media=len(scores)))
    write(root/'status.json',dict(status='COMPLETE',scored=len(scores),excluded_short=len(rows)-len(scores),
        controls=control_summary(controls)['status']))


def sparse_center_parity(old,official,cfg,model,transform,grid,row,root):
    """Run author main on the same exact preselected center5s input buffer."""
    import contextlib
    import io
    from types import SimpleNamespace
    import torch
    dest=old.SOURCE/'logs/sync_models'/old.EXP;dest.mkdir(parents=True,exist_ok=True)
    for name in [f'cfg-{old.EXP}.yaml',f'{old.EXP}.pt']:
        p=dest/name
        if not p.exists():p.symlink_to(old.WEIGHTS/name)
    names=['get_model','get_video_and_audio','prepare_inputs','decode_single_video_prediction','reconstruct_video_from_input','Recorder']
    original={k:getattr(official,k) for k in names};captured={};recorders=[]
    def selected_reader(*args,**kwargs):
        rgb,audio,meta=original['get_video_and_audio'](*args,**kwargs)
        selected=pretrim_official_center(dict(video=rgb,audio=audio,meta=meta,targets={}),row['center_start_5s'])
        return selected['video'],selected['audio'],meta
    def inputs(*args,**kwargs):
        a,v,t=original['prepare_inputs'](*args,**kwargs)
        captured['audio']=a.cpu().clone();captured['video']=v.cpu().clone()
        return a,v,t
    def decode(z,g,item):
        captured['logits']=z.detach().cpu().clone()
        captured['probabilities']=original['decode_single_video_prediction'](z,g,item).cpu().clone()
        return captured['probabilities']
    def recorder(*args,**kwargs):
        r=original['Recorder'](*args,**kwargs);recorders.append(r);return r
    official.get_model=lambda *args:(model,model)
    official.get_video_and_audio=selected_reader
    official.prepare_inputs=inputs;official.decode_single_video_prediction=decode
    official.reconstruct_video_from_input=lambda *args:None;official.Recorder=recorder
    stdout=io.StringIO()
    try:
        with torch.inference_mode(),contextlib.redirect_stdout(stdout):
            official.main(SimpleNamespace(exp_name=old.EXP,vid_path=str(old.media(row)),vfps=25,afps=16000,
                input_size=256,device='cuda',offset_sec=0.,v_start_i_sec=0.))
    finally:
        for recorder in recorders:recorder.eject()
        for k,v in original.items():setattr(official,k,v)
    a,v,t=old.tensors(official,transform,row)
    with torch.inference_mode():_,z=model(v,a,t)
    report=dict(status='PASS',id=row['id'],official_main_executed=True,
        reader_instrumentation='author reader followed by frozen integer center5s selection; main local v_start=0; full batch path uses same selection',
        audio_max_abs_diff=float((captured['audio']-a.cpu()).abs().max()),
        video_max_abs_diff=float((captured['video']-v.cpu()).abs().max()),
        logits_max_abs_diff=float((captured['logits']-z.cpu()).abs().max()),
        probabilities_max_abs_diff=float((captured['probabilities'].flatten()-z.cpu().softmax(-1).flatten()).abs().max()))
    assert report['audio_max_abs_diff']==report['video_max_abs_diff']==0
    assert report['logits_max_abs_diff']<=1e-5 and report['probabilities_max_abs_diff']<=1e-6
    (root/'official_example_stdout.txt').write_text(stdout.getvalue());write(root/'official_parity.json',report)


def run_avs():
    import cv2
    import importlib
    import numpy as np
    import torch
    from scipy.io import wavfile
    import avs_u_zh_pilot_20261027 as old
    root, rows=prepare('avs_u')
    torch.set_num_threads(4);torch.manual_seed(20261029)
    model,task,_=old.load_encoder(Path('/home/wjj/avs-u-20261027'),root,'cuda')
    from python_speech_features import logfbank
    avutils=importlib.import_module('avhubert.utils')
    crop_root=ROOT/'runs/cogenav_sync_audit_20261024_attempt01'
    detector=old.module('ears_avs_detector',crop_root/'autoavsr_detector.py').LandmarksDetector()
    cropper=old.module('ears_avs_crop',crop_root/'autoavsr_video_process.py').VideoProcess(convert_gray=False)
    transform=avutils.Compose([avutils.Normalize(0,255),
        avutils.CenterCrop((int(task.image_crop_size),int(task.image_crop_size))),
        avutils.Normalize(float(task.image_mean),float(task.image_std))])
    cache=root/'embeddings';cache.mkdir(exist_ok=True)
    scores=read(root/'scores.json') if (root/'scores.json').exists() else {}

    def audio_embedding(pcm,frames):
        feats=logfbank(pcm,samplerate=16000).astype(np.float32)
        order=int(task.stack_order_audio)
        if len(feats)%order: feats=np.pad(feats,((0,order-len(feats)%order),(0,0)))
        feats=feats.reshape(-1,feats.shape[1]*order)
        feats=np.pad(feats[:frames],((0,max(0,frames-len(feats))),(0,0)))
        x=torch.from_numpy(feats)
        if task.normalize:x=torch.nn.functional.layer_norm(x,x.shape[1:])
        with torch.inference_mode():
            out=model.extract_finetune({'audio':x.T[None].cuda(),'video':None},mask=False)[0]
        return out[0].cpu().numpy()

    for row in rows:
        key=row['id'];dest=cache/(key+'.npz')
        if key in scores: continue
        sr,pcm=wavfile.read(row['audio_path']);assert sr==16000 and pcm.ndim==1 and pcm.dtype==np.int16
        if dest.exists():
            with np.load(dest) as data:a,v=data['audio'],data['video']
        else:
            cap=cv2.VideoCapture(row['video_path']);frames=[]
            assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<1e-5
            while True:
                ok,f=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            cap.release();marks=detector(frames);detected=sum(m is not None for m in marks)
            assert detected/len(frames)>=.95
            patches=cropper(frames,marks)
            gray=np.stack([cv2.cvtColor(p,cv2.COLOR_RGB2GRAY) for p in patches])
            x=torch.from_numpy(np.ascontiguousarray(transform(gray))).float()[None,None].cuda()
            a=audio_embedding(pcm,len(gray))
            with torch.inference_mode():v=model.extract_finetune({'audio':None,'video':x},mask=False)[0][0].cpu().numpy()
            np.savez_compressed(dest,audio=a,video=v,detected=detected,protocol_sha256=sha(root/'protocol.json'),
                input_audio_sha256=row['audio_sha256'],input_video_sha256=row['video_sha256'])
            del frames,patches,gray,x
        result=dict(id=key,source=row['sample_id'],arm=row['arm'],frames=len(v),
            embedding_sha256=sha(dest),**old.similarities(a,v))
        if row['arm']=='N':
            control_path=cache/(key+'_controls.npz')
            shifted=dict(np.load(control_path)) if control_path.exists() else {}
            for amount in (-6400,6400):
                if str(amount) not in shifted:shifted[str(amount)]=audio_embedding(old.shift_pcm(pcm,amount),len(v))
            np.savez_compressed(control_path,**shifted)
            result['controls']=[dict(shift_samples=int(s),**old.similarities(aa,v)) for s,aa in shifted.items()]
        scores[key]=result;write(root/'scores.json',scores)
        write(root/'status.json',dict(status='RUNNING',scored=len(scores),total=len(rows),last=key))
        print('AVS',len(scores),len(rows),key,flush=True)
    controls=[dict(id=r['id'],shift_samples=c['shift_samples'],native=r['frame_mean'],shifted=c['frame_mean'],
        lower=c['frame_mean']<r['frame_mean']) for r in scores.values() if r['arm']=='N' for c in r['controls']]
    assert len(scores)==80
    assert_complete_controls(controls,rows)
    write(root/'known_shift_controls.json',control_summary(controls))
    write(root/'status.json',dict(status='COMPLETE',scored=len(scores),controls=control_summary(controls)['status']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['avs_u','synchformer','sparsesync'])
    p.add_argument('--run',type=Path,default=RUN);args=p.parse_args();RUN=args.run.resolve()
    {'avs_u':run_avs,'synchformer':run_synchformer,'sparsesync':run_sparse}[args.model]()
