"""Freeze diagnostic evidence after all producer processes have exited."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import tarfile
import numpy as np
from control import B,D,OLD,PARENT,RUN,read,write,sha


def main():
    result=read(RUN/'paired_comparison.json');assert result['status']=='PASS'
    offset_control=read(RUN/'offset_control_summary.json')
    plan=read(RUN/'plan.json')
    assert sha(Path(__file__).with_name('control.py'))==plan['source_sha256']
    for path,digest in plan['parent_pins'].items():assert sha(path)==digest
    # Independently recompute source rows from per-video endpoints, in NumPy.
    scores=read(RUN/'scores.json');by={(s['condition'],s['id']):s for s in scores}
    for condition,summary in result['conditions'].items():
        ns=[];ts=[]
        for sid in sorted(plan['common_frames']):
            n=next(r for r in plan['rows'] if r['source']==sid and r['arm']=='N')
            t=next(r for r in plan['rows'] if r['source']==sid and r['arm']=='T')
            ns.append(by[condition,n['id']]['C']);ts.append(by[condition,t['id']]['C'])
        np.testing.assert_allclose([summary['N'],summary['T'],summary['T_minus_N']],
                                   [np.mean(ns),np.mean(ts),np.mean(np.array(ts)-ns)],atol=1e-12,rtol=0)
    # Explicit endpoint controls: score-only gain must use the exact baseline visual.
    for row in plan['rows']:
        assert by['baseline',row['id']]['video_sha256']==by['scorer_gain_only',row['id']]['video_sha256']
    snapshot=RUN/'snapshot';snapshot.mkdir()
    origins=[]
    paths=[Path(__file__),Path(__file__).with_name('control.py'),Path(__file__).with_name('run.sh'),Path(__file__).with_name('shift_control.py'),Path(__file__).with_name('post.sh'),
           OLD/'generate.py',OLD/'attention.py',D/'repo/infer_flash.py',D/'repo/run_flash.sh',
           D/'download_manifest.json',D/'download_receipt.json',D/'environment.freeze.txt',
           D/'models/chinese-wav2vec2-base/preprocessor_config.json',B/'gpu_guard.py']
    paths += [B/'scorers/syncnet'/name for name in ['run_pipeline.py','run_syncnet.py','SyncNetInstance.py','SyncNetModel.py']]
    for i,p in enumerate(paths):
        assert p.is_file(),p
        dest=snapshot/(f'{i:02d}_'+p.name);shutil.copyfile(p,dest)
        origins.append(dict(path=str(p),archive_path=str(dest.relative_to(RUN)),sha256=sha(dest)))
    for env,prefix in [('generator',D/'env'),('syncnet',B/'envs/eval-base')]:
        (snapshot/(env+'_pip_freeze.txt')).write_bytes(subprocess.check_output([str(prefix/'bin/python'),'-m','pip','freeze']))
    weight=B/'scorers/syncnet/data/syncnet_v2.model'
    write(snapshot/'bindings.json',dict(sources=origins,syncnet_weight=dict(path=str(weight),sha256=sha(weight))))
    for r in plan['rows']:
        for name,source in [('audio',r['audio_path']),('canonical',r['canonical_audio_path']),('baseline_video',r['parent_video'])]:
            dest=RUN/'inputs'/name/Path(source).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    face=RUN/'inputs/3.png';shutil.copyfile(plan['face_path'],face);assert sha(face)==plan['face_sha256']
    # Static comparison plot is a scientific artifact, not an interactive dashboard.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(result['conditions']);fig,axes=plt.subplots(1,2,figsize=(11,4))
    names={'baseline':'Baseline','lufs_only':'LUFS only','cap_only':'81-frame cap only','seed_only':'Seed43 only','official':'Official settings','scorer_gain_only':'Scorer gain only'}
    positions=np.arange(len(labels));width=.35
    axes[0].bar(positions-width/2,[result['conditions'][k]['N'] for k in labels],width,label='Natural')
    axes[0].bar(positions+width/2,[result['conditions'][k]['T'] for k in labels],width,label='TTS C1')
    axes[0].set_ylabel('Sync-C (common prefix)');axes[0].legend()
    for condition in labels:
        pair=[r for r in result['pairs'] if r['condition']==condition]
        j=labels.index(condition)
        axes[1].scatter([j]*len(pair),[(r['N_change']+r['T_change'])/2 for r in pair],alpha=.6)
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set_ylabel('Mean N/T change vs baseline per source')
    for ax in axes:ax.set_xticks(positions,[names[k] for k in labels],rotation=32,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('EchoMimicV3-Flash: 6-source diagnostic, fixed common support')
    fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
    # Side-by-side first pair baseline / LUFS / official, first-mid-last frames.
    import cv2
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(6*180,3*205),'white');draw=ImageDraw.Draw(canvas)
    for column,(r,arm) in enumerate((r,a) for r in plan['rows'][:2] for a in ['baseline','lufs_only','official']):
        p=Path(r['parent_video']) if arm=='baseline' else RUN/'generation'/(arm+'__'+r['id'])/'video.mp4'
        cap=cv2.VideoCapture(str(p));f=plan['common_frames'][r['source']]
        for row,index in enumerate([0,f//2,f-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok
            image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((180,180))
            canvas.paste(image,(column*180,row*205+25));draw.text((column*180+2,row*205+4),r['arm']+' '+arm+' f'+str(index),fill='black')
        cap.release()
    canvas.save(RUN/'contact_sheet.png')
    body=['# EchoMimicV3-Flash official-default diagnostic','','Exploratory 6-source diagnostic (3 same-reference / 3 extended), one existing C1 TTS per source. Same static512 image. Not the full Chinese20 three-TTS confirmatory experiment.','',
          '| Condition | Natural Sync-C | TTS Sync-C | TTS−Natural | Natural change | TTS change |',
          '|---|---:|---:|---:|---:|---:|']
    for k,v in result['conditions'].items():body.append(f"| {k} | {v['N']:.3f} | {v['T']:.3f} | {v['T_minus_N']:+.3f} | {v['N_change']:+.3f} | {v['T_change']:+.3f} |")
    body += ['', 'All cells use the same first min(81, natural native frames, TTS native frames) per source and identical RMS24 PCM for generation contrasts. Scorer-gain-only changes baseline audio to LUFS23 without changing its visual stream. No padding, shifting, or TTS resynthesis.', '',
             'Official numeric settings use run_flash.sh at pinned revision7e89489ca51c0d008fc1963ec6c03fc5bd0b9397:8steps,CFG6,audioCFG3,shift5,TeaCache0.1/skip5,seed43,81frames,768 size cap; this512 image remains512. LUFS is applied to the whole waveform before truncation. CPU offload, BF16 loader repair and mask-preserving SDPA remain engineering adaptations; this is not a literal stock-entry benchmark.', '',
             'Both baseline replays exactly match decoded parent video frames and Sync-C. All generated video/audio media checks passed. FP32 raw matrices have independent FP64 reductions within1e-5. Frozen parent pins are unchanged. Six-source descriptive bootstrap intervals are in paired_comparison.json; no confirmatory p-values or perceived quality claims.']
    (RUN/'report.md').write_text('\n'.join(body)+'\n')
    write(RUN/'independent_audit.json',dict(status='PASS',source_summary_recomputed=True,
        score_only_visual_identical=True,parent_pins_unchanged=True,score_count=len(scores),
        offset_control_status=offset_control['status'],
        max_matrix_reduction_error=max(abs(r['C']-r['independent_C']) for r in scores)))
    records=[];rebuildable=[]
    for p in sorted(RUN.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(RUN)
        if any(part in ['pyframes','pytmp','pyavi'] for part in rel.parts):continue
        if rel.parts[0] in ('syncnet','offset_controls') and p.name in ('visual.mp4','input.mkv'):
            rebuildable.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)));continue
        records.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(RUN/'evidence_manifest.json',dict(status='PASS',files=records,rebuildable_media_not_archived=rebuildable,
        exclusions='Rebuildable SyncNet pyframes/pytmp/pyavi and exact lossless scoring prefix containers only; all original50 generated videos, 12baseline videos, PCM prefixes, pixel hashes, raw matrices and repair sources retained. Weights pinned, not copied.'))
    archive=RUN.parent/(RUN.name+'_evidence.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for rec in records:tar.add(RUN/rec['path'],arcname=rec['path'],recursive=False)
        tar.add(RUN/'evidence_manifest.json',arcname='evidence_manifest.json',recursive=False)
    receipt=dict(status='PASS',archive_path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                 manifest_sha256=sha(RUN/'evidence_manifest.json'),files=len(records)+1)
    write(RUN/'delivery_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
