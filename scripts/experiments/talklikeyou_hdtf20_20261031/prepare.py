"""Private run configuration and SHA/normalization validation before inference."""
from pathlib import Path
import hashlib,json,shutil
B=Path('/home/data2/wangjiajun/av-evaluation')
D=Path('/home/data2/wangjiajun/TalkLikeYouDeploy')
RUN=B/'jobs/talklikeyou_hdtf20_rms27_20261031_attempt01'
DATA=B/'datasets/hdtf20_cloud31_20261031'
GPU='GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def native_frames(samples):
    value=(samples//200-15)*25//80
    if value<=0:raise ValueError('No native complete mel windows')
    return value

def main():
    import numpy as np,soundfile as sf
    assert not RUN.exists(),'Fresh run required'
    source=json.loads((DATA/'bundle_manifest.json').read_text())
    assert len(source['rows'])==80 and len(source['real_rows'])==40
    for entry in source['files']:
        p=DATA/entry['path'];assert sha(p)==entry['sha256'],p
    assets=json.loads((D/'download_receipt.json').read_text());code=json.loads((D/'source_verification.json').read_text())
    for entry in assets['files']+code['files']:assert sha(D/'TalkLikeYou'/entry['path'])==entry['sha256']
    RUN.mkdir();rows=[]
    for r in source['rows']:
        row=dict(r)
        for key in ['audio_path','canonical_audio_path','face_path']:
            row[key]=str(DATA/row[key])
        pcm,sr=sf.read(row['audio_path'],dtype='int16');x=pcm.astype(float)/32768
        assert sr==16000 and pcm.ndim==1 and len(pcm)==row['samples']
        assert abs(20*np.log10(np.sqrt(np.mean(x*x)))+27)<.002 and abs(x).max()<=10**(-1/20)
        assert native_frames(len(pcm))==row['expected_frames']
        rows.append(row)
    real_rows=[]
    for r in source['real_rows']:
        row=dict(r)
        for key in ['audio_path','video_path']:row[key]=str(DATA/row[key])
        real_rows.append(row)
    protocol=source['protocol']
    protocol.update(run_id=RUN.name,status='FROZEN_BEFORE_GENERATION',
      generation=dict(model='TalkLikeYou',source_commit='d1088c49071148853c76d8a8f0c1fb97d895758b',weights_revision='652b744bbf86510c370ff6f302d046bdd63632e1',
        face_policy='each frozen HDTF window start image, same image for N and T',fps=25,seed=42,person_id=192,stage0=True,neutral=True,motion_scale=1.,guidance_scale=1.3,sampling_steps=1,pose_mode='audio',pose_sampling_steps=1,overlap=5,smooth=True,crop_scale=2.3,face_index=0,
        frame_policy='official complete native mel windows; no padding, time shift, stretching or score-based offset correction'),
      resources=dict(gpu_uuid=GPU,torch_memory_fraction=.60,cpu_threads=4,affinity=[0,1,2,3],execution='nohup; sequential GPU phases'),
      duplicate_audio_groups=[],unique_audio_files=len({r['audio_sha256'] for r in rows}))
    write(RUN/'protocol.json',protocol);write(RUN/'inputs.json',dict(rows=rows,real_rows=real_rows))
    eligible=[r for r in rows if min(r['samples']//640,r['expected_frames'])>=125]
    write(RUN/'predeclared_support.json',dict(status='FROZEN_BEFORE_GENERATION',classifier_eligible_ids=[r['id'] for r in eligible],
        classifier_complete_clips=[s for s in sorted({r['source'] for r in rows}) if all(r in eligible for r in rows if r['source']==s)],
        no_score_selection=True))
    shutil.copyfile(DATA/'cloud_complete.json',RUN/'cloud_complete.json')
    write(RUN/'preparation.json',dict(status='PASS',rows=80,clips=40,speakers=20,real_syncnet_controls=40,
        assets_sha_verified=len(assets['files']),source_sha_verified=len(code['files']),bundle_manifest_sha256=sha(DATA/'bundle_manifest.json')))
    print((RUN/'preparation.json').read_text())
if __name__=='__main__':main()
