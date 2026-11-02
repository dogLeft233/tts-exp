"""Verify a downloaded evidence archive, extract regular files, export standard media."""
from pathlib import Path, PurePosixPath
import argparse, csv, hashlib, json, shutil, tarfile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True);a=p.parse_args()
    receipt=read(a.receipt);assert sha(a.archive)==receipt['sha256'] and a.archive.stat().st_size==receipt['bytes']
    root=(a.output/'evidence').resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive) as tf:
        for member in tf:
            name=PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            target=(root/str(name)).resolve();assert target.is_relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            stream=tf.extractfile(member);assert stream is not None;data=stream.read()
            if target.exists():assert hashlib.sha256(data).hexdigest()==sha(target)
            else:target.write_bytes(data)
    manifest=read(root/'evidence_manifest.json');assert sha(root/'evidence_manifest.json')==receipt['manifest_sha256']
    for rec in manifest['files']:
        target=root/rec['path'];assert target.stat().st_size==rec['bytes'] and sha(target)==rec['sha256'],rec['path']
    exported=[]
    for row in read(root/'scoring_inputs.json')['rows']:
        folder='natural_raw' if row['arm']=='N' else 'tts_raw'
        video=root/'generation'/row['id']/'generated_aac.mp4'
        target=a.model_root/folder/row['source']/('rms24_20261030_'+row['id']+'.mp4')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(video)
        else:shutil.copyfile(video,target)
        exported.append(dict(id=row['id'],source=row['source'],arm=row['arm'],draw=row['draw'],video_path=str(target.resolve()),sha256=sha(target),
            scoring_video_path=str(root/'videos'/(row['id']+'.mp4')),audio_path=str(root/'audio'/(row['id']+'.wav'))))
    (a.output/'video_index.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in ['summary.json','report.md','paired_scores.csv','paired_scores.json','per_video_scores.csv','per_video_scores.json','independent_audit.json']:
        shutil.copyfile(root/name,a.output/name)
    result=dict(status='PASS',archive_sha256=sha(a.archive),verified_files=len(manifest['files']),exported_videos=len(exported),
        audio_note='Standard media are original generated AAC playback files. Lossless ALAC scoring videos and exact normalized driving WAVs remain in evidence/videos and evidence/audio.')
    (a.output/'local_readback.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
