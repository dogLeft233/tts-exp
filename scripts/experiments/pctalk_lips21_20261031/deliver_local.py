"""Verify downloaded evidence and reconstruct nine endpoints from native artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile
import numpy as np

METRICS=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']
FOLDERS=['syncnet','cogenav','peavs','syncreward','avs_u','mtd','vocalist','synchformer','sparsesync']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while block:=stream.read(8*1024**2):h.update(block)
    return h.hexdigest()

def collect(root,metric,folder):
    values={}
    if metric=='SyncNet':
        for x in read(root/folder/'scores.json'):
            curve=np.load(root/folder/(x['id']+'_distances.npy')).astype(float).mean(0)
            assert abs(float(np.median(curve)-curve.min())-x['C'])<2e-5
            values[x['id']]=x['C']
    elif metric=='CoGenAV':
        for x in read(root/folder/'scores.json'):
            if x.get('status','PASS')!='PASS':continue
            with np.load(root/folder/'embeddings'/(x['id']+'.npz')) as z:
                a=z['audio'].astype(float);v=z['video'].astype(float)
            score=np.maximum(0,(a*v).sum(-1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1))).mean()
            assert abs(float(score)-x['eq1'])<1e-6
            values[x['id']]=x['eq1']
    elif metric=='PEAVS':
        values={Path(x['filename']).stem:x['score'] for x in read(root/folder/'scores.json')['scores']}
        assert len(values)==42
    elif metric=='SyncReward':
        values={Path(x['video_path']).stem:x['reward'] for x in map(json.loads,(root/folder/'scores.jsonl').read_text().splitlines())}
    else:
        for x in read(root/folder/'batch_scores.json'):
            if x['status']!='PASS':continue
            base=root/folder/x['id'];results=read(base/'deployment_validation.json')['results']
            if metric=='AVS_u':
                with np.load(base/'raw_embeddings.npz') as z:
                    a=z['audio_0'].astype(float);v=z['video'].astype(float)
                score=((a*v).sum(-1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1))).mean()
                expected=results['0']['frame_mean'];assert abs(float(score)-expected)<1e-6
            elif metric in ['MTDVocaLiST','VocaLiST']:
                for shift,saved in results.items():
                    distances=np.load(base/('distances_'+f'{int(shift):+d}'+'.npy'))
                    k=np.count_nonzero(abs(distances.argmax(1)-15)<=1)/len(distances)
                    assert abs(k-saved['author_metrics']['K5'])<1e-7
                expected=results['0']['author_metrics']['K5']
            else:
                for saved in results.values():
                    z=np.array(saved['logits'],dtype=float);p=np.exp(z-z.max());p/=p.sum()
                    assert np.allclose(p,saved['probabilities'],rtol=1e-4,atol=2e-7)
                    assert abs(p[10]-saved['p_zero'])<2e-6
                expected=results['0']['p_zero']
            values[x['id']]=expected
    return values

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;receipt=read(out/'delivery_receipt.json');archive=out/'evidence.tar.gz'
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    root=out/'evidence';assert not root.exists();root.mkdir()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        tar.extractall(root,filter='data')
    files=read(root/'evidence_manifest.json')['files']
    for x in files:
        p=root/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
    inputs=read(root/'scoring_inputs.json');rows=inputs['rows'];aliases=inputs['aliases']
    pervideo={x['id']:x for x in read(root/'per_video_scores.json')}
    pairs=read(root/'paired_scores.json');summary=read(root/'summary.json');counts={}
    for metric,folder in zip(METRICS,FOLDERS):
        values=collect(root,metric,folder)
        if metric!='PEAVS':
            for alias,original in aliases.items():
                if original in values:values[alias]=values[original]
        assert all(np.isfinite(v) for v in values.values())
        for row in rows:
            saved=pervideo[row['id']][metric]
            if row['id'] not in values:assert saved is None
            else:assert abs(values[row['id']]-saved)<1e-12
        regrouped=[]
        for source in sorted({r['source'] for r in rows}):
            group=[r for r in rows if r['source']==source]
            if not all(r['id'] in values for r in group):continue
            n=[values[r['id']] for r in group if r['arm']=='N'];t=[values[r['id']] for r in group if r['arm']!='N']
            assert len(n)==1 and len(t)==1
            regrouped.append(dict(source=source,N=n[0],T=t[0]))
        assert len(regrouped)==len(pairs[metric])
        for rebuilt,saved in zip(regrouped,pairs[metric]):
            assert rebuilt['source']==saved['source']
            assert abs(rebuilt['N']-saved['N'])<1e-12 and abs(rebuilt['T']-saved['T'])<1e-12
        counts[metric]=dict(media=len(values),sources=len(regrouped))
    import soundfile as sf
    for row in rows:
        audio=root/'audio'/Path(row['audio_path']).name
        original=root/'original_audio'/(row['id']+'.wav')
        assert sha(audio)==row['audio_sha256']
        assert sha(original)==row['original_audio_sha256']
        assert sha(root/'images'/Path(row['image_path']).name)==row['image_sha256']
        before,sr=sf.read(original,dtype='int16');after,normalized_sr=sf.read(audio,dtype='int16')
        assert sr==normalized_sr==16000
        expected=np.rint(before.astype(np.float64)*row['scalar_gain']).astype('<i2')
        assert np.array_equal(after,expected)
        assert sha(root/'videos'/(row['id']+'.mp4'))==row['video_sha256']
    for name in ['report.md','summary.json','paired_scores.json','paired_scores.csv','per_video_scores.json','independent_audit.json']:
        shutil.copyfile(root/name,out/name)
    audit=dict(status='PASS',sha_verified_content_files=len(files),raw_native_endpoints_regrouped=counts,
        complete_media_bindings=42,notes='PEAVS/SyncReward native outputs and identities read directly; not independent network implementations. Softmax exact FP32 replay performed on server, independent FP64 check here.')
    (out/'local_raw_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
