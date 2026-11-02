"""Verify downloaded bytes, independently reconstruct all scores and export media."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import shutil
import sys
import tarfile
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while block:=stream.read(8*1024**2):h.update(block)
    return h.hexdigest()

def main():
    root=Path(sys.argv[1]).resolve()
    receipt=json.loads((root/'delivery_receipt.json').read_text())
    archive=root/'evidence.tar.gz'
    assert archive.stat().st_size==receipt['archive_bytes'] and sha(archive)==receipt['archive_sha256']
    evidence=root/'evidence'
    assert not evidence.exists()
    evidence.mkdir()
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        tar.extractall(evidence,filter='data')
    manifest=json.loads((evidence/'file_manifest.json').read_text())['files']
    for f in manifest:
        p=evidence/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
    inputs=json.loads((evidence/'scoring_inputs.json').read_text())
    scores={}
    for row in inputs['unique_rows']:
        z=np.load(evidence/'syncnet'/(row['id']+'_distances.npy')).astype(np.float64)
        curve=z.mean(axis=0)
        value=np.median(curve)-curve.min()
        saved=json.loads((evidence/'syncnet'/(row['id']+'.json')).read_text())
        assert abs(value-saved['C'])<2e-5 and 15-int(curve.argmin())==saved['offset']
        scores[row['id']]=saved['C']
    pairs=[]
    for source in sorted({r['source'] for r in inputs['rows']}):
        rows=[r for r in inputs['rows'] if r['source']==source]
        n=[r for r in rows if r['arm']=='N'];t=[r for r in rows if r['arm']!='N']
        assert len(n)==1 and len(t)==3
        def score(r):return scores[inputs['aliases'].get(r['id'],r['id'])]
        N=score(n[0]);T=sum(map(score,t))/3
        pairs.append(dict(source=source,strict_reference=n[0]['strict_reference'],N=N,T=T,delta=T-N))
    summary=json.loads((evidence/'summary.json').read_text())
    for group,ps in [('all20',pairs),('same_reference12',[p for p in pairs if p['strict_reference']]),
                     ('extended_reference8',[p for p in pairs if not p['strict_reference']])]:
        s=summary[group]
        assert len(ps)==s['sources']
        for key in ['N','T','delta']:
            assert abs(sum(p[key] for p in ps)/len(ps)-s[key])<1e-12
        d=np.array([p['delta'] for p in ps]);rng=np.random.default_rng(20261030)
        ci=np.quantile(d[rng.integers(0,len(ps),(10000,len(ps)))].mean(1),[.025,.975])
        assert np.allclose(ci,s['ci95'],atol=1e-12,rtol=0)
    videos=[]
    repo=Path(__file__).resolve().parents[3]
    for r in inputs['rows']:
        filename=Path(r['audio_path']).name
        wav=evidence/'audio'/filename
        assert sha(wav)==r['audio_sha256']
        clip=evidence/'videos'/(r['id']+'.mp4')
        assert sha(clip)==r['video_sha256']
        record=json.loads((evidence/'generation'/r['id']/'receipt.json').read_text())
        assert record['status']=='PASS' and record['decoded_pcm_exact'] and record['pts_zero_regular']
        raw=evidence/'generation'/r['id']/'generated_aac.mp4'
        assert sha(raw)==record['raw_video_sha256']
        dest=repo/'results/pctalk'/('natural_raw' if r['arm']=='N' else 'tts_raw')/r['source']/(root.name+'_'+r['id']+'.mp4')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(raw,dest)
        videos.append(dict(id=r['id'],source=r['source'],arm=r['arm'],path=str(dest),C=scores[inputs['aliases'].get(r['id'],r['id'])]))
    for name in ['summary.json','report.md','per_video_scores.csv','paired_scores.csv','score_audit.json','protocol.json']:
        shutil.copyfile(evidence/name,root/name)
    (root/'video_index.json').write_text(json.dumps(videos,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>PC-Talk 中文配对视频</title><style>body{max-width:1100px;margin:auto;font:16px sans-serif}section{margin:32px 0}video{width:47%;margin-right:2%;background:#111}</style><h1>PC-Talk 中文20来源</h1><p>按来源的绝对 TTS−自然 Sync-C 差排序。左自然，右TTS C1；其余draw附链接。分数使用完整原始PCM；此处为模型原生AAC播放片。</p>'
    for p in sorted(pairs,key=lambda x:-abs(x['delta'])):
        vs=[v for v in videos if v['source']==p['source']]
        n=next(v for v in vs if v['arm']=='N');ts=[v for v in vs if v['arm']!='N']
        page+=f'<section><h2>{html.escape(p["source"])}：N {p["N"]:.3f} / T均 {p["T"]:.3f} / Δ {p["delta"]:+.3f}</h2>'
        for v in [n,ts[0]]:
            link=os.path.relpath(v['path'],root)
            assert (root/link).is_file()
            page+=f'<video src="{html.escape(link,quote=True)}" controls preload="none"></video>'
        page+='<p>'+' · '.join(f'<a href="{html.escape(os.path.relpath(v["path"],root),quote=True)}">{html.escape(v["id"])}（C {v["C"]:.3f}）</a>' for v in ts)+'</p></section>'
    (root/'index.html').write_text(page)
    audit=dict(status='PASS',sha_verified_files=len(manifest),raw_score_matrices=len(scores),logical_media=len(videos),
               independently_recomputed_source_groups=list(summary),bootstrap_recomputed=True)
    (root/'local_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))

if __name__=='__main__':
    main()
