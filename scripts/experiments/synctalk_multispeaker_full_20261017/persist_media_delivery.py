"""Persist a full verified result archive plus local scores, arrays and watch previews."""
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
from execution_journal import atomic_json

def retain_local(path):
    path=Path(path)
    return path.suffix in ['.json','.py','.png','.mp4','.wav'] or (path.suffix=='.npy' and not path.name.startswith('raw_head_'))

def run(identity,seed):
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'] and seed in [42,2026]
    source=f'{REMOTE}/media_manifests/{identity}/seed{seed}'
    raw=subprocess.check_output(SSH+[f'cat {source}/manifest.json']);manifest=json.loads(raw)
    assert manifest['passed'] and manifest['identity']==identity and manifest['seed']==seed and manifest['formal_videos']==9 and manifest['grid_cells']==27
    out=ROOT/'result_delivery'/identity/f'seed{seed}'
    if (out/'verification.json').exists():
        proof=json.loads((out/'verification.json').read_text());assert proof['passed'] and proof['manifest_sha256']==hashlib.sha256(raw).hexdigest()
        for row in proof['local_files']:assert sha(out/'local'/row['path'])==row['sha256']
        if Path(proof['archive']).exists():assert sha(Path(proof['archive']))==proof['archive_sha256']
        else:
            retirement=json.loads((out/'archive_cache_retirement.json').read_text());assert retirement['passed'] and retirement['sha256']==proof['archive_sha256']
        return proof
    out.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(ROOT).free>manifest['raw_bytes']+128*1024**2,'reserve complete durable archive before transport'
    (out/'manifest.json').write_bytes(raw);files={row['path']:row for row in manifest['files']}
    partial=out/'results.tar.gz.partial'
    with partial.open('xb') as f:
        subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {REMOTE} --null -T {source}/files.nul'],stdout=f,check=True)
        f.flush();os.fsync(f.fileno())
    seen=set();thin=[]
    with tarfile.open(partial,'r|gz') as stream:
        for member in stream:
            assert member.isfile() and member.name in files and member.name not in seen
            row=files[member.name];assert row['bytes']==member.size
            target=out/'local'/member.name;assert target.resolve().is_relative_to((out/'local').resolve())
            keep=retain_local(member.name);h=hashlib.sha256();source_file=stream.extractfile(member)
            if keep:target.parent.mkdir(parents=True,exist_ok=True);output=target.open('xb')
            try:
                for block in iter(lambda:source_file.read(4*1024**2),b''):
                    h.update(block)
                    if keep:output.write(block)
            finally:
                if keep:output.close()
            assert h.hexdigest()==row['sha256'];seen.add(member.name)
            if keep:assert sha(target)==row['sha256'];thin.append(row)
    assert seen==files.keys()
    archive=out/'results.tar.gz';partial.rename(archive)
    proof=dict(identity=identity,seed=seed,passed=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),all_archived_member_bytes_verified=True,files=len(seen),local_files=thin,
        local_semantics='all scores/matrices/curves/source/portrait stills/watch-only preview; formal FFV1 and checkpoints are in verified archive',
        source_sha256=sha(Path(__file__)))
    atomic_json(out/'verification.json',proof);print(json.dumps({k:v for k,v in proof.items() if k!='local_files'}),flush=True);return proof

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
