"""Archive completed SyncNet evidence without redundant frame caches."""
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from prepare import RUN, sha, write

def main():
    assert json.loads((RUN/'score_audit.json').read_text())['status']=='PASS'
    assert json.loads((RUN/'generation_complete.json').read_text())['completed']==80
    pins=json.loads((RUN/'scorer_pins.json').read_text())
    target=RUN/'scorer_source_snapshot'
    target.mkdir(exist_ok=True)
    for path,digest in pins['files'].items():
        # Store source and weight hashes; copy source text, reuse official weights by hash.
        p=Path(path)
        if p.suffix in ['.py','.md','.sh']:
            out=target/p.name
            shutil.copyfile(p,out)
            assert sha(out)==digest
    files=[]
    for p in sorted(RUN.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(RUN)
        if rel.parts[:3] == ('syncnet','work','pyframes'):continue
        if 'official_output' in rel.parts:continue
        if p.name in ['file_manifest.json','delivery_receipt.json']:continue
        files.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(RUN/'file_manifest.json',dict(status='PASS',files=files,
        exclusions=['rebuildable syncnet/work/pyframes JPEGs','duplicate generation official_output/*.mp4'],
        scientific_scope='SyncNet only; other8 scorers not started; HDTF remains paused'))
    archive=RUN.with_name(RUN.name+'_evidence.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
        for f in files:tar.add(RUN/f['path'],arcname=f['path'],recursive=False)
        tar.add(RUN/'file_manifest.json',arcname='file_manifest.json',recursive=False)
    write(RUN/'delivery_receipt.json',dict(status='PASS',archive_path=str(archive),
        archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),files=len(files),
        scope='80 logical generation and SyncNet only, 41 unique; stopped before other scorers',
        ended=time.time()))
    print((RUN/'delivery_receipt.json').read_text())

if __name__=='__main__':
    main()
