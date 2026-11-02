"""Local persistent backup of a running SSH experiment; never starts training."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

REMOTE='/dev/shm/synctalk-may-training-pilot-20261017'
HOST='wjj@192.168.50.2'


def ssh_python(code):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST,
                           'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    return json.loads(result.stdout)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def tick(root,known):
    code=f'''import pathlib,json
r=pathlib.Path({REMOTE!r})
rows=[]
for p in r.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(r)
 if any(x in ['__pycache__','matplotlib-cache'] for x in rel.parts):continue
 if p.suffix in ['.partial','.lock']:continue
 if p.suffix=='.mkv' and not (p.parent/'render_receipt.json').exists():continue
 s=p.stat();rows.append(dict(path=str(rel),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
print(json.dumps(dict(files=rows,complete=(r/'experiment_complete.json').exists(),failed=(r/'supervisor_failure.json').exists())))'''
    listing=ssh_python(code)
    for row in listing['files']:
        relative=row['path']
        assert re.fullmatch(r'[a-zA-Z0-9_./-]+',relative) and '..' not in Path(relative).parts
        if known.get(relative)==[row['bytes'],row['mtime_ns']]:continue
        target=root/'remote'/relative
        assert target.resolve().is_relative_to(root.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        remote=REMOTE+'/'+relative
        digest=ssh_python(f"import hashlib,json;print(json.dumps(hashlib.sha256(open({remote!r},'rb').read()).hexdigest()))")
        temporary=target.with_name(target.name+'.download')
        copy=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                             HOST+':'+remote,str(temporary)],capture_output=True,text=True,timeout=300)
        if copy.returncode:raise RuntimeError(copy.stderr[-1000:])
        actual=sha(temporary)
        if actual!=digest:
            # Append-only logs can change during transfer; retry on the next pass.
            continue
        temporary.replace(target)
        known[relative]=[row['bytes'],row['mtime_ns']]
        with (root/'backup_receipts.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'file':relative,'sha256':actual,'bytes':target.stat().st_size,
                                  'verified_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})+'\n')
    (root/'backup_index.json').write_text(json.dumps(known,indent=2)+'\n')
    (root/'backup_status.json').write_text(json.dumps({'files':len(known),'last_check_utc':
                time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'worker_complete':listing['complete'],'worker_failed':listing['failed']},indent=2)+'\n')
    return listing['complete'] or listing['failed']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();a.root.mkdir(exist_ok=True)
    known=json.loads((a.root/'backup_index.json').read_text()) if (a.root/'backup_index.json').exists() else {}
    while True:
        try:
            stop=tick(a.root,known)
            if stop or a.once:return
        except Exception as error:
            with (a.root/'backup_errors.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'error':repr(error),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})+'\n')
            if a.once:raise
        time.sleep(60)


if __name__=='__main__':main()
