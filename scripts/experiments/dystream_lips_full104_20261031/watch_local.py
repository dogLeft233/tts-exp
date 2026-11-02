"""Durable local delivery after the detached server run; no chat notifications or retries of science."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shlex,shutil,subprocess,sys,time

ROOT=Path('/Users/wangjiajun/tts-exp')
OUT=ROOT/'results/dystream/lips_full104_rms24_nine_20261031_attempt01'
REMOTE='/home/data2/wangjiajun/av-evaluation/jobs/dystream_lips_full104_rms24_nine_20261031_attempt01'
REMOTE_SOURCE='/home/data2/wangjiajun/av-evaluation/experiments/dystream_lips_full104_20261031'
HELPER=Path('/tmp/tts192_remote.py')
HELPER_SHA='5486ca24e4e4f2e69bff7a3f0a46ef0bf498e866c0b518b1c8d917faed15951f'
S=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while x:=f.read(8*1024**2):h.update(x)
    return h.hexdigest()

def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def helper(*args):
    assert digest(HELPER)==HELPER_SHA,'SSH helper changed; inspect before using'
    return subprocess.run([sys.executable,str(HELPER),*map(str,args)],capture_output=True,text=True)

def status():
    program=f'''from pathlib import Path
import json
r=Path({REMOTE!r})
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {{}}
x=read(r/'status.json')
state='FAILED' if (r/'orchestrator_failed').exists() or (r/'queue_blocked.json').exists() else 'COMPLETE' if (r/'orchestrator_complete').exists() else 'RUNNING'
print(json.dumps(dict(state=state,stage=x.get('stage','generation'),native_models=x.get('models'),generation_available=len(list((r/'generation').glob('*/receipt.json'))),generation_planned=209,reused_generation=0,syncnet_available=len(list((r/'syncnet').glob('*_distances.npy'))),updated=__import__('time').time())))'''
    response=helper('exec',"/home/data2/wangjiajun/av-evaluation/envs/eval-base/bin/python -c "+shlex.quote(program))
    if response.returncode:return dict(state='NETWORK_RETRY',returncode=response.returncode)
    for line in reversed(response.stdout.splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    raise RuntimeError('No valid server status JSON')

def fetch_small(remote,local):
    for _ in range(6):
        r=helper('get',remote,local)
        if r.returncode==0:return
        time.sleep(10)
    raise RuntimeError('Small delivery transfer failed; retry watcher later')

def archive_resume(receipt):
    target=OUT/'evidence.tar.gz';part=OUT/'evidence.tar.gz.part'
    expected=dict(bytes=receipt['bytes'],sha256=receipt['sha256'],remote=receipt['archive_path'])
    meta=OUT/'download_expected.json'
    if meta.exists():assert json.loads(meta.read_text())==expected,'Archive identity changed'
    else:meta.write_text(json.dumps(expected,indent=2))
    if target.exists():
        assert target.stat().st_size==expected['bytes'] and digest(target)==expected['sha256'];return
    source=HELPER.read_text();assert digest(HELPER)==HELPER_SHA
    prefix=source.split("if sys.argv[1]=='getmany':",1)[0]
    assert prefix!=source
    while True:
        ns={};client=None
        try:
            # Reuse the already-audited host pin and runtime credential reader.
            # Its password variable is deleted by the prefix and never persisted.
            exec(compile(prefix,str(HELPER),'exec'),ns);client=ns['c']
            offset=part.stat().st_size if part.exists() else 0
            assert offset<=expected['bytes']
            with client.open_sftp() as sf:
                assert sf.stat(expected['remote']).st_size==expected['bytes']
                with sf.open(expected['remote'],'rb') as remote,part.open('ab') as local:
                    remote.seek(offset)
                    while offset<expected['bytes']:
                        data=remote.read(min(1024**2,expected['bytes']-offset))
                        if not data:raise OSError('Unexpected short archive read')
                        local.write(data);offset+=len(data)
            break
        except (OSError,EOFError) as error:
            write('local_delivery_status.json',dict(state='DOWNLOAD_RETRY',error_class=type(error).__name__,partial_bytes=part.stat().st_size if part.exists() else 0));time.sleep(10)
        except Exception as error:
            if type(error).__module__.startswith('paramiko'):
                write('local_delivery_status.json',dict(state='DOWNLOAD_RETRY',error_class=type(error).__name__));time.sleep(10)
            else:raise
        finally:
            if client is not None:client.close()
    assert part.stat().st_size==expected['bytes'] and digest(part)==expected['sha256']
    part.replace(target)

def command(script,*args):
    result=subprocess.run([sys.executable,str(S/script),*map(str,args)],capture_output=True,text=True)
    (OUT/(script+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,'Local check failed: '+script

def deliver():
    terminal="/home/data2/wangjiajun/av-evaluation/envs/eval-base/bin/python "+shlex.quote(REMOTE_SOURCE+'/terminal_check.py')
    for _ in range(3):
        result=helper('exec',terminal)
        if result.returncode==0:break
        time.sleep(10)
    assert result.returncode==0,'Terminal process checks not passed'
    for name in ['delivery_receipt.json','terminal_receipt.json']:fetch_small(REMOTE+'/'+name,OUT/name)
    archive_resume(json.loads((OUT/'delivery_receipt.json').read_text()))
    command('deliver_local.py','--output',OUT)
    scratch=OUT/'statistics_check_inputs';scratch.mkdir(exist_ok=True)
    for name in ['summary.json','paired_scores.json','scoring_inputs.json']:shutil.copyfile(OUT/'evidence'/name,scratch/name)
    command('verify_statistics.py','--run',scratch)
    shutil.copyfile(scratch/'independent_statistics_audit.json',OUT/'local_statistics_audit.json')
    command('present_local.py','--output',OUT)
    import html.parser
    class Links(html.parser.HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
    links=Links();links.feed((OUT/'index.html').read_text())
    assert all((OUT/url).resolve().exists() for url in links.links)
    files=['summary.json','paired_scores.json','speaker_scores.json','paired_scores.csv','report.md','index.html','speaker_differences.png','speaker_differences.pdf','local_raw_audit.json','local_statistics_audit.json','terminal_receipt.json']
    local_source=OUT/'local_source';local_source.mkdir(exist_ok=True)
    sources={}
    for p in S.iterdir():
        if p.is_file() and p.suffix in ['.py','.sh']:
            shutil.copyfile(p,local_source/p.name);sources[p.name]=digest(p)
    write('local_source_manifest.json',dict(files=sources))
    write('final_delivery_receipt.json',dict(status='PASS',paired_sources=104,speakers=21,media=209,native_scorers=9,
        archive_sha256=json.loads((OUT/'delivery_receipt.json').read_text())['sha256'],artifact_sha256={p:digest(OUT/p) for p in files},
        html_links_checked=len(links.links),server_processes_exited=True,
        visual_qa='Static plot and generated-contact-sheet human inspection not performed by the headless delivery process; pending agent review.',
        scientific_boundary='Speaker-clustered evaluator evidence, not104 independent participants, perceptual truth or complete leakage exclusion.'))
    write('local_delivery_status.json',dict(state='READY_FOR_REVIEW',complete=True,output=str(OUT),updated=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);last=None
    if not a.once:
        lock=(OUT/'local_watcher.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        current=status();write('server_status.json',current)
        signature={k:v for k,v in current.items() if k!='updated'}
        if signature!=last:print(json.dumps(current),flush=True);last=signature
        if a.once:return
        if current['state']=='FAILED':
            write('local_delivery_status.json',dict(state='SERVER_FAILED_REVIEW_REQUIRED',server=current,science_retried=False));return
        if current['state']=='COMPLETE':
            write('local_delivery_status.json',dict(state='DOWNLOADING_AND_VERIFYING'));deliver();return
        write('local_delivery_status.json',dict(state='WAITING_SERVER',server=current));time.sleep(45)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('local_delivery_status.json',dict(state='LOCAL_REVIEW_REQUIRED',error_class=type(error).__name__,message=str(error)));raise
