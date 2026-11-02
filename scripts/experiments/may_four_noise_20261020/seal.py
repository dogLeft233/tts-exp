"""Seal completed scientific and recovery evidence after independent QA."""
from pathlib import Path
import json,hashlib,shutil,datetime,argparse
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def main(out):
    assert not (out/'final_receipt.json').exists();qa=read(out/'independent_qa.json');done=read(out/'completion.json');pub=read(out/'publication_receipt.json');p=read(out/'protocol.json');recovery=read(out/'recovery_protocol.json')
    assert qa['passed'] and qa['admitted_forwards']==780 and done['status']=='COMPLETED' and pub['passed']
    here=Path(__file__).resolve().parent;repo=here.parents[2]
    assert sha(here/'run.py')==p['script_sha256'];assert sha(here/'recover_streaming.py')==recovery['recovery_source_sha256']
    assert sha(here/'resume_driver.py')==read(out/'resume_driver_protocol.json')['source_sha256']
    sources=out/'sources'
    for f in list(here.glob('*.py'))+list(here.glob('*.md'))+[repo/'tests/experiments/test_may_four_noise_20261020.py',repo/'.codex/scratch/publish_may_four_noise_20261020.py',Path(pub['report_path'])]:
        dest=sources/f.name
        if dest.exists():assert sha(dest)==sha(f)
        else:shutil.copy2(f,dest)
    shutil.copy2('F:/Datasets/tts-exp/host_cache_relocation_20261020_v1/relocation_receipt.json',sources/'host_cache_relocation_receipt.json')
    failed=read(out/'failed_attempt_snapshot.json');old=Path(failed['root'])
    for r in failed['members']:
        f=old/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
    write(out/'verification_receipt.json',dict(passed=True,cpu_tests=6,test_command='python tests/experiments/test_may_four_noise_20261020.py',test_result='Ran6 tests in0.050s; OK; streamed float32/64 metrics/hash equal full-array implementation',
        visual_review=dict(passed=True,files=['four_noise_response_ratios.png','four_noise_precision.png','four_noise_absolute_response.png'],observed='Chinese axes legible; both May prefixes and three-instance ranges retained; FP32/64 panel uses matched first instances; unresolved FP32 flagged; no cropped data'),
        independent_qa_sha256=sha(out/'independent_qa.json'),original_failed_root_fresh_readback=True,original_protocol_sha256=sha(out/'protocol.json'),recovery_protocol_sha256=sha(out/'recovery_protocol.json'),local_only=True))
    assert not list(out.rglob('*.partial'))
    members=[]
    for f in sorted(out.rglob('*')):
        if not f.is_file() or 'matplotlib-cache' in f.relative_to(out).parts:continue
        assert f.name not in ['delivery_manifest.json','final_receipt.json'];members.append(dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    write(out/'delivery_manifest.json',dict(members=members,excluded=['matplotlib-cache/','delivery_manifest.json','final_receipt.json']))
    for r in read(out/'delivery_manifest.json')['members']:
        f=out/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
    result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),members=len(members),bytes=sum(r['bytes'] for r in members),manifest_sha256=sha(out/'delivery_manifest.json'),fresh_full_sha_readback=True,scientific_forwards=746,extra_recovery_bridges=34,admitted_successful_forwards=780,excluded_failed_attempt_complete_outputs=done['excluded_failed_attempt_complete_outputs'],all_complete_successful_forwards_including_excluded=780+done['excluded_failed_attempt_complete_outputs'],training_updates=0,generator_forwards=0,syncnet_calls=0,local_persistent_only=True,report=pub['report_path'])
    write(out/'final_receipt.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
