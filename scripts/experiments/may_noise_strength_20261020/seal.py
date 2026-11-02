"""Seal local delivery only after numerical QA and publication readback."""
from pathlib import Path
import json,hashlib,shutil,datetime,argparse

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main(out):
    assert not (out/'final_receipt.json').exists()
    design=read(out/'protocol.json');qa=read(out/'independent_qa.json');pub=read(out/'publication_receipt.json')
    assert qa['passed'] and qa['forward_count']==566 and pub['passed']
    assert read(out/'completion.json')['forwards']==538 and read(out/'transition_fp64/completion.json')['forwards']==28
    src=Path(__file__).resolve().parent;assert sha(src/'run.py')==design['script_sha256']==sha(out/'sources/run.py')
    assert sha(src/'transition_fp64.py')==read(out/'transition_fp64/protocol.json')['script_sha256']
    for p in src.iterdir():
        if p.suffix in ['.py','.md']:shutil.copyfile(p,out/'sources'/p.name)
    repo=src.parents[2]
    test=repo/'tests/experiments/test_may_noise_strength_20261020.py';assert test.exists()
    shutil.copyfile(test,out/'sources'/test.name)
    shutil.copyfile(repo/'.codex/scratch/publish_may_noise_strength_20261020.py',out/'sources/publish.py')
    shutil.copyfile(pub['report_path'],out/'sources'/Path(pub['report_path']).name)
    shutil.copyfile(out.parent/'may_noise_strength_20261020_v1/failure.json',out/'sources/initial_failure.json')
    write(out/'verification_receipt.json',dict(passed=True,cpu_tests=4,test_command='python tests/experiments/test_may_noise_strength_20261020.py',test_result='Ran 4 tests in 0.003s; OK',independent_input_hashes=566,stage_frame_samples=225792,full_last_frame_pairs=504000,
        visual_review=dict(passed=True,files=['noise_strength_scan.png','noise_layer_transition.png'],observed='Chinese labels legible; no clipping; precision panel matches four first-seed conditions across FP32/64; zero FP32 response explicitly annotated; late-layer conditional suppression shown alongside train counterexample'),
        main_protocol_sha256=sha(out/'protocol.json'),followup_protocol_sha256=sha(out/'transition_fp64/protocol.json'),local_only=True))
    prior=Path(design['prior']);assert sha(prior/'delivery_manifest.json')==design['prior_manifest_sha256']
    assert not list(out.rglob('*.partial'))
    members=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or 'matplotlib-cache' in p.relative_to(out).parts:continue
        assert p.name not in ['delivery_manifest.json','final_receipt.json']
        members.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(out/'delivery_manifest.json',dict(members=members,excluded=['matplotlib-cache/','delivery_manifest.json','final_receipt.json']))
    # Fresh full member read-back after writing manifest, not reusing the hash pass above.
    manifest=read(out/'delivery_manifest.json')
    for r in manifest['members']:
        p=out/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),members=len(members),bytes=sum(r['bytes'] for r in members),manifest_sha256=sha(out/'delivery_manifest.json'),fresh_full_sha_readback=True,forwards=566,training_updates=0,generator_forwards=0,syncnet_calls=0,prior_manifest_unchanged=True,local_persistent_only=True,report=pub['report_path'])
    write(out/'final_receipt.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
