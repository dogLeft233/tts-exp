"""Windows/std-library archive SHA and every-member readback; no extraction."""
from pathlib import Path
import argparse,hashlib,json,tarfile,time

def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main(archive,receipt,out):
    expected=json.loads(Path(receipt).read_text());archive=Path(archive)
    with archive.open('rb') as f:actual=digest(f)
    assert actual==expected['sha256'] and archive.stat().st_size==expected['bytes']
    count=0;total=0
    with tarfile.open(archive,'r:') as tar:
        manifests=[m for m in tar.getmembers() if m.name.endswith('/artifact_manifest.json') and m.name.count('/')==1]
        assert len(manifests)==1
        raw=tar.extractfile(manifests[0]).read();assert hashlib.sha256(raw).hexdigest()==expected['manifest_sha'];manifest=json.loads(raw)
        required={manifest['run']+'/'+r['path']:r for r in manifest['files']}
        required.update({manifest['failed_attempt']+'/'+r['path']:r for r in manifest['failed_attempt_files']})
        found=set()
        for member in tar.getmembers():
            if not member.isfile():continue
            assert '..' not in Path(member.name).parts and not member.name.startswith('/')
            f=tar.extractfile(member);observed=digest(f);count+=1;total+=member.size
            if member.name in required:
                r=required[member.name];assert member.size==r['bytes'] and observed==r['sha256'],member.name;found.add(member.name)
        assert found==set(required)
    result={'status':'PASS','archive':str(archive.resolve()),'sha256':actual,'bytes':archive.stat().st_size,'readback_regular_members':count,'verified_manifest_members':len(required),'payload_bytes_read':total,'every_manifest_member_read_back':True,'no_extraction':True,'receipt_origin':str(Path(receipt).resolve()),'unix':time.time()}
    with Path(out).open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--receipt',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();main(a.archive,a.receipt,a.output)
