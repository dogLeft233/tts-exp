"""Download only the two already selected members from the fixed HDTF mirror revision."""
import hashlib,json,shutil,zipfile,zlib,time
from pathlib import Path
import requests
from source_inventory import RangeFile
from select_replacement_identity import sha


class BoundedRangeFile(RangeFile):
    def read(self,n=-1):
        remaining=min(self.size-self.position,n if n>=0 else self.size-self.position)
        parts=[]
        while remaining:
            count=min(remaining,4*1024**2)
            for attempt in range(5):
                try:part=super().read(count);break
                except requests.RequestException:
                    if attempt==4:raise
                    time.sleep(2**attempt)
            assert len(part)==count;parts.append(part);remaining-=count
        return b''.join(parts)


def run(root,candidate_name='selected_candidate.json',attempt=1):
    root=Path(root);policy=root/'identity_replacement_policy_v1'
    selected=json.loads((policy/candidate_name).read_text())
    contract=json.loads((policy/'policy.json').read_text());assert selected['policy_sha256']==sha(policy/'policy.json')
    identity=selected['candidate']['identity']
    queue=json.loads((policy/'ordered_candidates.json').read_text())['candidates']
    assert selected['candidate'] in queue and Path(candidate_name).name==candidate_name
    out=policy/f'acquisition_attempt{attempt:02d}';out.mkdir(exist_ok=False)
    revision=contract['mirror_revision'];repository='global-optima-research/HDTF'
    tree=requests.get(f'https://huggingface.co/api/datasets/{repository}/tree/{revision}?recursive=true&expand=false',timeout=45)
    tree.raise_for_status();entry=next(x for x in tree.json() if x['path']=='videos.zip')
    url=f'https://huggingface.co/datasets/{repository}/resolve/{revision}/videos.zip'
    source=BoundedRangeFile(url,entry['size']);receipts=[]
    with zipfile.ZipFile(source) as archive:
        members=[archive.getinfo('videos/'+selected['candidate'][role]['file']) for role in ['main','test']]
        assert shutil.disk_usage(root).free>sum(x.file_size for x in members)+128*1024**2
        (out/'archive_members.json').write_text(json.dumps(dict(repository=repository,revision=revision,archive_metadata=entry,
            policy_sha256=selected['policy_sha256'],members=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,
            crc32=f'{i.CRC:08x}',header_offset=i.header_offset) for i in members]),indent=2))
        for item in members:
            target=root/'source'/Path(item.filename).name;receipt_path=target.with_suffix('.receipt.json')
            if target.exists():
                row=json.loads(receipt_path.read_text());assert row['revision']==revision and row['crc32']==f'{item.CRC:08x}'
                assert target.stat().st_size==item.file_size and sha(target)==row['sha256']
            else:
                assert not receipt_path.exists();payload=archive.read(item)
                assert len(payload)==item.file_size and zlib.crc32(payload)==item.CRC
                partial=target.with_suffix('.mp4.partial');assert not partial.exists()
                with partial.open('xb') as f:f.write(payload)
                digest=hashlib.sha256(payload).hexdigest();assert sha(partial)==digest;partial.rename(target)
                row=dict(name=item.filename,bytes=len(payload),sha256=digest,crc32=f'{item.CRC:08x}',revision=revision,
                         policy_sha256=selected['policy_sha256'],source='pinned mirror transport; official original URL in frozen candidate')
                receipt_path.write_text(json.dumps(row,indent=2))
            receipts.append(row);print(json.dumps(row),flush=True)
    (out/'range_requests.json').write_text(json.dumps(source.requests,indent=2))
    result=dict(passed=True,identity=identity,sources=receipts,policy_sha256=selected['policy_sha256'],
                geometry_qualified=False,training_admitted=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',default='selected_candidate.json');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();run(a.root,a.candidate,a.attempt)
