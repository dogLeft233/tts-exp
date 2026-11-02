"""Inventory an HDTF mirror without downloading the full archive; CRC-check selected members."""
import argparse
import hashlib
import io
import json
import re
import zipfile
import zlib
from pathlib import Path

import requests


class RangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.position=0;self.session=requests.Session();self.requests=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('negative seek')
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0:return b''
        start=self.position;end=start+n-1
        # Cache-busting keeps signed range responses tied to their actual requested interval.
        response=self.session.get(self.url+f'?download=true&range_start={start}',headers={'Range':f'bytes={start}-{end}'},timeout=(20,120),stream=True)
        response.raise_for_status()
        assert response.status_code==206, 'refuse a full archive response to a range request'
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        payload=response.content;assert len(payload)==n
        self.requests.append(dict(start=start,end=end,bytes=n,sha256=hashlib.sha256(payload).hexdigest()))
        self.position+=n
        return payload


def main(root,members):
    root.mkdir(parents=True,exist_ok=True)
    info=requests.get('https://huggingface.co/api/datasets/global-optima-research/HDTF',timeout=30).json()
    revision=info['sha']
    tree=requests.get(f'https://huggingface.co/api/datasets/global-optima-research/HDTF/tree/{revision}?recursive=true&expand=false',timeout=30).json()
    entry=next(x for x in tree if x['path']=='videos.zip')
    url=f'https://huggingface.co/datasets/global-optima-research/HDTF/resolve/{revision}/videos.zip'
    remote=RangeFile(url,entry['size'])
    with zipfile.ZipFile(remote) as archive:
        catalog=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,crc32=f'{x.CRC:08x}',header_offset=x.header_offset) for x in archive.infolist()]
        (root/'archive_catalog.json').write_text(json.dumps(dict(repository='global-optima-research/HDTF',revision=revision,
            archive_metadata=entry,members=catalog),indent=2)+'\n')
        if not members:
            groups={}
            for item in catalog:
                if item['name'].endswith('.mp4'):
                    name=Path(item['name']).stem
                    key=re.sub(r'\d+$','',re.sub(r'_\d+$','',name))
                    groups.setdefault(key,[]).append(item)
            repeated={k:v for k,v in groups.items() if len(v)>=2}
            print(json.dumps(dict(revision=revision,archive_bytes=entry['size'],video_members=len(catalog),repeated_identities=repeated),indent=2),flush=True)
        else:
            for name in members:
                item=archive.getinfo(name);target=root/'source'/Path(name).name
                target.parent.mkdir(exist_ok=True)
                assert not target.exists(),str(target)
                payload=archive.read(name)
                assert len(payload)==item.file_size and zlib.crc32(payload)==item.CRC
                target.write_bytes(payload)
                receipt=dict(name=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),crc32=f'{item.CRC:08x}',revision=revision)
                (target.with_suffix('.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                print(json.dumps(receipt),flush=True)
    (root/'range_requests.json').write_text(json.dumps(remote.requests,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--member',action='append',default=[])
    args=parser.parse_args();main(args.root,args.member)
