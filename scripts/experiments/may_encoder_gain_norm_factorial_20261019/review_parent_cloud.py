"""Read-only independent Linux verification of an existing whole cloud archive."""
from pathlib import Path
import io,json,hashlib,tarfile,shutil,time
R=Path(__file__).resolve().parent
def main():
 cfg=json.loads((R/'parent_cloud_input.json').read_text());cloud=Path(cfg['cloud']);manifest=cfg['manifest'];parts=cfg['chunks']['parts'];bundle=cfg['chunks']['parent_bundle']
 expected={x['path']:x for x in manifest['files']};seen=set();started=time.monotonic()
 class Reader:
  def __init__(self):self.index=0;self.file=None;self.current=None;self.count=0;self.total=0;self.whole=hashlib.sha256();self.verified=[]
  def read(self,size=-1):
   assert size>=0,'bounded reader only';out=bytearray()
   while len(out)<size:
    if self.file is None:
     if self.index==len(parts):break
     row=parts[self.index];p=cloud/row['file'];assert p.stat().st_size==row['bytes'];self.file=p.open('rb');self.current=hashlib.sha256();self.count=0
    block=self.file.read(min(size-len(out),4*1024**2))
    if not block:
     row=parts[self.index];assert self.count==row['bytes'] and self.current.hexdigest()==row['sha256'];self.file.close();self.file=None;self.verified.append(row);self.index+=1;continue
    assert shutil.disk_usage('/').free>256*1024**2,'root space guard'
    self.count+=len(block);self.total+=len(block);self.current.update(block);self.whole.update(block);out.extend(block)
   return bytes(out)
 stream=Reader();bodybytes=0
 with tarfile.open(fileobj=stream,mode='r|gz') as archive:
  for member in archive:
   assert member.isfile() and not member.issym() and not member.islnk()
   if member.name=='delivery_manifest.json':
    archived=json.load(archive.extractfile(member));assert archived['files']==manifest['files'];continue
   assert member.name.startswith('remote/');name=member.name[7:];row=expected[name];assert name not in seen and member.size==row['bytes']
   h=hashlib.sha256();n=0
   with archive.extractfile(member) as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b);n+=len(b)
   assert n==row['bytes'] and h.hexdigest()==row['sha256'],name
   seen.add(name);bodybytes+=n
 while stream.read(4*1024**2):pass
 assert seen==set(expected) and stream.total==bundle['bytes'] and stream.whole.hexdigest()==bundle['sha256'] and len(stream.verified)==9
 result=dict(passed=True,independent_linux_cloud_readback=True,cloud=str(cloud),archive_sha256=stream.whole.hexdigest(),archive_bytes=stream.total,members=len(seen),body_bytes=bodybytes,parts_verified=9,elapsed_seconds=time.monotonic()-started,new_gpu_calls=0,deleted_files=0)
 (R/'parent_cloud_full_member_review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
